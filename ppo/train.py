# ruff: noqa: E402
"""Single-seed PPO v2 trainer matching exact Champion engine capabilities."""

import argparse
import json
import os
import sys
import time
import warnings
import numpy as np
import torch
import yaml

# Filter harmless JAX integer casting runtime warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*overflow encountered in cast.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="jax")

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ppo.agent import ActorCritic
from ppo.ppo import TRAINING_CONTRACT, compute_gae, update
from ppo.env import MJXVectorPyTorchWrapper


def save_checkpoint(
    path, agent, *, seed, env_name, obs_dim, act_dim, hidden_dim, config,
    total_env_steps, wall_time_seconds, critic_obs_dim=None, hidden_sizes=None,
):
    """Keep .pt compatible with existing evaluators and record its contract separately."""
    torch.save(agent.state_dict(), path)
    metadata = {
        "checkpoint_format": "raw_state_dict",
        "algorithm": "PPO",
        "training_contract": TRAINING_CONTRACT,
        "policy_distribution": "clipped_normal",
        "observation_normalization": "running_mean_variance",
        "reward_source": "mujoco_playground_state_reward",
        "physics_backend": "jax",
        "num_envs": int(config.get("num_envs", 8192)),
        "rollout_size": int(config.get("steps_per_epoch", 163840)),
        "seed": int(seed),
        "env_name": env_name,
        "obs_dim": int(obs_dim),
        "critic_obs_dim": int(critic_obs_dim) if critic_obs_dim is not None else int(obs_dim),
        "act_dim": int(act_dim),
        "hidden_dim": int(hidden_dim),
        "hidden_sizes": hidden_sizes or [hidden_dim, hidden_dim],
        "history_len": int(config.get("history_len", 1)),
        "episode_length": int(config.get("episode_length", 1000)),
        "total_env_steps": int(total_env_steps),
        "wall_time_seconds": float(wall_time_seconds),
        "config": config,
    }
    with open(f"{path}.meta.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def train_single_seed(seed, config, resume=False):
    """Train a single PPO seed using the exact Champion PPO v2 engine."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n=======================================================", flush=True)
    print(f"  Training Seed {seed} on Device: {device}", flush=True)
    print("=======================================================", flush=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    env_name = config.get("env_name", "Go1JoystickFlatTerrain")
    num_envs = config.get("num_envs", 8192)
    history_len = config.get("history_len", 1)
    episode_length = config.get("episode_length", 1000)

    env = MJXVectorPyTorchWrapper(
        env_name=env_name,
        num_envs=num_envs,
        history_len=history_len,
        episode_length=episode_length,
        device=str(device),
    )

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    critic_obs_dim = getattr(env, "critic_observation_dim", obs_dim)

    hidden_dim = config.get("hidden_dim", 512)
    hidden_sizes = config.get("hidden_sizes", [512, 256, 128])
    initial_log_std = config.get("initial_log_std", -1.0)

    agent = ActorCritic(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dim=hidden_dim,
        initial_log_std=initial_log_std,
        critic_obs_dim=critic_obs_dim,
        hidden_sizes=hidden_sizes,
    ).to(device)

    # Enable torch.compile for extra GPU acceleration if available
    try:
        agent = torch.compile(agent)
    except Exception:
        pass

    pi_lr = float(config.get("pi_lr", 3e-4))
    optimizer = torch.optim.Adam(agent.parameters(), lr=pi_lr)

    epochs = config.get("epochs", 1220)
    steps_per_epoch = config.get("steps_per_epoch", 163840)
    rollout_steps = steps_per_epoch // num_envs
    gamma = config.get("gamma", 0.97)
    lam = config.get("lam", 0.95)
    clip_ratio = config.get("clip_ratio", 0.2)
    max_grad_norm = config.get("max_grad_norm", 1.0)
    train_iters = config.get("train_iters", 4)
    batch_size = config.get("batch_size", 5120)

    checkpoint_dir = config.get("checkpoint_dir", "checkpoints/ppo_v2")
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, f"ppo_seed{seed}.pt")

    start_epoch = 1
    total_steps = 0
    start_time = time.time()

    if resume and os.path.exists(ckpt_path):
        print(f"Resuming from existing checkpoint: {ckpt_path}", flush=True)
        uncompiled = getattr(agent, "_orig_mod", agent)
        uncompiled.load_state_dict(torch.load(ckpt_path, map_location=device))
        meta_path = f"{ckpt_path}.meta.json"
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
                total_steps = meta.get("total_env_steps", 0)
                start_epoch = total_steps // steps_per_epoch + 1

    obs, info = env.reset(seed=seed)
    critic_obs = info.get("critic_obs", obs)

    obs_buffer = torch.zeros((rollout_steps, num_envs, obs_dim), device=device)
    critic_obs_buffer = torch.zeros((rollout_steps, num_envs, critic_obs_dim), device=device)
    actions_buffer = torch.zeros((rollout_steps, num_envs, act_dim), device=device)
    log_probs_buffer = torch.zeros((rollout_steps, num_envs), device=device)
    rewards_buffer = torch.zeros((rollout_steps, num_envs), device=device)
    values_buffer = torch.zeros((rollout_steps, num_envs), device=device)
    terminated_buffer = torch.zeros((rollout_steps, num_envs), device=device)
    truncated_buffer = torch.zeros((rollout_steps, num_envs), device=device)
    truncation_values_buffer = torch.zeros((rollout_steps, num_envs), device=device)

    history = {
        "seed": seed,
        "epochs": [],
        "rewards": [],
        "mean_step_rewards": [],
        "pol_losses": [],
        "val_losses": [],
        "entropies": [],
        "approx_kls": [],
        "clip_fractions": [],
        "linear_velocity_errors": [],
        "yaw_rate_errors": [],
        "mean_abs_actions": [],
        "positive_reward_fractions": [],
        "termination_fractions": [],
        "action_stds": [],
        "learning_rates": [],
        "update_epochs": [],
        "total_steps": [],
        "wall_times": [],
    }

    entropy_coefficient = float(config.get("ent_coef", 0.01))
    learning_rate = pi_lr

    for epoch in range(start_epoch, epochs + 1):
        epoch_rewards = []
        epoch_linear_velocity_errors = []
        epoch_yaw_rate_errors = []
        epoch_action_magnitudes = []
        epoch_positive_reward_fractions = []
        epoch_termination_fractions = []

        for step in range(rollout_steps):
            total_steps += num_envs
            with torch.no_grad():
                action, log_prob = agent.get_action(obs)
                value = agent.get_value(obs, critic_obs)

            next_obs, reward, terminated, truncated, next_info = env.step(action)
            next_critic_obs = next_info.get("critic_obs", next_obs)

            obs_buffer[step] = obs
            critic_obs_buffer[step] = critic_obs
            actions_buffer[step] = action
            log_probs_buffer[step] = log_prob
            rewards_buffer[step] = reward
            values_buffer[step] = value
            terminated_buffer[step] = terminated
            truncated_buffer[step] = truncated

            if truncated.any():
                with torch.no_grad():
                    trunc_val = agent.get_value(next_obs, next_critic_obs)
                truncation_values_buffer[step] = trunc_val
            else:
                truncation_values_buffer[step] = 0.0

            epoch_rewards.append(reward.mean().item())
            epoch_linear_velocity_errors.append(next_info["linear_velocity_error"].mean().item())
            epoch_yaw_rate_errors.append(next_info["yaw_rate_error"].mean().item())
            epoch_action_magnitudes.append(action.abs().mean().item())
            epoch_positive_reward_fractions.append((reward > 0).float().mean().item())
            epoch_termination_fractions.append(terminated.float().mean().item())

            obs = next_obs
            critic_obs = next_critic_obs

        with torch.no_grad():
            next_value = agent.get_value(obs, critic_obs)

        advantages, returns = compute_gae(
            rewards_buffer, values_buffer, terminated_buffer, next_value,
            gamma=gamma, lam=lam, truncated=truncated_buffer,
            truncation_values=truncation_values_buffer,
        )

        obs_tensor = obs_buffer.flatten(0, 1)
        critic_obs_tensor = critic_obs_buffer.flatten(0, 1)
        act_tensor = actions_buffer.flatten(0, 1)
        old_log_probs = log_probs_buffer.flatten(0, 1)
        adv_tensor = advantages.flatten(0, 1)
        ret_tensor = returns.flatten(0, 1)

        update_metrics = update(
            agent=agent,
            optimizer=optimizer,
            observations=obs_tensor,
            actions=act_tensor,
            old_log_probs=old_log_probs,
            returns=ret_tensor,
            advantages=adv_tensor,
            critic_observations=critic_obs_tensor,
            old_values=values_buffer.flatten(0, 1),
            epochs=train_iters,
            batch_size=batch_size,
            clip_ratio=clip_ratio,
            max_grad_norm=max_grad_norm,
            vf_coef=config.get("vf_coef", 0.5),
            ent_coef=entropy_coefficient,
            target_kl=config.get("target_kl", 0.02),
        )
        agent.update_observation_stats(obs_tensor, critic_obs_tensor)

        total_time = time.time() - start_time
        avg_reward = float(np.mean(epoch_rewards))

        history["epochs"].append(epoch)
        history["rewards"].append(avg_reward)
        history["mean_step_rewards"].append(avg_reward)
        history["pol_losses"].append(float(update_metrics["policy_loss"]))
        history["val_losses"].append(float(update_metrics["value_loss"]))
        history["entropies"].append(float(update_metrics["entropy"]))
        history["approx_kls"].append(float(update_metrics["approx_kl"]))
        history["clip_fractions"].append(float(update_metrics["clip_fraction"]))
        history["linear_velocity_errors"].append(float(np.mean(epoch_linear_velocity_errors)))
        history["yaw_rate_errors"].append(float(np.mean(epoch_yaw_rate_errors)))
        history["mean_abs_actions"].append(float(np.mean(epoch_action_magnitudes)))
        history["positive_reward_fractions"].append(float(np.mean(epoch_positive_reward_fractions)))
        history["termination_fractions"].append(float(np.mean(epoch_termination_fractions)))
        history["action_stds"].append(
            float(
                getattr(agent, "_orig_mod", agent).actor_log_std.detach()
                .clamp(agent._LOG_STD_MIN, agent._LOG_STD_MAX)
                .exp()
                .mean()
                .item()
            )
        )
        history["learning_rates"].append(float(learning_rate))
        history["update_epochs"].append(int(update_metrics["update_epochs"]))
        history["total_steps"].append(total_steps)
        history["wall_times"].append(float(total_time))

        if epoch % 5 == 0 or epoch == epochs:
            print(
                f"Seed {seed:02d} | Epoch {epoch:02d}/{epochs} | "
                f"Avg R: {avg_reward:.4f} | "
                f"Val Loss: {update_metrics['value_loss']:.4f} | "
                f"KL: {update_metrics['approx_kl']:.4f} | "
                f"LinErr: {history['linear_velocity_errors'][-1]:.3f} | "
                f"YawErr: {history['yaw_rate_errors'][-1]:.3f} | "
                f"|A|: {history['mean_abs_actions'][-1]:.3f} | "
                f"R>0: {history['positive_reward_fractions'][-1]:.2%} | "
                f"Done: {history['termination_fractions'][-1]:.2%} | "
                f"Sigma: {history['action_stds'][-1]:.3f} | "
                f"Steps: {total_steps} | "
                f"Time: {total_time:.1f}s",
                flush=True,
            )

        if epoch % 10 == 0 or epoch == epochs:
            save_checkpoint(
                ckpt_path,
                agent,
                seed=seed,
                env_name=env_name,
                obs_dim=obs_dim,
                act_dim=act_dim,
                hidden_dim=hidden_dim,
                config=config,
                total_env_steps=total_steps,
                wall_time_seconds=total_time,
                critic_obs_dim=critic_obs_dim,
                hidden_sizes=hidden_sizes,
            )

    env.close()
    return history


def main(config_path, seed_override=None, resume_override=False):
    with open(config_path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    seed = seed_override if seed_override is not None else config.get("seed", 2001)
    config = {**config, "seed": seed}

    history = train_single_seed(seed, config, resume=resume_override)

    checkpoint_dir = config.get("checkpoint_dir", "checkpoints/ppo_v2")
    output_path = os.path.join(checkpoint_dir, f"ppo_seed{seed}_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("\n=======================================================", flush=True)
    print(f"  Seed {seed} Training Complete!", flush=True)
    print(f"  Results saved to: {output_path}", flush=True)
    print("=======================================================", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a single PPO seed.")
    parser.add_argument("--config", default="configs/champion_v2.yaml")
    parser.add_argument("--seed", type=int, default=None, help="Override seed to train")
    parser.add_argument("--resume", action="store_true", help="Resume from existing checkpoint")
    args = parser.parse_args()
    main(args.config, seed_override=args.seed, resume_override=args.resume)
