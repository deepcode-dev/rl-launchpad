# ruff: noqa: E402
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
        "policy_distribution": "tanh_squashed_normal",
        "observation_normalization": "running_mean_variance",
        "reward_source": "mujoco_playground_state_reward",
        "physics_backend": "jax",
        "num_envs": int(config.get("num_envs", 32)),
        "rollout_size": int(config.get("steps_per_epoch", 4096)),
        "seed": seed,
        "env_name": env_name,
        "obs_dim": obs_dim,
        "critic_obs_dim": int(critic_obs_dim or obs_dim),
        "act_dim": act_dim,
        "hidden_dim": hidden_dim,
        "hidden_sizes": list(hidden_sizes or (hidden_dim, hidden_dim)),
        "history_len": config.get("history_len", 5),
        "episode_length": config.get("episode_length", 1000),
        "total_env_steps": total_env_steps,
        "wall_time_seconds": wall_time_seconds,
        "config": config,
    }
    with open(f"{path}.meta.json", "w") as f:
        json.dump(metadata, f, indent=2)


def train_single_seed(seed, config):
    env_name = config.get("env_name", "Go1JoystickFlatTerrain")
    num_envs = config.get("num_envs", 32)
    epochs = config.get("epochs", 50)
    steps_per_epoch = config.get("steps_per_epoch", 4096)
    batch_size = config.get("batch_size", 256)
    train_iters = config.get("train_iters", 10)
    gamma = config.get("gamma", 0.99)
    lam = config.get("lam", 0.95)
    clip_ratio = config.get("clip_ratio", 0.2)
    max_grad_norm = config.get("max_grad_norm", 0.5)
    hidden_dim = config.get("hidden_dim", 256)
    hidden_sizes = tuple(config.get("hidden_sizes", (hidden_dim, hidden_dim)))
    pi_lr = config.get("pi_lr", 0.0003)
    initial_log_std = config.get("initial_log_std", -0.5)
    ent_coef_start = config.get("ent_coef", 0.01)
    ent_coef_final = config.get("ent_coef_final", ent_coef_start)

    if epochs < 1 or num_envs < 1 or steps_per_epoch < 1:
        raise ValueError("epochs, num_envs, and steps_per_epoch must be positive")
    if steps_per_epoch % num_envs:
        raise ValueError("steps_per_epoch must be divisible by num_envs")
    configured_steps = config.get("total_timesteps_per_seed")
    actual_steps = epochs * steps_per_epoch
    if configured_steps is not None and int(configured_steps) != actual_steps:
        raise ValueError(
            "total_timesteps_per_seed must equal epochs * steps_per_epoch: "
            f"{configured_steps} != {actual_steps}"
        )

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n=======================================================", flush=True)
    print(f"  Training Seed {seed} on Device: {device}", flush=True)
    print("=======================================================", flush=True)

    env = MJXVectorPyTorchWrapper(
        env_name, num_envs=num_envs, seed=seed,
        history_len=config.get("history_len", 5),
        episode_length=config.get("episode_length", 1000),
        config_overrides=config.get("environment_overrides"),
    )
    obs_dim = env.observation_dim
    critic_obs_dim = env.privileged_observation_dim
    act_dim = env.action_dim

    agent = ActorCritic(
        obs_dim,
        act_dim,
        hidden_dim,
        initial_log_std=initial_log_std,
        critic_obs_dim=critic_obs_dim,
        hidden_sizes=hidden_sizes,
    ).to(device)
    initial_steps = 0
    checkpoint_dir = config.get("checkpoint_dir", "checkpoints/ppo_v2")
    resume_checkpoint = config.get("resume_checkpoint")
    if config.get("resume") and not resume_checkpoint:
        auto_ckpt = os.path.join(checkpoint_dir, f"ppo_seed{seed}.pt")
        if os.path.exists(auto_ckpt):
            resume_checkpoint = auto_ckpt

    if resume_checkpoint and os.path.exists(resume_checkpoint):
        state_dict = torch.load(resume_checkpoint, map_location=device, weights_only=True)
        agent.load_state_dict(state_dict, strict=True)
        meta_json = str(resume_checkpoint) + ".meta.json"
        if os.path.exists(meta_json):
            try:
                with open(meta_json) as f:
                    meta_data = json.load(f)
                    initial_steps = int(meta_data.get("total_env_steps", 0))
            except Exception:
                initial_steps = 19988480
        else:
            initial_steps = 19988480
        print(f"Resumed policy weights from: {resume_checkpoint} (initial_steps={initial_steps:,})", flush=True)
    optimizer = torch.optim.Adam(agent.parameters(), lr=pi_lr)

    num_steps = steps_per_epoch // num_envs
    total_steps = initial_steps
    start_time = time.time()

    obs, reset_info = env.reset(seed=seed)
    critic_obs = reset_info["privileged_observation"]
    agent.update_observation_stats(obs.to(device), critic_obs.to(device))

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
        "entropy_coefficients": [],
        "learning_rates": [],
        "update_epochs": [],
        "total_steps": [],
        "wall_times": [],
    }

    # Pre-allocate contiguous GPU rollout buffers to eliminate thousands of CUDA allocation kernels per epoch
    obs_buffer = torch.empty((num_steps, num_envs, obs_dim), device=device)
    critic_obs_buffer = torch.empty((num_steps, num_envs, critic_obs_dim), device=device)
    actions_buffer = torch.empty((num_steps, num_envs, act_dim), device=device)
    log_probs_buffer = torch.empty((num_steps, num_envs), device=device)
    values_buffer = torch.empty((num_steps, num_envs), device=device)
    rewards_buffer = torch.empty((num_steps, num_envs), device=device)
    terminated_buffer = torch.empty((num_steps, num_envs), device=device, dtype=torch.bool)
    truncated_buffer = torch.empty((num_steps, num_envs), device=device, dtype=torch.bool)
    truncation_values_buffer = torch.empty((num_steps, num_envs), device=device)

    learning_rate = pi_lr

    for epoch in range(1, epochs + 1):
        epoch_rewards = []
        epoch_linear_velocity_errors = []
        epoch_yaw_rate_errors = []
        epoch_action_magnitudes = []
        epoch_positive_reward_fractions = []
        epoch_termination_fractions = []

        if epoch > 1:
            last_kl = history["approx_kls"][-1]
            target_kl = config.get("target_kl", 0.02)
            if last_kl > target_kl * 2.0:
                learning_rate = max(learning_rate / 1.5, 1e-5)
            elif last_kl < target_kl / 2.0:
                learning_rate = min(learning_rate * 1.5, pi_lr)
        
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate

        schedule_fraction = (epoch - 1) / max(epochs - 1, 1)
        entropy_coefficient = (
            ent_coef_start
            + schedule_fraction * (ent_coef_final - ent_coef_start)
        )

        for step in range(num_steps):
            with torch.no_grad():
                obs_dev = obs.to(device)
                critic_obs_dev = critic_obs.to(device)
                action, log_prob = agent.get_action(obs_dev)
                value = agent.get_value(obs_dev, critic_obs_dev)

            next_obs, reward, done, truncated_step, step_info = env.step(action)
            truncation_value = torch.zeros(num_envs, device=device)
            if bool(truncated_step.any()):
                with torch.no_grad():
                    final_values = agent.get_value(
                        step_info["final_observation"].to(device),
                        step_info["final_privileged_observation"].to(device),
                    )
                truncation_value = torch.where(
                    truncated_step.to(device), final_values, truncation_value
                )

            obs_buffer[step] = obs_dev
            critic_obs_buffer[step] = critic_obs_dev
            actions_buffer[step] = action
            rewards_buffer[step] = reward.to(device)
            terminated_buffer[step] = done.to(device)
            truncated_buffer[step] = truncated_step.to(device)
            truncation_values_buffer[step] = truncation_value
            log_probs_buffer[step] = log_prob
            values_buffer[step] = value

            epoch_rewards.append(reward.mean().item())
            epoch_linear_velocity_errors.append(
                step_info["linear_velocity_error"].mean().item()
            )
            epoch_yaw_rate_errors.append(step_info["yaw_rate_error"].mean().item())
            epoch_action_magnitudes.append(step_info["mean_abs_action"].mean().item())
            epoch_positive_reward_fractions.append((reward > 0).float().mean().item())
            epoch_termination_fractions.append(done.float().mean().item())
            total_steps += num_envs
            obs = next_obs
            critic_obs = step_info["privileged_observation"]

        with torch.no_grad():
            next_value = agent.get_value(obs.to(device), critic_obs.to(device))

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
        history["linear_velocity_errors"].append(
            float(np.mean(epoch_linear_velocity_errors))
        )
        history["yaw_rate_errors"].append(float(np.mean(epoch_yaw_rate_errors)))
        history["mean_abs_actions"].append(float(np.mean(epoch_action_magnitudes)))
        history["positive_reward_fractions"].append(
            float(np.mean(epoch_positive_reward_fractions))
        )
        history["termination_fractions"].append(
            float(np.mean(epoch_termination_fractions))
        )
        history["action_stds"].append(
            float(
                agent.actor_log_std.detach()
                .clamp(agent._LOG_STD_MIN, agent._LOG_STD_MAX)
                .exp()
                .mean()
                .item()
            )
        )
        history["entropy_coefficients"].append(float(entropy_coefficient))
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

        # Auto-save main seed checkpoint every 10 epochs and at completion
        if epoch % 10 == 0 or epoch == epochs:
            checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            ckpt_path = os.path.join(checkpoint_dir, f"ppo_seed{seed}.pt")
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

    checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, f"ppo_seed{seed}.pt")
    save_checkpoint(
        ckpt_path, agent, seed=seed, env_name=env_name, obs_dim=obs_dim,
        act_dim=act_dim, hidden_dim=hidden_dim, config=config,
        total_env_steps=total_steps, wall_time_seconds=time.time() - start_time,
        critic_obs_dim=critic_obs_dim, hidden_sizes=hidden_sizes,
    )
    print(f"Saved Checkpoint: {ckpt_path}", flush=True)

    return history


def main(config_path="configs/default.yaml", seed_override=None, resume_override=False):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if resume_override:
        config["resume"] = True

    seeds = [int(seed_override)] if seed_override is not None else config.get(
        "seeds", [10, 11, 21, 67, 96]
    )
    config = {**config, "seeds": seeds, "seed": seeds[0]}
    expected_total = int(config.get("total_timesteps_per_seed", 0)) * len(seeds)
    if seed_override is not None:
        config["total_timesteps"] = expected_total
    configured_total = config.get("total_timesteps")
    if configured_total is not None and int(configured_total) != expected_total:
        raise ValueError(
            f"total_timesteps must equal per-seed steps * seed count: "
            f"{configured_total} != {expected_total}"
        )
    print(f"--- Starting Multi-Seed Training for Seeds: {seeds} ---", flush=True)

    all_histories = []
    total_start_time = time.time()

    for s in seeds:
        hist = train_single_seed(s, config)
        all_histories.append(hist)
        # Persist after every seed so an interrupted multi-hour run keeps all
        # completed measurements.
        checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        with open(os.path.join(checkpoint_dir, "ppo_multi_seed_results.json"), "w") as f:
            json.dump(all_histories, f, indent=2)

    output_path = os.path.join(config.get("checkpoint_dir", "checkpoints"), "ppo_multi_seed_results.json")
    with open(output_path, "w") as f:
        json.dump(all_histories, f, indent=2)

    total_duration = time.time() - total_start_time
    print("\n=======================================================", flush=True)
    print(f"  All {len(seeds)} Seeds Complete! Total Wall Time: {total_duration:.1f}s", flush=True)
    print(f"  Results saved to: {output_path}", flush=True)
    print("=======================================================", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train all configured PPO seeds.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=None, help="Override seed to train")
    parser.add_argument("--resume", action="store_true", help="Resume from existing checkpoint")
    args = parser.parse_args()
    main(args.config, seed_override=args.seed, resume_override=args.resume)
