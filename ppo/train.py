# ruff: noqa: E402
import argparse
import os
import sys
import time
import warnings
import json
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
    total_env_steps, wall_time_seconds,
):
    """Save evaluator-compatible weights plus separate reproducibility metadata."""
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
        "act_dim": act_dim,
        "hidden_dim": hidden_dim,
        "history_len": config.get("history_len", 5),
        "episode_length": config.get("episode_length", 1000),
        "total_env_steps": total_env_steps,
        "wall_time_seconds": wall_time_seconds,
        "config": config,
    }
    with open(f"{path}.meta.json", "w") as f:
        json.dump(metadata, f, indent=2)


def main(config_path="configs/default.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    seed = config.get("seed", config.get("seeds", [42])[0])
    num_envs = config.get("num_envs", 32)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    env_name = config.get("env_name", "Go1JoystickFlatTerrain")
    print(f"--- Starting Vectorized PPO Training ({num_envs} Parallel Envs) ---")
    print(f"Environment: {env_name} | Seed: {seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device}")

    # Instantiate vectorized environment wrapper
    env = MJXVectorPyTorchWrapper(
        env_name, num_envs=num_envs, seed=seed,
        history_len=config.get("history_len", 5),
        episode_length=config.get("episode_length", 1000),
    )
    obs_dim = env.observation_dim
    act_dim = env.action_dim
    print(f"Observation Dim: {obs_dim} | Action Dim: {act_dim}")

    hidden_dim = config.get("hidden_dim", 256)
    pi_lr = config.get("pi_lr", 0.0003)
    agent = ActorCritic(obs_dim, act_dim, hidden_dim).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=pi_lr)

    checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    steps_per_epoch = config.get("steps_per_epoch", 2048)
    num_steps = steps_per_epoch // num_envs  # Iterations per epoch per env
    if steps_per_epoch % num_envs:
        raise ValueError("steps_per_epoch must be divisible by num_envs")
    epochs = config.get("epochs", 10)
    if epochs < 1 or num_envs < 1 or steps_per_epoch < 1:
        raise ValueError("epochs, num_envs, and steps_per_epoch must be positive")
    configured_steps = config.get("total_timesteps_per_seed")
    actual_steps = epochs * steps_per_epoch
    if configured_steps is not None and int(configured_steps) != actual_steps:
        raise ValueError(
            "total_timesteps_per_seed must equal epochs * steps_per_epoch: "
            f"{configured_steps} != {actual_steps}"
        )
    batch_size = config.get("batch_size", 256)
    train_iters = config.get("train_iters", 10)
    gamma = config.get("gamma", 0.99)
    lam = config.get("lam", 0.95)
    clip_ratio = config.get("clip_ratio", 0.2)
    max_grad_norm = config.get("max_grad_norm", 0.5)

    total_steps = 0
    start_time = time.time()

    obs, _ = env.reset(seed=seed)
    agent.update_observation_stats(obs.to(device))

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        observations, actions, rewards, terminated, truncated_flags, truncation_values, log_probs, values = (
            [], [], [], [], [], [], [], []
        )
        epoch_rewards = []

        for step in range(num_steps):
            with torch.no_grad():
                obs_dev = obs.to(device)
                action, log_prob = agent.get_action(obs_dev)
                value = agent.get_value(obs_dev)

            action_cpu = action.cpu()
            next_obs, reward, done, truncated_step, step_info = env.step(action_cpu)
            truncation_value = torch.zeros(num_envs, device=device)
            if bool(truncated_step.any()):
                with torch.no_grad():
                    final_values = agent.get_value(step_info["final_observation"].to(device))
                truncation_value = torch.where(
                    truncated_step.to(device), final_values, truncation_value
                )

            observations.append(obs_dev)
            actions.append(action)
            rewards.append(reward.to(device))
            terminated.append(done.to(device))
            truncated_flags.append(truncated_step.to(device))
            truncation_values.append(truncation_value)
            log_probs.append(log_prob)
            values.append(value)

            epoch_rewards.append(reward.mean().item())
            total_steps += num_envs
            obs = next_obs

        # Bootstrap value for unfinished steps
        with torch.no_grad():
            next_value = agent.get_value(obs.to(device))

        # Keep [time, env] structure for GAE.  Flattening first would connect
        # one environment's value target to another environment's trajectory.
        rewards_tensor = torch.stack(rewards)
        values_tensor = torch.stack(values)
        terminated_tensor = torch.stack(terminated)
        truncated_tensor = torch.stack(truncated_flags)
        truncation_values_tensor = torch.stack(truncation_values)
        advantages, returns = compute_gae(
            rewards_tensor, values_tensor, terminated_tensor, next_value,
            gamma=gamma, lam=lam, truncated=truncated_tensor,
            truncation_values=truncation_values_tensor,
        )

        obs_tensor = torch.stack(observations).flatten(0, 1)
        act_tensor = torch.stack(actions).flatten(0, 1)
        old_log_probs = torch.stack(log_probs).flatten(0, 1)
        adv_tensor = advantages.flatten(0, 1)
        ret_tensor = returns.flatten(0, 1)

        # PPO parameter updates
        update_metrics = update(
            agent=agent,
            optimizer=optimizer,
            observations=obs_tensor,
            actions=act_tensor,
            old_log_probs=old_log_probs,
            returns=ret_tensor,
            advantages=adv_tensor,
            old_values=values_tensor.flatten(0, 1),
            epochs=train_iters,
            batch_size=batch_size,
            clip_ratio=clip_ratio,
            max_grad_norm=max_grad_norm,
            vf_coef=config.get("vf_coef", 0.5),
            ent_coef=config.get("ent_coef", 0.01),
        )
        agent.update_observation_stats(obs_tensor)

        epoch_time = time.time() - epoch_start_time
        avg_step_reward = np.mean(epoch_rewards)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Avg Reward: {avg_step_reward:.6f} | "
            f"Pol Loss: {update_metrics['policy_loss']:.4f} | "
            f"Val Loss: {update_metrics['value_loss']:.4f} | "
            f"Entropy: {update_metrics['entropy']:.4f} | "
            f"KL: {update_metrics['approx_kl']:.4f} | "
            f"ClipFrac: {update_metrics['clip_fraction']:.3f} | "
            f"Total Steps: {total_steps} | "
            f"Time: {epoch_time:.2f}s",
            flush=True
        )

        if epoch % 10 == 0 or epoch == epochs:
            checkpoint_path = os.path.join(checkpoint_dir, f"{env_name}_epoch_{epoch}.pt")
            save_checkpoint(
                checkpoint_path, agent, seed=seed, env_name=env_name, obs_dim=obs_dim,
                act_dim=act_dim, hidden_dim=hidden_dim, config=config,
                total_env_steps=total_steps, wall_time_seconds=time.time() - start_time,
            )
            print(f"  --> Saved Checkpoint: {checkpoint_path}", flush=True)

    final_path = os.path.join(checkpoint_dir, f"{env_name}_final.pt")
    save_checkpoint(
        final_path, agent, seed=seed, env_name=env_name, obs_dim=obs_dim,
        act_dim=act_dim, hidden_dim=hidden_dim, config=config,
        total_env_steps=total_steps, wall_time_seconds=time.time() - start_time,
    )
    print(f"\nVectorized Training Complete! Total wall time: {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train one from-scratch PPO policy.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
