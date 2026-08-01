# ruff: noqa: E402
"""Stable-Baselines3 PPO correctness baseline.

This is deliberately not the submitted from-scratch agent.  It consumes the
same production wrapper, bounded action contract, and native reward protocol so
that a result is a meaningful regression check rather than a different task.
"""

import argparse
import json
from pathlib import Path
import sys
import time

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO as SB3PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.evaluate import DEFAULT_EVAL_SEED, EVAL_EPISODES, load_config
from ppo.env import MJXVectorPyTorchWrapper


class RolloutMetricsCallback(BaseCallback):
    """Record measured native rollout reward against environment steps."""

    def __init__(self):
        super().__init__(verbose=0)
        self.started_at = time.perf_counter()
        self.history = {"total_steps": [], "mean_step_rewards": [], "wall_times": []}
        self._raw_rollout_rewards: list[float] = []

    def _on_rollout_start(self) -> None:
        self._raw_rollout_rewards = []

    def _on_step(self) -> bool:
        # Capture rewards before SB3 adds a value bootstrap to time-limit
        # transitions in its rollout buffer.  This keeps the published curve on
        # the same unmodified native-reward scale as the custom trainer.
        rewards = np.asarray(self.locals["rewards"], dtype=np.float64)
        self._raw_rollout_rewards.extend(rewards.reshape(-1).tolist())
        return True

    def _on_rollout_end(self) -> None:
        if not self._raw_rollout_rewards:
            raise RuntimeError("SB3 rollout ended without raw reward samples")
        self.history["total_steps"].append(int(self.num_timesteps))
        self.history["mean_step_rewards"].append(float(np.mean(self._raw_rollout_rewards)))
        self.history["wall_times"].append(float(time.perf_counter() - self.started_at))


class SB3GymWrapper(gym.Env):
    """Single-environment Gymnasium view of the project's production MJX wrapper."""

    metadata = {"render_modes": []}

    def __init__(
        self, env_name: str, seed: int, *, history_len: int = 5,
        episode_length: int = 1000,
    ):
        super().__init__()
        self._env = MJXVectorPyTorchWrapper(
            env_name, num_envs=1, seed=seed,
            history_len=history_len, episode_length=episode_length,
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._env.observation_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self._env.action_dim,), dtype=np.float32
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        del options
        obs, info = self._env.reset(seed=seed)
        return obs.squeeze(0).numpy(), info

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float32), self.action_space.low, self.action_space.high)
        obs, reward, terminated, truncated, info = self._env.step(action)
        terminated_value = bool(terminated.item())
        truncated_value = bool(truncated.item())
        if terminated_value or truncated_value:
            # The production vector wrapper has already reset this slot for the
            # next call. Gymnasium/SB3 still needs the real terminal observation
            # for this transition (especially to bootstrap time-limit values).
            terminal_observation = info["final_observation"].squeeze(0).numpy()
            info = dict(info, terminal_observation=terminal_observation)
            observation = terminal_observation
        else:
            observation = obs.squeeze(0).numpy()
        return (
            observation,
            float(reward.item()),
            terminated_value,
            truncated_value,
            info,
        )


class SB3MJXVecEnv(VecEnv):
    """SB3 VecEnv view of one batched MJX environment.

    The underlying production wrapper already autoresets completed slots.  This
    adapter converts Gymnasium's terminated/truncated split into SB3's VecEnv
    done flag while retaining the real terminal observation and time-limit bit.
    """

    def __init__(
        self, env_name: str, seed: int, *, num_envs: int,
        history_len: int = 5, episode_length: int = 1000,
    ):
        self._default_seed = int(seed)
        self._env = MJXVectorPyTorchWrapper(
            env_name, num_envs=num_envs, seed=seed,
            history_len=history_len, episode_length=episode_length,
        )
        self._pending_actions: np.ndarray | None = None
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._env.observation_dim,), dtype=np.float32
        )
        action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self._env.action_dim,), dtype=np.float32
        )
        super().__init__(num_envs, observation_space, action_space)

    def reset(self) -> np.ndarray:
        seed = next((value for value in self._seeds if value is not None), self._default_seed)
        obs, _ = self._env.reset(seed=seed)
        self.reset_infos = [{} for _ in range(self.num_envs)]
        self._reset_seeds()
        self._reset_options()
        return obs.numpy()

    def step_async(self, actions: np.ndarray) -> None:
        self._pending_actions = np.asarray(actions, dtype=np.float32)

    def step_wait(self):
        if self._pending_actions is None:
            raise RuntimeError("step_wait called before step_async")
        obs, rewards, terminated, truncated, raw_info = self._env.step(self._pending_actions)
        self._pending_actions = None
        terminated_np = terminated.numpy().astype(bool, copy=False)
        truncated_np = truncated.numpy().astype(bool, copy=False)
        dones = terminated_np | truncated_np
        final_observations = raw_info["final_observation"].numpy()
        infos: list[dict] = []
        for index in range(self.num_envs):
            info = {"TimeLimit.truncated": bool(truncated_np[index] and not terminated_np[index])}
            if dones[index]:
                info["terminal_observation"] = final_observations[index]
            infos.append(info)
        return obs.numpy(), rewards.numpy(), dones, infos

    def close(self) -> None:
        self._pending_actions = None

    def get_attr(self, attr_name: str, indices=None):
        values = list(self._get_indices(indices))
        if attr_name == "render_mode":
            value = None
        else:
            value = getattr(self._env, attr_name)
        return [value for _ in values]

    def set_attr(self, attr_name: str, value, indices=None) -> None:
        selected = list(self._get_indices(indices))
        if selected != list(range(self.num_envs)):
            raise NotImplementedError("Batched MJX attributes can only be set for every slot")
        setattr(self._env, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):
        selected = list(self._get_indices(indices))
        result = getattr(self._env, method_name)(*method_args, **method_kwargs)
        return [result for _ in selected]

    def env_is_wrapped(self, wrapper_class, indices=None):
        del wrapper_class
        return [False for _ in self._get_indices(indices)]


def train_sb3_seed(seed: int, config: dict) -> Path:
    """Trains one SB3 comparison checkpoint, never used by custom-PPO evaluation."""
    env_name = config.get("env_name", "Go1JoystickFlatTerrain")
    num_envs = int(config.get("num_envs", 128))
    n_steps = int(config.get("baseline_n_steps", 128))
    rollout_size = num_envs * n_steps
    total_timesteps = int(config["total_timesteps_per_seed"])
    if total_timesteps % rollout_size:
        raise ValueError(
            "total_timesteps_per_seed must be divisible by baseline_n_steps * num_envs "
            "so SB3 does not exceed the matched step budget"
        )
    env = SB3MJXVecEnv(
        env_name, seed, num_envs=num_envs,
        history_len=config.get("history_len", 5),
        episode_length=config.get("episode_length", 1000),
    )
    model = SB3PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=config.get("pi_lr", 3e-4),
        n_steps=n_steps,
        batch_size=config.get("batch_size", 256),
        n_epochs=config.get("train_iters", 10),
        gamma=config.get("gamma", 0.99),
        gae_lambda=config.get("lam", 0.95),
        clip_range=config.get("clip_ratio", 0.2),
        max_grad_norm=config.get("max_grad_norm", 0.5),
        seed=seed,
        device="cpu",
    )
    start = time.perf_counter()
    callback = RolloutMetricsCallback()
    model.learn(total_timesteps=total_timesteps, callback=callback)
    output = PROJECT_ROOT / "baselines" / f"sb3_ppo_seed{seed}"
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(output)
    metadata = {
        "protocol": "sb3-baseline-native-reward-v1",
        "baseline_only": True,
        "env_name": env_name,
        "observation_dim": env.observation_space.shape[0],
        "action_dim": env.action_space.shape[0],
        "num_envs": env.num_envs,
        "rollout_size": rollout_size,
        "history_len": int(config.get("history_len", 5)),
        "episode_length": int(config.get("episode_length", 1000)),
        "reward_source": "mujoco_playground_state_reward",
        "physics_backend": "jax",
        "training_seed": seed,
        "total_env_steps": int(model.num_timesteps),
        "wall_time_seconds": time.perf_counter() - start,
    }
    output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    training_history = {"seed": seed, **callback.history}
    output.with_name(output.name + "_training.json").write_text(
        json.dumps(training_history, indent=2), encoding="utf-8"
    )
    return output.with_suffix(".zip")


def evaluate_sb3_policy(model_path: str | Path, *, eval_seed: int) -> dict:
    """Uses the identical 50-episode deterministic native-reward protocol."""
    config = load_config()
    env_name = config.get("env_name", "Go1JoystickFlatTerrain")
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"SB3 baseline checkpoint not found: {model_path}")
    metadata_path = model_path.with_suffix(".metadata.json")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"SB3 baseline metadata not found: {metadata_path}")
    checkpoint_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if checkpoint_metadata.get("protocol") != "sb3-baseline-native-reward-v1":
        raise ValueError(f"Incompatible SB3 metadata protocol in {metadata_path}")

    env = SB3GymWrapper(
        env_name, seed=eval_seed, history_len=config.get("history_len", 5),
        episode_length=config.get("episode_length", 1000),
    )
    model = SB3PPO.load(model_path, device="cpu")
    if model.observation_space.shape != env.observation_space.shape:
        raise ValueError("SB3 checkpoint observation shape does not match evaluation environment")
    if model.action_space.shape != env.action_space.shape:
        raise ValueError("SB3 checkpoint action shape does not match evaluation environment")
    episode_seeds = [eval_seed + index for index in range(EVAL_EPISODES)]
    returns, lengths = [], []
    for episode_seed in episode_seeds:
        obs, _ = env.reset(seed=episode_seed)
        done = False
        total_return = 0.0
        length = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_return += reward
            length += 1
            done = terminated or truncated
        returns.append(total_return)
        lengths.append(length)

    result = {
        "protocol": "sb3-baseline-native-reward-v1",
        "baseline_only": True,
        "checkpoint": str(model_path),
        "checkpoint_metadata": checkpoint_metadata,
        "training_seed": int(checkpoint_metadata["training_seed"]),
        "env_name": env_name,
        "eval_seed": eval_seed,
        "episode_seeds": episode_seeds,
        "num_episodes": EVAL_EPISODES,
        "deterministic_actions": True,
        "mean_reward": float(np.mean(returns)),
        "std_reward": float(np.std(returns)),
        "mean_episode_length": float(np.mean(lengths)),
        "episode_rewards": [float(value) for value in returns],
        "episode_lengths": lengths,
    }
    output = model_path.with_name(model_path.stem + "_eval.json")
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate SB3 only as a correctness baseline.")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    config = load_config()
    seeds = config.get("seeds", [])
    if not seeds:
        raise ValueError("configs/default.yaml has no seeds.")

    model_paths = []
    for seed in seeds:
        path = PROJECT_ROOT / "baselines" / f"sb3_ppo_seed{seed}.zip"
        model_paths.append(path if args.evaluate_only else train_sb3_seed(seed, config))

    if not args.evaluate_only:
        histories = []
        for seed in seeds:
            history_path = PROJECT_ROOT / "baselines" / f"sb3_ppo_seed{seed}_training.json"
            histories.append(json.loads(history_path.read_text(encoding="utf-8")))
        (PROJECT_ROOT / "baselines" / "sb3_training_results.json").write_text(
            json.dumps(histories, indent=2), encoding="utf-8"
        )

    missing = [str(path) for path in model_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Cannot run the baseline comparison; missing:\n" + "\n".join(missing))
    results = [
        evaluate_sb3_policy(path, eval_seed=DEFAULT_EVAL_SEED)
        for path in model_paths
    ]
    summary = {
        "protocol": "sb3-baseline-native-reward-v1",
        "baseline_only": True,
        "num_episodes_per_seed": EVAL_EPISODES,
        "training_seeds": seeds,
        "grand_mean_return": float(np.mean([result["mean_reward"] for result in results])),
        "grand_std_return": float(np.std([result["mean_reward"] for result in results])),
        "mean_episode_length": float(np.mean([result["mean_episode_length"] for result in results])),
        "seed_results": results,
    }
    output = PROJECT_ROOT / "baselines" / "sb3_ppo_eval_summary.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved SB3 baseline-only summary to {output}")


if __name__ == "__main__":
    main()
