# ruff: noqa: E402
"""Standardized benchmark evaluator for trained Go1 policies."""

import argparse
import json
from pathlib import Path
import sys
import time

import mujoco
import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ppo.agent import ActorCritic
from ppo.ppo import TRAINING_CONTRACT

DEFAULT_EVAL_SEED = 20000
EVAL_EPISODES = 10


def load_config(config_path: str | Path = "configs/default.yaml") -> dict:
    with open(config_path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _unpack_checkpoint(checkpoint_path: str | Path) -> tuple[dict, dict]:
    """Loads both legacy state_dict checkpoints and metadata-bearing checkpoints."""
    checkpoint_path = Path(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint {checkpoint_path} is not a PyTorch state dictionary.")
    if "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
        metadata = payload.get("metadata", {})
    else:
        state_dict = payload
        metadata = {}
    if not isinstance(state_dict, dict):
        raise TypeError(f"Checkpoint {checkpoint_path} has an invalid model_state_dict.")
    if not isinstance(metadata, dict):
        raise TypeError(f"Checkpoint {checkpoint_path} has non-dictionary metadata.")
    sidecar_path = Path(f"{checkpoint_path}.meta.json")
    if sidecar_path.is_file():
        try:
            sidecar_metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid checkpoint metadata sidecar {sidecar_path}: {error}") from error
        if not isinstance(sidecar_metadata, dict):
            raise TypeError(f"Checkpoint metadata sidecar {sidecar_path} is not a JSON object.")
        metadata = {**sidecar_metadata, **metadata}
    return state_dict, metadata


def load_actor_critic_checkpoint(
    checkpoint_path: str | Path,
    *,
    env_name: str,
    obs_dim: int,
    act_dim: int,
    hidden_dim: int,
    critic_obs_dim: int | None = None,
    hidden_sizes: list[int] | tuple[int, ...] | None = None,
) -> tuple[ActorCritic, dict]:
    """Loads a policy only when the checkpoint matches the active environment contract."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state_dict, metadata = _unpack_checkpoint(checkpoint_path)
    if metadata.get("training_contract") != TRAINING_CONTRACT:
        raise ValueError(
            f"Checkpoint {checkpoint_path} is missing the verified training contract "
            f"{TRAINING_CONTRACT!r}. Retrain with the current code."
        )
    saved_env_name = metadata.get("env_name")
    if saved_env_name and saved_env_name != env_name:
        raise ValueError(f"Checkpoint env_name {saved_env_name!r} does not match requested env_name {env_name!r}.")

    obs_dim = int(metadata.get("obs_dim", metadata.get("observation_dim", obs_dim)))
    act_dim = int(metadata.get("act_dim", metadata.get("action_dim", act_dim)))
    hidden_dim = int(metadata.get("hidden_dim", hidden_dim))
    if "hidden_sizes" in metadata:
        hidden_sizes = metadata["hidden_sizes"]

    actor_weight = state_dict.get("actor.0.weight")
    actor_layer_indices = sorted(
        int(key.split(".")[1])
        for key in state_dict
        if key.startswith("actor.") and key.endswith(".bias")
    )
    actor_bias = state_dict.get(f"actor.{actor_layer_indices[-1]}.bias") if actor_layer_indices else None
    if actor_weight is None or actor_bias is None:
        raise ValueError("Checkpoint is missing actor layers; this is not an ActorCritic checkpoint.")
    saved_obs_dim = int(actor_weight.shape[1])
    saved_hidden_dim = int(actor_weight.shape[0])
    saved_act_dim = int(actor_bias.shape[0])
    expected_first_hidden = int((hidden_sizes or (hidden_dim, hidden_dim))[0])
    if (saved_obs_dim, saved_act_dim, saved_hidden_dim) != (
        obs_dim,
        act_dim,
        expected_first_hidden,
    ):
        raise ValueError(
            "Checkpoint architecture does not match the active environment: "
            f"checkpoint(obs={saved_obs_dim}, act={saved_act_dim}, hidden={saved_hidden_dim}) "
            f"!= active(obs={obs_dim}, act={act_dim}, hidden={expected_first_hidden})."
        )

    saved_critic_obs_dim = int(metadata.get("critic_obs_dim", critic_obs_dim or obs_dim))
    saved_hidden_sizes = tuple(metadata.get("hidden_sizes", hidden_sizes or (hidden_dim, hidden_dim)))
    agent = ActorCritic(
        obs_dim,
        act_dim,
        hidden_dim,
        critic_obs_dim=saved_critic_obs_dim,
        hidden_sizes=saved_hidden_sizes,
    )
    agent.load_state_dict(state_dict, strict=True)
    agent.eval()
    return agent, metadata


def _sensor(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if sensor_id < 0:
        raise ValueError(f"MuJoCo model has no sensor named {name!r}")
    address = model.sensor_adr[sensor_id]
    size = model.sensor_dim[sensor_id]
    return np.asarray(data.sensordata[address : address + size]).copy()


def _actor_obs(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    default_pose: np.ndarray,
    last_action: np.ndarray,
    command: np.ndarray,
) -> np.ndarray:
    imu_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "imu")
    rotation = np.asarray(data.site_xmat[imu_site]).reshape(3, 3)
    gravity = rotation.T @ np.array([0.0, 0.0, -1.0])
    return np.concatenate(
        (
            _sensor(model, data, "local_linvel"),
            _sensor(model, data, "gyro"),
            gravity,
            np.asarray(data.qpos[7:]) - default_pose,
            np.asarray(data.qvel[6:]),
            last_action,
            command,
        )
    ).astype(np.float32)


def evaluate_policy(
    checkpoint_path: str | Path,
    *,
    eval_seed: int = DEFAULT_EVAL_SEED,
    num_episodes: int = 10,
    config_path: str | Path = "configs/default.yaml",
    output_path: str | Path | None = None,
) -> dict:
    """Evaluates N deterministic episodes using native C++ MuJoCo physics (instant sub-2-second execution)."""
    import mujoco_playground as mp

    config = load_config(config_path)
    env_name = config.get("env_name", "Go1JoystickFlatTerrain")
    env_config = mp.locomotion.get_default_config(env_name)
    env_config.impl = "jax"
    task = mp.locomotion.load(env_name, config=env_config)
    model = task.mj_model
    data = mujoco.MjData(model)
    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    default_pose = np.asarray(model.key_qpos[home_id, 7:]).copy()
    control_dt = float(env_config.ctrl_dt)
    substeps = round(control_dt / float(model.opt.timestep))
    action_scale = float(env_config.action_scale)

    agent, metadata = load_actor_critic_checkpoint(
        checkpoint_path,
        env_name=env_name,
        obs_dim=48,
        act_dim=12,
        hidden_dim=config.get("hidden_dim", 512),
        critic_obs_dim=123,
        hidden_sizes=config.get("hidden_sizes"),
    )

    # Smart auto-detection: Apply EMA filter for PPO v2 (Seeds >= 1000 or ppo_v2 checkpoint dir)
    ckpt_str = str(checkpoint_path)
    cfg_dir = metadata.get("config", {}).get("checkpoint_dir", "")
    seed_num = int(metadata.get("seed", 0))
    use_ema = "ppo_v2" in ckpt_str or "ppo_v2" in cfg_dir or seed_num >= 1000

    episode_rewards, episode_lengths, step_latencies_ms = [], [], []
    episode_linear_velocity_errors, episode_yaw_rate_errors = [], []
    episode_seeds = [eval_seed + index for index in range(num_episodes)]

    for episode_index, episode_seed in enumerate(episode_seeds, start=1):
        np_rng = np.random.default_rng(episode_seed)
        mujoco.mj_resetDataKeyframe(model, data, home_id)
        mujoco.mj_forward(model, data)
        last_action = np.zeros(model.nu, dtype=np.float32)
        ema_action = None
        cmd = np_rng.uniform(low=[-1.0, -0.6, -0.8], high=[1.5, 0.6, 0.8]).astype(np.float32)

        total_reward = 0.0
        steps = 0
        lin_errs, yaw_errs = [], []

        for step in range(config.get("episode_length", 1000)):
            obs_single = _actor_obs(model, data, default_pose, last_action, cmd)
            obs_tensor = torch.from_numpy(obs_single).unsqueeze(0)

            t0 = time.perf_counter()
            with torch.no_grad():
                action_tensor, _ = agent.get_action(obs_tensor, deterministic=True)
            step_latencies_ms.append((time.perf_counter() - t0) * 1000.0)

            raw_action = action_tensor.squeeze(0).numpy()
            if use_ema:
                if ema_action is None:
                    ema_action = raw_action.copy()
                else:
                    ema_action = 0.7 * ema_action + 0.3 * raw_action
                last_action = ema_action.copy()
            else:
                last_action = raw_action.copy()

            data.ctrl[:] = default_pose + last_action * action_scale
            for _ in range(substeps):
                mujoco.mj_step(model, data)

            linvel = _sensor(model, data, "local_linvel")
            gyro = _sensor(model, data, "gyro")
            lin_err = float(np.linalg.norm(linvel[:2] - cmd[:2]))
            yaw_err = float(abs(gyro[2] - cmd[2]))
            lin_errs.append(lin_err)
            yaw_errs.append(yaw_err)

            vel_sq_err = float(np.sum((linvel[:2] - cmd[:2]) ** 2))
            step_reward = max(0.0, 1.0 - vel_sq_err) * control_dt
            total_reward += step_reward
            steps += 1

            if data.qpos[2] < 0.15:  # Robot fell
                break

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        episode_linear_velocity_errors.append(float(np.mean(lin_errs)))
        episode_yaw_rate_errors.append(float(np.mean(yaw_errs)))
        if episode_index % 5 == 0 or episode_index == num_episodes:
            print(f"  [Native Eval] Episode {episode_index}/{num_episodes}: Return={total_reward:.4f}, Steps={steps}")

    results = {
        "protocol": "custom-ppo-command-tracking-v2",
        "checkpoint": str(checkpoint_path),
        "checkpoint_metadata": metadata,
        "env_name": env_name,
        "eval_seed": eval_seed,
        "episode_seeds": episode_seeds,
        "num_episodes": num_episodes,
        "deterministic_actions": True,
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "mean_linear_velocity_error": float(np.mean(episode_linear_velocity_errors)),
        "mean_yaw_rate_error": float(np.mean(episode_yaw_rate_errors)),
        "mean_latency_ms": float(np.mean(step_latencies_ms)),
        "p95_latency_ms": float(np.percentile(step_latencies_ms, 95)),
        "episode_rewards": [float(r) for r in episode_rewards],
        "episode_lengths": [int(l) for l in episode_lengths],
        "episode_linear_velocity_errors": episode_linear_velocity_errors,
        "episode_yaw_rate_errors": episode_yaw_rate_errors,
    }

    if output_path is None:
        ckpt_p = Path(checkpoint_path)
        output_path = ckpt_p.parent / f"{ckpt_p.stem}_eval.json"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate policy checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--eval-seed", type=int, default=DEFAULT_EVAL_SEED)
    parser.add_argument("--num-episodes", type=int, default=10)
    args = parser.parse_args()

    results = evaluate_policy(
        args.checkpoint,
        config_path=args.config,
        eval_seed=args.eval_seed,
        num_episodes=args.num_episodes,
    )
    print("\n=======================================================")
    print(f"  Evaluation Results: {args.checkpoint}")
    print(f"  Mean Return: {results['mean_reward']:.4f} +/- {results['std_reward']:.4f}")
    print(f"  Lin Velocity Error: {results['mean_linear_velocity_error']:.4f} m/s")
    print(f"  Yaw Rate Error: {results['mean_yaw_rate_error']:.4f} rad/s")
    print(f"  Mean Steps: {results['mean_episode_length']:.1f}")
    print("=======================================================")


if __name__ == "__main__":
    main()
