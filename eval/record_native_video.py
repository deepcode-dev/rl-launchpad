# ruff: noqa: E402
"""Record an offscreen video using native MuJoCo C physics (identical to view_native_v2.py)."""

import argparse
from collections import deque
from pathlib import Path
import sys

import imageio.v2 as imageio
import mujoco
import mujoco_playground as mp
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.evaluate import load_actor_critic_checkpoint


def _sensor(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if sensor_id < 0:
        raise ValueError(f"MuJoCo model has no sensor named {name!r}")
    address = model.sensor_adr[sensor_id]
    size = model.sensor_dim[sensor_id]
    return np.asarray(data.sensordata[address : address + size]).copy()


def _actor_observation(
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


def record_native_demo_video(
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    command: tuple[float, float, float] = (1.0, 0.0, 0.0),
    max_steps: int = 500,
    fps: int = 50,
    width: int = 960,
    height: int = 720,
) -> dict:
    """Render a smooth native MuJoCo episode and save video/gif."""
    env_name = "Go1JoystickFlatTerrain"
    env_config = mp.locomotion.get_default_config(env_name)
    env_config.impl = "jax"
    task = mp.locomotion.load(env_name, config=env_config)
    model = task.mj_model
    data = mujoco.MjData(model)

    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    default_pose = np.asarray(model.key_qpos[home_id, 7:]).copy()

    ckpt_data = torch.load(checkpoint_path, map_location="cpu")
    meta = ckpt_data.get("metadata", {})
    ckpt_config = meta.get("config", {})
    history_len = int(ckpt_config.get("history_len", meta.get("history_len", 1)))
    hidden_dim = int(ckpt_config.get("hidden_dim", meta.get("hidden_dim", 512)))
    hidden_sizes = ckpt_config.get("hidden_sizes", meta.get("hidden_sizes", (512, 256, 128)))
    if isinstance(hidden_sizes, list):
        hidden_sizes = tuple(hidden_sizes)

    obs_dim = 48 * history_len
    agent, _ = load_actor_critic_checkpoint(
        checkpoint_path,
        env_name=env_name,
        obs_dim=obs_dim,
        act_dim=model.nu,
        hidden_dim=hidden_dim,
        critic_obs_dim=123,
        hidden_sizes=hidden_sizes,
    )

    command_array = np.asarray(command, dtype=np.float32)
    control_dt = float(env_config.ctrl_dt)
    substeps = round(control_dt / float(model.opt.timestep))
    action_scale = float(env_config.action_scale)

    mujoco.mj_resetDataKeyframe(model, data, home_id)
    mujoco.mj_forward(model, data)
    last_action = np.zeros(model.nu, dtype=np.float32)
    state = _actor_observation(model, data, default_pose, last_action, command_array)

    observation_history: deque[np.ndarray] = deque(maxlen=history_len)
    observation_history.extend(state.copy() for _ in range(history_len))

    renderer = mujoco.Renderer(model, height=height, width=width)
    frames: list[np.ndarray] = []
    ema_action = None

    try:
        for _ in range(max_steps):
            flat_obs = np.concatenate(tuple(observation_history))
            obs_tensor = torch.from_numpy(flat_obs).unsqueeze(0)
            with torch.no_grad():
                action, _ = agent.get_action(obs_tensor, deterministic=True)
            raw_action = action.squeeze(0).numpy()

            if ema_action is None:
                ema_action = raw_action.copy()
            else:
                ema_action = 0.7 * ema_action + 0.3 * raw_action

            last_action = ema_action.copy()
            data.ctrl[:] = default_pose + last_action * action_scale
            for _ in range(substeps):
                mujoco.mj_step(model, data)

            next_state = _actor_observation(
                model, data, default_pose, last_action, command_array
            )
            observation_history.append(next_state)

            renderer.update_scene(data, camera="track")
            frames.append(renderer.render().copy())

            if _sensor(model, data, "upvector")[-1] < 0.0:
                break
    finally:
        renderer.close()

    if not frames:
        raise RuntimeError("No renderable frames produced")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, frames, fps=fps)
    print(f"Rendered smooth native MuJoCo video: {output_path} ({len(frames)} frames)")
    return {"output": str(output_path), "frames": len(frames)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Record silky-smooth native MuJoCo video.")
    parser.add_argument("checkpoint")
    parser.add_argument("--output", default="write-up/demo-seed13039.mp4")
    parser.add_argument("--command", nargs=3, type=float, default=(1.0, 0.0, 0.0))
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--fps", type=int, default=50)
    args = parser.parse_args()
    record_native_demo_video(
        args.checkpoint,
        args.output,
        command=tuple(args.command),
        max_steps=args.max_steps,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
