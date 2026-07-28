# ruff: noqa: E402
"""Real-time native MuJoCo preview for PPO v2 policies (includes low-pass EMA filter).

The exact evaluator uses MJX. Native MuJoCo is used here for responsive
visual inspection of PPO v2 checkpoints on Windows.
"""

import argparse
from collections import deque
from pathlib import Path
import sys
import time

import mujoco
import mujoco.viewer
import mujoco_playground as mp
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.evaluate import load_actor_critic_checkpoint, load_config


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


def launch_native_viewer_v2(
    checkpoint_path: str | Path,
    *,
    config_path: str | Path,
    command: tuple[float, float, float],
) -> None:
    config = load_config(config_path)
    env_name = config.get("env_name", "Go1JoystickFlatTerrain")
    if env_name != "Go1JoystickFlatTerrain":
        raise ValueError("The native real-time preview currently supports Go1 flat terrain only")

    env_config = mp.locomotion.get_default_config(env_name)
    env_config.impl = "jax"
    task = mp.locomotion.load(env_name, config=env_config)
    model = task.mj_model
    data = mujoco.MjData(model)
    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_id < 0:
        raise ValueError("Go1 model is missing its home keyframe")
    default_pose = np.asarray(model.key_qpos[home_id, 7:]).copy()

    history_len = int(config.get("history_len", 1))
    meta_json = str(checkpoint_path) + ".meta.json"
    if Path(meta_json).exists():
        import json
        with open(meta_json) as f:
            meta = json.load(f)
            history_len = meta.get("history_len", history_len)

    obs_dim = 48 * history_len
    agent, _ = load_actor_critic_checkpoint(
        checkpoint_path,
        env_name=env_name,
        obs_dim=obs_dim,
        act_dim=model.nu,
        hidden_dim=config.get("hidden_dim", 256),
        critic_obs_dim=123,
        hidden_sizes=config.get("hidden_sizes"),
    )
    command_array = np.asarray(command, dtype=np.float32)
    control_dt = float(env_config.ctrl_dt)
    substeps = round(control_dt / float(model.opt.timestep))
    action_scale = float(env_config.action_scale)

    last_action = np.zeros(model.nu, dtype=np.float32)
    observation_history: deque[np.ndarray] = deque(maxlen=history_len)

    def reset() -> None:
        nonlocal last_action
        mujoco.mj_resetDataKeyframe(model, data, home_id)
        mujoco.mj_forward(model, data)
        last_action = np.zeros(model.nu, dtype=np.float32)
        state = _actor_observation(
            model, data, default_pose, last_action, command_array
        )
        observation_history.clear()
        observation_history.extend(state.copy() for _ in range(history_len))

    def key_callback(keycode: int) -> None:
        # Keycodes: Up=265/W=87, Down=264/S=83, Left=263/A=65, Right=262/D=68, Q=81, E=69, 1=49, 2=50, 3=51, 4=52, Space=32
        if keycode in (265, 87):  # Forward (Up / W)
            command_array[0] = min(command_array[0] + 0.1, 2.0)
        elif keycode in (264, 83):  # Reverse (Down / S)
            command_array[0] = max(command_array[0] - 0.1, -1.0)
        elif keycode in (263, 65):  # Yaw Left (Left / A)
            command_array[2] = min(command_array[2] + 0.2, 2.0)
        elif keycode in (262, 68):  # Yaw Right (Right / D)
            command_array[2] = max(command_array[2] - 0.2, -2.0)
        elif keycode == 81:  # Strafe Left (Q)
            command_array[1] = min(command_array[1] + 0.1, 1.0)
        elif keycode == 69:  # Strafe Right (E)
            command_array[1] = max(command_array[1] - 0.1, -1.0)
        elif keycode == 49:  # Preset 1: 0.5 m/s
            command_array[0], command_array[1], command_array[2] = 0.5, 0.0, 0.0
        elif keycode == 50:  # Preset 2: 1.0 m/s
            command_array[0], command_array[1], command_array[2] = 1.0, 0.0, 0.0
        elif keycode == 51:  # Preset 3: 1.5 m/s
            command_array[0], command_array[1], command_array[2] = 1.5, 0.0, 0.0
        elif keycode == 52:  # Preset 4: 2.0 m/s
            command_array[0], command_array[1], command_array[2] = 2.0, 0.0, 0.0
        elif keycode == 32:  # Spacebar: STOP
            command_array[:] = 0.0
        print(f"\r🎮 Command [vx={command_array[0]:.2f} m/s, vy={command_array[1]:.2f} m/s, yaw={command_array[2]:.2f} rad/s]", flush=True)

    reset()
    print(f"PPO v2 Checkpoint: {checkpoint_path}")
    print(f"Command [vx, vy, yaw]: {command_array.tolist()}")
    print("🎮 LIVE KEYBOARD STEERING ACTIVE (PPO v2 EMA Smooth Filter Enabled): Use W/A/S/D or Arrow Keys to drive!")
    print("Backend: native MuJoCo real-time preview (exact scoring still uses MJX)")

    step = 0
    timing_window_start = time.perf_counter()
    ema_action = None
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            frame_start = time.perf_counter()
            flat_observation = np.concatenate(tuple(observation_history))
            observation = torch.from_numpy(flat_observation).unsqueeze(0)
            with torch.no_grad():
                action, _ = agent.get_action(observation, deterministic=True)
            raw_action = action.squeeze(0).numpy()
            if ema_action is None:
                ema_action = raw_action.copy()
            else:
                ema_action = 0.7 * ema_action + 0.3 * raw_action
            last_action = ema_action.copy()
            data.ctrl[:] = default_pose + last_action * action_scale
            for _ in range(substeps):
                mujoco.mj_step(model, data)

            state = _actor_observation(
                model, data, default_pose, last_action, command_array
            )
            observation_history.append(state)
            viewer.sync()
            step += 1

            if step % 50 == 0:
                elapsed = time.perf_counter() - timing_window_start
                print(
                    f"step={step} simulated_fps={50 / elapsed:.1f} "
                    f"position=({data.qpos[0]:.2f}, {data.qpos[1]:.2f})"
                )
                timing_window_start = time.perf_counter()

            if _sensor(model, data, "upvector")[-1] < 0.0 or step >= 100000:
                print(f"reset at step={step}")
                reset()
                step = 0
                timing_window_start = time.perf_counter()

            remaining = control_dt - (time.perf_counter() - frame_start)
            if remaining > 0:
                time.sleep(remaining)


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time native MuJoCo policy preview for PPO v2")
    parser.add_argument("checkpoint")
    parser.add_argument("--config", default="configs/champion_v2.yaml")
    parser.add_argument(
        "--command",
        nargs=3,
        type=float,
        metavar=("VX", "VY", "YAW"),
        default=(0.8, 0.0, 0.0),
    )
    args = parser.parse_args()
    launch_native_viewer_v2(
        args.checkpoint,
        config_path=args.config,
        command=tuple(args.command),
    )


if __name__ == "__main__":
    main()
