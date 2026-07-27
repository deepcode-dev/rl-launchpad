"""Real-time native MuJoCo preview for a Brax PPO checkpoint."""

import argparse
from collections import deque
from pathlib import Path
import time

from brax.training.agents.ppo import checkpoint as ppo_checkpoint
import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import mujoco_playground as mp
import numpy as np

from eval.view_native import _actor_observation, _sensor


def launch(checkpoint_path: str, command: tuple[float, float, float], max_steps: int = 100000) -> None:
    env_name = "Go1JoystickFlatTerrain"
    env_config = mp.locomotion.get_default_config(env_name)
    env_config.impl = "jax"
    task = mp.locomotion.load(env_name, config=env_config)
    model = task.mj_model
    data = mujoco.MjData(model)
    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    default_pose = np.asarray(model.key_qpos[home_id, 7:]).copy()
    command_array = np.asarray(command, dtype=np.float32)
    ckpt_path = Path(checkpoint_path).resolve()
    if ckpt_path.is_dir() and not (ckpt_path / "ppo_network_config.json").exists():
        subdirs = sorted([d for d in ckpt_path.iterdir() if d.is_dir() and (d / "ppo_network_config.json").exists()])
        if subdirs:
            ckpt_path = subdirs[-1].resolve()
            print(f"Auto-selected latest Brax step checkpoint: {ckpt_path}")

    # Fix Brax Orbax JSON null mean_kernel_init_fn key
    json_path = ckpt_path / "ppo_network_config.json"
    if json_path.exists():
        import json
        with open(json_path, "r") as f:
            cfg_data = json.load(f)
        if cfg_data.get("network_factory_kwargs", {}).get("mean_kernel_init_fn") is None:
            cfg_data["network_factory_kwargs"]["mean_kernel_init_fn"] = "lecun_uniform"
            with open(json_path, "w") as f:
                json.dump(cfg_data, f)

    inference_fn = jax.jit(ppo_checkpoint.load_policy(str(ckpt_path.resolve())))
    rng = jax.random.PRNGKey(0)
    control_dt = float(env_config.ctrl_dt)
    substeps = round(control_dt / float(model.opt.timestep))
    action_scale = float(env_config.action_scale)
    observation_history: deque[np.ndarray] = deque(maxlen=1)
    last_action = np.zeros(model.nu, dtype=np.float32)

    def reset() -> None:
        nonlocal last_action
        mujoco.mj_resetDataKeyframe(model, data, home_id)
        mujoco.mj_forward(model, data)
        last_action = np.zeros(model.nu, dtype=np.float32)
        observation_history.clear()
        observation_history.append(
            _actor_observation(
                model, data, default_pose, last_action, command_array
            )
        )

    reset()
    print(f"Brax checkpoint: {checkpoint_path}")
    print("Controls: Up/Down/Left/Right or WASD to steer live! Spacebar to stop.")

    def key_callback(keycode: int) -> None:
        # Keycodes: Up=265/W=87, Down=264/S=83, Left=263/A=65, Right=262/D=68, Space=32
        if keycode in (265, 87, 119):  # Up / W
            command_array[0] = min(command_array[0] + 0.2, 1.5)
        elif keycode in (264, 83, 115):  # Down / S
            command_array[0] = max(command_array[0] - 0.2, -0.8)
        elif keycode in (263, 65, 97):  # Left / A
            command_array[2] = min(command_array[2] + 0.2, 1.2)
        elif keycode in (262, 68, 100):  # Right / D
            command_array[2] = max(command_array[2] - 0.2, -1.2)
        elif keycode == 32:  # Spacebar
            command_array[0] = 0.0
            command_array[1] = 0.0
            command_array[2] = 0.0
        print(f"\rCurrent Command [vx={command_array[0]:.1f}, vy={command_array[1]:.1f}, yaw={command_array[2]:.1f}]", end="")

    step = 0
    timing_window_start = time.perf_counter()
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            frame_start = time.perf_counter()
            observation = {
                "state": jnp.asarray(np.concatenate(tuple(observation_history)))[
                    None, :
                ]
            }
            action, _ = inference_fn(observation, rng)
            last_action = np.asarray(action[0])
            data.ctrl[:] = default_pose + last_action * action_scale
            for _ in range(substeps):
                mujoco.mj_step(model, data)
            observation_history.append(
                _actor_observation(
                    model, data, default_pose, last_action, command_array
                )
            )
            viewer.sync()
            step += 1

            if step % 50 == 0:
                elapsed = time.perf_counter() - timing_window_start
                print(
                    f"step={step} simulated_fps={50 / elapsed:.1f} "
                    f"position=({data.qpos[0]:.2f}, {data.qpos[1]:.2f})"
                )
                timing_window_start = time.perf_counter()
            if _sensor(model, data, "upvector")[-1] < 0.0 or step >= max_steps:
                print(f"\nreset at step={step}")
                reset()
                step = 0
                timing_window_start = time.perf_counter()

            remaining = control_dt - (time.perf_counter() - frame_start)
            if remaining > 0:
                time.sleep(remaining)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "checkpoint",
        help="Exact numbered Brax checkpoint directory containing ppo_network_config.json",
    )
    parser.add_argument(
        "--command",
        nargs=3,
        type=float,
        metavar=("VX", "VY", "YAW"),
        default=(0.8, 0.0, 0.0),
    )
    parser.add_argument("--max-steps", type=int, default=100000, help="Maximum steps before episode reset (default: 100,000)")
    args = parser.parse_args()
    launch(args.checkpoint, tuple(args.command), max_steps=args.max_steps)


if __name__ == "__main__":
    main()
