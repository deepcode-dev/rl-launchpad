# ruff: noqa: E402
"""Passive MuJoCo rendering of a checkpoint evaluated in the production wrapper."""

import argparse
from pathlib import Path
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.evaluate import DEFAULT_EVAL_SEED, load_actor_critic_checkpoint, load_config
from ppo.env import MJXVectorPyTorchWrapper


def launch_interactive_3d_viewer(
    checkpoint_path: str | Path,
    *,
    seed: int = DEFAULT_EVAL_SEED,
    config_path: str | Path = "configs/default.yaml",
) -> None:
    """Shows the same native-reward policy/env contract used by evaluation.

    The MuJoCo viewer is display-only: its mouse perturbations modify a separate
    C++ ``MjData`` object and therefore cannot affect the MJX simulation state.
    """
    config = load_config(config_path)
    env_name = config.get("env_name", "Go1JoystickFlatTerrain")
    history_len = config.get("history_len", 5)

    meta_json = str(checkpoint_path) + ".meta.json"
    if Path(meta_json).exists():
        import json
        with open(meta_json) as f:
            meta = json.load(f)
            history_len = meta.get("history_len", history_len)

    env = MJXVectorPyTorchWrapper(
        env_name, num_envs=1, seed=seed,
        history_len=history_len,
        episode_length=config.get("episode_length", 1000),
    )
    agent, metadata = load_actor_critic_checkpoint(
        checkpoint_path,
        env_name=env_name,
        obs_dim=env.observation_dim,
        act_dim=env.action_dim,
        hidden_dim=config.get("hidden_dim", 256),
        critic_obs_dim=env.privileged_observation_dim,
        hidden_sizes=config.get("hidden_sizes"),
    )

    command_array = np.array([0.8, 0.0, 0.0], dtype=np.float32)

    def key_callback(keycode: int) -> None:
        # 265: Up, 264: Down, 263: Left, 262: Right, 87: W, 83: S, 65: A, 68: D, 32: Space
        if keycode in (265, 87):
            command_array[0] = min(command_array[0] + 0.2, 1.5)
            print(f"🎮 Steering: Forward speed vx = {command_array[0]:.2f} m/s", flush=True)
        elif keycode in (264, 83):
            command_array[0] = max(command_array[0] - 0.2, -1.0)
            print(f"🎮 Steering: Reverse speed vx = {command_array[0]:.2f} m/s", flush=True)
        elif keycode in (263, 65):
            command_array[2] = min(command_array[2] + 0.3, 1.5)
            print(f"🎮 Steering: Yaw rate wz = {command_array[2]:.2f} rad/s", flush=True)
        elif keycode in (262, 68):
            command_array[2] = max(command_array[2] - 0.3, -1.5)
            print(f"🎮 Steering: Yaw rate wz = {command_array[2]:.2f} rad/s", flush=True)
        elif keycode == 32:
            command_array[:] = 0.0
            print("🎮 Steering: STOP (command = [0, 0, 0])", flush=True)

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Environment: {env_name} | obs={env.observation_dim} | actions={env.action_dim}")
    if metadata:
        print(f"Checkpoint metadata: {metadata}")
    print("🎮 LIVE KEYBOARD STEERING ACTIVE: Use W/A/S/D or Arrow Keys to drive! (Spacebar to Stop)")

    obs, _ = env.reset(seed=seed)
    mj_model = env.env.mj_model
    mj_data = mujoco.MjData(mj_model)

    episode_steps = 0
    episode_return = 0.0
    target_fps = 50.0  # 50 Hz control loop

    with mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            frame_start = time.perf_counter()
            
            # Inject steering command directly into active observation tensor
            if isinstance(obs, torch.Tensor):
                obs[0, -3:] = torch.from_numpy(command_array)

            with torch.no_grad():
                action, _ = agent.get_action(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_steps += 1
            episode_return += float(reward[0].item())

            # Synchronize 3D viewer with active JAX physics state
            mj_data.qpos[:] = np.asarray(env.states.data.qpos[0])
            mj_data.qvel[:] = np.asarray(env.states.data.qvel[0])
            mujoco.mj_forward(mj_model, mj_data)
            viewer.sync()

            if episode_steps % 50 == 0:
                print(f"step={episode_steps} native_return={episode_return:.3f}")
            if bool(terminated[0].item()) or bool(truncated[0].item()):
                print(f"episode complete: steps={episode_steps}, native_return={episode_return:.3f}")
                obs, _ = env.reset()
                episode_steps = 0
                episode_return = 0.0

            delay = 1.0 / target_fps - (time.perf_counter() - frame_start)
            if delay > 0:
                time.sleep(delay)


def main() -> None:
    parser = argparse.ArgumentParser(description="View a checkpoint using the production MJX wrapper.")
    parser.add_argument("checkpoint", help="ActorCritic checkpoint matching configs/default.yaml")
    parser.add_argument("--seed", type=int, default=DEFAULT_EVAL_SEED)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    launch_interactive_3d_viewer(args.checkpoint, seed=args.seed, config_path=args.config)


if __name__ == "__main__":
    main()
