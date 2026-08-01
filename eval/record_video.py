# ruff: noqa: E402
"""Record an offscreen demo from the same checkpoint/environment used by evaluation."""

import argparse
from pathlib import Path
import sys

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.evaluate import DEFAULT_EVAL_SEED, load_actor_critic_checkpoint, load_config
from ppo.env import MJXVectorPyTorchWrapper


def record_demo_video(
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    seed: int = DEFAULT_EVAL_SEED,
    max_steps: int = 500,
    fps: int = 50,
    width: int = 960,
    height: int = 720,
) -> dict:
    """Render one deterministic episode and return its measured rollout facts."""
    ckpt_data = torch.load(checkpoint_path, map_location="cpu")
    meta = ckpt_data.get("metadata", {})
    ckpt_config = meta.get("config", {})
    
    config = load_config()
    history_len = int(ckpt_config.get("history_len", meta.get("history_len", config.get("history_len", 1))))
    hidden_dim = int(ckpt_config.get("hidden_dim", meta.get("hidden_dim", config.get("hidden_dim", 512))))
    hidden_sizes = ckpt_config.get("hidden_sizes", meta.get("hidden_sizes", config.get("hidden_sizes", (512, 256, 128))))
    if isinstance(hidden_sizes, list):
        hidden_sizes = tuple(hidden_sizes)

    env_name = ckpt_config.get("env_name", config.get("env_name", "Go1JoystickFlatTerrain"))
    env = MJXVectorPyTorchWrapper(
        env_name, num_envs=1, seed=seed,
        history_len=history_len,
        episode_length=config.get("episode_length", 1000),
    )
    agent, _ = load_actor_critic_checkpoint(
        checkpoint_path, env_name=env_name, obs_dim=env.observation_dim,
        act_dim=env.action_dim, hidden_dim=hidden_dim, hidden_sizes=hidden_sizes,
    )
    obs, _ = env.reset(seed=seed)

    model = env.env.mj_model
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    frames: list[np.ndarray] = []
    episode_return = 0.0

    try:
        for step_i in range(max_steps):
            with torch.no_grad():
                action, _ = agent.get_action(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_return += float(reward.item())
            if bool(terminated.item()) or bool(truncated.item()):
                break

            data.qpos[:] = np.asarray(env.states.data.qpos[0])
            data.qvel[:] = np.asarray(env.states.data.qvel[0])
            mujoco.mj_forward(model, data)
            # The task XML provides a body-tracking camera; the renderer's
            # default free camera allows the robot to leave frame while moving.
            renderer.update_scene(data, camera="track")
            frames.append(renderer.render().copy())
    finally:
        renderer.close()

    if not frames:
        raise RuntimeError("The policy terminated before a renderable frame was produced")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, frames, fps=fps)
    result = {
        "checkpoint": str(checkpoint_path),
        "seed": seed,
        "frames": len(frames),
        "fps": fps,
        "native_return": episode_return,
    }
    print(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Record the reported custom-PPO checkpoint.")
    parser.add_argument("checkpoint")
    parser.add_argument("--output", default="write-up/demo.mp4")
    parser.add_argument("--seed", type=int, default=DEFAULT_EVAL_SEED)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--fps", type=int, default=50)
    args = parser.parse_args()
    record_demo_video(
        args.checkpoint, args.output, seed=args.seed,
        max_steps=args.max_steps, fps=args.fps,
    )


if __name__ == "__main__":
    main()
