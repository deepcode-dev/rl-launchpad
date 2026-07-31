"""Evaluate Brax 200M PPO baseline checkpoints for Rule R2 compliance."""

import argparse
import json
from pathlib import Path
import sys
import time

from brax.training.agents.ppo import checkpoint as ppo_checkpoint
import jax
import jax.numpy as jnp
import mujoco
import mujoco_playground as mp
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.evaluate import DEFAULT_EVAL_SEED
from eval.view_native import _actor_observation, _sensor


def evaluate_brax_checkpoint(ckpt_path: Path, eval_seed: int = DEFAULT_EVAL_SEED, num_episodes: int = 50) -> dict:
    env_name = "Go1JoystickFlatTerrain"
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

    json_path = ckpt_path / "ppo_network_config.json"
    if json_path.exists():
        with open(json_path, "r") as f:
            cfg_data = json.load(f)
        if cfg_data.get("network_factory_kwargs", {}).get("mean_kernel_init_fn") is None:
            cfg_data["network_factory_kwargs"]["mean_kernel_init_fn"] = "lecun_uniform"
            with open(json_path, "w") as f:
                json.dump(cfg_data, f)

    inference_fn = jax.jit(ppo_checkpoint.load_policy(str(ckpt_path.resolve())))
    rng = jax.random.PRNGKey(0)

    episode_rewards, episode_lengths = [], []
    episode_lin_errs, episode_yaw_errs = [], []
    episode_seeds = [eval_seed + idx for idx in range(num_episodes)]

    for ep_index, ep_seed in enumerate(episode_seeds, start=1):
        np_rng = np.random.default_rng(ep_seed)
        mujoco.mj_resetDataKeyframe(model, data, home_id)
        mujoco.mj_forward(model, data)
        last_action = np.zeros(model.nu, dtype=np.float32)
        cmd = np_rng.uniform(low=[-1.0, -0.6, -0.8], high=[1.5, 0.6, 0.8]).astype(np.float32)

        ep_reward = 0.0
        ep_steps = 0
        lin_errs, yaw_errs = [], []

        for step in range(1000):
            obs = _actor_observation(model, data, default_pose, last_action, cmd)
            rng, act_rng = jax.random.split(rng)
            obs_dict = {"state": jnp.asarray(obs)[None, :]}
            action, _ = inference_fn(obs_dict, act_rng)
            last_action = np.asarray(action[0])

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
            ep_reward += step_reward
            ep_steps += 1

            if data.qpos[2] < 0.15:  # Robot fell
                break

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_steps)
        episode_lin_errs.append(float(np.mean(lin_errs)))
        episode_yaw_errs.append(float(np.mean(yaw_errs)))
        if ep_index % 10 == 0 or ep_index == num_episodes:
            print(f"  [Brax 200M Eval] Episode {ep_index}/{num_episodes}: Return={ep_reward:.4f}, Steps={ep_steps}")

    return {
        "checkpoint": str(ckpt_path),
        "num_episodes": num_episodes,
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "mean_lin_err": float(np.mean(episode_lin_errs)),
        "mean_yaw_err": float(np.mean(episode_yaw_errs)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Brax 200M PPO baseline checkpoints.")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=PROJECT_ROOT / "baselines" / "brax_go1_200m",
        help="Directory containing Brax step checkpoints (default: baselines/brax_go1_200m).",
    )
    args = parser.parse_args()

    brax_dir = args.checkpoint_dir
    if not brax_dir.exists():
        raise FileNotFoundError(f"Brax checkpoint directory not found: {brax_dir}")

    step_dirs = sorted([d for d in brax_dir.iterdir() if d.is_dir() and (d / "ppo_network_config.json").exists()])
    if not step_dirs:
        # Check subdirectories
        step_dirs = sorted([d for d in brax_dir.rglob("*") if d.is_dir() and (d / "ppo_network_config.json").exists()])

    print(f"Found {len(step_dirs)} Brax 200M step checkpoints to evaluate.")
    latest_ckpt = step_dirs[-1]
    print(f"Evaluating Brax 200M checkpoint: {latest_ckpt.name} over 50 fixed episodes...")

    result = evaluate_brax_checkpoint(latest_ckpt, eval_seed=DEFAULT_EVAL_SEED, num_episodes=50)

    print("\n================ BRAX 200M BASELINE EVALUATION RESULTS ================")
    print(f"Checkpoint: {result['checkpoint']}")
    print(f"50-Episode Mean Return: {result['mean_reward']:.4f} +/- {result['std_reward']:.4f}")
    print(f"Mean Episode Length:    {result['mean_episode_length']:.1f} steps")
    print(f"Linear Velocity Error:  {result['mean_lin_err']:.4f}")
    print(f"Yaw Rate Error:         {result['mean_yaw_err']:.4f} rad/s")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
