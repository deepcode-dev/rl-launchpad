"""Train the stock MuJoCo Playground Go1 task with Brax PPO."""

from __future__ import annotations

import argparse
import functools
import json
from pathlib import Path
import time

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo_train
import jax
import mujoco_playground as mp
from mujoco_playground import wrapper
import yaml


def _install_jax_011_brax_compatibility() -> None:
    """Restore the replicated-put alias removed from JAX 0.11.

    Brax 0.14.2 still calls the former public name. JAX 0.11 retains the
    implementation internally, so this avoids downgrading JAX/CUDA packages.
    """
    try:
        jax.device_put_replicated
    except AttributeError:
        from jax._src.api import device_put_replicated

        jax.device_put_replicated = device_put_replicated


def _scalar_metrics(metrics: dict) -> dict[str, float]:
    result = {}
    for key, value in metrics.items():
        try:
            if getattr(value, "size", 1) == 1:
                result[key] = float(value)
        except (TypeError, ValueError):
            continue
    return result


def main(config_path: str) -> None:
    _install_jax_011_brax_compatibility()
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    env_name = config.get("env_name", "Go1JoystickFlatTerrain")
    env_config = mp.locomotion.get_default_config(env_name)
    env_config.impl = "jax"
    environment = mp.locomotion.load(env_name, config=env_config)

    hidden_sizes = tuple(config.get("hidden_sizes", (512, 256, 128)))
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=hidden_sizes,
        value_hidden_layer_sizes=hidden_sizes,
        policy_obs_key="state",
        value_obs_key="privileged_state",
        distribution_type="tanh_normal",
    )

    started = time.time()

    def progress(step: int, metrics: dict) -> None:
        payload = {
            "step": int(step),
            "elapsed_seconds": round(time.time() - started, 1),
            **_scalar_metrics(metrics),
        }
        print(json.dumps(payload, sort_keys=True), flush=True)

    print(f"JAX devices: {jax.devices()}", flush=True)
    print(f"Environment: {env_name}", flush=True)
    print(f"Config: {config_path}", flush=True)

    ppo_train.train(
        environment=environment,
        num_timesteps=int(config["num_timesteps"]),
        num_envs=int(config.get("num_envs", 8192)),
        episode_length=int(config.get("episode_length", 1000)),
        action_repeat=1,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        learning_rate=float(config.get("learning_rate", 3e-4)),
        entropy_cost=float(config.get("entropy_cost", 0.01)),
        discounting=float(config.get("discounting", 0.97)),
        unroll_length=int(config.get("unroll_length", 20)),
        batch_size=int(config.get("batch_size", 256)),
        num_minibatches=int(config.get("num_minibatches", 32)),
        num_updates_per_batch=int(config.get("num_updates_per_batch", 4)),
        normalize_observations=True,
        reward_scaling=float(config.get("reward_scaling", 1.0)),
        clipping_epsilon=float(config.get("clipping_epsilon", 0.2)),
        gae_lambda=float(config.get("gae_lambda", 0.95)),
        max_grad_norm=float(config.get("max_grad_norm", 1.0)),
        network_factory=network_factory,
        seed=int(config.get("seed", 10)),
        num_evals=int(config.get("num_evals", 5)),
        num_eval_envs=int(config.get("num_eval_envs", 128)),
        deterministic_eval=True,
        progress_fn=progress,
        save_checkpoint_path=config.get(
            "checkpoint_dir", "checkpoints/brax_go1_20m"
        ),
    )

    print(
        f"Training complete in {time.time() - started:.1f}s; "
        f"checkpoints: {config.get('checkpoint_dir')}",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/brax_go1_20m.yaml")
    args = parser.parse_args()
    main(args.config)
