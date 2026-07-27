"""Run one deterministic diagnostic rollout under the submission contract."""

import argparse

import torch

from eval.evaluate import DEFAULT_EVAL_SEED, load_actor_critic_checkpoint, load_config
from ppo.env import MJXVectorPyTorchWrapper


def render_policy_rollout(
    checkpoint_path: str,
    seed: int = DEFAULT_EVAL_SEED,
    config_path: str = "configs/default.yaml",
) -> dict:
    config = load_config(config_path)
    env = MJXVectorPyTorchWrapper(
        config["env_name"], num_envs=1, seed=seed,
        history_len=config.get("history_len", 5),
        episode_length=config.get("episode_length", 1000),
    )
    agent, _ = load_actor_critic_checkpoint(
        checkpoint_path, env_name=config["env_name"], obs_dim=env.observation_dim,
        act_dim=env.action_dim, hidden_dim=config.get("hidden_dim", 256),
        critic_obs_dim=env.privileged_observation_dim,
        hidden_sizes=config.get("hidden_sizes"),
    )
    obs, _ = env.reset(seed=seed)
    total_return = 0.0
    steps = 0
    while True:
        with torch.no_grad():
            action, _ = agent.get_action(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_return += float(reward.item())
        steps += 1
        if bool(terminated.item()) or bool(truncated.item()):
            break
    result = {"seed": seed, "native_return": total_return, "episode_length": steps}
    print(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--seed", type=int, default=DEFAULT_EVAL_SEED)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    render_policy_rollout(args.checkpoint, args.seed, args.config)


if __name__ == "__main__":
    main()
