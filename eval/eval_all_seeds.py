# ruff: noqa: E402
"""Evaluate every configured custom-PPO seed under one fixed protocol."""

import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.evaluate import DEFAULT_EVAL_SEED, EVAL_EPISODES, evaluate_policy, load_config


def main() -> None:
    config = load_config()
    training_seeds = config.get("seeds", [])
    if not training_seeds:
        raise ValueError("configs/default.yaml has no training seeds to evaluate.")

    checkpoints = []
    for seed in training_seeds:
        path1 = PROJECT_ROOT / "checkpoints" / "cluster_100m_v2" / f"ppo_seed{seed}.pt"
        path2 = PROJECT_ROOT / "checkpoints" / f"ppo_seed{seed}.pt"
        if path1.is_file():
            checkpoints.append(path1)
        elif path2.is_file():
            checkpoints.append(path2)
        else:
            checkpoints.append(path1)

    missing = [str(path) for path in checkpoints if not path.is_file()]
    if missing:
        raise FileNotFoundError("Refusing to publish a partial multi-seed result; missing:\n" + "\n".join(missing))

    seed_results = []
    for training_seed, checkpoint in zip(training_seeds, checkpoints):
        # Every checkpoint sees the same fixed episodes for a paired comparison.
        result = evaluate_policy(checkpoint, eval_seed=DEFAULT_EVAL_SEED)
        result["training_seed"] = training_seed
        seed_results.append(result)

    means = np.asarray([result["mean_reward"] for result in seed_results], dtype=np.float64)
    summary = {
        "protocol": "custom-ppo-native-reward-v1",
        "num_episodes_per_seed": EVAL_EPISODES,
        "training_seeds": training_seeds,
        "grand_mean_return": float(means.mean()),
        "grand_std_return": float(means.std()),
        "mean_episode_length": float(np.mean([result["mean_episode_length"] for result in seed_results])),
        "mean_latency_ms": float(np.mean([result["mean_latency_ms"] for result in seed_results])),
        "seed_results": seed_results,
    }
    output_path = PROJECT_ROOT / "checkpoints" / "ppo_eval_summary.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved complete {len(seed_results)}-seed summary to {output_path}")


if __name__ == "__main__":
    main()
