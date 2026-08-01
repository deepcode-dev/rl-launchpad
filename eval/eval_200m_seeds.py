"""Evaluate all three 200M cluster training seeds (10, 20, 30) for Rule R2 compliance."""

import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.evaluate import DEFAULT_EVAL_SEED, EVAL_EPISODES, evaluate_policy


def main() -> None:
    training_seeds = [10, 20, 30]
    checkpoints = [
        PROJECT_ROOT / "checkpoints" / "cluster_100m_v2" / f"ppo_seed{seed}.pt"
        for seed in training_seeds
    ]
    missing = [str(path) for path in checkpoints if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing 200M cluster checkpoints:\n" + "\n".join(missing)
        )

    seed_results = []
    print("Evaluating 200M multi-seed checkpoints over 50 fixed, disjoint episodes...")
    for training_seed, checkpoint in zip(training_seeds, checkpoints):
        print(f"  Evaluating Seed {training_seed} ({checkpoint.name})...")
        result = evaluate_policy(checkpoint, eval_seed=DEFAULT_EVAL_SEED, num_episodes=50)
        result["training_seed"] = training_seed
        seed_results.append(result)

    returns = [result["mean_reward"] for result in seed_results]
    print("\n================ MULTI-SEED EVALUATION RESULTS ================")
    for res in seed_results:
        print(
            f"Seed {res['training_seed']:2d}: Mean Return = {res['mean_reward']:.4f} | "
            f"Episode Length = {res['mean_episode_length']:.1f} steps | "
            f"Latency = {res['mean_latency_ms']:.2f} ms"
        )
    print("---------------------------------------------------------------")
    print(
        f"Grand Multi-Seed Mean Return: {np.mean(returns):.4f} +/- {np.std(returns):.4f}"
    )
    print("===============================================================\n")

    summary = {
        "protocol": "custom-ppo-200m-cluster-v1",
        "num_episodes_per_seed": 50,
        "training_seeds": training_seeds,
        "grand_mean_return": float(np.mean(returns)),
        "grand_std_return": float(np.std(returns)),
        "seed_results": seed_results,
    }
    output_path = PROJECT_ROOT / "checkpoints" / "cluster_100m_v2" / "eval_200m_summary.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved complete 200M multi-seed summary to {output_path}")


if __name__ == "__main__":
    main()
