"""Evaluate the Brax reference with the custom policy's EMA post-processing."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.eval_brax_seeds import DEFAULT_EVAL_SEED, evaluate_brax_checkpoint


BASELINE_DIRS = {
    10: PROJECT_ROOT / "baselines" / "brax_go1_200m",
    11: PROJECT_ROOT / "baselines" / "brax_go1_200m_seed11",
    12: PROJECT_ROOT / "baselines" / "brax_go1_200m_seed12",
}


def latest_checkpoint(root: Path) -> Path:
    candidates = [
        path
        for path in root.iterdir()
        if path.is_dir() and path.name.isdigit() and (path / "ppo_network_config.json").is_file()
    ]
    if not candidates:
        raise FileNotFoundError(f"No Brax checkpoint found in {root}")
    return max(candidates, key=lambda path: int(path.name))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "baselines" / "brax_go1_200m_ema_ablation.json",
    )
    parser.add_argument("--num-episodes", type=int, default=50)
    args = parser.parse_args()

    results = []
    for seed, root in BASELINE_DIRS.items():
        checkpoint = latest_checkpoint(root)
        print(f"Evaluating EMA ablation seed {seed}: {checkpoint.name}", flush=True)
        results.append(
            evaluate_brax_checkpoint(
                checkpoint,
                eval_seed=DEFAULT_EVAL_SEED,
                num_episodes=args.num_episodes,
                training_seed=seed,
                use_ema=True,
            )
        )

    means = [result["mean_reward"] for result in results]
    mean_return = sum(means) / len(means)
    summary = {
        "protocol": "brax-ppo-command-tracking-v2",
        "action_filter": "ema_0.7_0.3",
        "env_name": "Go1JoystickFlatTerrain",
        "training_seeds": sorted(BASELINE_DIRS),
        "num_episodes_per_seed": args.num_episodes,
        "eval_seed": DEFAULT_EVAL_SEED,
        "episode_seeds": list(range(DEFAULT_EVAL_SEED, DEFAULT_EVAL_SEED + args.num_episodes)),
        "total_timesteps_per_seed": 200_000_000,
        "grand_mean_return": mean_return,
        "grand_std_return": (
            sum((value - mean_return) ** 2 for value in means) / len(means)
        ) ** 0.5,
        "mean_episode_length": sum(result["mean_episode_length"] for result in results) / len(results),
        "mean_linear_velocity_error": sum(result["mean_lin_err"] for result in results) / len(results),
        "mean_yaw_rate_error": sum(result["mean_yaw_err"] for result in results) / len(results),
        "seed_results": results,
        "comparison_note": "Same evaluator and 0.7/0.3 EMA post-processing as the custom policy; baseline weights were trained without EMA.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved EMA ablation summary: {args.output}")


if __name__ == "__main__":
    main()
