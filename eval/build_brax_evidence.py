# ruff: noqa: E402
"""Evaluate the committed Brax checkpoints and build an auditable curve summary.

The final RL-track comparison needs measured checkpoint points, not a hand-drawn
learning curve. This script evaluates the four committed checkpoints for each
of the three baseline seeds using the same 50 fixed native-MuJoCo episodes as
the final benchmark.
"""

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


def checkpoint_dirs(root: Path) -> list[Path]:
    result = [
        path
        for path in root.iterdir()
        if path.is_dir() and path.name.isdigit() and (path / "ppo_network_config.json").is_file()
    ]
    return sorted(result, key=lambda path: int(path.name))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "baselines" / "brax_go1_200m_eval_summary.json",
    )
    args = parser.parse_args()

    all_results: list[dict] = []
    final_results: list[dict] = []
    for training_seed, root in BASELINE_DIRS.items():
        paths = checkpoint_dirs(root)
        if not paths:
            raise FileNotFoundError(f"No committed Brax checkpoints found in {root}")
        for checkpoint in paths:
            print(f"Evaluating seed {training_seed}, checkpoint {checkpoint.name}", flush=True)
            result = evaluate_brax_checkpoint(
                checkpoint,
                eval_seed=DEFAULT_EVAL_SEED,
                num_episodes=args.num_episodes,
                training_seed=training_seed,
            )
            all_results.append(result)
            if checkpoint == paths[-1]:
                final_results.append(result)

    final_means = [result["mean_reward"] for result in final_results]
    curve = []
    for step in sorted({result["checkpoint_step"] for result in all_results}):
        points = [result for result in all_results if result["checkpoint_step"] == step]
        means = [point["mean_reward"] for point in points]
        lin_means = [point["mean_lin_err"] for point in points]
        lin_mean = sum(lin_means) / len(lin_means)
        curve.append(
            {
                "checkpoint_step": step,
                "training_seeds": [point["training_seed"] for point in points],
                "mean_return": sum(means) / len(means),
                "std_return": (
                    sum((value - (sum(means) / len(means))) ** 2 for value in means) / len(means)
                ) ** 0.5,
                "mean_episode_length": sum(point["mean_episode_length"] for point in points) / len(points),
                "mean_linear_velocity_error": lin_mean,
                "std_linear_velocity_error": (
                    sum((value - lin_mean) ** 2 for value in lin_means) / len(lin_means)
                ) ** 0.5,
                "mean_yaw_rate_error": sum(point["mean_yaw_err"] for point in points) / len(points),
                "seed_results": points,
            }
        )

    final_mean = sum(final_means) / len(final_means)
    summary = {
        "protocol": "brax-ppo-command-tracking-v2",
        "env_name": "Go1JoystickFlatTerrain",
        "training_seeds": sorted(BASELINE_DIRS),
        "num_episodes_per_seed": args.num_episodes,
        "eval_seed": DEFAULT_EVAL_SEED,
        "episode_seeds": list(range(DEFAULT_EVAL_SEED, DEFAULT_EVAL_SEED + args.num_episodes)),
        "total_timesteps_per_seed": 200_000_000,
        "grand_mean_return": final_mean,
        "grand_std_return": (
            sum((value - final_mean) ** 2 for value in final_means) / len(final_means)
        ) ** 0.5,
        "mean_episode_length": sum(result["mean_episode_length"] for result in final_results) / len(final_results),
        "mean_linear_velocity_error": sum(result["mean_lin_err"] for result in final_results) / len(final_results),
        "mean_yaw_rate_error": sum(result["mean_yaw_err"] for result in final_results) / len(final_results),
        "seed_results": final_results,
        "training_curve": curve,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved measured Brax evidence summary: {args.output}")


if __name__ == "__main__":
    main()
