"""Plot only recorded, protocol-compatible evaluation summaries."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_summary(path: str | Path, expected_protocol: str) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required evaluation summary is missing: {path}")
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error
    if summary.get("protocol") != expected_protocol:
        raise ValueError(
            f"{path} uses protocol {summary.get('protocol')!r}, expected {expected_protocol!r}. "
            "Do not compare differently shaped rewards or evaluation procedures."
        )
    required = {"grand_mean_return", "grand_std_return", "mean_episode_length", "num_episodes_per_seed"}
    missing = sorted(required.difference(summary))
    if missing:
        raise ValueError(f"{path} is not a complete evaluation summary; missing {missing}")
    return summary


def load_training_histories(path: str | Path) -> list[dict]:
    """Load measured per-seed training points; never synthesize missing curves."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required training history is missing: {path}")
    histories = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(histories, list) or len(histories) < 3:
        raise ValueError(f"{path} must contain at least three training-seed histories")
    for history in histories:
        steps = history.get("total_steps")
        rewards = history.get("mean_step_rewards", history.get("rewards"))
        if not steps or not rewards or len(steps) != len(rewards):
            raise ValueError(f"Incomplete measured training history in {path}")
        if any(right <= left for left, right in zip(steps, steps[1:])):
            raise ValueError(f"Training steps must be strictly increasing in {path}")
    return histories


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot recorded native-reward custom/SB3 benchmark results.")
    parser.add_argument(
        "--custom-summary",
        default=PROJECT_ROOT / "checkpoints" / "ppo_eval_summary.json",
        type=Path,
    )
    parser.add_argument(
        "--sb3-summary",
        default=PROJECT_ROOT / "baselines" / "sb3_ppo_eval_summary.json",
        type=Path,
    )
    parser.add_argument(
        "--custom-training",
        default=PROJECT_ROOT / "checkpoints" / "ppo_multi_seed_results.json",
        type=Path,
    )
    parser.add_argument(
        "--sb3-training",
        default=PROJECT_ROOT / "baselines" / "sb3_training_results.json",
        type=Path,
    )
    parser.add_argument(
        "--output",
        default=PROJECT_ROOT / "write-up" / "benchmark_comparison.png",
        type=Path,
    )
    args = parser.parse_args()

    custom = load_summary(args.custom_summary, "custom-ppo-native-reward-v1")
    baseline = load_summary(args.sb3_summary, "sb3-baseline-native-reward-v1")
    custom_histories = load_training_histories(args.custom_training)
    baseline_histories = load_training_histories(args.sb3_training)
    if custom["num_episodes_per_seed"] != baseline["num_episodes_per_seed"]:
        raise ValueError("The summaries use a different number of episodes per seed.")

    labels = ["Custom PPO", "SB3 baseline"]
    summaries = [custom, baseline]
    means = [summary["grand_mean_return"] for summary in summaries]
    stds = [summary["grand_std_return"] for summary in summaries]
    lengths = [summary["mean_episode_length"] for summary in summaries]

    fig, (curve_axis, return_axis, length_axis) = plt.subplots(1, 3, figsize=(15, 4.5), dpi=160)
    colours = ["#1f77b4", "#d62728"]

    # Use the custom agent's recorded checkpoints as the common x-axis and
    # linearly interpolate only between measured baseline rollout points.
    common_steps = np.asarray(custom_histories[0]["total_steps"], dtype=np.float64)
    max_shared_step = min(
        min(history["total_steps"][-1] for history in custom_histories),
        min(history["total_steps"][-1] for history in baseline_histories),
    )
    common_steps = common_steps[common_steps <= max_shared_step]
    if common_steps.size < 2:
        raise ValueError("Training histories do not have a usable shared step range")

    for histories, label, colour in zip(
        (custom_histories, baseline_histories), labels, colours
    ):
        curves = []
        for history in histories:
            rewards = history.get("mean_step_rewards", history.get("rewards"))
            curves.append(np.interp(common_steps, history["total_steps"], rewards))
        curves = np.asarray(curves, dtype=np.float64)
        mean = curves.mean(axis=0)
        std = curves.std(axis=0)
        curve_axis.plot(common_steps, mean, label=label, color=colour)
        curve_axis.fill_between(common_steps, mean - std, mean + std, color=colour, alpha=0.2)
    curve_axis.set_title("Measured training reward")
    curve_axis.set_xlabel("environment steps")
    curve_axis.set_ylabel("native mean step reward")
    curve_axis.legend()
    curve_axis.grid(linestyle=":", alpha=0.5)
    return_axis.bar(labels, means, yerr=stds, capsize=5, color=colours)
    return_axis.set_title("Native-reward return across training seeds")
    return_axis.set_ylabel("mean episode return (error: across-seed std)")
    return_axis.grid(axis="y", linestyle=":", alpha=0.5)

    length_axis.bar(labels, lengths, color=colours)
    length_axis.set_title("Episode survival")
    length_axis.set_ylabel("mean episode length (steps)")
    length_axis.grid(axis="y", linestyle=":", alpha=0.5)

    fig.suptitle(f"Fixed deterministic protocol: {custom['num_episodes_per_seed']} episodes per seed")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    print(f"Saved recorded-data benchmark plot to {args.output}")


if __name__ == "__main__":
    main()
