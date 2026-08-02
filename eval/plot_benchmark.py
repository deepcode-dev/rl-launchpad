"""Plot measured custom PPO and Brax RL-track evidence.

The curve panel deliberately uses only values recorded during training or
checkpoint evaluation. It never invents a baseline trajectory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_json(path: str | Path) -> dict | list:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required evidence is missing: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error


def load_summary(path: str | Path, expected_protocol: str) -> dict:
    summary = load_json(path)
    if not isinstance(summary, dict):
        raise TypeError(f"Summary must be an object: {path}")
    required = {
        "grand_mean_return",
        "grand_std_return",
        "mean_episode_length",
        "num_episodes_per_seed",
        "mean_linear_velocity_error",
        "mean_yaw_rate_error",
    }
    missing = sorted(required.difference(summary))
    if missing:
        raise ValueError(f"{path} is incomplete; missing {missing}")
    if summary.get("protocol") != expected_protocol:
        raise ValueError(
            f"{path} uses protocol {summary.get('protocol')!r}; expected {expected_protocol!r}."
        )
    return summary


def load_training_history(path: str | Path) -> dict:
    histories = load_json(path)
    if not isinstance(histories, list) or not histories:
        raise ValueError(f"{path} must contain at least one measured history")
    history = histories[0]
    steps = history.get("total_steps")
    lin_errs = history.get("linear_velocity_errors")
    if not steps or not lin_errs or len(steps) != len(lin_errs):
        raise ValueError(f"{path} does not contain a complete measured LinErr history")
    if any(right <= left for left, right in zip(steps, steps[1:])):
        raise ValueError(f"Training steps must be strictly increasing in {path}")
    return history


def load_training_histories(path: str | Path) -> list[dict]:
    """Compatibility loader for callers that need every recorded history."""
    histories = load_json(path)
    if not isinstance(histories, list) or not histories:
        raise ValueError(f"{path} must contain at least one measured history")
    for index, history in enumerate(histories):
        steps = history.get("total_steps")
        lin_errs = history.get("linear_velocity_errors")
        if not steps or not lin_errs or len(steps) != len(lin_errs):
            raise ValueError(f"History {index} in {path} is incomplete")
        if any(right <= left for left, right in zip(steps, steps[1:])):
            raise ValueError(f"Training steps must be strictly increasing in {path}")
    return histories


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--custom-summary",
        default=PROJECT_ROOT / "NEW_checkpoints" / "ppo_v2" / "ppo_v2_eval_summary.json",
        type=Path,
    )
    parser.add_argument(
        "--baseline-summary",
        default=PROJECT_ROOT / "baselines" / "brax_go1_200m_eval_summary.json",
        type=Path,
    )
    parser.add_argument(
        "--custom-training",
        default=PROJECT_ROOT / "NEW_checkpoints" / "ppo_v2" / "ppo_multi_seed_results.json",
        type=Path,
    )
    parser.add_argument(
        "--output",
        default=PROJECT_ROOT / "write-up" / "benchmark_comparison.png",
        type=Path,
    )
    args = parser.parse_args()

    custom = load_summary(args.custom_summary, "custom-ppo-v2-champion")
    baseline = load_summary(args.baseline_summary, "brax-ppo-command-tracking-v2")
    histories = load_training_histories(args.custom_training)
    curve = baseline.get("training_curve")
    if not isinstance(curve, list) or not curve:
        raise ValueError("Brax summary has no measured checkpoint curve")
    for point in curve:
        for key in (
            "checkpoint_step",
            "mean_linear_velocity_error",
            "std_linear_velocity_error",
        ):
            if key not in point:
                raise ValueError(f"Brax curve point is missing {key!r}")

    labels = [
        f"Custom PPO ({len(custom['training_seeds'])} seeds)",
        f"Brax PPO ({len(baseline['training_seeds'])} seeds)",
    ]
    means = [custom["grand_mean_return"], baseline["grand_mean_return"]]
    stds = [custom["grand_std_return"], baseline["grand_std_return"]]
    lin_errs = [custom["mean_linear_velocity_error"], baseline["mean_linear_velocity_error"]]
    yaw_errs = [custom["mean_yaw_rate_error"], baseline["mean_yaw_rate_error"]]

    fig, (curve_axis, return_axis, err_axis, yaw_axis) = plt.subplots(
        1, 4, figsize=(20, 4.8), dpi=180
    )
    colours = ["#1f77b4", "#d62728"]

    # This is a true shared metric: both implementations report mean linear
    # velocity tracking error. The custom line is a mean +/- SD over the
    # three measured stdout histories; the Brax points are mean +/- SD over
    # three seeds.
    custom_steps_raw = [np.asarray(history["total_steps"], dtype=np.float64) for history in histories]
    common_steps = custom_steps_raw[0]
    if any(not np.array_equal(steps, common_steps) for steps in custom_steps_raw[1:]):
        raise ValueError("Custom histories must share the same checkpoint steps for a mean/std curve")
    custom_steps = common_steps / 1e6
    custom_matrix = np.asarray(
        [history["linear_velocity_errors"] for history in histories], dtype=np.float64
    )
    custom_lin_err = custom_matrix.mean(axis=0)
    custom_lin_std = custom_matrix.std(axis=0)
    baseline_steps = np.asarray([point["checkpoint_step"] for point in curve], dtype=np.float64) / 1e6
    baseline_lin_err = np.asarray(
        [point["mean_linear_velocity_error"] for point in curve], dtype=np.float64
    )
    baseline_lin_std = np.asarray(
        [point["std_linear_velocity_error"] for point in curve], dtype=np.float64
    )
    curve_axis.plot(
        custom_steps,
        custom_lin_err,
        color=colours[0],
        linewidth=1.8,
        label=(
            f"Custom rollout means ({len(histories)} seeds)"
            if len(histories) > 1
            else f"Custom rollout means (seed {histories[0]['seed']})"
        ),
    )
    if len(histories) > 1:
        curve_axis.fill_between(
            custom_steps,
            custom_lin_err - custom_lin_std,
            custom_lin_err + custom_lin_std,
            color=colours[0],
            alpha=0.16,
            label="Custom +/- seed SD",
        )
    curve_axis.plot(
        baseline_steps,
        baseline_lin_err,
        color=colours[1],
        marker="o",
        linewidth=2,
        label="Brax checkpoint means (3 seeds)",
    )
    curve_axis.fill_between(
        baseline_steps,
        baseline_lin_err - baseline_lin_std,
        baseline_lin_err + baseline_lin_std,
        color=colours[1],
        alpha=0.16,
        label="Brax ± seed SD",
    )
    curve_axis.set_title("Measured LinErr vs steps", fontsize=11, fontweight="bold")
    curve_axis.set_xlabel("Environment steps (millions)", fontsize=10)
    curve_axis.set_ylabel("Linear velocity error (m/s)", fontsize=10)
    curve_axis.set_xlim(0, max(200, float(max(baseline_steps)) + 2))
    curve_axis.set_ylim(bottom=0)
    curve_axis.legend(fontsize=8, loc="upper right")
    curve_axis.grid(linestyle=":", alpha=0.5)
    curve_axis.text(
        0.02,
        0.03,
        (
            "Custom curve is one logged trajectory; final bars are multi-seed."
            if len(histories) == 1
            else "Curves show mean +/- seed SD; final bars are multi-seed."
        ),
        transform=curve_axis.transAxes,
        fontsize=7.5,
        color="#475569",
    )

    bars = return_axis.bar(
        labels, means, yerr=stds, capsize=6, color=colours, width=0.55, edgecolor="black", alpha=0.85
    )
    return_axis.set_title("50-episode return", fontsize=11, fontweight="bold")
    return_axis.set_ylabel("Shared command-tracking return", fontsize=10)
    return_axis.set_ylim(15.0, 21.5)
    return_axis.grid(axis="y", linestyle=":", alpha=0.5)
    for bar, mean_val, std_val in zip(bars, means, stds):
        return_axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + std_val + 0.15,
            f"{mean_val:.3f} +/- {std_val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    err_bars = err_axis.bar(
        labels, lin_errs, color=colours, width=0.55, edgecolor="black", alpha=0.85
    )
    err_axis.set_title("Linear velocity error", fontsize=11, fontweight="bold")
    err_axis.set_ylabel("LinErr (m/s) [lower is better]", fontsize=10)
    err_axis.set_ylim(0.0, 0.14)
    err_axis.grid(axis="y", linestyle=":", alpha=0.5)
    for bar, err_val in zip(err_bars, lin_errs):
        err_axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.003,
            f"{err_val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )

    yaw_bars = yaw_axis.bar(
        labels, yaw_errs, color=colours, width=0.55, edgecolor="black", alpha=0.85
    )
    yaw_axis.set_title("Yaw-rate error", fontsize=11, fontweight="bold")
    yaw_axis.set_ylabel("YawErr (rad/s) [lower is better]", fontsize=10)
    yaw_axis.set_ylim(0.0, 0.10)
    yaw_axis.grid(axis="y", linestyle=":", alpha=0.5)
    for bar, yaw_val in zip(yaw_bars, yaw_errs):
        yaw_axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.002,
            f"{yaw_val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )

    fig.suptitle(
        "Go1 Locomotion: From-Scratch PyTorch PPO vs Brax PPO",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    print(f"Saved measured-data benchmark plot to {args.output}")


if __name__ == "__main__":
    main()
