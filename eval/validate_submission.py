"""Audit the canonical Launchpad RL-track submission artifacts.

This validator checks the current five-seed custom result against the current
three-seed Brax result and verifies the narrated demo when requested.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import wave

import imageio_ffmpeg

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CUSTOM_SUMMARY_PROTOCOL = "custom-ppo-v2-champion"
CUSTOM_EVAL_PROTOCOL = "custom-ppo-command-tracking-v2"
BASELINE_PROTOCOL = "brax-ppo-command-tracking-v2"
CUSTOM_SEEDS = [9033, 9006, 9018, 9016, 8009]
BASELINE_SEEDS = [10, 11, 12]
EVAL_SEED = 20_000
EVAL_EPISODES = 50
CUSTOM_STEPS = 199_884_800
BASELINE_BUDGET = 200_000_000


def load_json(path: Path) -> dict | list:
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact is missing: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error


def require_finite(values: list, label: str) -> None:
    if not values or not all(math.isfinite(float(value)) for value in values):
        raise ValueError(f"{label} contains missing or non-finite values")


def expected_episode_seeds() -> list[int]:
    return list(range(EVAL_SEED, EVAL_SEED + EVAL_EPISODES))


def validate_episode_result(
    result: dict,
    *,
    protocol: str,
    training_seed: int,
    checkpoint_steps: set[int],
    expected_action_filter: str | None = None,
) -> None:
    if result.get("protocol") != protocol:
        raise ValueError(f"Seed {training_seed} uses the wrong protocol: {result.get('protocol')!r}")
    if result.get("training_seed") not in (None, training_seed) and result.get("seed") != training_seed:
        raise ValueError(f"Mislabeled result for training seed {training_seed}")
    if result.get("eval_seed") != EVAL_SEED:
        raise ValueError(f"Seed {training_seed} uses the wrong evaluation seed")
    if result.get("episode_seeds") != expected_episode_seeds():
        raise ValueError(f"Seed {training_seed} does not use the fixed disjoint episode block")
    if result.get("num_episodes") != EVAL_EPISODES:
        raise ValueError(f"Seed {training_seed} does not contain exactly 50 episodes")
    if result.get("deterministic_actions") is not True:
        raise ValueError(f"Seed {training_seed} was not evaluated deterministically")
    if expected_action_filter is not None and result.get("action_filter") != expected_action_filter:
        raise ValueError(f"Seed {training_seed} has the wrong action filter")
    for key in ("episode_rewards", "episode_lengths"):
        values = result.get(key, [])
        if len(values) != EVAL_EPISODES:
            raise ValueError(f"Seed {training_seed} has an incomplete {key} array")
        require_finite(values, f"Seed {training_seed} {key}")
    if len(result.get("episode_linear_velocity_errors", result.get("episode_lin_errs", []))) != EVAL_EPISODES:
        raise ValueError(f"Seed {training_seed} has an incomplete LinErr array")
    if len(result.get("episode_yaw_rate_errors", result.get("episode_yaw_errs", []))) != EVAL_EPISODES:
        raise ValueError(f"Seed {training_seed} has an incomplete YawErr array")
    if checkpoint_steps:
        checkpoint_step = int(result.get("checkpoint_step", 0))
        if checkpoint_step not in checkpoint_steps:
            raise ValueError(f"Seed {training_seed} has an unexpected checkpoint step")


def validate_custom() -> dict:
    summary_path = PROJECT_ROOT / "NEW_checkpoints" / "ppo_v2" / "ppo_v2_eval_summary.json"
    summary = load_json(summary_path)
    if not isinstance(summary, dict) or summary.get("protocol") != CUSTOM_SUMMARY_PROTOCOL:
        raise ValueError("Canonical custom summary is missing or uses the wrong protocol")
    if summary.get("training_seeds") != CUSTOM_SEEDS:
        raise ValueError(f"Custom seeds do not match the canonical set: {summary.get('training_seeds')}")
    if summary.get("num_episodes_per_seed") != EVAL_EPISODES:
        raise ValueError("Custom summary does not contain exactly 50 episodes per seed")
    if len(summary.get("seed_results", [])) != len(CUSTOM_SEEDS):
        raise ValueError("Custom summary has an incomplete independent-seed set")

    for item in summary["seed_results"]:
        seed = int(item["seed"])
        result_path = PROJECT_ROOT / Path(str(item["json_path"]).replace("\\", "/"))
        result = load_json(result_path)
        if not isinstance(result, dict):
            raise ValueError(f"Custom result is not an object: {result_path}")
        validate_episode_result(
            result,
            protocol=CUSTOM_EVAL_PROTOCOL,
            training_seed=seed,
            checkpoint_steps=set(),
        )
        metadata = result.get("checkpoint_metadata", {})
        if metadata.get("total_env_steps") != CUSTOM_STEPS:
            raise ValueError(f"Custom seed {seed} has the wrong step budget")
        if float(metadata.get("wall_time_seconds", 0.0)) <= 0:
            raise ValueError(f"Custom seed {seed} has no training wall-time metadata")
        if metadata.get("reward_source") != "mujoco_playground_state_reward":
            raise ValueError(f"Custom seed {seed} has the wrong training reward source")
        if metadata.get("obs_dim") != 48 or metadata.get("critic_obs_dim") != 123 or metadata.get("act_dim") != 12:
            raise ValueError(f"Custom seed {seed} has the wrong observation/action contract")

    history_path = PROJECT_ROOT / "NEW_checkpoints" / "ppo_v2" / "ppo_multi_seed_results.json"
    histories = load_json(history_path)
    if not isinstance(histories, list) or not histories:
        raise ValueError("The custom training history is missing")
    for history in histories:
        steps = history.get("total_steps", [])
        if not steps or steps[-1] != CUSTOM_STEPS or any(right <= left for left, right in zip(steps, steps[1:])):
            raise ValueError("Custom training history is incomplete or non-monotonic")
    return summary


def validate_baseline() -> dict:
    path = PROJECT_ROOT / "baselines" / "brax_go1_200m_eval_summary.json"
    summary = load_json(path)
    if not isinstance(summary, dict) or summary.get("protocol") != BASELINE_PROTOCOL:
        raise ValueError("Canonical Brax summary is missing or uses the wrong protocol")
    if summary.get("training_seeds") != BASELINE_SEEDS:
        raise ValueError(f"Baseline seeds do not match the canonical set: {summary.get('training_seeds')}")
    if summary.get("num_episodes_per_seed") != EVAL_EPISODES:
        raise ValueError("Baseline summary does not contain exactly 50 episodes per seed")
    if summary.get("total_timesteps_per_seed") != BASELINE_BUDGET:
        raise ValueError("Baseline training budget is not 200M steps")
    if float(summary.get("wall_time_seconds_documented", 0.0)) <= 0:
        raise ValueError("Baseline wall-time accounting is missing")
    final_results = summary.get("seed_results", [])
    if len(final_results) != len(BASELINE_SEEDS):
        raise ValueError("Baseline summary has an incomplete independent-seed set")
    for result in final_results:
        seed = int(result["training_seed"])
        validate_episode_result(
            result,
            protocol=BASELINE_PROTOCOL,
            training_seed=seed,
            checkpoint_steps={200_540_160},
        )
    curve = summary.get("training_curve", [])
    if len(curve) < 4:
        raise ValueError("Baseline summary does not contain the measured checkpoint curve")
    return summary


def validate_ema_ablation() -> dict:
    path = PROJECT_ROOT / "baselines" / "brax_go1_200m_ema_ablation.json"
    summary = load_json(path)
    if not isinstance(summary, dict) or summary.get("protocol") != BASELINE_PROTOCOL:
        raise ValueError("EMA ablation summary is missing or uses the wrong protocol")
    if summary.get("action_filter") != "ema_0.7_0.3":
        raise ValueError("EMA ablation summary does not identify the custom filter")
    if summary.get("training_seeds") != BASELINE_SEEDS:
        raise ValueError("EMA ablation seed set is incomplete")
    if summary.get("num_episodes_per_seed") != EVAL_EPISODES:
        raise ValueError("EMA ablation does not contain exactly 50 episodes per seed")
    for result in summary.get("seed_results", []):
        validate_episode_result(
            result,
            protocol=BASELINE_PROTOCOL,
            training_seed=int(result["training_seed"]),
            checkpoint_steps={200_540_160},
            expected_action_filter="ema_0.7_0.3",
        )
    if len(summary.get("seed_results", [])) != len(BASELINE_SEEDS):
        raise ValueError("EMA ablation summary has an incomplete seed set")
    return summary


def validate_writeup_and_plot() -> int:
    writeup_path = PROJECT_ROOT / "write-up" / "submission.md"
    text = writeup_path.read_text(encoding="utf-8")
    if "@@" in text:
        raise ValueError("Submission still contains unresolved template tokens")
    word_count = len(re.findall(r"\b[\w'-]+\b", text))
    if word_count > 1_000:
        raise ValueError(f"Submission exceeds 1,000 words: {word_count}")
    for required in ("Reward and environment contract", "Honesty and trajectory", "GAE", "50 episodes", "200M"):
        if required.lower() not in text.lower():
            raise ValueError(f"Submission is missing required topic: {required}")
    plotter = (PROJECT_ROOT / "eval" / "plot_benchmark.py").read_text(encoding="utf-8")
    if "np.exp" in plotter or "hand-authored" in plotter:
        raise ValueError("Plotter still contains a synthetic curve path")
    plot = PROJECT_ROOT / "write-up" / "benchmark_comparison.png"
    if not plot.is_file() or plot.stat().st_size == 0:
        raise FileNotFoundError("Measured benchmark plot is missing")
    return word_count


def validate_demo(require_demo: bool) -> list[str]:
    if not require_demo:
        return []
    warnings: list[str] = []
    demo = PROJECT_ROOT / "write-up" / "demo.mp4"
    footage = PROJECT_ROOT / "write-up" / "policy-footage.mp4"
    if not demo.is_file() or not demo.stat().st_size:
        raise FileNotFoundError("Final demo video is missing")
    if not footage.is_file() or not footage.stat().st_size:
        raise FileNotFoundError("Canonical checkpoint footage is missing")
    reader = imageio_ffmpeg.read_frames(str(demo))
    metadata = next(reader)
    duration = float(metadata.get("duration", 0.0))
    if not 1.0 <= duration <= 120.0:
        raise ValueError(f"Demo must be no longer than two minutes; got {duration:.1f}s")
    audio_path = PROJECT_ROOT / "write-up" / "demo-narration.wav"
    silence_path = PROJECT_ROOT / "write-up" / "demo-silence.wav"
    valid_narration = False
    if audio_path.is_file() and audio_path.stat().st_size > 44:
        try:
            with wave.open(str(audio_path), "rb") as audio:
                valid_narration = audio.getnframes() >= int(audio.getframerate() * 60)
        except wave.Error:
            valid_narration = False
    if valid_narration:
        pass
    elif silence_path.is_file() and silence_path.stat().st_size > 44:
        with wave.open(str(silence_path), "rb") as audio:
            raw = audio.readframes(audio.getnframes())
        if not any(raw):
            warnings.append("Current demo uses silent audio with on-screen captions; add a human voice before upload")
    else:
        raise FileNotFoundError("Neither demo narration nor captioned-demo audio exists")
    script = (PROJECT_ROOT / "write-up" / "demo-script.md").read_text(encoding="utf-8")
    if "ppo_seed9033.pt" not in script:
        raise ValueError("Demo script does not identify the reported seed 9033 checkpoint")
    return warnings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-demo", action="store_true")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat disclosed presentation gaps as failures instead of warnings",
    )
    args = parser.parse_args()

    custom = validate_custom()
    baseline = validate_baseline()
    ema_ablation = validate_ema_ablation()
    word_count = validate_writeup_and_plot()
    warnings = validate_demo(args.require_demo)
    custom_histories = load_json(
        PROJECT_ROOT / "NEW_checkpoints" / "ppo_v2" / "ppo_multi_seed_results.json"
    )
    if len(custom_histories) < 3:
        warnings.append(
            f"Only {len(custom_histories)} complete custom training history is present; "
            "the plotted custom curve is not a multi-seed mean/std curve"
        )
    if args.strict and warnings:
        raise ValueError("Strict submission audit failed: " + " | ".join(warnings))
    print(
        f"Core submission audit passed: {len(custom['training_seeds'])} custom seeds, "
        f"{len(baseline['training_seeds'])} baseline seeds, {EVAL_EPISODES} episodes/seed, "
        f"{word_count} write-up words, and {len(ema_ablation['training_seeds'])}-seed EMA ablation."
    )
    for warning in warnings:
        print(f"WARNING: {warning}")


if __name__ == "__main__":
    main()
