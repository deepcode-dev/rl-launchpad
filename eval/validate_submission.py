"""Fail closed unless every local submission artifact obeys one protocol."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import wave

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CUSTOM_PROTOCOL = "custom-ppo-native-reward-v1"
BASELINE_PROTOCOL = "sb3-baseline-native-reward-v1"
EVAL_SEED = 20_000
EVAL_EPISODES = 50


def load_json(path: Path) -> dict | list:
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_summary(
    summary: dict,
    *,
    protocol: str,
    seeds: list[int],
    steps_per_seed: int,
) -> None:
    if summary.get("protocol") != protocol:
        raise ValueError(f"Expected protocol {protocol!r}, got {summary.get('protocol')!r}")
    if summary.get("training_seeds") != seeds:
        raise ValueError(f"Summary seeds do not match config: {summary.get('training_seeds')} != {seeds}")
    if summary.get("num_episodes_per_seed") != EVAL_EPISODES:
        raise ValueError("Summary does not contain exactly 50 episodes per training seed")
    results = summary.get("seed_results", [])
    if len(results) != len(seeds):
        raise ValueError("Summary has an incomplete set of independently trained seeds")

    expected_eval_seeds = list(range(EVAL_SEED, EVAL_SEED + EVAL_EPISODES))
    for training_seed, result in zip(seeds, results):
        if result.get("training_seed") != training_seed:
            raise ValueError(f"Mislabeled result for training seed {training_seed}")
        if result.get("protocol") != protocol:
            raise ValueError(f"Per-seed protocol mismatch for seed {training_seed}")
        if result.get("episode_seeds") != expected_eval_seeds:
            raise ValueError(f"Evaluation seed block mismatch for training seed {training_seed}")
        if result.get("num_episodes") != EVAL_EPISODES:
            raise ValueError(f"Wrong episode count for training seed {training_seed}")
        if len(result.get("episode_rewards", [])) != EVAL_EPISODES:
            raise ValueError(f"Missing episode returns for training seed {training_seed}")
        if len(result.get("episode_lengths", [])) != EVAL_EPISODES:
            raise ValueError(f"Missing episode lengths for training seed {training_seed}")
        if not all(math.isfinite(float(value)) for value in result["episode_rewards"]):
            raise ValueError(f"Non-finite return for training seed {training_seed}")
        metadata = result.get("checkpoint_metadata", {})
        if metadata.get("total_env_steps") != steps_per_seed:
            raise ValueError(f"Step-budget mismatch for training seed {training_seed}")
        if metadata.get("reward_source") != "mujoco_playground_state_reward":
            raise ValueError(f"Reward contract mismatch for training seed {training_seed}")


def validate_histories(path: Path, seeds: list[int], steps_per_seed: int) -> None:
    histories = load_json(path)
    if not isinstance(histories, list) or [item.get("seed") for item in histories] != seeds:
        raise ValueError(f"Training histories in {path} do not match configured seeds")
    for history in histories:
        steps = history.get("total_steps", [])
        rewards = history.get("mean_step_rewards", [])
        if not steps or steps[-1] != steps_per_seed or len(steps) != len(rewards):
            raise ValueError(f"Incomplete measured training history in {path}")
        if any(right <= left for left, right in zip(steps, steps[1:])):
            raise ValueError(f"Non-increasing training steps in {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-demo", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load((PROJECT_ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    seeds = [int(seed) for seed in config["seeds"]]
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("At least three distinct training seeds are required")
    steps_per_seed = int(config["total_timesteps_per_seed"])
    if steps_per_seed != int(config["epochs"]) * int(config["steps_per_epoch"]):
        raise ValueError("Configured step budget is internally inconsistent")

    custom = load_json(PROJECT_ROOT / "checkpoints" / "ppo_eval_summary.json")
    baseline = load_json(PROJECT_ROOT / "baselines" / "sb3_ppo_eval_summary.json")
    if not isinstance(custom, dict) or not isinstance(baseline, dict):
        raise TypeError("Evaluation summaries must be JSON objects")
    validate_summary(
        custom, protocol=CUSTOM_PROTOCOL, seeds=seeds, steps_per_seed=steps_per_seed
    )
    validate_summary(
        baseline, protocol=BASELINE_PROTOCOL, seeds=seeds, steps_per_seed=steps_per_seed
    )
    validate_histories(
        PROJECT_ROOT / "checkpoints" / "ppo_multi_seed_results.json", seeds, steps_per_seed
    )
    validate_histories(
        PROJECT_ROOT / "baselines" / "sb3_training_results.json", seeds, steps_per_seed
    )

    plot = PROJECT_ROOT / "write-up" / "benchmark_comparison.png"
    if not plot.is_file() or plot.stat().st_size == 0:
        raise FileNotFoundError("Measured benchmark plot is missing")
    submission = PROJECT_ROOT / "write-up" / "submission.md"
    text = submission.read_text(encoding="utf-8")
    if "@@" in text:
        raise ValueError("Submission still contains unresolved template tokens")
    word_count = len(re.findall(r"\b[\w'-]+\b", text))
    if word_count > 1000:
        raise ValueError(f"Submission exceeds 1,000 words: {word_count}")
    if args.require_demo:
        demo = PROJECT_ROOT / "write-up" / "demo.mp4"
        if not demo.is_file() or demo.stat().st_size == 0:
            raise FileNotFoundError("Recorded demo is missing")
        narration = PROJECT_ROOT / "write-up" / "demo-narration.wav"
        if not narration.is_file() or narration.stat().st_size == 0:
            raise FileNotFoundError("Demo narration is missing")
        with wave.open(str(narration), "rb") as audio:
            narration_seconds = audio.getnframes() / audio.getframerate()
        if not 60.0 <= narration_seconds <= 119.0:
            raise ValueError(
                f"Narrated demo must be between 60 and 119 seconds; got {narration_seconds:.1f}"
            )

    print(
        f"Submission audit passed: {len(seeds)} seeds, "
        f"{EVAL_EPISODES} episodes/seed, {word_count} words."
    )


if __name__ == "__main__":
    main()
