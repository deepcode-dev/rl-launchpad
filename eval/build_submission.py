"""Build the numeric submission sections only from verified result artifacts."""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path, protocol: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Submission evidence is missing: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("protocol") != protocol:
        raise ValueError(f"Unexpected protocol in {path}: {data.get('protocol')!r}")
    if len(data.get("seed_results", [])) < 3:
        raise ValueError(f"At least three complete training seeds are required in {path}")
    return data


def build_submission() -> Path:
    custom = _load(
        PROJECT_ROOT / "checkpoints" / "ppo_eval_summary.json",
        "custom-ppo-native-reward-v1",
    )
    baseline = _load(
        PROJECT_ROOT / "baselines" / "sb3_ppo_eval_summary.json",
        "sb3-baseline-native-reward-v1",
    )
    if custom["training_seeds"] != baseline["training_seeds"]:
        raise ValueError("Custom and baseline summaries use different training seeds")
    if custom["num_episodes_per_seed"] != baseline["num_episodes_per_seed"]:
        raise ValueError("Custom and baseline summaries use different episode counts")

    custom_metadata = [result["checkpoint_metadata"] for result in custom["seed_results"]]
    baseline_metadata = [result["checkpoint_metadata"] for result in baseline["seed_results"]]
    steps = {metadata["total_env_steps"] for metadata in custom_metadata}
    if len(steps) != 1:
        raise ValueError("Custom checkpoints do not use one matched environment-step budget")

    replacements = {
        "@@SEEDS@@": ", ".join(str(seed) for seed in custom["training_seeds"]),
        "@@EPISODES@@": str(custom["num_episodes_per_seed"]),
        "@@STEPS@@": f"{steps.pop():,}",
        "@@CUSTOM_MEAN@@": f"{custom['grand_mean_return']:.3f}",
        "@@CUSTOM_STD@@": f"{custom['grand_std_return']:.3f}",
        "@@CUSTOM_LENGTH@@": f"{custom['mean_episode_length']:.1f}",
        "@@CUSTOM_TIME@@": f"{sum(m['wall_time_seconds'] for m in custom_metadata) / len(custom_metadata):.1f}",
        "@@BASELINE_MEAN@@": f"{baseline['grand_mean_return']:.3f}",
        "@@BASELINE_STD@@": f"{baseline['grand_std_return']:.3f}",
        "@@BASELINE_LENGTH@@": f"{baseline['mean_episode_length']:.1f}",
        "@@BASELINE_TIME@@": f"{sum(m['wall_time_seconds'] for m in baseline_metadata) / len(baseline_metadata):.1f}",
    }
    template_path = PROJECT_ROOT / "write-up" / "submission_template.md"
    rendered = template_path.read_text(encoding="utf-8")
    for token, value in replacements.items():
        rendered = rendered.replace(token, value)
    unresolved = [token for token in replacements if token in rendered]
    if unresolved:
        raise ValueError(f"Unresolved submission tokens: {unresolved}")
    output = PROJECT_ROOT / "write-up" / "submission.md"
    output.write_text(rendered, encoding="utf-8")
    print(f"Built evidence-backed submission: {output}")
    return output


if __name__ == "__main__":
    build_submission()
