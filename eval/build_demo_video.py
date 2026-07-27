"""Assemble the narrated, at-most-two-minute submission demo."""

from __future__ import annotations

import argparse
import subprocess
import wave
from pathlib import Path

import imageio_ffmpeg
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT, FPS = 960, 720, 50


def make_slide(path: Path, title: str, lines: list[str], *, accent: str = "#38bdf8") -> None:
    fig = plt.figure(figsize=(WIDTH / 100, HEIGHT / 100), dpi=100, facecolor="#07111f")
    axis = fig.add_axes((0, 0, 1, 1))
    axis.set_facecolor("#07111f")
    axis.axis("off")
    axis.text(0.07, 0.84, title, color=accent, fontsize=30, fontweight="bold", va="top")
    y = 0.69
    for line in lines:
        axis.text(0.08, y, line, color="white", fontsize=19, va="top", wrap=True)
        y -= 0.105
    axis.text(0.07, 0.07, "Launchpad RL Track · Go1JoystickFlatTerrain", color="#94a3b8", fontsize=13)
    fig.savefig(path, dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)


def run_ffmpeg(arguments: list[str]) -> None:
    executable = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([executable, "-hide_banner", "-loglevel", "error", "-y", *arguments], check=True)


def audio_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as audio:
        return audio.getnframes() / audio.getframerate()


def build_demo(policy_footage: Path, narration: Path, output: Path) -> None:
    duration = audio_duration(narration)
    if duration > 119.0:
        raise ValueError(f"Narration is too long for the challenge limit: {duration:.1f}s")
    assets = PROJECT_ROOT / "write-up" / "demo-assets"
    assets.mkdir(parents=True, exist_ok=True)

    slides = [
        ("title", "From-Scratch PPO for Go1", ["Native MuJoCo Playground reward", "Three training seeds · fifty held-out episodes each"]),
        ("failures", "Why the first policy failed", ["GAE crossed vector trajectories", "Terminal robots never reset", "Reward semantics changed", "Gaussian actions were unbounded"]),
        ("contract", "Corrected training contract", ["GAE stays [time, environment]", "Termination and truncation bootstrap separately", "Per-slot autoreset and terminal observation", "Tanh actions · stored observation normalization"]),
        ("result", "Measured held-out result", ["Custom PPO: 10.913 return · 746.1 steps", "Matched SB3: 1.310 return · 281.9 steps", "Same 327,680-step budget per seed", "Long overtrained run retained as negative evidence"]),
        ("limits", "What this does not prove", ["Windows MJX simulation ran on CPU", "Viewer pushes do not alter MJX policy state", "Early falls remain in the JSON distributions", "No sim-to-real robustness claim"]),
    ]
    slide_paths: dict[str, Path] = {}
    for name, title, lines in slides:
        path = assets / f"{name}.png"
        make_slide(path, title, lines)
        slide_paths[name] = path

    # Percentages keep the visual timeline synchronized if the local TTS voice
    # changes speaking duration slightly.
    visual_plan = [
        ("image", slide_paths["title"], 0.08),
        ("video", policy_footage, 0.14),
        ("image", slide_paths["failures"], 0.16),
        ("image", slide_paths["contract"], 0.18),
        ("image", PROJECT_ROOT / "write-up" / "benchmark_comparison.png", 0.22),
        ("image", slide_paths["result"], 0.14),
        ("image", slide_paths["limits"], 0.08),
    ]
    segment_paths: list[Path] = []
    video_filter = (
        f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=decrease,"
        f"pad={WIDTH}:{HEIGHT}:(ow-iw)/2:(oh-ih)/2:color=#07111f,fps={FPS},format=yuv420p"
    )
    for index, (kind, source, share) in enumerate(visual_plan):
        segment = assets / f"segment-{index:02d}.mp4"
        segment_duration = duration * share + (0.25 if index == len(visual_plan) - 1 else 0.0)
        input_args = ["-loop", "1", "-i", str(source)] if kind == "image" else ["-stream_loop", "-1", "-i", str(source)]
        run_ffmpeg([
            *input_args, "-t", f"{segment_duration:.3f}", "-vf", video_filter,
            "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "20", str(segment),
        ])
        segment_paths.append(segment)

    concat_file = assets / "segments.txt"
    concat_file.write_text(
        "".join(f"file '{path.as_posix()}'\n" for path in segment_paths), encoding="utf-8"
    )
    visuals = assets / "visuals.mp4"
    run_ffmpeg(["-f", "concat", "-safe", "0", "-i", str(concat_file), "-c", "copy", str(visuals)])
    run_ffmpeg([
        "-i", str(visuals), "-i", str(narration), "-c:v", "copy", "-c:a", "aac",
        "-b:a", "160k", "-shortest", str(output),
    ])
    print(f"Built narrated demo ({duration:.1f}s): {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--footage", type=Path, default=PROJECT_ROOT / "write-up" / "policy-footage.mp4")
    parser.add_argument("--narration", type=Path, default=PROJECT_ROOT / "write-up" / "demo-narration.wav")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "write-up" / "demo.mp4")
    args = parser.parse_args()
    build_demo(args.footage, args.narration, args.output)


if __name__ == "__main__":
    main()
