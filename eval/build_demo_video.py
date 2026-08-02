"""Assemble a captioned or narrated, at-most-two-minute submission demo."""

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


def make_silent_audio(path: Path, duration: float, sample_rate: int = 16_000) -> None:
    """Create a deterministic silent track for captioned-demo fallback builds."""
    frame_count = int(round(duration * sample_rate))
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(b"\x00\x00" * frame_count)


def build_demo(policy_footage: Path, narration: Path | None, output: Path, *, silent: bool = False) -> None:
    if silent:
        duration = 107.0
        narration = PROJECT_ROOT / "write-up" / "demo-silence.wav"
        make_silent_audio(narration, duration)
    if narration is None:
        raise ValueError("Provide --narration or use --silent")
    duration = audio_duration(narration)
    if duration > 119.0:
        raise ValueError(f"Narration is too long for the challenge limit: {duration:.1f}s")
    assets = PROJECT_ROOT / "write-up" / "demo-assets"
    assets.mkdir(parents=True, exist_ok=True)

    slides = [
        ("title", "From-Scratch PPO for Go1", ["Five custom seeds · 200M steps each", "Fifty fixed held-out episodes per seed"]),
        ("failures", "Why the first policy failed", ["GAE crossed vector trajectories", "Terminal robots never reset", "Reward semantics changed", "Gaussian actions were unbounded"]),
        ("contract", "Corrected training contract", ["48-dim actor · 123-dim privileged critic", "GAE stays [time, environment]", "Stock reward for training; shared metric for eval", "Tanh actions · declared 0.7/0.3 EMA filter"]),
        ("loss", "PPO update in one line", ["L = -min(rA, clip(r, 0.8, 1.2)A)", "+ 0.5 Huber(V, target) - 0.01 entropy", "GAE is computed before flattening", "Four update passes · batch size 5,120"]),
        ("result", "Measured held-out result", ["Custom: 19.752 +/- 0.020 return · LinErr 0.0851", "Brax: 19.821 +/- 0.016 return · LinErr 0.0668", "200M steps · 5 custom / 3 baseline seeds", "Every reported episode reached 1,000 steps"]),
        ("limits", "What this does not prove", ["Custom mean training time: 13,561 seconds", "Brax reference: documented 589.3 seconds", "Curve logs are sampled every five epochs", "Simulation only; no sim-to-real claim"]),
    ]
    slide_paths: dict[str, Path] = {}
    for name, title, lines in slides:
        path = assets / f"{name}.png"
        make_slide(path, title, lines)
        slide_paths[name] = path

    # Percentages keep the visual timeline synchronized if the local TTS voice
    # changes speaking duration slightly.
    visual_plan = [
        ("image", slide_paths["title"], 0.05),
        ("video", policy_footage, 0.38),
        ("image", slide_paths["failures"], 0.10),
        ("image", slide_paths["contract"], 0.10),
        ("image", slide_paths["loss"], 0.10),
        ("image", PROJECT_ROOT / "write-up" / "benchmark_comparison.png", 0.14),
        ("image", slide_paths["result"], 0.08),
        ("image", slide_paths["limits"], 0.05),
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
    label = "captioned demo" if silent else "narrated demo"
    print(f"Built {label} ({duration:.1f}s): {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--footage", type=Path, default=PROJECT_ROOT / "write-up" / "policy-footage.mp4")
    parser.add_argument("--narration", type=Path, default=PROJECT_ROOT / "write-up" / "demo-narration.wav")
    parser.add_argument("--silent", action="store_true", help="Build a captioned demo with a silent audio track")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "write-up" / "demo.mp4")
    args = parser.parse_args()
    build_demo(args.footage, args.narration, args.output, silent=args.silent)


if __name__ == "__main__":
    main()
