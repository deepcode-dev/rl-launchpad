"""Build measured custom-PPO training histories from Slurm stdout logs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


LINE_RE = re.compile(
    r"Seed (?P<seed>\d+) \| Epoch\s+(?P<epoch>\d+)/(?P<epochs>\d+) "
    r"\| Avg R: (?P<reward>[0-9.]+) \| Val Loss: (?P<val_loss>[0-9.]+) "
    r"\| KL: (?P<kl>[0-9.]+) \| LinErr: (?P<lin_err>[0-9.]+) "
    r"\| YawErr: (?P<yaw_err>[0-9.]+) \| \|A\|: (?P<abs_action>[0-9.]+) "
    r"\| R>0: (?P<positive_reward>[0-9.]+)% \| Done: (?P<done>[0-9.]+)% "
    r"\| Sigma: (?P<sigma>[0-9.]+) \| Steps: (?P<steps>\d+) "
    r"\| Time: (?P<time>[0-9.]+)s"
)


def parse_log(path: Path) -> dict:
    rows: list[dict[str, float | int]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = LINE_RE.search(line)
        if match:
            groups = match.groupdict()
            rows.append(
                {
                    "seed": int(groups["seed"]),
                    "epoch": int(groups["epoch"]),
                    "reward": float(groups["reward"]),
                    "val_loss": float(groups["val_loss"]),
                    "kl": float(groups["kl"]),
                    "lin_err": float(groups["lin_err"]),
                    "yaw_err": float(groups["yaw_err"]),
                    "abs_action": float(groups["abs_action"]),
                    "positive_reward": float(groups["positive_reward"]) / 100.0,
                    "done": float(groups["done"]) / 100.0,
                    "sigma": float(groups["sigma"]),
                    "steps": int(groups["steps"]),
                    "time": float(groups["time"]),
                }
            )
    if not rows:
        raise ValueError(f"No training metrics found in {path}")
    seed = int(rows[0]["seed"])
    if any(int(row["seed"]) != seed for row in rows):
        raise ValueError(f"Multiple seeds found in {path}")
    if rows[-1]["steps"] != 199_884_800:
        raise ValueError(f"{path} does not reach the 199,884,800-step budget")

    def values(key: str) -> list[float | int]:
        return [row[key] for row in rows]

    return {
        "seed": seed,
        "epochs": values("epoch"),
        "rewards": values("reward"),
        "mean_step_rewards": values("reward"),
        "val_losses": values("val_loss"),
        "approx_kls": values("kl"),
        "linear_velocity_errors": values("lin_err"),
        "yaw_rate_errors": values("yaw_err"),
        "mean_abs_actions": values("abs_action"),
        "positive_reward_fractions": values("positive_reward"),
        "termination_fractions": values("done"),
        "action_stds": values("sigma"),
        "total_steps": values("steps"),
        "wall_times": values("time"),
        "source_log": path.name,
        "sampling_note": "Measured stdout metrics emitted every 5 training epochs.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    histories = [parse_log(path) for path in args.logs]
    seeds = [int(history["seed"]) for history in histories]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Duplicate training seeds: {seeds}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(histories, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(histories)} measured histories ({len(histories[0]['epochs'])} points each) to {args.output}")


if __name__ == "__main__":
    main()
