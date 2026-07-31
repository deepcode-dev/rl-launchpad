# 04 · Evaluate checkpoints

## Fixed evaluation contract

Every reported number uses the same protocol:

- **Physics**: native C++ MuJoCo (`impl="jax"` only during training).
- **Episodes**: 50 fixed, disjoint episodes, `eval_seed = 20000` (episode seeds
  are `20000 .. 20049`).
- **Reward**: simplified command-tracking contract
  `max(0, 1 − ‖v_xy − v_cmd‖²) · dt` evaluated on native MuJoCo — NOT the raw
  training reward. This is the shared "ruler" used to compare the custom PPO and
  the Brax baseline. (Training logs' reward numbers use the full stock reward and
  are only for monitoring.)
- A robot whose torso height drops below `0.15 m` counts as a fall; the episode
  ends.

## `eval/evaluate.py` — custom PyTorch PPO checkpoints

| CLI | Default | Meaning |
| --- | --- | --- |
| `--checkpoint` | *(required)* | `.pt` policy to evaluate |
| `--config` | `configs/default.yaml` | checkpoint contract / env dims |
| `--eval-seed` | `20000` | base of the 50 disjoint episode seeds |
| `--num-episodes` | `50` | number of episodes |

```powershell
uv run python eval/evaluate.py --checkpoint NEW_checkpoints/ppo_v2/ppo_seed9033.pt
uv run python eval/evaluate.py --checkpoint NEW_checkpoints/ppo_v2/ppo_seed9033.pt --num-episodes 50
```

Writes `<stem>_eval.json` next to the checkpoint: per-episode rewards/lengths,
mean/std return, linear-velocity error, yaw-rate error, latency percentiles, and
the checkpoint metadata. The CLI also prints a summary block.

## `eval/eval_brax_seeds.py` — Brax baseline checkpoints

Brax checkpoints are *step directories* (containing `ppo_network_config.json`),
not `.pt` files. This script finds the newest step dir in the given directory and
evaluates it under the same 50-episode native-MuJoCo contract.

| CLI | Default | Meaning |
| --- | --- | --- |
| `--checkpoint-dir` | `baselines/brax_go1_200m` | parent directory of the step dirs |

```powershell
uv run python eval/eval_brax_seeds.py                                                          # seed 10
uv run python eval/eval_brax_seeds.py --checkpoint-dir baselines/brax_go1_200m_seed11           # seed 11
uv run python eval/eval_brax_seeds.py --checkpoint-dir baselines/brax_go1_200m_seed12           # seed 12
```

## `eval/eval_all_seeds.py` — every custom seed in one shot

Reads `configs/default.yaml` (key `seeds`), evaluates each configured seed, and
writes a grand-mean summary to `checkpoints/ppo_eval_summary.json`.

```powershell
uv run python eval/eval_all_seeds.py
```

It refuses to run if any configured seed checkpoint is missing (no partial
multi-seed results).

## `eval/eval_200m_seeds.py`

Hard-coded evaluator for the older 200M cluster seeds `[10, 20, 30]`
(`checkpoints/cluster_100m_v2/ppo_seed<seed>.pt`). Prints per-seed results and a
grand mean, and writes `checkpoints/cluster_100m_v2/eval_200m_summary.json`:

```powershell
uv run python eval/eval_200m_seeds.py
```

> The champion set summary (`NEW_checkpoints/ppo_v2/ppo_v2_eval_summary.json`)
> is built by `scratch/build_ppo_v2_summary.py`, not by this script.

## `eval/validate_submission.py`

Checks the submission artifacts (eval summaries, training histories, demo).
Add `--require-demo` to require a demo video:

```powershell
uv run python eval/validate_submission.py --require-demo
```
