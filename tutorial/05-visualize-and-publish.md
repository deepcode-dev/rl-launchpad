# 05 · Visualize, record, and publish

All viewers run on Windows with a local display (GLFW). They load a checkpoint,
run the policy inside native C++ MuJoCo, and open a window.

## Live viewers

### `eval/view_native_v2.py` — champion custom PPO (recommended)

```powershell
uv run python eval/view_native_v2.py NEW_checkpoints/ppo_v2/ppo_seed9033.pt
```

| CLI | Default | Meaning |
| --- | --- | --- |
| `checkpoint` | *(positional)* | `.pt` policy |
| `--config` | `configs/champion_v2.yaml` | checkpoint contract |
| `--command` | `0.8 0.0 0.0` | initial `(VX, VY, YAW)` command |

Keyboard: **W/A/S/D** move, **Q/E** yaw, **1/2/3/4** speed presets.

### `eval/view_native.py` — generic custom PPO

```powershell
uv run python eval/view_native.py checkpoints/ppo_seed10.pt --config configs/default.yaml
```

### `eval/view_live.py` — MJX-wrapper viewer

Runs the production `MJXVectorPyTorchWrapper` (JAX physics) instead of native
MuJoCo:

```powershell
uv run python eval/view_live.py checkpoints/ppo_seed10.pt
```

### `eval/view_brax_native.py` — Brax baseline checkpoint

`checkpoint` is a Brax **step directory** (not a `.pt`):

```powershell
uv run python eval/view_brax_native.py baselines/brax_go1_200m/000200540160 --command 0.8 0 0
```

| CLI | Default | Meaning |
| --- | --- | --- |
| `--command` | `0.8 0.0 0.0` | `(VX, VY, YAW)` |
| `--max-steps` | `100000` | steps before episode reset |

## Record videos

### `eval/record_native_video.py` — silky native MuJoCo video

```powershell
uv run python eval/record_native_video.py NEW_checkpoints/ppo_v2/ppo_seed2005.pt `
  --output write-up/quadruped_walking.mp4 --command 1.0 0.0 0.0 --max-steps 500 --fps 50
```

| CLI | Default | Meaning |
| --- | --- | --- |
| `checkpoint` | *(positional)* | policy |
| `--output` | `write-up/quadruped_walking.mp4` | output path |
| `--command` | `1.0 0.0 0.0` | `(VX, VY, YAW)` |
| `--max-steps` | `500` | episode length to render |
| `--fps` | `50` | frames per second |

### `eval/record_video.py` — alternate recorder

```powershell
uv run python eval/record_video.py NEW_checkpoints/ppo_v2/ppo_seed9033.pt --output write-up/demo.mp4
```

(`checkpoint`, `--output`, `--seed` [default `20000`], `--max-steps` [default `500`], `--fps` [default `50`].)

### `eval/render_rollout.py` — single rollout images

```powershell
uv run python eval/render_rollout.py NEW_checkpoints/ppo_v2/ppo_seed9033.pt
```

(`checkpoint`, `--seed` [default `20000`], `--config` [default `configs/default.yaml`].)

### `eval/build_demo_video.py` — stitch footage + narration

```powershell
uv run python eval/build_demo_video.py
```

(`--footage` [default `write-up/policy-footage.mp4`], `--narration` [default
`write-up/demo-narration.wav`], `--output` [default `write-up/demo.mp4`].)

## Benchmark plot

### `eval/plot_benchmark.py`

Regenerates `write-up/benchmark_comparison.png` (4-panel: training curves, mean
return, LinErr, YawErr; custom vs Brax baseline):

```powershell
uv run python eval/plot_benchmark.py
```

| CLI | Default |
| --- | --- |
| `--custom-summary` | `NEW_checkpoints/ppo_v2/ppo_v2_eval_summary.json` |
| `--custom-training` | `NEW_checkpoints/ppo_v2/ppo_multi_seed_results.json` |
| `--sb3-summary` | `baselines/sb3_ppo_eval_summary.json` |
| `--sb3-training` | `baselines/sb3_training_results.json` |
| `--output` | `write-up/benchmark_comparison.png` |

## Submission

- `eval/build_submission.py` — assembles the submission report from the eval
  summaries (requires protocol-compatible summaries).
- `eval/validate_submission.py` — see [04-evaluate.md](04-evaluate.md).
