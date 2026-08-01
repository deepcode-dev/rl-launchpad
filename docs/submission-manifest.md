# Submission manifest

Use this list when preparing the RL sponsored-track upload. The repository
must include the source, pinned environment, canonical evaluation summaries,
and plot. The video is a separate deliverable but is also unignored locally so
it can be included if desired.

## Repository link

- `README.md`
- `pyproject.toml`
- `uv.lock`
- `ppo/agent.py`, `ppo/ppo.py`, `ppo/env.py`, `ppo/train_multi_seed.py`
- `configs/champion_v2.yaml`
- `eval/evaluate.py`, `eval/validate_submission.py`, `eval/plot_benchmark.py`
- `cluster/train_brax_go1.py` as baseline-only code
- `NEW_checkpoints/ppo_v2/ppo_seed9033.pt` and its sidecar metadata
- All five canonical custom evaluation JSON files and
  `NEW_checkpoints/ppo_v2/ppo_v2_eval_summary.json`
- `baselines/brax_go1_200m_eval_summary.json`
- `baselines/brax_go1_200m_ema_ablation.json`
- `write-up/benchmark_comparison.png`
- `write-up/submission.md`
- `docs/submission-audit.md`

## Separate demo upload

- `write-up/demo.mp4` — current 88-second narrated demo
- `write-up/policy-footage.mp4` — seed 9033 source footage
- `write-up/demo-narration.txt` — transcript used to generate the narration

Do not upload `write-up/demo-narration.wav`; the local file is an invalid
46-byte failed TTS output. Do not use the older unreferenced GIF as evidence.

## Before final upload

1. Use the committed 88-second narrated `demo.mp4`; it already includes the
   narration WAV and three labeled evaluation-style command clips from the
   reported seed 9033 checkpoint.
2. Add at least two more complete custom training histories, then rerun
   `eval/plot_benchmark.py` so the custom curve has mean and standard
   deviation across seeds.
3. Add the raw Brax training log if it is available.
4. Stage the new files that are currently untracked, then run:

```powershell
.venv\Scripts\python.exe eval\validate_submission.py --require-demo --strict
```
