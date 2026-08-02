# Submission manifest

Use this list when preparing the RL sponsored-track upload. The repository
contains the source, pinned environment, canonical evaluation summaries, and
measured plot. The video is also present locally for the separate upload.

## Repository link

- `README.md`
- `pyproject.toml`, `uv.lock`
- `ppo/agent.py`, `ppo/ppo.py`, `ppo/env.py`, `ppo/train_multi_seed.py`
- `configs/cluster_131k_v2.yaml`
- `eval/evaluate.py`, `eval/validate_submission.py`, `eval/plot_benchmark.py`
- `cluster/train_brax_go1.py` as baseline-only code
- `NEW_checkpoints/ppo_v2/ppo_v2_eval_summary.json`
- `NEW_checkpoints/ppo_v2/ppo_multi_seed_results.json`
- The three selected checkpoint/evaluation pairs for seeds 13039, 13079, and
  13027, including their `.pt.meta.json` sidecars
- `baselines/brax_go1_200m_eval_summary.json`
- `baselines/brax_go1_200m_ema_ablation.json`
- `write-up/benchmark_comparison.png`
- `write-up/submission.md`
- `docs/submission-audit.md`

## Separate demo upload

- `write-up/demo.mp4` — captioned two-minute six-clip montage of seeds 13039,
  13079, and 13027
- `write-up/policy-footage.mp4` — concatenated source footage from those seeds
- `write-up/demo-script.md` — reproducible capture commands and video contract

Spoken narration is optional and is not used by the current demo. Older
diagnostic media is not part of the evidence.

## Before final upload

1. Use the captioned `demo.mp4`; it contains two 1,000-step deterministic
   evaluation-style command episodes for each reported seed.
2. Rerun `eval/plot_benchmark.py` after any seed or history change.
3. Add the raw Brax training log only if it becomes available; its absence is
   already disclosed in the write-up.
4. Stage the final files and run:

```powershell
.venv\Scripts\python.exe eval\validate_submission.py --require-demo --strict
```
