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

- `write-up/demo.mp4` â€” current 120-second captioned six-clip montage of seeds 9033, 9016, and 8009
- `write-up/policy-footage.mp4` â€” concatenated source footage from those three seeds
- `write-up/demo-script.md` â€” reproducible capture commands and video contract

Spoken narration is optional and is not used by the current demo. The older
diagnostic media is not part of the evidence.

## Before final upload

1. Use the captioned `demo.mp4`; it contains six labeled deterministic
   evaluation-style command clips from reported checkpoints: two 1,000-step episodes for each seed.
2. Add at least two more complete custom training histories if the 131k-v2
   result becomes official, then rerun `eval/plot_benchmark.py` so the custom
   curve has mean and standard deviation across seeds.
3. Add the raw Brax training log if it is available.
4. Stage the final files and run:

```powershell
.venv\Scripts\python.exe eval\validate_submission.py --require-demo --strict
```
