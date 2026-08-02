# RL sponsored-track submission audit

Audit target: the Launchpad RL sponsored track requirements, checked against
the canonical 200M-step artifacts in this repository.

## Current evidence

| Requirement | Status | Evidence or remaining action |
|---|---|---|
| R1: from-scratch algorithm | Pass | `ppo/agent.py`, `ppo/ppo.py`, and `ppo/train_multi_seed.py`; Brax is kept under the baseline path. |
| R2: established baseline | Disclosed partial | Stock Brax is measured on the same task, dimensions, seeds, and evaluator. The custom EMA differs; post-hoc EMA ablation is recorded, but a fully matched baseline requires retraining Brax with EMA. |
| R3: reproducibility | Pass locally | `pyproject.toml`, `uv.lock`, configs, checkpoint metadata, and the validator are present. |
| R4: standardized final evaluation | Pass | Five custom and three Brax seeds each have 50 deterministic episodes from 20,000â€“20,049. Five Slurm logs provide 244 measured training points per custom seed, used for the custom mean/std curve. |
| R5: simulation and modifications | Pass with disclosure | Stock `Go1JoystickFlatTerrain`; wrapper autoreset, truncation handling, bounded actions, history, and EMA are declared. |
| R6: compute honesty | Disclosed partial | Custom wall times are in checkpoint metadata. Brax's 589.3-second time is documented, but its raw training log is not present. |
| Write-up | Pass on length/content | `write-up/submission.md` is under 1,000 words and contains architecture, loss, reward, modifications, evidence, failures, and limitations. |
| Video | Pass | `demo.mp4` is 120 seconds, uses three reported checkpoints (seeds 9033, 9016, and 8009), shows two 1,000-step deterministic evaluation-style command episodes per seed, and uses on-screen captions; narration is optional. |

## Verification commands

```powershell
.venv\Scripts\python.exe -m pytest -q --basetemp=.pytest_tmp -p no:cacheprovider
.venv\Scripts\ruff.exe check eval
.venv\Scripts\python.exe eval\validate_submission.py --require-demo
```

The non-strict validator reports the known gaps as warnings. Run this before
uploading after the remaining evidence is supplied:

```powershell
.venv\Scripts\python.exe eval\validate_submission.py --require-demo --strict
```

Strict mode should pass once at least three complete custom training histories
are present; spoken narration is optional because the demo uses on-screen
captions. A fully controlled EMA comparison additionally requires retraining
the baseline with the filter inside its environment.
