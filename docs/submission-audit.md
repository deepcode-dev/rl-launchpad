# RL sponsored-track submission audit

Audit target: the Launchpad RL sponsored track requirements, checked against
the canonical 131k-v2 artifacts in this repository.

## Current evidence

| Requirement | Status | Evidence |
|---|---|---|
| R1: from-scratch algorithm | Pass | `ppo/agent.py`, `ppo/ppo.py`, and `ppo/train_multi_seed.py`; Brax is baseline-only. |
| R2: established baseline | Disclosed partial | Stock Brax is measured on the same task, dimensions, seeds, and evaluator. The custom EMA differs; the post-hoc ablation is recorded, but a fully matched baseline requires retraining Brax with EMA. |
| R3: reproducibility | Pass locally | `pyproject.toml`, `uv.lock`, configs, checkpoint metadata, and the validator are present. |
| R4: standardized final evaluation | Pass | Three custom and three Brax seeds each have 50 deterministic episodes from 20,000-20,049. Three 131k-v2 Slurm logs provide 152 measured training points per custom seed for the mean/std curve. |
| R5: simulation and modifications | Pass with disclosure | Stock `Go1JoystickFlatTerrain`; wrapper autoreset, truncation handling, bounded actions, history, and EMA are declared. |
| R6: compute honesty | Disclosed partial | Custom wall times are in checkpoint metadata. Brax's 589.3-second time is documented, but its raw training log is not present. |
| Write-up | Pass on length/content | `write-up/submission.md` contains architecture, loss, reward, modifications, evidence, failures, limitations, and repository paths. |
| Video | Pass | `demo.mp4` is at most 120 seconds, uses seeds 13039, 13079, and 13027, shows two 1,000-step deterministic evaluation-style episodes per seed, and uses on-screen captions; narration is optional. |

## Verification commands

```powershell
.venv\Scripts\python.exe -m pytest -q --basetemp=.pytest_tmp -p no:cacheprovider
.venv\Scripts\ruff.exe check eval
.venv\Scripts\python.exe eval\validate_submission.py --require-demo --strict
```

The audit is designed to run from a fresh clone. The project depends on the
local Python environment, MuJoCo Playground, JAX, PyTorch, and GPU hardware
for training; committed evaluation artifacts are not mocked or dependent on
an external service.
