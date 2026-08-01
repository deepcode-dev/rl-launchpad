# 01 · Installation

## Local (Windows, CPU/GPU via CUDA wheels)

```powershell
uv sync --extra dev --locked
uv run pytest -q --basetemp=.pytest_tmp -p no:cacheprovider   # unit + contract tests
```

- Python is pinned to `>=3.12,<3.13` in `pyproject.toml`.
- `torch`, `torchvision`, `torchaudio` are pulled from the `pytorch-cu124` index
  (CUDA 12.4 wheels).
- `uv run <cmd>` places the repo root on `sys.path` for you, so
  `import ppo`, `import eval`, `import baselines` all work.

Smoke-test that physics + both ML stacks are importable:

```powershell
uv run python -c "import brax, jax, mujoco_playground as mp; import torch; print('ok', torch.cuda.is_available())"
```

> The pytest flags above (`--basetemp=.pytest_tmp -p no:cacheprovider`) redirect pytest's
> temp dirs and cache out of `%TEMP%`/`.pytest_cache`, which sandboxed Windows environments
> leave with restricted ACLs that raise `PermissionError: [WinError 5]` during test setup.

## Cluster (NUS SoC, `xlogin`)

```bash
bash cluster/setup.sh            # create/refresh .venv-cluster, pip-tmp handling
sbatch cluster/check_gpu.slurm   # verify JAX + PyTorch both see the GPU
squeue -u "$USER"                # watch it run
```

`cluster/setup.sh` installs into `.venv-cluster` (on `/mnt/scratch/$USER/...` when
available, otherwise `$HOME/.tmp/...`), disables pip's wheel cache to avoid
storing CUDA packages twice, and upgrades the shared cuDNN 9 runtime so that
PyTorch 2.6 does not downgrade it below what the JAX 0.11 CUDA wheel needs.

`cluster/check_gpu.slurm` prints `nvidia-smi`, JAX devices, and
`torch.cuda.is_available()`, and exits non-zero if either stack missed the GPU.

## `checkpoints/` is git-ignored

Everything under `checkpoints/` stays out of git (see `.gitignore`). If you want
artifacts in the repo, put them under `baselines/` (e.g. the Brax checkpoints) or
`NEW_checkpoints/` (the champion policies), which are tracked.
