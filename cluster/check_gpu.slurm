#!/usr/bin/env bash
#SBATCH --job-name=go1-gpu-check
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv-cluster/bin/activate

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nvidia-smi
srun python - <<'PY'
import jax
import torch

print("JAX devices:", jax.devices())
print("PyTorch CUDA available:", torch.cuda.is_available())
print("PyTorch device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
if not any(device.platform == "gpu" for device in jax.devices()):
    raise SystemExit("JAX did not detect the allocated GPU")
if not torch.cuda.is_available():
    raise SystemExit("PyTorch did not detect the allocated GPU")
PY
