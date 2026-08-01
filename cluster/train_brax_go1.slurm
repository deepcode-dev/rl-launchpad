#!/usr/bin/env bash
#SBATCH --job-name=go1-brax-ppo
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-80:1
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv-cluster/bin/activate

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export JAX_COMPILATION_CACHE_DIR="$HOME/.cache/jax"

CONFIG_PATH="${CONFIG_PATH:-configs/brax_go1_20m.yaml}"

echo "Host: $(hostname)"
echo "Config: $CONFIG_PATH"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

srun python -u cluster/train_brax_go1.py --config "$CONFIG_PATH"
