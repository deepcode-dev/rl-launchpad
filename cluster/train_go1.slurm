#!/usr/bin/env bash
#SBATCH --job-name=go1-custom-ppo
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load cuda 2>/dev/null || module load cuda/12.0 2>/dev/null || true
source .venv-cluster/bin/activate

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.65
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export JAX_COMPILATION_CACHE_DIR="$HOME/.cache/jax"

CONFIG_PATH="${CONFIG_PATH:-configs/cluster_smoke.yaml}"
TRAIN_SEED="${TRAIN_SEED:-10}"

echo "Host: $(hostname)"
echo "Config: $CONFIG_PATH"
echo "Seed: $TRAIN_SEED"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

RESUME_FLAG=""
if [ "${RESUME:-false}" = "true" ]; then
    RESUME_FLAG="--resume"
fi

srun python -u ppo/train_multi_seed.py --config "$CONFIG_PATH" --seed "$TRAIN_SEED" $RESUME_FLAG
