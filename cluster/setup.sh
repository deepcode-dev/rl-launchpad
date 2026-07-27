#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Login-node /tmp is quota-limited to 50 MB. Use the user's persistent scratch
# allocation for wheel downloads/extraction and avoid retaining a second cached
# copy of multi-gigabyte CUDA dependencies.
SCRATCH_ROOT="${RL_SCRATCH_DIR:-/mnt/scratch/$USER/rl-launchpad}"
if ! mkdir -p "$SCRATCH_ROOT/pip-tmp" 2>/dev/null; then
  SCRATCH_ROOT="$HOME/.tmp/rl-launchpad"
  mkdir -p "$SCRATCH_ROOT/pip-tmp"
fi
export TMPDIR="$SCRATCH_ROOT/pip-tmp"
export PIP_NO_CACHE_DIR=1

python3 -m venv .venv-cluster
source .venv-cluster/bin/activate
python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Install both frameworks with CUDA 12 support. The cluster driver is newer and
# remains backwards-compatible with these wheels.
python -m pip install --no-cache-dir \
  torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install --no-cache-dir "jax[cuda12]==0.11.0"
python -m pip install --no-cache-dir -e .
# torch 2.6 pins cuDNN 9.1, while the JAX CUDA wheel is compiled against
# cuDNN 9.8+. A newer minor release is backward-compatible with torch and is
# required by JAX. Install it last without letting pip re-resolve torch.
python -m pip install --no-cache-dir --upgrade --no-deps \
  "nvidia-cudnn-cu12>=9.8,<10"

python - <<'PY'
from importlib.metadata import version

print("torch:", version("torch"))
print("jax:", version("jax"))
print("Run cluster/check_gpu.slurm to verify CUDA from an allocated GPU node.")
PY
