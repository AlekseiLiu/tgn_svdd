#!/usr/bin/env bash
# Minimal conda environment setup for the canonical TGN-SVDD pipeline
# This installs only what the active (non-legacy) code path requires.
#
# Usage:
#   bash install_main_env.sh [env_name]
# Example:
#   bash install_main_env.sh tgn_svdd_main

set -euo pipefail

ENV_NAME="${1:-tgn_svdd_main}"

echo "Creating conda environment: ${ENV_NAME}"

# Detect CUDA capability via nvidia-smi (if present)
HAS_CUDA=0
if command -v nvidia-smi >/dev/null 2>&1; then
  if nvidia-smi >/dev/null 2>&1; then
    HAS_CUDA=1
  fi
fi

# Create env with Python 3.10 (widely compatible with PyTorch+PyG)
conda create -y -n "${ENV_NAME}" python=3.10

# Activate env (works in bash; for zsh/others adjust accordingly)
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

# Channels
conda config --env --add channels pytorch
conda config --env --add channels conda-forge

# Install PyTorch (GPU if available; CPU otherwise)
if [[ "${HAS_CUDA}" -eq 1 ]]; then
  echo "Installing GPU-enabled PyTorch"
  conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
else
  echo "Installing CPU-only PyTorch"
  conda install -y pytorch torchvision torchaudio cpuonly -c pytorch
fi

# Install remaining Python packages via pip (including torch-geometric)
# torch is already installed via conda; pip will satisfy remaining deps from src/requirements.txt

# Core scientific stack and utilities via pip to match src/requirements.txt
python -m pip install --upgrade pip
python -m pip install -r src/requirements.txt

# Verify critical imports
python - <<'PY'
import torch
import torch_geometric
import numpy, pandas, sklearn, tqdm, matplotlib, seaborn, scipy
print('OK: torch', torch.__version__)
print('OK: torch_geometric', torch_geometric.__version__)
print('OK: numpy')
print('OK: pandas')
print('OK: sklearn')
print('OK: tqdm')
print('OK: matplotlib')
print('OK: seaborn')
print('OK: scipy')
PY

echo "\nEnvironment '${ENV_NAME}' is ready. To use it:"
echo "  conda activate ${ENV_NAME}"
echo "  python -m src.main --quick-test --epochs 2 --verbose"
