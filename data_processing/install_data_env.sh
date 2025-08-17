#!/bin/bash
# Minimal environment setup for data processing
# Installs: pandas, numpy, scikit-learn, nfstream
# Usage: bash install_data_env.sh

ENV_NAME="tgn_data"
PYTHON_VERSION=3.8

# Create conda environment if it doesn't exist
echo "Creating conda environment: $ENV_NAME (Python $PYTHON_VERSION)"
conda create -y -n $ENV_NAME python=$PYTHON_VERSION

# Activate environment
echo "Activating environment: $ENV_NAME"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

# Install minimal dependencies
conda install -y pandas numpy scikit-learn
pip install nfstream

echo "\nEnvironment '$ENV_NAME' is ready. To activate, run:"
echo "  conda activate $ENV_NAME"
echo "\nYou can now run data processing scripts, e.g.:"
echo "  python cic_2017_preprocess.py"
