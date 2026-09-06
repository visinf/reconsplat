#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-reconsplat}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -y -n "$ENV_NAME" python=3.10
conda activate "$ENV_NAME"

pip install "setuptools<81"
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Build the custom CUDA rasterization kernel. This needs nvcc plus the matching CUDA 11.8
# headers/libraries in this env, without requiring a system-wide CUDA install.
conda config --set channel_priority flexible
conda install -y -c 'nvidia/label/cuda-11.8.0' cuda-toolkit=11.8.0

# Explicitly set target GPU architectures instead of relying on torch auto-detecting a GPU, 
# only needed if you are building from a cluster login/CPU node. 
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0"
pip install --no-build-isolation submodules/variational-gaussian-rasterization

echo "Done. Activate the env with: conda activate $ENV_NAME"
