#!/usr/bin/env bash
# =============================================================================
# setup_env.sh — Reconstruct the Robometer conda environment from scratch.
# =============================================================================
set -euo pipefail

ENV_NAME="${1:-huggingface}"
PYTHON_VERSION="3.12"
CUDA_VERSION="12.8"
TORCH_VERSION="2.8.0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================================"
echo " Robometer environment setup"
echo " env name  : ${ENV_NAME}"
echo "============================================================"

if conda info --envs | grep -qw "${ENV_NAME}"; then
    echo "[1/5] Conda env '${ENV_NAME}' already exists. Reusing."
else
    echo "[1/5] Creating conda env '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
    conda create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}"
fi

eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

echo "[2/5] Installing PyTorch..."
pip install --upgrade \
    "torch==${TORCH_VERSION}" \
    "torchvision" \
    "torchaudio" \
    --index-url "https://download.pytorch.org/whl/cu${CUDA_VERSION//./}"

echo "[3/5] Installing Python dependencies..."
pip install --upgrade pip setuptools wheel
pip install \
    "transformers>=4.57" \
    "accelerate>=1.9.0" \
    "peft" \
    "unsloth>=2025.10" \
    "bitsandbytes" \
    "safetensors" \
    "datasets" \
    "qwen-vl-utils[decord]" \
    "decord" \
    "opencv-python-headless" \
    "av" \
    "numpy" \
    "matplotlib" \
    "seaborn" \
    "pyrallis" \
    "pyyaml" \
    "rich" \
    "loguru" \
    "tqdm"

echo "[4/5] Installing robometer package (editable)..."
pip install -e "${SCRIPT_DIR}" --no-deps

echo "[5/5] Done!"
