#!/bin/bash

echo "Activating 'huggingface' conda environment..."
# Source conda to ensure activate works in the script
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate huggingface

echo "Installing required dependencies..."
pip install transformers accelerate safetensors qwen-vl-utils torchvision qwen-vl-utils[decord]

# Prompt for flash-attn
read -p "Do you want to install flash-attn for speed and memory savings? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Installing flash-attn (this may take a while)..."
    pip install flash-attn --no-build-isolation
fi

echo "Done! You can now run the inference script using:"
echo "conda activate huggingface"
echo "python run_roboreward.py --video path/to/video.mp4 --task 'Task description'"
