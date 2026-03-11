#!/usr/bin/env python3
"""
Static inference script for Robometer.
Loads the model, extracts frames from a video at 1.0 FPS,
runs a single batch inference, saves a plot and the predictions.
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from robometer.data.dataset_types import ProgressSample, Trajectory
from robometer.evals.eval_server import compute_batch_outputs
from robometer.evals.eval_viz_utils import create_combined_progress_success_plot, extract_frames
from robometer.utils.save import load_model_from_hf
from robometer.utils.setup_utils import setup_batch_collator

def main():
    parser = argparse.ArgumentParser(description="Static Robometer inference")
    parser.add_argument("--model-path", default="robometer/Robometer-4B")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--task", required=True, help="Task description")
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=512)
    parser.add_argument("--success-threshold", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    exp_config, tokenizer, processor, reward_model = load_model_from_hf(
        model_path=args.model_path, device=device,
    )
    reward_model.eval()
    batch_collator = setup_batch_collator(processor, tokenizer, exp_config, is_eval=True)

    # Extract frames
    frames = extract_frames(args.video, fps=args.fps, max_frames=args.max_frames)
    if frames is None or frames.size == 0:
        raise RuntimeError("Could not extract frames.")
    if frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 255).astype(np.uint8)

    T = int(frames.shape[0])
    print(f"Extracted {T} frames.")

    # Prepare trajectory
    traj = Trajectory(
        frames=frames,
        frames_shape=tuple(frames.shape),
        task=args.task,
        id="0",
        metadata={"subsequence_length": T},
        video_embeddings=None,
    )
    progress_sample = ProgressSample(trajectory=traj, sample_type="progress")
    batch = batch_collator([progress_sample])

    # Move to GPU
    progress_inputs = batch["progress_inputs"]
    for key, value in progress_inputs.items():
        if hasattr(value, "to"):
            progress_inputs[key] = value.to(device)

    # Determine config
    loss_config = getattr(exp_config, "loss", None)
    is_discrete = (
        getattr(loss_config, "progress_loss_type", "l2").lower() == "discrete"
        if loss_config else False
    )
    num_bins = (
        getattr(loss_config, "progress_discrete_bins", None)
        or getattr(exp_config.model, "progress_discrete_bins", 10)
    )

    # Run inference
    print("Running inference...")
    results = compute_batch_outputs(
        reward_model, tokenizer, progress_inputs,
        sample_type="progress", is_discrete_mode=is_discrete, num_bins=num_bins,
    )

    progress = np.array(results.get("progress_pred", [[]])[0])
    success = np.array(results.get("outputs_success", {}).get("success_probs", [[]])[0])

    print(f"Inference results (T={len(progress)}):")
    print(f"  Progress: {progress}")
    print(f"  Success:  {success}")

    # Plot
    success_binary = (success > args.success_threshold).astype(np.int32) if success.size > 0 else None
    fig = create_combined_progress_success_plot(
        progress_pred=progress,
        num_frames=T,
        success_binary=success_binary,
        success_probs=success if success.size > 0 else None,
        title=f"Robometer: {args.task}",
    )
    plot_path = Path(args.video).stem + "_plot.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    main()
