#!/usr/bin/env python3
"""
Reconstruction script for Robometer.
Runs both static inference (plot) and video rendering for a set of videos.
Produces:
  - outputs/<name>_progress.npy  (per-frame progress values)
  - outputs/<name>_success.npy   (per-frame success probabilities)
  - outputs/<name>_plot.png      (static matplotlib figure)
  - outputs/<name>_overlay.mp4   (video w/ progress/success bars burned in)
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from robometer.data.dataset_types import ProgressSample, Trajectory
from robometer.evals.eval_server import compute_batch_outputs
from robometer.evals.eval_viz_utils import create_combined_progress_success_plot, extract_frames
from robometer.utils.save import load_model_from_hf
from robometer.utils.setup_utils import setup_batch_collator


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_VIDEOS = {
    "video/peg.mp4": "insert the peg into the socket",
}


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def load_model(model_path, device):
    """Load model once, return everything needed for inference."""
    print(f"\n{'='*60}")
    print(f"Loading model: {model_path}")
    print(f"Device: {device}")
    print(f"{'='*60}")

    exp_config, tokenizer, processor, reward_model = load_model_from_hf(
        model_path=model_path, device=device,
    )
    reward_model.eval()
    batch_collator = setup_batch_collator(processor, tokenizer, exp_config, is_eval=True)

    loss_config = getattr(exp_config, "loss", None)
    is_discrete = (
        getattr(loss_config, "progress_loss_type", "l2").lower() == "discrete"
        if loss_config else False
    )
    num_bins = (
        getattr(loss_config, "progress_discrete_bins", None)
        or getattr(exp_config.model, "progress_discrete_bins", 10)
    )

    return {
        "reward_model": reward_model,
        "tokenizer": tokenizer,
        "processor": processor,
        "batch_collator": batch_collator,
        "is_discrete": is_discrete,
        "num_bins": num_bins,
        "device": device,
    }


def run_inference(ctx, video_path, task, fps=1.0, max_frames=512):
    """Run inference on a video, identical to try_robometer.py."""
    print(f"\n--- Extracting frames from {video_path} at {fps} fps (max {max_frames}) ---")
    frames = extract_frames(str(video_path), fps=fps, max_frames=max_frames)
    if frames is None or frames.size == 0:
        raise RuntimeError(f"Could not extract frames from {video_path}")
    if frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 255).astype(np.uint8)

    T = int(frames.shape[0])
    print(f"Extracted {T} frames, shape: {frames.shape}")

    traj = Trajectory(
        frames=frames,
        frames_shape=tuple(frames.shape),
        task=task,
        id="0",
        metadata={"subsequence_length": T},
        video_embeddings=None,
    )
    progress_sample = ProgressSample(trajectory=traj, sample_type="progress")
    batch = ctx["batch_collator"]([progress_sample])

    progress_inputs = batch["progress_inputs"]
    device = ctx["device"]
    for key, value in progress_inputs.items():
        if hasattr(value, "to"):
            progress_inputs[key] = value.to(device)

    print("Running inference...")
    t0 = time.time()
    results = compute_batch_outputs(
        ctx["reward_model"],
        ctx["tokenizer"],
        progress_inputs,
        sample_type="progress",
        is_discrete_mode=ctx["is_discrete"],
        num_bins=ctx["num_bins"],
    )
    dt = time.time() - t0
    print(f"Inference done in {dt:.2f}s")

    progress_pred = results.get("progress_pred", [[]])[0]
    outputs_success = results.get("outputs_success", {})
    success_probs = outputs_success.get("success_probs", [[]])[0] if outputs_success else []

    return np.array(progress_pred), np.array(success_probs), T


def save_plot(progress, success, num_frames, task, save_path):
    """Generate and save the static matplotlib figure."""
    success_binary = (success > 0.5).astype(np.int32) if success.size > 0 else None
    fig = create_combined_progress_success_plot(
        progress_pred=progress,
        num_frames=num_frames,
        success_binary=success_binary,
        success_probs=success if success.size > 0 else None,
        title=f"Robometer: {task}",
    )
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {save_path}")


def render_overlay_video(video_path, progress_arr, success_arr, output_path, task):
    """Burn progress/success bars onto the video using PyAV + OpenCV drawing."""
    import av
    import cv2
    from fractions import Fraction

    input_container = av.open(str(video_path))
    video_stream = input_container.streams.video[0]
    native_fps = float(video_stream.average_rate)
    total_frames = video_stream.frames
    ui_h = 100

    # Get dimensions from first frame
    first_frame = None
    for packet in input_container.demux(video_stream):
        for frm in packet.decode():
            first_frame = frm.to_ndarray(format="rgb24")
            break
        if first_frame is not None:
            break
    input_container.seek(0)

    h, w = first_frame.shape[:2]

    output_container = av.open(str(output_path), mode="w")
    out_stream = output_container.add_stream(
        "mpeg4", rate=Fraction(native_fps).limit_denominator(10000)
    )
    out_stream.width = w
    out_stream.height = h + ui_h
    out_stream.pix_fmt = "yuv420p"
    out_stream.options = {"qscale": "4"}

    frame_idx = 0
    n_pred = len(progress_arr)
    max_prog = 0.0
    max_succ = 0.0

    print(f"Rendering {total_frames} frames → {output_path}")

    for packet in input_container.demux(video_stream):
        for frm in packet.decode():
            raw_rgb = frm.to_ndarray(format="rgb24")

            pred_idx = min(int(frame_idx * n_pred / max(total_frames, 1)), n_pred - 1)
            progress = float(progress_arr[pred_idx]) if n_pred > 0 else 0.0
            success = float(success_arr[pred_idx]) if len(success_arr) > 0 else 0.0
            max_prog = max(max_prog, progress)
            max_succ = max(max_succ, success)

            # Draw on canvas
            canvas = np.zeros((h + ui_h, w, 3), dtype=np.uint8)
            canvas[:h, :w] = raw_rgb[:, :, ::-1]  # RGB → BGR

            bar_w = min(w - 40, 500)

            # Progress bar
            cv2.putText(canvas, f"Progress: {progress:.2f}  (peak: {max_prog:.2f})",
                        (20, h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.rectangle(canvas, (20, h + 25), (20 + bar_w, h + 42), (80, 80, 80), -1)
            cv2.rectangle(canvas, (20, h + 25),
                          (20 + int(bar_w * np.clip(progress, 0, 1)), h + 42),
                          (0, 200, 0), -1)

            # Success bar
            s_color = (0, 0, 220) if success < 0.5 else (0, 200, 0)
            cv2.putText(canvas, f"Success: {success:.2f}  (peak: {max_succ:.2f})",
                        (20, h + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.rectangle(canvas, (20, h + 63), (20 + bar_w, h + 80), (80, 80, 80), -1)
            cv2.rectangle(canvas, (20, h + 63),
                          (20 + int(bar_w * np.clip(success, 0, 1)), h + 80),
                          s_color, -1)

            # Task
            cv2.putText(canvas, f"Task: {task}",
                        (20, h + 97), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

            out_rgb = canvas[:, :, ::-1]  # back to RGB for PyAV
            av_frame = av.VideoFrame.from_ndarray(out_rgb, format="rgb24")
            av_frame.pts = frm.pts
            av_frame.time_base = frm.time_base
            for pkt in out_stream.encode(av_frame):
                output_container.mux(pkt)

            frame_idx += 1

    for pkt in out_stream.encode(None):
        output_container.mux(pkt)

    input_container.close()
    output_container.close()
    print(f"Overlay video saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct all Robometer outputs: static plots + overlay videos"
    )
    parser.add_argument("--model-path", default="robometer/Robometer-4B")
    parser.add_argument("--videos", nargs="*", default=None,
                        help="Video paths (space-separated). If omitted, uses defaults.")
    parser.add_argument("--tasks", nargs="*", default=None,
                        help="Corresponding task descriptions (must match --videos count)")
    parser.add_argument("--out-dir", default="outputs",
                        help="Output directory (default: outputs)")
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=512)
    args = parser.parse_args()

    # Build video→task mapping
    if args.videos:
        if args.tasks and len(args.tasks) == len(args.videos):
            video_task_map = dict(zip(args.videos, args.tasks))
        elif args.tasks:
            parser.error("--tasks must have the same number of entries as --videos")
        else:
            video_task_map = {v: "unknown task" for v in args.videos} # Fallback
    else:
        video_task_map = DEFAULT_VIDEOS

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model once
    ctx = load_model(args.model_path, device)

    # Process each video
    for video_path_str, task in video_task_map.items():
        video_path = Path(video_path_str)
        if not video_path.exists():
            print(f"\n⚠ Skipping {video_path} — file not found")
            continue

        name = video_path.stem
        print(f"\n{'='*60}")
        print(f"Processing: {video_path}  ({task})")
        print(f"{'='*60}")

        # 1. Run inference
        progress, success, num_frames = run_inference(
            ctx, video_path, task, fps=args.fps, max_frames=args.max_frames,
        )

        # Print per-frame values
        print(f"\nPer-frame progress ({len(progress)} frames):")
        print("  " + "  ".join(f"{v:.3f}" for v in progress))
        print(f"Per-frame success ({len(success)} frames):")
        print("  " + "  ".join(f"{v:.3f}" for v in success))
        print(f"\nMax progress: {progress.max():.3f}   Max success: {success.max():.3f}")
        print(f"Last frame → progress: {progress[-1]:.3f}   success: {success[-1]:.3f}")

        # 2. Save numpy arrays
        np.save(out_dir / f"{name}_progress.npy", progress)
        np.save(out_dir / f"{name}_success.npy", success)
        print(f"Saved {name}_progress.npy and {name}_success.npy")

        # 3. Generate static plot
        plot_path = out_dir / f"{name}_plot.png"
        save_plot(progress, success, num_frames, task, plot_path)

        # 4. Render overlay video
        overlay_path = out_dir / f"{name}_overlay.mp4"
        render_overlay_video(video_path, progress, success, overlay_path, task)

    print(f"\n{'='*60}")
    print(f"All done! Outputs in: {out_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
