#!/usr/bin/env python3
"""
Robometer visualization script.
Runs inference with the exact same pipeline as try_robometer.py,
then burns the per-frame predictions into a video using PyAV.
"""

import argparse
import cv2
import torch
import numpy as np
import traceback
import av
from pathlib import Path

from robometer.data.dataset_types import ProgressSample, Trajectory
from robometer.evals.eval_server import compute_batch_outputs
from robometer.evals.eval_viz_utils import extract_frames
from robometer.utils.save import load_model_from_hf
from robometer.utils.setup_utils import setup_batch_collator


# ---------------------------------------------------------------------------
# Inference  (identical to try_robometer.py)
# ---------------------------------------------------------------------------

def run_inference(model_path, video_path, task, fps=1.0, max_frames=512):
    """Run Robometer inference — identical pipeline to try_robometer.py."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading model {model_path}...")
    exp_config, tokenizer, processor, reward_model = load_model_from_hf(
        model_path=model_path,
        device=device,
    )
    reward_model.eval()
    batch_collator = setup_batch_collator(processor, tokenizer, exp_config, is_eval=True)

    print(f"Extracting frames from {video_path} at {fps} fps (max {max_frames} frames)...")
    frames = extract_frames(video_path, fps=fps, max_frames=max_frames)
    if frames is None or frames.size == 0:
        raise RuntimeError("Could not extract frames from video.")
    if frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 255).astype(np.uint8)

    T = int(frames.shape[0])
    print(f"Extracted {T} frames.")

    traj = Trajectory(
        frames=frames,
        frames_shape=tuple(frames.shape),
        task=task,
        id="0",
        metadata={"subsequence_length": T},
        video_embeddings=None,
    )
    progress_sample = ProgressSample(trajectory=traj, sample_type="progress")
    batch = batch_collator([progress_sample])

    progress_inputs = batch["progress_inputs"]
    for key, value in progress_inputs.items():
        if hasattr(value, "to"):
            progress_inputs[key] = value.to(device)

    loss_config = getattr(exp_config, "loss", None)
    is_discrete = (
        getattr(loss_config, "progress_loss_type", "l2").lower() == "discrete"
        if loss_config else False
    )
    num_bins = (
        getattr(loss_config, "progress_discrete_bins", None)
        or getattr(exp_config.model, "progress_discrete_bins", 10)
    )

    print("Running inference...")
    results = compute_batch_outputs(
        reward_model,
        tokenizer,
        progress_inputs,
        sample_type="progress",
        is_discrete_mode=is_discrete,
        num_bins=num_bins,
    )

    progress_pred = results.get("progress_pred", [[]])[0]
    outputs_success = results.get("outputs_success", {})
    success_probs = outputs_success.get("success_probs", [[]])[0] if outputs_success else []

    return np.array(progress_pred), np.array(success_probs)


# ---------------------------------------------------------------------------
# Rendering via PyAV (avoids OpenCV CUDA conflict)
# ---------------------------------------------------------------------------

def draw_ui_on_frame(frame_rgb, h, w, ui_h, progress, success, max_progress, max_success, task):
    """Overlay progress/success bars on a frame canvas."""
    canvas = np.zeros((h + ui_h, w, 3), dtype=np.uint8)
    canvas[:h, :w] = frame_rgb[:, :, ::-1]  # RGB → BGR for OpenCV drawing

    bar_w = min(w - 40, 500)

    # Progress bar (show current + max so far)
    label_progress = f"Progress: {progress:.2f}  (peak: {max_progress:.2f})"
    cv2.putText(canvas, label_progress, (20, h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.rectangle(canvas, (20, h + 25), (20 + bar_w, h + 42), (80, 80, 80), -1)
    cv2.rectangle(canvas, (20, h + 25), (20 + int(bar_w * np.clip(progress, 0, 1)), h + 42),
                  (0, 200, 0), -1)

    # Success bar (show current + max so far)
    color = (0, 0, 220) if success < 0.5 else (0, 200, 0)
    label_success = f"Success: {success:.2f}  (peak: {max_success:.2f})"
    cv2.putText(canvas, label_success, (20, h + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.rectangle(canvas, (20, h + 63), (20 + bar_w, h + 80), (80, 80, 80), -1)
    cv2.rectangle(canvas, (20, h + 63), (20 + int(bar_w * np.clip(success, 0, 1)), h + 80), color, -1)

    # Task label
    cv2.putText(canvas, f"Task: {task}", (20, h + 97), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    return canvas[:, :, ::-1]  # Return as RGB


def render_video(video_path, task, progress_arr, success_arr, output_path, playback_fps=None):
    """Read video with PyAV and burn progress/success bars using OpenCV drawing."""
    input_container = av.open(video_path)
    video_stream = input_container.streams.video[0]

    native_fps = float(video_stream.average_rate)
    total_frames = video_stream.frames
    display_fps = playback_fps or native_fps
    ui_h = 100

    print(f"Rendering {total_frames} frames at {display_fps:.1f} fps → {output_path}")

    # Get dimensions from first frame
    first_frame = None
    for packet in input_container.demux(video_stream):
        for frm in packet.decode():
            first_frame = frm.to_ndarray(format="rgb24")
            break
        if first_frame is not None:
            break
    input_container.seek(0)  # rewind

    h, w = first_frame.shape[:2]
    out_h = h + ui_h

    output_container = av.open(output_path, mode="w")
    from fractions import Fraction
    out_stream = output_container.add_stream("mpeg4", rate=Fraction(display_fps).limit_denominator(10000))
    out_stream.width = w
    out_stream.height = out_h
    out_stream.pix_fmt = "yuv420p"
    out_stream.options = {"qscale": "4"}

    frame_idx = 0
    n_pred = len(progress_arr)
    max_progress_so_far = 0.0
    max_success_so_far = 0.0

    for packet in input_container.demux(video_stream):
        for frm in packet.decode():
            raw_rgb = frm.to_ndarray(format="rgb24")

            # Map video frame index → 1fps prediction index
            pred_idx = int(frame_idx * n_pred / max(total_frames, 1))
            pred_idx = min(pred_idx, n_pred - 1)
            progress = float(progress_arr[pred_idx]) if n_pred > 0 else 0.0
            success = float(success_arr[pred_idx]) if len(success_arr) > 0 else 0.0

            max_progress_so_far = max(max_progress_so_far, progress)
            max_success_so_far = max(max_success_so_far, success)

            rendered_rgb = draw_ui_on_frame(raw_rgb, h, w, ui_h, progress, success,
                                           max_progress_so_far, max_success_so_far, task)

            av_frame = av.VideoFrame.from_ndarray(rendered_rgb, format="rgb24")
            av_frame.pts = frm.pts
            av_frame.time_base = frm.time_base
            for pkt in out_stream.encode(av_frame):
                output_container.mux(pkt)

            frame_idx += 1

    for pkt in out_stream.encode(None):
        output_container.mux(pkt)

    input_container.close()
    output_container.close()
    print(f"Output saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Robometer visualization")
    parser.add_argument("--model-path", default="robometer/Robometer-4B")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--task", required=True, help="Task description")
    parser.add_argument("--fps", type=float, default=1.0,
                        help="Frame extraction rate for inference (default: 1.0)")
    parser.add_argument("--max-frames", type=int, default=512,
                        help="Max frames for inference (default: 512)")
    parser.add_argument("--output", default="realtime_output.mp4", help="Output video path")
    parser.add_argument("--playback-fps", type=float, default=None,
                        help="Playback fps override (default: native video fps)")
    args = parser.parse_args()

    try:
        progress_arr, success_arr = run_inference(
            model_path=args.model_path,
            video_path=args.video,
            task=args.task,
            fps=args.fps,
            max_frames=args.max_frames,
        )
        print(f"Inference complete. Progress shape: {progress_arr.shape}, Success shape: {success_arr.shape}")
        print(f"\nPer-frame progress: {[f'{v:.3f}' for v in progress_arr]}")
        print(f"Per-frame success:  {[f'{v:.3f}' for v in success_arr]}")
        print(f"\nMax progress: {progress_arr.max():.3f}, Max success: {success_arr.max():.3f}")
        print(f"Final frame  → progress: {progress_arr[-1]:.3f}, success: {success_arr[-1]:.3f}")

        render_video(
            video_path=args.video,
            task=args.task,
            progress_arr=progress_arr,
            success_arr=success_arr,
            output_path=args.output,
            playback_fps=args.playback_fps,
        )
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
