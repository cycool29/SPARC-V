"""
extract_silhouettes.py — Step 1 of SCN pipeline (Windows-compatible).

Uses YOLOv8 instance segmentation (ultralytics) to extract human silhouette
boundary curves from video frames — a Windows-friendly replacement for the
original Mask R-CNN-based extraction stage.

Install:
    pip install ultralytics opencv-python numpy

YOLOv8 automatically downloads the model weights on first run.

Usage:
    python extract_silhouettes.py ^
        --video_dir  data/videos ^
        --output_dir data/silhouettes ^
        --device     cuda

Output structure:
    silhouettes/
      <class>/
        <video_name>/
                    frame_0000.npy   # (128, 2) uniform boundary sample  (x, y) pixels
          frame_0001.npy
          ...
          meta.json        # {"n_frames": T, "label": int, "class": str}
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Union

import cv2
import numpy as np

from utils import resample_contour_uniform


# ---------------------------------------------------------------------------
# YOLOv8 Segmentation extractor  (primary — Windows compatible)
# ---------------------------------------------------------------------------

class YOLOv8SilhouetteExtractor:
    """
    Extracts human silhouette boundary curves using YOLOv8-seg.

    Model options (downloaded automatically on first run):
        yolov8n-seg.pt  — nano,  fastest,  ~6 MB
        yolov8s-seg.pt  — small, balanced, ~22 MB  <- default
        yolov8m-seg.pt  — medium, accurate, ~52 MB
        yolov8x-seg.pt  — xlarge, best accuracy

    COCO class 0 = person.
    """

    PERSON_CLASS_ID = 0

    def __init__(
        self,
        model_name: str = "yolov8s-seg.pt",
        confidence: float = 0.5,
        device: str = "cpu",
        boundary_points: int = 128,
    ):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics not installed.\n"
                "Run:  pip install ultralytics"
            )

        print(f"Loading YOLOv8 model: {model_name}  (device={device})")
        self.model  = YOLO(model_name)
        self.conf   = confidence
        self.device = device
        self.boundary_points = boundary_points
        print("Model ready.")

    def extract_boundary(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Run YOLOv8 segmentation on a single BGR frame.

        Returns:
            boundary_pts: (N, 2) float32 array of (x, y) silhouette boundary
                          points, or empty (0, 2) array if no person detected.
        """
        results = self.model(
            frame_bgr,
            classes=[self.PERSON_CLASS_ID],
            conf=self.conf,
            device=self.device,
            verbose=False,
        )

        result = results[0]
        if result.masks is None or len(result.masks) == 0:
            return np.empty((0, 2), dtype=np.float32)

        H, W = frame_bgr.shape[:2]

        # Merge all person masks into one binary mask
        masks_data = result.masks.data.cpu().numpy()  # (K, H', W')
        combined   = masks_data.any(axis=0).astype(np.uint8)  # (H', W')

        # Resize to original frame size if needed
        if combined.shape != (H, W):
            combined = cv2.resize(combined, (W, H), interpolation=cv2.INTER_NEAREST)
        combined = (combined * 255).astype(np.uint8)

        # Extract contour boundary points
        contours, _ = cv2.findContours(
            combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if not contours:
            return np.empty((0, 2), dtype=np.float32)

        pts = np.concatenate(
            [c.reshape(-1, 2) for c in contours], axis=0
        ).astype(np.float32)
        return resample_contour_uniform(pts, self.boundary_points)


# ---------------------------------------------------------------------------
# OpenCV MOG2 fallback (zero dependencies beyond opencv-python)
# ---------------------------------------------------------------------------

class MOG2SilhouetteExtractor:
    """
    Background-subtraction silhouette extractor using OpenCV MOG2.
    Works on Windows with just opencv-python. Less accurate than YOLO
    but requires no model download and runs very fast.
    Best for static-camera datasets (HMDB, UCF101 often qualify).
    """

    def __init__(self, history: int = 200, var_threshold: int = 40, boundary_points: int = 128):
        self.bg   = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=False
        )
        self.kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        self.boundary_points = boundary_points

    def extract_boundary(self, frame_bgr: np.ndarray) -> np.ndarray:
        fg = self.bg.apply(frame_bgr)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  self.kernel_open)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self.kernel_close)

        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return np.empty((0, 2), dtype=np.float32)

        # Keep the largest contour (most likely the human subject)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]
        pts = np.concatenate(
            [c.reshape(-1, 2) for c in contours], axis=0
        ).astype(np.float32)
        return resample_contour_uniform(pts, self.boundary_points)


# ---------------------------------------------------------------------------
# Core: process a single video
# ---------------------------------------------------------------------------

def extract_video_silhouettes(
    video_path: str,
    output_dir: str,
    extractor,
    max_frames: int = 250,
) -> int:
    """
    Extract silhouette boundary points for every frame of a video.

    Paper Section 4.2: "we evenly extract around 250 frames" per video clip.

    Args:
        video_path:  path to .avi / .mp4 file
        output_dir:  directory to write per-frame .npy files
        extractor:   YOLOv8SilhouetteExtractor or MOG2SilhouetteExtractor
        max_frames:  cap on frames extracted (paper: ~250)

    Returns:
        n_saved: number of frames with non-empty silhouettes
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [WARN] Cannot open: {video_path}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = max(total_frames, 1)

    # Evenly-spaced frame indices (paper: "evenly extract ~250 frames")
    sample_count  = min(max_frames, total_frames)
    frame_indices = set(
        np.linspace(0, total_frames - 1, sample_count, dtype=int).tolist()
    )

    n_saved   = 0
    frame_idx = 0
    t0        = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in frame_indices:
            boundary = extractor.extract_boundary(frame)
            if boundary.shape[0] > 0:
                out_path = os.path.join(output_dir, f"frame_{n_saved:04d}.npy")
                np.save(out_path, boundary)
                n_saved += 1

        frame_idx += 1

    cap.release()
    elapsed = time.time() - t0
    print(f"    -> {n_saved} frames in {elapsed:.1f}s  [{output_dir}]")
    return n_saved


# ---------------------------------------------------------------------------
# Dataset-level processing
# ---------------------------------------------------------------------------

def process_dataset(
    video_dir: Union[str, Path],
    output_dir: Union[str, Path],
    extractor,
    max_frames: int = 250,
):
    """
    Walk video_dir (<class>/<video>.avi) and extract silhouettes for all videos.

    Expected input structure:
        video_dir/
          action_class_1/
            video1.avi
            video2.avi
          action_class_2/
            ...
    """
    video_dir  = Path(video_dir)
    output_dir = Path(output_dir)
    video_exts = {".avi", ".mp4", ".mov", ".mkv", ".wmv"}

    root_dirs = sorted([d for d in video_dir.iterdir() if d.is_dir()])
    video_exts = {".avi", ".mp4", ".mov", ".mkv", ".wmv"}

    # Supports either:
    # 1) video_dir/<class>/<video>
    # 2) video_dir/<split>/<class>/<video>
    split_layout = any(
        any(grandchild.is_dir() for grandchild in child.iterdir()) and
        not any(f.is_file() and f.suffix.lower() in video_exts for f in child.iterdir())
        for child in root_dirs
    )

    groups = []
    if split_layout:
        for split_dir in root_dirs:
            for class_dir in sorted([d for d in split_dir.iterdir() if d.is_dir()]):
                groups.append((split_dir.name, class_dir))
    else:
        for class_dir in root_dirs:
            groups.append((None, class_dir))

    class_names = sorted({class_dir.name for _, class_dir in groups})
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    n_classes = len(class_names)

    print(f"\nFound {n_classes} action classes in: {video_dir}")
    print(f"Output -> {output_dir}\n")

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "class_map.json", "w") as f:
        json.dump(class_to_idx, f, indent=2)

    total_videos = 0

    for split_name, class_dir in groups:
        label  = class_to_idx[class_dir.name]
        videos = sorted([
            f for f in class_dir.iterdir()
            if f.suffix.lower() in video_exts
        ])

        prefix = f"{split_name}/" if split_name else ""
        print(f"[{label+1:03d}/{n_classes}]  {prefix}{class_dir.name}  ({len(videos)} videos)")

        for video_path in videos:
            if split_name:
                rel_out = output_dir / split_name / class_dir.name / video_path.stem
            else:
                rel_out = output_dir / class_dir.name / video_path.stem

            if rel_out.exists() and any(rel_out.glob("frame_*.npy")):
                print(f"  [skip] {video_path.name}")
                continue

            print(f"  {video_path.name}")
            n_frames = extract_video_silhouettes(
                str(video_path), str(rel_out), extractor, max_frames=max_frames,
            )

            meta = {
                "n_frames": n_frames,
                "label":    label,
                "class":    class_dir.name,
                "video":    video_path.name,
            }
            rel_out.mkdir(parents=True, exist_ok=True)
            with open(rel_out / "meta.json", "w") as f:
                json.dump(meta, f, indent=2)

            total_videos += 1

    print(f"\nDone. Processed {total_videos} videos.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Extract silhouette boundary curves (Windows-compatible, YOLOv8)."
    )
    p.add_argument("--video_dir",   required=True,
                   help="Root dir: video_dir/<class>/<video.avi>")
    p.add_argument("--output_dir",  required=True,
                   help="Where to save .npy silhouette files")
    p.add_argument("--model",       default="yolov8s-seg.pt",
                   choices=["yolov8n-seg.pt", "yolov8s-seg.pt",
                            "yolov8m-seg.pt", "yolov8x-seg.pt"],
                   help="YOLOv8 model size (n=fastest, x=most accurate)")
    p.add_argument("--confidence",  type=float, default=0.5)
    p.add_argument("--device",      default="cpu",
                   help="cuda or cpu")
    p.add_argument("--max_frames",  type=int, default=250)
    p.add_argument("--boundary_points", type=int, default=128,
                   help="Uniform samples per silhouette boundary")
    p.add_argument("--fallback_mog2", action="store_true",
                   help="Use OpenCV MOG2 instead of YOLOv8 (no download needed)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.fallback_mog2:
        print("Using OpenCV MOG2 background subtraction.")
        extractor = MOG2SilhouetteExtractor(boundary_points=args.boundary_points)
    else:
        extractor = YOLOv8SilhouetteExtractor(
            model_name=args.model,
            confidence=args.confidence,
            device=args.device,
            boundary_points=args.boundary_points,
        )

    process_dataset(
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        extractor=extractor,
        max_frames=args.max_frames,
    )