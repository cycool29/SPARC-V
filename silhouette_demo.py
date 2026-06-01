"""
silhouette_demo.py — standalone silhouette visualization and smoke test.

This script is intentionally separate from the production extractor.
It previews YOLOv8 or MOG2 silhouette boundaries on a video and can
optionally save an annotated MP4 for quick verification.

Usage:
    python silhouette_demo.py --video data/videos/train/Fight/sample.mp4 --device cuda --show
    python silhouette_demo.py --video data/videos/train/Fight/sample.mp4 --device cuda --save_preview --preview_dir data/previews

Press 'q' to quit the live preview window.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np

from extract_silhouettes import MOG2SilhouetteExtractor, YOLOv8SilhouetteExtractor


def _draw_boundary(frame: np.ndarray, boundary: np.ndarray) -> np.ndarray:
    vis = frame.copy()
    if boundary.shape[0] > 0:
        pts = boundary.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    return vis


def preview_video(
    video_path: str,
    extractor,
    max_frames: int = 250,
    show: bool = True,
    save_preview: bool = False,
    preview_dir: str | None = None,
    output_fps: float = 10.0,
) -> None:
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    sample_count = min(max_frames, total_frames)
    frame_indices = set(np.linspace(0, total_frames - 1, sample_count, dtype=int).tolist())

    writer = None
    if save_preview:
        if preview_dir is None:
            preview_dir = str(Path(video_path).parent)
        os.makedirs(preview_dir, exist_ok=True)

    frame_idx = 0
    saved_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx in frame_indices:
            boundary = extractor.extract_boundary(frame)
            vis = _draw_boundary(frame, boundary)
            cv2.putText(
                vis,
                f"frame={frame_idx} saved={saved_idx} pts={boundary.shape[0]}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if save_preview:
                if writer is None:
                    h, w = vis.shape[:2]
                    preview_path = os.path.join(preview_dir, Path(video_path).stem + "_preview.mp4")
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(preview_path, fourcc, output_fps, (w, h))
                writer.write(vis)

            if show:
                cv2.imshow("Silhouette Demo", vis)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

            saved_idx += 1

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    if show:
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview extracted silhouettes on a single video.")
    parser.add_argument("--video", required=True, help="Path to a single video file")
    parser.add_argument("--model", default="yolov8s-seg.pt",
                        choices=["yolov8n-seg.pt", "yolov8s-seg.pt", "yolov8m-seg.pt", "yolov8x-seg.pt"])
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--boundary_points", type=int, default=128)
    parser.add_argument("--max_frames", type=int, default=250)
    parser.add_argument("--fallback_mog2", action="store_true", help="Use OpenCV MOG2 instead of YOLOv8")
    parser.add_argument("--show", action="store_true", help="Show live preview window")
    parser.add_argument("--save_preview", action="store_true", help="Save annotated preview MP4")
    parser.add_argument("--preview_dir", default=None, help="Where to save annotated previews")
    parser.add_argument("--output_fps", type=float, default=10.0, help="FPS for saved preview video")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.fallback_mog2:
        extractor = MOG2SilhouetteExtractor(boundary_points=args.boundary_points)
    else:
        extractor = YOLOv8SilhouetteExtractor(
            model_name=args.model,
            confidence=args.confidence,
            device=args.device,
            boundary_points=args.boundary_points,
        )

    preview_video(
        video_path=args.video,
        extractor=extractor,
        max_frames=args.max_frames,
        show=args.show,
        save_preview=args.save_preview,
        preview_dir=args.preview_dir,
        output_fps=args.output_fps,
    )
