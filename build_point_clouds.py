"""
build_point_clouds.py — Step 2 of SCN pipeline.

Converts per-frame silhouette boundary .npy files into a single
3D stacked point cloud per video, then resamples to a fixed N=4096 points
using Farthest Point Sampling.

Paper Section 3.1:
  "The silhouette curves are stacked to form a 3D curve volume along the
   time axis and resampled to a 3D point cloud with a fixed number of points.
   The coordinate of each point is denoted as (x, y, z), where (x, y)
   represents the point position in the corresponding frame, and z represents
   the time step of the frame."

Usage:
    python build_point_clouds.py \
        --silhouette_dir data/HMDB/silhouettes \
        --output_dir     data/HMDB/point_clouds \
        --n_points       4096
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from utils import build_point_cloud_from_frames, resample_contour_uniform


# ---------------------------------------------------------------------------
# Core: stack silhouette frames → 3D point cloud
# ---------------------------------------------------------------------------

def stack_silhouettes_to_point_cloud(
    frame_files: list,
    n_points: int = 4096,
    boundary_points: int = 128,
    temporal_window: int = 60,
) -> tuple:
    """
    Stack 2D silhouette boundary points from T frames into a 3D point cloud
    and resample to n_points via FPS.

    The 3D coordinate is (x, y, z) where:
      - (x, y) = 2D silhouette boundary position (normalised)
      - z      = normalised time step in [0, 1]

    As per Section 3.1, stacking is aligned by centres of gravity.

    Args:
        frame_files: sorted list of paths to per-frame .npy boundary arrays
        n_points:    target number of points after FPS resampling

    Returns:
        point_cloud:      (n_points, 3) float32 array
        density_inv:      (n_points,)   float32 density coefficients
    """
    if temporal_window > 0 and len(frame_files) > temporal_window:
        start = max(0, (len(frame_files) - temporal_window) // 2)
        frame_files = frame_files[start:start + temporal_window]

    frame_boundaries = []

    for t, fpath in enumerate(frame_files):
        pts_2d = np.load(fpath)   # (N_t, 2)  — (x, y) boundary pixels
        if pts_2d.shape[0] == 0:
            frame_boundaries.append(np.empty((0, 2), dtype=np.float32))
            continue

        frame_boundaries.append(resample_contour_uniform(pts_2d, boundary_points))

    return build_point_cloud_from_frames(frame_boundaries, n_points=n_points)


# ---------------------------------------------------------------------------
# Build Slow / Faster / Fastest temporal sub-samples for Slow-to-Fast SCN
# Paper Section 3.2.3
# ---------------------------------------------------------------------------

def build_slow_to_fast_samples(
    frame_files: list,
    n_points: int = 4096,
    boundary_points: int = 128,
    temporal_window: int = 60,
) -> dict:
    """
    Build three temporal-scale point clouds for the Slow-to-Fast architecture.

    Returns:
        dict with keys 'slow', 'faster', 'fastest', each (n_points, 3)
    """
    def _build(files):
        return stack_silhouettes_to_point_cloud(
            files,
            n_points,
            boundary_points=boundary_points,
            temporal_window=temporal_window,
        )

    # Slow:    s0, s1, s2, ...  (all frames)
    # Faster:  s0, s2, s4, ...  (every 2nd)
    # Fastest: s0, s3, s6, ...  (every 3rd)
    slow_files    = frame_files
    faster_files  = frame_files[::2]
    fastest_files = frame_files[::3]

    # Guarantee at least a few frames
    if len(faster_files)  < 4: faster_files  = frame_files
    if len(fastest_files) < 4: fastest_files = frame_files

    pc_slow,    d_slow    = _build(slow_files)
    pc_faster,  d_faster  = _build(faster_files)
    pc_fastest, d_fastest = _build(fastest_files)

    return {
        "slow":    (pc_slow,    d_slow),
        "faster":  (pc_faster,  d_faster),
        "fastest": (pc_fastest, d_fastest),
    }


# ---------------------------------------------------------------------------
# Dataset-level processing
# ---------------------------------------------------------------------------

def process_dataset(
    silhouette_dir: str,
    output_dir: str,
    n_points: int = 4096,
    boundary_points: int = 128,
    temporal_window: int = 60,
    slow_to_fast: bool = True,
):
    """
    Walk silhouette_dir (<class>/<video>/frame_XXXX.npy) and produce a
    single .npz file per video containing the stacked point cloud(s).

    Output .npz keys:
        If slow_to_fast=True:
            'slow_pts', 'slow_density',
            'faster_pts', 'faster_density',
            'fastest_pts', 'fastest_density',
            'label', 'class_name'
        Else:
            'points', 'density', 'label', 'class_name'
    """
    sil_dir = Path(silhouette_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    class_dirs = sorted([d for d in sil_dir.iterdir() if d.is_dir()])
    class_to_idx = {c.name: i for i, c in enumerate(class_dirs)}

    # Save class mapping
    with open(out_dir / "class_map.json", "w") as f:
        json.dump(class_to_idx, f, indent=2)

    print(f"Building point clouds for {len(class_dirs)} classes → {out_dir}")

    total = 0
    for class_dir in class_dirs:
        label = class_to_idx[class_dir.name]
        video_dirs = sorted([d for d in class_dir.iterdir() if d.is_dir()])

        for video_dir in video_dirs:
            out_path = out_dir / class_dir.name / f"{video_dir.name}.npz"
            if out_path.exists():
                print(f"  [SKIP] {out_path.name}")
                continue

            out_path.parent.mkdir(parents=True, exist_ok=True)

            # Collect sorted frame files
            frame_files = sorted(video_dir.glob("frame_*.npy"))
            if not frame_files:
                print(f"  [WARN] No frames in {video_dir}")
                continue

            frame_files = [str(f) for f in frame_files]
            print(f"  [{class_dir.name}] {video_dir.name}: {len(frame_files)} frames → ", end="")

            if slow_to_fast:
                samples = build_slow_to_fast_samples(
                    frame_files,
                    n_points,
                    boundary_points=boundary_points,
                    temporal_window=temporal_window,
                )
                np.savez_compressed(
                    str(out_path),
                    slow_pts       = samples["slow"][0],
                    slow_density   = samples["slow"][1],
                    faster_pts     = samples["faster"][0],
                    faster_density = samples["faster"][1],
                    fastest_pts    = samples["fastest"][0],
                    fastest_density= samples["fastest"][1],
                    label          = np.array(label, dtype=np.int64),
                    class_name     = np.array(class_dir.name),
                )
            else:
                pts, density = stack_silhouettes_to_point_cloud(
                    frame_files,
                    n_points,
                    boundary_points=boundary_points,
                    temporal_window=temporal_window,
                )
                np.savez_compressed(
                    str(out_path),
                    points     = pts,
                    density    = density,
                    label      = np.array(label, dtype=np.int64),
                    class_name = np.array(class_dir.name),
                )

            print(f"saved {out_path.name}")
            total += 1

    print(f"\nDone. Built {total} point cloud files.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Build 3D stacked silhouette point clouds.")
    parser.add_argument("--silhouette_dir", required=True,
                        help="Root dir of extracted silhouettes")
    parser.add_argument("--output_dir",     required=True,
                        help="Where to save .npz point cloud files")
    parser.add_argument("--n_points",  type=int, default=4096,
                        help="Number of points after FPS (paper default: 4096)")
    parser.add_argument("--boundary_points", type=int, default=128,
                        help="Uniform points sampled from each silhouette boundary")
    parser.add_argument("--temporal_window", type=int, default=60,
                        help="Number of consecutive frames to stack per video")
    parser.add_argument("--no_slow_to_fast", action="store_true",
                        help="Disable Slow-to-Fast multi-scale sampling")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_dataset(
        silhouette_dir=args.silhouette_dir,
        output_dir=args.output_dir,
        n_points=args.n_points,
        boundary_points=args.boundary_points,
        temporal_window=args.temporal_window,
        slow_to_fast=not args.no_slow_to_fast,
    )
