"""
preprocess_videos.py

Batch video preprocessing wrapper around ffmpeg for the SPARC-V pipeline.

Features:
- Re-encode to H.264 MP4
- Set frame rate (fps)
- Scale and pad to a fixed resolution while preserving aspect ratio
- Optional denoising and basic color adjustments
- Optional trimming
- Recursive folder processing and parallel workers

Usage:
    python preprocess_videos.py --input_dir data/raw_videos --output_dir data/videos --fps 10 --width 640 --height 480

Requires `ffmpeg` available on PATH.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def ffmpeg_exists() -> bool:
    return shutil.which("ffmpeg") is not None


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".mpg", ".mpeg"}


def build_vf_filter(width: int, height: int, denoise: bool = False, denoise_strength: float = 1.5, eq: Optional[str] = None) -> str:
    # scale to width preserving aspect, then pad to target WxH
    # use -2 to preserve even height when scaling
    scale = f"scale='min({width},iw)':'-2'"
    pad = f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"
    parts: List[str] = [scale, pad]
    if denoise:
        parts.insert(0, f"hqdn3d={denoise_strength}:{denoise_strength}:6:6")
    if eq:
        parts.append(eq)
    return ",".join(parts)


def make_output_path(inp: Path, input_root: Path, out_dir: Path, prefix: str = "") -> Path:
    """Preserve relative directory structure under output_dir."""
    try:
        rel = inp.relative_to(input_root)
    except ValueError:
        rel = Path(inp.name)

    out_name = f"{prefix}{rel.stem}.mp4"
    return out_dir / rel.parent / out_name


def process_video(
    inp: str,
    out: str,
    fps: int = 10,
    width: int = 640,
    height: int = 480,
    crf: int = 20,
    preset: str = "medium",
    denoise: bool = False,
    denoise_strength: float = 1.5,
    trim_start: Optional[float] = None,
    trim_duration: Optional[float] = None,
    overwrite: bool = True,
) -> None:
    if not ffmpeg_exists():
        raise RuntimeError("ffmpeg executable not found on PATH")

    vf = build_vf_filter(width, height, denoise=denoise, denoise_strength=denoise_strength)

    cmd = ["ffmpeg"]
    cmd += ["-y"] if overwrite else ["-n"]

    if trim_start is not None:
        cmd += ["-ss", str(trim_start)]

    cmd += ["-i", inp]

    if trim_duration is not None:
        cmd += ["-t", str(trim_duration)]

    cmd += ["-r", str(fps), "-vf", vf, "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-c:a", "aac", "-movflags", "+faststart", out]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed for {inp}: {e}")


def collect_videos(input_dir: Path, recursive: bool = True) -> List[Path]:
    vids: List[Path] = []
    if recursive:
        for p in input_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
                vids.append(p)
    else:
        for p in input_dir.iterdir():
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
                vids.append(p)
    vids.sort()
    return vids


def run_batch(
    input_dir: str,
    output_dir: str,
    fps: int = 10,
    width: int = 640,
    height: int = 480,
    crf: int = 20,
    preset: str = "medium",
    denoise: bool = False,
    denoise_strength: float = 1.5,
    trim_start: Optional[float] = None,
    trim_duration: Optional[float] = None,
    recursive: bool = True,
    workers: int = 1,
    overwrite: bool = True,
):
    inp = Path(input_dir)
    out = Path(output_dir)
    if not inp.exists():
        raise FileNotFoundError(f"input_dir {inp} does not exist")
    out.mkdir(parents=True, exist_ok=True)

    videos = collect_videos(inp, recursive=recursive)
    if not videos:
        print("No videos found to process.")
        return

    tasks = []
    for v in videos:
        out_path = make_output_path(v, inp, out)
        tasks.append((str(v), str(out_path)))

    def _worker(args):
        src, dst = args
        try:
            Path(dst).parent.mkdir(parents=True, exist_ok=True)
            process_video(
                src,
                dst,
                fps=fps,
                width=width,
                height=height,
                crf=crf,
                preset=preset,
                denoise=denoise,
                denoise_strength=denoise_strength,
                trim_start=trim_start,
                trim_duration=trim_duration,
                overwrite=overwrite,
            )
            return (src, dst, None)
        except Exception as e:
            return (src, dst, str(e))

    if workers and workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
            for src, dst, err in ex.map(_worker, tasks):
                if err:
                    print(f"[ERROR] {src} -> {dst}: {err}")
                else:
                    print(f"[OK] {src} -> {dst}")
    else:
        for args in tasks:
            src, dst, err = _worker(args)
            if err:
                print(f"[ERROR] {src} -> {dst}: {err}")
            else:
                print(f"[OK] {src} -> {dst}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch preprocess videos for SPARC-V pipeline")
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--crf", type=int, default=20)
    p.add_argument("--preset", type=str, default="medium")
    p.add_argument("--denoise", action="store_true")
    p.add_argument("--denoise_strength", type=float, default=1.5)
    p.add_argument("--trim_start", type=float, default=None)
    p.add_argument("--trim_duration", type=float, default=None)
    p.add_argument("--no_recursive", dest="recursive", action="store_false")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--no_overwrite", dest="overwrite", action="store_false")
    return p.parse_args()


def main():
    args = parse_args()
    if not ffmpeg_exists():
        print("ERROR: ffmpeg not found on PATH. Install ffmpeg before running.")
        sys.exit(1)

    run_batch(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        fps=args.fps,
        width=args.width,
        height=args.height,
        crf=args.crf,
        preset=args.preset,
        denoise=args.denoise,
        denoise_strength=args.denoise_strength,
        trim_start=args.trim_start,
        trim_duration=args.trim_duration,
        recursive=args.recursive,
        workers=args.workers,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
