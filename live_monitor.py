"""
live_monitor.py — local RTSP/LAN violence monitoring runtime.

This executable script wires the requested online pipeline:
  1) ingest an RTSP stream or local camera/video source
  2) detect and track people with YOLOv8n-seg + ByteTrack
  3) identify persistent interacting pairs by centroid proximity
  4) extract and uniformly resample pair silhouettes to 128 boundary points
  5) buffer 60 frames into a 3D spatio-temporal point cloud
  6) classify with the Slow-to-Fast SCN
  7) smooth probabilities across 4 windows and emit a local alert

The runtime is intentionally offline after model weights are installed.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from model import build_scn
from utils import build_point_cloud_from_frames, resample_contour_uniform


PairKey = Tuple[int, int]


@dataclass
class TrackObservation:
    track_id: int
    centroid: np.ndarray
    mask: np.ndarray


def _safe_track_id(value) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        value = value[0]
    if hasattr(value, "item"):
        value = value.item()
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _pair_key(a: int, b: int) -> PairKey:
    return (a, b) if a <= b else (b, a)


def _largest_contour(mask: np.ndarray) -> np.ndarray:
    mask_u8 = (mask.astype(np.uint8) * 255) if mask.dtype != np.uint8 else mask
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.empty((0, 2), dtype=np.float32)
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
    return contour


def _softmax_probability(logits: torch.Tensor, class_index: int) -> float:
    probs = torch.softmax(logits, dim=-1)
    class_index = max(0, min(class_index, probs.shape[-1] - 1))
    return float(probs[0, class_index].item())


class RollingAverage:
    def __init__(self, maxlen: int = 4):
        self.values: Deque[float] = deque(maxlen=maxlen)

    def add(self, value: float) -> float:
        self.values.append(float(value))
        return self.mean

    @property
    def mean(self) -> float:
        if not self.values:
            return 0.0
        return float(sum(self.values) / len(self.values))

    @property
    def is_full(self) -> bool:
        return len(self.values) == self.values.maxlen


class SPARCLiveMonitor:
    def __init__(
        self,
        checkpoint: str,
        n_classes: int,
        source: str,
        device: str = "cpu",
        yolo_model: str = "yolov8n-seg.pt",
        confidence: float = 0.5,
        target_fps: float = 10.0,
        boundary_points: int = 128,
        temporal_window: int = 60,
        pair_distance_ratio: float = 0.18,
        pair_persistence: int = 5,
        smoothing_windows: int = 4,
        alert_threshold: float = 0.5,
        positive_class: int = 1,
    ):
        from ultralytics import YOLO

        self.source = source
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.boundary_points = boundary_points
        self.temporal_window = temporal_window
        self.target_fps = target_fps
        self.frame_stride = 1
        self.confidence = confidence
        self.pair_distance_ratio = pair_distance_ratio
        self.pair_persistence = pair_persistence
        self.alert_threshold = alert_threshold
        self.positive_class = positive_class

        self.yolo = YOLO(yolo_model)

        self.model = build_scn(n_classes=n_classes, slow_to_fast=True).to(self.device)
        ckpt = torch.load(checkpoint, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

        self.pair_history: Dict[PairKey, int] = {}
        self.pair_buffers: Dict[PairKey, Deque[np.ndarray]] = defaultdict(lambda: deque(maxlen=temporal_window))
        self.smoothed_scores: Dict[PairKey, RollingAverage] = defaultdict(lambda: RollingAverage(maxlen=smoothing_windows))

        self.alert_log = Path("alerts.jsonl")

    def _run_tracker(self, frame: np.ndarray):
        results = self.yolo.track(
            frame,
            persist=True,
            classes=[0],
            conf=self.confidence,
            verbose=False,
            device=self.device.type,
            tracker="bytetrack.yaml",
        )
        return results[0]

    def _observations_from_result(self, result, frame_shape: Tuple[int, int]) -> List[TrackObservation]:
        if result.boxes is None or len(result.boxes) == 0:
            return []

        masks = None
        if result.masks is not None and len(result.masks) > 0:
            masks = result.masks.data.detach().cpu().numpy()

        observations: List[TrackObservation] = []
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        ids = result.boxes.id.detach().cpu().numpy() if result.boxes.id is not None else None

        for i, box in enumerate(boxes):
            track_id = _safe_track_id(ids[i]) if ids is not None else None
            if track_id is None:
                continue

            x1, y1, x2, y2 = box
            centroid = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)

            if masks is not None and i < masks.shape[0]:
                mask = masks[i]
                if mask.shape[:2] != frame_shape:
                    mask = cv2.resize(mask.astype(np.uint8), frame_shape[::-1], interpolation=cv2.INTER_NEAREST)
                mask = mask > 0.5
            else:
                mask = np.zeros(frame_shape, dtype=bool)
                x1i, y1i, x2i, y2i = map(int, [max(0, x1), max(0, y1), min(frame_shape[1] - 1, x2), min(frame_shape[0] - 1, y2)])
                mask[y1i:y2i, x1i:x2i] = True

            observations.append(TrackObservation(track_id=track_id, centroid=centroid, mask=mask))

        return observations

    def _current_pairs(self, observations: List[TrackObservation], frame_shape: Tuple[int, int]) -> Dict[PairKey, List[TrackObservation]]:
        if len(observations) < 2:
            self.pair_history = {}
            return {}

        diag = float(np.hypot(frame_shape[0], frame_shape[1]))
        max_distance = diag * self.pair_distance_ratio

        pair_map: Dict[PairKey, List[TrackObservation]] = {}
        next_history: Dict[PairKey, int] = {}

        for i in range(len(observations)):
            for j in range(i + 1, len(observations)):
                a = observations[i]
                b = observations[j]
                distance = float(np.linalg.norm(a.centroid - b.centroid))
                if distance > max_distance:
                    continue

                key = _pair_key(a.track_id, b.track_id)
                next_history[key] = self.pair_history.get(key, 0) + 1
                pair_map[key] = [a, b]

        self.pair_history = next_history
        return pair_map

    def _pair_boundary(self, pair_observations: List[TrackObservation]) -> np.ndarray:
        union_mask = np.zeros_like(pair_observations[0].mask, dtype=bool)
        for obs in pair_observations:
            union_mask |= obs.mask
        contour = _largest_contour(union_mask)
        return resample_contour_uniform(contour, self.boundary_points)

    def _classify_pair(self, pair_key: PairKey, boundaries: List[np.ndarray]) -> float:
        slow_pc, slow_den = build_point_cloud_from_frames(boundaries, n_points=4096)
        faster_pc, faster_den = build_point_cloud_from_frames(boundaries[::2] or boundaries, n_points=4096)
        fastest_pc, fastest_den = build_point_cloud_from_frames(boundaries[::3] or boundaries, n_points=4096)

        batch = {
            "slow_pts": torch.from_numpy(slow_pc[None]).to(self.device),
            "slow_density": torch.from_numpy(slow_den[None]).to(self.device),
            "faster_pts": torch.from_numpy(faster_pc[None]).to(self.device),
            "faster_density": torch.from_numpy(faster_den[None]).to(self.device),
            "fastest_pts": torch.from_numpy(fastest_pc[None]).to(self.device),
            "fastest_density": torch.from_numpy(fastest_den[None]).to(self.device),
        }

        with torch.no_grad():
            logits = self.model(batch)
        score = _softmax_probability(logits, self.positive_class)
        smoothed = self.smoothed_scores[pair_key].add(score)
        return smoothed

    def _notify(self, pair_key: PairKey, score: float, frame_index: int):
        message = {
            "timestamp": time.time(),
            "frame_index": frame_index,
            "pair": list(pair_key),
            "smoothed_score": score,
            "threshold": self.alert_threshold,
        }
        print(f"[ALERT] pair={pair_key} score={score:.3f} frame={frame_index}")
        try:
            with self.alert_log.open("a", encoding="utf-8") as f:
                f.write(json.dumps(message) + "\n")
        except OSError:
            pass

        try:
            import winsound

            winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
        except Exception:
            pass

    def run(self, display: bool = True):
        capture = cv2.VideoCapture(self.source)
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open source: {self.source}")

        source_fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        if source_fps > 0 and self.target_fps > 0:
            self.frame_stride = max(1, int(round(source_fps / self.target_fps)))

        frame_index = 0
        processed_index = 0

        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if frame_index % self.frame_stride != 0:
                frame_index += 1
                continue

            processed_index += 1
            result = self._run_tracker(frame)
            observations = self._observations_from_result(result, frame.shape[:2])
            pairs = self._current_pairs(observations, frame.shape[:2])

            active_pairs = set(self.pair_history.keys())
            for stale_pair in list(self.pair_buffers.keys()):
                if stale_pair not in active_pairs:
                    self.pair_buffers.pop(stale_pair, None)
                    self.smoothed_scores.pop(stale_pair, None)

            for pair_key, pair_observations in pairs.items():
                if self.pair_history.get(pair_key, 0) < self.pair_persistence:
                    continue

                boundary = self._pair_boundary(pair_observations)
                if boundary.shape[0] == 0:
                    continue

                buffer = self.pair_buffers[pair_key]
                buffer.append(boundary)
                if len(buffer) < self.temporal_window:
                    continue

                score = self._classify_pair(pair_key, list(buffer))
                if score >= self.alert_threshold and self.smoothed_scores[pair_key].is_full:
                    self._notify(pair_key, score, processed_index)

            if display:
                overlay = frame.copy()
                cv2.putText(
                    overlay,
                    f"processed={processed_index} pairs={len(pairs)}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("SPARC-V Live Monitor", overlay)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_index += 1

        capture.release()
        if display:
            cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="Run the local SPARC-V RTSP/LAN monitor.")
    parser.add_argument("--source", required=True, help="RTSP URL, camera index, or video file path")
    parser.add_argument("--checkpoint", required=True, help="Trained SCN checkpoint path")
    parser.add_argument("--n_classes", type=int, required=True, help="Number of classes in the trained model")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--yolo_model", default="yolov8n-seg.pt", help="Segmentation model for tracking")
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--target_fps", type=float, default=10.0)
    parser.add_argument("--boundary_points", type=int, default=128)
    parser.add_argument("--temporal_window", type=int, default=60)
    parser.add_argument("--pair_distance_ratio", type=float, default=0.18)
    parser.add_argument("--pair_persistence", type=int, default=5)
    parser.add_argument("--smoothing_windows", type=int, default=4)
    parser.add_argument("--alert_threshold", type=float, default=0.5)
    parser.add_argument("--positive_class", type=int, default=1)
    parser.add_argument("--no_display", action="store_true")
    return parser.parse_args()


def _coerce_source(source: str):
    try:
        return int(source)
    except ValueError:
        return source


def main():
    args = parse_args()
    monitor = SPARCLiveMonitor(
        checkpoint=args.checkpoint,
        n_classes=args.n_classes,
        source=_coerce_source(args.source),
        device=args.device,
        yolo_model=args.yolo_model,
        confidence=args.confidence,
        target_fps=args.target_fps,
        boundary_points=args.boundary_points,
        temporal_window=args.temporal_window,
        pair_distance_ratio=args.pair_distance_ratio,
        pair_persistence=args.pair_persistence,
        smoothing_windows=args.smoothing_windows,
        alert_threshold=args.alert_threshold,
        positive_class=args.positive_class,
    )
    monitor.run(display=not args.no_display)


if __name__ == "__main__":
    main()