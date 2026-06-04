"""
dataset.py — PyTorch Dataset classes for SCN training.

Supports HMDB51, JHMDB, and UCF101 in their standard 3-split format.
Loads pre-built .npz point cloud files (output of build_point_clouds.py).

Paper Section 4.1:
  "JHMDB, HMDB, and UCF101 have provided three training/testing splits.
   Following the standard protocols, we train and test using the provided splits."
"""

import json
import os
import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from utils import normalize_point_cloud_torch, farthest_point_sampling_torch


# ---------------------------------------------------------------------------
# Base point cloud dataset
# ---------------------------------------------------------------------------

class SilhouettePointCloudDataset(Dataset):
    """
    Loads pre-built .npz files produced by build_point_clouds.py.

    Each sample contains:
        slow_pts    (N, 3)  — all frames temporal scale
        faster_pts  (N, 3)  — every 2nd frame
        fastest_pts (N, 3)  — every 3rd frame
        slow_density    (N,)
        faster_density  (N,)
        fastest_density (N,)
        label       int
    """

    def __init__(
        self,
        data_dir: str,
        split_file: str,
        n_points: int = 4096,
        augment: bool = True,
        slow_to_fast: bool = True,
    ):
        """
        Args:
            data_dir:    root of .npz point cloud files (<class>/<video>.npz)
            split_file:  text file listing video paths relative to data_dir
                         (one per line: "<class>/<video_name>")
            n_points:    number of points per sample
            augment:     apply random jitter & rotation during training
            slow_to_fast: load 3 temporal scales (True) or single scale (False)
        """
        self.data_dir    = Path(data_dir)
        self.n_points    = n_points
        self.augment     = augment
        self.slow_to_fast = slow_to_fast

        # Load class mapping
        class_map_path = self.data_dir / "class_map.json"
        if class_map_path.exists():
            with open(class_map_path) as f:
                self.class_to_idx = json.load(f)
        else:
            self.class_to_idx = {}

        # Load sample list
        self.samples = self._load_split(split_file)
        print(f"  Dataset: {len(self.samples)} samples | augment={augment}")

    def _load_split(self, split_file: str) -> list:
        """Return list of (npz_path, label) tuples."""
        samples = []
        with open(split_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Expected format: "<class_name>/<video_name>" optionally with label
                parts = line.split()
                rel_path = parts[0]
                npz_path = self.data_dir / (rel_path + ".npz")

                # Infer label from class directory name
                class_name = Path(rel_path).parent.name
                label = self.class_to_idx.get(class_name, -1)
                if len(parts) > 1:
                    try:
                        label = int(parts[1])
                    except ValueError:
                        pass

                if npz_path.exists():
                    samples.append((str(npz_path), label))
                else:
                    print(f"  [WARN] Missing: {npz_path}")
        return samples

    # ------------------------------------------------------------------
    # Augmentation (applied during training only)
    # ------------------------------------------------------------------

    @staticmethod
    def _random_jitter(pts: torch.Tensor, sigma: float = 0.01) -> torch.Tensor:
        """Add small Gaussian noise to point coordinates."""
        noise = torch.randn_like(pts) * sigma
        return pts + noise

    @staticmethod
    def _random_scale(pts: torch.Tensor, lo: float = 0.9, hi: float = 1.1) -> torch.Tensor:
        scale = random.uniform(lo, hi)
        return pts * scale

    @staticmethod
    def _random_rotation_xy(pts: torch.Tensor) -> torch.Tensor:
        """Random rotation in the xy-plane (keeps temporal z intact)."""
        angle = random.uniform(0, 2 * np.pi)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        R = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a,  cos_a, 0],
            [0,      0,     1],
        ], dtype=torch.float32)
        return pts @ R.T

    def _augment(self, pts: torch.Tensor) -> torch.Tensor:
        pts = self._random_jitter(pts)
        pts = self._random_scale(pts)
        pts = self._random_rotation_xy(pts)
        return pts

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        npz_path, label = self.samples[idx]
        data = np.load(npz_path, allow_pickle=True)

        if self.slow_to_fast:
            slow_pts      = torch.from_numpy(data["slow_pts"].copy())
            slow_density  = torch.from_numpy(data["slow_density"].copy())
            faster_pts    = torch.from_numpy(data["faster_pts"].copy())
            faster_density= torch.from_numpy(data["faster_density"].copy())
            fastest_pts   = torch.from_numpy(data["fastest_pts"].copy())
            fastest_density=torch.from_numpy(data["fastest_density"].copy())

            if self.augment:
                slow_pts    = self._augment(slow_pts)
                faster_pts  = self._augment(faster_pts)
                fastest_pts = self._augment(fastest_pts)

            return {
                "slow_pts":       slow_pts,
                "slow_density":   slow_density,
                "faster_pts":     faster_pts,
                "faster_density": faster_density,
                "fastest_pts":    fastest_pts,
                "fastest_density":fastest_density,
                "label": torch.tensor(label, dtype=torch.long),
            }
        else:
            pts     = torch.from_numpy(data["points"].copy())
            density = torch.from_numpy(data["density"].copy())
            if self.augment:
                pts = self._augment(pts)
            return {
                "points":  pts,
                "density": density,
                "label":   torch.tensor(label, dtype=torch.long),
            }


# ---------------------------------------------------------------------------
# Convenience factory: build train/val DataLoaders for a given split
# ---------------------------------------------------------------------------

def get_dataloaders(
    data_dir: str,
    train_split_file: str,
    test_split_file: str,
    batch_size: int = 64,
    n_points: int = 4096,
    slow_to_fast: bool = True,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and test DataLoaders.

    Paper Section 4.2: batch_size=64.
    """
    train_ds = SilhouettePointCloudDataset(
        data_dir=data_dir,
        split_file=train_split_file,
        n_points=n_points,
        augment=True,
        slow_to_fast=slow_to_fast,
    )
    test_ds = SilhouettePointCloudDataset(
        data_dir=data_dir,
        split_file=test_split_file,
        n_points=n_points,
        augment=False,
        slow_to_fast=slow_to_fast,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=4,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
    )
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Split file generator (if official split files are not available)
# ---------------------------------------------------------------------------

def generate_split_files(
    data_dir: str,
    output_dir: str,
    val_ratio: float = 0.2,
    seed: int = 42,
):
    """
    Auto-generate train/test split files from the .npz directory structure
    when official splits are not available.
    """
    data_dir  = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load class mapping
    with open(data_dir / "class_map.json") as f:
        class_to_idx = json.load(f)

    def _collect_samples(prefix: Optional[str] = None):
        samples = []
        for class_name, label in class_to_idx.items():
            base_dir = data_dir / class_name if prefix is None else data_dir / prefix / class_name
            if not base_dir.exists():
                continue
            for npz_file in sorted(base_dir.glob("*.npz")):
                if prefix is None:
                    rel = f"{class_name}/{npz_file.stem}"
                else:
                    rel = f"{prefix}/{class_name}/{npz_file.stem}"
                samples.append((rel, label))
        return samples

    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    if train_dir.exists() and val_dir.exists():
        train_samples = _collect_samples("train")
        test_samples = _collect_samples("val")
        print("Detected train/val directory layout; using it directly for splits.")
    else:
        all_samples = _collect_samples(prefix=None)
        random.seed(seed)
        random.shuffle(all_samples)
        split_idx = int(len(all_samples) * (1 - val_ratio))
        train_samples = all_samples[:split_idx]
        test_samples  = all_samples[split_idx:]

    train_file = output_dir / "train_split1.txt"
    test_file  = output_dir / "test_split1.txt"

    with open(train_file, "w") as f:
        for rel, label in train_samples:
            f.write(f"{rel} {label}\n")

    with open(test_file, "w") as f:
        for rel, label in test_samples:
            f.write(f"{rel} {label}\n")

    print(f"Generated splits: {len(train_samples)} train, {len(test_samples)} test")
    print(f"  → {train_file}")
    print(f"  → {test_file}")
    return str(train_file), str(test_file)
