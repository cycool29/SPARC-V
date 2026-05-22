"""
test_pipeline.py — Smoke test for the SCN pipeline.

Generates synthetic silhouette point cloud data and runs a forward pass
and a few training iterations to verify the entire pipeline is functional.
No real video data needed.

Usage:
    python test_pipeline.py
"""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch


def generate_synthetic_dataset(root: str, n_classes: int = 5, n_videos: int = 20, n_points: int = 512):
    """Create a minimal synthetic dataset of .npz files."""
    root_path = Path(root)
    class_to_idx = {f"action_{i}": i for i in range(n_classes)}

    with open(root_path / "class_map.json", "w") as f:
        json.dump(class_to_idx, f)

    paths = []
    for cls_name, label in class_to_idx.items():
        cls_dir = root_path / cls_name
        cls_dir.mkdir(parents=True, exist_ok=True)
        for v in range(n_videos // n_classes):
            pts    = np.random.randn(n_points, 3).astype(np.float32)
            den    = np.abs(np.random.randn(n_points)).astype(np.float32) + 1e-3
            fname  = cls_dir / f"video_{v:03d}.npz"
            np.savez_compressed(
                str(fname),
                slow_pts       =pts, slow_density   =den,
                faster_pts     =pts, faster_density =den,
                fastest_pts    =pts, fastest_density=den,
                label          =np.array(label, dtype=np.int64),
                class_name     =np.array(cls_name),
            )
            paths.append((f"{cls_name}/video_{v:03d}", label))

    # Write split files
    np.random.shuffle(paths)
    split_dir = root_path / "splits"
    split_dir.mkdir(exist_ok=True)
    train_f = split_dir / "train.txt"
    test_f  = split_dir / "test.txt"
    with open(train_f, "w") as ft, open(test_f, "w") as fv:
        for i, (p, l) in enumerate(paths):
            if i < len(paths) * 0.8:
                ft.write(f"{p} {l}\n")
            else:
                fv.write(f"{p} {l}\n")

    return str(train_f), str(test_f)


def run_smoke_test():
    print("=" * 60)
    print("SCN Pipeline Smoke Test")
    print("=" * 60)

    # ----------------------------------------------------------------
    # 1. Test utility functions
    # ----------------------------------------------------------------
    print("\n[1/5] Testing utility functions...")
    from utils import (
        farthest_point_sampling,
        farthest_point_sampling_torch,
        compute_density_coefficients,
        normalize_point_cloud,
        knn_query,
        resample_contour_uniform,
        build_point_cloud_from_frames,
    )

    pts_np = np.random.randn(1000, 3).astype(np.float32)
    sampled = farthest_point_sampling(pts_np, 128)
    assert sampled.shape == (128, 3), f"FPS shape mismatch: {sampled.shape}"

    pts_t = torch.from_numpy(pts_np).unsqueeze(0)  # (1, 1000, 3)
    sampled_t = farthest_point_sampling_torch(pts_t, 128)
    assert sampled_t.shape == (1, 128, 3)

    den = compute_density_coefficients(sampled, bandwidth=0.1)
    assert den.shape == (128,)

    norm = normalize_point_cloud(pts_np)
    assert norm.max() <= 1.0 + 1e-5

    q  = torch.randn(2, 64, 3)
    s  = torch.randn(2, 256, 3)
    idx = knn_query(q, s, k=16)
    assert idx.shape == (2, 64, 16)

    contour = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
    contour_128 = resample_contour_uniform(contour, 128)
    assert contour_128.shape == (128, 2)

    pc_3d, density_3d = build_point_cloud_from_frames([contour_128] * 5, n_points=256)
    assert pc_3d.shape == (256, 3)
    assert density_3d.shape == (256,)

    print("  ✓ All utility functions pass.")

    # ----------------------------------------------------------------
    # 2. Test dataset loading
    # ----------------------------------------------------------------
    print("\n[2/5] Testing dataset loading...")
    from dataset import SilhouettePointCloudDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        train_split, test_split = generate_synthetic_dataset(
            tmpdir, n_classes=5, n_videos=20, n_points=512
        )

        ds = SilhouettePointCloudDataset(
            data_dir=tmpdir,
            split_file=train_split,
            n_points=512,
            augment=True,
            slow_to_fast=True,
        )
        assert len(ds) > 0, "Empty dataset"
        sample = ds[0]
        assert "slow_pts" in sample
        assert sample["slow_pts"].shape[-1] == 3
        assert "label" in sample
        print(f"  ✓ Dataset loaded: {len(ds)} samples, sample keys: {list(sample.keys())}")

        # ----------------------------------------------------------------
        # 3. Test model forward pass
        # ----------------------------------------------------------------
        print("\n[3/5] Testing model forward pass...")
        from model import build_scn

        model = build_scn(n_classes=5, slow_to_fast=True, n1=64, n2=16, k=8, base_channels=32)
        model.eval()
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Model parameters: {n_params:,}")

        # Build a micro-batch
        B = 2
        N = 512
        micro_batch = {
            "slow_pts":       torch.randn(B, N, 3),
            "slow_density":   torch.abs(torch.randn(B, N)) + 1e-3,
            "faster_pts":     torch.randn(B, N, 3),
            "faster_density": torch.abs(torch.randn(B, N)) + 1e-3,
            "fastest_pts":    torch.randn(B, N, 3),
            "fastest_density":torch.abs(torch.randn(B, N)) + 1e-3,
            "label": torch.zeros(B, dtype=torch.long),
        }

        with torch.no_grad():
            logits = model(micro_batch)
        assert logits.shape == (B, 5), f"Unexpected logits shape: {logits.shape}"
        print(f"  ✓ Forward pass: input (B={B}, N={N}) → logits {tuple(logits.shape)}")

        # ----------------------------------------------------------------
        # 4. Test backward pass
        # ----------------------------------------------------------------
        print("\n[4/5] Testing backward pass + optimizer step...")
        import torch.optim as optim

        model.train()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()

        for step in range(3):
            optimizer.zero_grad()
            logits = model(micro_batch)
            loss   = criterion(logits, micro_batch["label"])
            loss.backward()
            optimizer.step()
            print(f"  Step {step+1}: loss={loss.item():.4f}")

        print("  ✓ Backward pass and optimizer working.")

        # ----------------------------------------------------------------
        # 5. Test single-scale SCN
        # ----------------------------------------------------------------
        print("\n[5/5] Testing single-scale SCN...")
        single_model = build_scn(n_classes=5, slow_to_fast=False, n1=64, n2=16, k=8, base_channels=32)
        single_model.eval()
        single_batch = {
            "points":  torch.randn(B, N, 3),
            "density": torch.abs(torch.randn(B, N)) + 1e-3,
            "label":   torch.zeros(B, dtype=torch.long),
        }
        with torch.no_grad():
            logits = single_model(single_batch)
        assert logits.shape == (B, 5)
        print(f"  ✓ Single-scale SCN: logits {tuple(logits.shape)}")

    print("\n" + "=" * 60)
    print("All smoke tests passed! ✓")
    print("=" * 60)
    print("""
Next steps:
  1. python extract_silhouettes.py --video_dir data/HMDB/videos --output_dir data/HMDB/silhouettes
  2. python build_point_clouds.py  --silhouette_dir data/HMDB/silhouettes --output_dir data/HMDB/point_clouds
  3. python train.py --dataset HMDB --data_dir data/HMDB/point_clouds --n_classes 51 --epochs 100
  4. python evaluate.py --data_dir data/HMDB/point_clouds --checkpoint checkpoints/scn/best_model.pth --n_classes 51
""")


if __name__ == "__main__":
    run_smoke_test()
