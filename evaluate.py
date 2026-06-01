"""
evaluate.py — Evaluation for trained SCN models.

Reproduces the evaluation protocol from Section 4.1:
  "We test mean classification accuracy (in percentage) of the testing sets,
   i.e., the ratio of videos in a given class that is correctly classified
   averaged over all classes in the dataset."

  Three splits are run and averaged.

Usage:
    # Single split evaluation
    python evaluate.py \
        --data_dir  data/point_clouds \
        --split     checkpoints/scn/splits/test_split1.txt \
        --checkpoint checkpoints/scn/best_model.pth \
        --n_classes 2

    # All 3 splits (paper protocol)
    python evaluate.py \
        --data_dir  data/point_clouds \
        --splits    split1.txt split2.txt split3.txt \
        --checkpoints checkpoints/s1/best.pth checkpoints/s2/best.pth checkpoints/s3/best.pth \
        --n_classes 2
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import SilhouettePointCloudDataset
from model import build_scn


# ---------------------------------------------------------------------------
# Per-split evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_split(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: list = None,
) -> dict:
    """
    Compute overall accuracy, mean per-class accuracy, and per-class breakdown.
    """
    model.eval()
    all_preds  = []
    all_labels = []

    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        labels = batch["label"]
        logits = model(batch)
        preds  = logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    overall_acc = (all_preds == all_labels).mean() * 100

    # Mean per-class accuracy
    classes     = sorted(set(all_labels.tolist()))
    per_class   = {}
    for c in classes:
        mask         = all_labels == c
        c_acc        = (all_preds[mask] == all_labels[mask]).mean() * 100
        name         = class_names[c] if class_names and c < len(class_names) else str(c)
        per_class[name] = float(c_acc)

    mean_class_acc = np.mean(list(per_class.values()))

    return {
        "overall_accuracy": float(overall_acc),
        "mean_class_accuracy": float(mean_class_acc),
        "per_class_accuracy": per_class,
        "n_samples": len(all_labels),
    }


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def compute_confusion_matrix(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_classes: int,
) -> np.ndarray:
    model.eval()
    cm = np.zeros((n_classes, n_classes), dtype=int)

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            labels = batch["label"].cpu().numpy()
            preds  = model(batch).argmax(dim=-1).cpu().numpy()
            for true, pred in zip(labels, preds):
                if 0 <= true < n_classes and 0 <= pred < n_classes:
                    cm[true, pred] += 1

    return cm


# ---------------------------------------------------------------------------
# Multi-split evaluation (paper protocol: average over 3 splits)
# ---------------------------------------------------------------------------

def evaluate_three_splits(
    data_dir: str,
    split_files: list,
    checkpoint_files: list,
    n_classes: int,
    slow_to_fast: bool,
    n_points: int,
    batch_size: int,
    device: torch.device,
    class_names: list = None,
) -> dict:
    """
    Evaluate each split with its checkpoint and report the averaged accuracy.
    """
    assert len(split_files) == len(checkpoint_files), \
        "Must provide one checkpoint per split."

    split_results = []
    for i, (split_file, ckpt_file) in enumerate(zip(split_files, checkpoint_files)):
        print(f"\n--- Split {i+1} ---")
        print(f"  split: {split_file}")
        print(f"  ckpt:  {ckpt_file}")

        ds = SilhouettePointCloudDataset(
            data_dir=data_dir,
            split_file=split_file,
            n_points=n_points,
            augment=False,
            slow_to_fast=slow_to_fast,
        )
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

        model = build_scn(n_classes=n_classes, slow_to_fast=slow_to_fast).to(device)
        ckpt  = torch.load(ckpt_file, map_location=device)
        model.load_state_dict(ckpt["model"])

        result = evaluate_split(model, loader, device, class_names)
        split_results.append(result)

        print(f"  Overall accuracy:    {result['overall_accuracy']:.2f}%")
        print(f"  Mean class accuracy: {result['mean_class_accuracy']:.2f}%")

    # Average
    avg_overall    = np.mean([r["overall_accuracy"]    for r in split_results])
    avg_mean_class = np.mean([r["mean_class_accuracy"] for r in split_results])

    print(f"\n{'='*50}")
    print(f"3-Split Average Results:")
    print(f"  Overall accuracy:    {avg_overall:.2f}%")
    print(f"  Mean class accuracy: {avg_mean_class:.2f}%")
    print(f"{'='*50}")

    return {
        "splits": split_results,
        "avg_overall_accuracy":    float(avg_overall),
        "avg_mean_class_accuracy": float(avg_mean_class),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate trained SCN model(s).")

    p.add_argument("--data_dir",    required=True)
    p.add_argument("--n_classes",   type=int, required=True)
    p.add_argument("--n_points",    type=int, default=4096)
    p.add_argument("--batch_size",  type=int, default=64)
    p.add_argument("--device",      default="cuda")
    p.add_argument("--no_slow_to_fast", action="store_true")

    # Single-split mode
    p.add_argument("--split",       default=None,
                   help="Single test split file")
    p.add_argument("--checkpoint",  default=None,
                   help="Single checkpoint file")

    # Multi-split mode
    p.add_argument("--splits",       nargs="+", default=None,
                   help="Three test split files (paper protocol)")
    p.add_argument("--checkpoints",  nargs="+", default=None,
                   help="Three checkpoint files")

    p.add_argument("--class_map",   default=None,
                   help="Path to class_map.json (optional, for named output)")
    p.add_argument("--output",      default=None,
                   help="Save results JSON to this path")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load optional class names
    class_names = None
    class_map_path = args.class_map or str(Path(args.data_dir) / "class_map.json")
    if Path(class_map_path).exists():
        with open(class_map_path) as f:
            cm = json.load(f)
        class_names = [None] * (max(cm.values()) + 1)
        for name, idx in cm.items():
            class_names[idx] = name

    slow_to_fast = not args.no_slow_to_fast

    if args.splits and args.checkpoints:
        # Paper protocol: 3-split average
        results = evaluate_three_splits(
            data_dir=args.data_dir,
            split_files=args.splits,
            checkpoint_files=args.checkpoints,
            n_classes=args.n_classes,
            slow_to_fast=slow_to_fast,
            n_points=args.n_points,
            batch_size=args.batch_size,
            device=device,
            class_names=class_names,
        )
    else:
        # Single split
        assert args.split and args.checkpoint, \
            "Provide --split and --checkpoint, or --splits and --checkpoints."

        ds = SilhouettePointCloudDataset(
            data_dir=args.data_dir,
            split_file=args.split,
            n_points=args.n_points,
            augment=False,
            slow_to_fast=slow_to_fast,
        )
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

        model = build_scn(n_classes=args.n_classes, slow_to_fast=slow_to_fast).to(device)
        ckpt  = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])

        results = evaluate_split(model, loader, device, class_names)

        print(f"\nResults on {args.split}:")
        print(f"  Overall accuracy:    {results['overall_accuracy']:.2f}%")
        print(f"  Mean class accuracy: {results['mean_class_accuracy']:.2f}%")
        print(f"  Samples: {results['n_samples']}")

        if class_names:
            print("\nPer-class accuracy:")
            for name, acc in sorted(results["per_class_accuracy"].items()):
                print(f"  {name:30s}: {acc:.1f}%")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
