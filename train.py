"""
train.py — Training loop for SCN.

Implements the training setup from Section 4.2:
  - Optimizer: Adam
  - Initial LR: 0.001, reduced to 0.00075 mid-training, 0.0005 near end
  - Weight decay: 0.0001
  - Momentum: 0.8 (used for SGD fallback; Adam uses betas)
  - Batch size: 64
  - Dropout: 0.3 after each conv layer
  - Trains from scratch (no pre-training required)
  - Single GPU (Nvidia Titan Xp in the paper)

Usage:
    python train.py \
        --dataset HMDB \
    --data_dir data/point_clouds \
    --train_split checkpoints/scn/splits/train_split1.txt \
    --test_split  checkpoints/scn/splits/test_split1.txt \
    --n_classes 2 \
        --epochs 100 \
        --batch_size 64 \
    --output_dir checkpoints/scn
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from dataset import get_dataloaders, generate_split_files
from model import build_scn


# ---------------------------------------------------------------------------
# Learning rate schedule (paper Section 4.2)
# ---------------------------------------------------------------------------

class PaperLRScheduler:
    """
    Mimics the paper's schedule:
      epochs 0..mid_1:    lr = 0.001
      epochs mid_1..mid_2: lr = 0.00075
      epochs mid_2..end:   lr = 0.0005
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        total_epochs: int,
        lr_start: float  = 1e-3,
        lr_mid: float    = 7.5e-4,
        lr_end: float    = 5e-4,
        mid_frac_1: float = 0.5,
        mid_frac_2: float = 0.75,
    ):
        self.optimizer    = optimizer
        self.total_epochs = total_epochs
        self.lr_start     = lr_start
        self.lr_mid       = lr_mid
        self.lr_end       = lr_end
        self.boundary_1   = int(total_epochs * mid_frac_1)
        self.boundary_2   = int(total_epochs * mid_frac_2)

    def step(self, epoch: int):
        if epoch < self.boundary_1:
            lr = self.lr_start
        elif epoch < self.boundary_2:
            lr = self.lr_mid
        else:
            lr = self.lr_end
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr


# ---------------------------------------------------------------------------
# One training epoch
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    slow_to_fast: bool,
    micro_batch_size: int,
    epoch: int,
) -> dict:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    t0 = time.time()

    def _iter_micro_batches(batch: dict):
        batch_size = batch["label"].shape[0]
        step = max(1, int(micro_batch_size))
        for start in range(0, batch_size, step):
            end = min(start + step, batch_size)
            yield {
                k: (v[start:end] if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }

    for batch_idx, batch in enumerate(loader):
        optimizer.zero_grad()

        batch_samples = batch["label"].shape[0]
        micro_loss_sum = 0.0
        preds_all = []
        labels_all = []

        for micro_batch in _iter_micro_batches(batch):
            micro_batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in micro_batch.items()
            }
            labels = micro_batch["label"]

            logits = model(micro_batch)
            loss = criterion(logits, labels)
            scaled_loss = loss * (labels.size(0) / batch_samples)
            scaled_loss.backward()
            micro_loss_sum += loss.item() * labels.size(0)
            preds_all.append(logits.argmax(dim=-1).detach())
            labels_all.append(labels.detach())

        # Gradient clipping (optional but helpful for stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        labels = torch.cat(labels_all, dim=0)
        preds = torch.cat(preds_all, dim=0)
        correct = (preds == labels).sum().item()
        bs = labels.size(0)

        total_loss    += micro_loss_sum
        total_correct += correct
        total_samples += bs

        if (batch_idx + 1) % 20 == 0:
            print(
                f"  Epoch {epoch} [{batch_idx+1}/{len(loader)}] "
                f"loss={loss.item():.4f} "
                f"acc={correct/bs*100:.1f}%"
            )

    elapsed = time.time() - t0
    return {
        "loss":     total_loss / total_samples,
        "accuracy": total_correct / total_samples * 100,
        "time_s":   elapsed,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    micro_batch_size: int,
) -> dict:
    model.eval()
    total_loss    = 0.0
    total_correct = 0
    total_samples = 0

    # Per-class accuracy tracking (mean per-class accuracy, as in the paper)
    class_correct = {}
    class_total   = {}

    def _iter_micro_batches(batch: dict):
        batch_size = batch["label"].shape[0]
        step = max(1, int(micro_batch_size))
        for start in range(0, batch_size, step):
            end = min(start + step, batch_size)
            yield {
                k: (v[start:end] if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }

    for batch in loader:
        batch_loss = 0.0
        batch_total = 0

        for micro_batch in _iter_micro_batches(batch):
            micro_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                           for k, v in micro_batch.items()}
            labels = micro_batch["label"]

            logits = model(micro_batch)
            loss   = criterion(logits, labels)
            preds  = logits.argmax(dim=-1)
            correct = (preds == labels)

            batch_loss += loss.item() * labels.size(0)
            batch_total += labels.size(0)
            total_correct += correct.sum().item()
            total_samples += labels.size(0)

            # Per-class accumulation
            for c, p in zip(labels.cpu().numpy(), correct.cpu().numpy()):
                class_correct[c] = class_correct.get(c, 0) + int(p)
                class_total[c]   = class_total.get(c, 0) + 1

        total_loss += batch_loss

    # Mean per-class accuracy (as used in the paper)
    per_class_acc = [
        class_correct.get(c, 0) / class_total[c]
        for c in class_total
    ]
    mean_class_acc = sum(per_class_acc) / len(per_class_acc) * 100 if per_class_acc else 0.0

    return {
        "loss":            total_loss / max(total_samples, 1),
        "accuracy":        total_correct / max(total_samples, 1) * 100,
        "mean_class_acc":  mean_class_acc,
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Handle auto-generated splits
    train_split = args.train_split
    test_split  = args.test_split
    if train_split is None or not os.path.exists(train_split):
        print("Split files not found — auto-generating from data directory.")
        train_split, test_split = generate_split_files(
            data_dir=args.data_dir,
            output_dir=str(output_dir / "splits"),
        )

    # --- DataLoaders ---
    print(f"\nLoading dataset: {args.dataset}")
    train_loader, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        train_split_file=train_split,
        test_split_file=test_split,
        batch_size=args.batch_size,
        n_points=args.n_points,
        slow_to_fast=not args.no_slow_to_fast,
        num_workers=args.num_workers,
    )

    # --- Model ---
    model = build_scn(
        n_classes=args.n_classes,
        slow_to_fast=not args.no_slow_to_fast,
        n1=args.n1,
        n2=args.n2,
        k=args.k,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # --- Optimizer (Adam, as per paper Section 4.2) ---
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.momentum, 0.999),
    )

    lr_scheduler = PaperLRScheduler(
        optimizer,
        total_epochs=args.epochs,
        lr_start=args.lr,
    )

    criterion = nn.CrossEntropyLoss()

    # --- TensorBoard ---
    writer = SummaryWriter(log_dir=str(output_dir / "tb_logs"))

    # --- Resume from checkpoint ---
    start_epoch = 0
    best_acc    = 0.0
    ckpt_path   = output_dir / "latest.pth"
    best_path   = output_dir / "best_model.pth"

    if args.resume and ckpt_path.exists():
        print(f"Resuming from {ckpt_path}")
        ckpt = torch.load(str(ckpt_path), map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_acc    = ckpt.get("best_acc", 0.0)

    # --- Training loop ---
    history = []
    print(f"\nTraining for {args.epochs} epochs (start={start_epoch})\n{'='*60}")

    for epoch in range(start_epoch, args.epochs):
        lr = lr_scheduler.step(epoch)
        print(f"\nEpoch {epoch+1}/{args.epochs}  lr={lr:.6f}")

        # Train
        train_stats = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            slow_to_fast=not args.no_slow_to_fast,
            micro_batch_size=args.micro_batch_size,
            epoch=epoch + 1,
        )

        # Evaluate every `eval_freq` epochs
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            val_stats = evaluate(model, test_loader, criterion, device, args.micro_batch_size)
        else:
            val_stats = {"loss": float("nan"), "accuracy": float("nan"), "mean_class_acc": float("nan")}

        # Logging
        writer.add_scalar("train/loss",     train_stats["loss"],    epoch)
        writer.add_scalar("train/acc",      train_stats["accuracy"],epoch)
        writer.add_scalar("val/loss",       val_stats["loss"],      epoch)
        writer.add_scalar("val/acc",        val_stats["accuracy"],  epoch)
        writer.add_scalar("val/mean_class", val_stats["mean_class_acc"], epoch)
        writer.add_scalar("lr", lr, epoch)

        print(
            f"  Train: loss={train_stats['loss']:.4f}  acc={train_stats['accuracy']:.2f}%  "
            f"time={train_stats['time_s']:.1f}s"
        )
        print(
            f"  Val:   loss={val_stats['loss']:.4f}  acc={val_stats['accuracy']:.2f}%  "
            f"mean_class_acc={val_stats['mean_class_acc']:.2f}%"
        )

        # Save checkpoint
        ckpt = {
            "epoch":     epoch,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_acc":  best_acc,
            "args":      vars(args),
        }
        torch.save(ckpt, str(ckpt_path))

        if val_stats["mean_class_acc"] > best_acc:
            best_acc = val_stats["mean_class_acc"]
            torch.save(ckpt, str(best_path))
            print(f"  ★ New best mean-class accuracy: {best_acc:.2f}%  saved → {best_path}")

        history.append({"epoch": epoch + 1, "train": train_stats, "val": val_stats})

    writer.close()

    # Save training history
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best mean-class accuracy: {best_acc:.2f}%")
    print(f"Best checkpoint saved to: {best_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train SCN for video action recognition.")

    # Dataset
    p.add_argument("--dataset",     default="HMDB",
                   choices=["HMDB", "JHMDB", "UCF101"],
                   help="Dataset name")
    p.add_argument("--data_dir",    required=True,
                   help="Root dir of .npz point cloud files")
    p.add_argument("--train_split", default=None,
                   help="Train split file (auto-generated if not provided)")
    p.add_argument("--test_split",  default=None,
                   help="Test split file")
    p.add_argument("--n_classes",   type=int, default=51,
                   help="Number of action classes (HMDB=51, JHMDB=21, UCF101=101)")

    # Model
    p.add_argument("--n_points",        type=int, default=4096,
                   help="Points per point cloud (paper: 4096)")
    p.add_argument("--n1",              type=int, default=512,
                   help="FPS centroids layer 1 (paper: 512)")
    p.add_argument("--n2",              type=int, default=128,
                   help="FPS centroids layer 2 (paper: 128)")
    p.add_argument("--k",               type=int, default=20,
                   help="kNN neighbours")
    p.add_argument("--no_slow_to_fast", action="store_true",
                   help="Use single-scale SCN instead of Slow-to-Fast")
    p.add_argument("--dropout",         type=float, default=0.3,
                   help="Dropout probability (paper: 0.3)")

    # Training
    p.add_argument("--epochs",      type=int,   default=100)
    p.add_argument("--batch_size",  type=int,   default=64,
                   help="Batch size (paper: 64)")
    p.add_argument("--lr",          type=float, default=1e-3,
                   help="Initial learning rate (paper: 0.001)")
    p.add_argument("--weight_decay",type=float, default=1e-4,
                   help="Weight decay (paper: 0.0001)")
    p.add_argument("--momentum",    type=float, default=0.8,
                   help="Adam beta1 (paper momentum: 0.8)")
    p.add_argument("--eval_freq",   type=int,   default=5,
                   help="Evaluate every N epochs")
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--micro_batch_size", type=int, default=8,
                   help="Sub-batch size used inside each loaded batch to reduce GPU memory")

    # I/O
    p.add_argument("--output_dir",  default="checkpoints/scn",
                   help="Where to save checkpoints and logs")
    p.add_argument("--device",      default="cuda",
                   help="cuda or cpu")
    p.add_argument("--resume",      action="store_true",
                   help="Resume from latest checkpoint")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Print config
    print("=" * 60)
    print("SCN Training Configuration")
    print("=" * 60)
    for k, v in sorted(vars(args).items()):
        print(f"  {k:25s}: {v}")
    print("=" * 60)

    train(args)
