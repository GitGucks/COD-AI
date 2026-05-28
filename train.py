"""train.py — train or fine-tune the COD-AI ghost model.

Usage:
    python train.py                      # fine-tune from latest checkpoint
    python train.py --from-scratch       # ignore checkpoint, retrain clean
    python train.py --my-weight 2.0      # weight your own sessions twice as heavily
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import load_all_sessions, mark_sessions_used
from model import GhostModel
from paths import DATA_DIR, MODELS_DIR

BATCH_SIZE  = 32
MAX_EPOCHS  = 50
LR_SCRATCH  = 1e-3
LR_FINETUNE = 3e-4
PATIENCE    = 7   # early stopping: stop if val loss doesn't improve for this many epochs


def _next_version(models_dir: Path) -> int:
    existing = [p.stem for p in models_dir.glob("ghost_v*.pt")]
    nums = [int(m.group(1)) for s in existing if (m := re.search(r"v(\d+)", s))]
    return max(nums, default=0) + 1


def _loss(pred: dict, batch: dict, device: torch.device) -> torch.Tensor:
    mse = nn.functional.mse_loss
    bce = nn.functional.binary_cross_entropy

    ls_loss = mse(pred["left_stick"],  batch["left_stick"].to(device))
    rs_loss = mse(pred["right_stick"], batch["right_stick"].to(device))
    tr_loss = mse(pred["triggers"],    batch["triggers"].to(device))
    bt_loss = bce(pred["buttons"],     batch["buttons"].to(device))

    return ls_loss + rs_loss + tr_loss + 0.5 * bt_loss


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the COD-AI ghost model.")
    parser.add_argument("--from-scratch", action="store_true",
                        help="Ignore any existing checkpoint and train from scratch.")
    parser.add_argument("--my-weight", type=float, default=1.0,
                        help="Sample weight for your own sessions (player=me). Default 1.0.")
    parser.add_argument("--player", default="me",
                        help="Your player tag (must match what you used in collect.py).")
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nCOD-AI — Training")
    print(f"  Device:    {device}")
    print(f"  Batch:     {BATCH_SIZE}  |  Max epochs: {MAX_EPOCHS}  |  Patience: {PATIENCE}")

    print(f"\nLoading sessions from {DATA_DIR} ...")
    full_ds = load_all_sessions(DATA_DIR, my_player=args.player, my_weight=args.my_weight)
    n_val   = max(1, int(len(full_ds) * 0.10))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))
    print(f"  Total frames: {len(full_ds)}  (train={n_train}, val={n_val})")

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model = GhostModel().to(device)

    latest_ckpt = MODELS_DIR / "ghost_latest.pt"
    if not args.from_scratch and latest_ckpt.exists():
        print(f"\nFine-tuning from {latest_ckpt}")
        model.load_state_dict(torch.load(latest_ckpt, map_location=device))
        lr = LR_FINETUNE
    else:
        print("\nTraining from scratch (pretrained EfficientNet-B0 backbone).")
        lr = LR_SCRATCH

    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, patience=3, factor=0.5)

    best_val   = float("inf")
    no_improve = 0
    version    = _next_version(MODELS_DIR)

    print(f"\nTraining (will save as ghost_v{version:03d}.pt) ...\n")
    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.monotonic()

        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for batch in train_dl:
            frames = batch["frame"].to(device)
            pred   = model(frames)
            loss   = _loss(pred, batch, device)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            train_loss += loss.item() * len(frames)
        train_loss /= n_train

        # ── Validate ───────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dl:
                frames = batch["frame"].to(device)
                pred   = model(frames)
                val_loss += _loss(pred, batch, device).item() * len(frames)
        val_loss /= n_val

        elapsed = time.monotonic() - t0
        print(f"  epoch {epoch:3d}/{MAX_EPOCHS}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}  "
              f"({elapsed:.0f}s)")

        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val   = val_loss
            no_improve = 0
            ckpt_path  = MODELS_DIR / f"ghost_v{version:03d}.pt"
            torch.save(model.state_dict(), ckpt_path)
            # Update latest symlink (or copy on Windows where symlinks need elevation)
            try:
                if latest_ckpt.exists() or latest_ckpt.is_symlink():
                    latest_ckpt.unlink()
                latest_ckpt.symlink_to(ckpt_path.name)
            except (OSError, NotImplementedError):
                import shutil
                shutil.copy2(ckpt_path, latest_ckpt)
            print(f"    ✓ saved best checkpoint (val={best_val:.4f})")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"\n  Early stopping after {epoch} epochs (no improvement for {PATIENCE}).")
                break

    mark_sessions_used(DATA_DIR)
    print(f"\nDone. Best val loss: {best_val:.4f}")
    print(f"Checkpoint: {MODELS_DIR / f'ghost_v{version:03d}.pt'}")
    print(f"Latest:     {latest_ckpt}")


if __name__ == "__main__":
    main()
