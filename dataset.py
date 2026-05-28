from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from model import IMAGENET_MEAN, IMAGENET_STD


class SessionDataset(Dataset):
    """Loads one HDF5 session file.

    Each item is a dict:
        frame       FloatTensor (3, 224, 224)  normalised to ImageNet stats
        left_stick  FloatTensor (2,)
        right_stick FloatTensor (2,)
        triggers    FloatTensor (2,)
        buttons     FloatTensor (14,)
        weight      float  (per-player sample weight)
    """

    def __init__(self, h5_path: Path, weight: float = 1.0) -> None:
        self.h5_path = Path(h5_path)
        self.weight = weight

        # Load entire session into RAM once — avoids opening the HDF5 file
        # per sample which caused ~200s/epoch due to file open overhead.
        print(f"    loading {Path(h5_path).name} into RAM...", end=" ", flush=True)
        with h5py.File(self.h5_path, "r") as f:
            self._frames      = f["frames"][:]       # (N, 224, 224, 3) uint8
            self._left_stick  = f["left_stick"][:]   # (N, 2) float32
            self._right_stick = f["right_stick"][:]  # (N, 2) float32
            self._triggers    = f["triggers"][:]     # (N, 2) float32
            self._buttons     = f["buttons"][:]      # (N, 14) float32
        print(f"done ({len(self._frames)} frames)")

        mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
        std  = torch.tensor(IMAGENET_STD,  dtype=torch.float32).view(3, 1, 1)
        self._mean = mean
        self._std  = std

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, idx: int) -> dict:
        frame_t = torch.from_numpy(self._frames[idx].copy()).permute(2, 0, 1).float() / 255.0
        frame_t = (frame_t - self._mean) / self._std

        return {
            "frame":       frame_t,
            "left_stick":  torch.from_numpy(self._left_stick[idx].copy()),
            "right_stick": torch.from_numpy(self._right_stick[idx].copy()),
            "triggers":    torch.from_numpy(self._triggers[idx].copy()),
            "buttons":     torch.from_numpy(buttons.copy()),
            "weight":      self.weight,
        }


def load_all_sessions(
    data_dir: Path,
    my_player: str = "me",
    my_weight: float = 1.0,
) -> torch.utils.data.ConcatDataset:
    """Load every session_*.h5 in data_dir into a combined dataset.

    Sessions tagged with player == my_player get weight=my_weight; all
    others get weight=1.0.
    """
    data_dir = Path(data_dir)
    h5_files = sorted(data_dir.glob("session_*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No session_*.h5 files found in {data_dir}")

    datasets = []
    for h5_path in h5_files:
        meta_path = h5_path.with_suffix(".json")
        player = "me"
        if meta_path.exists():
            try:
                player = json.loads(meta_path.read_text()).get("player", "me")
            except Exception:
                pass
        w = my_weight if player == my_player else 1.0
        datasets.append(SessionDataset(h5_path, weight=w))
        print(f"  loaded {h5_path.name}  frames={len(datasets[-1])}  player={player!r}  weight={w}")

    return torch.utils.data.ConcatDataset(datasets)


def mark_sessions_used(data_dir: Path) -> None:
    """Flip used_in_training=true in every session metadata file."""
    for meta_path in sorted(Path(data_dir).glob("session_*.json")):
        try:
            data = json.loads(meta_path.read_text())
            data["used_in_training"] = True
            meta_path.write_text(json.dumps(data, indent=2))
        except Exception as exc:
            print(f"  warning: could not update {meta_path.name}: {exc}")
