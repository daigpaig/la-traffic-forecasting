"""METR-LA download, tensor extraction, chronological split, and sliding windows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset

# ``torch_geometric.datasets.METR_LA`` is not available in released PyG as of 2.7;
# fall back to the project-local implementation (see ``program.md``).
try:
    from torch_geometric.datasets import METR_LA  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover
    from .metr_la_dataset import METR_LA


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "METR_LA"


def default_metr_la_root() -> str:
    """Directory passed to :class:`METR_LA` (``data/METR_LA`` under the repo)."""
    return str(DEFAULT_DATA_ROOT)


def chronological_split_lengths(num_timesteps: int) -> Tuple[int, int, int]:
    """70% / 10% / 20% split along time (standard METR-LA practice)."""
    n_train = int(0.7 * num_timesteps)
    n_val = int(0.1 * num_timesteps)
    n_test = num_timesteps - n_train - n_val
    if min(n_train, n_val, n_test) <= 0:
        raise ValueError(f"Invalid split for T={num_timesteps}")
    return n_train, n_val, n_test


def load_metr_la_traffic(
    root: Optional[str] = None,
    *,
    force_reload: bool = False,
) -> torch.Tensor:
    """Download (if needed) and return speeds ``[T, 207, 1]`` float32."""
    root = root or default_metr_la_root()
    ds = METR_LA(root=root, force_reload=force_reload)
    return ds[0].traffic


def train_mean_std(traffic: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-sensor mean/std over the training time range only."""
    n_train, _, _ = chronological_split_lengths(traffic.shape[0])
    train = traffic[:n_train]
    mean = train.mean(dim=0, keepdim=True)
    std = train.std(dim=0, keepdim=True).clamp_min(1e-3)
    return mean, std


@dataclass
class SplitBounds:
    train: Tuple[int, int]
    val: Tuple[int, int]
    test: Tuple[int, int]


def split_time_bounds(num_timesteps: int) -> SplitBounds:
    n_train, n_val, _ = chronological_split_lengths(num_timesteps)
    return SplitBounds(
        train=(0, n_train),
        val=(n_train, n_train + n_val),
        test=(n_train + n_val, num_timesteps),
    )


class TrafficWindowDataset(Dataset):
    """Sliding windows inside ``[start, end)`` without crossing split boundaries."""

    def __init__(
        self,
        series: torch.Tensor,
        start: int,
        end: int,
        input_len: int = 12,
        output_len: int = 12,
    ) -> None:
        if end - start < input_len + output_len:
            raise ValueError("Segment too short for the requested horizons.")
        self.series = series
        self.start = start
        self.end = end
        self.input_len = input_len
        self.output_len = output_len
        self._len = (end - start) - input_len - output_len + 1

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        t0 = self.start + idx
        x = self.series[t0 : t0 + self.input_len, :, 0]
        y = self.series[
            t0 + self.input_len : t0 + self.input_len + self.output_len, :, 0
        ]
        return x, y
