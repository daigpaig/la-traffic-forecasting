"""Regression metrics for traffic forecasting.

METR-LA benchmark convention: pass ``null_val=0.0`` to mask out zero-speed
readings (missing/erroneous sensor data) when comparing to published results
(DCRNN, Graph WaveNet, STGCN).  The default ``null_val=None`` preserves the
original unmasked behaviour for backward compatibility.
"""

from __future__ import annotations

import torch


def _apply_mask(
    pred: torch.Tensor,
    target: torch.Tensor,
    null_val: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if null_val is None:
        return pred, target
    mask = target != null_val
    return pred[mask], target[mask]


def rmse(
    pred: torch.Tensor,
    target: torch.Tensor,
    null_val: float | None = None,
) -> float:
    """Root mean squared error.

    Args:
        null_val: If provided, exclude target positions equal to this value
                  before computing the metric (METR-LA standard: ``0.0``).
    """
    pred, target = _apply_mask(pred, target, null_val)
    return float(torch.sqrt(torch.mean((pred - target) ** 2)))


def mae(
    pred: torch.Tensor,
    target: torch.Tensor,
    null_val: float | None = None,
) -> float:
    """Mean absolute error.

    Args:
        null_val: If provided, exclude target positions equal to this value
                  before computing the metric (METR-LA standard: ``0.0``).
    """
    pred, target = _apply_mask(pred, target, null_val)
    return float(torch.mean(torch.abs(pred - target)))


def mape(
    pred: torch.Tensor,
    target: torch.Tensor,
    null_val: float = 0.0,
) -> float:
    """Mean absolute percentage error (always masks ``null_val`` to avoid ÷0).

    Args:
        null_val: Positions in ``target`` equal to this value are excluded
                  (default ``0.0``, which is the METR-LA standard).
    """
    pred, target = _apply_mask(pred, target, null_val)
    return float(torch.mean(torch.abs((pred - target) / target))) * 100.0


def r2_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Coefficient of determination."""
    ss_res = torch.sum((target - pred) ** 2)
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    if ss_tot <= 1e-12:
        return 0.0
    return float(1.0 - (ss_res / ss_tot))
