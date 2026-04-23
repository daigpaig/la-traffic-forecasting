"""Regression metrics for traffic forecasting."""

from __future__ import annotations

import torch


def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Root mean squared error (same shape tensors)."""
    return float(torch.sqrt(torch.mean((pred - target) ** 2)))


def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean absolute error."""
    return float(torch.mean(torch.abs(pred - target)))
