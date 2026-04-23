"""Vanilla LSTM baseline: 12-step lookback -> 12-step horizon, 207 independent sensor features."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from shared.data_loader import (
    TrafficWindowDataset,
    load_metr_la_traffic,
    split_time_bounds,
    train_mean_std,
)
from shared.evaluation import mae, rmse


class VanillaLSTM(nn.Module):
    """Single-stack LSTM + linear readout for multi-step prediction."""

    def __init__(
        self,
        in_features: int = 207,
        hidden_size: int = 64,
        num_layers: int = 2,
        horizon: int = 12,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.in_features = in_features
        self.lstm = nn.LSTM(
            in_features,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, horizon * in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        flat = self.head(last)
        return flat.view(-1, self.horizon, self.in_features)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> float:
    training = optimizer is not None
    model.train(training)
    total = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb)
        loss = criterion(pred, yb)
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total += loss.item() * xb.size(0)
        n += xb.size(0)
    return total / max(n, 1)


@torch.no_grad()
def evaluate_denormalized(
    model: nn.Module,
    loader: DataLoader,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb)
        preds.append(pred * std[:, :, 0] + mean[:, :, 0])
        targets.append(yb * std[:, :, 0] + mean[:, :, 0])
    pred = torch.cat(preds, dim=0)
    target = torch.cat(targets, dim=0)
    return rmse(pred, target), mae(pred, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    traffic = load_metr_la_traffic()
    mean, std = train_mean_std(traffic)
    traffic_norm = (traffic - mean) / std

    bounds = split_time_bounds(traffic.shape[0])
    in_len = out_len = 12

    train_ds = TrafficWindowDataset(
        traffic_norm, *bounds.train, input_len=in_len, output_len=out_len
    )
    val_ds = TrafficWindowDataset(
        traffic_norm, *bounds.val, input_len=in_len, output_len=out_len
    )
    test_ds = TrafficWindowDataset(
        traffic_norm, *bounds.test, input_len=in_len, output_len=out_len
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = VanillaLSTM(
        in_features=207,
        hidden_size=args.hidden,
        num_layers=args.layers,
        horizon=out_len,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, crit, opt, device)
        val_loss = run_epoch(model, val_loader, crit, None, device)
        print(f"epoch {epoch:02d}  train_mse {train_loss:.6f}  val_mse {val_loss:.6f}")

    test_rmse, test_mae = evaluate_denormalized(
        model, test_loader, mean.to(device), std.to(device), device
    )
    print(f"test_rmse {test_rmse:.6f}  test_mae {test_mae:.6f}")

    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "iteration_1.txt"
    log_path.write_text(
        f"test_rmse={test_rmse:.6f}\ntest_mae={test_mae:.6f}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
