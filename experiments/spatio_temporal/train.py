"""ST-GCN: Spatio-Temporal Graph Convolutional Network for traffic forecasting."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.nn import GCNConv, ChebConv

from shared.data_loader import (
    TrafficWindowDataset,
    load_metr_la_traffic,
    split_time_bounds,
    train_mean_std,
)
from shared.evaluation import mae, r2_score, rmse
from shared.spatial_utils import build_adjacency


class STGCN(nn.Module):
    """Spatio-Temporal Graph Convolutional Network.

    Architecture:
    - Per-node LSTM: each sensor's time series processed independently
    - GCN with residual skip connections for spatial aggregation
    - Linear readout to multi-step forecast
    """

    def __init__(
        self,
        num_nodes: int = 207,
        in_channels: int = 12,
        hidden_size: int = 64,
        out_channels: int = 12,
        graph_conv: str = "gcn",
        num_graph_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.out_channels = out_channels

        # Per-node temporal encoder: input is 1 feature (speed) per timestep
        self.node_lstm = nn.LSTM(1, hidden_size, batch_first=True)

        # Graph convolution layers
        self.graph_layers = nn.ModuleList()
        for _ in range(num_graph_layers):
            if graph_conv == "gcn":
                self.graph_layers.append(GCNConv(hidden_size, hidden_size))
            elif graph_conv == "cheb":
                self.graph_layers.append(ChebConv(hidden_size, hidden_size, K=3))
            else:
                raise ValueError(f"Unknown graph_conv: {graph_conv}")

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # Output projection
        self.readout = nn.Linear(hidden_size, out_channels)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, T_in, N) input sequence
            edge_index: (2, E) edge indices
            edge_weight: (E,) edge weights

        Returns:
            (B, T_out, N) output forecast
        """
        B, T_in, N = x.shape

        # Per-node temporal encoding: each sensor processed independently
        # (B, T, N) -> (B*N, T, 1)
        x = x.permute(0, 2, 1).reshape(B * N, T_in, 1)
        x, _ = self.node_lstm(x)   # (B*N, T, hidden)
        x = x[:, -1, :]            # (B*N, hidden) — final hidden state per node

        # Expand edge_index for batch
        batch_edge_index = []
        batch_edge_weight = []
        for b in range(B):
            offset = b * N
            batch_edge_index.append(edge_index + offset)
            if edge_weight is not None:
                batch_edge_weight.append(edge_weight)

        batch_edge_index = torch.cat(batch_edge_index, dim=1)
        if edge_weight is not None:
            batch_edge_weight = torch.cat(batch_edge_weight, dim=0)
        else:
            batch_edge_weight = None

        # Graph convolution with per-layer skip connections (prevent over-smoothing)
        for i, layer in enumerate(self.graph_layers):
            residual = x
            x = layer(x, batch_edge_index, batch_edge_weight)
            x = self.relu(x)
            x = x + residual   # Residual skip: preserve per-node identity
            if i < len(self.graph_layers) - 1:
                x = self.dropout(x)

        # Reshape back: (B*N, hidden) -> (B, N, hidden)
        x = x.reshape(B, N, -1)

        # Output projection: (B, N, hidden) -> (B, N, T_out)
        x = self.readout(x)

        return x.transpose(1, 2)  # (B, T_out, N)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> float:
    training = optimizer is not None
    model.train(training)
    total = 0.0
    n = 0

    # Move graph structure to device once
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device) if edge_weight is not None else None

    for xb, yb in loader:
        xb = xb.to(device)  # (B, T, N)
        yb = yb.to(device)  # (B, T, N)

        pred = model(xb, edge_index, edge_weight)
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
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
) -> tuple[float, float, float]:
    model.eval()
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []

    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device) if edge_weight is not None else None

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb, edge_index, edge_weight)

        # Denormalize
        preds.append(pred * std[:, :, 0] + mean[:, :, 0])
        targets.append(yb * std[:, :, 0] + mean[:, :, 0])

    pred = torch.cat(preds, dim=0)
    target = torch.cat(targets, dim=0)
    return rmse(pred, target), mae(pred, target), r2_score(pred, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--graph-conv", type=str, default="gcn", choices=["gcn", "cheb"])
    parser.add_argument("--graph-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--adj-type", type=str, default="physical", choices=["physical", "correlation"])
    parser.add_argument("--k-neighbors", type=int, default=10, help="K nearest neighbors")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", type=str, default="spatial_iter_1")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data (same as temporal baseline)
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

    # Build spatial adjacency
    print(f"Building {args.adj_type} adjacency with K={args.k_neighbors}...")
    edge_index, edge_weight = build_adjacency(
        adj_type=args.adj_type,
        k=args.k_neighbors,
        traffic=traffic,  # For correlation
    )
    print(f"Graph: {edge_index.shape[1]} edges, {207} nodes")

    # Initialize model
    model = STGCN(
        num_nodes=207,
        in_channels=in_len,
        hidden_size=args.hidden,
        out_channels=out_len,
        graph_conv=args.graph_conv,
        num_graph_layers=args.graph_layers,
        dropout=args.dropout,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.MSELoss()

    start_time = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, edge_index, edge_weight, crit, opt, device)
        val_loss = run_epoch(model, val_loader, edge_index, edge_weight, crit, None, device)
        print(f"epoch {epoch:02d}  train_mse {train_loss:.6f}  val_mse {val_loss:.6f}")

    elapsed_sec = time.perf_counter() - start_time

    test_rmse, test_mae, test_r2 = evaluate_denormalized(
        model, test_loader, edge_index, edge_weight, mean.to(device), std.to(device), device
    )
    print(
        f"test_rmse {test_rmse:.6f}  test_mae {test_mae:.6f}  "
        f"test_r2 {test_r2:.6f}  runtime_sec {elapsed_sec:.2f}"
    )

    # Log results
    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{args.run_name}.txt"
    log_path.write_text(
        (
            f"test_rmse={test_rmse:.6f}\n"
            f"test_mae={test_mae:.6f}\n"
            f"test_r2={test_r2:.6f}\n"
            f"runtime_sec={elapsed_sec:.2f}\n"
            f"epochs={args.epochs}\n"
            f"batch_size={args.batch_size}\n"
            f"lr={args.lr}\n"
            f"hidden={args.hidden}\n"
            f"graph_conv={args.graph_conv}\n"
            f"graph_layers={args.graph_layers}\n"
            f"dropout={args.dropout}\n"
            f"adj_type={args.adj_type}\n"
            f"k_neighbors={args.k_neighbors}\n"
            f"seed={args.seed}\n"
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
