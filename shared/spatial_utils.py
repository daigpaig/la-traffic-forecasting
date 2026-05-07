"""Spatial utilities for graph construction on METR-LA."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch

from .data_loader import DEFAULT_DATA_ROOT


def load_distance_matrix() -> np.ndarray:
    """Load the physical distance matrix (207 x 207) from METR-LA."""
    csv_path = DEFAULT_DATA_ROOT / "raw" / "distances_la.csv"
    sensor_ids_path = DEFAULT_DATA_ROOT / "raw" / "sensor_ids_la.txt"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Distance matrix not found at {csv_path}. "
            "Run a training script first to download METR-LA data."
        )

    # Load the 207 sensor IDs used in METR-LA
    sensor_ids_text = sensor_ids_path.read_text().strip()
    metr_sensor_ids = [int(sid) for sid in sensor_ids_text.split(',')]
    assert len(metr_sensor_ids) == 207, f"Expected 207 sensors, got {len(metr_sensor_ids)}"

    # Create ID to index mapping
    id_to_idx = {sensor_id: idx for idx, sensor_id in enumerate(metr_sensor_ids)}

    # Load distance matrix
    df = pd.read_csv(csv_path)

    # Build 207x207 matrix (only for METR-LA sensors)
    dist_matrix = np.full((207, 207), fill_value=np.inf, dtype=np.float32)
    np.fill_diagonal(dist_matrix, 0.0)

    for _, row in df.iterrows():
        from_id = int(row['from'])
        to_id = int(row['to'])

        # Only include edges between METR-LA sensors
        if from_id in id_to_idx and to_id in id_to_idx:
            i = id_to_idx[from_id]
            j = id_to_idx[to_id]
            d = float(row['cost'])
            dist_matrix[i, j] = d

    return dist_matrix


def physical_adjacency(
    dist_matrix: np.ndarray,
    k: int = 10,
    include_self: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build edge_index and edge_weights from K-nearest physical neighbors.

    Args:
        dist_matrix: (N, N) distance matrix in meters
        k: Number of nearest neighbors per node
        include_self: If True, add self-loops

    Returns:
        edge_index: (2, E) tensor of edges
        edge_weight: (E,) tensor of Gaussian-weighted edges
    """
    num_nodes = dist_matrix.shape[0]
    edges = []
    weights = []

    sigma = np.median(dist_matrix[dist_matrix < np.inf])

    for node in range(num_nodes):
        distances = dist_matrix[node]
        # Get K nearest neighbors (excluding self if distance is 0)
        nearest = np.argsort(distances)[:k+1]

        for neighbor in nearest:
            if neighbor == node and not include_self:
                continue
            dist = distances[neighbor]
            if np.isinf(dist):
                continue

            edges.append([node, neighbor])
            weight = np.exp(-dist**2 / (sigma**2))
            weights.append(weight)

    if len(edges) == 0:
        # Fallback: full graph
        edges = [[i, j] for i in range(num_nodes) for j in range(num_nodes)]
        weights = [1.0] * len(edges)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(weights, dtype=torch.float32)

    return edge_index, edge_weight


def correlation_adjacency(
    traffic: torch.Tensor,
    k: int = 10,
    include_self: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build edge_index and edge_weights from temporal correlation.

    Args:
        traffic: (T, N, 1) traffic tensor
        k: Number of most correlated neighbors per node
        include_self: If True, add self-loops

    Returns:
        edge_index: (2, E) tensor of edges
        edge_weight: (E,) tensor of correlation weights
    """
    # Compute pairwise Pearson correlation
    series = traffic[:, :, 0].numpy()  # (T, N)
    corr_matrix = np.corrcoef(series.T)  # (N, N)

    # Replace NaN with 0
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    num_nodes = corr_matrix.shape[0]
    edges = []
    weights = []

    for node in range(num_nodes):
        correlations = corr_matrix[node]
        # Get K most correlated neighbors
        # Use absolute correlation to capture both positive and negative
        abs_corr = np.abs(correlations)
        nearest = np.argsort(-abs_corr)[:k+1]

        for neighbor in nearest:
            if neighbor == node and not include_self:
                continue
            corr = correlations[neighbor]

            edges.append([node, neighbor])
            # Use absolute correlation as weight
            weights.append(abs(corr))

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(weights, dtype=torch.float32)

    return edge_index, edge_weight


def build_adjacency(
    adj_type: str,
    k: int = 10,
    traffic: torch.Tensor = None,
    dist_matrix: np.ndarray = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Unified adjacency builder.

    Args:
        adj_type: "physical" or "correlation"
        k: Number of neighbors per node
        traffic: Required if adj_type == "correlation"
        dist_matrix: Required if adj_type == "physical"

    Returns:
        edge_index: (2, E)
        edge_weight: (E,)
    """
    if adj_type == "physical":
        if dist_matrix is None:
            dist_matrix = load_distance_matrix()
        return physical_adjacency(dist_matrix, k=k)

    elif adj_type == "correlation":
        if traffic is None:
            raise ValueError("traffic tensor required for correlation adjacency")
        return correlation_adjacency(traffic, k=k)

    else:
        raise ValueError(f"Unknown adjacency type: {adj_type}")
