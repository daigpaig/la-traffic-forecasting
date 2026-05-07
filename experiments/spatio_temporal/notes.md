# Research Agent Notes: Spatio-Temporal GNN

## Mission

Beat temporal baseline RMSE 15.000 using Graph Neural Networks.

---

## Iteration 1: Baseline ST-GCN [FAILED]

**Date**: 2026-05-07
**Status**: ❌ FAILED

### Hypothesis
> "A standard ST-GCN with physical adjacency (K=5) and shallow network (1 GCN layer) should reduce spatial blurring compared to deeper networks."

### Rationale
- Previous experiments (K=10, 2 layers) achieved RMSE 17.6
- Hypothesis: Reducing K and depth will preserve node identity
- Expected: RMSE ~16.0 (improvement toward 15.0 target)

### Configuration
```python
Architecture: LSTM (temporal) + GCN (spatial) + Linear readout
Hidden size: 64
Graph layers: 1
Dropout: 0.1
Adjacency: physical distance, K=5 nearest neighbors
Learning rate: 1e-3
Epochs: 15
Batch size: 64
```

### Results
- **Test RMSE**: 18.755 mph
- **Test MAE**: 11.872 mph
- **Test R²**: 0.323
- **Runtime**: 139.43s

### Analysis: Why It Failed

**Problem 1: Architecture Flaw**
The current LSTM encoder processes the ENTIRE spatiotemporal sequence `(B, T, N)` as if N were features, not nodes:

```python
# Current (WRONG):
x, _ = self.temporal_lstm(x)  # (B, T_in, N) -> (B, T_in, hidden)
x = x[:, -1, :]  # Take last timestep: (B, hidden)
x = x.unsqueeze(1).expand(B, N, -1)  # Broadcast to all nodes!
```

**This loses all per-node information!** Every sensor gets the same temporal embedding.

**Problem 2: Spatial Blurring Still Present**
Even with K=5, graph convolution aggregates neighbors, causing over-smoothing. The model can't distinguish between sensors because they all start with identical temporal features.

**Problem 3: No Skip Connections**
Pure graph convolution with no residual connections forces information through smoothing layers without preserving original node features.

### Conclusion
The architecture is fundamentally broken. Broadcasting temporal embeddings destroys the spatial structure we're trying to leverage.

**Gap to target**: +3.755 RMSE (need to improve by 20% to beat temporal baseline)

---

## Iteration 2: Per-Node LSTM + GCN Skip Connection [AGENT HYPOTHESIS]

**Date**: 2026-05-07
**Status**: 🔄 IN PROGRESS

### Agent Hypothesis
> "Process each node's temporal sequence INDEPENDENTLY before applying graph convolution. Reshape input from (B, T, N) → (B*N, T, 1) so the LSTM sees one sensor at a time, preserving each sensor's unique rush-hour signature. Add a residual skip connection around each GCN layer to prevent over-smoothing."

### Agent Reasoning

Traffic sensors on the LA highway network have **location-specific temporal signatures** — sensor 42 near an on-ramp sees congestion building 10 minutes before sensor 87 on the open freeway. If we collapse all 207 sensors into a single LSTM as "features", we destroy this timing diversity: the LSTM must simultaneously model all 207 sensors' patterns through a single hidden state of size 64, which is massively underdetermined.

The spatial graph is only useful if each node *brings something unique* to the message-passing step. Broadcasting one shared embedding to every node makes the GCN pointless — all nodes start at the same value and remain nearly identical after aggregation. Skip connections ensure that even if GCN slightly mis-aggregates, the original per-node signal survives through the residual path.

**Why K=5 physical adjacency is the right starting point**: Traffic propagation on roads is local and directional. Each sensor's most relevant neighbors are the 4-6 sensors immediately upstream/downstream. K=5 limits blurring while still capturing the most informative spatial context.

### Original Analysis (confirmed)

### Key Insight
Traffic sensors have UNIQUE temporal signatures. We must:
1. **First**: Extract per-node temporal features independently
2. **Then**: Apply graph convolution for spatial context
3. **Finally**: Combine via residual/skip connections

### Proposed Architecture Change

**Old (Broken)**:
```
Input (B, T, N)
  → LSTM over T (treats N as features)
  → Broadcast to all nodes
  → GCN
  → Output
```

**New (Fixed)**:
```
Input (B, T, N)
  → Reshape to (B*N, T, 1)  [treat each node independently]
  → Per-node LSTM: (B*N, T, 1) → (B*N, hidden)
  → Reshape to (B, N, hidden)
  → GCN for spatial aggregation
  → Skip connection: output = GCN(x) + x
  → Output
```

### Implementation Plan

Modify `train.py` STGCN class:

```python
class STGCN(nn.Module):
    def __init__(self, ...):
        # Per-node temporal encoder
        self.node_lstm = nn.LSTM(1, hidden_size, batch_first=True)  # Input: 1 feature per node

        # Spatial layers with skip connections
        self.gcn1 = GCNConv(hidden_size, hidden_size)
        self.gcn2 = GCNConv(hidden_size, hidden_size)  # Optional second layer

        # Output projection
        self.readout = nn.Linear(hidden_size, out_channels)

    def forward(self, x, edge_index, edge_weight):
        B, T, N = x.shape

        # Process each node's time series independently
        x = x.permute(0, 2, 1).reshape(B * N, T, 1)  # (B*N, T, 1)
        x, _ = self.node_lstm(x)  # (B*N, T, hidden)
        x = x[:, -1, :]  # Take final timestep: (B*N, hidden)

        # Reshape for graph conv
        x = x.reshape(B, N, -1)  # (B, N, hidden)

        # Flatten for PyG
        x_flat = x.reshape(B * N, -1)  # (B*N, hidden)
        identity = x_flat  # Save for skip connection

        # Graph convolution with skip
        x_flat = self.gcn1(x_flat, batch_edge_index, batch_edge_weight)
        x_flat = F.relu(x_flat)
        x_flat = x_flat + identity  # Skip connection!

        # Reshape and output
        x = x_flat.reshape(B, N, -1)
        x = self.readout(x)  # (B, N, out_channels)
        return x.transpose(1, 2)  # (B, out_channels, N)
```

### Expected Outcome
- **Target**: RMSE < 17.0 (at minimum, improve from 18.755)
- **Best case**: RMSE ~15.5 (close to temporal baseline)
- **Reasoning**: Per-node processing preserves temporal patterns, skip connections prevent over-smoothing

### Configuration
```python
Architecture: Per-node LSTM + GCN with skip connections
Hidden size: 64
Graph layers: 1 (with skip connection)
Dropout: 0.0 (skip connections provide regularization)
Adjacency: physical distance, K=5
Learning rate: 1e-3
Epochs: 10 (5-minute budget)
Batch size: 64
```

### Results

| Metric | Value |
|---|---|
| **Test RMSE** | **12.083 mph** ✅ |
| Test MAE | 6.236 mph |
| Test R² | 0.719 |
| Runtime | 1149.34s (~19 min) |

**Beat temporal baseline (15.000) by 2.917 RMSE points. R² improved from 0.323 → 0.719.**

### Analysis: Why It Worked

Fixing per-node processing was the single most impactful change. The architecture now correctly models 207 independent temporal signals before spatial aggregation — the GCN can actually do its job because every node brings a distinct embedding. The residual skip prevents the GCN from over-smoothing, keeping each node's learned pattern intact while still incorporating neighbor context.

### Iteration 3 Candidates

- **Graph Attention (GAT)**: now that per-node features are meaningful, learned edge weights may further improve spatial selection
- **Correlation adjacency**: traffic speed correlations may better reflect actual road influence than raw physical distance
- **Deeper temporal encoder**: 2-layer LSTM per node (still with skip in GCN)
- **Longer training**: val_mse was still trending down at epoch 10 — more epochs may squeeze more out

---

## Research Philosophy

Following Karpathy's principle: **one file, one metric, fast iterations**.

Each experiment tests ONE hypothesis with clear expected outcomes. When something fails, analyze WHY before proposing the next fix.

The goal is not to throw hyperparameters at the wall—it's to build understanding of what works and why.

---

**Next Action**: Implement Iteration 2 architecture in `train.py` and run training.
