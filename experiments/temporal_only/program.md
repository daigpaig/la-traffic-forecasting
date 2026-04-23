# Temporal-only LSTM baseline (iteration 1)

## Objective

Establish a purely temporal LSTM baseline on METR-LA so later spatial models can be compared against a simple error floor.

## Data

- **Source**: METR-LA speeds in a PyTorch Geometric–style layout. The symbol `torch_geometric.datasets.METR_LA` is not present in current PyG releases (the upstream dataset PR was never merged). This repo uses `shared/metr_la_dataset.py`, and `shared/data_loader.py` imports upstream `METR_LA` when it exists, otherwise the local class.
- **Storage**: `shared/data_loader.default_metr_la_root()` → `data/METR_LA/` (raw zip from the standard Switch mirror, then `metr_la.h5`).
- **Tensor**: `[T, 207, 1]` with `T = 34272`.
- **Split**: Chronological 70% / 10% / 20% train / validation / test along time. Sliding windows are generated **inside** each segment so no window crosses a split boundary.
- **Normalization**: Per-sensor z-score using mean and standard deviation from the **training** segment only; the same mean/std is applied to validation and test.

## Model

- **File**: `experiments/temporal_only/train.py`
- **Architecture**: One vanilla `nn.LSTM` (207 input features per step, i.e. sensors treated as independent channels), then a linear head mapping the last hidden state to the next 12 steps × 207 speeds.
- **Horizon**: Input length 12 (1 hour), output length 12.

## Training (this iteration)

- **Epochs**: 5 (quick smoke run).
- **Loss**: MSE in normalized space; **reported test metrics** are RMSE and MAE after denormalizing predictions and targets to the original speed scale (see `shared/evaluation.py`).

## Results

- **Final test RMSE (original speed units)**: **16.509** (from `logs/iteration_1.txt` after a 5-epoch run on CPU with default hyperparameters: hidden 64, 2 LSTM layers, Adam `lr=1e-3`, batch size 64).
- **Final test MAE**: **9.328** (same run).
