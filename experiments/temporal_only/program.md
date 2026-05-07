# Program v1: Temporal-only baseline and Week 3 plan

## Goal

Establish a reliable temporal-only baseline on METR-LA and use early dry runs to decide the first set of hyperparameters before introducing spatial structure.

## Problem setup

- **Dataset**: METR-LA traffic speeds in `shared/metr_la_dataset.py`.
- **Tensor shape**: `[T, 207, 1]`, where `T=34272` at 5-minute resolution.
- **Split**: chronological 70%/10%/20% train/val/test, with split-safe windowing in `TrafficWindowDataset`.
- **Normalization**: per-sensor z-score from train split statistics only.
- **Prediction task**: 12-step input (1 hour) to 12-step output (1 hour ahead sequence).

## Baseline model (temporal only)

- **Code**: `experiments/temporal_only/train.py`
- **Architecture**: LSTM encoder over sensor features + linear head to multi-step forecast.
- **Optimization**: Adam with MSE loss in normalized space.
- **Reported metrics**: denormalized test RMSE, MAE, R2, and runtime (seconds).

## Dry-run experiments (Week 3)

All dry runs used CPU, 3 epochs, `batch_size=64`, `lr=1e-3`, seed 0.

1. `iteration_2`: hidden 64, layers 2, dropout 0.0
2. `iteration_3`: hidden 128, layers 2, dropout 0.0
3. `iteration_4`: hidden 64, layers 1, dropout 0.0
4. `iteration_5`: hidden 64, layers 2, dropout 0.2

### Results snapshot

- **Best RMSE**: `iteration_4` with RMSE **15.000**, MAE **8.912**, R2 **0.5669**, runtime **4.99s**
- **Worst RMSE**: `iteration_2` with RMSE **16.947**
- **Fastest run**: `iteration_4` at **4.99s**
- **Slowest run**: `iteration_3` at **14.13s**

## Interpretation and next steps

- Increasing hidden size from 64 to 128 increased runtime substantially but did not improve RMSE enough to justify cost at this stage.
- A simpler 1-layer LSTM performed best in this short-run regime, suggesting the deeper stack may be over-parameterized for quick training.
- Dropout at 0.2 did not help in 3-epoch tests; revisit with longer training before final conclusion.
- **Program v1 decision**: carry forward `hidden=64`, `layers=1`, `dropout=0.0` for the next longer baseline run (for example, 15-30 epochs with early stopping).
