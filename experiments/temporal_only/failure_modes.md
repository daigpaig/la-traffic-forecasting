# Failure mode list (temporal-only baseline)

- **Environment mismatch**: missing `python` alias or missing packages causes runs to fail before training starts.
- **Dataset download fragility**: remote host downtime or network issues can break first-time METR-LA fetch.
- **Split leakage risk**: any future refactor that allows windows to cross train/val/test boundaries will inflate metrics.
- **Normalization leakage**: using full-dataset mean/std instead of train-only stats will overstate generalization.
- **Undertraining misread as model weakness**: low epoch count can make good architectures look poor.
- **Overparameterization on quick runs**: deeper/wider LSTMs may underperform when not trained long enough.
- **Single-seed instability**: apparent improvements may disappear with a different random seed.
- **Metric mismatch**: optimizing MSE while presenting RMSE/MAE/R2 can hide trade-offs without a consistent selection rule.
- **Runtime blind spot**: choosing best RMSE without runtime tracking can lead to impractical model choices.
- **Non-stationarity**: temporal-only models struggle when traffic regimes shift abruptly (holidays, incidents, unusual weather).
