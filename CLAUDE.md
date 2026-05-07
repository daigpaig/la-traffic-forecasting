# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a STAT390 project focused on spatiotemporal traffic forecasting using the METR-LA dataset. The project follows an iterative research approach, starting with temporal-only baselines before introducing spatial structure.

**Dataset**: METR-LA traffic speeds at 5-minute resolution
- Shape: `[34272, 207, 1]` (timesteps, sensors, features)
- Chronological split: 70% train / 10% val / 20% test
- Task: 12-step input (1 hour) → 12-step output (1 hour ahead forecast)

## Common Commands

### Environment Setup
```bash
# Activate virtual environment (assumed to exist in .venv)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running Experiments

**Single training run:**
```bash
python experiments/temporal_only/train.py \
  --epochs 5 \
  --batch-size 64 \
  --lr 1e-3 \
  --hidden 64 \
  --layers 1 \
  --dropout 0.0 \
  --seed 0 \
  --run-name my_experiment
```

**Manual hyperparameter sweep:**
```bash
# Run from repo root
python experiments/temporal_only/autoresearch_loop.py --epochs 3 --seed 0
```

This executes multiple training runs sequentially with predefined hyperparameter configurations. Results are logged to `experiments/temporal_only/logs/<run-name>.txt`.

**Autonomous research loop (Karpathy-style):**
```bash
# Run autonomous optimization with 10 iterations, stop if RMSE reaches 14.0
python experiments/temporal_only/autonomous_research.py --max-iterations 10 --rmse-target 14.0
```

This implements true autonomous research:
- Reads previous logs and analyzes performance patterns
- Generates hypotheses for next experiments with reasoning
- Runs training automatically
- Commits to git only if RMSE improves
- Repeats until target reached or max iterations

See `experiments/temporal_only/AUTONOMOUS_README.md` for full documentation.

## Code Architecture

### Data Pipeline (`shared/`)

**`data_loader.py`** - Core data loading and splitting logic:
- `load_metr_la_traffic()`: Downloads and returns raw traffic tensor
- `train_mean_std()`: Computes per-sensor z-score statistics from training split only
- `split_time_bounds()`: Returns chronological 70/10/20 split boundaries
- `TrafficWindowDataset`: PyTorch Dataset that creates sliding windows within split boundaries (no leakage)

**`metr_la_dataset.py`** - Custom PyG-compatible dataset:
- Project-local implementation of METR-LA since torch_geometric doesn't include it in released versions
- Downloads from switch.ch mirror, processes HDF5 into `[T, 207, 1]` tensor
- Compatible with `torch_geometric.data.InMemoryDataset`

**`evaluation.py`** - Metrics:
- `rmse()`, `mae()`, `r2_score()` for denormalized traffic predictions

### Experiments (`experiments/temporal_only/`)

**`train.py`** - Main training script:
- `VanillaLSTM`: Single-stack LSTM encoder + linear head for multi-step forecasting
- Trains in normalized space (z-scored per sensor)
- Evaluates on test set in denormalized space (original mph units)
- Logs results to `logs/<run-name>.txt` with all hyperparameters

**`autoresearch_loop.py`** - Sequential hyperparameter sweep:
- Chains multiple `train.py` runs with preset configurations
- Default grid includes 4 variants: different hidden sizes, layers, and dropout
- All runs write individual log files for comparison

**`autonomous_research.py`** - Autonomous Karpathy-style research loop:
- Analyzes previous results and detects patterns (overfitting/underfitting/bottlenecks)
- Generates hypotheses with reasoning for next experiments
- Follows 3-phase strategy: architecture search → hyperparameter tuning → long training
- Git logic gate: commits only if RMSE improves, keeps failed logs for analysis
- Maintains `research_state.json` to track progress across runs

**`program.md`** - Research plan and results:
- Documents baseline approach and Week 3 dry-run experiments
- Key finding: 1-layer LSTM (hidden=64) performed best in 3-epoch regime

### Key Design Principles

1. **Split-safe windowing**: `TrafficWindowDataset` respects split boundaries to prevent train/val/test leakage
2. **Normalization scope**: Statistics computed from training split only, applied to all splits
3. **Evaluation in original units**: All reported metrics (RMSE, MAE, R2) are denormalized for interpretability
4. **Independent sensor modeling**: Current baseline treats 207 sensors as independent features (no spatial structure yet)
5. **Reproducibility**: All experiments use `torch.manual_seed()` and log full hyperparameters

## File Paths Convention

When running scripts, use paths relative to the project root (`/Users/daigomoriwake/Documents/stat390-project`). The code handles path resolution internally via `Path(__file__).resolve().parents[N]`.

Data is stored in `data/METR_LA/` and auto-downloaded on first use.
