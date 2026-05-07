# Autonomous Research Loop (Karpathy-style)

## Overview

This implements a true Karpathy-style autonomous research loop that:

1. **Reads** previous experiment logs
2. **Analyzes** performance patterns (overfitting/underfitting/bottlenecks)
3. **Generates** hypotheses for next experiments with reasoning
4. **Runs** training automatically
5. **Commits** to git if improved, keeps logs if regressed
6. **Repeats** autonomously until target reached or max iterations

## Usage

### Quick Start

```bash
# Run 10 autonomous iterations, stop if RMSE reaches 14.0
python experiments/temporal_only/autonomous_research.py --max-iterations 10 --rmse-target 14.0

# Run 5 iterations, target RMSE 12.0
python experiments/temporal_only/autonomous_research.py --max-iterations 5 --rmse-target 12.0
```

### What Happens

Each iteration:

1. **Analysis Phase**: Examines all previous logs, identifies best config, detects patterns:
   - `short_training_only`: All experiments used < 5 epochs
   - `simpler_is_better`: 1-layer models outperform deeper ones
   - `single_lr_only`: Only one learning rate tested

2. **Hypothesis Phase**: Generates next experiment based on patterns:
   - **Iterations 1-10**: Architecture search (layers, hidden size, dropout)
   - **Iterations 11-20**: Hyperparameter tuning (learning rate, batch size)
   - **Iterations 21+**: Long training runs to establish true baseline

3. **Execution Phase**: Runs `train.py` with proposed config

4. **Decision Phase** (Git Logic Gate):
   - ✓ **If RMSE improved**: Commits with detailed message
   - ✗ **If RMSE regressed**: Keeps log for analysis, doesn't commit

5. **State Tracking**: Saves progress to `research_state.json`

## Example Output

```
================================================================================
ITERATION 1
================================================================================

[1/5] Loading previous results...
Found 4 previous experiments

[2/5] Analyzing performance...
Analysis: {
  "best_rmse": 14.999995,
  "best_config": {"hidden": 64, "layers": 1, "dropout": 0.0, "lr": 0.001},
  "patterns": ["short_training_only", "simpler_is_better", "single_lr_only"]
}

[3/5] Generating hypothesis...
Reasoning: Best results came from simple 1-layer model. Testing if longer training improves performance.
Expected: RMSE should decrease with more epochs if model is underfitting
Config: {"hidden": 64, "layers": 1, "epochs": 15, ...}

[4/5] Running experiment: auto_iter_001...

[5/5] Evaluating results...
Result: RMSE=15.2435, MAE=9.1093, R²=0.5527

✗ REGRESSION: 0.2435 RMSE increase
Hypothesis did not improve performance. Keeping log for analysis.
```

## Key Features

### Intelligent Pattern Detection

- **Short training detection**: Suggests longer runs if all experiments used few epochs
- **Architecture analysis**: Compares simple vs complex models
- **Learning rate coverage**: Suggests exploration if only one LR tried
- **Overfitting detection**: Monitors train vs val loss divergence (future enhancement)

### Progressive Strategy

The system follows a 3-phase strategy:

**Phase 1 (iter 1-10): Architecture Search**
- Test different layer depths
- Explore hidden sizes
- Try dropout variations
- Keep training short (5-15 epochs) for fast iteration

**Phase 2 (iter 11-20): Hyperparameter Tuning**
- Optimize learning rate
- Test batch sizes
- Fine-tune regularization
- Medium training runs (15-30 epochs)

**Phase 3 (iter 21+): Convergence**
- Long training with best config (50+ epochs)
- Establish true baseline
- Get final error floor

### Git Integration

Only successful improvements are committed:

```bash
git log --oneline
a1b2c3d auto: iteration 7 - RMSE 14.532 (improved by 0.468)
d4e5f6g auto: iteration 4 - RMSE 15.000 (improved by 0.508)
```

Failed experiments are kept in logs/ for analysis but not committed.

## Comparison to Manual Loop

| Feature | Manual (`autoresearch_loop.py`) | Autonomous (`autonomous_research.py`) |
|---------|--------------------------------|--------------------------------------|
| Strategy | Predetermined grid | Adaptive based on results |
| Analysis | Human interprets logs | Automatic pattern detection |
| Git | Manual | Automatic (commit if improved) |
| Iterations | All configs run once | Progressive refinement |
| Stopping | Fixed number | Target RMSE or max iterations |

## Customization

### Adding New Patterns

Edit `analyze_performance()` to detect new patterns:

```python
# Example: Detect if batch size affects performance
if len(set(r.batch_size for r in results)) > 1:
    analysis["patterns"].append("batch_size_explored")
```

### Modifying Hypothesis Generation

Edit `generate_hypothesis()` to change search strategy:

```python
# Example: Try different LSTM variants
if iteration_num <= 5:
    return Hypothesis(
        config={"model_type": "GRU", ...},
        reasoning="Testing GRU vs LSTM",
        expected_outcome="GRU may train faster"
    )
```

### Adjusting Git Logic

Edit `git_commit()` to change commit behavior:

```python
# Example: Only commit if improvement > threshold
if improvement > 0.5:  # Only commit if RMSE drops by 0.5+
    git_commit(message)
```

## Current Best Result

After initial test (2 iterations):

- **Best Run**: `iteration_4`
- **RMSE**: 15.0000
- **Config**: hidden=64, layers=1, dropout=0.0, lr=0.001, epochs=3

The autonomous loop correctly identified this as the best baseline and is now testing:
1. Longer training (15 epochs) - resulted in slight regression
2. Light dropout (0.1) with 10 epochs - also regressed

**Next Steps**: The system will continue exploring different hypotheses to break through the RMSE 15.0 barrier.

## Integration with Capstone

This autonomous loop implements the "Karpathy-style" research approach for your Northwestern capstone:

1. **PI (You)**: Set max iterations, RMSE target, review results
2. **Agent (Script)**: Runs autonomously, makes decisions, commits improvements
3. **Outcome**: Optimized temporal-only baseline without manual hyperparameter search

Once the temporal baseline is optimized, you can:
1. Create `experiments/spatiotemporal/` with same autonomous loop
2. Compare final RMSE of both branches
3. Quantify the value of spatial structure
