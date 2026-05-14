# Complete Experiment Log Bundle — Spatio-Temporal GNN on METR-LA

**Branch:** `autoresearch/may7-gnn`  ·  **Run dates:** 2026-05-07 to 2026-05-14
**Substrate (held constant across all iterations):**
METR-LA `[34272, 207, 1]` · chronological 70/10/20 split · 12-step → 12-step window · per-sensor z-score from train-only stats · Adam + MSE in normalized space · `lr=1e-3` · `batch=64` · `seed=0` · CPU · denormalized RMSE/MAE/R² on test.

---

## A. Tracked iterations (canonical)

These are the seven hypothesis-driven runs documented in `notes.md` and (Iter 2 onward) `results.tsv`. Each is a separate git commit; the keep/discard logic gate compares `test_rmse` to the current best.

### Iter 1 — broken arch (broadcast LSTM)
- Hypothesis: shallow ST-GCN with K=5 physical adjacency reduces over-smoothing.
- Config: GCN, 1 graph layer, K=5 physical, hidden=64, 1 LSTM layer, dropout=0.1, 15 epochs.
- Result: RMSE **18.755**, MAE 11.872, R² 0.323, runtime 139.4 s.
- Decision: **discard** (worse than 15.0 baseline; root cause = single shared LSTM embedding broadcast to all 207 nodes).

### Iter 2 — per-node LSTM + GCN skip  (commit `99bf97d`)
- Hypothesis: reshape input to `(B*N, T, 1)` so each sensor gets its own LSTM pass; add residual skip around GCN.
- Config: GCN, 1 graph layer + skip, K=5 physical, hidden=64, 1 LSTM layer, 10 epochs.
- Result: RMSE **12.083**, MAE 6.236, R² 0.719, runtime 1149.3 s.
- Decision: **keep** — first model to beat the 15.0 baseline (−2.917 RMSE).

### Iter 3 — GAT 1-head + skip  (commit `b3c96ad`)
- Hypothesis: learned attention picks better neighbors than fixed Gaussian edge weights.
- Config: GAT (1 head), K=5 physical, hidden=64, 1 LSTM layer, 5 epochs.
- Result: RMSE **12.055**, MAE 5.846, R² 0.720, runtime 746.4 s.
- Decision: **keep** — marginal gain (+0.028 RMSE).

### Iter 4 — GAT + correlation adjacency  (commit `7936d9a`)
- Hypothesis: Pearson-correlation K=5 of training speeds beats physical distance.
- Config: GAT, K=5 **correlation**, hidden=64, 1 LSTM layer, 5 epochs.
- Result: RMSE **11.940**, MAE 5.772, R² 0.726, runtime 953.0 s.
- Decision: **keep** — topology change gave 4× the gain of the GCN→GAT swap.

### Iter 5 — 2-layer LSTM  (commit `5d44064`)
- Hypothesis: a 2-layer LSTM per node captures higher-order temporal dynamics.
- Config: as Iter 4 but `lstm_layers=2`, dropout=0.1, 5 epochs.
- Result: RMSE **11.990**, MAE 6.511, R² 0.723, runtime 1399.7 s.
- Decision: **discard** — val_mse oscillating 0.350–0.364; MAE jumped +0.74 mph; slower for no gain.

### Iter 6 — hidden=96  (commit `9941881`)
- Hypothesis: widen capacity (64 → 96) without deepening.
- Config: as Iter 4 but `hidden=96`, 5 epochs.
- Result: RMSE **12.122**, MAE 5.815, R² 0.717, runtime 1213.8 s.
- Decision: **discard** — val_mse diverged epochs 4–5; over-parameterized at 5-epoch budget.

### Iter 7 — 10 epochs of Iter 4 config  (commit `65d829a`)
- Hypothesis: Iter 4 val_mse still declining at epoch 4; more epochs at the same architecture will reach a lower minimum.
- Config: as Iter 4 but `epochs=10`.
- Result: RMSE **11.912**, MAE 6.244, R² 0.727, runtime 2287.8 s.
- Decision: **keep** — new best; val_mse hit new floor 0.347 at epoch 7. **Caveat:** the +0.028 gain comes at +2.4× the wall-clock of Iter 4. Marginal.

---

## B. Pre-iteration exploratory runs (logs/spatial_iter_001 … _009)

Before the loop was formalized, nine logs were written while sweeping the broken-arch configuration. They are not separate hypotheses — they are the harness re-running variants of the same architectural mistake. Kept for transparency.

| Log file | Config | Test RMSE | Note |
|---|---|---|---|
| spatial_iter_001.txt | GCN, 2 layers, K=10, 5 ep | 17.600 | early sweep |
| spatial_iter_002.txt | GCN, 2 layers, K=15, 10 ep | 18.929 | early sweep |
| spatial_iter_003.txt | identical to 002 | 18.929 | duplicate re-run |
| spatial_iter_004.txt | GCN, 1 layer, K=5, dropout=0.1, 15 ep | 18.755 | matches canonical Iter 1 |
| spatial_iter_005.txt | identical to 004 | 18.755 | duplicate |
| spatial_iter_006.txt | identical to 004 | 18.755 | duplicate |
| spatial_iter_007.txt | identical to 004 | 18.755 | duplicate |
| spatial_iter_008.txt | identical to 004 | 18.755 | duplicate |
| spatial_iter_009.txt | identical to 004 | 18.755 | duplicate |

**Observation:** files 004–009 are six identical re-runs of the same broken config. This is a logger-duplication failure — the harness re-fired the same job without varying inputs. Counted as a procedural near-miss (no result was wrong, but compute was wasted).

---

## C. Empty / failed-log files

- `logs/spatial_iter_7.txt` — 0 bytes. The intended Iter 7 log path collided with the harness's existing numbering and the file was created but never written. The actual Iter 7 metrics are in `logs/spatial_iter7_longer_training.txt`. **Procedural failure → near-crash** (the run completed, but the standard log path is empty; downstream tooling that reads `spatial_iter_7.txt` would silently see no data).

---

## D. Cross-referenced files

- `results.tsv` — canonical machine-readable tracker (Iter 2 onward).
- `notes.md` — per-iteration hypothesis + reasoning + post-mortem.
- `metadata.json` — best-config snapshot (currently stale: still points to Iter 4; should be refreshed to Iter 7).
- `results_plot.png` — RMSE-by-iteration plot with keep/discard overlay (regenerated 2026-05-14, now includes Iter 7).
- `make_plot.py` — script that produced `results_plot.png`.
- `experiment_matrix.md` — Week 4 result matrix (Iter 1–6); superseded by this bundle for Iter 7.
- `failure_memo.md` — Week 4 failure analysis + error taxonomy.

---

## E. Headline metrics

| | RMSE (mph) | MAE | R² |
|---|---|---|---|
| Temporal-only LSTM baseline (Week 3) | 15.000 | 8.912 | 0.567 |
| **Spatial best (Iter 7)** | **11.912** | 6.244 | 0.727 |
| Δ vs baseline | **−3.088 (−20.6%)** | −2.668 | +0.160 |
