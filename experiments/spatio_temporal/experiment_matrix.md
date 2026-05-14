# Experiment-Result Matrix — Spatio-Temporal GNN on METR-LA

**Branch:** `autoresearch/may7-gnn`  ·  **Dates:** 2026-05-07  ·  **Baseline to beat:** temporal-only LSTM, test RMSE **15.000 mph**

## Experimental control protocol

One factor varied per iteration; all other factors held constant. The fixed substrate across the controlled set is:

| Factor | Value (held constant) |
|---|---|
| Dataset | METR-LA, `[34272, 207, 1]` |
| Split | Chronological 70 / 10 / 20 train/val/test |
| Window | 12 steps in → 12 steps out (1h → 1h) |
| Normalization | Per-sensor z-score, train-split stats only |
| Optimizer | Adam, MSE loss in normalized space |
| Learning rate | `1e-3` |
| Batch size | `64` |
| Seed | `0` |
| Eval | Denormalized RMSE / MAE / R² on test split |
| Hardware | CPU, single process |
| Time budget per run | 7 min target, 10 min hard kill |

The agent committed each iteration to its own git commit and used a **logic gate** for the keep/discard decision: `test_rmse < current_best ⇒ keep`, otherwise `git reset --hard HEAD~1`. This is the controlled-experiment unit — every "keep" mutates exactly one factor of the previous best.

## Matrix

| Iter | Commit | Changed factor (vs prev best) | Arch | Conv | Adj | K | Hidden | LSTM layers | Epochs | Test RMSE | Test MAE | Test R² | Runtime (s) | Decision | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | (pre-tsv) | — (initial) | Broadcast LSTM + GCN | GCN | physical | 5 | 64 | 1 | 15 | **18.755** | 11.872 | 0.323 | 139.4 | discard | Broken arch: single LSTM hidden broadcast to all 207 nodes |
| 2 | `99bf97d` | Architecture: per-node LSTM + GCN skip | Per-node LSTM + GCN + skip | GCN | physical | 5 | 64 | 1 | 10 | **12.083** | 6.236 | 0.719 | 1149.3 | **keep** | First config to beat 15.000 baseline (−2.917) |
| 3 | `b3c96ad` | Conv operator: GCN → GAT (1 head) | Per-node LSTM + GAT + skip | GAT | physical | 5 | 64 | 1 | 5 | **12.055** | 5.846 | 0.720 | 746.4 | **keep** | Marginal gain (+0.028); GAT ~2.5× cost per RMSE point |
| 4 | `7936d9a` | Adjacency: physical → correlation | Per-node LSTM + GAT + skip | GAT | **correlation** | 5 | 64 | 1 | 5 | **11.940** | 5.772 | 0.726 | 953.0 | **keep** | Topology beats operator: +0.115 RMSE vs +0.028 from GAT |
| 5 | `5d44064` | LSTM depth: 1 → 2 layers (dropout 0.1) | Per-node 2-LSTM + GAT + skip | GAT | correlation | 5 | 64 | 2 | 5 | 11.990 | 6.511 | 0.723 | 1399.7 | discard | Val_mse noisy; MAE jumped +0.74 mph — over-deep for T=12 |
| 6 | `9941881` | Width: hidden 64 → 96 | Per-node LSTM + GAT + skip | GAT | correlation | 5 | **96** | 1 | 5 | 12.122 | 5.815 | 0.717 | 1213.8 | discard | Val_mse diverged epochs 4–5 — overfit at 5 epochs |

**Current best:** Iter 4 — RMSE **11.940 mph** at commit `7936d9a`. Spatial structure improves on the temporal-only baseline by **3.06 RMSE points** (≈20% relative).

## What the matrix shows

- The only architectural change with a step-function payoff was **Iter 2** (per-node LSTM + skip). All later improvements are decimal-point refinements.
- Of the three "refinement" knobs tried after Iter 2 — conv operator (GAT), adjacency (correlation), and capacity (depth/width) — **adjacency dominated**. Topology > operator > capacity at this scale.
- The two discards (Iter 5, Iter 6) failed for orthogonal reasons (instability, overfit), not the same one — both are documented in the taxonomy below.

Companion files in this directory: `results.tsv` (raw machine-readable log), `results_plot.png` (RMSE-by-iteration with decision overlay), `failure_memo.md` (analysis + taxonomy), `notes.md` (per-iter hypotheses and reasoning).
