# Failure Analysis Memo — Spatio-Temporal GNN on METR-LA

**Author:** Daigo Moriwake  ·  **Branch:** `autoresearch/may7-gnn`  ·  **Date:** 2026-05-14

## TL;DR

Six controlled iterations were run on the GNN sandbox. One (Iter 2) was a step-function win; three subsequent "keeps" were sub-0.15-RMSE refinements; two were outright failures. The failure modes split cleanly into five categories — three architectural, two procedural. The single most important finding is not architectural: **graph topology (which neighbors) mattered more than the convolution operator (how to aggregate them).**

## Error taxonomy

Each failure observed during the spatial sweep falls into one of five categories. The taxonomy is descriptive — every entry is grounded in at least one of the 6 logged iterations.

### T1 — Representation collapse
*Information about distinct entities is destroyed before the spatial step can use it.*

- **Symptom:** Adding the graph does not help (RMSE stays near 18 mph) even with K, depth, and dropout swept.
- **Mechanism:** A single LSTM over `(B, T, N)` produces one shared embedding that is then broadcast to all 207 sensors. The GCN aggregates near-identical vectors, so message passing carries no signal.
- **Example:** Iter 1. RMSE 18.755 with R² 0.323 — *worse* than the temporal-only baseline at 15.000.
- **Fix that worked:** Per-node LSTM via `(B*N, T, 1)` reshape (Iter 2: RMSE 12.083).
- **Diagnostic to use next time:** Inspect input variance across nodes *after* the temporal encoder. If it collapses, the GNN can't help.

### T2 — Optimization instability under short windows
*Deeper temporal modules are unstable when the input sequence is too short to constrain them.*

- **Symptom:** Val MSE oscillates between epochs and MAE rises even when RMSE looks flat.
- **Mechanism:** A 2-layer LSTM on `T=12` has roughly the same depth as the sequence length. Gradient signal is weak; the second layer behaves like noise.
- **Example:** Iter 5. RMSE 11.990 (≈ Iter 4) but **MAE +0.74 mph** and val_mse oscillating between 0.350 and 0.364. Slower (1400 s vs 953 s) for worse fit.
- **Fix that worked:** Revert to 1 LSTM layer (Iter 4 config restored as best).
- **Diagnostic to use next time:** Watch MAE alongside RMSE — RMSE can mask calibration failures because squared error de-emphasizes the bulk of points.

### T3 — Capacity overshoot at fixed training budget
*A wider/deeper model overfits before training has time to compensate.*

- **Symptom:** Train loss continues to drop while val loss starts climbing — classic divergence — but only in the last 1–2 epochs of the budget.
- **Mechanism:** Hidden 96 adds ≈50% parameters to the LSTM + GAT stack. Five epochs is not enough to regularize through SGD noise alone.
- **Example:** Iter 6. RMSE 12.122; val_mse trajectory 0.351 → 0.352 → 0.357 → 0.364 across epochs 3–5.
- **Fix that worked:** Keep hidden=64 at 5 epochs. (Open question: would hidden=96 + longer training + early stop close the gap? Untested — see "what was not tried.")
- **Diagnostic to use next time:** Tie capacity changes to either (a) more epochs, or (b) explicit regularization. Don't change them in isolation.

### T4 — Topology misspecification
*The graph defines what counts as "near" — wrong adjacency caps how much the conv operator can recover.*

- **Symptom:** GAT (learned attention) helps less than expected (+0.028 RMSE).
- **Mechanism:** Physical K=5 nearest neighbors includes sensors on parallel streets whose traffic is structurally decoupled. No attention weight can fix the fact that the *candidate set* of neighbors is wrong.
- **Example:** Iter 3 (physical + GAT): 12.055 → Iter 4 (correlation + GAT): 11.940. Correlation adjacency contributed **4× more** RMSE improvement than swapping GCN→GAT did, at the same training cost.
- **Fix that worked:** Pearson-correlation K=5 adjacency from training-split speeds.
- **Diagnostic to use next time:** When trying conv operators, also try at least one alternative adjacency in the same sweep — they interact.

### T5 — Compute-quality mismatch (procedural)
*A small RMSE gain is bought with a disproportionate runtime cost.*

- **Symptom:** Per the program's "simplicity gate," a kept change should be either better *and* cheaper, or much better. Iter 3 violated the spirit: +0.028 RMSE for +2.5× cost over Iter 2's pre-vectorized GCN.
- **Mechanism:** GAT computes attention per edge per head. Per-edge ops dominate on dense K=5 graphs with 207 nodes × 64 batch.
- **Example:** Iter 3 was *kept* by the strict gate (`test_rmse < current_best`) but is the weakest of the three keeps under a cost-adjusted reading.
- **Fix going forward:** Treat any kept change with <0.1 RMSE gain as a candidate for re-evaluation when the surrounding architecture changes. If GAT's benefit is purely from attention, smaller/lighter alternatives (edge-weight reweighting on GCN) should be tested.
- **Diagnostic to use next time:** Track `Δ RMSE / Δ runtime` per iteration. Iter 3's number is poor.

### Summary table

| Code | Category | Iter(s) | Root cause | Detect via |
|---|---|---|---|---|
| T1 | Representation collapse | 1 | Broadcasting one embedding to all nodes | Per-node variance after encoder |
| T2 | Optimization instability under short windows | 5 | LSTM depth ≥ sequence length | MAE rising while RMSE flat; noisy val_mse |
| T3 | Capacity overshoot at fixed budget | 6 | Width ↑ without epochs ↑ | Val_mse divergence in last 1–2 epochs |
| T4 | Topology misspecification | 3 (partial), 4 (fix) | Wrong neighbor candidate set | Operator change underperforms adjacency change |
| T5 | Compute-quality mismatch (procedural) | 3 | Logic gate ignores cost | Track Δ RMSE / Δ runtime |

## What the agent did well

- **Held the substrate constant.** Lr, batch, seed, window length, normalization, and split protocol never changed across the six iterations. The matrix is comparable row-to-row.
- **Diagnosed before fixing.** Iter 1 → Iter 2 was not a sweep; it was an architectural reading of the broken model. The `notes.md` entry for Iter 2 names the broadcasting bug explicitly before any code change.
- **Operator vs. topology separation.** Iter 3 (operator) and Iter 4 (topology) were run as separate steps. This is what made the "topology > operator" claim defensible — the alternative (changing both at once) would have left it ambiguous.
- **Self-imposed budget hygiene.** Each run had a 7-minute target and 10-minute hard kill; runtime was logged on every iteration; the discard gate fired automatically without operator intervention.
- **Failed runs were retained in `logs/` rather than overwritten**, which made the multi-failure pattern in `spatial_iter_004` through `spatial_iter_009` visible (same config logged 6 times — see "what was done badly").

## What the agent did badly

- **Logger duplication.** `spatial_iter_004.txt` through `spatial_iter_009.txt` log identical configs and identical metrics six times. This looks like a re-run loop in the harness without de-duplication. It pollutes the log directory and would have caused a real bug if any of those re-runs had stochastic variation that was then averaged into a "stable" reading.
- **Single-seed throughout.** Every iteration used `seed=0`. The 0.115 RMSE gain in Iter 4 (the keystone finding) is *not* seed-averaged. A 3-seed re-run of Iter 2 vs Iter 4 is the most valuable next experiment, and it wasn't run.
- **No early-stopping during the budget.** Iter 6 (hidden=96) failed by overfitting in the last two epochs. With early stopping on val_mse this iteration might have been a "keep." The procedure does not distinguish "architecture is bad" from "stopping rule is wrong."
- **Discarded configurations were not re-tested after Iter 4.** Once correlation adjacency was confirmed best, the prior "discards" (physical adjacency variants, deeper LSTM, wider hidden) were never re-evaluated under the new graph. Some of them may interact differently with correlation adjacency than they did with physical.
- **Simplicity gate was not enforced numerically.** The program file declares a simplicity gate, but the keep/discard decision used only RMSE. Iter 3 should arguably have been rejected on cost-per-improvement grounds.

## What was not tried (open work)

- Multi-seed reruns of Iter 2 and Iter 4 (variance estimate on the keystone gain).
- Longer training (≥10 epochs) of the Iter 4 config — Iter 7 was queued in `notes.md` but no result is logged.
- ChebConv (`K=2`), DropEdge, and learned adjacency — all on the hypothesis backlog in `program.md` and untouched.
- A real ablation: per-node LSTM with **no** graph layer, to isolate the spatial contribution from the temporal-encoding fix. Without this, "+2.917 RMSE from spatial structure" is overstated — some of that gain may be from the encoder rewrite alone.

## Recommendation for next iteration

The single most valuable run is the **temporal-encoder ablation**: per-node LSTM, *no* graph convolution, same correlation-adjacency dataset. If RMSE is anywhere near 12.0, then most of the spatial improvement is actually a temporal-encoder improvement and the Week 5 framing needs to change.

After that: 3-seed re-run of Iter 4 to put a confidence interval on the headline number.

## Files

- `experiment_matrix.md` — controlled set + result matrix (this submission)
- `results_plot.png` — RMSE-by-iteration plot with keep/discard overlay (this submission)
- `failure_memo.md` — this memo
- `results.tsv` — raw tab-separated log
- `notes.md` — per-iteration hypotheses and reasoning
- `logs/spatial_iter_*.txt` — raw metrics per run
