# Ablation & Comparison Table — Spatio-Temporal GNN on METR-LA

**Branch:** `autoresearch/may7-gnn`  ·  **As of:** 2026-05-21  ·  **Baseline:** temporal-only LSTM, test RMSE **15.000 mph**

Substrate held constant unless explicitly varied: METR-LA `[34272, 207, 1]` · 70/10/20 chronological split · 12→12 window · per-sensor z-score from train-only stats · Adam + MSE in normalized space · `lr=1e-3` · `batch=64` · `seed=0` · CPU · denormalized RMSE/MAE/R² on test.

---

## A. Completed iterations (controlled, one-factor-at-a-time)

| Iter | Commit | Changed factor (vs prev best) | Conv | Adj | K | Hidden | LSTM layers | Epochs | Test RMSE | Test MAE | Test R² | Runtime (s) | Δ RMSE / Δ runtime ratio | Decision |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | (pre-tsv) | — initial (broken arch) | GCN | physical | 5 | 64 | 1 | 15 | 18.755 | 11.872 | 0.323 | 139.4 | — | discard |
| 2 | `99bf97d` | Per-node LSTM + GCN skip (arch fix) | GCN | physical | 5 | 64 | 1 | 10 | **12.083** | 6.236 | 0.719 | 1149.3 | huge / +1010 s | **keep** |
| 3 | `b3c96ad` | Conv: GCN → GAT (1 head) | GAT | physical | 5 | 64 | 1 | 5 | **12.055** | 5.846 | 0.720 | 746.4 | −0.028 / −403 s (faster) | **keep-marginal** |
| 4 | `7936d9a` | Adj: physical → correlation | GAT | correlation | 5 | 64 | 1 | 5 | **11.940** | 5.772 | 0.726 | 953.0 | −0.115 / +207 s | **keep** |
| 5 | `5d44064` | LSTM depth: 1 → 2 (dropout 0.1) | GAT | correlation | 5 | 64 | 2 | 5 | 11.990 | 6.511 | 0.723 | 1399.7 | +0.050 / +447 s | discard (T2) |
| 6 | `9941881` | Width: hidden 64 → 96 | GAT | correlation | 5 | 96 | 1 | 5 | 12.122 | 5.815 | 0.717 | 1213.8 | +0.182 / +261 s | discard (T3) |
| 7 | `702f7b7` | Epochs: 5 → 10 (Iter 4 config) | GAT | correlation | 5 | 64 | 1 | 10 | **11.912** | 6.244 | 0.727 | 2287.8 | −0.028 / +1335 s | **keep-marginal** |

**Failure-mode codes** (see `failure_memo.md`): T2 = optimization instability under short windows; T3 = capacity overshoot at fixed budget.

## B. Per-factor attribution (decomposing the 3.088-RMSE gain over baseline)

Each row reports the marginal contribution of one factor, isolated from the others where possible. Read alongside `what_worked_memo.md`.

| Factor | Compared rows | Marginal Δ RMSE | % of total gain (3.088) | Verdict |
|---|---|---|---|---|
| Per-node LSTM + skip (the arch fix) | Iter 1 → Iter 2 | −6.672 (broken→fixed scale) | dominant — accounts for crossing the baseline | step-function win |
| Conv operator: GCN → GAT | Iter 2 → Iter 3 (note: also changed epochs 10→5) | −0.028 (confounded) | <1% | weak, cost-heavy |
| Adjacency: physical → correlation | Iter 3 → Iter 4 | −0.115 | ~4% | the only post-arch lever that produced real movement |
| LSTM depth: 1 → 2 | Iter 4 → Iter 5 | +0.050 | negative | discard |
| Width: hidden 64 → 96 | Iter 4 → Iter 6 | +0.182 | negative | discard |
| Training length: 5 → 10 epochs | Iter 4 → Iter 7 | −0.028 | ~1% | weak; signals architectural saturation |

**Read across the table:** essentially all the gain is from the per-node encoding fix (Iter 2) and from picking the right neighbors (Iter 4). Everything else is decimal noise.

**Caveat:** The Iter 2 → Iter 3 comparison is confounded — both the conv operator (GCN → GAT) and epochs (10 → 5) changed. The −0.028 RMSE for GAT therefore includes the effect of *training less*. A clean operator-only comparison is not yet on file and is a candidate ablation (see §C).

## C. Planned ablations (Tier 1 — methodological controls, not yet run)

These are the missing measurements that determine whether the headline claim ("a GNN reduces test RMSE by 20.6%") is correctly attributed. Run order matters: A1 first, then A2/A3 in parallel.

| ID | Hypothesis being tested | Config (vs Iter 7 best) | Predicted RMSE | Decision rule | Estimated runtime |
|---|---|---|---|---|---|
| **A1** | "Most of the gain is the temporal encoder, not the graph." | Drop the GAT layer entirely. Per-node LSTM (hidden=64, 1 layer, 10 ep) → linear readout. No skip needed (no GCN to skip around). | 12.0–12.8 | If RMSE ≤ 12.3, the spatial contribution of the best model is ≤ 0.4 RMSE — re-frame the project around the temporal encoder. | ~600 s |
| **A2** | "Iter 7's 11.912 is within seed noise of Iter 4's 11.940." | Iter 7 config, seeds {0, 1, 2}. | mean ≈ 11.9, std unknown | Report mean ± std. If std > 0.05, the 10-epoch advantage is not significant. | ~6900 s (3× Iter 7) |
| **A3** | "Iter 4 itself is robust across seeds." | Iter 4 config (5 epochs), seeds {0, 1, 2}. | mean ≈ 11.95, std unknown | Pair with A2 to compute the seed-noise scale of the 0.028 gap. | ~2860 s |
| **A4** | "GCN→GAT (Iter 3) was a real gain, not a confound with epochs." | Iter 2 config but with epochs=5 (matching Iter 3) — to isolate the operator effect from training length. | 12.1–12.3 | If RMSE is within ±0.05 of Iter 3's 12.055, the operator change was negligible; the apparent gain came from training-length confound. | ~570 s |

## D. Planned structural changes (Tier 2 — conditional on Tier 1 results)

To be run **after** A1–A4 land. Each follows the multi-metric logic gate from `program.md`.

| ID | Change | Rationale | Discard threshold |
|---|---|---|---|
| **S1** | Sparser correlation graph K=3 (was K=5) | Topology was the dominant lever in Iter 4; lower K reduces over-smoothing further. | RMSE ≥ Iter 7 best |
| **S2** | Two GAT layers + per-layer skip (drafted as Iter 8 in `notes.md` but not yet executed) | Expand receptive field 1-hop → 2-hop; per-layer skip bounds the downside. | RMSE ≥ Iter 7 best |
| **S3** | ChebConv K=2 | Localized spectral filter; cheaper than GAT; tests whether learned attention specifically was the value or just the receptive field. | RMSE ≥ Iter 7 best **and** runtime ≥ Iter 7 |
| **S4** | Learned adjacency (parameterize a K=5 mask end-to-end) | Highest-payoff structural change on backlog; only worth running after Tier 1 confirms the spatial contribution is real. | RMSE ≥ Iter 7 best |

## E. Comparison to the temporal baseline (headline)

| Model | Test RMSE | Test MAE | Test R² | Δ RMSE vs baseline |
|---|---|---|---|---|
| Temporal-only LSTM (Week 3) | 15.000 | 8.912 | 0.567 | — |
| Iter 2 — per-node LSTM + GCN + skip | 12.083 | 6.236 | 0.719 | −2.917 (−19.4%) |
| Iter 7 — per-node LSTM + GAT + correlation, 10 ep (current best) | **11.912** | 6.244 | 0.727 | **−3.088 (−20.6%)** |

**Caveat (will be removed once A1 lands):** The "spatial" attribution above assumes the gap between Iter 2 and the temporal baseline is from spatial structure. Until A1 runs, that gap may include unmeasured contribution from the per-node temporal encoder change alone.

---

**Companion files:** `results.tsv` (raw log), `notes.md` (per-iter reasoning), `failure_memo.md` (T1–T5 taxonomy), `what_worked_memo.md` (per-change attribution narrative), `experiment_log_bundle.md` (full Iter 1–7 detail + exploratory pre-iteration runs), `project_statement.md` (revised research questions), `program.md` (agent loop + backlog), `two_week_plan.md` (remaining schedule).
