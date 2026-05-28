# Project Statement (Revised) — Spatio-Temporal Traffic Forecasting on METR-LA

**Author:** Daigo Moriwake  ·  **Course:** STAT390  ·  **Revision date:** 2026-05-21
**Current branch:** `autoresearch/may7-gnn`  ·  **Current best test RMSE:** 11.912 mph (Iter 7)

---

## 1. Problem

Forecast freeway traffic speeds on the METR-LA sensor network 1 hour ahead. The dataset is `[34272 timesteps, 207 sensors, 1 feature]` at 5-minute resolution; the task takes a 12-step window as input and predicts the next 12 steps. Evaluation is denormalized RMSE / MAE / R² on a chronological 70/10/20 split, with per-sensor z-score normalization fit on the training split only.

## 2. Original framing (Week 1) vs. revised framing (Week 5)

**Original framing:** "Build a spatio-temporal GNN that beats a temporal-only LSTM baseline." Implicit assumption: the gain comes from the *spatial* model.

**Revised framing:** "Quantify *where* the gain comes from — temporal encoder, graph topology, or graph operator — and report each independently." The Week 4/5 experimental record shows the assumption above is unsafe: most of the headline RMSE reduction may live in the per-node temporal encoder, not in spatial message passing. Until the temporal-encoder-only ablation is run, the spatial contribution is overstated.

## 3. What is known (as of Iter 7)

- **Temporal-only LSTM baseline:** test RMSE 15.000.
- **Spatial best (Iter 7):** test RMSE 11.912 — a 20.6% relative reduction.
- **Decomposition of the 3.088 RMSE gap** (per `what_worked_memo.md`):
  - Per-node LSTM + GCN skip vs. the broken broadcast baseline accounts for the bulk of the improvement (Iter 1 → Iter 2: −6.67 RMSE on the broken-baseline scale; equivalently the headline gain over the temporal-only baseline).
  - Correlation adjacency vs. physical adjacency contributes −0.115 RMSE.
  - GCN → GAT contributes −0.028 RMSE at 2.5× the runtime.
  - Longer training (5 → 10 epochs) contributes another −0.028 RMSE.
- **Confirmed dead ends:** 2-layer per-node LSTM (instability on T=12 windows), hidden=96 at 5-epoch budget (overfitting).
- **Open finding:** validation MSE plateaus around 0.347 at epoch 7 of the best config — the architecture has saturated. Further gains require structural change, not more compute.

## 4. Revised research questions

| # | Question | Why it matters |
|---|---|---|
| Q1 | How much of the 3.088-RMSE gain over the temporal baseline survives if we remove the graph layer entirely (per-node LSTM only)? | Determines whether this is a "spatial" or "temporal-encoding" project. |
| Q2 | Is the Iter 4 → Iter 7 advantage (0.028 RMSE) larger than seed noise? | Without a multi-seed bound, the headline number is single-seed and not defensible. |
| Q3 | Does sparsifying the correlation graph (K=3) or expanding the receptive field (2 GAT layers, ChebConv K=2) move the plateau? | Iter 7 said "spatial receptive field is the bottleneck" — these are the cheapest tests of that claim. |
| Q4 | Does learned adjacency (parameterize the K=5 mask) close the gap further? | Highest-payoff structural change on the backlog; only worth running after Q1/Q2/Q3. |

## 5. Deliverables (Week 5 → Week 7 / project end)

1. **Ablation table** with confidence intervals, including the temporal-encoder-only control (`ablation_table.md`).
2. **Revised agent strategy** (`program.md`) — reordered backlog reflecting what was learned in Weeks 3–5.
3. **Final two-week plan** (`two_week_plan.md`) with concrete deliverables per week.
4. **Final write-up** (Week 7) — re-framed around the decomposition in §3 rather than the original "GNNs help" claim.

## 6. Scope of work that will *not* be done

To keep the project finishable in the remaining two weeks:

- No new datasets (PEMS-BAY, etc.). All experiments stay on METR-LA.
- No multi-GPU or distributed training. CPU-only, single process, as throughout.
- No production deployment artifact. The output is a research write-up + reproducible repo.
- No more than 12 additional training runs in the remaining budget — the substrate is mature; further sweeps have diminishing return.

## 7. Success criteria

The project is considered complete and defensible if:

- (a) The temporal-encoder-only ablation has been run and the spatial contribution is **stated correctly** in the final write-up — regardless of whether it shrinks the headline.
- (b) The headline RMSE is reported with a multi-seed standard deviation, not as a single point estimate.
- (c) At least one structural change beyond Iter 7 has been *attempted* (kept or discarded with documented reasoning).
- (d) All claims in the final write-up trace back to a row in `ablation_table.md` and a commit in `results.tsv`.

This is a deliberately weaker bar than "beat 11.912" — the methodological work (a, b, d) is what makes the project a research artifact rather than a leaderboard chase.
