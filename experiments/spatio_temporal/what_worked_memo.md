# "What Actually Worked" Memo — Spatio-Temporal GNN on METR-LA

**Author:** Daigo Moriwake  ·  **Branch:** `autoresearch/may7-gnn`  ·  **Date:** 2026-05-14

## The headline

Across seven hypothesis-driven iterations (plus nine exploratory pre-iteration runs), exactly **two** changes account for essentially all of the improvement over the temporal-only baseline:

1. **Per-node temporal encoding** (Iter 2): −6.67 RMSE
2. **Correlation-based graph adjacency** (Iter 4): −0.14 RMSE

Together they bring test RMSE from 18.755 (broken initial arch) → 15.000 (Week 3 baseline) → 11.912 (current best). Everything else — GAT vs GCN, deeper LSTM, wider hidden, longer training — was either marginal or actively harmful.

## Per-change attribution

| Change | Where introduced | Δ RMSE | Verdict |
|---|---|---|---|
| Per-node LSTM via `(B*N, T, 1)` reshape | Iter 2 | **−6.672** vs Iter 1 | Step-function win. The single most important architectural decision. |
| Residual skip around GCN | Iter 2 | (bundled with above) | Necessary partner to per-node encoding; prevents over-smoothing. |
| GCN → GAT (1 head) | Iter 3 | −0.028 | Weakly positive; cost: 2.5× runtime. Probably not worth it on its own. |
| Physical → correlation adjacency, K=5 | Iter 4 | **−0.115** | Largest post-Iter-2 gain. The graph topology was the bottleneck, not the conv operator. |
| 1 → 2 LSTM layers | Iter 5 | +0.050 | Harmful. T=12 is too short to stabilize a 2-layer recurrent stack. |
| Hidden 64 → 96 | Iter 6 | +0.182 | Harmful. Overfits at 5-epoch budget. |
| 5 → 10 epochs (Iter 4 config) | Iter 7 | −0.028 | Weakly positive; cost: 2.4× runtime. Marginal, and signals the architecture has saturated. |

## What actually worked (and why)

**1. Treating each sensor as its own time series before message passing.**
The original architecture fed `(B, T, N)` into one LSTM, treating the 207 sensors as features. Output was a single hidden vector that was broadcast back to every node. The GNN then aggregated 207 near-identical vectors and could not improve the prediction — message passing has nothing to say when all the messages are the same. Reshaping to `(B*N, T, 1)` and running the LSTM per node gave each sensor a distinct embedding that *carried information worth aggregating*. This is the only change with a step-function payoff and it is the change everything else builds on.

**2. Choosing neighbors by speed correlation rather than physical distance.**
The METR-LA road network puts physically-close sensors on parallel surface streets that are structurally decoupled in traffic terms. K=5 physical neighbors therefore polluted each node's message set with irrelevant signal. Replacing the adjacency with K=5 Pearson-correlation neighbors (computed on training-split speeds only) selects sensors that actually behave like one another — same highway, same direction, same time-of-day phase. The change is cheap (it's a different precomputed edge list, no extra parameters) and it produced a +0.115 RMSE gain on top of a model that was already past the baseline. This is the clearest evidence that **graph topology matters more than the convolution operator** in this regime.

**3. The keep/discard logic gate as a forcing function.**
The procedural rule "commit each experiment; if RMSE worsens, `git reset --hard HEAD~1`" pushed every change to defend itself on its own merit, immediately. This is what kept Iter 5 and Iter 6 from contaminating the codebase, and it's why I can attribute each gain to a single factor instead of arguing about combined effects.

## What didn't work (and why)

**Deeper temporal modeling under short windows (Iter 5).** A 2-layer LSTM on 12-step inputs has roughly the same depth as the sequence length. Gradients didn't propagate cleanly; val_mse oscillated; MAE rose by +0.74 mph even though RMSE looked nearly flat. The lesson is procedural: when MAE and RMSE disagree, trust MAE — RMSE can absorb calibration failures because squared error de-emphasizes the body of the distribution.

**Wider hidden states at fixed budget (Iter 6).** Hidden 96 added ~50% more parameters to the LSTM + GAT stack. Five epochs was not enough to regularize through SGD noise; val_mse climbed in the last two epochs. Crucially, this iteration was not paired with more epochs or with early stopping, so it cannot be ruled out as a useful change under a different training schedule — but it cannot currently be kept.

**The GCN → GAT swap (Iter 3).** Kept by the strict gate but the weakest of the four keeps. Attention helps marginally, costs 2.5× per epoch, and is dwarfed by the adjacency change that followed it. If the simplicity gate were enforced numerically (Δ RMSE / Δ runtime), Iter 3 would be on the chopping block.

**Longer training (Iter 7).** A real but marginal improvement (+0.028 RMSE) at +2.4× wall-clock vs Iter 4. The interesting result is the *signal*: val_mse plateaued at 0.347 by epoch 7 and stopped declining. The architecture has saturated. More epochs will not help further; only structural changes will.

## What was never tested (and should be)

The single most valuable missing experiment, in priority order:

1. **Temporal-encoder-only ablation.** Per-node LSTM with **no** graph convolution at all. If this gets RMSE near 12, then most of the apparent spatial gain is really a temporal-encoder gain and the framing of the project shifts. Without this control, "+3.088 RMSE from spatial structure" is partly unverified.
2. **3-seed reruns of Iter 4 and Iter 7.** Every result so far is from `seed=0`. A 0.028-RMSE gap (Iter 4 → Iter 7) is well within plausible seed noise.
3. **Sparser adjacency (K=3) on correlation graph.** Iter 4 hinted that topology is the dominant lever; lowering K reduces over-smoothing risk further.
4. **ChebConv K=2, DropEdge, learned adjacency.** All on the backlog in `program.md`; none touched.
5. **Iter 6 (hidden=96) under 10-epoch budget.** Failed once at 5 epochs; the failure mode was overfitting at the end of the budget — exactly the case where more epochs *might* close the gap.

## Procedural lessons

- **Track the substrate explicitly.** Every iteration's log includes `seed`, `lr`, `batch_size`, `epochs`, etc. This is what made the matrix row-to-row comparable and what made attribution possible.
- **Diagnose before changing code.** Iter 2 was preceded by an explicit written diagnosis of the broadcasting bug. The fix landed once. Compare: the nine pre-iteration logs (`spatial_iter_001` … `_009`) include six identical re-runs of the same broken config — what blind sweeping looks like.
- **Watch all three metrics.** RMSE selection alone made the loop almost miss Iter 5's calibration failure (MAE rose +0.74 mph while RMSE moved <0.05). The keep/discard gate is correct, but the human reader should always glance at MAE too.
- **A weak keep is a warning.** Iter 3 (+0.028), Iter 7 (+0.028) — when gains shrink to noise scale, the architecture is signaling that the next move must be structural, not parametric.

## Where to go next

The architecture has plateaued. Three branches are worth opening in priority order:

1. The temporal-encoder-only ablation, **before** anything else — to verify how much of the spatial gain is actually spatial.
2. Multi-seed runs to put a confidence interval on the 11.912 number.
3. Structural changes to the spatial side: sparser adjacency, learned adjacency, or ChebConv. The Iter 7 plateau says "more epochs won't help"; only the receptive field will.
