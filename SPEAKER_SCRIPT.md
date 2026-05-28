# STAT 390 — Speaker Script

_Total ~6:50 at normal pace. Time cue at top of each slide._

---

## Slide 1: Title

[~30s]

The task is to predict freeway speeds in Los Angeles one hour ahead, across two hundred and seven sensors, using only the previous hour of speeds.

Over sixteen iterations of experiments, test error dropped from fifteen miles per hour at the baseline to about eleven point nine. The more important result, which I'll walk through, is that around ninety-four percent of that improvement came from a change that has nothing to do with the graph structure itself.

---

## Slide 2: Formulation

[~40s]

My initial research question was the standard one: can a spatio-temporal graph network outperform a temporal-only LSTM on this dataset? Implicit in that framing is the assumption that any improvement comes from the spatial component.

A few weeks in, the question I was actually able to answer was different. It became: where does the improvement come from — the temporal encoder, the graph topology, or the graph operator — and how much of each effect is statistically real, versus seed noise.

The setup that made this possible: a fixed dataset and evaluation protocol, one mutable file, and a deliverable that decomposes contributions rather than reporting a single best number.

---

## Slide 3: Data & Baseline

[~45s]

A short grounding in the data. Everything on the left is held constant across every run, which is what makes the results comparable from one iteration to the next. The dataset is METR-LA freeway speeds — thirty-four thousand timesteps across two hundred and seven sensors, at five-minute resolution. The forecasting window is one hour of input, predicting one hour ahead.

One detail worth noting is normalization. I compute the z-score statistics from the training split only, which prevents temporal leakage. Evaluation is denormalized back to miles per hour, so all the metrics on later slides are directly interpretable.

The baseline is a one-layer LSTM that treats the sensors as features. It scores fifteen RMSE, around nine MAE, and an R-squared of zero point five-seven. It's a deliberately simple temporal floor — every later number in the project is measured against it.

---

## Slide 4: Loop Design

[~60s]

This is the structure of the loop. I want to spend a little time on it because the attribution claims later depend on it.

The design is intentionally minimal — one file, one metric, and fast iterations. Each change has to defend itself on its own commit, which prevents two improvements from being quietly bundled together. The cycle is: read previous results, write down a hypothesis, edit the code, commit, run, parse the log, and evaluate against the logic gate.

The gate has two conditions: a change is kept only if RMSE improves and mean absolute error doesn't rise by more than three tenths of a mile per hour. The MAE guard catches calibration regressions — that's the failure mode that ended iteration five. A separate simplicity gate flags changes that improve accuracy only marginally but increase runtime substantially.

On the right is the tier system. Tier one runs are methodological controls — ablations and multi-seed reruns — and these are never discarded. Tier two and three are optimization runs, subject to the gate. The rule is that no architecture experiment runs while a Tier-one measurement is still open, because the value of any architectural change depends on the spatial contribution being real.

---

## Slide 5: Experiment Trace

[~45s]

This is the experiment trace. Each point is one completed iteration plotted at its test RMSE, color-coded by the decision made.

Iteration one used a flawed architecture and scored worse than the baseline, at eighteen point eight. Iteration two corrected the architecture and dropped to twelve point one. That single transition accounts for nearly all of the improvement in the project. Every subsequent iteration moves the metric by less than two tenths of a point.

The grey point at iteration nine is the one to flag. That was an ablation using the same per-node temporal encoder but no graph layer at all. It lands almost exactly at iteration two's score, which sets up the result on the next slide.

---

## Slide 6: Final Result

[~60s]

This slide presents the main result. Iteration nine — the ablation with no graph layer — scored twelve point zero eight. That means the graph layer, added on top of the temporal encoder, accounts only for the difference between twelve point zero eight and eleven point nine two.

Decomposed against the baseline: about ninety-four percent of the three-point-one improvement comes from the per-node temporal encoder, and roughly five percent from the correlation graph. The five-percent contribution is statistically real — about eight point six sigma across seeds — but it's a small effect, and the original framing of the project did not correctly identify where the improvement was coming from.

I can make that decomposition with confidence because the seed-noise floor is about two-hundredths of a point of RMSE and a quarter point of MAE. The iteration-eight result that initially looked like an MAE improvement was within that noise band on a single seed.

---

## Slide 7: Worked vs Failed

[~55s]

Two changes produced most of the improvement. The first is per-node temporal encoding. In the original architecture, every sensor was passed through one shared LSTM and a single embedding was broadcast across all of them, which meant the graph layer was averaging two hundred near-identical vectors. Giving each sensor its own pass through the LSTM produced the only step-function improvement in the project. The second is correlation-based adjacency — choosing neighbors that behave similarly outperformed using physically adjacent ones, at no additional computational cost.

On the right are five named failure modes, each grounded in a specific iteration: representation collapse, optimization instability, capacity overshoot, topology misspecification, and compute-quality mismatch. I'll also note two procedural issues: single-seed evaluation early on, and a logging bug that re-ran the same configuration multiple times.

---

## Slide 8: Reflection

[~55s]

To close, four methodological lessons.

First, the research question changed during the project — an ablation, rather than an optimization step, was what reframed it. Second, the keep-discard gate is necessary but not sufficient; selection on a single seed produces apparent statistical significance, which is why I now require at least three seeds for small claims. Third, monitoring a single metric is risky — RMSE alone nearly missed a calibration failure in iteration five. Fourth, compute-quality tradeoffs need an explicit gate of their own.

The broader takeaway is that a logic-gated experimental loop can still be misled if the gate isn't designed to ask whether a result is real, in addition to whether it's better.

On limitations: the defensible final number is eleven point nine two RMSE. The project uses one dataset, a fixed window, and CPU compute only. Thank you.

---
