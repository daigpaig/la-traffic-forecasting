# Program: ST-GNN Auto-Researcher for METR-LA  (revised 2026-05-21)

## Mission

Beat the temporal-only LSTM baseline (test RMSE **15.000 mph**) on METR-LA using Graph Neural Networks, **and** decompose the gain into temporal-encoder vs. graph-topology vs. graph-operator components. The decomposition is now a first-class deliverable, not an afterthought — see `project_statement.md` §3.

## Current state (Iter 7, commit `702f7b7`)

| | RMSE | MAE | R² | Config |
|---|---|---|---|---|
| Temporal baseline | 15.000 | 8.912 | 0.567 | LSTM hidden=64, layers=1 |
| **Spatial best (Iter 7)** | **11.912** | 6.244 | 0.727 | Per-node LSTM + GAT (1 head) + skip, K=5 correlation, hidden=64, 10 epochs |

Validation MSE plateaus near 0.347 at epoch 7. **The architecture has saturated at single-seed**; further gains need structural change (receptive field, learned adjacency) or multi-seed re-evaluation.

---

## Setup

```bash
# Branch per research tag (e.g., may7-gnn)
git checkout autoresearch/may7-gnn

# Results log (already exists)
cat results.tsv
```

**In-scope (sandbox):** `experiments/spatio_temporal/train.py` — change anything here.

**Off-limits (fixed):** `shared/data_loader.py`, `shared/evaluation.py`, `shared/spatial_utils.py`. Do not touch.

---

## The Loop (revised)

```
LOOP:
```

**1. Read.** Check `results.tsv`, `notes.md`, `failure_memo.md`, `what_worked_memo.md`. What failed, what was close, what hasn't been tried.

**2. Hypothesize.** One change. Write the hypothesis + expected RMSE band + discard threshold in `notes.md` *before* touching code.

**3. Hack.** Modify `experiments/spatio_temporal/train.py` only.

**4. Commit.**
```bash
git commit -am "experiment: <brief description>"
```

**5. Run.**
```bash
python experiments/spatio_temporal/train.py \
  --epochs <N> --run-name <tag> [other flags] > run.log 2>&1
```
Time budget per run: **target 7 min, hard kill 10 min** unless the hypothesis explicitly requires longer (e.g., Iter 7-style longer-training runs).

**6. Parse result.**
```bash
grep "^test_rmse" run.log
# Format: test_rmse X.XXXXXX  test_mae X.XXXXXX  test_r2 X.XXXX  runtime_sec XX.XX
```

**7. Logic gate (revised — now multi-metric).**

- **KEEP if:** `test_rmse < current_best` *and* `test_mae` did not rise by >0.3 mph vs current best. Watching MAE catches the calibration-collapse failure mode (T2) that pure RMSE selection missed in Iter 5.
- **DISCARD otherwise:** `git reset --hard HEAD~1`. Log status `discard` with a one-line reason.
- **Simplicity gate (new, enforced):** for any kept change with `Δ RMSE < 0.1`, also require `runtime ≤ 1.5× previous best`. If not, log as `keep-marginal` and flag for re-evaluation when surrounding architecture changes.

**8. Log.** Append to `results.tsv`:
```
<commit>   <test_rmse>   <runtime_sec>   keep|discard|keep-marginal   <description>
```
Then update `notes.md` with the post-hoc analysis (what the val_mse trajectory showed, why it worked or failed).

---

## Constraints (revised)

**Reproducibility.** Headline numbers must be multi-seed (≥3 seeds). Single-seed runs are allowed for *screening* hypotheses but cannot be the basis of a final claim.

**No pausing.** Don't ask permission to continue. If you run out of ideas, re-read this file and try a structural change from the backlog.

**Simplicity gate.** Enforced numerically (see step 7).

**Substrate.** Lr, batch, normalization, window, split protocol stay fixed unless the hypothesis is explicitly about them.

---

## Known failure modes (carried forward; see `failure_memo.md` for details)

- **T1 — Representation collapse** (Iter 1): broadcasting one embedding to all nodes destroys per-node identity. *Resolved* by per-node reshape in Iter 2; do not regress.
- **T2 — Optimization instability under short windows** (Iter 5): LSTM depth ≥ sequence length. *Detect via MAE rising while RMSE flat.*
- **T3 — Capacity overshoot at fixed budget** (Iter 6): width ↑ without epochs ↑. *Detect via val_mse divergence in last 1–2 epochs.* (Open: re-test under longer training.)
- **T4 — Topology misspecification** (Iter 3 partial): adjacency dominates operator choice. *When changing the operator, also re-test at least one adjacency variant.*
- **T5 — Compute-quality mismatch** (Iter 3): kept on RMSE alone but cost-inefficient. *Now enforced by the simplicity gate above.*

---

## Hypothesis backlog (re-ordered 2026-05-21)

Re-ordered around what the Week 4/5 record actually showed. Ablations and methodological controls now precede new architectures, because no further structural change is defensible without them.

**Tier 1 — Methodological controls (run first, before any new architecture):**
- [ ] **Temporal-encoder-only ablation.** Per-node LSTM, no graph layer. If RMSE ≈ 12, the project's framing changes.
- [ ] **Multi-seed (≥3) re-run of Iter 7.** Put a standard deviation on the headline 11.912 number.
- [ ] **Multi-seed re-run of Iter 4.** Bound the Iter 4 → Iter 7 gap (0.028 RMSE) against seed noise; decides whether 10 epochs is "real" or noise.

**Tier 2 — Structural changes to the spatial side:**
- [ ] **Sparser correlation graph (K=3).** Topology was the dominant lever; reducing K reduces over-smoothing further.
- [ ] **Two GAT layers + per-layer skip.** Expand receptive field 1-hop → 2-hop; previously drafted as Iter 8 in `notes.md` but not yet run.
- [ ] **ChebConv K=2.** Localized spectral filter; less global smoothing than GCN, cheaper than GAT.
- [ ] **Learned adjacency.** Parameterize a K=5 mask end-to-end. Highest-payoff but highest-risk; defer until Tier 1 done.

**Tier 3 — Revisit prior discards under new substrate:**
- [ ] **Iter 6 (hidden=96) at 10-epoch budget with early stopping.** Failed once at 5 epochs by overfitting; the mode was end-of-budget divergence, exactly the case more epochs + early stop might fix.
- [ ] **DropEdge regularization.** Drop a fraction of edges each batch; mitigates over-smoothing across deeper graph stacks.

---

## Agent strategy (revised)

The agent's behavior changes between Tiers:

**Tier 1 runs** are not subject to the keep/discard gate — they are measurements, not optimizations. Always commit, always log, never `git reset`. Their job is to put error bars on existing claims, not to advance the branch.

**Tier 2 runs** use the (revised) multi-metric logic gate from §The Loop, step 7.

**Tier 3 runs** require a written justification in `notes.md` for why this previously-discarded change might succeed under the new substrate — otherwise they're noise.

The agent should **not** propose a Tier 2 or Tier 3 change while a Tier 1 deliverable is still missing. The order is enforced because the value of any further architectural improvement is conditional on the spatial contribution being real, which Tier 1 settles.
