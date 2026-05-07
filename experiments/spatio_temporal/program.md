# Program: ST-GNN Auto-Researcher for METR-LA

## Mission

Beat the temporal-only LSTM baseline (test RMSE **15.000 mph**) on METR-LA using Graph Neural Networks. Quantify whether spatial structure helps and, if so, which architecture choices drive the gain.

---

## Setup

```bash
# Branch per research tag (e.g., may7-gnn)
git checkout -b autoresearch/<tag>

# Initialize results log (tab-separated)
echo -e "commit\ttest_rmse\truntime_sec\tstatus\tdescription" > results.tsv
```

**In-scope (sandbox):** `experiments/spatio_temporal/train.py` — change anything here.

**Off-limits (fixed):** `shared/data_loader.py`, `shared/evaluation.py`, `shared/spatial_utils.py`. Do not touch these.

---

## The Loop

```
LOOP FOREVER:
```

**1. Read.** Check `results.tsv` and `notes.md`. What failed, what was close, what hasn't been tried.

**2. Hypothesize.** One spatial hypothesis. Write the reasoning in `notes.md` before touching code.

**3. Hack.** Modify `experiments/spatio_temporal/train.py`.

**4. Commit.**
```bash
git commit -am "experiment: <brief description>"
```

**5. Run.**
```bash
python experiments/spatio_temporal/train.py \
  --epochs <N> --run-name <tag> [other flags] > run.log 2>&1
```
Time budget: **7 minutes**. If the process exceeds 10 minutes, kill it and log as `crash`.

**6. Parse result.**
```bash
grep "^test_rmse" run.log
# Output format: test_rmse X.XXXXXX  test_mae X.XXXXXX  test_r2 X.XXXX  runtime_sec XX.XX
```
If the line is missing or the file ends in a traceback: read the last 50 lines, fix a trivial typo if obvious, otherwise `git reset --hard HEAD~1` and move on.

**7. Logic gate.**

- `test_rmse < current_best`: **KEEP.** Update `results.tsv` with status `keep`. You have advanced the branch.
- `test_rmse >= current_best`: **DISCARD.** `git reset --hard HEAD~1`. Log with status `discard` and one-line reason.

**8. Log.**
```
<commit>   <test_rmse>   <runtime_sec>   keep|discard   <description>
```

---

## Constraints

**Simplicity gate.** A 0.001 RMSE gain that adds ugly complexity is a `discard`. A 0.001 gain from *deleting* code is a `keep`. Prefer the smaller, cleaner model.

**No pausing.** Do not ask for permission to continue. If you run out of ideas, re-read this file and try a more radical change (different conv operator, different adjacency, learned graph structure).

**Reproducibility.** Always pass `--seed 0`. Results that can't be reproduced don't count.

**VRAM.** If a run OOMs, it's a crash. Reduce `--hidden` or `--batch-size` and retry.

---

## Known Failure Modes

**Spatial blurring.** GCN aggregates neighbors and over-smooths node representations — nodes lose their individual identity. Symptom: more neighbors → worse RMSE. Fix: reduce K, use shallower graphs, add residual skip connections.

**Information loss (broadcasting).** Encoding all 207 sensors into a single LSTM hidden state, then broadcasting to all nodes, means every node gets the same temporal embedding before GCN. The spatial message-passing becomes meaningless. Fix: per-node LSTM (`(B*N, T, 1)` reshape before encoding).

**Adjacency mismatch.** Physical road distance ≠ traffic speed correlation. Two sensors 1 km apart on parallel roads may be uncorrelated; two sensors 10 km apart on the same highway may be tightly coupled. Fix: try correlation adjacency.

---

## Hypotheses Backlog

Roughly ordered by expected payoff. Cross off as you go.

- [ ] **Graph Attention (GAT):** learned edge weights should select useful neighbors instead of treating all K equally
- [ ] **Correlation adjacency:** build graph from Pearson correlation of training speeds; may better reflect actual road influence
- [ ] **Deeper per-node LSTM (2 layers):** more temporal capacity per sensor before spatial aggregation
- [ ] **Longer training:** val_mse was still declining at epoch 10 in iteration 2 — squeeze more out of the current architecture
- [ ] **ChebConv (K=2):** localized spectral filters; less global smoothing than standard GCN
- [ ] **Dropout on graph edges (DropEdge):** regularizes spatial aggregation, prevents over-reliance on any single neighbor
- [ ] **Larger hidden (96, 128):** now that per-node architecture is correct, capacity may matter
- [ ] **Learned adjacency:** parameterize the adjacency matrix and learn it end-to-end

---

## Current Best

| | RMSE | Config |
|---|---|---|
| Temporal baseline | 15.000 | LSTM hidden=64, layers=1 |
| Spatial best (iter 2) | **12.083** | Per-node LSTM + GCN skip, hidden=64, K=5, epochs=10 |

Spatial is currently **+2.917 ahead** of the temporal baseline.
