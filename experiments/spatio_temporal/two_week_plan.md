# Final Two-Week Plan (2026-05-21 → 2026-06-04)

**Author:** Daigo Moriwake  ·  **Branch:** `autoresearch/may7-gnn`
**Current best:** Iter 7, test RMSE 11.912 mph (commit `702f7b7`)
**End-state goal:** A defensible write-up with multi-seed error bars, the temporal-encoder ablation done, and at least one Tier 2 structural change attempted — see `project_statement.md` §7 for the full success criteria.

The plan trades the open-ended "beat 11.912" framing for a methodologically tight close-out. Architecture hunting takes a back seat to measurement.

---

## Week 6  (2026-05-21 → 2026-05-28) — Measurement

**Theme:** Settle what is and isn't real about the existing results before adding anything new.

| Day | Deliverable | Owner | Est. compute | Done when |
|---|---|---|---|---|
| Thu 05-21 | This plan + revised `project_statement.md`, `program.md`, `ablation_table.md` committed | self | 0 | `git log` shows the four files in one commit |
| Fri 05-22 | **A1: temporal-encoder-only ablation** run, logged, written up in `notes.md` | self | ~600 s | `results.tsv` row added; one-paragraph result note in `notes.md` |
| Sat 05-23 | **A4: GAT vs GCN at matched epochs=5** (isolates the Iter 2→3 operator effect from the training-length confound) | self | ~570 s | `results.tsv` row added |
| Sun 05-24 | **A2: Iter 7 config, seeds {0, 1, 2}** | self | ~6900 s (overnight) | mean ± std in `ablation_table.md` |
| Mon 05-25 | **A3: Iter 4 config, seeds {0, 1, 2}** | self | ~2860 s | mean ± std in `ablation_table.md`; seed-noise scale of the Iter 4↔7 gap is now known |
| Tue 05-26 | Update `ablation_table.md` §B with the corrected attributions; flag any claims that need re-wording in the final write-up | self | 0 | §B columns "marginal Δ RMSE" updated; new "verdict" column |
| Wed 05-27 | **Week 6 checkpoint memo** (`week6_memo.md`): which Tier 1 conclusions hold, which surprised, what changes in the final write-up | self | 0 | Memo committed, ≤ 2 pages |
| Thu 05-28 | Decide which **one** Tier 2 change to run in Week 7 (S1, S2, or S3) based on Week 6 findings — record the decision and reasoning in `notes.md` | self | 0 | Decision logged with the reasoning, *before* any code change |

**Week 6 stop criteria** (one or more triggers an early stop and a re-plan):
- A1 returns RMSE > 12.5 → spatial contribution is real and meaningful; proceed as planned.
- A1 returns RMSE ≤ 12.0 → most of the gain is the temporal encoder; **re-frame** the final write-up before running anything else.
- A2 returns std > 0.10 RMSE → seed noise dominates the recent "keeps"; the project's claims about iter-to-iter improvement need to be re-stated in terms of confidence intervals, not point estimates.

---

## Week 7  (2026-05-29 → 2026-06-04) — One structural attempt + write-up

**Theme:** Take one defensible structural shot, then close out. No new ideas after Monday.

| Day | Deliverable | Owner | Est. compute | Done when |
|---|---|---|---|---|
| Fri 05-29 | Implement the Tier 2 change selected at end of Week 6 (S1, S2, or S3). Single commit; reviewable diff. | self | 0 (code only) | `train.py` changed and committed |
| Sat 05-30 | Run the selected Tier 2 change, single seed (screening). Apply the multi-metric logic gate from `program.md`. | self | ~2300 s | `results.tsv` row added with keep/discard/keep-marginal |
| Sun 05-31 | If kept: 2 additional seeds to confirm. If discarded: log the post-mortem in `notes.md` and move on. | self | 0 or ~4600 s | seed mean ± std logged, or post-mortem note added |
| Mon 06-01 | **First-draft final write-up** (`final_writeup.md`) — re-framed per the revised project statement. Pulls numbers from `ablation_table.md` only; no fresh experiments. | self | 0 | Draft committed; every number traces to a `results.tsv` row |
| Tue 06-02 | Self-review pass: walk every claim against the matrix; remove anything not backed by a row. | self | 0 | Diff against Mon draft shows only deletions / wording changes, no new claims |
| Wed 06-03 | **Reproducibility check:** rerun Iter 7 from a clean checkout, confirm RMSE within 0.01 of recorded. | self | ~2300 s | Repro log committed; matches `results.tsv` |
| Thu 06-04 | Final submission: `final_writeup.md`, plot regenerated to include Tier 1 + Tier 2 additions, all companion files referenced. | self | 0 | Final tag pushed |

**Hard freeze:** No new training runs after Wed 06-03 12:00 except the reproducibility re-run. Any further changes are documentation-only.

---

## Compute budget (estimate)

| Bucket | Runs | Wall-clock | Notes |
|---|---|---|---|
| Week 6 Tier 1 ablations | A1 + A4 + A2 (3 seeds) + A3 (3 seeds) | ~10 970 s ≈ 3.1 h | A2 is the only one that needs an overnight slot |
| Week 7 Tier 2 attempt | S* screening + ≤2 follow-up seeds | ~2300–6900 s ≈ 0.6–1.9 h | Range depends on keep/discard at screening |
| Reproducibility re-run | Iter 7 | ~2300 s ≈ 0.6 h | |
| **Total budget** | ~9 runs | ≈ 4.3–5.6 h CPU | Well under the prior weeks' burn rate |

## Risk register

| Risk | Probability | Mitigation |
|---|---|---|
| A1 reframes the project late (would force write-up changes) | medium | Run A1 *first* (Fri 05-22) so the framing is settled by Wed 05-27 |
| Tier 2 attempt fails to keep, no fallback ready | low | The plan does not require a Tier 2 win — discard with a written post-mortem is a valid outcome (§7c of `project_statement.md`) |
| Compute slot for A2 (overnight 3-seed run) is interrupted | low | A2 is the bulkiest single job; if interrupted, fall back to 2 seeds, document the reduced sample size |
| Reproducibility re-run produces RMSE differing from log by > 0.1 | low | Investigate immediately; non-reproducibility is itself a finding and must be reported in the write-up if it occurs |
| Scope creep — temptation to chase one more architecture in Week 7 | high | Hard freeze on new runs after 06-03 12:00. New ideas go into a "future work" section, not into the experiment queue. |

## What this plan deliberately excludes

- **No PEMS-BAY transfer experiments.** Out of scope; would require a fresh data pipeline.
- **No multi-step horizon ablation.** Window stays fixed at 12 → 12.
- **No new conv operators beyond the one Tier 2 choice.** ChebConv / DropEdge / learned adjacency are at most one of them, not all.
- **No re-runs of Iter 5 / Iter 6 under different training regimes.** They go into "future work" with a written reason.

The constraint is the schedule, not the ideas. Discipline here is the deliverable.
