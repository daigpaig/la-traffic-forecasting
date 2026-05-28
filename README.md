# Where Does the Gain Come From? Spatio-Temporal Forecasting on METR-LA

STAT390 capstone. Quantifies, via a 16-iteration controlled ablation, where the improvement of a spatio-temporal GNN over a temporal-only LSTM actually comes from on the METR-LA traffic-forecasting benchmark.

**Headline result (test RMSE, mph):**

| Stage | RMSE | Source of gain |
|---|---|---|
| Temporal-only LSTM baseline | 15.000 | — |
| Per-node LSTM (no graph) | 12.084 | **94% of total gain** |
| + correlation-GAT K=5, 3-seed | **11.920 ± 0.019** | + 5% from graph |

Operator choice (GAT vs. ChebConv K=2) and most sub-0.05 RMSE deltas are within seed noise.

## Deliverables

| File | What it is |
|---|---|
| `report/main.pdf` | Final 4-page NeurIPS-format report + appendices (experiments record, results table, reflection memo, reproducibility) |
| `report/main.tex` | LaTeX source for the report |
| `experiments/spatio_temporal/results.tsv` | Raw log of every committed iteration |
| `experiments/spatio_temporal/notes.md` | Per-iteration hypothesis, configuration, result, decision |
| `experiments/spatio_temporal/ablation_table.md` | Controlled comparison matrix |
| `experiments/spatio_temporal/program.md` | Agent-loop protocol (Tier 1 / Tier 2 discipline) |
| `experiments/spatio_temporal/project_statement.md` | Revised research questions and success criteria |

## Repository layout

```
.
├── README.md                       # this file
├── CLAUDE.md                       # agent-collaboration notes
├── requirements.txt                # pip deps
├── data/                           # METR-LA cache (gitignored; auto-downloaded)
├── shared/                         # data, eval, graph utils used by both tracks
│   ├── data_loader.py              # split-safe windowing + per-sensor z-score
│   ├── evaluation.py               # RMSE / MAE / R²
│   ├── metr_la_dataset.py          # PyG-compatible loader
│   └── spatial_utils.py            # physical & correlation adjacency
├── experiments/
│   ├── temporal_only/              # B0 baseline (Week 3)
│   │   └── train.py                # vanilla LSTM
│   └── spatio_temporal/            # main project track
│       ├── train.py                # per-node LSTM + GAT/ChebConv
│       ├── results.tsv             # raw per-iteration log
│       ├── notes.md                # per-iteration write-ups (Iter 1–16)
│       ├── ablation_table.md       # one-factor-at-a-time matrix
│       ├── program.md              # agent loop / decision protocol
│       ├── project_statement.md    # revised research statement
│       ├── two_week_plan.md        # Week 6 / Week 7 schedule
│       ├── what_worked_memo.md     # mid-project attribution memo
│       ├── failure_memo.md         # T1–T5 failure taxonomy
│       ├── experiment_log_bundle.md# pre-iteration exploratory runs
│       └── results_plot.png        # iteration trajectory figure
└── report/
    ├── main.tex                    # NeurIPS-format report source
    ├── main.pdf                    # compiled report (4 pages + appendices)
    ├── neurips_2024.sty            # NeurIPS style file
    └── results_plot.png            # figure (copy for compile)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

METR-LA is downloaded automatically on first run into `data/METR_LA/`.

## Reproducing the headline number

The reported headline of `RMSE 11.920 ± 0.019` is the 3-seed mean ± sample std of the Iter-7 configuration (per-node 1-layer LSTM, hidden 64, 1× GAT, correlation adjacency K=5, 10 epochs).

```bash
for s in 0 1 2; do
  python experiments/spatio_temporal/train.py \
      --epochs 10 --hidden 64 --lstm-layers 1 \
      --conv gat --adj correlation --k 5 \
      --seed $s --run-name iter7_s${s}
done
```

Each row of `experiments/spatio_temporal/results.tsv` carries the commit hash; `git checkout <hash>` recovers the exact code that produced that row.

## Reproducing the report PDF

The report compiles with `tectonic` (sudo-free LaTeX engine):

```bash
brew install tectonic       # macOS
cd report
tectonic main.tex
```

Anywhere else, any modern LaTeX distribution (TeX Live 2022+, MacTeX) compiles `main.tex` directly:

```bash
cd report
pdflatex main.tex
```

The report uses NeurIPS final mode (`\usepackage[final]{neurips_2024}`), which shows author info. To produce an anonymized version for review, remove the `[final]` option from that line and recompile.

## Methodology in one paragraph

Each iteration is a controlled one-factor-at-a-time change relative to the previous best. Tier 1 (methodological controls — temporal-only ablation and multi-seed re-runs of headline configurations) are committed unconditionally. Tier 2 (architectural changes) use single-seed screening; a screening result earns multi-seed confirmation only if it exceeds the seed-noise floor (currently $\sigma_{\mathrm{RMSE}} = 0.019$). The seed-noise floor was established post-hoc from the 3-seed Iter-7 study and is what permits attribution at $\pm 0.02$ resolution. Full protocol: `experiments/spatio_temporal/program.md`.
