"""Regenerate results_plot.png with all current iterations (1..N)."""
from pathlib import Path
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent

iters = [
    dict(n=1, rmse=18.755, mae=11.872, status="discard", label="Broken arch\n(broadcast LSTM)"),
    dict(n=2, rmse=12.083, mae=6.236,  status="keep",    label="Per-node LSTM\n+ GCN skip"),
    dict(n=3, rmse=12.055, mae=5.846,  status="keep",    label="GAT 1-head\n+ skip"),
    dict(n=4, rmse=11.940, mae=5.772,  status="keep",    label="GAT + corr.\nadj"),
    dict(n=5, rmse=11.990, mae=6.511,  status="discard", label="2-layer LSTM\n(unstable)"),
    dict(n=6, rmse=12.122, mae=5.815,  status="discard", label="hidden=96\n(overfit)"),
    dict(n=7, rmse=11.912, mae=6.244,  status="keep",    label="10 epochs\n★ best"),
]

BASELINE = 15.000
KEEP_COLOR    = "#4ade80"
DISCARD_COLOR = "#f87171"
BASELINE_COLOR= "#fb923c"
MAE_COLOR     = "#c084fc"

plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(13, 7))

xs   = [r["n"]    for r in iters]
ys   = [r["rmse"] for r in iters]
maes = [r["mae"]  for r in iters]

ax.plot(xs, ys, color="#9ca3af", linewidth=1.5, zorder=1)

for r in iters:
    color = KEEP_COLOR if r["status"] == "keep" else DISCARD_COLOR
    marker = "o" if r["status"] == "keep" else "X"
    ax.scatter(r["n"], r["rmse"], s=160, c=color, marker=marker,
               edgecolors="white", linewidths=1.2, zorder=3)

ax.axhline(BASELINE, color=BASELINE_COLOR, linestyle="--", linewidth=2,
           label=f"Temporal baseline  {BASELINE:.1f}", zorder=2)

ax2 = ax.twinx()
ax2.plot(xs, maes, color=MAE_COLOR, linewidth=1.4, linestyle=":",
         marker="D", markersize=5, alpha=0.85, label="MAE (right axis)")
ax2.set_ylabel("MAE (mph)", color=MAE_COLOR)
ax2.tick_params(axis="y", colors=MAE_COLOR)
ax2.set_ylim(4.0, 12.5)

offsets = {
    1: ( 0.25, -0.4),
    2: ( 0.0,  -0.9),
    3: ( 0.0,   0.55),
    4: (-0.15, -0.95),
    5: ( 0.05,  0.55),
    6: (-0.2,   0.55),
    7: ( 0.05, -0.95),
}
for r in iters:
    dx, dy = offsets[r["n"]]
    edge = KEEP_COLOR if r["status"] == "keep" else DISCARD_COLOR
    ax.annotate(
        f"{r['rmse']:.3f}\n{r['label']}",
        xy=(r["n"], r["rmse"]),
        xytext=(r["n"] + dx, r["rmse"] + dy),
        fontsize=9, ha="center", va="center",
        color="white",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#1f2937",
                  edgecolor=edge, linewidth=1.2),
    )

best = min(iters, key=lambda r: r["rmse"])
delta = BASELINE - best["rmse"]
ax.annotate(
    f"Best: {best['rmse']:.3f} mph\n−{delta:.3f} vs baseline",
    xy=(best["n"], best["rmse"]),
    xytext=(best["n"] - 1.4, best["rmse"] + 1.6),
    fontsize=10, color=KEEP_COLOR, ha="center",
    bbox=dict(boxstyle="round,pad=0.45", facecolor="#0b1220",
              edgecolor=KEEP_COLOR, linewidth=1.6),
    arrowprops=dict(arrowstyle="->", color=KEEP_COLOR, lw=1.4),
)

ax.set_xlabel("Iteration")
ax.set_ylabel("Test RMSE (mph)")
ax.set_title("ST-GNN Auto-Research: RMSE by Iteration\nMETR-LA Traffic Forecasting")
ax.set_xticks(xs)
ax.set_xticklabels([f"Iter {n}" for n in xs])
ax.set_ylim(10, 20)
ax.grid(alpha=0.15)

from matplotlib.lines import Line2D
legend_handles = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=KEEP_COLOR,
           markeredgecolor="white", markersize=10, label="Keep", linestyle=""),
    Line2D([0], [0], marker="X", color="w", markerfacecolor=DISCARD_COLOR,
           markeredgecolor="white", markersize=10, label="Discard", linestyle=""),
    Line2D([0], [0], color=BASELINE_COLOR, linestyle="--", linewidth=2,
           label=f"Temporal baseline  {BASELINE:.1f}"),
    Line2D([0], [0], color=MAE_COLOR, linestyle=":", marker="D",
           markersize=5, label="MAE (right axis)"),
]
ax.legend(handles=legend_handles, loc="upper right", framealpha=0.9)

fig.tight_layout()
out = HERE / "results_plot.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"Wrote {out}")
