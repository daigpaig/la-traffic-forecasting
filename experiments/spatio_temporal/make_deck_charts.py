"""Generate clean-academic dot-plot charts for the STAT390 AutoResearch deck."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, FancyArrowPatch
from matplotlib.lines import Line2D
import os

OUT = "/sessions/funny-amazing-davinci/mnt/outputs"
os.makedirs(OUT, exist_ok=True)

NAVY  = "#1F3A5F"; KEEP = "#2A9D8F"; DISC = "#C44536"
MARG  = "#E9A23B"; TIER1 = "#8A94A6"; GRID = "#D7DCE2"
TEXT  = "#22303F"; MUTED = "#5C6B7A"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 13,
    "text.color": TEXT, "axes.edgecolor": "#B7C0CB",
    "axes.labelcolor": TEXT, "xtick.color": TEXT, "ytick.color": TEXT,
    "axes.linewidth": 1.0,
})

# ----------------------------------------------------------------------
# CHART 1 — Dot plot: RMSE by iteration
# ----------------------------------------------------------------------
iters  = ["1","2","3","4","5","6","7","8","9","14","15"]
labels = ["1","2","3","4","5","6","7","8","9\n(no graph)","14","15"]
rmse   = [18.755,12.083,12.055,11.940,11.990,12.122,11.912,11.896,12.084,11.957,11.869]
status = ["disc","keep","marg","keep","disc","disc","keep","keep","tier1","disc","marg"]
cmap = {"keep":KEEP,"disc":DISC,"marg":MARG,"tier1":TIER1}
mmap = {"keep":"o","disc":"X","marg":"D","tier1":"s"}

fig, ax = plt.subplots(figsize=(12, 6.2))

# baseline line
ax.axhline(15.000, color=NAVY, ls="--", lw=1.8, zorder=2)
ax.text(len(iters)-0.6, 15.18, "Temporal LSTM baseline   15.00 mph",
        ha="right", va="bottom", color=NAVY, fontsize=12, fontweight="bold")

# thin connecting trace
ax.plot(range(len(iters)), rmse, color="#C9D2DC", lw=1.6, zorder=2)

# dots
for i, (v, st) in enumerate(zip(rmse, status)):
    ax.scatter(i, v, s=240, marker=mmap[st], color=cmap[st],
               edgecolor="white", linewidth=1.6, zorder=4)
    dy = 0.30 if v < 14 else -0.55
    ax.text(i, v + dy, f"{v:.2f}", ha="center",
            va="bottom" if dy > 0 else "top",
            fontsize=11, fontweight="bold", color=TEXT, zorder=5)

# highlight iter 9 = iter 2 callout
i9, i2 = 8, 1
ax.annotate("Iter 9 (no graph) lands\non top of Iter 2",
            xy=(i9, 12.084), xytext=(6.4, 13.4),
            fontsize=11, color=MUTED, fontstyle="italic",
            ha="center",
            arrowprops=dict(arrowstyle="-", color=TIER1, lw=1.2, linestyle="--"))
ax.plot([i2, i9], [12.083, 12.084], color=TIER1, lw=1.2, ls=":", alpha=0.7, zorder=3)

ax.set_ylim(11, 19.6)
ax.set_xticks(range(len(iters)))
ax.set_xticklabels(labels, fontsize=11.5)
ax.set_ylabel("Test RMSE (mph)", fontsize=14, fontweight="bold")
ax.set_xlabel("Iteration", fontsize=14, fontweight="bold")
ax.set_title("One step-function improvement, then sub-noise drift",
             fontsize=15, fontweight="bold", color=NAVY, pad=14)
ax.yaxis.grid(True, color=GRID, lw=0.9, zorder=0)
ax.set_axisbelow(True)
for sp in ["top","right"]: ax.spines[sp].set_visible(False)

legend = [
    Line2D([0],[0], marker="o", color="w", markerfacecolor=KEEP,  markersize=11, label="Keep"),
    Line2D([0],[0], marker="D", color="w", markerfacecolor=MARG,  markersize=10, label="Keep-marginal"),
    Line2D([0],[0], marker="X", color="w", markerfacecolor=DISC,  markersize=12, label="Discard"),
    Line2D([0],[0], marker="s", color="w", markerfacecolor=TIER1, markersize=11, label="Tier-1 ablation"),
]
ax.legend(handles=legend, loc="upper right", frameon=True, framealpha=0.95,
          edgecolor=GRID, fontsize=11.5, ncol=2)
fig.tight_layout()
fig.savefig(f"{OUT}/chart_rmse_by_iter.png", dpi=200, facecolor="white")
plt.close(fig)

# ----------------------------------------------------------------------
# CHART 2 — Decomposition dot plot
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 6.2))

xs = [0, 1, 2]
ys = [15.000, 12.084, 11.920]
labels2 = ["Temporal\nbaseline", "+ Per-node\ntemporal encoder", "+ Correlation\ngraph layer"]
colors2 = [NAVY, KEEP, MARG]

# step lines connecting stages
for i in range(len(xs)-1):
    # horizontal segment at the higher level
    ax.plot([xs[i], xs[i+1]], [ys[i], ys[i]], color="#C9D2DC", lw=1.4, ls=":", zorder=2)
    # vertical drop
    ax.plot([xs[i+1], xs[i+1]], [ys[i], ys[i+1]], color=colors2[i+1], lw=2.2, zorder=3)
    # delta label
    delta = ys[i+1] - ys[i]
    xmid = xs[i+1] + 0.06
    ymid = (ys[i] + ys[i+1]) / 2
    txt = f"{delta:+.2f} RMSE"
    if i == 0:
        ax.text(xmid, ymid, txt + "\n(94% of gain)", fontsize=13.5,
                fontweight="bold", color=KEEP, va="center", ha="left")
    else:
        ax.text(xmid, ymid, txt + "\n(5%, 8.6σ)", fontsize=12.5,
                fontweight="bold", color="#9A6B14", va="center", ha="left")

# dots
for x, y, c in zip(xs, ys, colors2):
    ax.scatter(x, y, s=420, color=c, edgecolor="white", linewidth=2.2, zorder=5)
    ax.text(x, y + 0.32, f"{y:.2f}", ha="center", va="bottom",
            fontsize=14, fontweight="bold", color=TEXT)

# error bar on graph stage (±0.019)
ax.errorbar(xs[2], ys[2], yerr=0.019, fmt="none",
            ecolor=MARG, elinewidth=2, capsize=6, capthick=2, zorder=4)
ax.text(xs[2], ys[2] - 0.32, "± 0.019 (3-seed)", ha="center", va="top",
        fontsize=10.5, color=MUTED, fontstyle="italic")

ax.set_xticks(xs)
ax.set_xticklabels(labels2, fontsize=12.5)
ax.set_xlim(-0.45, 2.65)
ax.set_ylim(11, 16.2)
ax.set_ylabel("Test RMSE (mph)", fontsize=14, fontweight="bold")
ax.set_title("Decomposing the 3.08 RMSE improvement over baseline",
             fontsize=15, fontweight="bold", color=NAVY, pad=14)
ax.yaxis.grid(True, color=GRID, lw=0.9, zorder=0)
ax.set_axisbelow(True)
for sp in ["top","right"]: ax.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig(f"{OUT}/chart_decomposition.png", dpi=200, facecolor="white")
plt.close(fig)

print("charts written")
