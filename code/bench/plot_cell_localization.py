"""plot_cell_localization.py — RQ5 figure: probe vs random baseline cell localization."""
import os, json
os.environ.setdefault("TMPDIR", "/data/users/pengminjie/Backup/tmp")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.family"] = "serif"
rcParams["font.size"] = 9
rcParams["axes.linewidth"] = 0.8
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42

ROOT = "/home/pengminjie/Backup/paper/ICDE26"
RES = ROOT + "/results/cell_localization.json"
OUT = ROOT + "/latex/figures/cell_localization.pdf"

d = json.load(open(RES))
pb = d["probe"]
rb = d["random_baseline"]

metrics = ["Top-1", "Top-3", "MRR", "Cell-AUROC"]
probe_vals = [pb["top1_recall"], pb["top3_recall"], pb["mrr"], pb["cell_auroc_pooled"]]
rand_vals = [rb["top1_recall"], rb["top3_recall"], rb["mrr"], rb["cell_auroc_pooled"]]

x = np.arange(len(metrics))
w = 0.38

fig, ax = plt.subplots(figsize=(3.5, 2.4))
c_probe = "#2c5f8a"
c_rand = "#bdbdbd"
b1 = ax.bar(x - w / 2, probe_vals, w, label="Hidden-state probe", color=c_probe,
            edgecolor="black", linewidth=0.5)
b2 = ax.bar(x + w / 2, rand_vals, w, label="Random", color=c_rand,
            edgecolor="black", linewidth=0.5)

for bars, vals in [(b1, probe_vals), (b2, rand_vals)]:
    for rect, v in zip(bars, vals):
        ax.annotate(f"{v:.2f}", xy=(rect.get_x() + rect.get_width() / 2, v),
                    xytext=(0, 1.5), textcoords="offset points",
                    ha="center", va="bottom", fontsize=6.5)

ax.axhline(0.5, color="0.55", lw=0.6, ls="--", zorder=0)
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylabel("Score")
ymax = max(max(probe_vals), max(rand_vals))
ax.set_ylim(0, min(1.05, ymax + 0.22))
ax.legend(frameon=False, fontsize=7, loc="upper left", ncol=1,
          handlelength=1.1, borderaxespad=0.2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(length=2.5)
fig.tight_layout(pad=0.4)
fig.savefig(OUT, bbox_inches="tight")
print("wrote", OUT)
print("probe", probe_vals)
print("rand ", rand_vals)
