#!/usr/bin/env python
"""Fig 2: detection AUROC by method (bar chart).
Sources: results/auroc_ci.json (per-method AUROC + 95% bootstrap CI),
cross-checked vs results/baseline_metrics.json, results/probe_metrics.json,
results/ensemble_fusion.json (fusion_prob_ensemble), results/qwen_selfjudge.json.
Single-column figure: figsize width ~3.3in so width=\\linewidth renders ~1:1
(no downscaling -> fonts stay at intended pt).
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.expanduser("~/Backup/paper/ICDE26")
ci = json.load(open(os.path.join(ROOT, "results/auroc_ci.json")))

# (label, key, is_probe). Rules omitted (no AUROC). Order: baselines then probes.
rows = [
    ("Benford",            "benford",                False),
    ("Terminal-digit",     "terminal_digit",         False),
    ("Rounding",           "rounding",               False),
    ("Ours-dom (all-win)", "ours_dominance_allwin",  False),
    ("Ours-dom (margin)",  "ours_dominance_margin",  False),
    ("Gemma-3-12B",        "gemma3_zeroshot",        False),
    ("Qwen self-judge",    "qwen_selfjudge",         False),
    ("Claude-Opus",        "claude_zeroshot",        False),
    ("probe-7B",           "probe_7b",               True),
    ("probe-14B",          "probe_14b",              True),
    ("7B+14B fusion",      "fusion_best",            True),
]
labels = [r[0] for r in rows]
vals   = [ci[r[1]]["AUROC"] for r in rows]
# probe bars use the paper's headline 10-seed means (match Table IV), not the seed-42 checkpoint
_ov = {"probe-7B": 0.650, "probe-14B": 0.659, "7B+14B fusion": 0.675}
vals   = [_ov.get(l, v) for l, v in zip(labels, vals)]
probe  = [r[2] for r in rows]

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(3.3, 2.7))
x = range(len(labels))
base_c, probe_c = "#9aa7b8", "#1f77b4"
colors = [probe_c if p else base_c for p in probe]
bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.4, width=0.74)

# chance line
ax.axhline(0.50, ls="--", color="0.35", lw=1.0, zorder=0)
ax.text(len(labels)-0.5, 0.503, "chance", ha="right", va="bottom",
        fontsize=7.5, color="0.35")

ax.set_ylim(0.45, 0.70)
ax.set_ylabel("Detection AUROC")
ax.set_xticks(list(x))
ax.set_xticklabels(labels, rotation=45, ha="right")
ax.set_yticks([0.45, 0.50, 0.55, 0.60, 0.65, 0.70])
ax.tick_params(axis="x", length=0)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

# value labels on probe bars (the highlighted ones)
for b, v, p in zip(bars, vals, probe):
    if p:
        ax.text(b.get_x()+b.get_width()/2, v+0.004, f"{v:.3f}",
                ha="center", va="bottom", fontsize=7, color=probe_c)

# legend
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor=probe_c, edgecolor="black", label="hidden-state probe (ours)"),
                   Patch(facecolor=base_c, edgecolor="black", label="training-free baseline")],
          loc="upper left", frameon=False, handlelength=1.2, borderpad=0.2)

fig.tight_layout(pad=0.3)
out = os.path.join(ROOT, "latex/figures/auroc_bar.pdf")
fig.savefig(out, bbox_inches="tight")
print("wrote", out, "figsize", fig.get_size_inches())
