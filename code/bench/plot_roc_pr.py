#!/usr/bin/env python
"""Fig 3: ROC (top) + Precision-Recall (bottom), STACKED single-column.
Source: results/ablations/roc_pr_curves.json
(keys probe_14b, probe_7b, gemma3_zeroshot, claude_opus_zeroshot, rounding;
each has roc{fpr,tpr}, pr{precision,recall}, AUROC, AUPRC).
Stacked layout (3.3w x 4.2h) so each panel is full column width at
width=\\linewidth -> ~1:1 render, large fonts.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.expanduser("~/Backup/paper/ICDE26")
d = json.load(open(os.path.join(ROOT, "results/ablations/roc_pr_curves.json")))

# (key, label, color, linewidth, zorder)
series = [
    ("probe_14b",           "14B probe",        "#1f77b4", 1.8, 5),
    ("probe_7b",            "7B probe",         "#2ca02c", 1.5, 4),
    ("claude_opus_zeroshot","Claude-Opus",      "#d62728", 1.2, 3),
    ("rounding",            "Rounding",         "#9467bd", 1.2, 2),
    ("gemma3_zeroshot",     "Gemma-3-12B",      "#8c564b", 1.2, 1),
]

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

fig, (ax_roc, ax_pr) = plt.subplots(2, 1, figsize=(3.3, 4.4))

# ---- ROC (top) full 0-1 with diagonal ----
ax_roc.plot([0, 1], [0, 1], ls="--", color="0.5", lw=1.0, zorder=0)
for k, lab, c, lw, z in series:
    s = d[k]
    ax_roc.plot(s["roc"]["fpr"], s["roc"]["tpr"], color=c, lw=lw, zorder=z,
                label=f"{lab} ({s['AUROC']:.3f})")
ax_roc.set_xlim(0, 1); ax_roc.set_ylim(0, 1)
ax_roc.set_xlabel("False positive rate")
ax_roc.set_ylabel("True positive rate")
ax_roc.set_title("ROC")
ax_roc.legend(loc="lower right", frameon=False, handlelength=1.3,
              labelspacing=0.25, borderpad=0.2)
for sp in ("top", "right"): ax_roc.spines[sp].set_visible(False)

# ---- PR (bottom) precision y-axis starts at 0.45, chance line at 0.50 ----
ax_pr.axhline(0.50, ls="--", color="0.5", lw=1.0, zorder=0)
ax_pr.text(0.02, 0.505, "chance", ha="left", va="bottom", fontsize=7, color="0.5")
for k, lab, c, lw, z in series:
    s = d[k]
    ax_pr.plot(s["pr"]["recall"], s["pr"]["precision"], color=c, lw=lw, zorder=z,
               label=f"{lab} ({s['AUPRC']:.3f})")
ax_pr.set_xlim(0, 1); ax_pr.set_ylim(0.45, 1.0)
ax_pr.set_xlabel("Recall")
ax_pr.set_ylabel("Precision")
ax_pr.set_title("Precision–Recall")
ax_pr.legend(loc="upper right", frameon=False, handlelength=1.3,
             labelspacing=0.25, borderpad=0.2)
for sp in ("top", "right"): ax_pr.spines[sp].set_visible(False)

fig.tight_layout(pad=0.4, h_pad=1.0)
out = os.path.join(ROOT, "latex/figures/roc_pr.pdf")
fig.savefig(out, bbox_inches="tight")
print("wrote", out, "figsize", fig.get_size_inches())
