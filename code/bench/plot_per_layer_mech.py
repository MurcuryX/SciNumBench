#!/usr/bin/env python
"""Fig 4: per-layer mechanism (3 curves over layers 0-48).
Source: results/mechanism_per_layer.json -> per_layer[] with
layer, linear_probe_auroc, verdict_proj_auroc, residual_probe_auroc.
Reference lines: self-judge 0.5123 (summary.self_judge_auroc), chance 0.50.
Single-column figsize width ~3.3in.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.expanduser("~/Backup/paper/ICDE26")
d = json.load(open(os.path.join(ROOT, "results/mechanism_per_layer.json")))
pl = d["per_layer"]
layer = [r["layer"] for r in pl]
lin   = [r["linear_probe_auroc"]   for r in pl]
res   = [r["residual_probe_auroc"] for r in pl]
vp    = [r["verdict_proj_auroc"]   for r in pl]
sj    = d["summary"]["self_judge_auroc"]   # 0.5123
chance= d["summary"]["chance"]             # 0.50

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(3.3, 2.6))

# reference lines
ax.axhline(chance, ls=":", color="0.45", lw=1.0, zorder=0)
ax.text(48, chance-0.003, "chance", ha="right", va="top", fontsize=7, color="0.45")
ax.axhline(sj, ls="--", color="#d62728", lw=1.0, zorder=0)
ax.text(0, sj+0.002, f"self-judge {sj:.3f}", ha="left", va="bottom",
        fontsize=7, color="#d62728")

ax.plot(layer, lin, color="#1f77b4", lw=1.8, label="linear probe", zorder=5)
ax.plot(layer, res, color="#2ca02c", lw=1.3, ls="--", label="residual probe (ablated)", zorder=4)
ax.plot(layer, vp,  color="#ff7f0e", lw=1.5, label="verdict projection", zorder=3)

ax.set_xlim(0, 48); ax.set_ylim(0.45, 0.66)
ax.set_xlabel("Layer")
ax.set_ylabel("AUROC")
ax.set_yticks([0.45, 0.50, 0.55, 0.60, 0.65])
ax.legend(loc="lower right", frameon=False, handlelength=1.6,
          labelspacing=0.25, borderpad=0.2)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)

fig.tight_layout(pad=0.3)
out = os.path.join(ROOT, "latex/figures/per_layer_mech.pdf")
fig.savefig(out, bbox_inches="tight")
print("wrote", out, "figsize", fig.get_size_inches())
