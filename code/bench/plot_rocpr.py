import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open("results/ablations/roc_pr_curves.json"))
styles = {
    "probe_14b":            dict(label="Hidden-state probe (14B)", color="#b5202b", lw=2.3, ls="-",  z=6),
    "probe_7b":             dict(label="Hidden-state probe (7B)",  color="#2f5e96", lw=1.9, ls="-",  z=5),
    "claude_opus_zeroshot": dict(label="Claude-Opus judge",        color="#e08a2b", lw=1.6, ls="--", z=4),
    "gemma3_zeroshot":      dict(label="Gemma-3-12B judge",        color="#2f9e57", lw=1.6, ls="--", z=3),
    "rounding":             dict(label="Rounding forensic",        color="#8a8f98", lw=1.5, ls=":",  z=2),
}
order = ["probe_14b", "probe_7b", "claude_opus_zeroshot", "gemma3_zeroshot", "rounding"]

plt.rcParams.update({"font.size": 9, "font.family": "serif",
                     "axes.linewidth": 0.8, "pdf.fonttype": 42})
fig, (axr, axp) = plt.subplots(1, 2, figsize=(7.0, 2.9))

# ---- ROC (full 0-1, needs the diagonal) ----
for k in order:
    s = styles[k]; r = d[k]["roc"]
    axr.plot(r["fpr"], r["tpr"], color=s["color"], lw=s["lw"], ls=s["ls"], zorder=s["z"],
             label=f'{s["label"]} ({d[k]["AUROC"]:.2f})')
axr.plot([0, 1], [0, 1], color="#b0b6bd", lw=1.0, ls=(0, (4, 4)), zorder=0)
axr.set_xlim(0, 1); axr.set_ylim(0, 1)
axr.set_xlabel("False positive rate"); axr.set_ylabel("True positive rate")
axr.set_title("ROC", fontsize=10)
axr.grid(True, lw=0.4, alpha=0.35)
axr.legend(fontsize=6.6, loc="lower right", frameon=False, handlelength=1.8)

# ---- Precision-Recall (precision axis starts mid-range, not 0) ----
for k in order:
    s = styles[k]; pr = d[k]["pr"]
    axp.plot(pr["recall"], pr["precision"], color=s["color"], lw=s["lw"], ls=s["ls"], zorder=s["z"],
             label=f'{s["label"]} ({d[k]["AUPRC"]:.2f})')
axp.axhline(0.5, color="#b0b6bd", lw=1.0, ls=(0, (4, 4)), zorder=0)  # balanced-class chance
axp.set_xlim(0, 1); axp.set_ylim(0.45, 1.005)   # start from the middle
axp.set_xlabel("Recall"); axp.set_ylabel("Precision")
axp.set_title("Precision–Recall", fontsize=10)
axp.grid(True, lw=0.4, alpha=0.35)
axp.legend(fontsize=6.6, loc="upper right", frameon=False, handlelength=1.8)

fig.tight_layout(pad=0.6)
fig.savefig("latex/figures/roc_pr.pdf", bbox_inches="tight")
print("saved latex/figures/roc_pr.pdf  | PR precision ylim = 0.45..1.005 (mid-start)")
