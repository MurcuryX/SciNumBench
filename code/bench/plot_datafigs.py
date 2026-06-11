import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 10, "font.family": "serif", "axes.linewidth": 0.8, "pdf.fonttype": 42})
BLUE="#2f5e96"; RED="#b5202b"; GREEN="#2f9e57"; GRAY="#9aa7b4"; ORANGE="#e08a2b"

# ---------- 1. detectability vs fabrication magnitude (stealth curve) ----------
def fig_magnitude(ax):
    bins=["low\n(1-2 cells)","med\n(3 cells)","high\n(4-7 cells)"]
    auroc=[0.6459,0.7003,0.7076]; n=[578,208,214]
    x=range(3)
    ax.bar(x,auroc,width=0.6,color=GREEN,edgecolor="#1f7d43",linewidth=0.8,zorder=3)
    ax.axhline(0.5,color=GRAY,ls=(0,(4,4)),lw=1,zorder=1)
    ax.text(2.45,0.505,"chance",fontsize=7.5,color=GRAY,va="bottom",ha="right")
    for i,(a,c) in enumerate(zip(auroc,n)):
        ax.text(i,a+0.004,f"{a:.3f}",ha="center",va="bottom",fontsize=9,fontweight="bold")
        ax.text(i,0.515,f"n={c}",ha="center",va="bottom",fontsize=7.5,color="#3a5a45")
    ax.set_xticks(list(x)); ax.set_xticklabels(bins,fontsize=8.5)
    ax.set_ylim(0.48,0.74); ax.set_ylabel("AUROC (probe vs honest)")
    ax.set_title("(a) Detectability vs fabrication magnitude",fontsize=10)
    ax.set_xlabel("edited cells per table")
    ax.grid(axis="y",lw=0.4,alpha=0.35)

# ---------- 2. reference-oracle headroom ----------
def fig_oracle(ax):
    labels=["chance","probe\n(reference-free)","oracle\n(+ honest twin)"]
    vals=[0.50,0.6704,0.9727]; cols=[GRAY,RED,GREEN]
    x=range(3)
    ax.bar(x,vals,width=0.6,color=cols,edgecolor="#444",linewidth=0.6,zorder=3)
    for i,v in enumerate(vals):
        ax.text(i,v+0.008,f"{v:.2f}",ha="center",va="bottom",fontsize=9.5,fontweight="bold")
    ax.annotate("",xy=(1,0.6704),xytext=(2,0.9727),
                arrowprops=dict(arrowstyle="<->",color="#555",lw=1.1))
    ax.text(1.5,0.83,"gap 0.30",fontsize=8.5,ha="center",color="#555",style="italic")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels,fontsize=8.5)
    ax.set_ylim(0.45,1.02); ax.set_ylabel("AUROC")
    ax.set_title("(b) The signal is there; reference-free is the hard part",fontsize=9.5)
    ax.grid(axis="y",lw=0.4,alpha=0.35)

# ---------- 3. inference cost ----------
def fig_efficiency(ax):
    labels=["probe\n(prefill only)","zero-shot\njudge","CoT\njudge"]
    ms=[78.26,534.1,8592.7]; cols=[GREEN,ORANGE,RED]
    x=range(3)
    ax.bar(x,ms,width=0.6,color=cols,edgecolor="#444",linewidth=0.6,zorder=3)
    ax.set_yscale("log")
    for i,v in enumerate(ms):
        ax.text(i,v*1.15,f"{v:.0f} ms" if v<1000 else f"{v/1000:.1f} s",ha="center",va="bottom",fontsize=9,fontweight="bold")
    ax.text(1,300,"6.8x",ha="center",fontsize=9,color="#7a5a1a",fontweight="bold")
    ax.text(2,4000,"109x",ha="center",fontsize=9,color=RED,fontweight="bold")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels,fontsize=8.5)
    ax.set_ylim(40,20000); ax.set_ylabel("ms / table  (log scale)")
    ax.set_title("(c) Inference cost (same Qwen2.5-14B, A100)",fontsize=10)
    ax.grid(axis="y",lw=0.4,alpha=0.35,which="both")

# combined preview
fig,axes=plt.subplots(1,3,figsize=(13,3.4))
fig_magnitude(axes[0]); fig_oracle(axes[1]); fig_efficiency(axes[2])
fig.tight_layout(pad=1.0)
fig.savefig("figures/preview_datafigs.png",dpi=140,bbox_inches="tight")
print("saved figures/preview_datafigs.png")

# individual PDFs for the paper
for name,fn in [("magnitude",fig_magnitude),("oracle",fig_oracle),("efficiency",fig_efficiency)]:
    f,a=plt.subplots(figsize=(3.4,2.7)); fn(a); f.tight_layout(pad=0.4)
    f.savefig(f"latex/figures/{name}.pdf",bbox_inches="tight"); plt.close(f)
    print("saved latex/figures/%s.pdf"%name)
