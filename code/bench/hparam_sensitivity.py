"""
hparam_sensitivity.py — Hyperparameter sensitivity / robustness ablation
for the 14B hidden-state probe (reviewer-requested).

Reuses the CANONICAL protocol from ablations_planb.train_eval, refactored so
hyperparameters (lr, weight_decay, dropout, hidden dims, batch size) can be
varied. Vary ONE at a time around the anchor default; 3 seeds {0,1,2}.

Protocol (unchanged from headline):
- Input phi: concat of the 4 tapped layers of qwen14b feats -> R^20480.
- Labels / src_table_id from data/splits/mapping.jsonl.
- StandardScaler fit on train.
- 15% GROUP-DISJOINT (by src_table_id) val carve-out for early-stop / model select.
- BCEWithLogits pos_weight, Adam, early stop on val AUROC.
- TEST (2000 rows) is NEVER used for selection.

CPU only, threads capped at 8.
"""
import os
os.environ.setdefault("TMPDIR", "/data/users/pengminjie/Backup/tmp")
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"
import json
import numpy as np
import torch
import torch.nn as nn
torch.set_num_threads(8)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

ROOT = "/home/pengminjie/Backup/paper/ICDE26"
FEATDIR = ROOT + "/data/features"
MAPPING = ROOT + "/data/splits/mapping.jsonl"
RESULTS = ROOT + "/results"
FIGDIR = ROOT + "/latex/figures"
VAL_FRAC = 0.15
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGDIR, exist_ok=True)

# ---- anchor (default) config ----
DEF_LR = 1e-3
DEF_WD = 1e-3
DEF_DROP = 0.3
DEF_HIDDEN = (256, 64)
DEF_BATCH = 128
SEEDS = [0, 1, 2]


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.use_deterministic_algorithms(True, warn_only=True)


def make_net(d, hidden=DEF_HIDDEN, dropout=DEF_DROP):
    """MLP: Linear(d,h1)-ReLU-Drop - [Linear(h1,h2)-ReLU-Drop] - Linear(last,1).
    Supports 1 or 2 hidden layers."""
    layers = []
    prev = d
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
        prev = h
    layers += [nn.Linear(prev, 1)]
    return nn.Sequential(*layers)


def load_labels():
    lab = {}
    with open(MAPPING) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            lab[r["example_id"]] = {"label": int(r["label"]),
                                    "src": int(r["src_table_id"])}
    return lab


def load_raw(tag, split, labmap):
    d = np.load(f"{FEATDIR}/{tag}_{split}.npz", allow_pickle=True)
    feats = d["features"].astype(np.float32)  # (N, L, H)
    eids = d["example_ids"].astype(str)
    y = np.array([labmap[e]["label"] for e in eids], dtype=np.int64)
    src = np.array([labmap[e]["src"] for e in eids], dtype=np.int64)
    return feats, y, eids, src


def train_eval(Xtr_all, ytr_all, src_tr, Xte, yte, seed=0,
               epochs=200, lr=DEF_LR, wd=DEF_WD, patience=25,
               dropout=DEF_DROP, hidden=DEF_HIDDEN, batch=DEF_BATCH):
    """Canonical training loop (replica of ablations_planb.train_eval) with
    dropout/hidden/batch exposed. Returns test AUROC (float)."""
    set_seed(seed)
    uniq_src = np.array(sorted(set(src_tr.tolist())))
    rng = np.random.RandomState(seed)
    perm_src = uniq_src.copy()
    rng.shuffle(perm_src)
    n_val_src = int(round(len(perm_src) * VAL_FRAC))
    val_src = set(perm_src[:n_val_src].tolist())
    val_mask = np.array([s in val_src for s in src_tr], bool)
    if val_mask.sum() == 0 or (~val_mask).sum() == 0:
        val_mask = np.zeros(len(src_tr), bool)
        val_mask[: max(1, int(round(len(src_tr) * VAL_FRAC)))] = True
    Xtr, ytr = Xtr_all[~val_mask], ytr_all[~val_mask]
    Xva, yva = Xtr_all[val_mask], ytr_all[val_mask]

    scaler = StandardScaler().fit(Xtr)
    Ttr = torch.tensor(scaler.transform(Xtr), dtype=torch.float32)
    Tva = torch.tensor(scaler.transform(Xva), dtype=torch.float32)
    Tte = torch.tensor(scaler.transform(Xte), dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)

    net = make_net(Xtr.shape[1], hidden=hidden, dropout=dropout)
    n_pos, n_neg = float(ytr.sum()), float((1 - ytr).sum())
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n_neg / max(1.0, n_pos)]))
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)

    g = torch.Generator()
    g.manual_seed(seed)
    ntr = Ttr.shape[0]
    best_auc, best_state, bad = -1.0, None, 0
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(ntr, generator=g)
        for i in range(0, ntr, batch):
            bidx = perm[i:i + batch]
            opt.zero_grad()
            loss = crit(net(Ttr[bidx]).squeeze(-1), ytr_t[bidx])
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            sva = torch.sigmoid(net(Tva).squeeze(-1)).numpy()
        try:
            auc = roc_auc_score(yva, sva)
        except ValueError:
            auc = 0.5
        if auc > best_auc + 1e-4:
            best_auc = auc
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        ste = torch.sigmoid(net(Tte).squeeze(-1)).numpy()
    test_auroc = float(roc_auc_score(yte, ste))
    return test_auroc


def run_seeds(Xtr, ytr, src_tr, Xte, yte, **kw):
    aucs = [train_eval(Xtr, ytr, src_tr, Xte, yte, seed=s, **kw) for s in SEEDS]
    return aucs


def main():
    labmap = load_labels()
    Xtr3, ytr, _, src_tr = load_raw("qwen14b", "train", labmap)
    Xte3, yte, _, src_te = load_raw("qwen14b", "test", labmap)
    Ntr, L, H = Xtr3.shape
    Xtr_all = Xtr3.reshape(Ntr, L * H)
    Xte_all = Xte3.reshape(Xte3.shape[0], L * H)
    assert not (set(src_tr.tolist()) & set(src_te.tolist())), "train/test src LEAK"
    print(f"[setup] train {Xtr_all.shape} test {Xte_all.shape} dim={L*H}")

    # ---- anchor (default) over seeds ----
    print("[anchor] default config over seeds", SEEDS, "...")
    anchor_aucs = run_seeds(Xtr_all, ytr, src_tr, Xte_all, yte)
    anchor_mean = float(np.mean(anchor_aucs))
    anchor_std = float(np.std(anchor_aucs))
    print(f"[anchor] AUROC per-seed={[round(a,4) for a in anchor_aucs]} "
          f"mean={anchor_mean:.4f} std={anchor_std:.4f}")

    out = {
        "protocol": "14B 4-layer concat (R^20480); StandardScaler; 15% group-disjoint "
                    "val carve-out by src_table_id for early-stop/model-select; test 2000 rows "
                    "NEVER used for selection. Replica of ablations_planb.train_eval.",
        "default_config": {"lr": DEF_LR, "weight_decay": DEF_WD, "dropout": DEF_DROP,
                           "hidden": list(DEF_HIDDEN), "batch": DEF_BATCH},
        "seeds": SEEDS,
        "anchor": {"AUROC_per_seed": [round(a, 4) for a in anchor_aucs],
                   "test_AUROC_mean": round(anchor_mean, 4),
                   "test_AUROC_std": round(anchor_std, 4)},
        "sweeps": {},
    }

    all_means = [anchor_mean]

    def sweep(name, key, values, kwfn):
        print(f"\n[sweep] {name}: {values}")
        rows = []
        for v in values:
            kw = kwfn(v)
            aucs = run_seeds(Xtr_all, ytr, src_tr, Xte_all, yte, **kw)
            m, s = float(np.mean(aucs)), float(np.std(aucs))
            rows.append({"value": v if not isinstance(v, tuple) else list(v),
                         "AUROC_per_seed": [round(a, 4) for a in aucs],
                         "test_AUROC_mean": round(m, 4),
                         "test_AUROC_std": round(s, 4),
                         "is_anchor": kw == {} or all(
                             kw.get(k, dflt) == dflt for k, dflt in
                             [("lr", DEF_LR), ("wd", DEF_WD), ("dropout", DEF_DROP),
                              ("hidden", DEF_HIDDEN), ("batch", DEF_BATCH)])})
            all_means.append(m)
            print(f"  {key}={v}: {[round(a,4) for a in aucs]} -> {m:.4f}+-{s:.4f}")
        out["sweeps"][name] = rows

    sweep("learning_rate", "lr", [3e-4, 1e-3, 3e-3, 1e-2], lambda v: {"lr": v})
    sweep("weight_decay", "wd", [0.0, 1e-4, 1e-3, 1e-2], lambda v: {"wd": v})
    sweep("dropout", "dropout", [0.0, 0.1, 0.3, 0.5], lambda v: {"dropout": v})
    sweep("hidden_dims", "hidden", [(128, 32), (256, 64), (512, 128), (256,)],
          lambda v: {"hidden": v})
    sweep("batch_size", "batch", [64, 128, 256], lambda v: {"batch": v})

    out["overall"] = {
        "min_AUROC_mean": round(float(np.min(all_means)), 4),
        "max_AUROC_mean": round(float(np.max(all_means)), 4),
        "range": round(float(np.max(all_means) - np.min(all_means)), 4),
        "seed_noise_std_at_anchor": round(anchor_std, 4),
    }
    print(f"\n[overall] min={out['overall']['min_AUROC_mean']} "
          f"max={out['overall']['max_AUROC_mean']} "
          f"range={out['overall']['range']} anchor_std={anchor_std:.4f}")

    json.dump(out, open(f"{RESULTS}/hparam_sensitivity.json", "w"), indent=2)
    print(f"[write] {RESULTS}/hparam_sensitivity.json")

    make_figure(out, anchor_mean, anchor_std)


def make_figure(out, anchor_mean, anchor_std):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
    })

    panels = [
        ("learning_rate", "learning rate", True),
        ("weight_decay", "weight decay", "wd"),
        ("dropout", "dropout", False),
        ("hidden_dims", "hidden dims", "cat"),
        ("batch_size", "batch size", False),
    ]
    # 2x3 grid; last cell holds legend/text
    fig, axes = plt.subplots(2, 3, figsize=(3.3, 2.6), sharey=True)
    axes = axes.ravel()
    ylo, yhi = 0.55, 0.70
    band_lo, band_hi = anchor_mean - anchor_std, anchor_mean + anchor_std

    for ax, (name, xlabel, mode) in zip(axes, panels):
        rows = out["sweeps"][name]
        means = [r["test_AUROC_mean"] for r in rows]
        stds = [r["test_AUROC_std"] for r in rows]
        if mode == "cat":
            xs = list(range(len(rows)))
            labels = ["x".join(str(int(z)) for z in r["value"]) if isinstance(r["value"], list)
                      else str(r["value"]) for r in rows]
            ax.set_xticks(xs)
            ax.set_xticklabels(labels, rotation=40, ha="right")
        elif mode is True:  # log scale (lr)
            xs = [r["value"] for r in rows]
            ax.set_xscale("log")
        elif mode == "wd":
            # wd has a 0 -> plot on index axis with labels (log can't show 0)
            xs = list(range(len(rows)))
            labels = [("0" if r["value"] == 0 else f"{r['value']:g}") for r in rows]
            ax.set_xticks(xs)
            ax.set_xticklabels(labels, rotation=40, ha="right")
        else:
            xs = [r["value"] for r in rows]

        ax.axhspan(band_lo, band_hi, color="0.80", alpha=0.6, lw=0, zorder=0)
        ax.axhline(anchor_mean, ls="--", color="C3", lw=0.8, zorder=1)
        ax.errorbar(xs, means, yerr=stds, fmt="o-", ms=2.5, lw=0.9,
                    capsize=1.5, color="C0", zorder=2)
        ax.set_ylim(ylo, yhi)
        ax.set_xlabel(xlabel)
        ax.tick_params(length=2, pad=1.5)
        ax.grid(True, axis="y", lw=0.3, alpha=0.4)

    axes[0].set_ylabel("test AUROC")
    axes[3].set_ylabel("test AUROC")

    # 6th cell: legend / annotation
    lax = axes[5]
    lax.axis("off")
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color="C0", marker="o", ms=3, lw=0.9, label="mean$\\pm$std (3 seeds)"),
        Line2D([0], [0], color="C3", ls="--", lw=0.8, label=f"anchor {anchor_mean:.3f}"),
        Patch(facecolor="0.80", alpha=0.6, label=f"seed noise $\\pm${anchor_std:.3f}"),
    ]
    lax.legend(handles=handles, loc="center", frameon=False, fontsize=6.5,
               handlelength=1.6, borderaxespad=0)

    fig.tight_layout(pad=0.3, w_pad=0.4, h_pad=0.6)
    outp = f"{FIGDIR}/hparam_sensitivity.pdf"
    fig.savefig(outp, bbox_inches="tight")
    print(f"[write] {outp}")


if __name__ == "__main__":
    main()
