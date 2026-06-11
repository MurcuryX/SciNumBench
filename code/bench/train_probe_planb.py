"""
train_probe_planb.py — Step 3 (Plan B): train + eval hidden-state MLP probe.

Reuses the MLP architecture/training conventions from
code/bench/method_hidden_probe_enhanced.py:mlp_variant
  net = [Linear(d,256), ReLU, Dropout(0.3), Linear(256,64), ReLU, Dropout(0.3), Linear(64,1)]
  BCEWithLogitsLoss(pos_weight=n_neg/n_pos), Adam(lr=1e-3, wd=1e-4)
  StandardScaler fit on TRAIN only, early-stop on val AUROC (patience=20, max 200 epochs).

Adapted to the new Step-2 feature format:
  data/features/qwen{7b,14b}_{train,test}.npz  key 'features' (N, L, H) float16,
    'example_ids', 'tapped_layer_indices'.
  Feature phi = CONCAT tapped layers -> (N, L*H).
Labels joined from data/splits/mapping.jsonl by example_id.
New splits have only train/test; carve a 15% val slice out of TRAIN
(stratified, seeded) for early stopping / threshold selection. TEST untouched.
"""
import os, json, pickle
os.environ.setdefault("TMPDIR", "/data/users/pengminjie/Backup/tmp")
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = "/home/pengminjie/Backup/paper/ICDE26"
FEATDIR = ROOT + "/data/features"
MAPPING = ROOT + "/data/splits/mapping.jsonl"
MODELDIR = ROOT + "/models"
RESULTS = ROOT + "/results"
SEED = 42
VAL_FRAC = 0.15

os.makedirs(MODELDIR, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)


def set_seed(s=SEED):
    np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_labels():
    """example_id -> {label, src_table_id}."""
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


def load_concat(tag, split, labmap):
    d = np.load(f"{FEATDIR}/{tag}_{split}.npz", allow_pickle=True)
    feats = d["features"].astype(np.float32)        # (N, L, H)
    N, L, H = feats.shape
    X = feats.reshape(N, L * H)                     # concat tapped layers
    eids = d["example_ids"].astype(str)
    y = np.array([labmap[e]["label"] for e in eids], dtype=np.int64)
    src = np.array([labmap[e]["src"] for e in eids], dtype=np.int64)
    return X, y, eids, src, list(map(int, d["tapped_layer_indices"]))


def metrics_from_preds(y, pred):
    y = np.asarray(y); pred = np.asarray(pred)
    TP = int(np.sum((pred == 1) & (y == 1)))
    FP = int(np.sum((pred == 1) & (y == 0)))
    FN = int(np.sum((pred == 0) & (y == 1)))
    TN = int(np.sum((pred == 0) & (y == 0)))
    P = TP / (TP + FP) if TP + FP else 0.0
    R = TP / (TP + FN) if TP + FN else 0.0
    F1 = 2 * P * R / (P + R) if P + R else 0.0
    Acc = (TP + TN) / len(y) if len(y) else 0.0
    return dict(TP=TP, FP=FP, FN=FN, TN=TN,
                precision=round(P, 4), recall=round(R, 4), F1=round(F1, 4),
                accuracy=round(Acc, 4))


def f1_opt_threshold(y, score):
    cands = np.unique(score)
    best_thr, best_f1 = 0.5, -1.0
    for thr in cands:
        pred = (score >= thr).astype(int)
        m = metrics_from_preds(y, pred)
        if m["F1"] > best_f1:
            best_f1 = m["F1"]; best_thr = float(thr)
    return best_thr


def make_net(d):
    return nn.Sequential(
        nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(64, 1),
    )


def train_one(tag, labmap, epochs=200, lr=1e-3, wd=1e-3, patience=25):
    set_seed(SEED)
    Xtr_all, ytr_all, eids_tr, src_tr, taps = load_concat(tag, "train", labmap)
    Xte, yte, eids_te, src_te, _ = load_concat(tag, "test", labmap)

    # GROUP-disjoint 15% val carve-out by src_table_id (seeded). Each src_table
    # contributes a paired pos+neg example; if a table is split across train/val
    # the probe memorizes table identity and val AUROC collapses below 0.5.
    # Holding out whole tables mirrors the train/test protocol (test is already
    # src-disjoint from train).
    uniq_src = np.array(sorted(set(src_tr.tolist())))
    rng = np.random.RandomState(SEED)
    perm_src = uniq_src.copy(); rng.shuffle(perm_src)
    n_val_src = int(round(len(perm_src) * VAL_FRAC))
    val_src = set(perm_src[:n_val_src].tolist())
    val_mask = np.array([s in val_src for s in src_tr], bool)
    Xtr, ytr = Xtr_all[~val_mask], ytr_all[~val_mask]
    Xva, yva = Xtr_all[val_mask], ytr_all[val_mask]

    # leak self-checks: train/test src-disjoint; val src-disjoint from train-core
    assert not (set(src_tr.tolist()) & set(src_te.tolist())), "train/test src LEAK"
    assert not (val_src & set(src_tr[~val_mask].tolist())), "val/train-core src LEAK"

    scaler = StandardScaler().fit(Xtr)
    Ttr = torch.tensor(scaler.transform(Xtr), dtype=torch.float32)
    Tva = torch.tensor(scaler.transform(Xva), dtype=torch.float32)
    Tte = torch.tensor(scaler.transform(Xte), dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)

    net = make_net(Xtr.shape[1])
    n_pos, n_neg = float(ytr.sum()), float((1 - ytr).sum())
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n_neg / max(1.0, n_pos)]))
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)

    # mini-batch SGD: many gradient steps per epoch (full-batch stalls on
    # 10k-20k-dim inputs). Shuffle each epoch with the global RNG (seeded).
    g = torch.Generator(); g.manual_seed(SEED)
    batch = 128
    ntr = Ttr.shape[0]
    best_auc, best_state, best_ep, bad = -1.0, None, -1, 0
    curve = []
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(ntr, generator=g)
        ep_loss = 0.0
        for i in range(0, ntr, batch):
            bidx = perm[i:i + batch]
            opt.zero_grad()
            loss = crit(net(Ttr[bidx]).squeeze(-1), ytr_t[bidx])
            loss.backward(); opt.step()
            ep_loss += float(loss.item()) * len(bidx)
        ep_loss /= ntr
        net.eval()
        with torch.no_grad():
            sva = torch.sigmoid(net(Tva).squeeze(-1)).numpy()
        auc = roc_auc_score(yva, sva)
        curve.append((ep, round(ep_loss, 4), round(float(auc), 4)))
        if auc > best_auc + 1e-4:
            best_auc = auc; best_state = {k: v.clone() for k, v in net.state_dict().items()}
            best_ep = ep; bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    net.load_state_dict(best_state); net.eval()
    with torch.no_grad():
        ste = torch.sigmoid(net(Tte).squeeze(-1)).numpy()
        sva_best = torch.sigmoid(net(Tva).squeeze(-1)).numpy()

    # F1-optimal threshold chosen on VAL (no test peeking)
    thr_f1 = f1_opt_threshold(yva, sva_best)

    test_auroc = round(float(roc_auc_score(yte, ste)), 4)
    test_auprc = round(float(average_precision_score(yte, ste)), 4)
    m05 = metrics_from_preds(yte, (ste >= 0.5).astype(int))
    mf1 = metrics_from_preds(yte, (ste >= thr_f1).astype(int))

    # save model + scaler
    torch.save({"state_dict": net.state_dict(), "in_dim": Xtr.shape[1],
                "arch": "Linear-256-ReLU-Drop0.3-Linear-64-ReLU-Drop0.3-Linear-1",
                "tapped_layers": taps, "seed": SEED,
                "f1_opt_threshold": thr_f1, "best_val_epoch": best_ep},
               f"{MODELDIR}/probe_{tag_short(tag)}.pt")
    with open(f"{MODELDIR}/scaler_{tag_short(tag)}.pkl", "wb") as fh:
        pickle.dump(scaler, fh)

    preds = {eids_te[i]: float(ste[i]) for i in range(len(eids_te))}
    with open(f"{RESULTS}/probe_preds_{tag_short(tag)}.json", "w") as fh:
        json.dump(preds, fh)

    block = {
        "tag": tag, "in_dim": Xtr.shape[1], "tapped_layers": taps,
        "n_train": int(len(ytr)), "n_val": int(len(yva)), "n_test": int(len(yte)),
        "n_val_src_tables": int(n_val_src), "val_split": "group-disjoint by src_table_id",
        "best_val_epoch": best_ep, "best_val_AUROC": round(float(best_auc), 4),
        "epochs_ran": len(curve),
        "test": {
            "AUROC": test_auroc, "AUPRC": test_auprc,
            "at_0.5": m05,
            "f1_opt_threshold": round(thr_f1, 4),
            "at_f1_opt": mf1,
            "confusion_at_0.5": {"TP": m05["TP"], "FP": m05["FP"],
                                 "FN": m05["FN"], "TN": m05["TN"]},
        },
        "val_curve_tail": curve[-5:],
        "val_curve_head": curve[:5],
    }
    print(f"[{tag}] in_dim={Xtr.shape[1]} bestVALep={best_ep} valAUROC={best_auc:.4f} "
          f"testAUROC={test_auroc} testAUPRC={test_auprc} "
          f"F1@0.5={m05['F1']} F1@opt={mf1['F1']}(thr={thr_f1:.3f})")
    return block


def tag_short(tag):
    return "14b" if "14b" in tag else "7b"


def main():
    labmap = load_labels()
    out = {"method": "frozen Qwen hidden-state concat -> StandardScaler -> small MLP probe",
           "seed": SEED, "val_frac": VAL_FRAC,
           "val_protocol": "15% of train src_table_ids held out (group-disjoint)",
           "mlp_arch": "Linear(d,256)-ReLU-Drop0.3-Linear(256,64)-ReLU-Drop0.3-Linear(64,1)",
           "optimizer": "Adam lr=1e-3 wd=1e-3 batch=128 BCEWithLogits pos_weight",
           "early_stop": "val AUROC patience=25 max_epochs=200",
           "reused_from": "code/bench/method_hidden_probe_enhanced.py:mlp_variant"}
    out["qwen14b"] = train_one("qwen14b", labmap)   # HEADLINE
    out["qwen7b"] = train_one("qwen7b", labmap)
    with open(f"{RESULTS}/probe_metrics.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("[SAVE] " + RESULTS + "/probe_metrics.json")


if __name__ == "__main__":
    main()
