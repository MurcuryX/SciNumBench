"""
pseudo_reference.py — EXPLORATORY: pseudo-reference contrastive detector.

Goal: approximate the honest reference (which gives oracle AUROC 0.973) by
RETRIEVING similar honest TRAIN tables, then test whether contrast features
[phi || (phi - Rbar) || scalar distances] beat the reference-free probe (~0.66).

Reuses train_probe_planb conventions:
  phi = concat 4 tapped layers (20480-d)
  StandardScaler on train, group-disjoint 15% val by src_table_id (seed 42),
  MLP Linear(d,256)-ReLU-Drop0.3-Linear(256,64)-ReLU-Drop0.3-Linear(64,1),
  Adam lr=1e-3 wd=1e-3 batch=128, BCEWithLogits pos_weight, early-stop val AUROC.

LEAKAGE SAFEGUARDS:
  * Retrieval pool = TRAIN split HONEST tables only (label==0, split==train).
  * Exclude any neighbor sharing the query's src_table_id (so a table never sees
    its own honest twin = no oracle leak; for train queries this excludes self).
  * Mahalanobis stats (mean/cov) fit on the HONEST-TRAIN pool only.
  * Val carve-out is group-disjoint by src (same as headline). Test untouched.

CPU-only (cached features). Threads capped to 8.
"""
import os, json, pickle, time, sys
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ.setdefault("TMPDIR", "/data/users/pengminjie/Backup/tmp")
import numpy as np
import torch
torch.set_num_threads(8)
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = "/home/pengminjie/Backup/paper/ICDE26"
FEATDIR = ROOT + "/data/features"
MAPPING = ROOT + "/data/splits/mapping.jsonl"
RESULTS = ROOT + "/results"
SEED = 42
VAL_FRAC = 0.15
KS = [5, 10, 20]
SEEDS = [42, 43, 44]  # for mean+/-std on the best variant


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s)
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_labels():
    lab = {}
    with open(MAPPING) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            lab[r["example_id"]] = {"label": int(r["label"]),
                                    "src": int(r["src_table_id"]),
                                    "split": r["split"]}
    return lab


def load_concat(tag, split, labmap):
    d = np.load(f"{FEATDIR}/{tag}_{split}.npz", allow_pickle=True)
    feats = d["features"].astype(np.float32)
    N, L, H = feats.shape
    X = feats.reshape(N, L * H)
    eids = d["example_ids"].astype(str)
    y = np.array([labmap[e]["label"] for e in eids], dtype=np.int64)
    src = np.array([labmap[e]["src"] for e in eids], dtype=np.int64)
    return X, y, eids, src, list(map(int, d["tapped_layer_indices"]))


# ---------------------------------------------------------------------------
def make_net(d):
    return nn.Sequential(
        nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(64, 1),
    )


def group_val_mask(src_tr, seed):
    uniq_src = np.array(sorted(set(src_tr.tolist())))
    rng = np.random.RandomState(seed)
    perm_src = uniq_src.copy(); rng.shuffle(perm_src)
    n_val_src = int(round(len(perm_src) * VAL_FRAC))
    val_src = set(perm_src[:n_val_src].tolist())
    return np.array([s in val_src for s in src_tr], bool), val_src


def train_eval(Xtr_all, ytr_all, src_tr, Xte, yte, src_te, seed,
               epochs=200, patience=25):
    set_seed(seed)
    val_mask, val_src = group_val_mask(src_tr, seed)
    Xtr, ytr = Xtr_all[~val_mask], ytr_all[~val_mask]
    Xva, yva = Xtr_all[val_mask], ytr_all[val_mask]
    assert not (set(src_tr[~val_mask].tolist()) & set(src_te.tolist())), "train/test LEAK"
    assert not (val_src & set(src_tr[~val_mask].tolist())), "val/train LEAK"

    scaler = StandardScaler().fit(Xtr)
    Ttr = torch.tensor(scaler.transform(Xtr), dtype=torch.float32)
    Tva = torch.tensor(scaler.transform(Xva), dtype=torch.float32)
    Tte = torch.tensor(scaler.transform(Xte), dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)

    net = make_net(Xtr.shape[1])
    n_pos, n_neg = float(ytr.sum()), float((1 - ytr).sum())
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n_neg / max(1.0, n_pos)]))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-3)
    g = torch.Generator(); g.manual_seed(seed)
    batch = 128; ntr = Ttr.shape[0]
    best_auc, best_state, best_ep, bad = -1.0, None, -1, 0
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(ntr, generator=g)
        for i in range(0, ntr, batch):
            bidx = perm[i:i + batch]
            opt.zero_grad()
            loss = crit(net(Ttr[bidx]).squeeze(-1), ytr_t[bidx])
            loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            sva = torch.sigmoid(net(Tva).squeeze(-1)).numpy()
        auc = roc_auc_score(yva, sva)
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
    test_auroc = float(roc_auc_score(yte, ste))
    test_auprc = float(average_precision_score(yte, ste))
    return {"val_AUROC": float(best_auc), "test_AUROC": test_auroc,
            "test_AUPRC": test_auprc, "best_ep": best_ep}


def main():
    t0 = time.time()
    labmap = load_labels()
    tag = "qwen14b"
    Xtr_all, ytr_all, eids_tr, src_tr, taps = load_concat(tag, "train", labmap)
    Xte, yte, eids_te, src_te, _ = load_concat(tag, "test", labmap)
    print(f"[load] train {Xtr_all.shape} test {Xte.shape} taps={taps} "
          f"({time.time()-t0:.1f}s)", flush=True)

    # ---- honest-train retrieval pool ----
    pool_mask = (ytr_all == 0)
    Xpool = Xtr_all[pool_mask]
    src_pool = src_tr[pool_mask]
    print(f"[pool] honest-train pool {Xpool.shape}, n_src={len(set(src_pool.tolist()))}", flush=True)
    assert len(set(src_pool.tolist())) == Xpool.shape[0], "expect 1 honest table per src"

    # Mahalanobis stats on honest pool. D=20480 too big for full cov; use
    # diagonal-shrinkage: Sigma = diag(var)+eps. inv_chol = diag(1/sqrt(var+eps)).
    pool_mean = Xpool.mean(axis=0)
    pool_var = Xpool.var(axis=0) + 1e-3
    inv_sd = (1.0 / np.sqrt(pool_var)).astype(np.float32)  # vector form, avoid DxD matrix

    # patch build_pseudo_ref maha to use vector inv_sd via a light wrapper
    def maha_vec(x):
        z = (x - pool_mean) * inv_sd
        return float(np.sqrt((z * z).sum()))

    # ---- build pseudo-reference for train & test queries ----
    def build(Xq, src_q, name):
        Nq, D = Xq.shape
        qn = Xq / (np.linalg.norm(Xq, axis=1, keepdims=True) + 1e-8)
        pn = Xpool / (np.linalg.norm(Xpool, axis=1, keepdims=True) + 1e-8)
        Kmax = max(KS)
        Rbar = {K: np.zeros((Nq, D), np.float32) for K in KS}
        d1_cos = np.zeros(Nq, np.float32)
        meanK_cos = {K: np.zeros(Nq, np.float32) for K in KS}
        typicality = np.zeros(Nq, np.float32)
        maha = np.zeros(Nq, np.float32)
        l2_d1 = np.zeros(Nq, np.float32)
        B = 256
        for i0 in range(0, Nq, B):
            i1 = min(i0 + B, Nq)
            sims = qn[i0:i1] @ pn.T
            for j in range(i1 - i0):
                qi = i0 + j
                mask = src_pool != src_q[qi]
                s = sims[j].copy(); s[~mask] = -2.0
                typicality[qi] = -float(s[mask].mean())
                order = np.argsort(-s)
                topKmax = order[:Kmax]
                d1_cos[qi] = 1.0 - float(s[topKmax[0]])
                for K in KS:
                    idxK = topKmax[:K]
                    Rbar[K][qi] = Xpool[idxK].mean(axis=0)
                    meanK_cos[K][qi] = 1.0 - float(s[idxK].mean())
                l2 = np.linalg.norm(Xq[qi] - Xpool[topKmax], axis=1)
                l2_d1[qi] = float(l2.min())
                maha[qi] = maha_vec(Xq[qi])
            if i0 % 1024 == 0:
                print(f"  [{name}] {i1}/{Nq} ({time.time()-t0:.1f}s)", flush=True)
        return {"Rbar": Rbar, "d1_cos": d1_cos, "meanK_cos": meanK_cos,
                "typicality": typicality, "maha": maha, "l2_d1": l2_d1}

    Btr = build(Xtr_all, src_tr, "train")
    Bte = build(Xte, src_te, "test")
    print(f"[retrieval done] {time.time()-t0:.1f}s", flush=True)

    def scal_block(Bd, K):
        # [d1_cos, meanK_cos, typicality, maha, l2_d1]  -> (N,5)
        return np.stack([Bd["d1_cos"], Bd["meanK_cos"][K], Bd["typicality"],
                         Bd["maha"], Bd["l2_d1"]], axis=1).astype(np.float32)

    oracle = json.load(open(f"{RESULTS}/ablations/reference_upperbound.json"))

    results = {
        "method": "pseudo-reference contrastive detector (retrieval of honest-train tables)",
        "feature": "phi=concat 4 tapped layers (20480-d)",
        "retrieval": "cosine NN over HONEST-TRAIN pool, same-src EXCLUDED",
        "Ks_tried": KS, "protocol": "train_probe_planb (StandardScaler, group-disjoint 15% val by src, MLP, seed42)",
        "scalar_feats": ["d1_cos", "meanK_cos", "typicality(neg mean cos to pool)",
                         "maha(diag-shrink Mahalanobis to honest pool)", "l2_d1"],
        "leakage_safeguards": {
            "retrieval_pool": "train-split honest tables only (2518)",
            "same_src_excluded": True,
            "maha_stats_fit_on": "honest-train pool only",
            "val": "group-disjoint by src_table_id, test untouched",
        },
        "baseline_headline_AUROC": 0.6704,
        "baseline_headline_10seed": "0.659 +/- 0.007",
        "oracle_reference_AUROC": oracle["oracle_test_AUROC"],
        "oracle_reference_AUPRC": oracle["oracle_test_AUPRC"],
        "variants": {},
    }

    # ---------- (a) baseline: phi only (single seed 42, sanity) ----------
    print("=== (a) baseline phi-only ===", flush=True)
    rb = train_eval(Xtr_all, ytr_all, src_tr, Xte, yte, src_te, SEED)
    results["variants"]["a_baseline_phi"] = {"K": None, **rb}
    print(rb, flush=True)

    # ---------- (b) retrieval-contrast: [phi || Delta || scal] per K ----------
    # ---------- (c) contrast-only: [Delta || scal] per K ----------
    # ---------- (d) distances-only: [scal] per K ----------
    for variant, builder in [
        ("b_retrieval_contrast", lambda K, Xq, Bd: np.concatenate(
            [Xq, Xq - Bd["Rbar"][K], scal_block(Bd, K)], axis=1)),
        ("c_contrast_only", lambda K, Xq, Bd: np.concatenate(
            [Xq - Bd["Rbar"][K], scal_block(Bd, K)], axis=1)),
        ("d_distances_only", lambda K, Xq, Bd: scal_block(Bd, K)),
    ]:
        print(f"=== {variant} ===", flush=True)
        perK = {}
        for K in KS:
            Xtr_v = builder(K, Xtr_all, Btr)
            Xte_v = builder(K, Xte, Bte)
            r = train_eval(Xtr_v, ytr_all, src_tr, Xte_v, yte, src_te, SEED)
            r["in_dim"] = int(Xtr_v.shape[1])
            perK[K] = r
            print(f"  K={K} val={r['val_AUROC']:.4f} test={r['test_AUROC']:.4f}", flush=True)
        bestK = max(perK, key=lambda k: perK[k]["val_AUROC"])  # pick K on VAL
        results["variants"][variant] = {
            "per_K": perK, "best_K_by_val": bestK,
            "test_AUROC": perK[bestK]["test_AUROC"],
            "test_AUPRC": perK[bestK]["test_AUPRC"],
            "val_AUROC": perK[bestK]["val_AUROC"],
        }

    # ---------- multi-seed on best variant (b) at its best K + baseline ----------
    bestK_b = results["variants"]["b_retrieval_contrast"]["best_K_by_val"]
    print(f"=== multi-seed: baseline & b@K={bestK_b} over {SEEDS} ===", flush=True)
    b_auroc, a_auroc = [], []
    Xtr_b = np.concatenate([Xtr_all, Xtr_all - Btr["Rbar"][bestK_b], scal_block(Btr, bestK_b)], axis=1)
    Xte_b = np.concatenate([Xte, Xte - Bte["Rbar"][bestK_b], scal_block(Bte, bestK_b)], axis=1)
    for s in SEEDS:
        ra = train_eval(Xtr_all, ytr_all, src_tr, Xte, yte, src_te, s)
        rbm = train_eval(Xtr_b, ytr_all, src_tr, Xte_b, yte, src_te, s)
        a_auroc.append(ra["test_AUROC"]); b_auroc.append(rbm["test_AUROC"])
        print(f"  seed={s} a={ra['test_AUROC']:.4f} b={rbm['test_AUROC']:.4f}", flush=True)
    results["multiseed"] = {
        "seeds": SEEDS,
        "a_baseline_phi": {"AUROCs": a_auroc, "mean": float(np.mean(a_auroc)), "std": float(np.std(a_auroc))},
        f"b_retrieval_contrast_K{bestK_b}": {"AUROCs": b_auroc, "mean": float(np.mean(b_auroc)), "std": float(np.std(b_auroc))},
        "delta_b_minus_a_mean": float(np.mean(b_auroc) - np.mean(a_auroc)),
    }

    # ---------- verdict ----------
    a_mean = float(np.mean(a_auroc)); a_std = float(np.std(a_auroc))
    b_mean = float(np.mean(b_auroc))
    delta = b_mean - a_mean
    headline = 0.659; headline_std = 0.007
    helps = delta > 2 * max(a_std, headline_std)  # beyond ~2x seed noise
    results["verdict"] = {
        "best_variant": "b_retrieval_contrast",
        "best_K": bestK_b,
        "b_mean_AUROC": b_mean,
        "a_mean_AUROC": a_mean,
        "delta_mean": delta,
        "seed_noise_std_a": a_std,
        "headline_10seed": f"{headline}+/-{headline_std}",
        "retrieval_helps_beyond_seed_noise": bool(helps),
        "note": "delta must exceed ~2x seed std to count as a real gain.",
    }
    results["runtime_sec"] = round(time.time() - t0, 1)
    with open(f"{RESULTS}/pseudo_reference.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print("[SAVE] " + RESULTS + "/pseudo_reference.json", flush=True)
    print(json.dumps(results["verdict"], indent=2), flush=True)


if __name__ == "__main__":
    main()
