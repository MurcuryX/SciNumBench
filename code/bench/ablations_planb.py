"""
ablations_planb.py — Step 5 (Plan B): ablations + final result tables.

Reuses the EXACT MLP arch / training / scaler logic from train_probe_planb.py
(refactored into train_eval()). Headline = 14B (4-layer concat).

Outputs under results/ablations/ and results/final_*.{json,md}.
"""
import os, json, pickle, collections
os.environ.setdefault("TMPDIR", "/data/users/pengminjie/Backup/tmp")
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

ROOT = "/home/pengminjie/Backup/paper/ICDE26"
FEATDIR = ROOT + "/data/features"
MAPPING = ROOT + "/data/splits/mapping.jsonl"
RESULTS = ROOT + "/results"
ABL = RESULTS + "/ablations"
DB = ROOT + "/data/arxiv_data.db"
SEED = 42
VAL_FRAC = 0.15
os.makedirs(ABL, exist_ok=True)


def set_seed(s=SEED):
    np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.use_deterministic_algorithms(True, warn_only=True)


def make_net(d):
    return nn.Sequential(
        nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(64, 1),
    )


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
    """Return per-layer features (N,L,H), y, eids, src, tapped layers."""
    d = np.load(f"{FEATDIR}/{tag}_{split}.npz", allow_pickle=True)
    feats = d["features"].astype(np.float32)  # (N, L, H)
    eids = d["example_ids"].astype(str)
    y = np.array([labmap[e]["label"] for e in eids], dtype=np.int64)
    src = np.array([labmap[e]["src"] for e in eids], dtype=np.int64)
    taps = list(map(int, d["tapped_layer_indices"]))
    return feats, y, eids, src, taps


def train_eval(Xtr_all, ytr_all, src_tr, Xte, yte, seed=SEED,
               epochs=200, lr=1e-3, wd=1e-3, patience=25, return_scores=False):
    """EXACT replica of train_probe_planb.train_one's training loop.
    Group-disjoint val carve-out by src_table_id (seeded by `seed`).
    Returns dict with test AUROC/AUPRC + metrics; optionally test scores."""
    set_seed(seed)
    uniq_src = np.array(sorted(set(src_tr.tolist())))
    rng = np.random.RandomState(seed)
    perm_src = uniq_src.copy(); rng.shuffle(perm_src)
    n_val_src = int(round(len(perm_src) * VAL_FRAC))
    val_src = set(perm_src[:n_val_src].tolist())
    val_mask = np.array([s in val_src for s in src_tr], bool)
    # guard: tiny subsets may yield empty val -> fall back to no early stop
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

    net = make_net(Xtr.shape[1])
    n_pos, n_neg = float(ytr.sum()), float((1 - ytr).sum())
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n_neg / max(1.0, n_pos)]))
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)

    g = torch.Generator(); g.manual_seed(seed)
    batch = 128
    ntr = Ttr.shape[0]
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
        try:
            auc = roc_auc_score(yva, sva)
        except ValueError:
            auc = 0.5
        if auc > best_auc + 1e-4:
            best_auc = auc
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
            best_ep = ep; bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state); net.eval()
    with torch.no_grad():
        ste = torch.sigmoid(net(Tte).squeeze(-1)).numpy()
        sva_best = torch.sigmoid(net(Tva).squeeze(-1)).numpy()
    thr_f1 = f1_opt_threshold(yva, sva_best)
    test_auroc = float(roc_auc_score(yte, ste)) if len(set(yte.tolist())) > 1 else None
    test_auprc = float(average_precision_score(yte, ste)) if len(set(yte.tolist())) > 1 else None
    m05 = metrics_from_preds(yte, (ste >= 0.5).astype(int))
    mf1 = metrics_from_preds(yte, (ste >= thr_f1).astype(int))
    out = dict(AUROC=None if test_auroc is None else round(test_auroc, 4),
               AUPRC=None if test_auprc is None else round(test_auprc, 4),
               best_val_AUROC=round(float(best_auc), 4), best_val_epoch=best_ep,
               n_train=int(len(ytr)), n_val=int(len(yva)), n_test=int(len(yte)),
               f1_opt_threshold=round(float(thr_f1), 4),
               at_0p5=m05, at_f1_opt=mf1)
    if return_scores:
        return out, ste
    return out


# ---------- magnitude proxy from provenance / DB ----------
def load_fraud_meta():
    """src_table_id -> {strategy, n_cells_changed, provenance(list)}."""
    import sqlite3
    con = sqlite3.connect(DB); cur = con.cursor()
    meta = {}
    for sid, strat, ncc, prov in cur.execute(
            "SELECT src_table_id, strategy, n_cells_changed, provenance FROM llm_fraud"):
        try:
            pv = json.loads(prov) if prov else []
        except Exception:
            pv = []
        meta[int(sid)] = {"strategy": strat, "n_cells_changed": ncc, "prov": pv}
    con.close()
    return meta


def rel_delta_from_prov(prov):
    """Magnitude proxy: max relative |new-orig|/|orig| over edited cells (fallback abs)."""
    best = 0.0
    for ch in prov:
        try:
            o = float(str(ch.get("orig")).replace("%", "").replace(",", ""))
            n = float(str(ch.get("new")).replace("%", "").replace(",", ""))
        except Exception:
            continue
        denom = abs(o) if abs(o) > 1e-9 else 1.0
        rd = abs(n - o) / denom
        best = max(best, rd)
    return best


def main():
    labmap = load_labels()
    fraud = load_fraud_meta()
    summary = {"seed": SEED}

    # base 14B raw features
    Xtr3, ytr, eids_tr, src_tr, taps14 = load_raw("qwen14b", "train", labmap)
    Xte3, yte, eids_te, src_te, _ = load_raw("qwen14b", "test", labmap)
    Ntr, L, H = Xtr3.shape
    Xtr_all = Xtr3.reshape(Ntr, L * H)
    Xte_all = Xte3.reshape(Xte3.shape[0], L * H)
    assert not (set(src_tr.tolist()) & set(src_te.tolist())), "train/test src LEAK"

    # ===== SANITY: reproduce headline 14B =====
    print("[SANITY] reproducing 14B headline ...")
    base, base_scores = train_eval(Xtr_all, ytr, src_tr, Xte_all, yte, return_scores=True)
    print("[SANITY] 14B concat AUROC=", base["AUROC"], "AUPRC=", base["AUPRC"],
          "(expect ~0.670)")
    summary["sanity_14b_headline"] = base

    # ============ ABLATION 1: learning curve ============
    print("\n[ABL1] learning curve ...")
    fracs = [0.1, 0.25, 0.5, 0.75, 1.0]
    seeds = [42, 1, 2]
    uniq_src = np.array(sorted(set(src_tr.tolist())))
    lc = {"fractions": fracs, "seeds": seeds, "metric": "test AUROC on full 2000-row test",
          "subsample": "by src_table_id GROUPS (paired pos+neg kept together)",
          "points": []}
    for fr in fracs:
        aucs = []
        n_tab = None
        for sd in seeds:
            if fr >= 0.999:
                sel_src = set(uniq_src.tolist())
            else:
                rng = np.random.RandomState(sd)
                perm = uniq_src.copy(); rng.shuffle(perm)
                k = max(2, int(round(len(perm) * fr)))
                sel_src = set(perm[:k].tolist())
            mask = np.array([s in sel_src for s in src_tr], bool)
            n_tab = len(sel_src)
            r = train_eval(Xtr_all[mask], ytr[mask], src_tr[mask], Xte_all, yte, seed=sd)
            aucs.append(r["AUROC"])
            if fr >= 0.999:
                break  # full set identical across seeds for selection; still 1 train per seed
        # for full fraction, run all seeds too (training rng differs)
        if fr >= 0.999:
            aucs = []
            for sd in seeds:
                r = train_eval(Xtr_all, ytr, src_tr, Xte_all, yte, seed=sd)
                aucs.append(r["AUROC"])
        lc["points"].append({
            "fraction": fr, "n_train_tables": int(n_tab),
            "n_train_examples": int(np.array([s in (set(uniq_src.tolist()) if fr>=0.999 else sel_src) for s in src_tr]).sum()),
            "AUROC_per_seed": [round(a, 4) for a in aucs],
            "AUROC_mean": round(float(np.mean(aucs)), 4),
            "AUROC_std": round(float(np.std(aucs)), 4)})
        print(f"  frac={fr} ntab={n_tab} AUROC={np.mean(aucs):.4f}+-{np.std(aucs):.4f} {aucs}")
    json.dump(lc, open(f"{ABL}/learning_curve.json", "w"), indent=2)
    summary["learning_curve"] = [{"frac": p["fraction"], "n_tables": p["n_train_tables"],
                                  "AUROC": p["AUROC_mean"], "std": p["AUROC_std"]} for p in lc["points"]]

    # ============ ABLATION 2: magnitude stratification ============
    print("\n[ABL2] magnitude stratification (fixed full-train 14B probe) ...")
    # use the base headline scores (full train) already computed
    # proxies: n_cells_changed AND max relative delta
    pos_idx = np.where(yte == 1)[0]
    neg_idx = np.where(yte == 0)[0]
    neg_scores = base_scores[neg_idx]  # shared honest negatives
    ncc_arr = np.array([fraud.get(int(src_te[i]), {}).get("n_cells_changed") or 0 for i in pos_idx])
    rd_arr = np.array([rel_delta_from_prov(fraud.get(int(src_te[i]), {}).get("prov", [])) for i in pos_idx])
    pos_scores = base_scores[pos_idx]

    def bin_report(values, vname, pos_scores, neg_scores, src_pos):
        # tertiles
        qs = np.quantile(values, [1/3, 2/3])
        bins = {"low": values <= qs[0],
                "med": (values > qs[0]) & (values <= qs[1]),
                "high": values > qs[1]}
        rep = {"proxy": vname, "tertile_cutoffs": [round(float(qs[0]), 4), round(float(qs[1]), 4)], "bins": {}}
        for bn, bm in bins.items():
            ps = pos_scores[bm]
            if len(ps) == 0:
                continue
            y = np.concatenate([np.ones(len(ps)), np.zeros(len(neg_scores))])
            sc = np.concatenate([ps, neg_scores])
            au = float(roc_auc_score(y, sc)) if len(set(y.tolist())) > 1 else None
            # detection rate at the headline F1-opt threshold (0.3191) and 0.5
            dr05 = float(np.mean(ps >= 0.5))
            dr_f1 = float(np.mean(ps >= base["f1_opt_threshold"]))
            rep["bins"][bn] = {"n_pos": int(len(ps)),
                               "value_range": [round(float(values[bm].min()), 4), round(float(values[bm].max()), 4)],
                               "AUROC_vs_honest_neg": None if au is None else round(au, 4),
                               "mean_pos_score": round(float(ps.mean()), 4),
                               "detect_rate@0.5": round(dr05, 4),
                               "detect_rate@f1opt": round(dr_f1, 4)}
        return rep

    mag = {"note": "fixed full-train 14B probe scores; each positive bin vs the SAME 1000 honest negatives",
           "headline_f1opt_threshold": base["f1_opt_threshold"],
           "by_n_cells_changed": bin_report(ncc_arr.astype(float), "n_cells_changed", pos_scores, neg_scores, src_te[pos_idx]),
           "by_relative_delta": bin_report(rd_arr, "max_relative_|new-orig|/|orig|_over_edited_cells", pos_scores, neg_scores, src_te[pos_idx])}
    json.dump(mag, open(f"{ABL}/magnitude_strat.json", "w"), indent=2)
    for proxy in ["by_n_cells_changed", "by_relative_delta"]:
        print(f"  {proxy}:")
        for bn, b in mag[proxy]["bins"].items():
            print(f"    {bn}: n={b['n_pos']} AUROC={b['AUROC_vs_honest_neg']} detect@0.5={b['detect_rate@0.5']}")
    summary["magnitude_strat"] = {p: {bn: b["AUROC_vs_honest_neg"] for bn, b in mag[p]["bins"].items()}
                                   for p in ["by_n_cells_changed", "by_relative_delta"]}

    # ============ ABLATION 3: layer ablation ============
    print("\n[ABL3] layer ablation ...")
    def layer_ablation(tag):
        Xtr_r, ytr_, _, src_tr_, taps = load_raw(tag, "train", labmap)
        Xte_r, yte_, _, _, _ = load_raw(tag, "test", labmap)
        n1, Ll, Hl = Xtr_r.shape
        res = {"tapped_layers": taps, "variants": {}}
        # single layers
        for li, lay in enumerate(taps):
            r = train_eval(Xtr_r[:, li, :], ytr_, src_tr_, Xte_r[:, li, :], yte_)
            res["variants"][f"single_layer_{lay}"] = {"AUROC": r["AUROC"], "AUPRC": r["AUPRC"]}
        # last layer only
        r = train_eval(Xtr_r[:, -1, :], ytr_, src_tr_, Xte_r[:, -1, :], yte_)
        res["variants"][f"last_layer_only_{taps[-1]}"] = {"AUROC": r["AUROC"], "AUPRC": r["AUPRC"]}
        # all concat (headline)
        r = train_eval(Xtr_r.reshape(n1, Ll * Hl), ytr_, src_tr_,
                       Xte_r.reshape(Xte_r.shape[0], Ll * Hl), yte_)
        res["variants"]["all_concat"] = {"AUROC": r["AUROC"], "AUPRC": r["AUPRC"]}
        return res
    layab = {"qwen14b": layer_ablation("qwen14b"), "qwen7b": layer_ablation("qwen7b")}
    json.dump(layab, open(f"{ABL}/layer_ablation.json", "w"), indent=2)
    for tag in ["qwen14b", "qwen7b"]:
        print(f"  {tag}: " + ", ".join(f"{k}={v['AUROC']}" for k, v in layab[tag]["variants"].items()))
    summary["layer_ablation"] = {tag: {k: v["AUROC"] for k, v in layab[tag]["variants"].items()}
                                  for tag in ["qwen14b", "qwen7b"]}

    # ============ ABLATION 4: reference upper-bound (oracle) ============
    print("\n[ABL4] reference upper-bound ...")
    # Implemented as documented ceiling proxy: the existing task already is
    # reference-paired at the dataset level (each fabricated example has its honest
    # twin via src_table_id). We compute an ANALYSIS upper bound: a probe given the
    # PAIRED reference signal = [fabricated 14B feature  ||  (fabricated - honest_twin)]
    # where the honest twin's cached 14B feature (the paired negative of the same
    # src_table_id) is concatenated as a difference vector. This is the cleanest
    # reference-aware feature realizable WITHOUT re-extraction, using only cached feats.
    # Build per-split src_table_id -> (pos_feat, neg_feat) using the concat features.
    def build_paired(Xall, y, src):
        pos = {}; neg = {}
        for i in range(len(y)):
            (pos if y[i] == 1 else neg)[int(src[i])] = Xall[i]
        sids = sorted(set(pos) & set(neg))
        Xpos = np.stack([pos[s] for s in sids])
        Xneg = np.stack([neg[s] for s in sids])
        return sids, Xpos, Xneg
    sids_tr, Xpos_tr, Xneg_tr = build_paired(Xtr_all, ytr, src_tr)
    sids_te, Xpos_te, Xneg_te = build_paired(Xte_all, yte, src_te)
    # oracle feature: [feat || feat - twin]; positives use pos-neg diff, negatives use neg-pos diff
    def oracle_feats(Xpos, Xneg):
        # for each table: a fabricated row (label1) feat=[Xpos || Xpos-Xneg]
        #                 an honest    row (label0) feat=[Xneg || Xneg-Xpos]
        fpos = np.concatenate([Xpos, Xpos - Xneg], axis=1)
        fneg = np.concatenate([Xneg, Xneg - Xpos], axis=1)
        X = np.concatenate([fpos, fneg], axis=0)
        y = np.concatenate([np.ones(len(fpos)), np.zeros(len(fneg))])
        s = np.concatenate([np.array(sids_tr if X is None else None)]) if False else None
        return X, y
    Xtr_o = np.concatenate([np.concatenate([Xpos_tr, Xpos_tr - Xneg_tr], 1),
                            np.concatenate([Xneg_tr, Xneg_tr - Xpos_tr], 1)], 0)
    ytr_o = np.concatenate([np.ones(len(Xpos_tr)), np.zeros(len(Xneg_tr))])
    src_o = np.concatenate([np.array(sids_tr), np.array(sids_tr)])
    Xte_o = np.concatenate([np.concatenate([Xpos_te, Xpos_te - Xneg_te], 1),
                            np.concatenate([Xneg_te, Xneg_te - Xpos_te], 1)], 0)
    yte_o = np.concatenate([np.ones(len(Xpos_te)), np.zeros(len(Xneg_te))])
    oracle = train_eval(Xtr_o, ytr_o.astype(np.int64), src_o, Xte_o, yte_o.astype(np.int64))
    ref = {"definition": "ANALYSIS UPPER-BOUND (oracle): detector is given the honest reference. "
           "Feature = [fabricated 14B concat feat || (feat - paired_honest_twin_feat)], "
           "labels paired by src_table_id (each table => 1 pos row fab vs its twin, 1 neg row honest vs its twin). "
           "Uses ONLY cached features, no re-extraction. Group-disjoint train/test by src_table_id preserved.",
           "n_train_pairs": len(sids_tr), "n_test_pairs": len(sids_te),
           "in_dim": int(Xtr_o.shape[1]),
           "oracle_test_AUROC": oracle["AUROC"], "oracle_test_AUPRC": oracle["AUPRC"],
           "headline_no_reference_AUROC": base["AUROC"],
           "gap": round((oracle["AUROC"] or 0) - (base["AUROC"] or 0), 4)}
    json.dump(ref, open(f"{ABL}/reference_upperbound.json", "w"), indent=2)
    print(f"  oracle AUROC={oracle['AUROC']} vs headline(no-ref)={base['AUROC']} gap={ref['gap']}")
    summary["reference_upperbound"] = {"oracle_AUROC": oracle["AUROC"], "headline_AUROC": base["AUROC"], "gap": ref["gap"]}

    # ============ ABLATION 5: strategy-OOD ============
    print("\n[ABL5] strategy-OOD ...")
    def strat_of(sid):
        return fraud.get(int(sid), {}).get("strategy")
    # strategy per positive example (negatives have no strategy; pair them by src)
    # train positives' strategies
    strat_tr_pos = np.array([strat_of(s) for s in src_tr])  # only meaningful where y==1
    strat_counts_tr = collections.Counter(strat_tr_pos[ytr == 1].tolist())
    strat_te_pos = np.array([strat_of(s) for s in src_te])
    strat_counts_te = collections.Counter(strat_te_pos[yte == 1].tolist())
    # restrict to main strategies with enough support
    main_strats = [s for s, c in strat_counts_te.items() if c >= 5 and s is not None]
    ood = {"strategy_counts_train_pos": {str(k): v for k, v in strat_counts_tr.items()},
           "strategy_counts_test_pos": {str(k): v for k, v in strat_counts_te.items()},
           "held_out_strategies_evaluated": main_strats,
           "in_distribution_AUROC_full": base["AUROC"], "per_strategy": {}}
    for hs in main_strats:
        # TRAIN: all positives whose strategy != hs, plus ALL negatives (paired by src grouping kept).
        # We keep negatives whose src pairs with a kept positive's table OR all negatives (honest table
        # identity is the same pool). To preserve pairing/grouping we keep neg rows whose src is NOT a
        # held-out positive's table; train negatives come from training tables only (already src-disjoint
        # from test). Simplest correct: train pos = (y==1 & strat!=hs); train neg = (y==0) of training set.
        tr_mask = ((ytr == 1) & (strat_tr_pos != hs)) | (ytr == 0)
        # test: held-out strategy positives + ALL honest negatives (shared)
        te_mask = ((yte == 1) & (strat_te_pos == hs)) | (yte == 0)
        r = train_eval(Xtr_all[tr_mask], ytr[tr_mask], src_tr[tr_mask],
                       Xte_all[te_mask], yte[te_mask])
        ood["per_strategy"][str(hs)] = {
            "n_test_pos_heldout": int(((yte == 1) & (strat_te_pos == hs)).sum()),
            "n_test_neg": int((yte == 0).sum()),
            "n_train_pos_kept": int(((ytr == 1) & (strat_tr_pos != hs)).sum()),
            "OOD_AUROC": r["AUROC"], "OOD_AUPRC": r["AUPRC"]}
        print(f"  held-out={hs}: OOD AUROC={r['AUROC']} (n_test_pos={ood['per_strategy'][str(hs)]['n_test_pos_heldout']})")
    json.dump(ood, open(f"{ABL}/strategy_ood.json", "w"), indent=2)
    summary["strategy_ood"] = {k: v["OOD_AUROC"] for k, v in ood["per_strategy"].items()}
    summary["strategy_ood"]["in_distribution_full"] = base["AUROC"]

    # ============ ABLATION 6: ROC/PR curves data ============
    print("\n[ABL6] ROC/PR curve dumps ...")
    def load_scores(path, is_probe):
        d = json.load(open(path))
        sc = {}
        for k, v in d.items():
            sc[k] = float(v) if is_probe else float(v["score"])
        return sc
    methods = {
        "probe_14b": ("results/probe_preds_14b.json", True),
        "probe_7b": ("results/probe_preds_7b.json", True),
        "gemma3_zeroshot": ("results/baseline_preds_gemma3_zeroshot.json", False),
        "claude_opus_zeroshot": ("results/baseline_preds_claude_zeroshot.json", False),
        "rounding": ("results/baseline_preds_rounding.json", False),
    }
    ylab = {e: labmap[e]["label"] for e in labmap}
    curves = {"note": "fpr/tpr/thresholds (ROC) and precision/recall (PR) on the 2000-row test; "
              "scores from saved preds; labels from mapping.jsonl"}
    for name, (path, isp) in methods.items():
        sc = load_scores(f"{ROOT}/{path}", isp)
        eids = [e for e in sc if e in ylab]
        y = np.array([ylab[e] for e in eids])
        s = np.array([sc[e] for e in eids])
        fpr, tpr, thr = roc_curve(y, s)
        prec, rec, pthr = precision_recall_curve(y, s)
        curves[name] = {
            "n": int(len(y)),
            "AUROC": round(float(roc_auc_score(y, s)), 4),
            "AUPRC": round(float(average_precision_score(y, s)), 4),
            "roc": {"fpr": [round(float(x), 5) for x in fpr],
                    "tpr": [round(float(x), 5) for x in tpr],
                    "thresholds": [round(float(x), 6) for x in thr]},
            "pr": {"precision": [round(float(x), 5) for x in prec],
                   "recall": [round(float(x), 5) for x in rec],
                   "thresholds": [round(float(x), 6) for x in pthr]}}
        print(f"  {name}: n={len(y)} AUROC={curves[name]['AUROC']} AUPRC={curves[name]['AUPRC']}")
    json.dump(curves, open(f"{ABL}/roc_pr_curves.json", "w"), indent=2)

    # ============ FINAL MAIN TABLE ============
    print("\n[FINAL] main table ...")
    bm = json.load(open(f"{RESULTS}/baseline_metrics.json"))
    rowmap = {r.get("baseline"): r for r in bm["rows"]}
    # ordered display rows
    order = [("rules", "Rules (regex)"), ("benford", "Benford"),
             ("terminal_digit", "Terminal-digit"), ("rounding", "Rounding"),
             ("ours_dominance_allwin", "Ours-dominance (allwin)"),
             ("ours_dominance_margin", "Ours-dominance (margin)"),
             ("gemma3_12b_zeroshot", "Gemma3-12B zero-shot"),
             ("claude_opus_zeroshot", "Claude Opus zero-shot"),
             ("hidden_probe_7B", "Hidden-probe 7B (ours)"),
             ("hidden_probe_14B", "Hidden-probe 14B (ours, headline)")]
    main_rows = []
    for key, disp in order:
        r = rowmap.get(key)
        if r is None:
            continue
        main_rows.append({"method": disp, "key": key,
                          "AUROC": r.get("AUROC"), "AUPRC": r.get("AUPRC"),
                          "F1": r.get("F1"), "accuracy": r.get("accuracy", r.get("acc")),
                          "precision": r.get("precision"), "recall": r.get("recall"),
                          "headline": key == "hidden_probe_14B"})
    main_tbl = {"test_set": bm["test_set"], "n": 2000,
                "metric_notes": bm["metric_notes"], "rows": main_rows}
    json.dump(main_tbl, open(f"{RESULTS}/final_main_table.json", "w"), indent=2)
    # markdown
    def fmt(x):
        return "-" if x is None else f"{x:.4f}"
    lines = ["# Final Main Table — SciNumBench fabrication detection (2000-row src-disjoint test)", "",
             "| Method | AUROC | AUPRC | F1 | Acc | Prec | Recall |",
             "|---|---|---|---|---|---|---|"]
    for r in main_rows:
        nm = f"**{r['method']}**" if r["headline"] else r["method"]
        vals = [fmt(r["AUROC"]), fmt(r["AUPRC"]), fmt(r["F1"]), fmt(r["accuracy"]),
                fmt(r["precision"]), fmt(r["recall"])]
        if r["headline"]:
            vals = [f"**{v}**" for v in vals]
        lines.append(f"| {nm} | " + " | ".join(vals) + " |")
    lines += ["", "Notes: probe rows at threshold 0.5 (F1-opt variants in probe_metrics.json). "
              "Statistical baselines at native unsupervised decision; LLM zero-shot at Yes/No verdict."]
    open(f"{RESULTS}/final_main_table.md", "w").write("\n".join(lines))
    print("\n".join(lines))

    # ============ FINAL ABLATION TABLE ============
    print("\n[FINAL] ablation table ...")
    al = ["# Final Ablation Table (Plan B, 14B headline unless noted)", ""]
    al += ["## 1. Learning curve (test AUROC vs #train tables, mean over seeds 42,1,2)",
           "| frac | #tables | AUROC mean | std |", "|---|---|---|---|"]
    for p in lc["points"]:
        al.append(f"| {p['fraction']} | {p['n_train_tables']} | {p['AUROC_mean']:.4f} | {p['AUROC_std']:.4f} |")
    al += ["", "## 2. Magnitude stratification (AUROC of each positive bin vs shared honest negs)",
           "| proxy | low | med | high |", "|---|---|---|---|"]
    for proxy in ["by_n_cells_changed", "by_relative_delta"]:
        b = mag[proxy]["bins"]
        al.append(f"| {proxy} | {fmt(b.get('low',{}).get('AUROC_vs_honest_neg'))} | "
                  f"{fmt(b.get('med',{}).get('AUROC_vs_honest_neg'))} | "
                  f"{fmt(b.get('high',{}).get('AUROC_vs_honest_neg'))} |")
    al += ["", "## 3. Layer ablation (test AUROC)", "| variant | 14B | 7B |", "|---|---|---|"]
    v14 = layab["qwen14b"]["variants"]; v7 = layab["qwen7b"]["variants"]
    # align by descriptive key
    keys14 = list(v14.keys())
    for k in keys14:
        al.append(f"| {k} | {fmt(v14[k]['AUROC'])} | "
                  f"{fmt(v7.get(k, {}).get('AUROC'))} |")
    al += ["", "## 4. Reference upper-bound (oracle, paired-difference feature)",
           f"oracle AUROC = **{ref['oracle_test_AUROC']}** vs headline (no reference) {base['AUROC']} "
           f"(gap +{ref['gap']}).", ""]
    al += ["## 5. Strategy-OOD (train on other strategies, test held-out)",
           f"In-distribution (full) AUROC = {base['AUROC']}", "",
           "| held-out strategy | n_test_pos | OOD AUROC |", "|---|---|---|"]
    for hs, v in ood["per_strategy"].items():
        al.append(f"| {hs} | {v['n_test_pos_heldout']} | {fmt(v['OOD_AUROC'])} |")
    open(f"{RESULTS}/final_ablation_table.md", "w").write("\n".join(al))
    final_abl = {"learning_curve": lc["points"], "magnitude": summary["magnitude_strat"],
                 "layer_ablation": summary["layer_ablation"],
                 "reference_upperbound": summary["reference_upperbound"],
                 "strategy_ood": summary["strategy_ood"]}
    json.dump(final_abl, open(f"{RESULTS}/final_ablation_table.json", "w"), indent=2)
    print("\n".join(al))

    json.dump(summary, open(f"{ABL}/_summary.json", "w"), indent=2)
    print("\n[DONE] all ablations + tables written.")


if __name__ == "__main__":
    main()
