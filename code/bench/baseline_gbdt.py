"""
baseline_gbdt.py — supervised GBDT/XGBoost baseline (non-LLM supervised baseline)

Supervised binary classification (label) on qwen7b_feats.npz hidden-state
features (mp14/mp21/mp28/last).
  - train to fit, val to select layer/threshold, test for main metrics, cross_test for cross-domain FPR
  - variants: (a) best single layer (selected by val AUROC) (b) four-layer concat; sklearn HGB + XGBoost
  - primary threshold-independent metrics AUROC / AP; operating-point threshold selected on val with an FPR constraint (prevents all-positive collapse)
  - candidate precision on the DIST candidate subset, compared against M6=0.92 / probe=0.57

Reuses evaluate.py conventions: table-level P/R/F1/Acc, clean FPR, per-family recall, cross_test FPR.
Outputs results/baseline_gbdt.json (with per-bench_id predictions).
"""
import os
# Cap thread pools BEFORE importing numpy/sklearn: uncapped OpenMP oversubscribes
# cores and thrashes when another job shares the box (observed hang). 8 is plenty.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "32")
import sys, json, time, argparse, sqlite3
import numpy as np
from collections import Counter
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = "/home/pengminjie/Backup/paper/ICDE26"
sys.path.insert(0, ROOT + "/code/bench")
from parser import StructuredTable  # noqa: E402

NPZ = ROOT + "/data/features/qwen7b_feats.npz"
DB = ROOT + "/data/arxiv_data.db"
OUT = ROOT + "/results/baseline_gbdt.json"
LAYERS = ["mp14", "mp21", "mp28", "last"]
FAMILIES = ["GRIM", "PVAL", "CI", "PCT", "DIST"]
SEED = 42
FPR_CAP = 0.20  # operating-point val FPR cap (same magnitude as M1=0.19/M6=0.20)


def metrics(y, pred):
    y = np.asarray(y); pred = np.asarray(pred)
    TP = int(((pred == 1) & (y == 1)).sum()); FP = int(((pred == 1) & (y == 0)).sum())
    FN = int(((pred == 0) & (y == 1)).sum()); TN = int(((pred == 0) & (y == 0)).sum())
    P = TP / (TP + FP) if TP + FP else 0.0
    R = TP / (TP + FN) if TP + FN else 0.0
    F1 = 2 * P * R / (P + R) if P + R else 0.0
    Acc = (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) else 0.0
    FPR = FP / (FP + TN) if (FP + TN) else 0.0
    return dict(TP=TP, FP=FP, FN=FN, TN=TN, P=P, R=R, F1=F1, Acc=Acc, FPR=FPR)


def safe_auc(y, prob):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, prob))


def safe_ap(y, prob):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return None
    return float(average_precision_score(y, prob))


def per_family_recall(fam, y, pred):
    fam = np.asarray(fam); y = np.asarray(y); pred = np.asarray(pred)
    out = {}
    for f in FAMILIES:
        m = (y == 1) & (fam == f)
        n = int(m.sum())
        out[f] = dict(n=n, recall=(float((pred[m] == 1).mean()) if n else None))
    return out


def pick_threshold(yval, prob_val, fpr_cap=FPR_CAP):
    """Select operating-point threshold on val: among thresholds with FPR<=cap, take max F1;
    if no threshold satisfies cap, fall back to the threshold whose FPR is closest to cap.
    Returns (thr, sel_F1, sel_FPR, mode). Candidate thresholds use val predicted-probability quantiles."""
    cands = np.unique(np.concatenate([
        np.linspace(0.02, 0.98, 49),
        np.quantile(prob_val, np.linspace(0.01, 0.99, 99))]))
    feas = []
    allm = []
    for thr in cands:
        m = metrics(yval, (prob_val >= thr).astype(int))
        allm.append((thr, m["F1"], m["FPR"]))
        if m["FPR"] <= fpr_cap:
            feas.append((thr, m["F1"], m["FPR"]))
    if feas:
        thr, f1, fpr = max(feas, key=lambda x: x[1])
        return float(thr), float(f1), float(fpr), "fpr_capped"
    # fallback: minimize |FPR - cap|
    thr, f1, fpr = min(allm, key=lambda x: abs(x[2] - fpr_cap))
    return float(thr), float(f1), float(fpr), "fpr_nearest"


def dist_candidate_ids():
    """Reuse the finetune_dist candidate definition: table cells with meansd and mean-2SD<0 (mean>0, sd>0), test split."""
    conn = sqlite3.connect(DB)
    rows = conn.execute(
        """SELECT sb.bench_id, sb.label, sb.corruption_family, sb.corrupted_grid, pt.caption
           FROM scinum_bench sb LEFT JOIN paper_tables pt ON sb.src_table_id=pt.id
           WHERE sb.dataset_split='test'""").fetchall()
    conn.close()
    cand = {}
    for b, y, fam, cg, cap in rows:
        try:
            g = json.loads(cg)
        except Exception:
            continue
        if len(g) < 2:
            continue
        st = StructuredTable(g[0], g[1:], caption=cap or "")
        is_cand = False
        for (i, j) in st.positions_of("meansd"):
            c = st.cells[i][j]
            if c["sd"] > 0 and c["mean"] > 0 and c["mean"] - 2 * c["sd"] < 0:
                is_cand = True
                break
        if is_cand:
            cand[int(b)] = dict(label=int(y), is_dist=1 if (fam or "") == "DIST" else 0,
                                clean=1 if int(y) == 0 else 0)
    return cand


def fit_prob(Xtr, ytr, Xeval_list, model="hgb"):
    """Standardize (fit on train) -> train -> predict probabilities for a set of eval splits. Returns list of prob arrays."""
    sc = StandardScaler().fit(Xtr)
    Xtr_s = sc.transform(Xtr).astype(np.float32)
    if model == "hgb":
        # NOTE: class_weight / sample_weight trigger a pathological >10x slowdown
        # (hang) in this sklearn 1.9 HGB build, so we do NOT weight here. The
        # imbalance is mild (pos_rate~0.6, 1.5:1); AUROC/AP are prevalence-robust
        # and the FPR-capped val threshold handles the operating point.
        clf = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.07, max_depth=None, l2_regularization=1.0,
            max_leaf_nodes=63, early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=15, random_state=SEED)
    elif model == "xgb":
        import xgboost as xgb
        pos = int((ytr == 1).sum()); neg = int((ytr == 0).sum())
        spw = float(neg) / float(pos) if pos else 1.0
        clf = xgb.XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8,
            colsample_bytree=0.6, reg_lambda=2.0, min_child_weight=2,
            scale_pos_weight=spw, tree_method="hist", random_state=SEED,
            n_jobs=8, eval_metric="logloss")
    else:
        raise ValueError(model)
    clf.fit(Xtr_s, ytr)
    return [clf.predict_proba(sc.transform(X).astype(np.float32))[:, 1] for X in Xeval_list]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0, help="if >0, subsample each split to N rows")
    ap.add_argument("--full", action="store_true", help="force full (smoke=0)")
    ap.add_argument("--no-xgb", action="store_true", help="skip xgboost")
    args = ap.parse_args()
    if args.full:
        args.smoke = 0
    run_xgb = not args.no_xgb

    t0 = time.time()
    d = np.load(NPZ, allow_pickle=True)
    bid = d["bench_id"].astype(int)
    label = d["label"].astype(int)
    split = d["split"].astype(str)
    family = d["family"].astype(str)
    source = d["source"].astype(str)

    idx = {s: np.where(split == s)[0] for s in ["train", "val", "test", "cross_test"]}
    if args.smoke:
        rng = np.random.RandomState(SEED)
        for s in idx:
            if len(idx[s]) > args.smoke:
                idx[s] = np.sort(rng.choice(idx[s], args.smoke, replace=False))
    print("[DATA] " + " ".join(f"{s}={len(idx[s])}" for s in idx), flush=True)

    def feats(layer, ix):
        return d[layer][ix].astype(np.float32)

    def feats_concat(ix):
        return np.concatenate([d[L][ix].astype(np.float32) for L in LAYERS], axis=1)

    ytr = label[idx["train"]]; yval = label[idx["val"]]; yte = label[idx["test"]]
    ycross = label[idx["cross_test"]]
    fam_te = family[idx["test"]]

    results = {"reference": {
        "M1_rules": {"P": 0.875, "R": 0.887, "F1": 0.881, "FPR": 0.190,
                     "cross_test_FPR": 0.000, "DIST_recall": 0.64},
        "M6_finetune_mix": {"P": 0.876, "R": 0.926, "F1": 0.900, "FPR": 0.196,
                            "cross_test_FPR": 0.004, "DIST_recall": 0.83,
                            "DIST_candidate_P": 0.92, "clean_cand_FPR": 0.06},
        "probe_M2": {"P": 0.678, "R": 0.647, "F1": 0.662, "FPR": 0.461,
                     "cross_test_FPR": 0.407, "DIST_candidate_P": 0.57},
    }, "fpr_cap": FPR_CAP, "variants": {}, "predictions": {}}

    def run_variant(name, get, model="hgb"):
        print(f"\n[VARIANT] {name} (model={model})", flush=True)
        ts = time.time()
        Xtr = get(idx["train"]); Xval = get(idx["val"]); Xte = get(idx["test"])
        Xcross = get(idx["cross_test"])
        prob_val, prob_te, prob_cross = fit_prob(Xtr, ytr, [Xval, Xte, Xcross], model)

        # threshold-independent primary metrics
        auroc_te = safe_auc(yte, prob_te); ap_te = safe_ap(yte, prob_te)
        auroc_val = safe_auc(yval, prob_val)

        # operating point: FPR-capped F1 on val
        thr, sel_f1, sel_fpr, thr_mode = pick_threshold(yval, prob_val)
        pred_te = (prob_te >= thr).astype(int)
        pred_cross = (prob_cross >= thr).astype(int)
        pred_val = (prob_val >= thr).astype(int)
        m_test = metrics(yte, pred_te)
        m_val = metrics(yval, pred_val)
        m_cross = metrics(ycross, pred_cross)
        # also report at fixed thr=0.5
        m_test_05 = metrics(yte, (prob_te >= 0.5).astype(int))
        m_cross_05 = metrics(ycross, (prob_cross >= 0.5).astype(int))

        pfr = per_family_recall(fam_te, yte, pred_te)
        by_src = {}
        for src in ["pmc", "arxiv"]:
            sm = source[idx["test"]] == src
            if sm.sum():
                by_src[src] = metrics(yte[sm], pred_te[sm])["R"]

        out = dict(
            model=model,
            test_AUROC=auroc_te, test_AP=ap_te, val_AUROC=auroc_val,
            thr=thr, thr_mode=thr_mode, val_sel_F1=sel_f1, val_sel_FPR=sel_fpr,
            test=m_test, val=m_val, test_at_thr05=m_test_05,
            cross_test_FPR=m_cross["FPR"], cross_test_FPR_at_thr05=m_cross_05["FPR"],
            cross_test_n=len(ycross),
            per_family_recall=pfr, test_recall_by_source=by_src,
            secs=round(time.time() - ts, 1))
        print(f"  AUROC={auroc_te:.3f} AP={ap_te:.3f} | thr={thr:.3f}({thr_mode}) "
              f"val_selF1={sel_f1:.3f} val_selFPR={sel_fpr:.3f}", flush=True)
        print(f"  test P={m_test['P']:.3f} R={m_test['R']:.3f} F1={m_test['F1']:.3f} "
              f"Acc={m_test['Acc']:.3f} FPR={m_test['FPR']:.3f} | "
              f"@0.5 F1={m_test_05['F1']:.3f} FPR={m_test_05['FPR']:.3f} | "
              f"cross_FPR={m_cross['FPR']:.3f} | {out['secs']}s", flush=True)
        results["variants"][name] = out
        for ix_arr, prob_arr, pred_arr in [
                (idx["test"], prob_te, pred_te), (idx["cross_test"], prob_cross, pred_cross)]:
            for k, ii in enumerate(ix_arr):
                results["predictions"].setdefault(name, {})[int(bid[ii])] = \
                    dict(prob=round(float(prob_arr[k]), 4), pred=int(pred_arr[k]),
                         label=int(label[ii]), family=str(family[ii]), split=str(split[ii]))
        return out

    # ---- single layer: pick best by val AUROC ----
    layer_val_auroc = {}
    for L in LAYERS:
        pv, = fit_prob(feats(L, idx["train"]), ytr, [feats(L, idx["val"])], "hgb")
        au = safe_auc(yval, pv)
        layer_val_auroc[L] = au
        print(f"  [layer-select] {L}: val_AUROC={au:.3f}", flush=True)
    best_layer = max(layer_val_auroc, key=lambda k: layer_val_auroc[k])
    results["layer_select_val_auroc"] = layer_val_auroc
    results["best_single_layer"] = best_layer
    print(f"[LAYER-SELECT] best single layer = {best_layer}", flush=True)

    run_variant(f"hgb_single_{best_layer}", lambda ix: feats(best_layer, ix), "hgb")
    run_variant("hgb_concat4", feats_concat, "hgb")

    if run_xgb:
        try:
            import xgboost  # noqa
            run_variant(f"xgb_single_{best_layer}", lambda ix: feats(best_layer, ix), "xgb")
            run_variant("xgb_concat4", feats_concat, "xgb")
        except Exception as e:
            print(f"[XGB] skipped: {e}", flush=True)
            results["xgb_error"] = str(e)

    # ---- DIST candidate precision (compared against M6=0.92 / probe=0.57) ----
    cand = dist_candidate_ids()
    cand_ids = set(cand.keys())
    cand_eval = {}
    for vname, pmap in results["predictions"].items():
        ids = [b for b in cand_ids if str(pmap.get(b, {}).get("split", "")) == "test"]
        if not ids:
            continue
        yc = np.array([cand[b]["is_dist"] for b in ids])  # 1 if DIST fake
        pc = np.array([pmap[b]["pred"] for b in ids])
        m = metrics(yc, pc)
        clean_ids = [b for b in ids if cand[b]["clean"] == 1]
        clean_fp = float(np.mean([pmap[b]["pred"] for b in clean_ids])) if clean_ids else None
        cand_eval[vname] = dict(n_cand=len(ids), n_dist_fake=int(yc.sum()),
                                n_clean_cand=len(clean_ids),
                                candidate_P=m["P"], candidate_R=m["R"],
                                clean_cand_FPR=clean_fp)
        print(f"[DIST-CAND] {vname}: n={len(ids)} dist_fake={int(yc.sum())} "
              f"P={m['P']:.3f} R={m['R']:.3f} clean_cand_FPR="
              f"{clean_fp if clean_fp is None else round(clean_fp,3)} (vs M6 P=0.92/FPR=0.06, probe P=0.57)",
              flush=True)
    results["dist_candidate"] = cand_eval

    # ---- integrity self-check ----
    checks = {}
    for vname, v in results["variants"].items():
        t = v["test"]
        degenerate = (t["TP"] + t["FP"] == 0) or (t["FN"] + t["TN"] == 0) \
            or (t["FPR"] >= 0.98 and t["R"] >= 0.98)
        below_probe = (t["F1"] < 0.662)
        checks[vname] = dict(degenerate=bool(degenerate),
                             below_probe_F1=bool(below_probe),
                             test_F1=t["F1"], test_AUROC=v["test_AUROC"])
        if degenerate:
            print(f"[WARN] {vname} test prediction DEGENERATE (all-pos/all-neg)!", flush=True)
        if below_probe:
            print(f"[NOTE] {vname} test F1={t['F1']:.3f} < probe 0.662", flush=True)
    results["integrity_check"] = checks

    results["meta"] = dict(npz=NPZ, n_total=len(bid), seed=SEED,
                           smoke=args.smoke, full=(args.smoke == 0),
                           total_secs=round(time.time() - t0, 1))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
