"""
method_hidden_probe_enhanced.py — SciNumBench (ICDE26) enhanced main-method experiment.

Reuses existing npz split/eval_id (TEST=600, never re-split), running:
  1. 7B multi-layer concat (mp14+mp21+mp28[+last]) LR+XGB
  2. 14B best single layer + multi-layer concat LR+XGB (requires 14b npz first)
  3. MLP head comparison (best features)
  4. (optional) 7B+14B concat upper-bound exploration

Layers/thresholds/hyperparameters are selected only on VAL; TEST is evaluated only once at the end.
All TEST verdicts are written as {eval_id: ...} for official score_llmfraud verification.
Summary -> results/method_hidden_probe_enhanced.json, with verdicts written per variant.

CLI:
  python method_hidden_probe_enhanced.py --do 7b            # 7B only (no GPU)
  python method_hidden_probe_enhanced.py --do all           # all (requires 14b npz)
"""
import os
os.environ.setdefault("TMPDIR", "/data/users/pengminjie/Backup/tmp")
import sys
import json
import argparse
import numpy as np

ROOT = "/home/pengminjie/Backup/paper/ICDE26"
FEATDIR = ROOT + "/data/features"
RESULTS = ROOT + "/results"
NPZ_7B = FEATDIR + "/hidden_probe_feats.npz"
NPZ_14B = FEATDIR + "/hidden_probe_14b_feats.npz"
OUT_JSON = RESULTS + "/method_hidden_probe_enhanced.json"
SEED = 42

BASELINE_AUROC = 0.627
BASELINE_F1 = 0.49
BASELINE_BEST_F1 = 0.36  # ours-margin judge


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
    FPR = FP / (FP + TN) if (FP + TN) else 0.0
    return dict(TP=TP, FP=FP, FN=FN, TN=TN,
                P=round(P, 4), R=round(R, 4), F1=round(F1, 4),
                Acc=round(Acc, 4), FPR=round(FPR, 4))


def choose_threshold(y_val, score_val, max_fpr=0.2):
    cands = np.unique(score_val)
    best = None; best_relaxed = None
    for thr in cands:
        pred = (score_val >= thr).astype(int)
        m = metrics_from_preds(y_val, pred)
        if best_relaxed is None or m["F1"] > best_relaxed[1]["F1"]:
            best_relaxed = (thr, m)
        if m["FPR"] <= max_fpr:
            if best is None or m["F1"] > best[1]["F1"]:
                best = (thr, m)
    if best is not None:
        return float(best[0]), best[1], True
    return float(best_relaxed[0]), best_relaxed[1], False


def load_split(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    split = d["split"].astype(str)
    y = d["label"].astype(int)
    strategy = d["strategy"].astype(str)
    eval_id = d["eval_id"].astype(int)
    src = d["src_table_id"].astype(int)
    sid = d["sid"].astype(str)
    tr = split == "train"; va = split == "val"; te = split == "test"
    s_tr, s_va, s_te = set(src[tr]), set(src[va]), set(src[te])
    assert not (s_tr & s_va) and not (s_tr & s_te) and not (s_va & s_te), \
        f"LEAK in {npz_path}!"
    return d, dict(split=split, y=y, strategy=strategy, eval_id=eval_id,
                   src=src, sid=sid, tr=tr, va=va, te=te,
                   s_tr=s_tr, s_va=s_va, s_te=s_te)


def build_feature(d, keys):
    """Concatenate several keys (each already (N,H)) -> (N, sum H), float32."""
    mats = [d[k].astype(np.float32) for k in keys]
    return np.concatenate(mats, axis=1)


def eval_clf_variant(name, X, meta, clf_kind, want_perstrat=True):
    """Uniformly train/select-threshold/evaluate one variant. clf_kind in {'lr','xgb'}. Returns dict + (test verdicts dict)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    tr, va, te = meta["tr"], meta["va"], meta["te"]
    y = meta["y"]

    if clf_kind == "lr":
        scaler = StandardScaler().fit(X[tr])
        Xtr, Xva, Xte = scaler.transform(X[tr]), scaler.transform(X[va]), scaler.transform(X[te])
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000, penalty="l2")
        clf.fit(Xtr, y[tr])
        sva = clf.predict_proba(Xva)[:, 1]; ste = clf.predict_proba(Xte)[:, 1]
    elif clf_kind == "xgb":
        from xgboost import XGBClassifier
        n_pos, n_neg = int(y[tr].sum()), int((1 - y[tr]).sum())
        spw = n_neg / max(1, n_pos)
        clf = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
                            eval_metric="logloss", tree_method="hist", n_jobs=8,
                            random_state=SEED)
        clf.fit(X[tr], y[tr])
        sva = clf.predict_proba(X[va])[:, 1]; ste = clf.predict_proba(X[te])[:, 1]
    else:
        raise ValueError(clf_kind)

    val_auc = round(float(roc_auc_score(y[va], sva)), 4)
    test_auc = round(float(roc_auc_score(y[te], ste)), 4)
    thr, val_op, feas = choose_threshold(y[va], sva, max_fpr=0.2)
    pred_te = (ste >= thr).astype(int)
    m_op = metrics_from_preds(y[te], pred_te)
    m_05 = metrics_from_preds(y[te], (ste >= 0.5).astype(int))
    degen = len(set(pred_te.tolist())) <= 1

    per_strat = {}
    if want_perstrat:
        strat = meta["strategy"]; te_idx = np.where(te)[0]
        for s in sorted(set(strat[te])):
            pos_mask = (strat[te] == s) & (y[te] == 1)
            n = int(pos_mask.sum())
            if n == 0:
                continue
            caught = int(np.sum(pred_te[pos_mask] == 1))
            per_strat[s] = dict(n=n, recall=round(caught / n, 4))

    verdicts = {}
    eids = meta["eval_id"][te]
    for j in range(len(eids)):
        verdicts[str(int(eids[j]))] = "FABRICATED" if pred_te[j] == 1 else "PLAUSIBLE"

    block = {"name": name, "clf": clf_kind, "val_AUROC": val_auc, "test_AUROC": test_auc,
             "thr": round(thr, 4), "thr_feasible_fpr_le_0.2": feas,
             "val_op": val_op, "test_op": m_op, "test_at_0.5": m_05,
             "degenerate": degen, "per_strategy_recall": per_strat,
             "beats_0627": test_auc > BASELINE_AUROC, "beats_049_F1": m_op["F1"] > BASELINE_F1}
    print(f"  [{clf_kind.upper():3s}] {name:28s} valAUC={val_auc:.4f} "
          f"testAUC={test_auc:.4f} F1={m_op['F1']:.4f} thr={thr:.3f} degen={degen}")
    return block, verdicts


def select_best_single(d, meta, keys, clf_kind="lr"):
    """Train LR per key, select best single layer by val AUROC. Returns best_key, val_auroc_dict."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    tr, va = meta["tr"], meta["va"]; y = meta["y"]
    val_auc = {}
    for k in keys:
        X = d[k].astype(np.float32)
        scaler = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000)
        clf.fit(scaler.transform(X[tr]), y[tr])
        sva = clf.predict_proba(scaler.transform(X[va]))[:, 1]
        val_auc[k] = round(float(roc_auc_score(y[va], sva)), 4)
    best = max(val_auc, key=val_auc.get)
    return best, val_auc


def mlp_variant(name, X, meta, hidden=(256, 64), epochs=200, lr=1e-3, wd=1e-4, patience=20):
    """Small MLP head, early-stop on val AUROC."""
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    torch.manual_seed(SEED); np.random.seed(SEED)
    tr, va, te = meta["tr"], meta["va"], meta["te"]; y = meta["y"]
    scaler = StandardScaler().fit(X[tr])
    Xtr = torch.tensor(scaler.transform(X[tr]), dtype=torch.float32)
    Xva = torch.tensor(scaler.transform(X[va]), dtype=torch.float32)
    Xte = torch.tensor(scaler.transform(X[te]), dtype=torch.float32)
    ytr = torch.tensor(y[tr], dtype=torch.float32)

    layers = []; dim = X.shape[1]
    for h in hidden:
        layers += [nn.Linear(dim, h), nn.ReLU(), nn.Dropout(0.3)]; dim = h
    layers += [nn.Linear(dim, 1)]
    net = nn.Sequential(*layers)
    n_pos, n_neg = float(y[tr].sum()), float((1 - y[tr]).sum())
    pos_weight = torch.tensor([n_neg / max(1.0, n_pos)])
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)

    best_auc = -1; best_state = None; best_sva = None; bad = 0
    for ep in range(epochs):
        net.train(); opt.zero_grad()
        logit = net(Xtr).squeeze(-1)
        loss = crit(logit, ytr); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            sva = torch.sigmoid(net(Xva).squeeze(-1)).numpy()
        auc = roc_auc_score(y[va], sva)
        if auc > best_auc + 1e-4:
            best_auc = auc; best_state = {k: v.clone() for k, v in net.state_dict().items()}
            best_sva = sva; bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state); net.eval()
    with torch.no_grad():
        ste = torch.sigmoid(net(Xte).squeeze(-1)).numpy()
    val_auc = round(float(best_auc), 4)
    test_auc = round(float(roc_auc_score(y[te], ste)), 4)
    thr, val_op, feas = choose_threshold(y[va], best_sva, max_fpr=0.2)
    pred_te = (ste >= thr).astype(int)
    m_op = metrics_from_preds(y[te], pred_te)
    m_05 = metrics_from_preds(y[te], (ste >= 0.5).astype(int))
    degen = len(set(pred_te.tolist())) <= 1
    strat = meta["strategy"]; per_strat = {}
    for s in sorted(set(strat[te])):
        pos_mask = (strat[te] == s) & (y[te] == 1); n = int(pos_mask.sum())
        if n == 0: continue
        per_strat[s] = dict(n=n, recall=round(int(np.sum(pred_te[pos_mask] == 1)) / n, 4))
    verdicts = {}; eids = meta["eval_id"][te]
    for j in range(len(eids)):
        verdicts[str(int(eids[j]))] = "FABRICATED" if pred_te[j] == 1 else "PLAUSIBLE"
    block = {"name": name, "clf": "mlp", "hidden": list(hidden), "val_AUROC": val_auc,
             "test_AUROC": test_auc, "thr": round(thr, 4), "thr_feasible_fpr_le_0.2": feas,
             "val_op": val_op, "test_op": m_op, "test_at_0.5": m_05, "degenerate": degen,
             "per_strategy_recall": per_strat, "beats_0627": test_auc > BASELINE_AUROC,
             "beats_049_F1": m_op["F1"] > BASELINE_F1}
    print(f"  [MLP] {name:28s} valAUC={val_auc:.4f} testAUC={test_auc:.4f} "
          f"F1={m_op['F1']:.4f} thr={thr:.3f} degen={degen}")
    return block, verdicts


def save_verdicts(tag, verdicts):
    path = RESULTS + f"/verdicts_enh_{tag}.json"
    json.dump(verdicts, open(path, "w"))
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--do", choices=["7b", "all"], default="all")
    a = ap.parse_args()

    variants = {}
    verdict_paths = {}

    # ───────── 7B ─────────
    d7, m7 = load_split(NPZ_7B)
    print(f"  npz split: train={m7['tr'].sum()} val={m7['va'].sum()} test={m7['te'].sum()}  "
          f"zero-overlap tv={len(m7['s_tr']&m7['s_va'])} tt={len(m7['s_tr']&m7['s_te'])} "
          f"vt={len(m7['s_va']&m7['s_te'])}")
    keys7_mp = ["mp14", "mp21", "mp28"]
    keys7_all = ["mp14", "last14", "mp21", "last21", "mp28", "last28"]

    # single-layer baseline recheck (mp21)
    for clf in ("lr", "xgb"):
        b, v = eval_clf_variant("7b_single_mp21", d7["mp21"].astype(np.float32), m7, clf)
        variants[f"7b_single_mp21_{clf}"] = b
        verdict_paths[f"7b_single_mp21_{clf}"] = save_verdicts(f"7b_single_mp21_{clf}", v)

    # multi-layer concat (mp only)
    Xcat_mp = build_feature(d7, keys7_mp)
    for clf in ("lr", "xgb"):
        b, v = eval_clf_variant("7b_concat_mp", Xcat_mp, m7, clf)
        variants[f"7b_concat_mp_{clf}"] = b
        verdict_paths[f"7b_concat_mp_{clf}"] = save_verdicts(f"7b_concat_mp_{clf}", v)

    # multi-layer concat (all mp+last)
    Xcat_all = build_feature(d7, keys7_all)
    for clf in ("lr", "xgb"):
        b, v = eval_clf_variant("7b_concat_all", Xcat_all, m7, clf)
        variants[f"7b_concat_all_{clf}"] = b
        verdict_paths[f"7b_concat_all_{clf}"] = save_verdicts(f"7b_concat_all_{clf}", v)

    # MLP on 7B best (concat mp)
    try:
        b, v = mlp_variant("7b_concat_mp_mlp", Xcat_mp, m7)
        variants["7b_concat_mp_mlp"] = b
        verdict_paths["7b_concat_mp_mlp"] = save_verdicts("7b_concat_mp_mlp", v)
        b, v = mlp_variant("7b_single_mp21_mlp", d7["mp21"].astype(np.float32), m7)
        variants["7b_single_mp21_mlp"] = b
        verdict_paths["7b_single_mp21_mlp"] = save_verdicts("7b_single_mp21_mlp", v)
    except Exception as e:
        print(f"  [MLP 7B skipped] {e}")

    # ───────── 14B ─────────
    if a.do == "all" and os.path.exists(NPZ_14B):
        d14, m14 = load_split(NPZ_14B)
        # alignment check: 14B test eval_id set should match 7B
        same_test = set(m14["eval_id"][m14["te"]].tolist()) == set(m7["eval_id"][m7["te"]].tolist())
        same_train_sid = set(m14["sid"][m14["tr"]].tolist()) == set(m7["sid"][m7["tr"]].tolist())
        print(f"  align: test eval_id matches 7B={same_test}  train sid matches={same_train_sid}  "
              f"zero-overlap tt={len(m14['s_tr']&m14['s_te'])}")
        mp_keys14 = sorted([k for k in d14.files if k.startswith("mp")],
                           key=lambda x: int(x[2:]))
        last_keys14 = sorted([k for k in d14.files if k.startswith("last")],
                             key=lambda x: int(x[4:]))
        all_keys14 = mp_keys14 + last_keys14
        print(f"  14B layers present: mp={mp_keys14} last={last_keys14}")

        # best single layer: selected by val AUROC (over all mp+last)
        best14, val_auc14 = select_best_single(d14, m14, all_keys14)
        print(f"  [SELECT 14B single] best={best14} val_auroc={val_auc14}")
        for clf in ("lr", "xgb"):
            b, v = eval_clf_variant(f"14b_single_{best14}", d14[best14].astype(np.float32), m14, clf)
            b["selected_from_val"] = val_auc14
            variants[f"14b_single_{clf}"] = b
            verdict_paths[f"14b_single_{clf}"] = save_verdicts(f"14b_single_{clf}", v)

        # multi-layer concat (all mp)
        Xcat14_mp = build_feature(d14, mp_keys14)
        for clf in ("lr", "xgb"):
            b, v = eval_clf_variant("14b_concat_mp", Xcat14_mp, m14, clf)
            variants[f"14b_concat_mp_{clf}"] = b
            verdict_paths[f"14b_concat_mp_{clf}"] = save_verdicts(f"14b_concat_mp_{clf}", v)

        # multi-layer concat (all mp+last)
        Xcat14_all = build_feature(d14, all_keys14)
        for clf in ("lr", "xgb"):
            b, v = eval_clf_variant("14b_concat_all", Xcat14_all, m14, clf)
            variants[f"14b_concat_all_{clf}"] = b
            verdict_paths[f"14b_concat_all_{clf}"] = save_verdicts(f"14b_concat_all_{clf}", v)

        # MLP on 14B concat mp
        try:
            b, v = mlp_variant("14b_concat_mp_mlp", Xcat14_mp, m14)
            variants["14b_concat_mp_mlp"] = b
            verdict_paths["14b_concat_mp_mlp"] = save_verdicts("14b_concat_mp_mlp", v)
        except Exception as e:
            print(f"  [MLP 14B skipped] {e}")

        # ───── 7B+14B concat upper-bound exploration (requires sid alignment) ─────
        try:
            # align the two feature sets by sid key (order may differ)
            sid7 = m7["sid"]; sid14 = m14["sid"]
            if set(sid7.tolist()) == set(sid14.tolist()):
                order14 = {s: i for i, s in enumerate(sid14)}
                perm = np.array([order14[s] for s in sid7])
                X7 = build_feature(d7, keys7_mp)
                X14 = build_feature(d14, mp_keys14)[perm]  # reorder to 7B order
                Xcomb = np.concatenate([X7, X14], axis=1)
                for clf in ("lr", "xgb"):
                    b, v = eval_clf_variant("combo_7b14b_concat_mp", Xcomb, m7, clf)
                    variants[f"combo_7b14b_{clf}"] = b
                    verdict_paths[f"combo_7b14b_{clf}"] = save_verdicts(f"combo_7b14b_{clf}", v)
            else:
                print("  [combo skipped] sid sets do not match")
        except Exception as e:
            print(f"  [combo skipped] {e}")
    elif a.do == "all":
        print(f"\n[14B SKIPPED] {NPZ_14B} does not exist, producing 7B part only.")

    # ───────── summary ─────────
    # find best variant (by test AUROC)
    best_name = max(variants, key=lambda k: variants[k]["test_AUROC"])
    best_f1_name = max(variants, key=lambda k: variants[k]["test_op"]["F1"])

    summary_table = {k: dict(test_AUROC=variants[k]["test_AUROC"],
                             test_F1=variants[k]["test_op"]["F1"],
                             test_P=variants[k]["test_op"]["P"],
                             test_R=variants[k]["test_op"]["R"],
                             test_FPR=variants[k]["test_op"]["FPR"],
                             val_AUROC=variants[k]["val_AUROC"],
                             degenerate=variants[k]["degenerate"],
                             beats_0627=variants[k]["beats_0627"],
                             beats_049_F1=variants[k]["beats_049_F1"])
                     for k in variants}

    result = dict(
        note="Enhanced main method: 7B multi-layer concat / 14B single+concat / MLP / 7B+14B combo. Reuses existing npz split/eval_id, TEST=600 not re-split.",
        baselines=dict(prev_best_7b_single_AUROC=BASELINE_AUROC, prev_best_7b_single_F1=BASELINE_F1,
                       judge_ours_margin_best_F1=BASELINE_BEST_F1),
        models=dict(qwen7b="Qwen/Qwen2.5-7B-Instruct (hidden=3584, 28L sampled at 14/21/28)",
                    qwen14b="Qwen/Qwen2.5-14B-Instruct (hidden=5120, 48L)"),
        variants=variants,
        summary_table=summary_table,
        best_by_test_AUROC=dict(name=best_name, **summary_table[best_name]),
        best_by_test_F1=dict(name=best_f1_name, **summary_table[best_f1_name]),
        verdict_paths=verdict_paths,
        feature_npz=dict(qwen7b=NPZ_7B, qwen14b=NPZ_14B if os.path.exists(NPZ_14B) else None),
    )
    json.dump(result, open(OUT_JSON, "w"), ensure_ascii=False, indent=2)
    print(f"[BEST by AUROC] {best_name}: AUROC={summary_table[best_name]['test_AUROC']} "
          f"F1={summary_table[best_name]['test_F1']}")
    print(f"[BEST by F1]    {best_f1_name}: AUROC={summary_table[best_f1_name]['test_AUROC']} "
          f"F1={summary_table[best_f1_name]['test_F1']}")


if __name__ == "__main__":
    main()
