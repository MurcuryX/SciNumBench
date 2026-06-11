"""
method_hidden_probe.py — SciNumBench (ICDE26) main method: frozen-LLM hidden-state
probe to detect boost-ours fabrication.

Idea: feed each table (serialize output, identical to baseline/judge input) to a frozen
Qwen2.5-7B-Instruct, take mean-pool and last-token hidden states from several layers as
features; train L2 logistic regression (+ GBDT control) on TRAIN, select layer/pooling and
operating threshold by VAL AUROC, evaluate on TEST(600).

Strict leakage-free split:
  TEST = the 600 rows in results/llmfraud_eval.jsonl (300 src_table_id), reusing their
         eval_id and text.
  TRAIN = all other status='ok' llm_fraud rows (src_table_id not in the test 300),
          pos=fabricated_grid(label1), neg=original_grid(label0), serialized via serialize().
  VAL  = ~15% split from TRAIN by src_table_id (same src_table never crosses splits).

Stages (--stage):
  extract : extract hidden states -> data/features/hidden_probe_feats.npz
  train   : train probe / select layer / tune threshold / evaluate TEST -> results/method_hidden_probe_llmfraud.json + verdicts
  all     : run sequentially (default)
"""
import os
os.environ.setdefault("HF_HOME", "/data/shared_models/huggingface")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("TMPDIR", "/data/users/pengminjie/Backup/tmp")

import sys
import json
import time
import argparse
import sqlite3
import random
from collections import Counter

import numpy as np

ROOT = "/home/pengminjie/Backup/paper/ICDE26"
sys.path.insert(0, ROOT + "/code/bench")
from clean_serialize import serialize  # noqa: E402

DB = ROOT + "/data/arxiv_data.db"
EVAL = ROOT + "/results/llmfraud_eval.jsonl"
FEATDIR = ROOT + "/data/features"
FEATNPZ = FEATDIR + "/hidden_probe_feats.npz"
RESULTS = ROOT + "/results"
OUT_JSON = RESULTS + "/method_hidden_probe_llmfraud.json"
OUT_VERDICTS = RESULTS + "/verdicts_hidden_probe_llmfraud.json"

MODEL = "Qwen/Qwen2.5-7B-Instruct"
LAYERS = [14, 21, 28]
MAXTOK = 1024
BATCH = 16
VAL_FRAC = 0.15
SEED = 42


# ──────────────────────────── data split ────────────────────────────
def load_test_records():
    """Load the 600 test rows, reusing eval_id/text/label/strategy/src_table_id."""
    recs = []
    with open(EVAL) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            recs.append({
                "sid": f"test_{r['eval_id']}",
                "eval_id": int(r["eval_id"]),
                "src_table_id": int(r["src_table_id"]),
                "label": int(r["label"]),
                "strategy": r.get("strategy") or "",
                "split": "test",
                "text": r["text"],
            })
    return recs


def build_train_records(test_src_ids):
    """From DB take all status='ok' rows whose src_table_id is not in test, pos=fab/neg=orig.
    Then split ~15% by src_table_id as VAL. Returns records (split in {train,val})."""
    conn = sqlite3.connect(DB)
    rows = conn.execute(
        """SELECT src_table_id, strategy, caption,
                  original_grid, fabricated_grid, provenance
           FROM llm_fraud WHERE status='ok' ORDER BY src_table_id""").fetchall()
    conn.close()

    train_src = []
    rowmap = {}
    for (sid, strat, cap, og, fg, prov) in rows:
        sid = int(sid)
        if sid in test_src_ids:
            continue
        train_src.append(sid)
        rowmap[sid] = (strat or "", cap or "",
                       json.loads(og), json.loads(fg),
                       json.loads(prov) if prov else [])

    train_src = sorted(set(train_src))
    rng = random.Random(SEED)
    shuffled = train_src[:]
    rng.shuffle(shuffled)
    n_val = int(round(len(shuffled) * VAL_FRAC))
    val_src = set(shuffled[:n_val])

    recs = []
    for sid in train_src:
        strat, cap, og, fg, prov = rowmap[sid]
        split = "val" if sid in val_src else "train"
        # positive: fabricated (with provenance, matching build_llmfraud_eval)
        recs.append({
            "sid": f"{split}_{sid}_pos", "eval_id": -1, "src_table_id": sid,
            "label": 1, "strategy": strat, "split": split,
            "text": serialize(fg, cap, provenance=prov),
        })
        # negative: original (no provenance passed)
        recs.append({
            "sid": f"{split}_{sid}_neg", "eval_id": -1, "src_table_id": sid,
            "label": 0, "strategy": strat, "split": split,
            "text": serialize(og, cap),
        })
    return recs


def split_selfcheck(all_recs):
    """Print zero-intersection of src_table_id across the three splits + pos/neg counts; raise on anomaly."""
    by_split = {"train": set(), "val": set(), "test": set()}
    counts = {s: Counter() for s in by_split}
    for r in all_recs:
        by_split[r["split"]].add(r["src_table_id"])
        counts[r["split"]][r["label"]] += 1

    print("\n===== split self-check =====")
    for s in ("train", "val", "test"):
        print(f"  {s:5s}: src_table_id={len(by_split[s])}  "
              f"pos={counts[s][1]} neg={counts[s][0]} total={sum(counts[s].values())}")
    inter_tv = by_split["train"] & by_split["val"]
    inter_tt = by_split["train"] & by_split["test"]
    inter_vt = by_split["val"] & by_split["test"]
    print(f"  intersection: train&val={len(inter_tv)}  train&test={len(inter_tt)}  "
          f"val&test={len(inter_vt)}")
    if inter_tv or inter_tt or inter_vt:
        raise SystemExit("[FATAL] src_table_id leaks across splits!")
    print("  zero-intersection confirmed OK")
    return {s: dict(src=len(by_split[s]), pos=counts[s][1], neg=counts[s][0]) for s in by_split}


# ──────────────────────────── extract hidden states ────────────────────────────
def extract():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    os.makedirs(FEATDIR, exist_ok=True)
    test_recs = load_test_records()
    test_src_ids = {r["src_table_id"] for r in test_recs}
    train_recs = build_train_records(test_src_ids)
    all_recs = train_recs + test_recs
    split_stats = split_selfcheck(all_recs)

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, output_hidden_states=True).cuda().eval()
    H = model.config.hidden_size
    nL = model.config.num_hidden_layers
    print(f"[LOAD] hidden={H} layers={nL}", flush=True)

    N = len(all_recs)
    feats_mp = {L: np.zeros((N, H), dtype=np.float16) for L in LAYERS}
    feats_last = {L: np.zeros((N, H), dtype=np.float16) for L in LAYERS}
    meta = {k: [] for k in ("sid", "eval_id", "src_table_id", "label", "strategy", "split")}

    t0 = time.time()
    for s in range(0, N, BATCH):
        chunk = all_recs[s:s + BATCH]
        texts = [r["text"] for r in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=MAXTOK).to("cuda")
        with torch.no_grad():
            out = model(**enc)
        hs = out.hidden_states  # tuple(len=nL+1), each (B,T,H)
        mask = enc["attention_mask"].unsqueeze(-1).to(torch.bfloat16)
        denom = mask.sum(1).clamp(min=1)
        lastpos = enc["attention_mask"].sum(1) - 1
        idx = torch.arange(len(chunk), device=enc["attention_mask"].device)
        for L in LAYERS:
            mp = (hs[L] * mask).sum(1) / denom
            feats_mp[L][s:s + len(chunk)] = mp.to(torch.float16).cpu().numpy()
            lt = hs[L][idx, lastpos]
            feats_last[L][s:s + len(chunk)] = lt.to(torch.float16).cpu().numpy()
        for r in chunk:
            for k in meta:
                meta[k].append(r[k])
        if s % (BATCH * 25) == 0:
            el = time.time() - t0
            print(f"  {s+len(chunk)}/{N}  {(s+len(chunk))/max(1e-6,el):.0f} tables/s", flush=True)

    save = {}
    for L in LAYERS:
        save[f"mp{L}"] = feats_mp[L]
        save[f"last{L}"] = feats_last[L]
    for k, v in meta.items():
        save[k] = np.array(v)
    np.savez(FEATNPZ, **save)
    # save split stats for the train stage to reference
    json.dump(split_stats, open(FEATDIR + "/split_stats.json", "w"))


# ──────────────────────────── probe training / evaluation ────────────────────────────
def _metrics_from_preds(y, pred):
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


def _choose_threshold(y_val, score_val, max_fpr=0.2):
    """Select threshold on val: maximize F1 within FPR<=max_fpr; fall back to max F1 if no feasible point."""
    from sklearn.metrics import roc_curve
    order = np.argsort(-score_val)
    cands = np.unique(score_val[order])
    best = None
    best_relaxed = None
    for thr in cands:
        pred = (score_val >= thr).astype(int)
        m = _metrics_from_preds(y_val, pred)
        if best_relaxed is None or m["F1"] > best_relaxed[1]["F1"]:
            best_relaxed = (thr, m)
        if m["FPR"] <= max_fpr:
            if best is None or m["F1"] > best[1]["F1"]:
                best = (thr, m)
    if best is not None:
        return float(best[0]), best[1], True
    return float(best_relaxed[0]), best_relaxed[1], False


def train():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    try:
        from xgboost import XGBClassifier
        HAVE_XGB = True
    except Exception:
        HAVE_XGB = False

    d = np.load(FEATNPZ, allow_pickle=True)
    split = d["split"].astype(str)
    y = d["label"].astype(int)
    strategy = d["strategy"].astype(str)
    eval_id = d["eval_id"].astype(int)
    src = d["src_table_id"].astype(int)

    tr = split == "train"
    va = split == "val"
    te = split == "test"

    # self-check: re-confirm zero-intersection from npz
    s_tr, s_va, s_te = set(src[tr]), set(src[va]), set(src[te])
    assert not (s_tr & s_va) and not (s_tr & s_te) and not (s_va & s_te), "LEAK in npz!"
    print("\n===== npz split confirmation =====")
    print(f"  train: n={tr.sum()} src={len(s_tr)} pos={int(y[tr].sum())} neg={int((1-y[tr]).sum())}")
    print(f"  val  : n={va.sum()} src={len(s_va)} pos={int(y[va].sum())} neg={int((1-y[va]).sum())}")
    print(f"  test : n={te.sum()} src={len(s_te)} pos={int(y[te].sum())} neg={int((1-y[te]).sum())}")
    print(f"  zero-intersection: train&val={len(s_tr&s_va)} train&test={len(s_tr&s_te)} val&test={len(s_va&s_te)}")

    feat_keys = [f"{p}{L}" for L in LAYERS for p in ("mp", "last")]

    # ── train LR per (layer, pooling), select by VAL AUROC ──
    val_auroc = {}
    fitted = {}
    for fk in feat_keys:
        X = d[fk].astype(np.float32)
        scaler = StandardScaler().fit(X[tr])
        Xtr, Xva = scaler.transform(X[tr]), scaler.transform(X[va])
        clf = LogisticRegression(C=1.0, class_weight="balanced",
                                 max_iter=2000, penalty="l2")
        clf.fit(Xtr, y[tr])
        sva = clf.predict_proba(Xva)[:, 1]
        auc = roc_auc_score(y[va], sva)
        val_auroc[fk] = round(float(auc), 4)
        fitted[fk] = (scaler, clf)
        print(f"  [LR] feat={fk:8s} val_AUROC={auc:.4f}")

    best_fk = max(val_auroc, key=val_auroc.get)
    print(f"\n[SELECT] best LR feature = {best_fk}  val_AUROC={val_auroc[best_fk]}")

    scaler, clf = fitted[best_fk]
    Xb = d[best_fk].astype(np.float32)
    sva = clf.predict_proba(scaler.transform(Xb[va]))[:, 1]
    ste = clf.predict_proba(scaler.transform(Xb[te]))[:, 1]

    thr, val_op, feasible = _choose_threshold(y[va], sva, max_fpr=0.2)
    print(f"[THR] val-selected thr={thr:.4f} (FPR<=0.2 feasible={feasible})  "
          f"val op metrics={val_op}")

    test_auroc = round(float(roc_auc_score(y[te], ste)), 4)
    pred_op = (ste >= thr).astype(int)
    m_op = _metrics_from_preds(y[te], pred_op)
    pred_05 = (ste >= 0.5).astype(int)
    m_05 = _metrics_from_preds(y[te], pred_05)

    print(f"\n===== TEST (LR, feat={best_fk}) =====")
    print(f"  AUROC={test_auroc}")
    print(f"  @val-thr({thr:.3f}): {m_op}")
    print(f"  @0.5      : {m_05}")

    # per-strategy recall (on positives, computed over the test mask)
    full_pred_op = np.zeros(len(y), int)
    full_pred_op[te] = pred_op
    per_strategy = {}
    for s in sorted(set(strategy[te])):
        mask_pos = te & (strategy == s) & (y == 1)
        n = int(mask_pos.sum())
        if n == 0:
            continue
        caught = int(np.sum(full_pred_op[mask_pos] == 1))
        per_strategy[s] = dict(n=n, recall=round(caught / n, 4))
    print("  per-strategy recall @val-thr:")
    for s, v in per_strategy.items():
        print(f"    {s:35s} recall={v['recall']} (n={v['n']})")

    degenerate = len(set(pred_op.tolist())) <= 1

    # ── GBDT / XGB control (same best feature) ──
    xgb_block = None
    if HAVE_XGB:
        n_pos, n_neg = int(y[tr].sum()), int((1 - y[tr]).sum())
        spw = n_neg / max(1, n_pos)
        xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8,
                            scale_pos_weight=spw, eval_metric="logloss",
                            tree_method="hist", n_jobs=8)
        xgb.fit(Xb[tr], y[tr])
        sva_x = xgb.predict_proba(Xb[va])[:, 1]
        ste_x = xgb.predict_proba(Xb[te])[:, 1]
        thr_x, val_op_x, feas_x = _choose_threshold(y[va], sva_x, max_fpr=0.2)
        m_op_x = _metrics_from_preds(y[te], (ste_x >= thr_x).astype(int))
        auc_x = round(float(roc_auc_score(y[te], ste_x)), 4)
        xgb_block = dict(feature=best_fk, val_AUROC=round(float(roc_auc_score(y[va], sva_x)), 4),
                         test_AUROC=auc_x, thr=round(thr_x, 4),
                         test_op=m_op_x, val_op=val_op_x)
        print(f"\n[XGB] feat={best_fk} test_AUROC={auc_x} @thr({thr_x:.3f}) {m_op_x}")

    # ── write verdicts (LR @val-thr) ──
    # pred_op / ste / eval_id[te] are all the test subset in te-mask order (length 600), aligned.
    test_eval_ids = eval_id[te]
    verdicts = {}
    for j in range(len(test_eval_ids)):
        verdicts[str(int(test_eval_ids[j]))] = "FABRICATED" if pred_op[j] == 1 else "PLAUSIBLE"
    json.dump(verdicts, open(OUT_VERDICTS, "w"))

    # ── baseline comparison (known values) ──
    baseline_cmp = {
        "note": "existing training-free baselines are all ~random on these 600",
        "rule_control": {"recall": 0.0},
        "gemma": {"recall": 0.08},
        "haiku": {"recall": 0.16},
        "ours_margin_judge_bestF1": 0.36,
        "typical_Acc_range": [0.50, 0.52],
    }

    split_stats = {}
    try:
        split_stats = json.load(open(FEATDIR + "/split_stats.json"))
    except Exception:
        pass

    result = {
        "method": "frozen Qwen2.5-7B-Instruct hidden-state probe (L2 logistic regression)",
        "model": MODEL,
        "layers": LAYERS,
        "poolings": ["mean-pool", "last-token"],
        "splits": {
            "train": dict(n=int(tr.sum()), src=len(s_tr),
                          pos=int(y[tr].sum()), neg=int((1 - y[tr]).sum())),
            "val": dict(n=int(va.sum()), src=len(s_va),
                        pos=int(y[va].sum()), neg=int((1 - y[va]).sum())),
            "test": dict(n=int(te.sum()), src=len(s_te),
                         pos=int(y[te].sum()), neg=int((1 - y[te]).sum())),
            "zero_intersection": dict(train_val=len(s_tr & s_va),
                                      train_test=len(s_tr & s_te),
                                      val_test=len(s_va & s_te)),
        },
        "val_AUROC_by_feature": val_auroc,
        "best_feature": best_fk,
        "operating_threshold": round(thr, 4),
        "threshold_feasible_fpr_le_0.2": feasible,
        "val_op_metrics": val_op,
        "test": {
            "AUROC": test_auroc,
            "at_val_threshold": m_op,
            "at_0.5": m_05,
            "per_strategy_recall": per_strategy,
            "degenerate_all_one_class": degenerate,
        },
        "xgb_control": xgb_block,
        "vs_baseline": baseline_cmp,
        "verdicts_path": OUT_VERDICTS,
        "feature_npz": FEATNPZ,
    }
    json.dump(result, open(OUT_JSON, "w"), ensure_ascii=False, indent=2)
    print(f"\n[CONCLUSION] best={best_fk} TEST AUROC={test_auroc} "
          f"F1={m_op['F1']} (vs baseline best F1 0.36) degenerate={degenerate}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["extract", "train", "all"], default="all")
    a = ap.parse_args()
    if a.stage in ("extract", "all"):
        extract()
    if a.stage in ("train", "all"):
        train()


if __name__ == "__main__":
    main()
