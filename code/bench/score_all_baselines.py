"""
score_all_baselines.py — compute probe-comparable metrics for every Step-4 baseline
on the 2000-row NEW test split, keyed by example_id, labels from mapping.jsonl.

Metrics (same as probe): AUROC, AUPRC (where a continuous score exists),
plus at the baseline's native binary decision: F1/accuracy/precision/recall + TPR/FPR.
For score-bearing baselines we ALSO report F1@0.5-on-score is N/A; the binary verdict
is the paper's decision rule, so binary metrics use 'pred'. AUROC/AUPRC use 'score'.

Copies in the probe rows (14B, 7B) from probe_metrics.json for side-by-side.
Writes results/baseline_metrics.json.
"""
import json
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = "/home/pengminjie/Backup/paper/ICDE26"
RESULTS = ROOT + "/results"
SPLITS = ROOT + "/data/splits"


def load_labels():
    lab = {}
    for line in open(f"{SPLITS}/mapping.jsonl"):
        r = json.loads(line)
        if r.get("split") and r["split"] != "test":
            continue
        lab[r["example_id"]] = int(r["label"])
    return lab


def binary_metrics(y, pred):
    TP = int(((pred == 1) & (y == 1)).sum())
    FP = int(((pred == 1) & (y == 0)).sum())
    FN = int(((pred == 0) & (y == 1)).sum())
    TN = int(((pred == 0) & (y == 0)).sum())
    prec = TP / (TP + FP) if TP + FP else 0.0
    rec = TP / (TP + FN) if TP + FN else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = (TP + TN) / len(y)
    tpr = rec
    fpr = FP / (FP + TN) if FP + TN else 0.0
    return dict(TP=TP, FP=FP, FN=FN, TN=TN,
                precision=round(prec, 4), recall=round(rec, 4),
                F1=round(f1, 4), accuracy=round(acc, 4),
                TPR=round(tpr, 4), FPR=round(fpr, 4))


def score_baseline(name, preds_file, lab, has_score=True):
    preds = json.load(open(preds_file))
    eids = list(lab.keys())
    y = np.array([lab[e] for e in eids])
    p = np.array([preds[e]["pred"] for e in eids])
    row = {"baseline": name, "preds_file": preds_file, "n": len(y)}
    row.update(binary_metrics(y, p))
    if has_score:
        s = np.array([preds[e]["score"] for e in eids], dtype=float)
        # AUROC undefined if score constant
        if len(np.unique(s)) > 1:
            row["AUROC"] = round(float(roc_auc_score(y, s)), 4)
            row["AUPRC"] = round(float(average_precision_score(y, s)), 4)
        else:
            row["AUROC"] = None
            row["AUPRC"] = None
    else:
        row["AUROC"] = None
        row["AUPRC"] = None
    row["pred_dist"] = {"pred1": int(p.sum()), "pred0": int((p == 0).sum())}
    return row


def main():
    lab = load_labels()
    print(f"[LABELS] {len(lab)} examples  pos={sum(lab.values())} neg={len(lab)-sum(lab.values())}")

    baselines = [
        ("rules", f"{RESULTS}/baseline_preds_rules.json", True),
        ("benford", f"{RESULTS}/baseline_preds_benford.json", True),
        ("terminal_digit", f"{RESULTS}/baseline_preds_terminal_digit.json", True),
        ("rounding", f"{RESULTS}/baseline_preds_rounding.json", True),
        ("ours_dominance_allwin", f"{RESULTS}/baseline_preds_ours_dominance_allwin.json", True),
        ("ours_dominance_margin", f"{RESULTS}/baseline_preds_ours_dominance_margin.json", True),
        ("gemma3_12b_zeroshot", f"{RESULTS}/baseline_preds_gemma3_zeroshot.json", True),
    ]
    rows = []
    import os
    for name, f, hs in baselines:
        if not os.path.exists(f):
            print(f"[SKIP] {name}: {f} not found yet")
            continue
        r = score_baseline(name, f, lab, hs)
        rows.append(r)
        print(f"[{name}] AUROC={r['AUROC']} AUPRC={r['AUPRC']} F1={r['F1']} "
              f"acc={r['accuracy']} P={r['precision']} R={r['recall']} "
              f"TPR={r['TPR']} FPR={r['FPR']} pred1={r['pred_dist']['pred1']}")

    # copy in probe rows
    pm = json.load(open(f"{RESULTS}/probe_metrics.json"))
    for tag, label in [("qwen14b", "hidden_probe_14B"), ("qwen7b", "hidden_probe_7B")]:
        t = pm[tag]["test"]
        a05 = t["at_0.5"]
        rows.append({
            "baseline": label,
            "preds_file": f"results/probe_preds_{'14b' if tag=='qwen14b' else '7b'}.json",
            "n": pm[tag]["n_test"],
            "AUROC": t["AUROC"], "AUPRC": t["AUPRC"],
            "precision": a05["precision"], "recall": a05["recall"],
            "F1": a05["F1"], "accuracy": a05["accuracy"],
            "TP": a05["TP"], "FP": a05["FP"], "FN": a05["FN"], "TN": a05["TN"],
            "TPR": round(a05["recall"], 4),
            "FPR": round(a05["FP"] / (a05["FP"] + a05["TN"]), 4),
            "threshold": "0.5", "note": "probe @0.5; also has f1_opt_threshold variant",
        })

    rows.append({"baseline": "haiku", "status": "PENDING-server0",
                 "note": "training-free LLM judge; must run via server0 control plane (GFW)"})

    out = {
        "test_set": "data/splits/{test_model,test,mapping}.jsonl (2000 rows, src-disjoint)",
        "label_source": "data/splits/mapping.jsonl (1=fabricated 0=honest, 1000/1000)",
        "metric_notes": (
            "AUROC/AUPRC from continuous 'score'; precision/recall/F1/acc/TPR/FPR at each "
            "baseline's native unsupervised binary decision (p<0.05 / fixed quantile / Yes-No). "
            "Probe rows copied from probe_metrics.json at threshold 0.5."),
        "rows": rows,
    }
    json.dump(out, open(f"{RESULTS}/baseline_metrics.json", "w"), ensure_ascii=False, indent=2)
    print(f"\n[WRITE] {RESULTS}/baseline_metrics.json  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
