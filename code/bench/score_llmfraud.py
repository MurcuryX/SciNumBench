"""
score_llmfraud.py — scorer for the llm_fraud (boost-ours) evaluation set.

Reads verdicts (LLM-judge / any detector output) and compares against the label
in results/llmfraud_eval.jsonl.
Convention aligned with score_verdicts.py: FABRICATED -> pred=1, PLAUSIBLE -> pred=0;
also accepts 0/1.

Main interface:
    score(verdicts_path, eval_path=<default results/llmfraud_eval.jsonl>) -> dict
  verdicts : {eval_id(str|int): "FABRICATED"|"PLAUSIBLE"} or {eval_id: 0/1}.
  Missing eval_id is treated as unjudged -> defaults to PLAUSIBLE (pred=0), counted in n_missing.

Metrics: P / R / F1 / Acc / FPR, confusion matrix (TP/FP/FN/TN), per-strategy recall,
n_missing, degenerate flag (all predictions in one class).

CLI: python score_llmfraud.py <verdicts.json> [--eval results/llmfraud_eval.jsonl]
"""
import sys
import json
from collections import Counter

ROOT = "/home/pengminjie/Backup/paper/ICDE26"
DEFAULT_EVAL = ROOT + "/results/llmfraud_eval.jsonl"


def _metrics(pairs):
    """pairs: list of (y, pred). Returns P/R/F1/Acc/FPR + confusion matrix counts."""
    TP = sum(1 for y, p in pairs if p == 1 and y == 1)
    FP = sum(1 for y, p in pairs if p == 1 and y == 0)
    FN = sum(1 for y, p in pairs if p == 0 and y == 1)
    TN = sum(1 for y, p in pairs if p == 0 and y == 0)
    P = TP / (TP + FP) if TP + FP else 0.0
    R = TP / (TP + FN) if TP + FN else 0.0
    F1 = 2 * P * R / (P + R) if P + R else 0.0
    Acc = (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) else 0.0
    FPR = FP / (FP + TN) if (FP + TN) else 0.0
    return dict(TP=TP, FP=FP, FN=FN, TN=TN,
                P=round(P, 4), R=round(R, 4), F1=round(F1, 4),
                Acc=round(Acc, 4), FPR=round(FPR, 4))


def _load_verdicts(path):
    """Normalize verdicts -> {int eval_id: pred(1=FABRICATED)}. Accepts string or 0/1."""
    raw = json.load(open(path))
    vd = {}
    for k, v in raw.items():
        if isinstance(v, str):
            pred = 1 if v.strip().upper() == "FABRICATED" else 0
        else:
            pred = 1 if int(v) == 1 else 0
        vd[int(k)] = pred
    return vd


def _load_eval(path):
    """Read llmfraud_eval.jsonl -> list of {eval_id, label, strategy}."""
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            recs.append({"eval_id": int(r["eval_id"]),
                         "label": int(r["label"]),
                         "strategy": r.get("strategy")})
    return recs


def score(verdicts_path, eval_path=DEFAULT_EVAL):
    vd = _load_verdicts(verdicts_path)
    gold = _load_eval(eval_path)

    pairs = []
    preds = []
    missing = 0
    strat_total = Counter()   # recall denominator per strategy, positives only (label=1)
    strat_caught = Counter()
    for g in gold:
        eid = g["eval_id"]
        y = g["label"]
        if eid not in vd:
            missing += 1
        p = vd.get(eid, 0)   # missing -> PLAUSIBLE
        pairs.append((y, p))
        preds.append(p)
        if y == 1:
            strat_total[g["strategy"]] += 1
            if p == 1:
                strat_caught[g["strategy"]] += 1

    m = _metrics(pairs)
    pred_counts = Counter(preds)
    degenerate = len([c for c in pred_counts.values() if c > 0]) <= 1

    per_strategy_recall = {
        s: dict(n=strat_total[s],
                recall=round(strat_caught[s] / strat_total[s], 4) if strat_total[s] else None)
        for s in sorted(strat_total)
    }

    result = {
        "verdicts_path": verdicts_path,
        "eval_path": eval_path,
        "n": len(pairs),
        "n_verdicts": len(vd),
        "n_missing": missing,
        **m,
        "confusion_matrix": {"TP": m["TP"], "FP": m["FP"], "FN": m["FN"], "TN": m["TN"]},
        "pred_dist": {"pred1_FABRICATED": pred_counts.get(1, 0),
                      "pred0_PLAUSIBLE": pred_counts.get(0, 0)},
        "degenerate_all_one_class": degenerate,
        "per_strategy_recall": per_strategy_recall,
    }

    print(f"\n===== score_llmfraud {verdicts_path} =====")
    print(f"  n={result['n']} verdicts={result['n_verdicts']} missing(->PLAUSIBLE)={missing}")
    print(f"  P={m['P']} R={m['R']} F1={m['F1']} Acc={m['Acc']} FPR={m['FPR']}")
    print(f"  confusion: TP={m['TP']} FP={m['FP']} FN={m['FN']} TN={m['TN']}")
    print(f"  pred_dist: FABRICATED={pred_counts.get(1,0)} PLAUSIBLE={pred_counts.get(0,0)}"
          f"  {'  [DEGENERATE: all one class]' if degenerate else ''}")
    print("  per-strategy recall (on positives):")
    for s, d in per_strategy_recall.items():
        print(f"    {s:35s} recall={d['recall']} (n={d['n']})")

    return result


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("verdicts_path")
    ap.add_argument("--eval", dest="eval_path", default=DEFAULT_EVAL)
    a = ap.parse_args()
    out = score(a.verdicts_path, a.eval_path)
    print("\n[JSON]", json.dumps(out, ensure_ascii=False))
