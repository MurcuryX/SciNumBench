"""
score_verdicts.py — unified verdict scoring functions (shared by Haiku / Gemma / any LLM-judge)

Conventions match evaluate.py / eval_prompted.py / baseline_gbdt.py exactly:
  verdict "FABRICATED" -> pred=1 (fake) ; "PLAUSIBLE" -> pred=0 (clean).
  Table-level: P / R / F1 / Acc / FPR.
  per-family recall (on test split, by corruption_family).
  cross_test FPR (cross_test split is all clean, FP/(FP+TN)).
  DIST candidate candidate_P (on the DIST candidate subset, treat FABRICATED as the
                        prediction for DIST-fake and compute precision), and report clean_cand_FPR.

Main interface:
    score(verdicts_path, subset="test") -> dict
  verdicts_path : json of {bench_id(str|int): "FABRICATED"|"PLAUSIBLE"} produced by judge_gemma.py / Haiku.
  subset        : "test"        -> main-set metrics (P/R/F1/Acc/FPR + per-family recall),
                                   and automatically appends cross_test FPR (if verdicts cover cross_test).
                  "dist_cand"   -> DIST candidate-layer metrics (candidate_P/R + clean_cand_FPR).
                  "cross_test"  -> cross-domain FPR only.
  Returns a dict (subset, counts, metrics); also prints a human-readable summary.

verdicts only need to cover the target subset's bench_ids; missing bench_ids are treated as unjudged -> default PLAUSIBLE (pred=0),
and n_missing is reported to help detect jsonl/verdict misalignment.
"""
import sys
import json
import sqlite3
from collections import Counter

ROOT = "/home/pengminjie/Backup/paper/ICDE26"
DB = ROOT + "/data/arxiv_data.db"
sys.path.insert(0, ROOT + "/code/bench")
from parser import StructuredTable  # noqa: E402

FAMILIES = ["GRIM", "PVAL", "CI", "PCT", "DIST"]


# ───────────────────────── metrics (same conventions as baseline_gbdt.metrics) ─────────────────────────
def _metrics(pairs):
    """pairs: list of (y, pred). Returns P/R/F1/Acc/FPR + counts."""
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
    raw = json.load(open(path))
    # normalize key to int, value to pred (1=FABRICATED)
    vd = {}
    for k, v in raw.items():
        vu = str(v).strip().upper()
        vd[int(k)] = 1 if vu == "FABRICATED" else 0
    return vd


def _is_dist_candidate(st):
    """DIST candidate definition, consistent with export_for_judge / baseline_gbdt / cot_detect."""
    for (i, j) in st.positions_of("meansd"):
        c = st.cells[i][j]
        if c["sd"] > 0 and c["mean"] > 0 and c["mean"] - 2 * c["sd"] < 0:
            return True
    return False


def _fetch(split):
    conn = sqlite3.connect(DB)
    rows = conn.execute(
        """SELECT sb.bench_id, sb.label, sb.corruption_family, sb.dataset_split,
                  sb.source, sb.corrupted_grid, pt.caption
           FROM scinum_bench sb LEFT JOIN paper_tables pt ON sb.src_table_id=pt.id
           WHERE sb.dataset_split=?""", (split,)).fetchall()
    conn.close()
    return rows


def score(verdicts_path, subset="test"):
    vd = _load_verdicts(verdicts_path)

    def pred_for(bid):
        # missing verdict -> default PLAUSIBLE (pred=0)
        return vd.get(int(bid), 0)

    result = {"verdicts_path": verdicts_path, "subset": subset,
              "n_verdicts": len(vd)}

    if subset == "test":
        rows = _fetch("test")
        pairs = []
        fam_total = Counter(); fam_caught = Counter()
        missing = 0
        for bid, label, fam, split, source, cg, cap in rows:
            if int(bid) not in vd:
                missing += 1
            p = pred_for(bid)
            y = int(label)
            pairs.append((y, p))
            if y == 1:
                fam_total[fam] += 1
                if p == 1:
                    fam_caught[fam] += 1
        m = _metrics(pairs)
        result.update(m)
        result["n_missing"] = missing
        result["per_family_recall"] = {
            f: dict(n=fam_total[f],
                    recall=(round(fam_caught[f] / fam_total[f], 4) if fam_total[f] else None))
            for f in FAMILIES}
        # append cross_test FPR (if covered)
        cross = _fetch("cross_test")
        if cross and any(int(b) in vd for b, *_ in cross):
            cpairs = [(int(lab), pred_for(b)) for b, lab, *_ in cross]
            cm = _metrics(cpairs)
            result["cross_test_FPR"] = cm["FPR"]
            result["cross_test_n"] = len(cpairs)

        print(f"\n===== score [{subset}] {verdicts_path} =====")
        print(f"  n={len(pairs)} verdicts={len(vd)} missing(->PLAUSIBLE)={missing}")
        print(f"  P={m['P']} R={m['R']} F1={m['F1']} Acc={m['Acc']} FPR={m['FPR']} "
              f"(TP={m['TP']} FP={m['FP']} FN={m['FN']} TN={m['TN']})")
        print("  per-family recall:")
        for f in FAMILIES:
            d = result["per_family_recall"][f]
            print(f"    {f:5s} {d['recall']} (n={d['n']})")
        if "cross_test_FPR" in result:
            print(f"  cross_test FPR={result['cross_test_FPR']} (n={result['cross_test_n']})")

    elif subset == "cross_test":
        rows = _fetch("cross_test")
        missing = sum(1 for b, *_ in rows if int(b) not in vd)
        pairs = [(int(lab), pred_for(b)) for b, lab, *_ in rows]
        m = _metrics(pairs)
        result.update(m); result["n_missing"] = missing
        result["cross_test_FPR"] = m["FPR"]
        print(f"\n===== score [{subset}] {verdicts_path} =====")
        print(f"  n={len(pairs)} missing={missing}  cross_test FPR={m['FPR']} "
              f"(FP={m['FP']} TN={m['TN']})")

    elif subset == "dist_cand":
        rows = _fetch("test")
        cand = []  # (is_dist_fake, clean, pred)
        missing = 0
        for bid, label, fam, split, source, cg, cap in rows:
            g = json.loads(cg)
            if len(g) < 2:
                continue
            st = StructuredTable(g[0], g[1:], caption=cap or "")
            if not _is_dist_candidate(st):
                continue
            if int(bid) not in vd:
                missing += 1
            p = pred_for(bid)
            is_dist = 1 if (fam or "") == "DIST" else 0
            clean = 1 if int(label) == 0 else 0
            cand.append((is_dist, clean, p))
        # candidate precision: judgment of DIST-fake (FABRICATED=pred1 vs is_dist_fake)
        pairs = [(is_dist, p) for is_dist, clean, p in cand]
        m = _metrics(pairs)
        clean_ids = [(clean, p) for is_dist, clean, p in cand if clean == 1]
        clean_fpr = (sum(p for clean, p in clean_ids) / len(clean_ids)) if clean_ids else None
        result.update(dict(
            n_cand=len(cand),
            n_dist_fake=sum(is_dist for is_dist, clean, p in cand),
            n_clean_cand=len(clean_ids),
            candidate_P=m["P"], candidate_R=m["R"],
            clean_cand_FPR=(round(clean_fpr, 4) if clean_fpr is not None else None),
            n_missing=missing))
        print(f"\n===== score [{subset}] {verdicts_path} =====")
        print(f"  n_cand={result['n_cand']} dist_fake={result['n_dist_fake']} "
              f"clean_cand={result['n_clean_cand']} missing={missing}")
        print(f"  candidate_P={result['candidate_P']} candidate_R={result['candidate_R']} "
              f"clean_cand_FPR={result['clean_cand_FPR']}  (reference: M6 P=0.92 / probe P=0.57)")
    else:
        raise ValueError(f"unknown subset: {subset}")

    return result


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("verdicts_path")
    ap.add_argument("--subset", default="test", choices=["test", "cross_test", "dist_cand"])
    a = ap.parse_args()
    out = score(a.verdicts_path, a.subset)
    print("\n[JSON]", json.dumps(out, ensure_ascii=False))
