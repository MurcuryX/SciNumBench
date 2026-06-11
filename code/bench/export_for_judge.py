"""
export_for_judge.py — export LLM-judge candidate subsets to results/

Produces two jsonl files:
  judge_test.jsonl      : all 2106 rows of the test split.
  judge_dist_cand.jsonl : DIST candidate subset (rule-based, identical to
                          baseline_gbdt.py / cot_detect.py / M6: within the test
                          split, any meansd cell satisfies mean>0 & sd>0 &
                          mean-2SD<0). Expected n_cand=538, dist_fake=258.

Each line: {bench_id, label, family, split, source, text}
  text = clean_serialize.serialize(corrupted_grid, caption, provenance)  (cleaned + serialized table)
judge_dist_cand has an extra field: is_dist_fake (1 if corruption_family=="DIST" else 0).

After export, prints line counts of both files + label/family distribution.
"""
import os
import sys
import json
import sqlite3

ROOT = "/home/pengminjie/Backup/paper/ICDE26"
sys.path.insert(0, ROOT + "/code/bench")
from parser import StructuredTable          # noqa: E402
from clean_serialize import serialize       # noqa: E402

DB = ROOT + "/data/arxiv_data.db"
RESULTS = ROOT + "/results"
OUT_TEST = RESULTS + "/judge_test.jsonl"
OUT_DIST = RESULTS + "/judge_dist_cand.jsonl"


def is_dist_candidate(st):
    """DIST candidate definition (identical to baseline_gbdt.dist_candidate_ids /
    cot_detect / extract_14b_dist): any meansd cell with sd>0 and mean>0 and mean-2SD<0."""
    for (i, j) in st.positions_of("meansd"):
        c = st.cells[i][j]
        if c["sd"] > 0 and c["mean"] > 0 and c["mean"] - 2 * c["sd"] < 0:
            return True
    return False


def fetch_rows(split):
    conn = sqlite3.connect(DB)
    rows = conn.execute(
        """SELECT sb.bench_id, sb.label, sb.corruption_family, sb.dataset_split,
                  sb.source, sb.corrupted_grid, sb.provenance, pt.caption
           FROM scinum_bench sb LEFT JOIN paper_tables pt ON sb.src_table_id=pt.id
           WHERE sb.dataset_split=? ORDER BY sb.bench_id""", (split,)).fetchall()
    conn.close()
    return rows


def build_record(bid, label, fam, split, source, cg, prov, cap):
    g = json.loads(cg)
    provenance = json.loads(prov) if prov else None
    text = serialize(g, cap or "", provenance=provenance)
    return {
        "bench_id": int(bid),
        "label": int(label),
        "family": fam or "clean",
        "split": split,
        "source": source,
        "text": text,
    }


def distribution(records, dist_key=None):
    from collections import Counter
    lab = Counter(r["label"] for r in records)
    fam = Counter(r["family"] for r in records)
    out = f"    n={len(records)} | label: " + dict(lab).__repr__() + " | family: " + dict(fam).__repr__()
    if dist_key:
        dk = Counter(r[dist_key] for r in records)
        out += f" | {dist_key}: " + dict(dk).__repr__()
    return out


def main():
    os.makedirs(RESULTS, exist_ok=True)
    test_rows = fetch_rows("test")

    # --- judge_test.jsonl: all test rows ---
    test_recs = []
    for (bid, label, fam, split, source, cg, prov, cap) in test_rows:
        test_recs.append(build_record(bid, label, fam, split, source, cg, prov, cap))
    with open(OUT_TEST, "w") as f:
        for r in test_recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # --- judge_dist_cand.jsonl: DIST candidate subset ---
    dist_recs = []
    for (bid, label, fam, split, source, cg, prov, cap) in test_rows:
        g = json.loads(cg)
        if len(g) < 2:
            continue
        st = StructuredTable(g[0], g[1:], caption=cap or "")
        if not is_dist_candidate(st):
            continue
        rec = build_record(bid, label, fam, split, source, cg, prov, cap)
        rec["is_dist_fake"] = 1 if (fam or "") == "DIST" else 0
        dist_recs.append(rec)
    with open(OUT_DIST, "w") as f:
        for r in dist_recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[SAVE] {OUT_TEST}")
    print(distribution(test_recs))
    print(f"[SAVE] {OUT_DIST}")
    print(distribution(dist_recs, dist_key="is_dist_fake"))

    n_cand = len(dist_recs)
    n_dist_fake = sum(r["is_dist_fake"] for r in dist_recs)
    print(f"\n[CHECK] DIST candidates n_cand={n_cand} (expected 538)  dist_fake={n_dist_fake} (expected 258)  "
          f"{'OK' if (n_cand == 538 and n_dist_fake == 258) else '!!! mismatch'}")


if __name__ == "__main__":
    main()
