"""
add_cross_domain.py — add a CS cross-domain precision track to scinum_bench

Use unused arXiv tables (forensic_usable=0, numeric-dense real tables) as clean hard negatives:
dataset_split='cross_test', label=0, corruption_family='clean', ood_role='cross_neg'.
Papers are disjoint from the main set at the paper level (to prevent leakage). Purpose:
measure the detector's false-positive rate on normal CS tables / cross-domain robustness.
Separate track, not mixed into the main set's balanced metrics.
"""
import sqlite3, sys, json, random
sys.path.insert(0, "/home/pengminjie/Backup/paper/ICDE26/code/bench")
from parser import StructuredTable

DB = "/home/pengminjie/Backup/paper/ICDE26/data/arxiv_data.db"
SEED = 42
TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
NUMERIC_TYPES = {"meansd", "ci", "pval", "pct", "countpct", "count", "num"}


def numeric_frac(st):
    tc = st.type_counts()
    nonempty = sum(v for k, v in tc.items() if k != "empty")
    if nonempty == 0:
        return 0.0
    num = sum(v for k, v in tc.items() if k in NUMERIC_TYPES)
    return num / nonempty


def main():
    conn = sqlite3.connect(DB)
    # Clear old cross_test rows (idempotent re-run)
    conn.execute("DELETE FROM scinum_bench WHERE dataset_split='cross_test'")
    conn.commit()
    used_papers = set(r[0] for r in conn.execute("SELECT DISTINCT arxiv_id FROM scinum_bench"))

    # Candidates: arxiv tables from papers not used in the main set, with some size
    rows = conn.execute("""SELECT id, arxiv_id, caption, table_json
                           FROM paper_tables
                           WHERE source='arxiv' AND IFNULL(forensic_usable,0)=0
                             AND rows>=3 AND cols>=2""").fetchall()

    rng = random.Random(SEED)
    rng.shuffle(rows)
    picked = []
    seen_papers = set()
    for rid, aid, cap, tj in rows:
        if aid in used_papers:
            continue
        st = StructuredTable.from_json(tj, caption=cap, src_table_id=rid, arxiv_id=aid)
        if st is None or st.nrows < 3 or st.ncols < 2:
            continue
        if numeric_frac(st) < 0.30:   # must be a numeric-dense real table (qualified hard negative)
            continue
        picked.append((rid, aid, st, cap or ""))
        seen_papers.add(aid)
        if len(picked) >= TARGET:
            break

    buf = []
    for rid, aid, st, cap in picked:
        grid = st.to_grid()
        meta = json.dumps({"col_n": st.col_n, "table_n": st.table_n,
                           "types": st.type_counts()}, ensure_ascii=False)
        gj = json.dumps(grid, ensure_ascii=False)
        buf.append((aid, rid, "arxiv", "cross_test", 0, "clean", None, "cross_neg",
                    gj, gj, "[]", 0, meta))
    conn.executemany("""INSERT OR IGNORE INTO scinum_bench
        (arxiv_id,src_table_id,source,dataset_split,label,corruption_family,
         forensic_rule,ood_role,original_grid,corrupted_grid,provenance,n_cells_changed,table_meta)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""", buf)
    conn.commit()

    # Report: full-database domain composition
    print("\n=== scinum_bench domain composition after adding cross_test ===")
    tot = conn.execute("SELECT COUNT(*) FROM scinum_bench").fetchone()[0]
    print(f"total samples: {tot}")
    for src, n in conn.execute("SELECT source,COUNT(*) FROM scinum_bench GROUP BY source"):
        print(f"  {src:6s} {n} ({100*n/tot:.0f}%)")
    print("\nsplit x label:")
    for sp, lab, n in conn.execute("SELECT dataset_split,label,COUNT(*) FROM scinum_bench GROUP BY dataset_split,label ORDER BY dataset_split,label"):
        print(f"  {sp:11s} label={lab}  {n}")
    conn.close()


if __name__ == "__main__":
    main()
