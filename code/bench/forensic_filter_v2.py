"""
forensic_filter_v2.py — re-label forensic_usable / forensic_tags

Criterion changed from a regex-density threshold to "whether a strictly
falsifiable fabrication can be injected":
  forensic_usable=1  iff the table supports >=1 strict family (GRIM/DIST/CI/PCT/PVAL)
  forensic_tags      = comma-separated supported families (excludes SURF; SURF is not a positive sample)
Recompute over all pmc + arxiv tables. Deterministic seed.
"""
import sqlite3, sys, random, time
sys.path.insert(0, "/home/pengminjie/Backup/paper/ICDE26/code/bench")
from parser import StructuredTable
import corruptors as C, build_bench as B

DB = "/home/pengminjie/Backup/paper/ICDE26/data/arxiv_data.db"
SEED = 42
RIG = ["GRIM", "DIST", "CI", "PCT", "PVAL"]   # SURF excluded


def rng(rid, fam):
    return random.Random(hash((rid, fam, SEED)) & 0xFFFFFFFF)


def supported(st, rid, cap, tags):
    out = []
    for fam in RIG:
        g, s = C.attempt(st, fam, rng(rid, fam))
        if g is not None and s and B.verify_fake(fam, st.to_grid(), g, s, cap or "", tags or "")[0]:
            out.append(fam)
    return out


def main():
    conn = sqlite3.connect(DB)
    # Ensure forensic_usable / forensic_tags columns exist
    cols = [r[1] for r in conn.execute("pragma table_info(paper_tables)")]
    if "forensic_usable" not in cols:
        conn.execute("ALTER TABLE paper_tables ADD COLUMN forensic_usable INTEGER DEFAULT 0")
    if "forensic_tags" not in cols:
        conn.execute("ALTER TABLE paper_tables ADD COLUMN forensic_tags TEXT")
    conn.commit()
    # Reset all to 0 before recompute
    conn.execute("UPDATE paper_tables SET forensic_usable=0, forensic_tags=NULL")
    conn.commit()

    rows = conn.execute("""SELECT pt.id, pt.arxiv_id, pt.caption, pt.table_json, pt.source
                           FROM paper_tables pt""").fetchall()
    total = len(rows)
    from collections import Counter
    by_src = Counter(); fam_by_src = {}
    upd = []
    t0 = time.time()
    done = 0
    for rid, aid, cap, tj, src in rows:
        st = StructuredTable.from_json(tj, caption=cap, src_table_id=rid, arxiv_id=aid)
        sup = supported(st, rid, cap, "") if st is not None else []
        if sup:
            upd.append((1, ",".join(sup), rid))
            by_src[src] += 1
            fam_by_src.setdefault(src, Counter())
            for f in sup:
                fam_by_src[src][f] += 1
        done += 1
    conn.executemany("UPDATE paper_tables SET forensic_usable=?, forensic_tags=? WHERE id=?", upd)
    conn.commit()
    for src in ("pmc", "arxiv"):
        print(f"  {src}: usable={by_src[src]}  families {dict(fam_by_src.get(src, {}))}", flush=True)
    tot = conn.execute("SELECT COUNT(*) FROM paper_tables WHERE forensic_usable=1").fetchone()[0]
    print(f"  total usable: {tot}", flush=True)
    conn.close()


if __name__ == "__main__":
    main()
