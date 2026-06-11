"""
build_bench.py — SciNumBench v2 construction (split + conditional assignment + provenance persistence + self-check)

Data flow: paper_tables(pmc, forensic_usable=1) -> paper-level 60/20/20 stratified split
        -> within-split seeded greedy conditional assignment (1 per table, rare-first, clean~=40%)
        -> tier-aware corruption + cell-level provenance -> mandatory self-check -> write scinum_bench

Usage:
  python build_bench.py            # exits with error if data already exists
  python build_bench.py --rebuild  # wipe and rebuild
Fully non-interactive (tmux safe).
"""

import argparse
import copy
import json
import random
import re
import sqlite3
import sys
from collections import Counter, defaultdict

sys.path.insert(0, "/home/pengminjie/Backup/paper/ICDE26/code/bench")
from parser import StructuredTable
import corruptors as C

DB = "/home/pengminjie/Backup/paper/ICDE26/data/arxiv_data.db"
SEED = 42
SPLIT_RATIO = {"train": 0.60, "val": 0.20, "test": 0.20}
FAKE_FRAC = 0.60
# Only strictly falsifiable families (SURF not used as positives). rare-first.
RIG_FAMILIES = ["GRIM", "PVAL", "DIST", "CI", "PCT"]
RARITY_ORDER = ["GRIM", "PVAL", "DIST", "CI", "PCT"]  # rare first
OOD_FAMILIES = {"PCT", "CI"}          # held-out unseen families
OOD_TRAIN_FAMS = {"clean", "GRIM", "PVAL", "DIST"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS scinum_bench (
    bench_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    arxiv_id          TEXT NOT NULL,
    src_table_id      INTEGER NOT NULL,
    source            TEXT NOT NULL DEFAULT 'pmc',
    dataset_split     TEXT NOT NULL,
    label             INTEGER NOT NULL,           -- 0 clean / 1 fake
    corruption_family TEXT NOT NULL,              -- clean/SURF/GRIM/PVAL/DIST/PCT/CI
    forensic_rule     TEXT,
    ood_role          TEXT NOT NULL DEFAULT 'none',-- train_pool/ood_test/none
    original_grid     TEXT NOT NULL,
    corrupted_grid    TEXT NOT NULL,
    provenance        TEXT NOT NULL,              -- JSON list of cell spans
    n_cells_changed   INTEGER NOT NULL DEFAULT 0,
    table_meta        TEXT,
    UNIQUE(src_table_id, dataset_split, corruption_family)
);
"""


def corr_rng(rid, fam):
    return random.Random(hash((rid, fam, SEED)) & 0xFFFFFFFF)


def verify_fake(fam, original_grid, corrupted_grid, spans, caption, tags):
    """(1) provenance can exactly reconstruct corrupted from original (2) the injected violation is actually detectable."""
    # (1) reconstruct
    rebuilt = copy.deepcopy(original_grid)
    for sp in spans:
        rebuilt[sp["r"]][sp["c"]] = sp["new"]
    if rebuilt != corrupted_grid:
        return False, "provenance cannot exactly reconstruct"
    if not spans:
        return False, "no span"
    # (2) violation detectable
    st = StructuredTable(corrupted_grid[0], corrupted_grid[1:], caption=caption, forensic_tags=tags)
    sp = spans[0]
    i, j = sp["r"] - 1, sp["c"]
    if i < 0 or i >= st.nrows or j >= st.ncols:
        return False, "span out of bounds"
    cell = st.cells[i][j]
    if fam == "GRIM":
        n = st.n_for(j)
        mt = re.match(r"^\s*(-?\d+(?:\.\d+)?)", cell["raw"])
        if not mt or cell["type"] != "meansd" or not n:
            return False, "GRIM structure lost"
        dec = len(mt.group(1).split(".")[1]) if "." in mt.group(1) else 0
        return (not C._grim_consistent(cell["mean"], n, dec)), "GRIM"
    if fam == "CI":
        return (cell["type"] == "ci" and not (cell["lo"] <= cell["point"] <= cell["hi"])), "CI"
    if fam == "DIST":
        return (cell["type"] == "meansd" and cell["mean"] - 2 * cell["sd"] < 0), "DIST"
    if fam == "PCT":
        if cell["type"] == "countpct":
            d = cell["denom"] or st.n_for(j)
            if d:
                return (abs(100.0 * cell["count"] / d - cell["pct"]) > 8), "PCT"
        return (sp["new"] != sp["orig"]), "PCT-sum"
    if fam == "PVAL":
        if cell["type"] != "pval":
            return False, "PVAL structure lost"
        ci = [st.cells[i][jj] for jj in range(st.ncols) if st.cells[i][jj]["type"] == "ci"]
        if not ci:
            return False, "PVAL no CI in same row (not strict)"
        cc = ci[0]
        excludes = not (cc["lo"] <= 1.0 <= cc["hi"])  # whether CI excludes null=1
        sig = cell["value"] < 0.05                     # whether p is significant
        return (excludes != sig), "PVAL"               # only a real violation when the two contradict
    if fam == "SURF":
        return (sp["new"] != sp["orig"]), "SURF"
    return False, "unknown family"


def stratum_key(support_set):
    """Paper stratification key = the rarest family it can support (spreads rare capabilities like GRIM evenly across the three splits)."""
    for fam in RARITY_ORDER:
        if fam in support_set:
            return fam
    return "cleanonly"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    conn = sqlite3.connect(DB)
    conn.execute(SCHEMA)
    existing = conn.execute("SELECT COUNT(*) FROM scinum_bench").fetchone()[0]
    if existing > 0:
        if not args.rebuild:
            print(f"[STOP] scinum_bench already has {existing} rows. Add --rebuild to wipe and rebuild.")
            return
        conn.execute("DELETE FROM scinum_bench")
        conn.execute("DELETE FROM sqlite_sequence WHERE name='scinum_bench'")
        conn.commit()

    # ── load + pre-parse + precompute strict families supported per table (deterministic rng) ──
    rows = conn.execute("""SELECT id,arxiv_id,caption,table_json,forensic_tags,source
                           FROM paper_tables WHERE forensic_usable=1""").fetchall()
    tables = {}      # rid -> dict(st, support, results)
    paper_tabs = defaultdict(list)  # arxiv_id -> [rid]
    for rid, aid, cap, tj, tags, src in rows:
        st = StructuredTable.from_json(tj, caption=cap, src_table_id=rid, arxiv_id=aid, forensic_tags=tags)
        if st is None:
            continue
        support = {}
        for fam in RIG_FAMILIES:
            grid, spans = C.attempt(st, fam, corr_rng(rid, fam))
            if grid is not None and spans:
                ok, _ = verify_fake(fam, st.to_grid(), grid, spans, cap or "", tags or "")
                if ok:
                    support[fam] = (grid, spans)
        if not support:
            continue  # skip the rare cases where tags disagree with the actual measurement
        tables[rid] = {"st": st, "aid": aid, "cap": cap or "", "tags": tags or "",
                       "support": support, "source": src}
        paper_tabs[aid].append(rid)

    # ── paper-level 60/20/20 stratified split ──
    paper_support = {}
    for aid, rids in paper_tabs.items():
        s = set()
        for rid in rids:
            s |= set(tables[rid]["support"].keys())
        paper_support[aid] = s
    strata = defaultdict(list)
    for aid in paper_tabs:
        strata[stratum_key(paper_support[aid])].append(aid)
    rng = random.Random(SEED)
    paper_split = {}
    for key, papers in strata.items():
        papers = sorted(papers)
        rng.shuffle(papers)
        n = len(papers)
        ntr = int(n * SPLIT_RATIO["train"])
        nva = int(n * SPLIT_RATIO["val"])
        for p in papers[:ntr]:
            paper_split[p] = "train"
        for p in papers[ntr:ntr + nva]:
            paper_split[p] = "val"
        for p in papers[ntr + nva:]:
            paper_split[p] = "test"
    split_tabs = defaultdict(list)
    for rid, info in tables.items():
        split_tabs[paper_split[info["aid"]]].append(rid)

    # ── within-split seeded greedy conditional assignment (rare-first, 1 per table, clean~=40%) ──
    arng = random.Random(SEED + 1)
    assign = {}  # rid -> family ("clean" or fam)
    for split in ("train", "val", "test"):
        rids = sorted(split_tabs[split])
        arng.shuffle(rids)
        N = len(rids)
        fake_target = round(FAKE_FRAC * N)
        assigned_here = 0
        unassigned = set(rids)
        for k, fam in enumerate(RARITY_ORDER):
            remaining_fams = len(RARITY_ORDER) - k
            quota = max(0, -(-(fake_target - assigned_here) // remaining_fams))  # ceil even split
            pool = [r for r in rids if r in unassigned and fam in tables[r]["support"]]
            arng.shuffle(pool)
            take = pool[:min(quota, len(pool))]
            for r in take:
                assign[r] = fam
                unassigned.discard(r)
            assigned_here += len(take)
        for r in unassigned:
            assign[r] = "clean"

    # ── OOD role ──
    def ood_role(rid):
        fam = assign[rid]
        sp = paper_split[tables[rid]["aid"]]
        if sp in ("val", "test") and fam in OOD_FAMILIES:
            return "ood_test"
        if sp == "train" and fam in OOD_TRAIN_FAMS:
            return "train_pool"
        return "none"

    # ── generate + self-check + persist ──
    buf = []
    relabeled = 0
    for rid, info in tables.items():
        st = info["st"]; aid = info["aid"]; cap = info["cap"]; tags = info["tags"]
        split = paper_split[aid]
        fam = assign[rid]
        orig = st.to_grid()
        meta = json.dumps({"col_n": st.col_n, "table_n": st.table_n,
                           "tags": sorted(st.tags), "types": st.type_counts()}, ensure_ascii=False)
        if fam == "clean":
            corrupted, spans, label, rule = orig, [], 0, None
        else:
            grid, spans = info["support"][fam]
            ok, rule = verify_fake(fam, orig, grid, spans, cap, tags)
            if not ok:  # fallback: self-check failed -> degrade to clean (never emit a corruption label)
                corrupted, spans, label, rule, fam = orig, [], 0, None, "clean"
                relabeled += 1
            else:
                corrupted, label = grid, 1
        buf.append((aid, rid, info["source"], split, label, fam,
                    (C.RULE.get(fam) if label else None),
                    ood_role(rid),
                    json.dumps(orig, ensure_ascii=False),
                    json.dumps(corrupted, ensure_ascii=False),
                    json.dumps(spans, ensure_ascii=False),
                    len(spans), meta))
    conn.executemany("""INSERT OR IGNORE INTO scinum_bench
        (arxiv_id,src_table_id,source,dataset_split,label,corruption_family,
         forensic_rule,ood_role,original_grid,corrupted_grid,provenance,n_cells_changed,table_meta)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""", buf)
    conn.commit()
    if relabeled:
        print(f"[SELFCHECK] {relabeled} tables failed self-check -> degraded to clean (no corruption labels emitted)")

    report(conn)
    conn.close()


def report(conn):
    tot = conn.execute("SELECT COUNT(*) FROM scinum_bench").fetchone()[0]
    print(f"total samples: {tot}")
    print("\nsource × label:")
    for src, lab, n in conn.execute("SELECT source,label,COUNT(*) FROM scinum_bench GROUP BY source,label ORDER BY source,label"):
        print(f"  {src:6s} label={lab}  {n}")
    print("\nsplit × label:")
    for sp, lab, n in conn.execute("SELECT dataset_split,label,COUNT(*) FROM scinum_bench GROUP BY dataset_split,label ORDER BY dataset_split,label"):
        print(f"  {sp:5s} label={lab}  {n}")
    print("\nsplit × family:")
    cur = conn.execute("SELECT dataset_split,corruption_family,COUNT(*) FROM scinum_bench GROUP BY dataset_split,corruption_family ORDER BY dataset_split")
    d = defaultdict(dict)
    for sp, fam, n in cur:
        d[sp][fam] = n
    fams = ["clean"] + RIG_FAMILIES
    print("  split   " + "  ".join(f"{f:>5s}" for f in fams))
    for sp in ("train", "val", "test"):
        print(f"  {sp:5s}   " + "  ".join(f"{d[sp].get(f,0):5d}" for f in fams))
    print("\nOOD tracks:")
    for role, n in conn.execute("SELECT ood_role,COUNT(*) FROM scinum_bench GROUP BY ood_role"):
        print(f"  {role:11s} {n}")
    # self-check review: sample fakes to verify provenance can reconstruct
    bad = 0
    for orig, corr, prov in conn.execute("SELECT original_grid,corrupted_grid,provenance FROM scinum_bench WHERE label=1 LIMIT 99999"):
        o = json.loads(orig); cc = json.loads(corr); sp = json.loads(prov)
        rebuilt = copy.deepcopy(o)
        for s in sp:
            rebuilt[s["r"]][s["c"]] = s["new"]
        if rebuilt != cc:
            bad += 1
    print(f"\n[VERIFY] provenance reconstruction check: {'all passed' if bad==0 else f'{bad} failed'}")


if __name__ == "__main__":
    main()
