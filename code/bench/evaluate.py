"""
evaluate.py — evaluate rule-based detectors on SciNumBench

Table-level prediction = any rule fires -> fake. Reports:
  main test set: P/R/F1/acc + per-family recall (overall / matched-rule) + cell localization accuracy
  false-positive count of each detector on clean data
  cross_test (8000 CS clean): cross-domain false-positive rate
"""
import sqlite3, json, sys
from collections import Counter
sys.path.insert(0, "/home/pengminjie/Backup/paper/ICDE26/code/bench")
from parser import StructuredTable
import detectors as D

DB = "/home/pengminjie/Backup/paper/ICDE26/data/arxiv_data.db"
conn = sqlite3.connect(DB)
rows = conn.execute("""SELECT sb.dataset_split, sb.label, sb.corruption_family, sb.source,
                              sb.ood_role, sb.corrupted_grid, sb.provenance, pt.caption
                       FROM scinum_bench sb LEFT JOIN paper_tables pt ON sb.src_table_id=pt.id""").fetchall()


def parse(cg, cap):
    g = json.loads(cg)
    if len(g) < 2:
        return None
    return StructuredTable(g[0], g[1:], caption=cap or "")


def run(rowset, title):
    TP = FP = FN = TN = 0
    fam_total = Counter(); fam_caught = Counter(); fam_match = Counter()
    loc_total = loc_hit = 0
    fp_by_det = Counter()
    for split, label, fam, source, ood, cg, prov, cap in rowset:
        st = parse(cg, cap)
        flags = D.scan_all(st) if st is not None else []
        pred = 1 if flags else 0
        if label == 1:
            fam_total[fam] += 1
            if pred:
                TP += 1; fam_caught[fam] += 1
                if any(f["rule"] == fam for f in flags):
                    fam_match[fam] += 1
                pv = json.loads(prov)
                if pv:
                    truth = {(p["r"], p["c"]) for p in pv}
                    loc_total += 1
                    if any((f["r"], f["c"]) in truth for f in flags):
                        loc_hit += 1
            else:
                FN += 1
        else:
            if pred:
                FP += 1
                for f in flags:
                    fp_by_det[f["rule"]] += 1
            else:
                TN += 1
    print(f"\n===== {title} =====")
    n_pos = TP + FN; n_neg = FP + TN
    prec = TP / (TP + FP) if TP + FP else 0
    rec = TP / n_pos if n_pos else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    acc = (TP + TN) / (n_pos + n_neg) if (n_pos + n_neg) else 0
    print(f"  fake={n_pos} clean={n_neg} | TP={TP} FP={FP} FN={FN} TN={TN}")
    print(f"  Precision={prec:.3f}  Recall={rec:.3f}  F1={f1:.3f}  Acc={acc:.3f}")
    if n_neg:
        print(f"  clean false-positive rate (FPR)={FP/n_neg:.3f}")
    if fam_total:
        print("  per-family recall (overall / matched-rule):")
        for fam in ["GRIM", "PVAL", "DIST", "CI", "PCT"]:
            if fam_total[fam]:
                print(f"    {fam:5s} {fam_caught[fam]}/{fam_total[fam]}={fam_caught[fam]/fam_total[fam]:.2f}"
                      f"  matched {fam_match[fam]/fam_total[fam]:.2f}")
    if loc_total:
        print(f"  cell localization accuracy (among caught fakes, flag lands on true provenance cell): {loc_hit}/{loc_total}={loc_hit/loc_total:.3f}")
    if fp_by_det:
        print("  false positives by detector:", dict(fp_by_det))


# main test set
run([r for r in rows if r[0] == "test"], "MAIN TEST (medical+CS, balanced)")
# family OOD (ood_test within val+test)
run([r for r in rows if r[4] == "ood_test"], "FAMILY OOD (ood_test: PCT+CI held out)")
# cross-domain precision (cross_test is all clean)
run([r for r in rows if r[0] == "cross_test"], "CROSS-DOMAIN PRECISION (cross_test: 8000 CS clean)")
# split test by source
for src in ("pmc", "arxiv"):
    run([r for r in rows if r[0] == "test" and r[3] == src], f"TEST | {src} subset")
conn.close()
