"""
build_hybrid.py — selective hybrid detector (rules + blind-spot probe)

All rules are kept. The probe is enabled only when [rules miss AND a rule
blind-spot candidate exists] (threshold calibrated on val):
  DIST blind-spot candidate = a meansd cell with mean-2SD<0 (rules abstain due
    to missing non-negativity keyword)
  PCT  blind-spot candidate = a countpct cell whose denominator cannot be
    recovered (rules abstain due to missing denominator)
The probe's influence is confined to rule blind spots, avoiding a global 47%
false-positive rate.
"""
import sqlite3, json, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, "/home/pengminjie/Backup/paper/ICDE26/code/bench")
from parser import StructuredTable
import detectors as D

DB = "/home/pengminjie/Backup/paper/ICDE26/data/arxiv_data.db"
d = np.load("/home/pengminjie/Backup/paper/ICDE26/data/features/qwen7b_prompted.npz", allow_pickle=True)
bid = d["bench_id"].astype(int); label = d["label"].astype(int); split = d["split"].astype(str)
tr = split == "train"
X = d["p28"].astype(np.float32)
sc = StandardScaler().fit(X[tr])
clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(X[tr]), label[tr])
proba_all = clf.predict_proba(sc.transform(X))[:, 1]
proba = {int(b): float(p) for b, p in zip(bid, proba_all)}


def analyze(st):
    """Return (rule_pred, dist_candidate, pct_candidate)."""
    rp = 1 if D.scan_all(st) else 0
    dist_cand = False
    for (i, j) in st.positions_of("meansd"):
        c = st.cells[i][j]
        if c["sd"] > 0 and c["mean"] > 0 and c["mean"] - 2 * c["sd"] < 0:
            dist_cand = True; break
    pct_cand = False
    for (i, j) in st.positions_of("countpct"):
        c = st.cells[i][j]
        denom = c["denom"] or st.n_for(j)
        if not denom:           # rules abstain due to missing denominator
            pct_cand = True; break
    return rp, dist_cand, pct_cand


conn = sqlite3.connect(DB)


def load(splitname):
    rows = conn.execute("""SELECT sb.bench_id, sb.label, sb.corruption_family, sb.corrupted_grid, pt.caption
                           FROM scinum_bench sb LEFT JOIN paper_tables pt ON sb.src_table_id=pt.id
                           WHERE sb.dataset_split=?""", (splitname,)).fetchall()
    out = []
    for b, lab, fam, cg, cap in rows:
        g = json.loads(cg)
        st = StructuredTable(g[0], g[1:], caption=cap or "") if len(g) >= 2 else None
        rp, dc, pc = analyze(st) if st else (0, False, False)
        out.append(dict(bid=b, y=lab, fam=fam or "", rule=rp, dist=dc, pct=pc, pr=proba.get(b, 0.0)))
    return out


val = load("val"); test = load("test"); cross = load("cross_test")


def hybrid_pred(r, thr):
    if r["rule"]:
        return 1
    if (r["dist"] or r["pct"]) and r["pr"] > thr:
        return 1
    return 0


def f1(rows, thr):
    TP = FP = FN = TN = 0
    for r in rows:
        p = hybrid_pred(r, thr); y = r["y"]
        TP += (p == 1 and y == 1); FP += (p == 1 and y == 0)
        FN += (p == 0 and y == 1); TN += (p == 0 and y == 0)
    P = TP/(TP+FP) if TP+FP else 0; R = TP/(TP+FN) if TP+FN else 0
    return (2*P*R/(P+R) if P+R else 0), P, R, FP/max(1, FP+TN)


# Calibrate threshold (maximize F1 on val)
best_thr, best = 0.5, -1
for thr in np.linspace(0.30, 0.95, 66):
    fv, *_ = f1(val, thr)
    if fv > best:
        best, best_thr = fv, thr
print(f"[CALIB] val best threshold={best_thr:.2f} (val F1={best:.3f})")


def report(name, rows):
    # rules alone
    TP=FP=FN=TN=0
    for r in rows:
        p=r["rule"]; y=r["y"]
        TP+=(p==1 and y==1);FP+=(p==1 and y==0);FN+=(p==0 and y==1);TN+=(p==0 and y==0)
    Pr=TP/(TP+FP) if TP+FP else 0;Rr=TP/(TP+FN) if TP+FN else 0
    F1r=2*Pr*Rr/(Pr+Rr) if Pr+Rr else 0; fprr=FP/max(1,FP+TN)
    # hybrid
    fh,Ph,Rh,fprh=f1(rows,best_thr)
    print(f"\n=== {name} (n={len(rows)}) ===")
    print(f"  rules  : F1={F1r:.3f} R={Rr:.3f} FPR={fprr:.3f}")
    print(f"  hybrid : F1={fh:.3f} R={Rh:.3f} FPR={fprh:.3f}")
    # improvement on DIST/PCT
    for fam in ["DIST", "PCT"]:
        fk = [r for r in rows if r["fam"] == fam and r["y"] == 1]
        if fk:
            rr = np.mean([r["rule"] for r in fk]); hr = np.mean([hybrid_pred(r, best_thr) for r in fk])
            print(f"    {fam} recall: rules {rr:.2f} -> hybrid {hr:.2f} (n={len(fk)})")


report("main TEST", test)
report("cross_test (all clean)", cross)
conn.close()
