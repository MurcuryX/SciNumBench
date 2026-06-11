"""
specialize_dist.py — specialized DIST probe

Task: given a table with a cell where mean-2SD<0 (a DIST candidate), decide
  whether it is a fabricated impossible SD (DIST fake) or a real large variance (legitimate clean).
Positives = DIST family fakes; negatives = real clean tables that have a DIST candidate
  (and other-family fakes whose candidates are genuine).
Try feature mp21 (mean-pool, SD value in the table text) and p28 (decision position); pick the best on val.
Compare: rule / general-probe hybrid / DIST-specialized hybrid.
"""
import sqlite3, json, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, "/home/pengminjie/Backup/paper/ICDE26/code/bench")
from parser import StructuredTable
import detectors as D

DB = "/home/pengminjie/Backup/paper/ICDE26/data/arxiv_data.db"
fa = np.load("/home/pengminjie/Backup/paper/ICDE26/data/features/qwen7b_feats.npz", allow_pickle=True)
fp = np.load("/home/pengminjie/Backup/paper/ICDE26/data/features/qwen7b_prompted.npz", allow_pickle=True)
bid = fa["bench_id"].astype(int)
idx = {int(b): k for k, b in enumerate(bid)}
FEATS = {"mp21": fa["mp21"].astype(np.float32), "mp28": fa["mp28"].astype(np.float32),
         "p28": fp["p28"].astype(np.float32)}

conn = sqlite3.connect(DB)
rows = conn.execute("""SELECT sb.bench_id, sb.label, sb.corruption_family, sb.dataset_split,
                              sb.source, sb.corrupted_grid, pt.caption
                       FROM scinum_bench sb LEFT JOIN paper_tables pt ON sb.src_table_id=pt.id
                       WHERE sb.dataset_split IN ('train','val','test')""").fetchall()


def feat_row(st):
    rp = 1 if D.scan_all(st) else 0
    dist_cand = any(st.cells[i][j]["sd"] > 0 and st.cells[i][j]["mean"] > 0
                    and st.cells[i][j]["mean"] - 2 * st.cells[i][j]["sd"] < 0
                    for (i, j) in st.positions_of("meansd"))
    return rp, dist_cand


data = []
for b, lab, fam, sp, src, cg, cap in rows:
    g = json.loads(cg)
    if len(g) < 2 or b not in idx:
        continue
    st = StructuredTable(g[0], g[1:], caption=cap or "")
    rp, dc = feat_row(st)
    data.append(dict(b=b, y=lab, fam=fam or "", sp=sp, src=src, rule=rp, dist=dc, k=idx[b]))

tr = [r for r in data if r["sp"] == "train"]
va = [r for r in data if r["sp"] == "val"]
te = [r for r in data if r["sp"] == "test"]

# Specialized training set: among candidate tables, DIST fake=1, clean=0
# (other-family fakes have genuine candidates -> also 0)
spec_tr = [r for r in tr if r["dist"] and (r["fam"] == "DIST" or r["y"] == 0)]
y_tr = np.array([1 if r["fam"] == "DIST" else 0 for r in spec_tr])

# Select feature layer (on val, specialized F1 over the candidate subset)
spec_va = [r for r in va if r["dist"]]
yva = np.array([1 if r["fam"] == "DIST" else 0 for r in spec_va])
best = None
for fs, M in FEATS.items():
    sc = StandardScaler().fit(M[[r["k"] for r in spec_tr]])
    clf = LogisticRegression(max_iter=3000, class_weight="balanced").fit(sc.transform(M[[r["k"] for r in spec_tr]]), y_tr)
    pv = clf.predict(sc.transform(M[[r["k"] for r in spec_va]]))
    tp = int(((pv == 1) & (yva == 1)).sum()); fp_ = int(((pv == 1) & (yva == 0)).sum()); fn = int(((pv == 0) & (yva == 1)).sum())
    P = tp/(tp+fp_) if tp+fp_ else 0; R = tp/(tp+fn) if tp+fn else 0; F1 = 2*P*R/(P+R) if P+R else 0
    print(f"  [VAL candidates] {fs}: specialized F1={F1:.3f} (P={P:.2f} R={R:.2f})")
    if best is None or F1 > best[0]:
        best = (F1, fs, sc, clf)
_, fs, sc, clf = best
M = FEATS[fs]
print(f"  >>> specialized probe uses {fs}")
proba = {r["b"]: clf.predict_proba(sc.transform(M[[r["k"]]]))[0, 1] for r in data}

# Threshold calibration (hybrid F1 over the full val set)
def hyb(r, thr):
    if r["rule"]:
        return 1
    if r["dist"] and proba[r["b"]] > thr:
        return 1
    return 0
def score(rows, thr):
    TP=FP=FN=TN=0
    for r in rows:
        p=hyb(r,thr);y=r["y"];TP+=(p==1 and y==1);FP+=(p==1 and y==0);FN+=(p==0 and y==1);TN+=(p==0 and y==0)
    P=TP/(TP+FP) if TP+FP else 0;R=TP/(TP+FN) if TP+FN else 0
    return (2*P*R/(P+R) if P+R else 0),P,R,FP/max(1,FP+TN)
bt,bf=0.5,-1
for thr in np.linspace(0.30,0.95,66):
    f,*_=score(va,thr)
    if f>bf: bf,bt=f,thr
print(f"[CALIB] threshold={bt:.2f} (val F1={bf:.3f})\n")

# Report test
def rule_only(rows):
    TP=FP=FN=TN=0
    for r in rows:
        p=r["rule"];y=r["y"];TP+=(p==1 and y==1);FP+=(p==1 and y==0);FN+=(p==0 and y==1);TN+=(p==0 and y==0)
    P=TP/(TP+FP) if TP+FP else 0;R=TP/(TP+FN) if TP+FN else 0
    return (2*P*R/(P+R) if P+R else 0),P,R,FP/max(1,FP+TN)
f1r,Pr,Rr,fprr=rule_only(te)
f1h,Ph,Rh,fprh=score(te,bt)
print("=== Main set TEST ===")
print(f"  rule          : F1={f1r:.3f} P={Pr:.3f} R={Rr:.3f} FPR={fprr:.3f}")
print(f"  DIST-spec hybrid: F1={f1h:.3f} P={Ph:.3f} R={Rh:.3f} FPR={fprh:.3f}")
dfk=[r for r in te if r["fam"]=="DIST" and r["y"]==1]
print(f"  DIST recall: rule {np.mean([r['rule'] for r in dfk]):.2f} -> hybrid {np.mean([hyb(r,bt) for r in dfk]):.2f} (n={len(dfk)})")
# clean DIST candidate false positives
cc=[r for r in te if r["y"]==0 and r["dist"]]
print(f"  clean DIST candidates {len(cc)}, hybrid false positives {int(sum(hyb(r,bt) for r in cc))} ({np.mean([hyb(r,bt) for r in cc]):.2f})")
conn.close()
