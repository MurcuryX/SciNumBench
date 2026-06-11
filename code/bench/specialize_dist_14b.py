"""
specialize_dist_14b.py — 14B specialized DIST probe + hybrid evaluation
"""
import sqlite3, json, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, "/home/pengminjie/Backup/paper/ICDE26/code/bench")
from parser import StructuredTable
import detectors as D

DB = "/home/pengminjie/Backup/paper/ICDE26/data/arxiv_data.db"
f = np.load("/home/pengminjie/Backup/paper/ICDE26/data/features/qwen14b_dist.npz", allow_pickle=True)
bid = f["bench_id"].astype(int); lab = f["label"].astype(int)
fam = f["family"].astype(str); spl = f["split"].astype(str)
FEATS = {"p24": f["p24"].astype(np.float32), "p36": f["p36"].astype(np.float32), "p48": f["p48"].astype(np.float32)}
kidx = {int(b): i for i, b in enumerate(bid)}

# 14B zero-shot judge performance on candidates
judge = (f["judge_yes"] > f["judge_no"]).astype(int)
for sp in ["test"]:
    m = spl == sp
    y = (fam[m] == "DIST").astype(int)  # among candidates, fake DIST vs not
    jp = judge[m]
    tp = int(((jp == 1) & (y == 1)).sum()); fp = int(((jp == 1) & (y == 0)).sum()); fn = int(((jp == 0) & (y == 1)).sum())
    P = tp/(tp+fp) if tp+fp else 0; R = tp/(tp+fn) if tp+fn else 0
    print(f"[14B zero-shot judge / candidate test] DIST P={P:.2f} R={R:.2f}")

# specialized probe: train candidates, positive=fake DIST, negative=clean (label0)
trm = spl == "train"
postr = trm & (fam == "DIST"); negtr = trm & (lab == 0)
selm = postr | negtr
ytr = (fam[selm] == "DIST").astype(int)
vam = spl == "val"
yva = (fam[vam] == "DIST").astype(int)  # note: val candidates include other-family fakes, treated as negatives
print(f"[SPEC14B] train candidates {int(selm.sum())} (pos {int(ytr.sum())}/neg {int((ytr==0).sum())})")
best = None
for fs, M in FEATS.items():
    sc = StandardScaler().fit(M[selm]); clf = LogisticRegression(max_iter=3000, class_weight="balanced").fit(sc.transform(M[selm]), ytr)
    pv = clf.predict(sc.transform(M[vam]))
    tp = int(((pv == 1) & (yva == 1)).sum()); fp = int(((pv == 1) & (yva == 0)).sum()); fn = int(((pv == 0) & (yva == 1)).sum())
    P = tp/(tp+fp) if tp+fp else 0; R = tp/(tp+fn) if tp+fn else 0; F1 = 2*P*R/(P+R) if P+R else 0
    print(f"  [VAL candidate] {fs}: specialized F1={F1:.3f} P={P:.2f} R={R:.2f}")
    if best is None or F1 > best[0]:
        best = (F1, fs, sc, clf)
_, fs, sc, clf = best
M = FEATS[fs]; print(f"  >>> 14B specialized using {fs}")
proba = {int(bid[i]): float(clf.predict_proba(sc.transform(M[i:i+1]))[0, 1]) for i in range(len(bid))}

# full test/val rule + cand
conn = sqlite3.connect(DB)
def load(sp):
    rs = conn.execute("""SELECT sb.bench_id,sb.label,sb.corruption_family,sb.corrupted_grid,pt.caption
                         FROM scinum_bench sb LEFT JOIN paper_tables pt ON sb.src_table_id=pt.id
                         WHERE sb.dataset_split=?""", (sp,)).fetchall()
    out = []
    for b, y, fm, cg, cap in rs:
        g = json.loads(cg); st = StructuredTable(g[0], g[1:], caption=cap or "") if len(g) >= 2 else None
        rp = 1 if (st and D.scan_all(st)) else 0
        dc = bool(st and any(st.cells[i][j]["sd"] > 0 and st.cells[i][j]["mean"] > 0 and st.cells[i][j]["mean"]-2*st.cells[i][j]["sd"] < 0 for (i, j) in st.positions_of("meansd")))
        out.append(dict(b=b, y=y, fam=fm or "", rule=rp, dist=dc))
    return out
val = load("val"); test = load("test")
def hyb(r, thr):
    if r["rule"]: return 1
    if r["dist"] and proba.get(r["b"], 0) > thr: return 1
    return 0
def sc_(rows, thr):
    TP=FP=FN=TN=0
    for r in rows:
        p=hyb(r,thr);y=r["y"];TP+=(p==1 and y==1);FP+=(p==1 and y==0);FN+=(p==0 and y==1);TN+=(p==0 and y==0)
    P=TP/(TP+FP) if TP+FP else 0;R=TP/(TP+FN) if TP+FN else 0
    return (2*P*R/(P+R) if P+R else 0),P,R,FP/max(1,FP+TN)
bt,bf=0.5,-1
for thr in np.linspace(0.30,0.95,66):
    fv,*_=sc_(val,thr)
    if fv>bf: bf,bt=fv,thr
print(f"[CALIB] threshold={bt:.2f} val F1={bf:.3f}\n")
f1h,Ph,Rh,fprh=sc_(test,bt)
print("=== Main set TEST ===")
print(f"  rule             : F1=0.881 P=0.875 R=0.887 FPR=0.190 (baseline)")
print(f"  7B specialized hybrid    : F1=0.900 P=0.862 R=0.941 FPR=0.226 (previous round)")
print(f"  14B specialized hybrid   : F1={f1h:.3f} P={Ph:.3f} R={Rh:.3f} FPR={fprh:.3f}")
dfk=[r for r in test if r["fam"]=="DIST" and r["y"]==1]
print(f"  DIST recall: rule {np.mean([r['rule'] for r in dfk]):.2f} -> 14B hybrid {np.mean([hyb(r,bt) for r in dfk]):.2f}")
cc=[r for r in test if r["y"]==0 and r["dist"]]
print(f"  clean DIST candidates {len(cc)}, 14B hybrid false positives {int(sum(hyb(r,bt) for r in cc))} ({np.mean([hyb(r,bt) for r in cc]):.2f})  <- candidate-level precision shown here")
conn.close()
