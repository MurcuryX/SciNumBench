"""
analyze_complement.py — rule vs probe complementarity analysis
Key question: can the probe catch the fakes that rules miss? (true test of non-redundant signal)
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

# train probe (p28)
X = d["p28"].astype(np.float32)
sc = StandardScaler().fit(X[tr])
clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(X[tr]), label[tr])
probe_pred_all = clf.predict(sc.transform(X))
probe = {int(b): int(p) for b, p in zip(bid, probe_pred_all)}

# rule prediction (test rows)
conn = sqlite3.connect(DB)
rows = conn.execute("""SELECT sb.bench_id, sb.label, sb.corruption_family, sb.corrupted_grid, pt.caption
                       FROM scinum_bench sb LEFT JOIN paper_tables pt ON sb.src_table_id=pt.id
                       WHERE sb.dataset_split='test'""").fetchall()

def rule_pred(cg, cap):
    g = json.loads(cg)
    if len(g) < 2:
        return 0
    st = StructuredTable(g[0], g[1:], caption=cap or "")
    return 1 if D.scan_all(st) else 0

import numpy as np
y = []; rp = []; pp = []; fam = []
for b, lab, f, cg, cap in rows:
    y.append(lab); rp.append(rule_pred(cg, cap)); pp.append(probe.get(b, 0)); fam.append(f or "")
y = np.array(y); rp = np.array(rp); pp = np.array(pp); fam = np.array(fam)

def stat(name, pred):
    TP = int(((pred == 1) & (y == 1)).sum()); FP = int(((pred == 1) & (y == 0)).sum())
    FN = int(((pred == 0) & (y == 1)).sum()); TN = int(((pred == 0) & (y == 0)).sum())
    P = TP/(TP+FP) if TP+FP else 0; R = TP/(TP+FN) if TP+FN else 0
    F1 = 2*P*R/(P+R) if P+R else 0
    print(f"  {name:14s} P={P:.3f} R={R:.3f} F1={F1:.3f} FPR={FP/max(1,(y==0).sum()):.3f}")

print("=== Individual ===")
stat("rule", rp); stat("probe", pp)
print("=== Ensemble ===")
stat("rule OR probe", ((rp == 1) | (pp == 1)).astype(int))

print("\n=== Key: how many rule-missed fakes can the probe recover? ===")
rule_miss = (y == 1) & (rp == 0)   # rule false negatives
print(f"rule-missed fake count: {rule_miss.sum()}")
if rule_miss.sum():
    print(f"  caught by probe: {int(((rule_miss) & (pp == 1)).sum())} "
          f"({((pp[rule_miss] == 1).mean()):.2f})")
    # per-family rule misses + probe recovery
    for f in ["DIST", "PCT", "GRIM", "PVAL", "CI"]:
        m = rule_miss & (fam == f)
        if m.sum():
            print(f"    {f:5s} rule missed {int(m.sum())}, probe recovered {int((pp[m]==1).sum())} ({(pp[m]==1).mean():.2f})")
print("\n=== Cost: among clean correctly judged by rules, how many does the probe falsely flag? ===")
rule_tn = (y == 0) & (rp == 0)
print(f"clean correctly judged by rules: {rule_tn.sum()}, probe false positives among them: {int((pp[rule_tn]==1).sum())} ({(pp[rule_tn]==1).mean():.2f})")
conn.close()
