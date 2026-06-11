"""
train_probe.py — train a linear probe on Qwen hidden states, evaluate and compare against a rule baseline

Probe = standardization + logistic regression (sklearn); train on train / select best layer on val / report on test+ood+cross.
Tests whether LLM internal representations can detect numerical inconsistency and generalize (cross-domain cross_test false positives, ood_test).
"""
import os, numpy as np
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

F = "/home/pengminjie/Backup/paper/ICDE26/data/features/qwen7b_feats.npz"
d = np.load(F, allow_pickle=True)
label = d["label"].astype(int)
split = d["split"].astype(str)
family = d["family"].astype(str)
source = d["source"].astype(str)
ood = d["ood"].astype(str)
FEATSETS = ["mp14", "mp21", "mp28", "last"]


def metrics(y, p):
    TP = int(((p == 1) & (y == 1)).sum()); FP = int(((p == 1) & (y == 0)).sum())
    FN = int(((p == 0) & (y == 1)).sum()); TN = int(((p == 0) & (y == 0)).sum())
    prec = TP / (TP + FP) if TP + FP else 0
    rec = TP / (TP + FN) if TP + FN else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    acc = (TP + TN) / max(1, TP + FP + FN + TN)
    return dict(TP=TP, FP=FP, FN=FN, TN=TN, P=prec, R=rec, F1=f1, Acc=acc)


tr = split == "train"
va = split == "val"
te = split == "test"
cross = split == "cross_test"

best = None
for fs in FEATSETS:
    X = d[fs].astype(np.float32)
    sc = StandardScaler().fit(X[tr])
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    clf.fit(sc.transform(X[tr]), label[tr])
    pv = clf.predict(sc.transform(X[va]))
    m = metrics(label[va], pv)
    print(f"[VAL] {fs}: F1={m['F1']:.3f} Acc={m['Acc']:.3f}")
    if best is None or m["F1"] > best[1]:
        best = (fs, m["F1"], sc, clf)

fs, _, sc, clf = best
print(f"\n>>> selected feature layer {fs}\n")
X = d[fs].astype(np.float32)


def report(maskname, mask):
    y = label[mask]
    if mask.sum() == 0:
        return
    p = clf.predict(sc.transform(X[mask]))
    m = metrics(y, p)
    print(f"===== {maskname} (n={int(mask.sum())}) =====")
    print(f"  P={m['P']:.3f} R={m['R']:.3f} F1={m['F1']:.3f} Acc={m['Acc']:.3f} "
          f"| TP{m['TP']} FP{m['FP']} FN{m['FN']} TN{m['TN']}")
    if (y == 0).sum():
        print(f"  clean FPR={m['FP']/max(1,(y==0).sum()):.3f}")
    fam = family[mask]
    fams = [f for f in ["GRIM", "PVAL", "DIST", "CI", "PCT"] if (fam == f).sum()]
    if fams:
        line = "  per-family recall: "
        for f in fams:
            fm = (fam == f) & (y == 1)
            if fm.sum():
                rec = (p[fm] == 1).mean()
                line += f"{f}={rec:.2f}({int(fm.sum())}) "
        print(line)


report("main TEST", te)
report("family OOD (ood_test)", ood == "ood_test")
report("cross-domain precision (cross_test CS clean)", cross)
for s in ("pmc", "arxiv"):
    report(f"TEST · {s}", te & (source == s))
