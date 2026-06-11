"""
eval_prompted.py — Zero-shot LLM-judge + task-prompt probe evaluation
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

F = "/home/pengminjie/Backup/paper/ICDE26/data/features/qwen7b_prompted.npz"
d = np.load(F, allow_pickle=True)
label = d["label"].astype(int); split = d["split"].astype(str)
family = d["family"].astype(str); source = d["source"].astype(str); ood = d["ood"].astype(str)


def metrics(y, p):
    TP = int(((p == 1) & (y == 1)).sum()); FP = int(((p == 1) & (y == 0)).sum())
    FN = int(((p == 0) & (y == 1)).sum()); TN = int(((p == 0) & (y == 0)).sum())
    P = TP / (TP + FP) if TP + FP else 0; R = TP / (TP + FN) if TP + FN else 0
    F1 = 2 * P * R / (P + R) if P + R else 0; Acc = (TP + TN) / max(1, TP + FP + FN + TN)
    return TP, FP, FN, TN, P, R, F1, Acc


def show(tag, y, p):
    if len(y) == 0:
        return
    TP, FP, FN, TN, P, R, F1, Acc = metrics(y, p)
    fpr = FP / max(1, (y == 0).sum())
    print(f"  {tag:28s} P={P:.3f} R={R:.3f} F1={F1:.3f} Acc={Acc:.3f} FPR={fpr:.3f} (n={len(y)})")


tr = split == "train"; va = split == "val"; te = split == "test"; cross = split == "cross_test"
oodm = ood == "ood_test"

print("========== Zero-shot LLM-judge (logit yes>no) ==========")
judge = (d["judge_yes"] > d["judge_no"]).astype(int)
show("Main TEST", label[te], judge[te])
show("Family OOD (ood_test)", label[oodm], judge[oodm])
show("Cross-domain cross_test", label[cross], judge[cross])
show("TEST·pmc", label[te & (source == "pmc")], judge[te & (source == "pmc")])
show("TEST·arxiv", label[te & (source == "arxiv")], judge[te & (source == "arxiv")])

print("\n========== Task-prompt probe (decision-position hidden state + LogReg) ==========")
best = None
for fs in ["p14", "p21", "p28"]:
    X = d[fs].astype(np.float32)
    sc = StandardScaler().fit(X[tr])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(X[tr]), label[tr])
    pv = clf.predict(sc.transform(X[va])); _, _, _, _, _, _, f1, _ = metrics(label[va], pv)
    print(f"  [VAL] {fs}: F1={f1:.3f}")
    if best is None or f1 > best[0]:
        best = (f1, fs, sc, clf)
_, fs, sc, clf = best
print(f"  >>> selected {fs}")
X = d[fs].astype(np.float32)
P = clf.predict(sc.transform(X))
show("Main TEST", label[te], P[te])
show("Family OOD (ood_test)", label[oodm], P[oodm])
show("Cross-domain cross_test", label[cross], P[cross])
show("TEST·pmc", label[te & (source == "pmc")], P[te & (source == "pmc")])
show("TEST·arxiv", label[te & (source == "arxiv")], P[te & (source == "arxiv")])

# per-family recall (probe, test)
print("\n  Probe per-family recall (TEST):")
for f in ["GRIM", "PVAL", "DIST", "CI", "PCT"]:
    m = te & (family == f) & (label == 1)
    if m.sum():
        print(f"    {f:5s} {(P[m]==1).mean():.2f} (n={int(m.sum())})")
