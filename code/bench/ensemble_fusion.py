"""
ensemble_fusion.py — seed-ensembling + 7B/14B fusion of the hidden-state probe.

Uses ONLY cached features (CPU). Reuses train_eval() from ablations_planb.py
(EXACT MLP arch / training / scaler / group-disjoint val early-stop).

Experiments (all on the fixed 2000-row test; AUROC/AUPRC/F1):
  1. Single-seed reference: 14B, 7B, seeds 0..9 -> mean+-std AUROC.
  2. Seed ensemble: average probs across 10 seeds -> ensemble AUROC (14B, 7B).
  3. Fusion:
     (a) FEATURE-level: probe on concat[7B||14B] (31232-dim), seeds 0..9,
         single mean+-std AND seed-ensemble.
     (b) PROB-level late fusion: avg(14B-ensemble prob, 7B-ensemble prob).
  4. Best config: full metrics (AUROC/AUPRC/F1 @0.5 and @F1-opt) + delta over
     single-seed 14B mean (0.661 headline). Save its per-test probs.

Outputs:
  results/ensemble_fusion.json
  results/probe_preds_best.json
"""
import os, json
# Cap ALL thread pools BEFORE importing numpy/torch/sklearn (prior run hit
# 34-thread oversubscription via openblas). Force every BLAS backend to 8.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_v] = "8"
os.environ.setdefault("TMPDIR", "/data/users/pengminjie/Backup/tmp")
try:
    from threadpoolctl import threadpool_limits
    threadpool_limits(8)
except Exception:
    pass
import numpy as np
import torch
torch.set_num_threads(8)
from sklearn.metrics import roc_auc_score, average_precision_score

import sys
sys.path.insert(0, "/home/pengminjie/Backup/paper/ICDE26/code/bench")
from ablations_planb import (train_eval, load_labels, load_raw,
                             metrics_from_preds, f1_opt_threshold)

ROOT = "/home/pengminjie/Backup/paper/ICDE26"
RESULTS = ROOT + "/results"
SEEDS = list(range(10))


def msd(vals):
    a = np.array(vals, dtype=float)
    return round(float(a.mean()), 4), round(float(a.std()), 4)


def full_metrics(y, score):
    """AUROC/AUPRC + metrics @0.5 and @ F1-opt threshold (threshold picked on test
    here purely to report the F1-optimal operating point of the ensembled scores)."""
    auroc = float(roc_auc_score(y, score))
    auprc = float(average_precision_score(y, score))
    m05 = metrics_from_preds(y, (score >= 0.5).astype(int))
    thr = f1_opt_threshold(y, score)
    mf1 = metrics_from_preds(y, (score >= thr).astype(int))
    return dict(AUROC=round(auroc, 4), AUPRC=round(auprc, 4),
                F1_at_0p5=m05["F1"], at_0p5=m05,
                f1_opt_threshold=round(float(thr), 4),
                F1_at_f1opt=mf1["F1"], at_f1_opt=mf1)


def run_seeds(Xtr, ytr, src_tr, Xte, yte, tag):
    """Train probe for each seed; return (per-seed metric dicts, per-seed test prob arrays)."""
    per_seed = []
    prob_stack = []
    for sd in SEEDS:
        out, ste = train_eval(Xtr, ytr, src_tr, Xte, yte, seed=sd, return_scores=True)
        per_seed.append({"seed": sd, "AUROC": out["AUROC"], "AUPRC": out["AUPRC"],
                         "F1_at_f1opt": out["at_f1_opt"]["F1"]})
        prob_stack.append(ste)
        print(f"  [{tag}] seed={sd} AUROC={out['AUROC']} AUPRC={out['AUPRC']}", flush=True)
    return per_seed, np.stack(prob_stack)  # (10, Nte)


def main():
    labmap = load_labels()
    summary = {"seeds": SEEDS, "test_n": None, "configs": {}}

    # ---- load canonical features (row order identical across 7b/14b) ----
    X7tr3, ytr, eids_tr, src_tr, _ = load_raw("qwen7b", "train", labmap)
    X7te3, yte, eids_te, src_te, _ = load_raw("qwen7b", "test", labmap)
    X14tr3, ytr14, _, src_tr14, _ = load_raw("qwen14b", "train", labmap)
    X14te3, yte14, _, _, _ = load_raw("qwen14b", "test", labmap)
    assert np.array_equal(ytr, ytr14) and np.array_equal(yte, yte14)
    assert not (set(src_tr.tolist()) & set(src_te.tolist())), "train/test src LEAK"
    summary["test_n"] = int(len(yte))

    X7tr = X7tr3.reshape(len(ytr), -1)
    X7te = X7te3.reshape(len(yte), -1)
    X14tr = X14tr3.reshape(len(ytr), -1)
    X14te = X14te3.reshape(len(yte), -1)
    Xfus_tr = np.concatenate([X7tr, X14tr], axis=1)
    Xfus_te = np.concatenate([X7te, X14te], axis=1)
    print(f"dims: 7B={X7tr.shape[1]} 14B={X14tr.shape[1]} fusion={Xfus_tr.shape[1]}", flush=True)

    # ===== 1+2: single-seed + seed-ensemble, 14B and 7B =====
    print("\n[14B] seeds ...", flush=True)
    ps14, P14 = run_seeds(X14tr, ytr, src_tr, X14te, yte, "14B")
    print("\n[7B] seeds ...", flush=True)
    ps7, P7 = run_seeds(X7tr, ytr, src_tr, X7te, yte, "7B")

    m14, s14 = msd([p["AUROC"] for p in ps14])
    m7, s7 = msd([p["AUROC"] for p in ps7])
    ap14_m, ap14_s = msd([p["AUPRC"] for p in ps14])
    ap7_m, ap7_s = msd([p["AUPRC"] for p in ps7])

    ens14 = P14.mean(axis=0)
    ens7 = P7.mean(axis=0)
    ens14_m = full_metrics(yte, ens14)
    ens7_m = full_metrics(yte, ens7)

    summary["configs"]["single_14b"] = {
        "AUROC_mean": m14, "AUROC_std": s14, "AUPRC_mean": ap14_m, "AUPRC_std": ap14_s,
        "per_seed_AUROC": [p["AUROC"] for p in ps14]}
    summary["configs"]["single_7b"] = {
        "AUROC_mean": m7, "AUROC_std": s7, "AUPRC_mean": ap7_m, "AUPRC_std": ap7_s,
        "per_seed_AUROC": [p["AUROC"] for p in ps7]}
    summary["configs"]["ensemble_14b"] = {**ens14_m, "delta_vs_single_14b_mean": round(ens14_m["AUROC"] - m14, 4)}
    summary["configs"]["ensemble_7b"] = {**ens7_m, "delta_vs_single_7b_mean": round(ens7_m["AUROC"] - m7, 4)}

    # ===== 3a: FEATURE-level fusion =====
    print("\n[FUSION-feat] seeds ...", flush=True)
    psF, PF = run_seeds(Xfus_tr, ytr, src_tr, Xfus_te, yte, "FUSION-feat")
    mF, sF = msd([p["AUROC"] for p in psF])
    apF_m, apF_s = msd([p["AUPRC"] for p in psF])
    ensF = PF.mean(axis=0)
    ensF_m = full_metrics(yte, ensF)
    summary["configs"]["fusion_feature_single"] = {
        "AUROC_mean": mF, "AUROC_std": sF, "AUPRC_mean": apF_m, "AUPRC_std": apF_s,
        "per_seed_AUROC": [p["AUROC"] for p in psF], "in_dim": int(Xfus_tr.shape[1])}
    summary["configs"]["fusion_feature_ensemble"] = {
        **ensF_m, "delta_vs_single_14b_mean": round(ensF_m["AUROC"] - m14, 4)}

    # ===== 3b: PROB-level late fusion (avg of 14B-ens and 7B-ens probs) =====
    print("\n[FUSION-prob] late fusion ...", flush=True)
    ens_lf = (ens14 + ens7) / 2.0
    lf_m = full_metrics(yte, ens_lf)
    summary["configs"]["fusion_prob_ensemble"] = {
        **lf_m, "delta_vs_single_14b_mean": round(lf_m["AUROC"] - m14, 4),
        "note": "avg(14B-seed-ensemble prob, 7B-seed-ensemble prob)"}

    # ===== 4: pick best config by AUROC =====
    cand = {
        "ensemble_14b": (ens14_m["AUROC"], ens14),
        "ensemble_7b": (ens7_m["AUROC"], ens7),
        "fusion_feature_ensemble": (ensF_m["AUROC"], ensF),
        "fusion_prob_ensemble": (lf_m["AUROC"], ens_lf),
        # single-seed means are not score arrays; include 14B/7B single-mean as scalar refs only
    }
    best_name = max(cand, key=lambda k: cand[k][0])
    best_auroc, best_probs = cand[best_name]
    best_full = full_metrics(yte, best_probs)
    seed_noise = s14  # single-seed 14B std
    gain = round(best_auroc - m14, 4)
    summary["best"] = {
        "config": best_name,
        "metrics": best_full,
        "delta_over_single_14b_mean": gain,
        "delta_over_0p661_headline": round(best_auroc - 0.661, 4),
        "single_14b_seed_std": seed_noise,
        "gain_exceeds_seed_noise": bool(gain > seed_noise)}

    # save best per-test-example probs keyed by example_id
    best_pred = {str(eids_te[i]): float(best_probs[i]) for i in range(len(eids_te))}
    json.dump(best_pred, open(f"{RESULTS}/probe_preds_best.json", "w"))

    json.dump(summary, open(f"{RESULTS}/ensemble_fusion.json", "w"), indent=2)

    # ---- console report ----
    print("\n================ REPORT ================")
    print(f"single 14B: AUROC {m14}+-{s14} (AUPRC {ap14_m}+-{ap14_s})")
    print(f"single  7B: AUROC {m7}+-{s7} (AUPRC {ap7_m}+-{ap7_s})")
    print(f"ens   14B: AUROC {ens14_m['AUROC']} (delta {summary['configs']['ensemble_14b']['delta_vs_single_14b_mean']:+})")
    print(f"ens    7B: AUROC {ens7_m['AUROC']} (delta {summary['configs']['ensemble_7b']['delta_vs_single_7b_mean']:+})")
    print(f"fusion-feat single: AUROC {mF}+-{sF} ; ensemble {ensF_m['AUROC']} (delta vs 14B {summary['configs']['fusion_feature_ensemble']['delta_vs_single_14b_mean']:+})")
    print(f"fusion-prob (late): AUROC {lf_m['AUROC']} (delta vs 14B {summary['configs']['fusion_prob_ensemble']['delta_vs_single_14b_mean']:+})")
    print(f"BEST = {best_name}: {best_full}")
    print(f"  delta over single-14B-mean = {gain:+}; over 0.661 = {summary['best']['delta_over_0p661_headline']:+}")
    print(f"  seed-noise(14B std) = {seed_noise}; gain>noise = {summary['best']['gain_exceeds_seed_noise']}")
    print("saved: results/ensemble_fusion.json, results/probe_preds_best.json")


if __name__ == "__main__":
    main()
