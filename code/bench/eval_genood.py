#!/usr/bin/env python
"""
eval_genood.py — Generator-OOD evaluation.

A) HEADLINE: load the EXISTING Qwen-trained probe (models/probe_14b.pt +
   scaler_14b.pkl) and apply it UNCHANGED to the gemma-authored OOD features
   (qwen14b_genood.npz). No retraining.
B) OPTIONAL: seed-ensemble (14B) and 7B/14B prob-level fusion, by reusing
   ablations_planb.train_eval (trains on the ORIGINAL train features, scores the
   genood set as the test). Mirrors results/ensemble_fusion.json best config.

Labels for the OOD set come from data/splits/gen_ood_mapping.jsonl.
Writes results/generator_ood.json.
"""
import os, json, pickle
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
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score

import sys
ROOT = "/home/pengminjie/Backup/paper/ICDE26"
sys.path.insert(0, ROOT + "/code/bench")
from ablations_planb import (train_eval, load_raw, metrics_from_preds,
                             f1_opt_threshold)

FEATDIR = ROOT + "/data/features"
MODELDIR = ROOT + "/models"
RESULTS = ROOT + "/results"
GENOOD_MAP = ROOT + "/data/splits/gen_ood_mapping.jsonl"
SEEDS = list(range(10))

# in-distribution Qwen-authored test reference (from probe_metrics / ensemble_fusion)
INDIST = {"single_14b_AUROC": 0.661, "ensemble_14b_AUROC": 0.675,
          "saved_probe_14b_test_AUROC": 0.6704}


def make_net(d):
    return nn.Sequential(
        nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(64, 1))


def load_genood_labels():
    lab = {}
    with open(GENOOD_MAP) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                lab[r["example_id"]] = int(r["label"])
    return lab


def load_genood_concat(tag, labmap):
    d = np.load(f"{FEATDIR}/{tag}_genood.npz", allow_pickle=True)
    feats = d["features"].astype(np.float32)        # (N,L,H)
    N, L, H = feats.shape
    X = feats.reshape(N, L * H)
    eids = d["example_ids"].astype(str)
    y = np.array([labmap[e] for e in eids], dtype=np.int64)
    return X, y, eids


def full_metrics(y, score, thr_fixed=None):
    auroc = float(roc_auc_score(y, score))
    auprc = float(average_precision_score(y, score))
    m05 = metrics_from_preds(y, (score >= 0.5).astype(int))
    thr = thr_fixed if thr_fixed is not None else f1_opt_threshold(y, score)
    mf1 = metrics_from_preds(y, (score >= thr).astype(int))
    return dict(AUROC=round(auroc, 4), AUPRC=round(auprc, 4),
                F1_at_0p5=m05["F1"], at_0p5=m05,
                threshold=round(float(thr), 4),
                F1_at_thr=mf1["F1"], at_thr=mf1)


def main():
    out = {"in_distribution_reference": INDIST, "configs": {}}
    glab = load_genood_labels()

    # ----- A) HEADLINE: existing saved probe applied unchanged -----
    Xood14, yood, eids_ood = load_genood_concat("qwen14b", glab)
    out["n_ood"] = int(len(yood))
    out["n_pos"] = int(yood.sum()); out["n_neg"] = int((1 - yood).sum())
    ckpt = torch.load(f"{MODELDIR}/probe_14b.pt", map_location="cpu", weights_only=False)
    with open(f"{MODELDIR}/scaler_14b.pkl", "rb") as fh:
        scaler = pickle.load(fh)
    assert Xood14.shape[1] == ckpt["in_dim"], f"dim mismatch {Xood14.shape[1]} vs {ckpt['in_dim']}"
    net = make_net(ckpt["in_dim"]); net.load_state_dict(ckpt["state_dict"]); net.eval()
    Tood = torch.tensor(scaler.transform(Xood14), dtype=torch.float32)
    with torch.no_grad():
        sood = torch.sigmoid(net(Tood).squeeze(-1)).numpy()
    thr_saved = ckpt.get("f1_opt_threshold", 0.5)
    out["configs"]["saved_probe_14b"] = full_metrics(yood, sood, thr_fixed=thr_saved)
    out["configs"]["saved_probe_14b"]["threshold_source"] = "f1_opt from in-dist train/val (frozen)"
    out["configs"]["saved_probe_14b"]["f1_opt_on_ood"] = full_metrics(yood, sood)
    # save per-example probs
    with open(f"{RESULTS}/probe_preds_genood.json", "w") as fh:
        json.dump({eids_ood[i]: float(sood[i]) for i in range(len(eids_ood))}, fh)
    print(f"[A saved_probe_14b] OOD AUROC={out['configs']['saved_probe_14b']['AUROC']} "
          f"AUPRC={out['configs']['saved_probe_14b']['AUPRC']}", flush=True)

    # ----- B) seed-ensemble 14B + 7B/14B prob-fusion (train on orig train, score OOD) -----
    from ablations_planb import load_labels as load_indist_labels
    ilab = load_indist_labels()
    try:
        # 14B ensemble
        X14tr3, ytr, eids_tr, src_tr, _ = load_raw("qwen14b", "train", ilab)
        X14tr = X14tr3.reshape(X14tr3.shape[0], -1)
        probs14 = []
        for sd in SEEDS:
            _o, s = train_eval(X14tr, ytr, src_tr, Xood14, yood, seed=sd, return_scores=True)
            probs14.append(s)
            print(f"  [ens14 seed={sd}] OOD AUROC={_o['AUROC']}", flush=True)
        ens14 = np.stack(probs14).mean(0)
        out["configs"]["ensemble_14b"] = full_metrics(yood, ens14)

        # 7B ensemble (needs qwen7b_genood.npz)
        sevenb_path = f"{FEATDIR}/qwen7b_genood.npz"
        if os.path.exists(sevenb_path):
            Xood7, yood7, eids_ood7 = load_genood_concat("qwen7b", glab)
            assert np.array_equal(yood7, yood) and list(eids_ood7) == list(eids_ood), "7b/14b OOD row mismatch"
            X7tr3, ytr7, _, src_tr7, _ = load_raw("qwen7b", "train", ilab)
            X7tr = X7tr3.reshape(X7tr3.shape[0], -1)
            assert np.array_equal(ytr7, ytr), "7b/14b train label mismatch"
            probs7 = []
            for sd in SEEDS:
                _o, s = train_eval(X7tr, ytr7, src_tr7, Xood7, yood7, seed=sd, return_scores=True)
                probs7.append(s)
                print(f"  [ens7 seed={sd}] OOD AUROC={_o['AUROC']}", flush=True)
            ens7 = np.stack(probs7).mean(0)
            out["configs"]["ensemble_7b"] = full_metrics(yood, ens7)
            out["configs"]["fusion_7b14b_prob"] = full_metrics(yood, 0.5 * (ens14 + ens7))
            print(f"  [fusion] OOD AUROC={out['configs']['fusion_7b14b_prob']['AUROC']}", flush=True)
        else:
            out["configs"]["ensemble_7b"] = "skipped: qwen7b_genood.npz absent"
    except Exception as e:
        out["configs"]["ensemble_error"] = repr(e)
        print("[B ENSEMBLE ERROR]", repr(e), flush=True)

    # ----- comparison block -----
    saved = out["configs"]["saved_probe_14b"]
    out["comparison"] = {
        "in_dist_single_14b_AUROC": INDIST["single_14b_AUROC"],
        "in_dist_saved_probe_14b_AUROC": INDIST["saved_probe_14b_test_AUROC"],
        "ood_saved_probe_14b_AUROC": saved["AUROC"],
        "auroc_drop_vs_saved": round(INDIST["saved_probe_14b_test_AUROC"] - saved["AUROC"], 4),
    }
    if "ensemble_14b" in out["configs"] and isinstance(out["configs"]["ensemble_14b"], dict):
        out["comparison"]["in_dist_ensemble_14b_AUROC"] = INDIST["ensemble_14b_AUROC"]
        out["comparison"]["ood_ensemble_14b_AUROC"] = out["configs"]["ensemble_14b"]["AUROC"]
    if "fusion_7b14b_prob" in out["configs"] and isinstance(out["configs"]["fusion_7b14b_prob"], dict):
        out["comparison"]["ood_fusion_7b14b_AUROC"] = out["configs"]["fusion_7b14b_prob"]["AUROC"]

    ood_auroc = saved["AUROC"]
    out["verdict"] = (
        "ROBUST to fabrication source: probe trained on Qwen-authored fabrications still "
        "detects gemma-3-12B-authored fabrications well above chance."
        if ood_auroc >= 0.60 else
        ("PARTIAL: above chance but degraded vs in-distribution."
         if ood_auroc >= 0.55 else
         "NOT ROBUST: collapses toward chance (~0.5); probe likely keyed on Qwen authorship fingerprint."))
    out["probe_used"] = {"checkpoint": "models/probe_14b.pt", "scaler": "models/scaler_14b.pkl",
                         "arch": ckpt.get("arch"), "tapped_layers": ckpt.get("tapped_layers"),
                         "encoder": "Qwen/Qwen2.5-14B-Instruct (frozen)", "retrained": False}

    with open(f"{RESULTS}/generator_ood.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("\n[SAVE] " + RESULTS + "/generator_ood.json", flush=True)
    print(json.dumps(out["comparison"], indent=2), flush=True)
    print("VERDICT:", out["verdict"], flush=True)


if __name__ == "__main__":
    main()
