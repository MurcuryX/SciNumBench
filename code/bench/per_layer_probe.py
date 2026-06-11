"""
per_layer_probe.py — train the probe on EACH single layer's mean-pooled vector
(5120-dim) and record TEST AUROC. Reuses ablations_planb.train_eval (seed42,
group-disjoint val by src_table_id, same MLP). Produces AUROC-vs-layer curve.

Input : data/features/qwen14b_alllayers_{train,test}.npz
Output: results/per_layer_probe.json
"""
import os
os.environ.setdefault("TMPDIR", "/data/users/pengminjie/Backup/tmp")
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
          "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "8"
import sys, json, time
import numpy as np

ROOT = "/home/pengminjie/Backup/paper/ICDE26"
FEATDIR = ROOT + "/data/features"
MAPPING = ROOT + "/data/splits/mapping.jsonl"
OUT = ROOT + "/results/per_layer_probe.json"
sys.path.insert(0, ROOT + "/code/bench")

import torch
torch.set_num_threads(8)
import ablations_planb as AB

SEED = 42
EXTRA_SEEDS = [42, 1, 7]  # 3-seed band on representative layers


def load_labels():
    lab = {}
    with open(MAPPING) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                lab[r["example_id"]] = {"label": int(r["label"]), "src": int(r["src_table_id"])}
    return lab


def load_all(split, labmap):
    d = np.load(f"{FEATDIR}/qwen14b_alllayers_{split}.npz", allow_pickle=True)
    feats = d["features"].astype(np.float32)  # (N, L, H)
    eids = d["example_ids"].astype(str)
    y = np.array([labmap[e]["label"] for e in eids], dtype=np.int64)
    src = np.array([labmap[e]["src"] for e in eids], dtype=np.int64)
    layers = list(map(int, d["layer_indices"]))
    return feats, y, eids, src, layers


def main():
    labmap = load_labels()
    Ftr, ytr, eids_tr, src_tr, layers = load_all("train", labmap)
    Fte, yte, eids_te, src_te, layers_te = load_all("test", labmap)
    assert layers == layers_te
    Ntr, L, H = Ftr.shape
    print(f"[DATA] train {Ftr.shape} test {Fte.shape} layers={layers[0]}..{layers[-1]} "
          f"pos_rate_test={yte.mean():.3f}", flush=True)

    curve = []
    t0 = time.time()
    for li, lyr in enumerate(layers):
        Xtr = Ftr[:, li, :]
        Xte = Fte[:, li, :]
        res = AB.train_eval(Xtr, ytr, src_tr, Xte, yte, seed=SEED)
        curve.append({"layer": int(lyr), "auroc": res["AUROC"], "auprc": res["AUPRC"]})
        print(f"  layer {lyr:2d}  AUROC={res['AUROC']}  AUPRC={res['AUPRC']}  "
              f"[{(time.time()-t0)/60:.1f}min]", flush=True)

    aurocs = [c["auroc"] for c in curve]
    peak_idx = int(np.argmax(aurocs))
    peak = curve[peak_idx]
    # 3-seed band on peak +/- representative (peak, a mid, a late)
    band_layers = sorted(set([peak["layer"],
                              layers[len(layers) // 2],
                              layers[-5] if len(layers) >= 5 else layers[-1]]))
    band = {}
    for lyr in band_layers:
        li = layers.index(lyr)
        Xtr = Ftr[:, li, :]; Xte = Fte[:, li, :]
        vals = []
        for sd in EXTRA_SEEDS:
            r = AB.train_eval(Xtr, ytr, src_tr, Xte, yte, seed=sd)
            vals.append(r["AUROC"])
        band[str(lyr)] = {"seeds": EXTRA_SEEDS, "aurocs": vals,
                          "mean": round(float(np.mean(vals)), 4),
                          "std": round(float(np.std(vals)), 4)}
        print(f"  [band] layer {lyr}: {vals} mean={np.mean(vals):.4f} std={np.std(vals):.4f}", flush=True)

    out = dict(model="Qwen2.5-14B-Instruct", n_train=int(Ntr), n_test=int(len(yte)),
               n_layers=L, seed=SEED, curve=curve,
               peak_layer=peak["layer"], peak_auroc=peak["auroc"],
               concat4_reference=0.66, ensemble_reference=0.675,
               three_seed_band=band)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"[DONE] {OUT}  peak layer={peak['layer']} AUROC={peak['auroc']}", flush=True)


if __name__ == "__main__":
    main()
