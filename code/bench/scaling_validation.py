"""scaling_validation.py — data-scaling learning-curve validation, FIXED test.

Baseline   = old train only (2518 src tables) -> reproduce ~0.6704 (sanity).
Expanded   = old + new (~3305 src tables).
Intermediate group-subsampled points between them (extra trend points).
ALL evaluated on the SAME fixed 2000-row OLD test features (byte-identical).

Reuses ablations_planb.train_eval verbatim. Runs seeds 42,1,2 at baseline &
expanded. 14B headline + 7B. Writes results/scaling_validation.json.
"""
import os, sys, json, time
os.environ.setdefault("TMPDIR", "/data/users/pengminjie/Backup/tmp")
# cap threads: 35-core oversubscription thrashes the small CPU MLP; cap to a sane count
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "8"
import numpy as np
ROOT = "/home/pengminjie/Backup/paper/ICDE26"
sys.path.insert(0, ROOT + "/code/bench")
from ablations_planb import train_eval  # exact Step-3 hyperparams + grouped val
import torch
torch.set_num_threads(8)
FEATDIR = ROOT + "/data/features"
MAP_V2 = ROOT + "/data/splits/mapping_v2.jsonl"
RESULTS = ROOT + "/results"
SEEDS = [42, 1, 2]

def load_labels_v2():
    lab = {}
    with open(MAP_V2) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            lab[r["example_id"]] = {"label": int(r["label"]), "src": int(r["src_table_id"])}
    return lab

def load_npz(tag, suffix, labmap):
    d = np.load(f"{FEATDIR}/{tag}_{suffix}.npz", allow_pickle=True)
    feats = d["features"].astype(np.float32)  # (N,L,H)
    eids = d["example_ids"].astype(str)
    y = np.array([labmap[e]["label"] for e in eids], dtype=np.int64)
    src = np.array([labmap[e]["src"] for e in eids], dtype=np.int64)
    N, L, H = feats.shape
    return feats.reshape(N, L * H), y, eids, src

def run_model(tag, labmap):
    # old train + new train + fixed old test
    Xtr_old, ytr_old, eid_old, src_old = load_npz(tag, "train", labmap)
    Xtr_new, ytr_new, eid_new, src_new = load_npz(tag, "newtrain", labmap)
    Xte, yte, eid_te, src_te = load_npz(tag, "test", labmap)

    # NaN checks
    for nm, X in (("old_train", Xtr_old), ("new_train", Xtr_new), ("test", Xte)):
        n_nan = int(np.isnan(X).sum())
        assert n_nan == 0, f"{tag} {nm} has {n_nan} NaN"

    # alignment: new eids disjoint from old train & test
    assert not (set(eid_new) & set(eid_old)), "new∩oldtrain eid leak"
    assert not (set(eid_new) & set(eid_te)), "new∩test eid leak"
    # test src disjoint from all train src (old+new)
    all_train_src = set(src_old.tolist()) | set(src_new.tolist())
    assert not (all_train_src & set(src_te.tolist())), f"{tag} train/test src LEAK"

    Xtr_exp = np.concatenate([Xtr_old, Xtr_new], axis=0)
    ytr_exp = np.concatenate([ytr_old, ytr_new], axis=0)
    src_exp = np.concatenate([src_old, src_new], axis=0)

    n_tab_old = len(set(src_old.tolist()))
    n_tab_exp = len(all_train_src)

    def multiseed(Xtr, ytr, src_tr, label):
        aucs, auprcs = [], []
        for sd in SEEDS:
            t0 = time.time()
            r = train_eval(Xtr, ytr, src_tr, Xte, yte, seed=sd)
            print(f"    [{tag} {label} seed{sd}] AUROC={r['AUROC']} ({time.time()-t0:.0f}s)", flush=True)
            aucs.append(r["AUROC"]); auprcs.append(r["AUPRC"])
        return {"AUROC_per_seed": aucs, "AUPRC_per_seed": auprcs,
                "AUROC_mean": round(float(np.mean(aucs)), 4),
                "AUROC_std": round(float(np.std(aucs)), 4),
                "AUPRC_mean": round(float(np.mean(auprcs)), 4)}

    points = []
    # baseline (old only) all seeds
    b = multiseed(Xtr_old, ytr_old, src_old, "baseline")
    b.update({"name": "baseline_old_only", "n_train_tables": n_tab_old,
              "n_train_examples": int(len(ytr_old))})
    points.append(b)

    # intermediate group-subsampled points (old union new), seeds=42,1,2 with grouped subsample
    uniq_exp = np.array(sorted(all_train_src))
    for n_tab_target in (2900,):
        aucs = []
        for sd in SEEDS:
            rng = np.random.RandomState(sd)
            perm = uniq_exp.copy(); rng.shuffle(perm)
            sel = set(perm[:n_tab_target].tolist())
            mask = np.array([s in sel for s in src_exp], bool)
            t0 = time.time()
            r = train_eval(Xtr_exp[mask], ytr_exp[mask], src_exp[mask], Xte, yte, seed=sd)
            print(f"    [{tag} interm{n_tab_target} seed{sd}] AUROC={r['AUROC']} ({time.time()-t0:.0f}s)", flush=True)
            aucs.append(r["AUROC"])
        points.append({"name": f"intermediate_{n_tab_target}", "n_train_tables": n_tab_target,
                       "n_train_examples": int(2 * n_tab_target),
                       "AUROC_per_seed": aucs,
                       "AUROC_mean": round(float(np.mean(aucs)), 4),
                       "AUROC_std": round(float(np.std(aucs)), 4)})

    # expanded (old+new) all seeds
    e = multiseed(Xtr_exp, ytr_exp, src_exp, "expanded")
    e.update({"name": "expanded_old_plus_new", "n_train_tables": n_tab_exp,
              "n_train_examples": int(len(ytr_exp))})
    points.append(e)

    delta = round(e["AUROC_mean"] - b["AUROC_mean"], 4)
    noise = round(max(b["AUROC_std"], e["AUROC_std"]), 4)
    return {"points": points, "delta_expanded_minus_baseline": delta,
            "seed_noise_std": noise, "gain_exceeds_noise": bool(delta > noise),
            "n_train_tables_baseline": n_tab_old, "n_train_tables_expanded": n_tab_exp}

def main():
    labmap = load_labels_v2()
    out = {"test_fixed": True, "n_new_src": 787, "seeds": SEEDS,
           "test_n_rows": 2000, "test_n_tables": 1000,
           "note": "Old test features unchanged (byte-identical). New 787 src tables ALL in train. "
                   "Grouped-by-src val carve-out inside train_eval (seed-dependent).",
           "models": {}}
    for tag in ("qwen14b", "qwen7b"):
        print(f"=== {tag} ===", flush=True)
        res = run_model(tag, labmap)
        out["models"][tag] = res
        for p in res["points"]:
            print(f"  {p['name']:24s} ntab={p['n_train_tables']:5d} "
                  f"AUROC={p['AUROC_mean']:.4f}+-{p['AUROC_std']:.4f} {p['AUROC_per_seed']}", flush=True)
        print(f"  DELTA(exp-base)={res['delta_expanded_minus_baseline']} "
              f"noise={res['seed_noise_std']} gain>noise={res['gain_exceeds_noise']}", flush=True)
    json.dump(out, open(f"{RESULTS}/scaling_validation.json", "w"), indent=2)
    print(f"[DONE] -> {RESULTS}/scaling_validation.json")

if __name__ == "__main__":
    main()
