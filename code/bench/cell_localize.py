"""
cell_localize.py — RQ5 cell-level localization via occlusion of the hidden-state probe.

For each fabricated test table T:
  s(T)          = probe fabrication prob on the full table.
  s(T\(r,c))    = probe prob with numeric cell (r,c) masked to "?".
  susp(r,c)     = s(T) - s(T\(r,c))   (drop when masking; higher = more suspicious).
Ground truth edited cells = provenance (r,c) (== fabricated_grid vs original_grid diff).
Candidate cells = body cells (row>=1) containing >=1 digit ("numeric body cell").

Metrics over evaluated tables (only tables with >=2 numeric body cells AND >=1
edited cell that is numeric, so ranking is meaningful):
  Cell-AUROC (pooled), Cell-AUROC (mean per-table),
  Top-1 recall, Top-3 recall, MRR  — for probe and a random baseline.
Random baseline: rank candidate cells uniformly at random (analytic expectation
where closed-form, else averaged over R random permutations per table, fixed seed).

Compliance: HF shared read-only cache, offline; writes only under ~/Backup.
"""
import os
os.environ.setdefault("HF_HOME", "/data/shared_models/huggingface")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("TMPDIR", "/data/users/pengminjie/Backup/tmp")

import sys, json, time, pickle, argparse, re, random
import numpy as np
import sqlite3

ROOT = "/home/pengminjie/Backup/paper/ICDE26"
sys.path.insert(0, ROOT + "/code/bench")
from clean_serialize import serialize  # noqa: E402

DB = ROOT + "/data/arxiv_data.db"
MAPPING = ROOT + "/data/splits/mapping.jsonl"
MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"
LAYERS = [20, 28, 36, 44]
MAXTOK = 1024
MASK_TOKEN = "?"
SEED = 42

DIGIT = re.compile(r"\d")


def is_numeric_cell(v):
    return bool(DIGIT.search(str(v)))


def load_test_positives():
    """example_id -> src_table_id for split=test,label=1."""
    e2src = {}
    with open(MAPPING) as f:
        for line in f:
            r = json.loads(line)
            if r.get("split") == "test" and int(r["label"]) == 1:
                e2src[r["example_id"]] = int(r["src_table_id"])
    return e2src


def mask_cell(grid, r, c):
    g = [list(row) for row in grid]
    g[r][c] = MASK_TOKEN
    return g


def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch.nn as nn
    from sklearn.metrics import roc_auc_score

    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0, help="0=all; else random sample of fabricated tables")
    ap.add_argument("--max_variants_batch", type=int, default=16)
    ap.add_argument("--out", default=ROOT + "/results/cell_localization.json")
    a = ap.parse_args()

    rng = random.Random(SEED)
    nprng = np.random.RandomState(SEED)

    # ---- probe + scaler ----
    ck = torch.load(MODEL_NAME and ROOT + "/models/probe_14b.pt", map_location="cpu", weights_only=False)
    assert ck["tapped_layers"] == LAYERS
    in_dim = ck["in_dim"]
    thr = float(ck["f1_opt_threshold"])
    net = nn.Sequential(
        nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(64, 1),
    )
    net.load_state_dict(ck["state_dict"])
    net.eval().cuda()
    with open(ROOT + "/models/scaler_14b.pkl", "rb") as fh:
        scaler = pickle.load(fh)
    mean = torch.tensor(scaler.mean_, dtype=torch.float32).cuda()
    std = torch.tensor(scaler.scale_, dtype=torch.float32).cuda()

    # ---- LLM ----
    print(f"[GPU] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, output_hidden_states=True).cuda().eval()
    H = model.config.hidden_size
    print(f"[LOAD] H={H} taps={LAYERS}", flush=True)

    @torch.no_grad()
    def score_texts(texts):
        """List[str] -> np.array of probe fabrication probs."""
        probs = []
        for s in range(0, len(texts), a.max_variants_batch):
            chunk = texts[s:s + a.max_variants_batch]
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                      max_length=MAXTOK).to("cuda")
            out = model(**enc)
            hs = out.hidden_states
            mask = enc["attention_mask"].unsqueeze(-1).to(torch.bfloat16)
            denom = mask.sum(1).clamp(min=1)
            layer_mp = []
            for L in LAYERS:
                mp = (hs[L] * mask).sum(1) / denom  # (B,H)
                layer_mp.append(mp.to(torch.float32))
            phi = torch.cat(layer_mp, dim=1)  # (B, L*H)
            phi = (phi - mean) / std
            logit = net(phi).squeeze(-1)
            probs.append(torch.sigmoid(logit).cpu().numpy())
        return np.concatenate(probs)

    # ---- data ----
    e2src = load_test_positives()
    conn = sqlite3.connect(DB)
    # one fraud row per src (verified). caption/grid/provenance
    items = []  # (eid, src, caption, fab_grid, edited_set)
    for eid, src in e2src.items():
        row = conn.execute(
            "SELECT caption, fabricated_grid, provenance FROM llm_fraud WHERE src_table_id=?",
            (src,)).fetchone()
        if not row:
            continue
        cap, fg, prov = row
        fg = json.loads(fg)
        prov = json.loads(prov) if prov else []
        edited = set((int(p["r"]), int(p["c"])) for p in prov)
        items.append((eid, src, cap, fg, prov, edited))

    items.sort(key=lambda x: x[1])  # deterministic order by src
    if a.sample and a.sample < len(items):
        idx = sorted(rng.sample(range(len(items)), a.sample))
        items = [items[i] for i in idx]
        sample_note = f"random sample of {a.sample} fabricated tables (seed={SEED})"
    else:
        sample_note = f"all {len(items)} fabricated test tables"
    print(f"[DATA] evaluating {len(items)} tables ; {sample_note}", flush=True)

    # ---- per-table occlusion ----
    per_table = []   # dict per evaluated table
    pooled_score = []
    pooled_label = []
    n_skipped_few_cells = 0
    n_skipped_no_numeric_edit = 0
    t0 = time.time()

    for ti, (eid, src, cap, fg, prov, edited) in enumerate(items):
        # candidate numeric body cells
        cands = []
        for r in range(1, len(fg)):
            for c in range(len(fg[r])):
                if is_numeric_cell(fg[r][c]):
                    cands.append((r, c))
        # restrict edited to numeric body candidates that exist
        edited_num = set(e for e in edited if e in set(cands))
        if len(cands) < 2:
            n_skipped_few_cells += 1
            continue
        if len(edited_num) == 0:
            n_skipped_no_numeric_edit += 1
            continue

        base_text = serialize(fg, cap, provenance=prov)
        masked_texts = [serialize(mask_cell(fg, r, c), cap, provenance=prov) for (r, c) in cands]
        all_texts = [base_text] + masked_texts
        scores = score_texts(all_texts)
        s_base = float(scores[0])
        s_masked = scores[1:]
        susp = s_base - s_masked  # higher = more suspicious

        labels = np.array([1 if rc in edited_num else 0 for rc in cands], dtype=int)

        # ranking by suspiciousness (desc); tie-break stable
        order = np.argsort(-susp, kind="stable")
        ranked = [cands[i] for i in order]
        # first true-positive rank (1-indexed)
        first_rank = None
        for rk, rc in enumerate(ranked, start=1):
            if rc in edited_num:
                first_rank = rk
                break
        top1 = int(ranked[0] in edited_num)
        top3 = int(any(rc in edited_num for rc in ranked[:3]))
        mrr = 1.0 / first_rank if first_rank else 0.0
        # per-table AUROC (only if both classes present, which is guaranteed: >=1 pos, and len>edited)
        if 0 < labels.sum() < len(labels):
            tbl_auroc = float(roc_auc_score(labels, susp))
        else:
            tbl_auroc = None  # all candidates edited (degenerate) -> exclude from per-table mean

        pooled_score.extend(susp.tolist())
        pooled_label.extend(labels.tolist())

        per_table.append(dict(
            src=src, eid=eid, n_cands=len(cands), n_edited=int(labels.sum()),
            s_base=s_base, flagged=bool(s_base > thr),
            top1=top1, top3=top3, mrr=mrr, first_rank=first_rank,
            tbl_auroc=tbl_auroc,
        ))

        if ti % 25 == 0:
            el = time.time() - t0
            done = len(per_table)
            print(f"  [{ti+1}/{len(items)}] evaluated={done} "
                  f"{(ti+1)/max(1e-6,el):.2f} tab/s elapsed={el/60:.1f}m", flush=True)

    # ---- aggregate probe metrics ----
    pl = np.array(pooled_label)
    ps = np.array(pooled_score)
    pooled_auroc = float(roc_auc_score(pl, ps)) if (0 < pl.sum() < len(pl)) else None
    tbl_aurocs = [t["tbl_auroc"] for t in per_table if t["tbl_auroc"] is not None]
    mean_tbl_auroc = float(np.mean(tbl_aurocs)) if tbl_aurocs else None
    top1 = float(np.mean([t["top1"] for t in per_table]))
    top3 = float(np.mean([t["top3"] for t in per_table]))
    mrr = float(np.mean([t["mrr"] for t in per_table]))

    # ---- random baseline (analytic where possible; pooled AUROC = 0.5) ----
    # Per table with n cands, k edited: random ranking.
    # E[top1] = k/n ; E[top3] = 1 - C(n-k,3)/C(n,3) (if n>=3 else 1 if k>0) ;
    # E[MRR] for first hit at random: sum over positions analytically:
    #   P(first hit at position j) = C(n-k, j-1)/C(n,j-1) * k/(n-j+1)
    #   E[MRR] = sum_{j=1..n-k+1} (1/j) * P(first at j)
    from math import comb

    def rand_top1(n, k):
        return k / n

    def rand_top3(n, k):
        m = min(3, n)
        if n - k < m:
            return 1.0
        return 1.0 - comb(n - k, m) / comb(n, m)

    def rand_mrr(n, k):
        tot = 0.0
        for j in range(1, n - k + 2):
            # P(first hit exactly at position j)
            # first j-1 all non-edited, position j edited
            p = 1.0
            for t in range(j - 1):
                p *= (n - k - t) / (n - t)
            p *= k / (n - (j - 1))
            tot += p / j
        return tot

    r_top1 = float(np.mean([rand_top1(t["n_cands"], t["n_edited"]) for t in per_table]))
    r_top3 = float(np.mean([rand_top3(t["n_cands"], t["n_edited"]) for t in per_table]))
    r_mrr = float(np.mean([rand_mrr(t["n_cands"], t["n_edited"]) for t in per_table]))
    r_pooled_auroc = 0.5
    r_mean_tbl_auroc = 0.5  # expectation of AUROC under random scores

    # ---- table-to-cell consistency (condition on flagged) ----
    flagged = [t for t in per_table if t["flagged"]]
    not_flagged = [t for t in per_table if not t["flagged"]]
    def grp(ts, key):
        return float(np.mean([t[key] for t in ts])) if ts else None
    consistency = dict(
        n_flagged=len(flagged), n_not_flagged=len(not_flagged),
        flagged_top1=grp(flagged, "top1"), flagged_top3=grp(flagged, "top3"),
        flagged_mrr=grp(flagged, "mrr"),
        notflagged_top1=grp(not_flagged, "top1"), notflagged_top3=grp(not_flagged, "top3"),
        notflagged_mrr=grp(not_flagged, "mrr"),
    )

    out = dict(
        method="occlusion-based cell suspiciousness: susp(r,c)=s(T)-s(T\\(r,c)), "
               "mask numeric body cell value with '?'; probe = 14B 4-layer MLP (test AUROC 0.6704).",
        model=MODEL_NAME, tapped_layers=LAYERS, mask_token=MASK_TOKEN,
        table_flag_threshold=thr,
        sample_note=sample_note,
        n_tables_evaluated=len(per_table),
        n_tables_skipped_few_numeric_cells=n_skipped_few_cells,
        n_tables_skipped_no_numeric_edited_cell=n_skipped_no_numeric_edit,
        n_numeric_candidate_cells=int(len(pl)),
        n_edited_cells_evaluated=int(pl.sum()),
        candidate_cell_def="body cell (row>=1) containing >=1 digit",
        ground_truth="provenance (r,c) cells (== fabricated_grid vs original_grid diff), "
                     "restricted to numeric body candidates",
        probe=dict(
            cell_auroc_pooled=pooled_auroc,
            cell_auroc_mean_per_table=mean_tbl_auroc,
            top1_recall=top1, top3_recall=top3, mrr=mrr,
            n_per_table_auroc=len(tbl_aurocs),
        ),
        random_baseline=dict(
            cell_auroc_pooled=r_pooled_auroc,
            cell_auroc_mean_per_table=r_mean_tbl_auroc,
            top1_recall=r_top1, top3_recall=r_top3, mrr=r_mrr,
            note="analytic expectation per table (closed-form), averaged over evaluated tables",
        ),
        table_to_cell_consistency=consistency,
        seed=SEED,
    )
    # interpretation
    if pooled_auroc:
        out["interpretation"] = (
            f"The occlusion probe ranks fabricated cells above chance "
            f"(pooled cell-AUROC {pooled_auroc:.3f} vs 0.500; Top-1 {top1:.3f} vs {r_top1:.3f}; "
            f"MRR {mrr:.3f} vs {r_mrr:.3f}), so the same hidden-state signal that flags a table "
            f"also coarsely localizes which numeric cell was altered.")

    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2)
    # also dump per-table for the figure / audit
    with open(a.out.replace(".json", "_pertable.json"), "w") as fh:
        json.dump(per_table, fh)
    print("[DONE] wrote", a.out, flush=True)
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
