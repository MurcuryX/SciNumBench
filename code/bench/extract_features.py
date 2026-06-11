"""
extract_features.py — extract hidden-state features for each SciNumBench table using Qwen2.5-7B-Instruct.

Setup:
  HF_HOME points to a read-only shared cache (Qwen only, no download/write); offline mode avoids any network access;
  features are written only to ~/Backup/paper/ICDE26/data/features/.

Approach: serialize each table (corrupted_grid, the version presented to the detector) into text -> forward pass ->
  take mean-pool over selected layers (by attention mask) + last-token hidden state. Saved as float16 npz.
Probes are later trained on these features to test whether the LLM internal representation encodes the
non-redundant signal of numerical inconsistency.
"""
import os
os.environ.setdefault("HF_HOME", "/data/shared_models/huggingface")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("TMPDIR", "/data/users/pengminjie/Backup/tmp")
import sqlite3, json, sys, time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "Qwen/Qwen2.5-7B-Instruct"
DB = "/home/pengminjie/Backup/paper/ICDE26/data/arxiv_data.db"
OUTDIR = "/home/pengminjie/Backup/paper/ICDE26/data/features"
LAYERS = [14, 21, 28]      # hidden_states: 0=emb, 1..28=layers (Qwen2.5-7B has 28 layers)
MAXTOK = 1024
BATCH = 16
os.makedirs(OUTDIR, exist_ok=True)


def grid_to_text(grid):
    head = " | ".join(str(c) for c in grid[0])
    body = "\n".join(" | ".join(str(c) for c in row) for row in grid[1:])
    return f"Table:\n{head}\n{body}"


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, output_hidden_states=True).cuda().eval()
    H = model.config.hidden_size
    print(f"[LOAD] hidden={H}, layers={model.config.num_hidden_layers}", flush=True)

    conn = sqlite3.connect(DB)
    rows = conn.execute("""SELECT bench_id, corrupted_grid, label, dataset_split,
                                  corruption_family, source, ood_role
                           FROM scinum_bench ORDER BY bench_id""").fetchall()
    N = len(rows)
    print(f"[DATA] {N} rows", flush=True)

    feats_mp = {L: np.zeros((N, H), dtype=np.float16) for L in LAYERS}
    feats_last = np.zeros((N, H), dtype=np.float16)
    meta = {"bench_id": [], "label": [], "split": [], "family": [], "source": [], "ood": []}
    t0 = time.time()
    for s in range(0, N, BATCH):
        chunk = rows[s:s + BATCH]
        texts = [grid_to_text(json.loads(r[1])) for r in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=MAXTOK).to("cuda")
        with torch.no_grad():
            out = model(**enc)
        hs = out.hidden_states  # tuple(len=layers+1) each (B,T,H)
        mask = enc["attention_mask"].unsqueeze(-1).to(torch.bfloat16)  # (B,T,1)
        denom = mask.sum(1).clamp(min=1)
        for L in LAYERS:
            mp = (hs[L] * mask).sum(1) / denom        # mean-pool
            feats_mp[L][s:s + len(chunk)] = mp.to(torch.float16).cpu().numpy()
        # last token (right padding -> use mask to find the last valid position per row)
        lastpos = enc["attention_mask"].sum(1) - 1
        lt = hs[LAYERS[-1]][torch.arange(len(chunk)), lastpos]
        feats_last[s:s + len(chunk)] = lt.to(torch.float16).cpu().numpy()
        for r in chunk:
            meta["bench_id"].append(r[0]); meta["label"].append(r[2])
            meta["split"].append(r[3]); meta["family"].append(r[4] or "")
            meta["source"].append(r[5]); meta["ood"].append(r[6] or "none")

    save = {f"mp{L}": feats_mp[L] for L in LAYERS}
    save["last"] = feats_last
    for k, v in meta.items():
        save[k] = np.array(v)
    np.savez(os.path.join(OUTDIR, "qwen7b_feats.npz"), **save)
    conn.close()


if __name__ == "__main__":
    main()
