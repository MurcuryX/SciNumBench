"""
extract_planb.py — Plan B Step 2: extract frozen-LLM multi-layer mean-pooled
hidden-state features for the NEW anti-leak splits (train_model.jsonl /
test_model.jsonl, keyed by example_id).

Reuses the existing pipeline's tapped-layer convention EXACTLY:
  7B  (28 layers): LAYERS = [14, 21, 28]        (~50/75/100% depth)
  14B (48 layers): LAYERS = [20, 28, 36, 44]    (matching prior extract_14b_probe.py)
Mean-pool over tokens (attention-mask aware, padding excluded). Stored
separately per layer (NOT pre-concat): features[N, n_layers, H].

Compliance: HF shared read-only cache, offline; writes only under ~/Backup.
Usage: python extract_planb.py --model {7b,14b}
"""
import os
os.environ.setdefault("HF_HOME", "/data/shared_models/huggingface")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("TMPDIR", "/data/users/pengminjie/Backup/tmp")

import sys
import json
import time
import argparse
import numpy as np

ROOT = "/home/pengminjie/Backup/paper/ICDE26"
SPLITDIR = ROOT + "/data/splits"
FEATDIR = ROOT + "/data/features"

CFG = {
    "7b": dict(model="Qwen/Qwen2.5-7B-Instruct", layers=[14, 21, 28], batch=16,
               prefix="qwen7b"),
    "14b": dict(model="Qwen/Qwen2.5-14B-Instruct", layers=[20, 28, 36, 44], batch=8,
                prefix="qwen14b"),
}
MAXTOK = 1024


def load_split(name):
    """name in {train,test}. Returns (example_ids list, texts list) in file order."""
    eids, texts = [], []
    with open(f"{SPLITDIR}/{name}_model.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            eids.append(str(r["example_id"]))
            texts.append(r["text"])
    return eids, texts


def extract_split(model, tok, layers, H, name, eids, texts, batch, out_npz):
    import torch
    N = len(texts)
    # features[N, n_layers, H], float16
    feats = np.zeros((N, len(layers), H), dtype=np.float16)
    t0 = time.time()
    peak_mb = 0
    for s in range(0, N, batch):
        chunk_texts = texts[s:s + batch]
        enc = tok(chunk_texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=MAXTOK).to("cuda")
        with torch.no_grad():
            out = model(**enc)
        hs = out.hidden_states  # tuple(len=nL+1), each (B,T,H)
        mask = enc["attention_mask"].unsqueeze(-1).to(torch.bfloat16)
        denom = mask.sum(1).clamp(min=1)
        for li, L in enumerate(layers):
            mp = (hs[L] * mask).sum(1) / denom  # (B,H)
            feats[s:s + len(chunk_texts), li, :] = mp.to(torch.float16).cpu().numpy()
        peak_mb = max(peak_mb, torch.cuda.max_memory_allocated() // (1024 * 1024))
        if (s // batch) % 25 == 0:
            el = time.time() - t0
            print(f"  [{name}] {s+len(chunk_texts)}/{N}  "
                  f"{(s+len(chunk_texts))/max(1e-6,el):.0f} ex/s  peak={peak_mb}MB", flush=True)
    np.savez(out_npz,
             features=feats,
             example_ids=np.array(eids),
             tapped_layer_indices=np.array(layers, dtype=np.int64),
             hidden_dim=np.int64(H))
    el = time.time() - t0
    print(f"[DONE {name}] -> {out_npz}  shape={feats.shape} dtype={feats.dtype}  "
          f"{el/60:.1f} min  peak={peak_mb}MB", flush=True)
    return peak_mb, el


def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["7b", "14b"], required=True)
    a = ap.parse_args()
    cfg = CFG[a.model]
    os.makedirs(FEATDIR, exist_ok=True)

    print(f"[GPU] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
          f"device={torch.cuda.get_device_name(0)}", flush=True)
    print(f"[LOAD] {cfg['model']} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"], torch_dtype=torch.bfloat16, output_hidden_states=True).cuda().eval()
    H = model.config.hidden_size
    nL = model.config.num_hidden_layers
    print(f"[LOAD] hidden={H} layers={nL} tap={cfg['layers']}", flush=True)
    assert all(0 < L <= nL for L in cfg["layers"]), "layer index out of range"

    for name in ("train", "test"):
        eids, texts = load_split(name)
        print(f"[SPLIT] {name}: N={len(texts)}", flush=True)
        out_npz = f"{FEATDIR}/{cfg['prefix']}_{name}.npz"
        extract_split(model, tok, cfg["layers"], H, name, eids, texts,
                      cfg["batch"], out_npz)


if __name__ == "__main__":
    main()
