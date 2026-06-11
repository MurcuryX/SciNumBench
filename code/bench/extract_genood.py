#!/usr/bin/env python
"""
extract_genood.py — Encode the gemma-authored OOD split with QWEN (same model,
same tap layers, same mean-pool) as extract_planb.py. The table is AUTHORED by
gemma but ENCODED by Qwen, then classified by the Qwen-trained probe.

Usage: python extract_genood.py --model {7b,14b}
Out:   data/features/qwen{7b,14b}_genood.npz  (features[N,L,H] float16, example_ids, taps, H)
"""
import os
os.environ.setdefault("HF_HOME", "/data/shared_models/huggingface")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("TMPDIR", "/data/users/pengminjie/Backup/tmp")

import json, time, argparse
import numpy as np

ROOT = "/home/pengminjie/Backup/paper/ICDE26"
SPLITDIR = ROOT + "/data/splits"
FEATDIR = ROOT + "/data/features"

CFG = {
    "7b": dict(model="Qwen/Qwen2.5-7B-Instruct", layers=[14, 21, 28], batch=16, prefix="qwen7b"),
    "14b": dict(model="Qwen/Qwen2.5-14B-Instruct", layers=[20, 28, 36, 44], batch=8, prefix="qwen14b"),
}
MAXTOK = 1024
SPLITFILE = SPLITDIR + "/gen_ood_model.jsonl"


def load_split():
    eids, texts = [], []
    with open(SPLITFILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            eids.append(str(r["example_id"]))
            texts.append(r["text"])
    return eids, texts


def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["7b", "14b"], required=True)
    a = ap.parse_args()
    cfg = CFG[a.model]
    os.makedirs(FEATDIR, exist_ok=True)

    print(f"[GPU] CVD={os.environ.get('CUDA_VISIBLE_DEVICES')} device={torch.cuda.get_device_name(0)}", flush=True)
    print(f"[LOAD] {cfg['model']} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"], torch_dtype=torch.bfloat16, output_hidden_states=True).cuda().eval()
    H = model.config.hidden_size
    nL = model.config.num_hidden_layers
    layers = cfg["layers"]
    assert all(0 < L <= nL for L in layers), "layer index out of range"
    print(f"[LOAD] hidden={H} layers={nL} tap={layers}", flush=True)

    eids, texts = load_split()
    N = len(texts)
    print(f"[SPLIT] genood: N={N}", flush=True)
    feats = np.zeros((N, len(layers), H), dtype=np.float16)
    t0 = time.time(); peak_mb = 0
    batch = cfg["batch"]
    for s in range(0, N, batch):
        chunk = texts[s:s + batch]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=MAXTOK).to("cuda")
        with torch.no_grad():
            out = model(**enc)
        hs = out.hidden_states
        mask = enc["attention_mask"].unsqueeze(-1).to(torch.bfloat16)
        denom = mask.sum(1).clamp(min=1)
        for li, L in enumerate(layers):
            mp = (hs[L] * mask).sum(1) / denom
            feats[s:s + len(chunk), li, :] = mp.to(torch.float16).cpu().numpy()
        peak_mb = max(peak_mb, torch.cuda.max_memory_allocated() // (1024 * 1024))
        if (s // batch) % 10 == 0:
            el = time.time() - t0
            print(f"  [genood] {s+len(chunk)}/{N} {(s+len(chunk))/max(1e-6,el):.0f} ex/s peak={peak_mb}MB", flush=True)
    out_npz = f"{FEATDIR}/{cfg['prefix']}_genood.npz"
    np.savez(out_npz, features=feats, example_ids=np.array(eids),
             tapped_layer_indices=np.array(layers, dtype=np.int64), hidden_dim=np.int64(H))
    nan = int(np.isnan(feats.astype(np.float32)).sum())
    print(f"[DONE] -> {out_npz} shape={feats.shape} dtype={feats.dtype} NaN={nan} "
          f"{(time.time()-t0)/60:.1f}min peak={peak_mb}MB", flush=True)


if __name__ == "__main__":
    main()
