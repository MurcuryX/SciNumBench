"""extract_newtrain.py — extract features for the NEW train rows only.
Reads data/splits/new_train_model.jsonl, writes data/features/qwen{7b,14b}_newtrain.npz.
Identical tap layers / pooling / dtype as extract_planb.py.
"""
import os
os.environ.setdefault("HF_HOME", "/data/shared_models/huggingface")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("TMPDIR", "/data/users/pengminjie/Backup/tmp")
import sys, json, time, argparse
import numpy as np
ROOT = "/home/pengminjie/Backup/paper/ICDE26"
SPLITDIR = ROOT + "/data/splits"
FEATDIR = ROOT + "/data/features"
CFG = {
    "7b": dict(model="Qwen/Qwen2.5-7B-Instruct", layers=[14, 21, 28], batch=16, prefix="qwen7b"),
    "14b": dict(model="Qwen/Qwen2.5-14B-Instruct", layers=[20, 28, 36, 44], batch=8, prefix="qwen14b"),
}
MAXTOK = 1024

def load_new():
    eids, texts = [], []
    with open(f"{SPLITDIR}/new_train_model.jsonl") as f:
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
    assert all(0 < L <= nL for L in cfg["layers"])
    eids, texts = load_new()
    N = len(texts)
    print(f"[SPLIT] newtrain: N={N}", flush=True)
    feats = np.zeros((N, len(cfg["layers"]), H), dtype=np.float16)
    t0 = time.time(); peak_mb = 0
    batch = cfg["batch"]
    for s in range(0, N, batch):
        chunk = texts[s:s + batch]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=MAXTOK).to("cuda")
        with torch.no_grad():
            out = model(**enc)
        hs = out.hidden_states
        mask = enc["attention_mask"].unsqueeze(-1).to(torch.bfloat16)
        denom = mask.sum(1).clamp(min=1)
        for li, L in enumerate(cfg["layers"]):
            mp = (hs[L] * mask).sum(1) / denom
            feats[s:s + len(chunk), li, :] = mp.to(torch.float16).cpu().numpy()
        peak_mb = max(peak_mb, torch.cuda.max_memory_allocated() // (1024 * 1024))
        if (s // batch) % 25 == 0:
            el = time.time() - t0
            print(f"  [newtrain] {s+len(chunk)}/{N}  {(s+len(chunk))/max(1e-6,el):.0f} ex/s peak={peak_mb}MB", flush=True)
    out_npz = f"{FEATDIR}/{cfg['prefix']}_newtrain.npz"
    np.savez(out_npz, features=feats, example_ids=np.array(eids),
             tapped_layer_indices=np.array(cfg["layers"], dtype=np.int64),
             hidden_dim=np.int64(H))
    el = time.time() - t0
    n_nan = int(np.isnan(feats.astype(np.float32)).sum())
    print(f"[DONE newtrain] -> {out_npz} shape={feats.shape} {el/60:.1f}min peak={peak_mb}MB NaN={n_nan}", flush=True)

if __name__ == "__main__":
    main()
