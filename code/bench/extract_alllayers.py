"""
extract_alllayers.py — dump ALL transformer layers' mean-pooled hidden states for
Qwen2.5-14B-Instruct, for train_model.jsonl (5036) and test_model.jsonl (2000).

Mean-pool over tokens (attention-mask aware). Stores features[N, L_all, H] float16,
where L_all = num_hidden_states (49 for 14B: embeddings + 48 layers). Keeps
example_ids (file order, aligned with split) + layer_indices (0..48).

Output: data/features/qwen14b_alllayers_{train,test}.npz
Offline shared HF cache; writes only under ~/Backup (data disk).
"""
import os
os.environ.setdefault("HF_HOME", "/data/shared_models/huggingface")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("TMPDIR", "/data/users/pengminjie/Backup/tmp")
import sys, json, time
import numpy as np

ROOT = "/home/pengminjie/Backup/paper/ICDE26"
SPLITDIR = ROOT + "/data/splits"
FEATDIR = ROOT + "/data/features"
MODEL = "Qwen/Qwen2.5-14B-Instruct"
MAXTOK = 1024
BATCH = 8


def load_split(name):
    eids, texts = [], []
    with open(f"{SPLITDIR}/{name}_model.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                eids.append(str(r["example_id"]))
                texts.append(r["text"])
    return eids, texts


def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f"[GPU] CVD={os.environ.get('CUDA_VISIBLE_DEVICES')} "
          f"dev={torch.cuda.get_device_name(0)}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, output_hidden_states=True).cuda().eval()
    H = model.config.hidden_size
    nL = model.config.num_hidden_layers
    L_all = nL + 1  # incl. embedding layer (hidden_states[0])
    print(f"[LOAD] H={H} num_hidden_layers={nL} L_all={L_all}", flush=True)

    os.makedirs(FEATDIR, exist_ok=True)
    for name in ("train", "test"):
        eids, texts = load_split(name)
        N = len(texts)
        feats = np.zeros((N, L_all, H), dtype=np.float16)
        t0 = time.time(); peak = 0
        for s in range(0, N, BATCH):
            ct = texts[s:s + BATCH]
            enc = tok(ct, return_tensors="pt", padding=True, truncation=True,
                      max_length=MAXTOK).to("cuda")
            with torch.no_grad():
                out = model(**enc)
            hs = out.hidden_states  # tuple len L_all, each (B,T,H)
            mask = enc["attention_mask"].unsqueeze(-1).to(torch.bfloat16)
            denom = mask.sum(1).clamp(min=1)
            for li in range(L_all):
                mp = (hs[li] * mask).sum(1) / denom
                feats[s:s + len(ct), li, :] = mp.to(torch.float16).cpu().numpy()
            peak = max(peak, torch.cuda.max_memory_allocated() // (1024 * 1024))
            if (s // BATCH) % 25 == 0:
                el = time.time() - t0
                print(f"  [{name}] {s+len(ct)}/{N} {(s+len(ct))/max(1e-6,el):.0f}ex/s peak={peak}MB", flush=True)
        nan_ct = int(np.isnan(feats.astype(np.float32)).sum())
        out_npz = f"{FEATDIR}/qwen14b_alllayers_{name}.npz"
        np.savez(out_npz, features=feats, example_ids=np.array(eids),
                 layer_indices=np.arange(L_all, dtype=np.int64), hidden_dim=np.int64(H))
        print(f"[DONE {name}] {out_npz} shape={feats.shape} NaN={nan_ct} "
              f"{(time.time()-t0)/60:.1f}min peak={peak}MB", flush=True)


if __name__ == "__main__":
    main()
