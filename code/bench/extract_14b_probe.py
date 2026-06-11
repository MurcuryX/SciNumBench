"""
extract_14b_probe.py — extract Qwen2.5-14B-Instruct hidden states, using exactly the same samples/inputs as the 7B probe.

Reuses load_test_records / build_train_records / split_selfcheck from method_hidden_probe.py,
ensuring train/val/test samples (sid/src_table_id/split/label) match the serialized text and the 7B run record-by-record.
14B has 48 layers; scans the layers specified by LAYERS_14B x mean-pool/last-token.

Alignment check: after extraction, compares this script's test eval_id set against the 7B npz, and train sid consistency.

Output: data/features/hidden_probe_14b_feats.npz (mp{L}/last{L} + sid/eval_id/src_table_id/label/strategy/split).
"""
import os
os.environ.setdefault("HF_HOME", "/data/shared_models/huggingface")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("TMPDIR", "/data/users/pengminjie/Backup/tmp")

import sys
import json
import time
import numpy as np

ROOT = "/home/pengminjie/Backup/paper/ICDE26"
sys.path.insert(0, ROOT + "/code/bench")
import method_hidden_probe as mhp  # reuse the same data/serialization logic

FEATDIR = ROOT + "/data/features"
OUT_NPZ = FEATDIR + "/hidden_probe_14b_feats.npz"
NPZ_7B = FEATDIR + "/hidden_probe_feats.npz"

MODEL = "Qwen/Qwen2.5-14B-Instruct"
# 48 layers; evenly scan several layers (including mid-to-late, covering relative positions ~0.5/0.75/1.0 analogous to 7B's 14/21/28)
LAYERS_14B = [20, 28, 36, 44]
MAXTOK = 1024
BATCH = 8  # 14B is larger, so reduce the batch size


def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    os.makedirs(FEATDIR, exist_ok=True)
    test_recs = mhp.load_test_records()
    test_src_ids = {r["src_table_id"] for r in test_recs}
    train_recs = mhp.build_train_records(test_src_ids)
    all_recs = train_recs + test_recs
    mhp.split_selfcheck(all_recs)

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, output_hidden_states=True).cuda().eval()
    H = model.config.hidden_size
    nL = model.config.num_hidden_layers
    print(f"[LOAD] hidden={H} layers={nL} scan={LAYERS_14B}", flush=True)
    assert all(0 < L <= nL for L in LAYERS_14B)

    N = len(all_recs)
    feats_mp = {L: np.zeros((N, H), dtype=np.float16) for L in LAYERS_14B}
    feats_last = {L: np.zeros((N, H), dtype=np.float16) for L in LAYERS_14B}
    meta = {k: [] for k in ("sid", "eval_id", "src_table_id", "label", "strategy", "split")}

    t0 = time.time()
    for s in range(0, N, BATCH):
        chunk = all_recs[s:s + BATCH]
        texts = [r["text"] for r in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=MAXTOK).to("cuda")
        with torch.no_grad():
            out = model(**enc)
        hs = out.hidden_states
        mask = enc["attention_mask"].unsqueeze(-1).to(torch.bfloat16)
        denom = mask.sum(1).clamp(min=1)
        lastpos = enc["attention_mask"].sum(1) - 1
        idx = torch.arange(len(chunk), device=enc["attention_mask"].device)
        for L in LAYERS_14B:
            mp = (hs[L] * mask).sum(1) / denom
            feats_mp[L][s:s + len(chunk)] = mp.to(torch.float16).cpu().numpy()
            lt = hs[L][idx, lastpos]
            feats_last[L][s:s + len(chunk)] = lt.to(torch.float16).cpu().numpy()
        for r in chunk:
            for k in meta:
                meta[k].append(r[k])
    save = {}
    for L in LAYERS_14B:
        save[f"mp{L}"] = feats_mp[L]
        save[f"last{L}"] = feats_last[L]
    for k, v in meta.items():
        save[k] = np.array(v)
    np.savez(OUT_NPZ, **save)

    # ── alignment check against 7B ──
    d7 = np.load(NPZ_7B, allow_pickle=True)
    sp7 = d7["split"].astype(str); eid7 = d7["eval_id"].astype(int); sid7 = d7["sid"].astype(str)
    te7 = sp7 == "test"; tr7 = sp7 == "train"
    sid14 = np.array(meta["sid"]); sp14 = np.array(meta["split"]); eid14 = np.array(meta["eval_id"])
    te14 = sp14 == "test"; tr14 = sp14 == "train"
    same_test = set(eid14[te14].tolist()) == set(eid7[te7].tolist())
    same_train = set(sid14[tr14].tolist()) == set(sid7[tr7].tolist())
    same_all_sid = set(sid14.tolist()) == set(sid7.tolist())
    print(f"\n[ALIGN] test eval_id matches 7B={same_test}  train sid matches={same_train}  "
          f"all sid matches={same_all_sid}")
    if not (same_test and same_train and same_all_sid):
        print("[WARN] alignment check did not fully pass; please verify.")


if __name__ == "__main__":
    main()
