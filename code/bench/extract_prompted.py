"""
extract_prompted.py — task-prompted extraction + zero-shot LLM-judge

Wrap each table in a "find numerical inconsistency" instruction, apply_chat_template +
add_generation_prompt, left padding, and a single forward pass yields:
  1. Hidden states at the decision position (last token) across layers -> task-prompted probe features
  2. yes/no logits at the last position -> zero-shot judge prediction (no token-by-token generation)
"""
import os
os.environ.setdefault("HF_HOME", "/data/shared_models/huggingface")
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("TMPDIR", "/data/users/pengminjie/Backup/tmp")
import sqlite3, json, time
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "Qwen/Qwen2.5-7B-Instruct"
DB = "/home/pengminjie/Backup/paper/ICDE26/data/arxiv_data.db"
OUTDIR = "/home/pengminjie/Backup/paper/ICDE26/data/features"
LAYERS = [14, 21, 28]
MAXTOK = 1100
BATCH = 16
INSTR = ("Below is a table from a scientific paper. Decide whether it contains a "
         "numerical/statistical inconsistency (e.g. a mean impossible for the reported N, "
         "a confidence interval that does not contain its point estimate, percentages "
         "inconsistent with the counts, or a p-value contradicting the confidence interval). "
         "Answer with a single word: Yes or No.")
os.makedirs(OUTDIR, exist_ok=True)


def table_text(grid):
    return "\n".join(" | ".join(str(c) for c in row) for row in grid)


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, output_hidden_states=True).cuda().eval()
    H = model.config.hidden_size

    def first_ids(words):
        s = set()
        for w in words:
            ids = tok.encode(w, add_special_tokens=False)
            if ids:
                s.add(ids[0])
        return list(s)
    yes_ids = first_ids(["Yes", " Yes", "yes", " yes", "YES"])
    no_ids = first_ids(["No", " No", "no", " no", "NO"])
    print(f"[LOAD] hidden={H} | yes_ids={yes_ids} no_ids={no_ids}", flush=True)

    conn = sqlite3.connect(DB)
    rows = conn.execute("""SELECT bench_id, corrupted_grid, label, dataset_split,
                                  corruption_family, source, ood_role
                           FROM scinum_bench ORDER BY bench_id""").fetchall()
    N = len(rows)
    print(f"[DATA] {N} rows", flush=True)

    feats = {L: np.zeros((N, H), dtype=np.float16) for L in LAYERS}
    j_yes = np.zeros(N, dtype=np.float32); j_no = np.zeros(N, dtype=np.float32)
    meta = {"bench_id": [], "label": [], "split": [], "family": [], "source": [], "ood": []}
    t0 = time.time()
    for s in range(0, N, BATCH):
        chunk = rows[s:s + BATCH]
        prompts = []
        for r in chunk:
            txt = table_text(json.loads(r[1]))
            msg = [{"role": "user", "content": INSTR + "\n\n" + txt}]
            prompts.append(tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                  max_length=MAXTOK).to("cuda")
        with torch.no_grad():
            out = model(**enc)
        # left padding -> last position [:, -1] is the decision token
        for L in LAYERS:
            feats[L][s:s + len(chunk)] = out.hidden_states[L][:, -1, :].to(torch.float16).cpu().numpy()
        logits = out.logits[:, -1, :].float()
        j_yes[s:s + len(chunk)] = logits[:, yes_ids].max(1).values.cpu().numpy()
        j_no[s:s + len(chunk)] = logits[:, no_ids].max(1).values.cpu().numpy()
        for r in chunk:
            meta["bench_id"].append(r[0]); meta["label"].append(r[2]); meta["split"].append(r[3])
            meta["family"].append(r[4] or ""); meta["source"].append(r[5]); meta["ood"].append(r[6] or "none")

    save = {f"p{L}": feats[L] for L in LAYERS}
    save["judge_yes"] = j_yes; save["judge_no"] = j_no
    for k, v in meta.items():
        save[k] = np.array(v)
    np.savez(os.path.join(OUTDIR, "qwen7b_prompted.npz"), **save)
    conn.close()


if __name__ == "__main__":
    main()
