"""
qwen_selfjudge.py — Qwen2.5-14B-Instruct ZERO-SHOT SELF-JUDGE on the new 2000-row test.

Same model whose hidden states the probe reads. Reuses judge_gemma.ZS_INSTR verbatim
and parse_verdict_zeroshot (Yes->FABRICATED). Greedy decode (single forced first token
read via logits). P(fabricated) = softmax over first-token logits of {Yes,No} variants,
exactly mirroring judge_gemma_prob.py so AUROC is comparable.

Plain HF transformers (no vLLM in scinum-gpu env). Offline shared HF cache.
Output: results/qwen_selfjudge.json
"""
import os
os.environ.setdefault("HF_HOME", "/data/shared_models/huggingface")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("TMPDIR", "/data/users/pengminjie/Backup/tmp")
import sys, json, math, time
import numpy as np

ROOT = "/home/pengminjie/Backup/paper/ICDE26"
sys.path.insert(0, ROOT + "/code/bench")
import judge_gemma as JG  # ZS_INSTR, parse_verdict_zeroshot, load_jsonl

MODEL = "Qwen/Qwen2.5-14B-Instruct"
INPUT = ROOT + "/data/splits/judge_input_test.jsonl"
MAPPING = ROOT + "/data/splits/mapping.jsonl"
OUT = ROOT + "/results/qwen_selfjudge.json"
MAXTOK = 1024
BATCH = 16


def load_labels():
    lab = {}
    with open(MAPPING) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                lab[str(r["example_id"])] = int(r["label"])
    return lab


def metrics(y, score, pred):
    from sklearn.metrics import roc_auc_score, average_precision_score
    y = np.asarray(y); score = np.asarray(score); pred = np.asarray(pred)
    TP = int(((pred == 1) & (y == 1)).sum()); FP = int(((pred == 1) & (y == 0)).sum())
    FN = int(((pred == 0) & (y == 1)).sum()); TN = int(((pred == 0) & (y == 0)).sum())
    P = TP / (TP + FP) if TP + FP else 0.0
    R = TP / (TP + FN) if TP + FN else 0.0
    F1 = 2 * P * R / (P + R) if P + R else 0.0
    Acc = (TP + TN) / len(y)
    return dict(AUROC=round(float(roc_auc_score(y, score)), 4),
                AUPRC=round(float(average_precision_score(y, score)), 4),
                F1=round(F1, 4), accuracy=round(Acc, 4),
                precision=round(P, 4), recall=round(R, 4),
                TP=TP, FP=FP, FN=FN, TN=TN)


def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f"[GPU] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
          f"dev={torch.cuda.get_device_name(0)}", flush=True)

    recs = JG.load_jsonl(INPUT)
    labels = load_labels()
    print(f"[DATA] {len(recs)} tables; labels loaded={len(labels)}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # so last position = first generated token slot
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16).cuda().eval()

    # Yes/No token id sets (variants with/without leading space, casing)
    def ids_for(words):
        s = set()
        for w in words:
            for v in (w, " " + w):
                t = tok.encode(v, add_special_tokens=False)
                if len(t) == 1:
                    s.add(t[0])
        return s
    yes_ids = ids_for(["Yes", "yes", "YES", "Y"])
    no_ids = ids_for(["No", "no", "NO", "N"])
    print(f"[TOK] yes_ids={sorted(yes_ids)} no_ids={sorted(no_ids)}", flush=True)

    # Build chat prompts (verbatim ZS_INSTR)
    prompts = []
    for r in recs:
        content = JG.ZS_INSTR + "\n\n" + r["text"]
        msg = [{"role": "user", "content": content}]
        prompts.append(tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))

    per = {}
    y_all, score_all, pred_all = [], [], []
    parse_fail = 0
    t0 = time.time()
    N = len(recs)
    for s in range(0, N, BATCH):
        bp = prompts[s:s + BATCH]
        br = recs[s:s + BATCH]
        enc = tok(bp, return_tensors="pt", padding=True, truncation=True,
                  max_length=MAXTOK).to("cuda")
        with torch.no_grad():
            out = model(**enc)
        logits = out.logits[:, -1, :].float()  # (B, V) first-gen-token logits
        # argmax decoded token -> verdict (greedy)
        argmax_ids = logits.argmax(-1).tolist()
        for bi, (r, am) in enumerate(zip(br, argmax_ids)):
            eid = str(r["bench_id"])
            gen_tok = tok.decode([am])
            verdict = JG.parse_verdict_zeroshot(gen_tok)
            row_logits = logits[bi]
            yes_lp = max([row_logits[t].item() for t in yes_ids], default=-1e9)
            no_lp = max([row_logits[t].item() for t in no_ids], default=-1e9)
            if yes_lp > -1e8 or no_lp > -1e8:
                m = max(yes_lp, no_lp)
                ey, en = math.exp(yes_lp - m), math.exp(no_lp - m)
                p_fab = ey / (ey + en)
            else:
                parse_fail += 1
                p_fab = 1.0 if verdict == "FABRICATED" else 0.0
            per[eid] = {"example_id": eid, "verdict": verdict, "p_fab": float(p_fab)}
            if eid in labels:
                y_all.append(labels[eid]); score_all.append(float(p_fab))
                pred_all.append(1 if verdict == "FABRICATED" else 0)
        if (s // BATCH) % 10 == 0:
            el = time.time() - t0
            print(f"  {s+len(br)}/{N}  {(s+len(br))/max(1e-6,el):.1f} ex/s", flush=True)

    M = metrics(y_all, score_all, pred_all)
    nfab = sum(1 for v in per.values() if v["verdict"] == "FABRICATED")
    result = dict(model=MODEL, n=len(per), n_labeled=len(y_all),
                  verdict_FABRICATED=nfab, verdict_PLAUSIBLE=len(per) - nfab,
                  parse_fail=parse_fail, metrics=M,
                  per_example=[per[str(r["bench_id"])] for r in recs])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(result, open(OUT, "w"), indent=1)
    print(f"[DONE] {OUT}", flush=True)
    print(f"[METRICS] {json.dumps(M)}", flush=True)
    print(f"[DIST] FAB={nfab} PLAUS={len(per)-nfab} parse_fail={parse_fail} "
          f"wall={(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
