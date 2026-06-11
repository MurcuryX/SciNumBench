"""
judge_gemma_llmfraud.py — Gemma-3-12B-it GENERAL zero-shot LLM-judge baseline (training-free)
for the llm_fraud (boost-ours) eval set.

IMPORTANT: the prompt is intentionally GENERIC. It only asks whether any number in the
table has been fabricated/manipulated/tampered with. It must NOT leak any fabrication
mechanism (no mention of "authors' method", "comparison table", "boosting own results",
specific corruption families, granularity/CI/p-value cues, etc.).

Decision is read from next-token logits: we compare P(Yes) vs P(No) on the first
generated token (greedy, max_tokens=1, logprobs over vocab), avoiding free-form parsing.
  Yes -> FABRICATED, No -> PLAUSIBLE.

Input  : results/llmfraud_judge_input.jsonl  (each line: {eval_id, text})
Output : verdict JSON {eval_id(str): "FABRICATED"|"PLAUSIBLE"}

Compliance: HF cache shared read-only; offline-first; writes only under ~/Backup.
"""
import os
os.environ.setdefault("HF_HOME", "/data/shared_models/huggingface")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TMPDIR", "/data/users/pengminjie/Backup/tmp")
import sys
import json
import time
import argparse

MODEL = "google/gemma-3-12b-it"

# ── GENERIC audit prompt. No mechanism leakage. ──
GENERIC_INSTR = (
    "You are auditing a scientific table. Has any number in this table been "
    "fabricated, manipulated, or tampered with? Consider only the table content "
    "shown. Answer with a single word: Yes or No."
)


def load_jsonl(path):
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def build_prompts(recs, tokenizer):
    prompts = []
    for r in recs:
        content = GENERIC_INSTR + "\n\n" + r["text"]
        msg = [{"role": "user", "content": content}]
        prompts.append(tokenizer.apply_chat_template(
            msg, tokenize=False, add_generation_prompt=True))
    return prompts


def _yes_no_from_logprobs(logprob_dict, tokenizer):
    """Given vLLM logprobs dict {token_id: Logprob} for the first generated token,
    return ('FABRICATED'|'PLAUSIBLE', p_yes, p_no, top_token_str).
    We aggregate any token whose decoded text starts with yes/no (case-insensitive,
    stripped) to be robust to leading spaces / BOS quirks."""
    import math
    p_yes = -float("inf")
    p_no = -float("inf")
    top_tok = None
    top_lp = -float("inf")
    for tid, lp in logprob_dict.items():
        # vLLM Logprob has .logprob and .decoded_token (may be None)
        text = getattr(lp, "decoded_token", None)
        if text is None:
            text = tokenizer.decode([tid])
        lpv = lp.logprob
        if lpv > top_lp:
            top_lp = lpv
            top_tok = repr(text)
        s = text.strip().lower()
        if s.startswith("yes"):
            p_yes = max(p_yes, lpv)
        elif s.startswith("no"):
            p_no = max(p_no, lpv)
    verdict = "FABRICATED" if p_yes >= p_no else "PLAUSIBLE"
    py = math.exp(p_yes) if p_yes != -float("inf") else 0.0
    pn = math.exp(p_no) if p_no != -float("inf") else 0.0
    return verdict, py, pn, top_tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--logprobs", type=int, default=20)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    recs = load_jsonl(args.input)
    if args.limit:
        recs = recs[:args.limit]
    print(f"[DATA] {len(recs)} tables from {args.input}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = build_prompts(recs, tok)

    # Next-token logit read: greedy, 1 token, with top-k logprobs.
    sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=args.logprobs)

    llm = LLM(model=args.model, dtype="bfloat16",
              gpu_memory_utilization=args.gpu_mem,
              max_model_len=args.max_model_len,
              tensor_parallel_size=args.tp,
              trust_remote_code=True)

    t0 = time.time()
    outputs = llm.generate(prompts, sp)

    verdicts = {}
    n_no_yesno = 0  # cases where neither yes nor no appeared in top-k
    dbg = []
    for r, o in zip(recs, outputs):
        out0 = o.outputs[0]
        lp_list = out0.logprobs  # list per generated token
        if lp_list and lp_list[0]:
            verdict, py, pn, top_tok = _yes_no_from_logprobs(lp_list[0], tok)
            if py == 0.0 and pn == 0.0:
                # neither yes nor no in top-k: fall back to generated text
                n_no_yesno += 1
                gen = out0.text.strip().lower()
                verdict = "FABRICATED" if gen.startswith("yes") else "PLAUSIBLE"
        else:
            gen = out0.text.strip().lower()
            verdict = "FABRICATED" if gen.startswith("yes") else "PLAUSIBLE"
            py = pn = 0.0
            top_tok = repr(out0.text)
        verdicts[str(r["eval_id"])] = verdict
        dbg.append((r["eval_id"], verdict, round(py, 4), round(pn, 4), top_tok))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(verdicts, f, indent=1)

    n_fab = sum(1 for v in verdicts.values() if v == "FABRICATED")
    n_pla = len(verdicts) - n_fab
    print(f"[DONE] {args.out}  n={len(verdicts)}  FABRICATED={n_fab}  PLAUSIBLE={n_pla}  "
          f"no_yesno_in_topk={n_no_yesno}  ({(time.time()-t0)/60:.1f} min)", flush=True)
    # sample debug
    print("[SAMPLE] eval_id verdict p_yes p_no top_token", flush=True)
    for row in dbg[:20]:
        print("  ", row, flush=True)


if __name__ == "__main__":
    main()
