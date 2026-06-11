"""
judge_gemma.py — Gemma-3-12B-it LLM-judge baseline (zero-shot + CoT) via vLLM

Runs two modes over a given jsonl (produced by export_for_judge.py; each line has bench_id/text/...):
  --mode zeroshot : reuses the judge wording from extract_prompted.py (comparable to Qwen M3/M4),
                    asking "Yes/No whether a numerical inconsistency is present", mapping Yes->FABRICATED / No->PLAUSIBLE.
  --mode cot      : reuses the step-by-step reasoning structure from cot_detect.py (generalized to all inconsistency types),
                    requiring a final line "VERDICT: FABRICATED|PLAUSIBLE".

Outputs verdict JSON: {bench_id(str): "FABRICATED"|"PLAUSIBLE"}, for uniform scoring by score_verdicts.py.
"""
import os
os.environ.setdefault("HF_HOME", "/data/shared_models/huggingface")
os.environ.setdefault("TMPDIR", "/data/users/pengminjie/Backup/tmp")
# Default to offline; the caller lifts OFFLINE if local weights are missing.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import sys
import json
import re
import time
import argparse

ROOT = "/home/pengminjie/Backup/paper/ICDE26"
MODEL = "google/gemma-3-12b-it"

# zero-shot judge wording: verbatim reuse of the INSTR from extract_prompted.py / extract_14b_dist.py
ZS_INSTR = ("Below is a table from a scientific paper. Decide whether it contains a "
            "numerical/statistical inconsistency (e.g. a mean impossible for the reported N, "
            "a confidence interval that does not contain its point estimate, percentages "
            "inconsistent with the counts, or a p-value contradicting the confidence interval). "
            "Answer with a single word: Yes or No.")

# CoT prompt: aligned with cot_detect.py's "step-by-step reasoning + final-line VERDICT" structure, generalized to all families
COT_PROMPT = """You audit a table from a scientific paper for fabricated / inconsistent statistics.

Table:
{table}

Check for any numerical or statistical inconsistency, e.g.:
- a reported mean that is impossible for the stated sample size N (granularity),
- a confidence interval that does not contain its point estimate,
- counts inconsistent with the reported percentages,
- a p-value contradicting the significance implied by its confidence interval,
- a mean ± SD where mean − 2·SD < 0 for an inherently non-negative, roughly normal quantity (impossible negative values).

Reason briefly step by step over the relevant cells, then decide.
End with exactly one line: VERDICT: FABRICATED  or  VERDICT: PLAUSIBLE"""


def load_jsonl(path):
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def parse_verdict_zeroshot(text):
    """Yes->FABRICATED, No->PLAUSIBLE; decided from the first word of the generated text."""
    t = text.strip().lower()
    m = re.search(r"\b(yes|no)\b", t)
    if m:
        return "FABRICATED" if m.group(1) == "yes" else "PLAUSIBLE"
    # fallback: presence of fabricat/inconsist leans FABRICATED
    return "FABRICATED" if re.search(r"fabricat|inconsist", t) else "PLAUSIBLE"


def parse_verdict_cot(text):
    m = re.findall(r"VERDICT:\s*(FABRICATED|PLAUSIBLE)", text, re.I)
    if m:
        return m[-1].upper()
    return "FABRICATED" if "fabricat" in text.lower() else "PLAUSIBLE"


def build_prompts(recs, mode, tokenizer):
    prompts = []
    for r in recs:
        if mode == "zeroshot":
            content = ZS_INSTR + "\n\n" + r["text"]
        else:
            content = COT_PROMPT.format(table=r["text"])
        msg = [{"role": "user", "content": content}]
        prompts.append(tokenizer.apply_chat_template(
            msg, tokenize=False, add_generation_prompt=True))
    return prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="jsonl from export_for_judge.py")
    ap.add_argument("--mode", choices=["zeroshot", "cot"], required=True)
    ap.add_argument("--out", required=True, help="verdict json output path (under ~/Backup)")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--tp", type=int, default=1, help="tensor parallel size")
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N rows")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    recs = load_jsonl(args.input)
    if args.limit:
        recs = recs[:args.limit]
    print(f"[DATA] {len(recs)} tables from {args.input}  mode={args.mode}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = build_prompts(recs, args.mode, tok)

    # zero-shot needs only a very short generation; CoT needs reasoning space. greedy (temperature=0) ensures reproducibility.
    max_new = 8 if args.mode == "zeroshot" else 512
    sp = SamplingParams(temperature=0.0, max_tokens=max_new)

    llm = LLM(model=args.model, dtype="bfloat16",
              gpu_memory_utilization=args.gpu_mem,
              max_model_len=args.max_model_len,
              tensor_parallel_size=args.tp,
              trust_remote_code=True)

    t0 = time.time()
    outputs = llm.generate(prompts, sp)
    # vLLM preserves input order
    parse = parse_verdict_zeroshot if args.mode == "zeroshot" else parse_verdict_cot
    verdicts = {}
    for r, o in zip(recs, outputs):
        gen = o.outputs[0].text
        verdicts[str(r["bench_id"])] = parse(gen)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(verdicts, f, indent=1)
    n_fab = sum(1 for v in verdicts.values() if v == "FABRICATED")
    print(f"[DONE] {args.out}  n={len(verdicts)}  FABRICATED={n_fab}  "
          f"PLAUSIBLE={len(verdicts)-n_fab}  ({(time.time()-t0)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
