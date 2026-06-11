"""
judge_gemma_prob.py — gemma-3-12B-it zero-shot judge on the NEW test split.

REUSES judge_gemma.py verbatim: same ZS_INSTR prompt, same chat template, same
parse_verdict_zeroshot mapping (Yes->FABRICATED / No->PLAUSIBLE), greedy temp=0.
ADDS: logprobs on the first generated token so we can derive a continuous
P(fabricated) = softmax over {Yes,No} logprobs -> AUROC comparable to the probe.

Input : data/splits/judge_input_test.jsonl  ({bench_id=example_id, text})
Output: results/baseline_preds_gemma3_zeroshot.json  keyed by example_id:
        {score=P(fabricated), verdict, pred}
        + results/verdicts_gemma_zeroshot_newtest.json (verdict-only, paper-style)
"""
import os
os.environ.setdefault("HF_HOME", "/data/shared_models/huggingface")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TMPDIR", "/data/users/pengminjie/Backup/tmp")
import sys, json, math, time

ROOT = "/home/pengminjie/Backup/paper/ICDE26"
sys.path.insert(0, ROOT + "/code/bench")
import judge_gemma as JG   # reuse ZS_INSTR, parse_verdict_zeroshot, load_jsonl

MODEL = "google/gemma-3-12b-it"
INPUT = ROOT + "/data/splits/judge_input_test.jsonl"
OUT_PREDS = ROOT + "/results/baseline_preds_gemma3_zeroshot.json"
OUT_VERD = ROOT + "/results/verdicts_gemma_zeroshot_newtest.json"


def main():
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    recs = JG.load_jsonl(INPUT)
    print(f"[DATA] {len(recs)} tables from {INPUT} mode=zeroshot", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    prompts = []
    for r in recs:
        content = JG.ZS_INSTR + "\n\n" + r["text"]
        msg = [{"role": "user", "content": content}]
        prompts.append(tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))

    # greedy + logprobs to recover Yes/No probability for AUROC
    sp = SamplingParams(temperature=0.0, max_tokens=8, logprobs=20)
    llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.90,
              max_model_len=8192, tensor_parallel_size=1, trust_remote_code=True)

    t0 = time.time()
    outputs = llm.generate(prompts, sp)

    preds, verdicts = {}, {}
    parse_fail = 0
    for r, o in zip(recs, outputs):
        gen = o.outputs[0].text
        verdict = JG.parse_verdict_zeroshot(gen)
        # probability of fabricated from first-token logprobs over Yes/No variants
        p_fab = None
        try:
            first_lp = o.outputs[0].logprobs[0]  # dict {token_id: Logprob}
            yes_lp, no_lp = -1e9, -1e9
            for tid, lp in first_lp.items():
                t = lp.decoded_token.strip().lower()
                if t in ("yes", "y"):
                    yes_lp = max(yes_lp, lp.logprob)
                elif t in ("no", "n"):
                    no_lp = max(no_lp, lp.logprob)
            if yes_lp > -1e8 or no_lp > -1e8:
                m = max(yes_lp, no_lp)
                ey, en = math.exp(yes_lp - m), math.exp(no_lp - m)
                p_fab = ey / (ey + en)
        except Exception:
            p_fab = None
        if p_fab is None:
            parse_fail += 1
            p_fab = 1.0 if verdict == "FABRICATED" else 0.0
        eid = str(r["bench_id"])
        preds[eid] = {"score": float(p_fab), "verdict": verdict,
                      "pred": 1 if verdict == "FABRICATED" else 0}
        verdicts[eid] = verdict

    json.dump(preds, open(OUT_PREDS, "w"), ensure_ascii=False, indent=0)
    json.dump(verdicts, open(OUT_VERD, "w"), ensure_ascii=False, indent=1)
    nfab = sum(1 for v in verdicts.values() if v == "FABRICATED")
    print(f"[DONE] {OUT_PREDS} n={len(preds)} FABRICATED={nfab} PLAUSIBLE={len(preds)-nfab} "
          f"prob_parse_fail={parse_fail} ({(time.time()-t0)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
