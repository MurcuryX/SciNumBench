"""
cot_detect.py — CoT-reasoning DIST detection (tests whether explicit reasoning
can break the linear-probe ceiling).

For DIST candidate tables in val+test, feed cells with mean-2SD<0 together with
their row/column labels to the 7B model, and have it reason step-by-step about
whether the variable should be non-negative and roughly normal, to decide
FABRICATED/PLAUSIBLE. Compared against the rule-based / 7B-probe hybrid. Since
both use the same 7B, this isolates reasoning vs probe.
"""
import os
os.environ.setdefault("HF_HOME", "/data/shared_models/huggingface")
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("TMPDIR", "/data/users/pengminjie/Backup/tmp")
import sqlite3, json, sys, time, re
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, "/home/pengminjie/Backup/paper/ICDE26/code/bench")
from parser import StructuredTable
import detectors as D

MODEL = "Qwen/Qwen2.5-7B-Instruct"
DB = "/home/pengminjie/Backup/paper/ICDE26/data/arxiv_data.db"
OUT = "/home/pengminjie/Backup/paper/ICDE26/data/features/cot_verdicts.json"
BATCH = 8

PROMPT = """You audit a scientific table for fabricated statistics. Values are reported as mean ± SD. For the flagged value(s) below, mean − 2×SD < 0.

Table:
{table}

Flagged value(s) (mean ± SD with mean−2·SD < 0):
{flags}

Reason briefly:
1. What real-world quantity is each flagged variable?
2. Is that quantity inherently non-negative AND roughly normally distributed (e.g. age, weight, height, BMI, time, score, concentration)? Or can it be negative / strongly skewed (e.g. change score, difference, correlation, count of rare events)?
3. If non-negative & roughly normal, mean−2SD<0 implies impossible negative values → the SD is implausibly large → FABRICATED. Otherwise → PLAUSIBLE.

End with exactly one line: VERDICT: FABRICATED  or  VERDICT: PLAUSIBLE"""


def table_text(grid):
    return "\n".join(" | ".join(str(c) for c in row) for row in grid)


def dist_flags(st):
    out = []
    for (i, j) in st.positions_of("meansd"):
        c = st.cells[i][j]
        if c["sd"] > 0 and c["mean"] > 0 and c["mean"] - 2 * c["sd"] < 0:
            rl = st.data[i][0] if st.data[i] else ""
            cl = st.columns[j] if j < len(st.columns) else ""
            out.append(f'- variable "{rl} {cl}".strip(): mean={c["mean"]}, SD={c["sd"]} (mean−2SD={c["mean"]-2*c["sd"]:.2f})')
    return out


def main():
    conn = sqlite3.connect(DB)
    rows = conn.execute("""SELECT sb.bench_id, sb.label, sb.corruption_family, sb.dataset_split,
                                  sb.corrupted_grid, pt.caption
                           FROM scinum_bench sb LEFT JOIN paper_tables pt ON sb.src_table_id=pt.id
                           WHERE sb.dataset_split IN ('val','test')""").fetchall()
    cands = []
    for b, lab, fam, sp, cg, cap in rows:
        g = json.loads(cg)
        if len(g) < 2:
            continue
        st = StructuredTable(g[0], g[1:], caption=cap or "")
        fl = dist_flags(st)
        if fl:
            cands.append((b, lab, fam or "", sp, table_text(g), "\n".join(fl)))
    print(f"[DATA] DIST candidates (val+test): {len(cands)}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()

    verdicts = {}
    t0 = time.time()
    for s in range(0, len(cands), BATCH):
        chunk = cands[s:s + BATCH]
        prompts = [tok.apply_chat_template(
            [{"role": "user", "content": PROMPT.format(table=c[4], flags=c[5])}],
            tokenize=False, add_generation_prompt=True) for c in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1300).to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=300, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        for k, c in enumerate(chunk):
            txt = tok.decode(gen[k][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            m = re.findall(r"VERDICT:\s*(FABRICATED|PLAUSIBLE)", txt, re.I)
            v = (m[-1].upper() if m else ("FABRICATED" if "fabricat" in txt.lower() else "PLAUSIBLE"))
            verdicts[c[0]] = v
        if s % (BATCH * 10) == 0:
            print(f"  {s+len(chunk)}/{len(cands)}  {(s+len(chunk))/max(1e-6,time.time()-t0):.2f} tables/s", flush=True)

    json.dump(verdicts, open(OUT, "w"))

    # Evaluation
    cmap = {c[0]: (c[1], c[2], c[3]) for c in cands}
    tec = [c for c in cands if c[3] == "test"]
    # Candidate level: precision/recall of CoT FABRICATED verdicts on test candidates (vs true DIST fakes)
    yc = np.array([1 if c[2] == "DIST" else 0 for c in tec])
    pc = np.array([1 if verdicts[c[0]] == "FABRICATED" else 0 for c in tec])
    tp = int(((pc == 1) & (yc == 1)).sum()); fp = int(((pc == 1) & (yc == 0)).sum()); fn = int(((pc == 0) & (yc == 1)).sum())
    P = tp/(tp+fp) if tp+fp else 0; R = tp/(tp+fn) if tp+fn else 0
    print(f"\n[candidate level test] CoT: P={P:.3f} R={R:.3f} (DIST fakes {int(yc.sum())}/non {int((yc==0).sum())})")
    print(f"  comparison: 7B probe candidate level P≈0.57 (clean candidate false-positive 0.86)")
    # False positives on clean DIST candidates
    cleanc = [c for c in tec if c[1] == 0]
    if cleanc:
        fpr = np.mean([1 if verdicts[c[0]] == "FABRICATED" else 0 for c in cleanc])
        print(f"  clean DIST candidates {len(cleanc)}, CoT false positives {int(sum(1 for c in cleanc if verdicts[c[0]]=='FABRICATED'))} ({fpr:.2f})")

    # Full hybrid: non-candidate -> rules; candidate -> rules OR CoT-FABRICATED
    allrows = conn.execute("""SELECT sb.bench_id,sb.label,sb.corruption_family,sb.corrupted_grid,pt.caption
                              FROM scinum_bench sb LEFT JOIN paper_tables pt ON sb.src_table_id=pt.id
                              WHERE sb.dataset_split='test'""").fetchall()
    TP=FP=FN=TN=0; dist_caught=dist_tot=0
    for b, y, fm, cg, cap in allrows:
        g = json.loads(cg); st = StructuredTable(g[0], g[1:], caption=cap or "") if len(g) >= 2 else None
        rp = 1 if (st and D.scan_all(st)) else 0
        isc = b in verdicts
        pred = 1 if rp or (isc and verdicts[b] == "FABRICATED") else 0
        TP += (pred==1 and y==1); FP += (pred==1 and y==0); FN += (pred==0 and y==1); TN += (pred==0 and y==0)
        if fm == "DIST" and y == 1:
            dist_tot += 1; dist_caught += (pred == 1)
    P=TP/(TP+FP) if TP+FP else 0;R=TP/(TP+FN) if TP+FN else 0;F1=2*P*R/(P+R) if P+R else 0
    print("\n=== main set TEST full hybrid ===")
    print(f"  rules         : F1=0.881 R=0.887 FPR=0.190")
    print(f"  7B probe hybrid: F1=0.900 R=0.941 FPR=0.226")
    print(f"  CoT reasoning hybrid: F1={F1:.3f} P={P:.3f} R={R:.3f} FPR={FP/max(1,FP+TN):.3f}")
    print(f"  DIST recall: rules 0.64 -> CoT hybrid {dist_caught/max(1,dist_tot):.2f}")
    conn.close()


if __name__ == "__main__":
    main()
