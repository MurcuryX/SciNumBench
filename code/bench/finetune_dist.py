"""
finetune_dist.py — LoRA fine-tune Qwen2.5-7B for DIST discrimination.

Train on DIST candidates to learn "fabricated large SD (FABRICATED) vs
legitimate large variance (PLAUSIBLE)".
Hypothesis: fine-tuning can learn each variable's typical variance from
samples (age SD~2-5, fabricated ~11-13), breaking the candidate-layer
ceiling of the frozen probe/CoT (P~0.5).
Compliance: base shared read-only, LoRA adapter stored in ~/Backup, system disk not written.
"""
import os
os.environ.setdefault("HF_HOME", "/data/shared_models/huggingface")
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("TMPDIR", "/data/users/pengminjie/Backup/tmp")
import sqlite3, json, sys, time, re, random
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
sys.path.insert(0, "/home/pengminjie/Backup/paper/ICDE26/code/bench")
from parser import StructuredTable
import detectors as D

MODEL = "Qwen/Qwen2.5-7B-Instruct"
DB = "/home/pengminjie/Backup/paper/ICDE26/data/arxiv_data.db"
ADAPTER = "/home/pengminjie/Backup/paper/ICDE26/models/lora_dist"
EPOCHS = 3; BS = 4; ACC = 2; LR = 1e-4; MAXLEN = 1100
INSTR = ("You audit a scientific table for fabricated statistics (values are mean ± SD). "
         "For the flagged variable(s), mean − 2×SD < 0. If the variable is inherently "
         "non-negative and roughly normal (age, weight, height, BMI, time, score, "
         "concentration...), such a large SD is implausible → FABRICATED. If it can be "
         "negative or strongly skewed (change score, difference, correlation...) → PLAUSIBLE.")


def ttext(g): return "\n".join(" | ".join(str(c) for c in r) for r in g)
def flags(st):
    o = []
    for (i, j) in st.positions_of("meansd"):
        c = st.cells[i][j]
        if c["sd"] > 0 and c["mean"] > 0 and c["mean"] - 2 * c["sd"] < 0:
            rl = st.data[i][0] if st.data[i] else ""; cl = st.columns[j] if j < len(st.columns) else ""
            o.append(f'- "{(rl+" "+cl).strip()}": mean={c["mean"]}, SD={c["sd"]}')
    return o, bool(o)


def load_split(conn, splits):
    rs = conn.execute(f"""SELECT sb.bench_id,sb.label,sb.corruption_family,sb.dataset_split,sb.corrupted_grid,pt.caption
                          FROM scinum_bench sb LEFT JOIN paper_tables pt ON sb.src_table_id=pt.id
                          WHERE sb.dataset_split IN ({','.join('?'*len(splits))})""", splits).fetchall()
    out = []
    for b, y, fam, sp, cg, cap in rs:
        g = json.loads(cg)
        if len(g) < 2: continue
        st = StructuredTable(g[0], g[1:], caption=cap or "")
        fl, ok = flags(st)
        if ok:
            out.append(dict(b=b, y=y, fam=fam or "", sp=sp, st=st, g=g,
                            prompt=INSTR + "\n\nTable:\n" + ttext(g) + "\n\nFlagged:\n" + "\n".join(fl)))
    return out


def main():
    conn = sqlite3.connect(DB)
    tr = load_split(conn, ["train"]); te = load_split(conn, ["test"])
    print(f"[DATA] train candidates {len(tr)} / test candidates {len(te)}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "right"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).cuda()
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora); model.print_trainable_parameters()

    def build(ex):
        verdict = "FABRICATED" if ex["fam"] == "DIST" else "PLAUSIBLE"
        pstr = tok.apply_chat_template([{"role": "user", "content": ex["prompt"]}],
                                       tokenize=False, add_generation_prompt=True)
        pids = tok.encode(pstr, add_special_tokens=False)
        tids = tok.encode("VERDICT: " + verdict, add_special_tokens=False) + [tok.eos_token_id]
        ids = (pids + tids)[:MAXLEN]
        lab = ([-100] * len(pids) + tids)[:MAXLEN]
        return ids, lab

    examples = [build(e) for e in tr]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    model.train()
    rng = random.Random(42)
    step = 0; t0 = time.time()
    for ep in range(EPOCHS):
        order = list(range(len(examples))); rng.shuffle(order)
        for bi in range(0, len(order), BS):
            batch = [examples[k] for k in order[bi:bi + BS]]
            mx = max(len(x[0]) for x in batch)
            ii = torch.full((len(batch), mx), tok.pad_token_id, dtype=torch.long)
            ll = torch.full((len(batch), mx), -100, dtype=torch.long)
            am = torch.zeros((len(batch), mx), dtype=torch.long)
            for r, (ids, lab) in enumerate(batch):
                ii[r, :len(ids)] = torch.tensor(ids); ll[r, :len(lab)] = torch.tensor(lab); am[r, :len(ids)] = 1
            out = model(input_ids=ii.cuda(), attention_mask=am.cuda(), labels=ll.cuda())
            (out.loss / ACC).backward(); step += 1
            if step % ACC == 0:
                opt.step(); opt.zero_grad()
            if step % 40 == 0:
                print(f"  ep{ep} step{step} loss={out.loss.item():.3f} {(time.time()-t0)/60:.1f}min", flush=True)
    os.makedirs(ADAPTER, exist_ok=True); model.save_pretrained(ADAPTER)

    # Eval: generate verdict for test candidates
    model.eval(); tok.padding_side = "left"
    verds = {}
    for s in range(0, len(te), 8):
        ch = te[s:s + 8]
        prompts = [tok.apply_chat_template([{"role": "user", "content": e["prompt"]}], tokenize=False, add_generation_prompt=True) for e in ch]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=MAXLEN).to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=12, do_sample=False, pad_token_id=tok.pad_token_id)
        for k, e in enumerate(ch):
            t = tok.decode(gen[k][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            verds[e["b"]] = "FABRICATED" if "FABRICAT" in t.upper() else "PLAUSIBLE"
    json.dump(verds, open("/home/pengminjie/Backup/paper/ICDE26/data/features/ft_verdicts.json", "w"))

    yc = np.array([1 if e["fam"] == "DIST" else 0 for e in te])
    pc = np.array([1 if verds[e["b"]] == "FABRICATED" else 0 for e in te])
    tp = int(((pc==1)&(yc==1)).sum()); fp = int(((pc==1)&(yc==0)).sum()); fn = int(((pc==0)&(yc==1)).sum())
    P = tp/(tp+fp) if tp+fp else 0; R = tp/(tp+fn) if tp+fn else 0
    print(f"\n[candidate-layer test] fine-tune: P={P:.3f} R={R:.3f}  (vs probe P0.57/CoT P0.48)")
    cleanc = [e for e in te if e["y"] == 0]
    print(f"  clean DIST candidates {len(cleanc)}, fine-tune false positives {int(sum(1 for e in cleanc if verds[e['b']]=='FABRICATED'))} "
          f"({np.mean([1 if verds[e['b']]=='FABRICATED' else 0 for e in cleanc]):.2f})  (vs probe0.86/CoT0.73)")

    # Full mixture
    allr = conn.execute("""SELECT sb.bench_id,sb.label,sb.corruption_family,sb.corrupted_grid,pt.caption
                           FROM scinum_bench sb LEFT JOIN paper_tables pt ON sb.src_table_id=pt.id WHERE sb.dataset_split='test'""").fetchall()
    TP=FP=FN=TN=0; dc=dt=0
    for b, y, fm, cg, cap in allr:
        g = json.loads(cg); st = StructuredTable(g[0], g[1:], caption=cap or "") if len(g) >= 2 else None
        rp = 1 if (st and D.scan_all(st)) else 0
        pred = 1 if rp or (b in verds and verds[b] == "FABRICATED") else 0
        TP+=(pred==1 and y==1);FP+=(pred==1 and y==0);FN+=(pred==0 and y==1);TN+=(pred==0 and y==0)
        if fm=="DIST" and y==1: dt+=1; dc+=(pred==1)
    P=TP/(TP+FP) if TP+FP else 0;R=TP/(TP+FN) if TP+FN else 0;F1=2*P*R/(P+R) if P+R else 0
    print("\n=== Main set TEST full mixture ===")
    print(f"  rules F1=0.881 | probe mixture F1=0.900 | CoT mixture F1=0.899")
    print(f"  fine-tune mixture: F1={F1:.3f} P={P:.3f} R={R:.3f} FPR={FP/max(1,FP+TN):.3f}")
    print(f"  DIST recall: rules0.64 -> fine-tune mixture {dc/max(1,dt):.2f}")
    conn.close()


if __name__ == "__main__":
    main()
