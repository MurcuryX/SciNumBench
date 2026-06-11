"""
extract_14b_dist.py — extract Qwen2.5-14B decision-token hidden states only for DIST candidate tables.
"""
import os
os.environ.setdefault("HF_HOME", "/data/shared_models/huggingface")
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("TMPDIR", "/data/users/pengminjie/Backup/tmp")
import sqlite3, json, sys, time
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, "/home/pengminjie/Backup/paper/ICDE26/code/bench")
from parser import StructuredTable

MODEL = "Qwen/Qwen2.5-14B-Instruct"
DB = "/home/pengminjie/Backup/paper/ICDE26/data/arxiv_data.db"
OUT = "/home/pengminjie/Backup/paper/ICDE26/data/features/qwen14b_dist.npz"
LAYERS = [24, 36, 48]
MAXTOK = 1100; BATCH = 8
INSTR = ("Below is a table from a scientific paper. Decide whether it contains a "
         "numerical/statistical inconsistency (e.g. a mean impossible for the reported N, "
         "a confidence interval that does not contain its point estimate, percentages "
         "inconsistent with the counts, or a p-value contradicting the confidence interval). "
         "Answer with a single word: Yes or No.")


def table_text(grid):
    return "\n".join(" | ".join(str(c) for c in row) for row in grid)


def is_dist_cand(st):
    return any(st.cells[i][j]["sd"] > 0 and st.cells[i][j]["mean"] > 0
              and st.cells[i][j]["mean"] - 2 * st.cells[i][j]["sd"] < 0
              for (i, j) in st.positions_of("meansd"))


conn = sqlite3.connect(DB)
rows = conn.execute("""SELECT sb.bench_id, sb.label, sb.corruption_family, sb.dataset_split,
                              sb.corrupted_grid, pt.caption
                       FROM scinum_bench sb LEFT JOIN paper_tables pt ON sb.src_table_id=pt.id
                       WHERE sb.dataset_split IN ('train','val','test')""").fetchall()
cands = []
for b, lab, fam, sp, cg, cap in rows:
    g = json.loads(cg)
    if len(g) < 2:
        continue
    st = StructuredTable(g[0], g[1:], caption=cap or "")
    if is_dist_cand(st):
        cands.append((b, lab, fam or "", sp, g))
print(f"[DATA] DIST candidates {len(cands)}", flush=True)

tok = AutoTokenizer.from_pretrained(MODEL)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                             output_hidden_states=True).cuda().eval()
H = model.config.hidden_size
print(f"[LOAD] 14B hidden={H} layers={model.config.num_hidden_layers}", flush=True)
yes_ids = [tok.encode(w, add_special_tokens=False)[0] for w in ["Yes", " Yes", "yes"]]
no_ids = [tok.encode(w, add_special_tokens=False)[0] for w in ["No", " No", "no"]]

N = len(cands)
feats = {L: np.zeros((N, H), dtype=np.float16) for L in LAYERS}
jy = np.zeros(N, np.float32); jn = np.zeros(N, np.float32)
bid = np.array([c[0] for c in cands]); lab = np.array([c[1] for c in cands])
fam = np.array([c[2] for c in cands]); spl = np.array([c[3] for c in cands])
t0 = time.time()
for s in range(0, N, BATCH):
    chunk = cands[s:s + BATCH]
    prompts = [tok.apply_chat_template([{"role": "user", "content": INSTR + "\n\n" + table_text(c[4])}],
                                       tokenize=False, add_generation_prompt=True) for c in chunk]
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=MAXTOK).to("cuda")
    with torch.no_grad():
        out = model(**enc)
    for L in LAYERS:
        feats[L][s:s + len(chunk)] = out.hidden_states[L][:, -1, :].to(torch.float16).cpu().numpy()
    lg = out.logits[:, -1, :].float()
    jy[s:s + len(chunk)] = lg[:, yes_ids].max(1).values.cpu().numpy()
    jn[s:s + len(chunk)] = lg[:, no_ids].max(1).values.cpu().numpy()
    if s % (BATCH * 30) == 0:
        print(f"  {s+len(chunk)}/{N}  {(s+len(chunk))/max(1e-6,time.time()-t0):.1f} tables/s", flush=True)

save = {f"p{L}": feats[L] for L in LAYERS}
save.update(judge_yes=jy, judge_no=jn, bench_id=bid, label=lab, family=fam, split=spl)
np.savez(OUT, **save)
print(f"[DONE] {OUT}, elapsed {(time.time()-t0)/60:.1f} min", flush=True)
conn.close()
