#!/usr/bin/env python
"""
gen_fraud_gemma.py — GENERATOR-OOD fabrication set authored by gemma-3-12B-it.

Reuses the EXACT fabrication task/gate logic from code/bench/llm_fraud_gen.py
(boost-ours: inflate "ours", deflate baselines; same favorable()/gate checks,
same SELECT eligibility, same JSON-grid parse). ONLY differences:
  - generator model -> google/gemma-3-12b-it (gemma chat template; gemma has NO
    system role, so SYS is merged into the user turn).
  - SOURCE TABLES restricted to paper_tables ids NOT already a src_table_id in
    the existing `llm_fraud` table (Qwen used those 4305) -> generator-OOD on
    BOTH author and source tables.
  - writes to a SEPARATE table `llm_fraud_gemma`; NEVER touches `llm_fraud`.
"""
import argparse, json, os, random, re

os.environ.setdefault("HF_HOME", "/data/shared_models/huggingface")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TMPDIR", "/data/users/pengminjie/Backup/tmp")

import sqlite3
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DB = "/home/pengminjie/Backup/paper/ICDE26/data/arxiv_data.db"
MODEL = "google/gemma-3-12b-it"

SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_fraud_gemma (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  src_table_id INTEGER, arxiv_id TEXT, source TEXT, caption TEXT,
  n_rows INTEGER, n_cols INTEGER, method_axis TEXT, our_index INTEGER,
  strategy TEXT, rationale TEXT,
  original_grid TEXT, fabricated_grid TEXT, provenance TEXT, n_cells_changed INTEGER,
  model TEXT, status TEXT, raw_llm TEXT, UNIQUE(src_table_id)
);
"""

SELECT_SQL = """
SELECT id, arxiv_id, source, caption, table_json
FROM paper_tables
WHERE source='arxiv'
  AND ( lower(caption) LIKE '%comparison%' OR lower(caption) LIKE '%ablation%'
     OR lower(caption) LIKE '%baseline%'   OR lower(caption) LIKE '%state-of-the-art%'
     OR lower(caption) LIKE '%compared%'   OR lower(caption) LIKE '%vs.%'
     OR lower(caption) LIKE '%vs %'        OR lower(caption) LIKE '%versus%'
     OR lower(caption) LIKE '%performance%' OR lower(caption) LIKE '%result%'
     OR lower(caption) LIKE '%accuracy%'   OR lower(caption) LIKE '%f1%'
     OR lower(caption) LIKE '%score%'      OR lower(caption) LIKE '%evaluation%'
     OR lower(caption) LIKE '%benchmark%'  OR lower(caption) LIKE '%method%' )
  AND rows BETWEEN 3 AND 14 AND cols BETWEEN 3 AND 9
ORDER BY id
"""

_NUMRE = re.compile(r"(?P<pre>[^\d\-+]*)(?P<num>[-+]?\d+(?:\.\d+)?)(?P<suf>.*)$", re.S)
_TWO_NUM = re.compile(r"^\s*[-+]?\d+(?:\.\d+)?\s+[-+]?\d+")
MAX_REL = 0.08

_METRIC_WORDS = {"latency", "accuracy", "acc", "f1", "precision", "prec", "recall", "rec",
    "error", "err", "loss", "time", "score", "std", "stddev", "mean", "median", "size",
    "rate", "auc", "psnr", "ssim", "lpips", "bleu", "rouge", "throughput", "flops",
    "params", "parameters", "perplexity", "ppl", "mse", "mae", "rmse", "map", "mrr",
    "ndcg", "cost", "speed", "memory", "mem", "fpr", "tpr", "iou", "dice"}
_STRONG_METRIC = ("latency", "accuracy", "precision", "recall", "throughput",
                  "perplexity", "runtime", "inference time", "error rate",
                  "f1 score", "std dev", "std. dev")


def is_numeric_label(s):
    s = s.strip()
    return bool(s) and bool(re.fullmatch(r"[+-]?[\d.,]+\s*[KMBGkmbg]?%?", s))


def looks_like_metric(s):
    w = [t for t in re.sub(r"[^a-z0-9 ]", " ", s.lower()).split() if t]
    return len(w) > 0 and all(t in _METRIC_WORDS for t in w)


def grid_from_json(tj):
    o = json.loads(tj)
    cols = [str(c) for c in o.get("columns", [])]
    data = [[("" if c is None else str(c)) for c in row] for row in o.get("data", [])]
    return [cols] + data


def primary_number(cell):
    s = cell.strip()
    if not s or _TWO_NUM.match(s):
        return None
    m = _NUMRE.match(s)
    if not m:
        return None
    suf = m.group("suf").strip()
    if suf and not re.fullmatch(r"[%\*†‡a-cA-C]?\s*(pp|ms|s|x|×|M|K|B|G|%)?\.?", suf):
        return None
    return m.group("pre"), m.group("num"), m.group("suf")


def decimals(numstr):
    return len(numstr.split(".")[1]) if "." in numstr else 0


def fmt_like(value, numstr):
    d = decimals(numstr)
    return f"{value:.{d}f}" if d > 0 else str(int(round(value)))


def table_is_clean(grid):
    body = [c for row in grid[1:] for c in row]
    if len(body) < 6:
        return False
    nums = sum(1 for c in body if primary_number(c))
    empt = sum(1 for c in body if not c.strip())
    merged = sum(1 for c in body if _TWO_NUM.match(c.strip()))
    if nums == 0 or nums / len(body) < 0.40 or empt / len(body) > 0.35 or merged / max(nums, 1) > 0.20:
        return False
    return True


def gate_structure(grid, axis, oi):
    if axis == "row":
        labels = [grid[r][0].strip() for r in range(1, len(grid)) if grid[r]]
        ours = grid[oi][0].strip() if 0 <= oi < len(grid) and grid[oi] else ""
    else:
        hdr = grid[0]
        labels = [hdr[c].strip() for c in range(1, len(hdr))]
        ours = hdr[oi].strip() if 0 <= oi < len(hdr) else ""
    nlab = [l for l in labels if l]
    if nlab and sum(is_numeric_label(l) for l in nlab) / len(nlab) >= 0.6:
        return False, "numeric_sweep"
    if len(re.findall(r"[A-Za-z]", ours)) < 3:
        return False, "ours_too_few_letters"
    if is_numeric_label(ours):
        return False, "ours_numeric"
    if re.search(r"w/o|without|remov", ours.lower()):
        return False, "ours_ablation_removed"
    if looks_like_metric(ours):
        return False, "ours_is_metric"
    if any(m in ours.lower() for m in _STRONG_METRIC):
        return False, "ours_metric_substr"
    return True, "ok"


def arrow_polarity(grid, axis):
    pol = {}
    if axis == "row":
        for c, h in enumerate(grid[0]):
            if "↓" in h:
                pol[c] = "lower"
            elif "↑" in h:
                pol[c] = "higher"
    else:
        for r in range(len(grid)):
            lab = grid[r][0] if grid[r] else ""
            if "↓" in lab:
                pol[r] = "lower"
            elif "↑" in lab:
                pol[r] = "higher"
    return pol


def favorable(is_ours, pol, ov, nv):
    if pol not in ("higher", "lower"):
        return False
    up = nv > ov
    if is_ours:
        return (pol == "higher" and up) or (pol == "lower" and not up)
    return (pol == "higher" and not up) or (pol == "lower" and up)


def apply_changes(grid, axis, oi, pol, changes):
    fab = [list(row) for row in grid]
    prov = []
    seen = set()
    for ch in changes:
        try:
            r, c = int(ch["row"]), int(ch["col"])
            nv = float(ch["new_value"])
        except Exception:
            continue
        if r <= 0 or r >= len(grid) or c < 0 or c >= len(grid[r]) or (r, c) in seen:
            continue
        pn = primary_number(grid[r][c])
        if pn is None:
            continue
        pre, numstr, suf = pn
        ov = float(numstr)
        if axis == "row":
            is_ours = (r == oi); p = pol.get(c, "na")
        else:
            is_ours = (c == oi); p = pol.get(r, "na")
        if not favorable(is_ours, p, ov, nv):
            continue
        if ov != 0 and abs(nv - ov) / abs(ov) > MAX_REL:
            continue
        if nv < 0 or (("%" in suf) and ov <= 100.0 and nv > 100.0):
            continue
        new_cell = pre + fmt_like(nv, numstr) + suf
        if new_cell.strip() == grid[r][c].strip():
            continue
        fab[r][c] = new_cell
        prov.append({"r": r, "c": c, "orig": grid[r][c], "new": new_cell,
                     "is_ours": is_ours, "polarity": p})
        seen.add((r, c))
    if not prov:
        return None, "no_valid_change"
    return fab, prov


def render_grid(grid):
    out = []
    for r, row in enumerate(grid):
        tag = "row 0 (header)" if r == 0 else f"row {r}"
        out.append(f"{tag}: " + " | ".join(f"[{c}] {v}" for c, v in enumerate(row)))
    return "\n".join(out)


SYS = ("You simulate a dishonest researcher who fabricates their OWN results table to make "
       "their proposed method look stronger. This builds a fraud-detection benchmark; the "
       "table you receive is a real published table.")

USER_TMPL = """Below is a table from a paper.

CAPTION: {caption}

TABLE (row 0 is the header; cells written as [colindex] value):
{rendered}

STEP 1 — GATE. Decide if this is a table where the AUTHORS compare their OWN proposed method/model against alternatives, AND one specific row or column is clearly the authors' own method ("ours": often named after the paper, marked "Ours"/"Proposed", or the obvious best). If it is NOT such a comparison (e.g. descriptive statistics, a dataset/data table, or no identifiable "ours"), respond exactly: {{"is_comparison": false}}

STEP 2 — If it IS such a comparison, report structure and polarity:
  - "method_axis": "row" if each method is a ROW, "col" if each method is a COLUMN.
  - "our_index": index of the authors' own method (row index if method_axis="row", else column index).
  - "polarity": for every METRIC, whether higher or lower is better. If method_axis="row", key by metric COLUMN index; if "col", key by metric ROW index. Use "higher", "lower", or "na". (Accuracy/F1/score=higher; Error/Loss/Latency/Time/Params/FLOPs/perplexity/std=lower.)

STEP 3 — Fabricate 1 to 4 NUMERIC edits, each clearly FAVORING the authors (improve their own method, and/or worsen competing baselines). Subtle, realistic nudges only; keep values plausible.

Respond with STRICT JSON only:
{{"is_comparison": true, "method_axis": "row|col", "our_index": <int>, "polarity": {{"<idx>": "higher|lower|na"}}, "strategy": "boost_ours|sink_baseline|both", "rationale": "<one sentence>", "changes": [{{"row": <int>, "col": <int>, "new_value": <number>}}]}}"""


def extract_json(text):
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except Exception:
                    return None
    return None


def store(con, rec):
    con.execute(
        "INSERT OR REPLACE INTO llm_fraud_gemma(src_table_id,arxiv_id,source,caption,n_rows,n_cols,"
        "method_axis,our_index,strategy,rationale,original_grid,fabricated_grid,provenance,"
        "n_cells_changed,model,status,raw_llm) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rec)


def gen_batch(model, tok, prompts, max_new=320):
    # gemma-3 has NO system role -> merge SYS into the user turn.
    texts = [tok.apply_chat_template(
        [{"role": "user", "content": SYS + "\n\n" + p}],
        tokenize=False, add_generation_prompt=True) for p in prompts]
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=3072).to("cuda:0")
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=True,
                             temperature=0.7, top_p=0.9, pad_token_id=tok.pad_token_id)
    inlen = enc.input_ids.shape[1]
    return [tok.decode(out[i][inlen:], skip_special_tokens=True) for i in range(out.shape[0])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    con = sqlite3.connect(DB, timeout=120)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA busy_timeout=120000")
    con.executescript(SCHEMA)
    if args.rebuild:
        con.execute("DROP TABLE IF EXISTS llm_fraud_gemma")
        con.executescript(SCHEMA)
        con.commit()

    # EXCLUDE every src_table_id already used by Qwen in llm_fraud (generator-OOD
    # on source tables too), plus anything already done in llm_fraud_gemma.
    qwen_used = {r[0] for r in con.execute("SELECT src_table_id FROM llm_fraud")}
    gemma_done = {r[0] for r in con.execute("SELECT src_table_id FROM llm_fraud_gemma")}
    exclude = qwen_used | gemma_done
    print(f"[EXCLUDE] qwen_used={len(qwen_used)} gemma_done={len(gemma_done)}", flush=True)

    rows = [r for r in con.execute(SELECT_SQL).fetchall() if r[0] not in exclude]
    random.Random(args.seed).shuffle(rows)
    print(f"[SELECT] OOD candidates {len(rows)} (disjoint from llm_fraud), target ok={args.n}, batch={args.batch}", flush=True)

    print(f"[LOAD] {MODEL} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda:0")
    model.eval(); torch.manual_seed(args.seed)

    ctr = {"ok": 0, "gate": 0, "fail": 0, "dirty": 0}
    fr = {}
    seen = 0

    def process(tid, aid, src, caption, grid, gen):
        obj = extract_json(gen)
        if not obj:
            ctr["fail"] += 1; fr["json"] = fr.get("json", 0) + 1; return
        if not obj.get("is_comparison"):
            ctr["gate"] += 1; return
        axis = obj.get("method_axis")
        try:
            oi = int(obj.get("our_index"))
        except Exception:
            ctr["fail"] += 1; fr["bad_our_index"] = fr.get("bad_our_index", 0) + 1; return
        if axis not in ("row", "col") or not (0 < oi < (len(grid) if axis == "row" else len(grid[0]))):
            ctr["fail"] += 1; fr["bad_axis_idx"] = fr.get("bad_axis_idx", 0) + 1; return
        gok, greason = gate_structure(grid, axis, oi)
        if not gok:
            ctr["gate"] += 1; fr[greason] = fr.get(greason, 0) + 1; return
        pol = {}
        for k, v in (obj.get("polarity") or {}).items():
            try:
                pol[int(k)] = v
            except Exception:
                pass
        pol.update(arrow_polarity(grid, axis))
        res = apply_changes(grid, axis, oi, pol, obj.get("changes", []))
        if res[0] is None:
            ctr["fail"] += 1; fr[res[1]] = fr.get(res[1], 0) + 1; return
        fab, prov = res
        ctr["ok"] += 1
        store(con, (tid, aid, src, caption, len(grid), len(grid[0]),
                    axis, oi, obj.get("strategy"), obj.get("rationale"),
                    json.dumps(grid, ensure_ascii=False), json.dumps(fab, ensure_ascii=False),
                    json.dumps(prov, ensure_ascii=False), len(prov), MODEL, "ok", gen[:1500]))

    buf = []

    def flush():
        if not buf:
            return
        gens = gen_batch(model, tok, [b[5] for b in buf])
        for (tid, aid, src, caption, grid, _), gen in zip(buf, gens):
            try:
                process(tid, aid, src, caption, grid, gen)
            except Exception as e:
                ctr["fail"] += 1; fr["exc"] = fr.get("exc", 0) + 1
                print(f"  [EXC] tid={tid}: {e}", flush=True)
        con.commit()
        buf.clear()

    for (tid, aid, src, caption, tj) in rows:
        if ctr["ok"] >= args.n:
            break
        seen += 1
        try:
            grid = grid_from_json(tj)
        except Exception:
            ctr["dirty"] += 1; continue
        if len(grid) < 2 or not table_is_clean(grid):
            ctr["dirty"] += 1; continue
        prompt = USER_TMPL.format(caption=(caption or "")[:400], rendered=render_grid(grid))
        buf.append((tid, aid, src, caption, grid, prompt))
        if len(buf) >= args.batch:
            flush()
            print(f"  ... seen={seen} ok={ctr['ok']} gate={ctr['gate']} fail={ctr['fail']} dirty={ctr['dirty']}", flush=True)
    flush()

    print(f"\n[DONE] ok={ctr['ok']} gate={ctr['gate']} fail={ctr['fail']} dirty={ctr['dirty']} seen={seen} fail_reasons={fr}", flush=True)
    con.close()


if __name__ == "__main__":
    main()
