"""
build_llmfraud_eval.py — assemble the llm_fraud table (boost-ours fabrication)
into a labeled training-free baseline evaluation set.

Data source: llm_fraud (status='ok'). Fabrication nature: in method-comparison /
ablation tables, the authors' own method (our_index) metrics are nudged in a
favorable direction (polarity higher/lower), or baselines are made worse; after
editing the table remains self-consistent and the numbers stay plausible.

Construction protocol:
  - Fixed seed=42, select 300 distinct src_table_id (status='ok').
  - Each src_table_id yields one pair of samples:
      positive label=1: fabricated_grid;
      negative label=0: corresponding original_grid (true unedited table = hard negative).
    => 600 records, balanced.
  - text = clean_serialize.serialize(grid, caption, provenance) (clean + serialize).
    Negatives are not passed provenance (the original table has no fabricated span).

Output (written to results/):
  llmfraud_eval.jsonl       : all fields (eval_id, src_table_id, label, strategy, method_axis,
                              our_index, polarity{dict}, provenance (positives only), text).
  llmfraud_judge_input.jsonl: desensitized, only {eval_id, text}. Never contains label/strategy/our_index/provenance.

polarity field protocol: for spans with is_ours==True in the table provenance, aggregate
polarity counts, e.g. {"higher": n, "lower": m}; positive and negative of the same table
share the same polarity dict (describing fabrication intent; the negative is a reference and
still carries polarity for analysis, but negative provenance=None does not leak the edited cells).
"""
import os
import sys
import json
import random
import sqlite3
import statistics
from collections import Counter

ROOT = "/home/pengminjie/Backup/paper/ICDE26"
sys.path.insert(0, ROOT + "/code/bench")
from clean_serialize import serialize          # noqa: E402

DB = ROOT + "/data/arxiv_data.db"
RESULTS = ROOT + "/results"
OUT_FULL = RESULTS + "/llmfraud_eval.jsonl"
OUT_JUDGE = RESULTS + "/llmfraud_judge_input.jsonl"

N_TABLES = 300
SEED = 42


def polarity_dict(provenance):
    """Aggregate polarity counts over spans with is_ours==True -> dict {polarity: count}."""
    d = Counter()
    for span in provenance or []:
        if span.get("is_ours") is True:
            pol = span.get("polarity")
            if pol:
                d[pol] += 1
    return dict(d)


def main():
    os.makedirs(RESULTS, exist_ok=True)
    conn = sqlite3.connect(DB)
    # Fetch src_table_id for all ok rows (each src_table_id is unique within the ok subset)
    ids = [r[0] for r in conn.execute(
        "SELECT src_table_id FROM llm_fraud WHERE status='ok' ORDER BY src_table_id").fetchall()]
    rng = random.Random(SEED)
    chosen = sorted(rng.sample(ids, N_TABLES))

    qmarks = ",".join("?" * len(chosen))
    rows = conn.execute(
        f"""SELECT src_table_id, strategy, method_axis, our_index, caption,
                   original_grid, fabricated_grid, provenance
            FROM llm_fraud
            WHERE status='ok' AND src_table_id IN ({qmarks})
            ORDER BY src_table_id""", chosen).fetchall()
    conn.close()

    full_recs = []
    judge_recs = []
    eval_id = 0
    for (src_id, strategy, axis, our_index, caption,
         orig_g, fab_g, prov) in rows:
        og = json.loads(orig_g)
        fg = json.loads(fab_g)
        provenance = json.loads(prov) if prov else []
        pol = polarity_dict(provenance)

        # Positive: fabricated, label=1, with provenance (protects edited rows during serialization)
        eval_id += 1
        pos_text = serialize(fg, caption or "", provenance=provenance)
        full_recs.append({
            "eval_id": eval_id,
            "src_table_id": int(src_id),
            "label": 1,
            "strategy": strategy,
            "method_axis": axis,
            "our_index": int(our_index),
            "polarity": pol,
            "provenance": provenance,
            "text": pos_text,
        })
        judge_recs.append({"eval_id": eval_id, "text": pos_text})

        # Negative: original (truly unedited), label=0, hard negative, no provenance passed
        eval_id += 1
        neg_text = serialize(og, caption or "")
        full_recs.append({
            "eval_id": eval_id,
            "src_table_id": int(src_id),
            "label": 0,
            "strategy": strategy,
            "method_axis": axis,
            "our_index": int(our_index),
            "polarity": pol,
            "provenance": None,
            "text": neg_text,
        })
        judge_recs.append({"eval_id": eval_id, "text": neg_text})

    with open(OUT_FULL, "w") as f:
        for r in full_recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(OUT_JUDGE, "w") as f:
        for r in judge_recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Statistics
    n_pos = sum(1 for r in full_recs if r["label"] == 1)
    n_neg = sum(1 for r in full_recs if r["label"] == 0)
    n_src = len({r["src_table_id"] for r in full_recs})
    strat = Counter(r["strategy"] for r in full_recs if r["label"] == 1)
    lens = [len(r["text"]) for r in full_recs]
    med = statistics.median(lens)
    mx = max(lens)
    tok_est_max = mx / 4

    print(f"[SAVE] {OUT_FULL}  ({len(full_recs)} lines)")
    print(f"[SAVE] {OUT_JUDGE} ({len(judge_recs)} lines)")
    print(f"  pos(label=1)={n_pos}  neg(label=0)={n_neg}  distinct src_table_id={n_src}")
    print(f"  strategy dist (pos): {dict(strat)}")
    print(f"  text len chars: median={med} max={mx}  (est tokens max={tok_est_max:.0f})")
    print(f"  text over-8k-token? {'YES !!!' if tok_est_max > 8000 else 'no (ok)'}")

    # Desensitization self-check: judge input fields must be only eval_id/text
    bad = set()
    for r in judge_recs:
        bad |= (set(r.keys()) - {"eval_id", "text"})
    print(f"  judge_input fields = {sorted(set().union(*(r.keys() for r in judge_recs)))} "
          f"-> {'LEAK !!!' if bad else 'desensitized OK'}")


if __name__ == "__main__":
    main()
