#!/usr/bin/env python
"""
build_genood_split.py — Build the generator-OOD split from llm_fraud_gemma.

Same pairing + serialization + re-key scheme as build_splits.py:
  each gemma src_table_id -> pos(label1)=fabricated_grid+provenance, neg(label0)=original_grid.
  example_id = "ex_" + sha1(f"{SEED}:{src_table_id}:{label}").hexdigest()[:16]
  text = clean_serialize.serialize(grid, caption, provenance=...).
Outputs:
  data/splits/gen_ood_model.jsonl   {example_id, text}      (model-facing)
  data/splits/gen_ood_mapping.jsonl {example_id, label, src_table_id, arxiv_id}
"""
import os, sys, json, hashlib, sqlite3
from collections import Counter

ROOT = "/home/pengminjie/Backup/paper/ICDE26"
sys.path.insert(0, ROOT + "/code/bench")
from clean_serialize import serialize  # noqa: E402

DB = ROOT + "/data/arxiv_data.db"
OUTDIR = ROOT + "/data/splits"
SEED = 42


def make_example_id(salt, src_id, label):
    h = hashlib.sha1("{}:{}:{}".format(salt, src_id, label).encode()).hexdigest()
    return "ex_" + h[:16]


def main():
    con = sqlite3.connect(DB)
    rows = con.execute(
        "SELECT src_table_id, arxiv_id, caption, original_grid, fabricated_grid, provenance "
        "FROM llm_fraud_gemma WHERE status='ok' ORDER BY src_table_id").fetchall()

    # disjointness assertions vs Qwen train/test (llm_fraud) and the trained splits
    qwen_src = {int(r[0]) for r in con.execute("SELECT src_table_id FROM llm_fraud")}
    mapping_src = set()
    with open(OUTDIR + "/mapping.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                mapping_src.add(int(json.loads(line)["src_table_id"]))
    con.close()

    gemma_src = {int(r[0]) for r in rows}
    assert gemma_src.isdisjoint(qwen_src), "gemma src overlaps llm_fraud (Qwen)!"
    assert gemma_src.isdisjoint(mapping_src), "gemma src overlaps trained splits!"

    model_recs, mapping = [], []
    for (src_id, arxiv_id, caption, orig_g, fab_g, prov) in rows:
        src_id = int(src_id)
        og = json.loads(orig_g)
        fg = json.loads(fab_g)
        provenance = json.loads(prov) if prov else []
        for label, grid, prov_use in ((1, fg, provenance), (0, og, None)):
            ex_id = make_example_id(SEED, src_id, label)
            text = serialize(grid, caption or "", provenance=prov_use)
            model_recs.append({"example_id": ex_id, "text": text})
            mapping.append({"example_id": ex_id, "label": label,
                            "src_table_id": src_id, "arxiv_id": arxiv_id})

    with open(OUTDIR + "/gen_ood_model.jsonl", "w") as f:
        for r in model_recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(OUTDIR + "/gen_ood_mapping.jsonl", "w") as f:
        for r in mapping:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    c = Counter(r["label"] for r in mapping)
    eids = [r["example_id"] for r in mapping]
    dup = len(eids) - len(set(eids))
    print("=== build_genood_split report ===")
    print("gemma src_tables (ok)      = {}".format(len(gemma_src)))
    print("examples emitted           = {}  (pos={} neg={})".format(len(mapping), c[1], c[0]))
    print("example_id duplicates      = {}  (MUST be 0)".format(dup))
    print("src disjoint vs Qwen       = {}".format(gemma_src.isdisjoint(qwen_src)))
    print("src disjoint vs trained    = {}".format(gemma_src.isdisjoint(mapping_src)))
    print("files: gen_ood_model.jsonl  gen_ood_mapping.jsonl")


if __name__ == "__main__":
    main()
