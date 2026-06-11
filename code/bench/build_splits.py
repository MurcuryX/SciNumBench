"""
build_splits.py — Plan B Step 1: rebuild anti-leak train/test splits.

Pairing convention follows build_llmfraud_eval.py:
  Source = llm_fraud (status='ok', 3518 rows, each src_table_id unique).
  Each src_table_id yields one content-matched pair:
      positive label=1: fabricated_grid + provenance  (fabricated table)
      negative label=0: original_grid   (honest un-tampered version of the SAME table = hard negative)
  text = clean_serialize.serialize(grid, caption, provenance=...)  (aligned with eval set; negatives carry no provenance).

Anti-leak:
  (a) split by src_table_id; train/test src_table_id sets are DISJOINT; a (pos,neg) pair always lands in the SAME split.
  (b) global shuffle (seed=42).
  (c) re-key: every emitted example gets an opaque example_id (sha1-derived); the model-facing
      record carries NO src_table_id / arxiv_id / label / provenance / strategy etc.;
      identity mapping is stored separately in splits/mapping.jsonl for our own bookkeeping.

Outputs (data/splits/):
  train.jsonl, test.jsonl             : full fields (incl label/provenance/src_table_id) for our training/eval.
  train_model.jsonl, test_model.jsonl : desensitized, only {example_id, text} for feeding the model (zero leak).
  mapping.jsonl                       : {example_id, split, label, src_table_id, arxiv_id} bookkeeping map.
"""
import os
import sys
import json
import random
import hashlib
import sqlite3
from collections import Counter

ROOT = "/home/pengminjie/Backup/paper/ICDE26"
sys.path.insert(0, ROOT + "/code/bench")
from clean_serialize import serialize          # noqa: E402

DB = ROOT + "/data/arxiv_data.db"
OUTDIR = ROOT + "/data/splits"
SEED = 42
N_TEST_IDS = 1000     # ~1000 src_table_id -> ~2000 test rows; remaining ~2518 ids -> train
STATUS_OK = "ok"

FORBIDDEN_MODEL_FIELDS = {
    "src_table_id", "arxiv_id", "label", "provenance", "strategy",
    "method_axis", "our_index", "source", "rationale", "original_grid",
    "fabricated_grid", "n_cells_changed", "model", "raw_llm",
}


def make_example_id(salt, src_id, label):
    """Opaque id derived from (seed, src_table_id, label); no reversible plaintext link to src_table_id."""
    h = hashlib.sha1("{}:{}:{}".format(salt, src_id, label).encode()).hexdigest()
    return "ex_" + h[:16]


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    conn = sqlite3.connect(DB)
    rows = conn.execute(
        "SELECT src_table_id, arxiv_id, caption, original_grid, fabricated_grid, provenance "
        "FROM llm_fraud WHERE status=? ORDER BY src_table_id", (STATUS_OK,)).fetchall()
    conn.close()

    ids = sorted({int(r[0]) for r in rows})
    assert len(ids) == len(rows), "src_table_id not unique; pairing convention broken"

    rng = random.Random(SEED)
    shuffled_ids = ids[:]
    rng.shuffle(shuffled_ids)
    test_ids = set(shuffled_ids[:N_TEST_IDS])
    train_ids = set(shuffled_ids[N_TEST_IDS:])
    assert test_ids.isdisjoint(train_ids)

    full = {"train": [], "test": []}
    model = {"train": [], "test": []}
    mapping = []

    for (src_id, arxiv_id, caption, orig_g, fab_g, prov) in rows:
        src_id = int(src_id)
        split = "test" if src_id in test_ids else "train"
        og = json.loads(orig_g)
        fg = json.loads(fab_g)
        provenance = json.loads(prov) if prov else []

        for label, grid, prov_use in ((1, fg, provenance), (0, og, None)):
            ex_id = make_example_id(SEED, src_id, label)
            text = serialize(grid, caption or "", provenance=prov_use)
            full[split].append({
                "example_id": ex_id,
                "label": label,
                "text": text,
                "src_table_id": src_id,
                "arxiv_id": arxiv_id,
                "provenance": prov_use,
            })
            model[split].append({"example_id": ex_id, "text": text})
            mapping.append({
                "example_id": ex_id, "split": split, "label": label,
                "src_table_id": src_id, "arxiv_id": arxiv_id,
            })

    # global shuffle (scramble pos/neg and src order); same rng order keeps full/model rows aligned
    for split in ("train", "test"):
        order = list(range(len(full[split])))
        rng.shuffle(order)
        full[split] = [full[split][i] for i in order]
        model[split] = [model[split][i] for i in order]

    def dump(path, recs):
        with open(path, "w") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dump(OUTDIR + "/train.jsonl", full["train"])
    dump(OUTDIR + "/test.jsonl", full["test"])
    dump(OUTDIR + "/train_model.jsonl", model["train"])
    dump(OUTDIR + "/test_model.jsonl", model["test"])
    dump(OUTDIR + "/mapping.jsonl", mapping)

    # ── verify + report ──
    def counts(recs):
        c = Counter(r["label"] for r in recs)
        sids = {r["src_table_id"] for r in recs}
        return c[1], c[0], len(sids)

    tr_pos, tr_neg, tr_src = counts(full["train"])
    te_pos, te_neg, te_src = counts(full["test"])
    inter = {r["src_table_id"] for r in full["train"]} & {r["src_table_id"] for r in full["test"]}

    leak = set()
    for split in ("train", "test"):
        for r in model[split]:
            leak |= (set(r.keys()) & FORBIDDEN_MODEL_FIELDS)
            leak |= (set(r.keys()) - {"example_id", "text"})
    all_ids = [r["example_id"] for r in mapping]
    dup = len(all_ids) - len(set(all_ids))

    verdict = "LEAK!!!" if leak else "clean OK"
    print("=== build_splits report ===")
    print("OUTDIR={}".format(OUTDIR))
    print("TRAIN: rows={}  pos={} neg={}  src_ids={}".format(len(full["train"]), tr_pos, tr_neg, tr_src))
    print("TEST : rows={}  pos={} neg={}  src_ids={}".format(len(full["test"]), te_pos, te_neg, te_src))
    print("src_table_id overlap(train INTERSECT test) = {}  (MUST be 0)".format(len(inter)))
    print("total src_ids = {}  (expect {})".format(tr_src + te_src, len(ids)))
    print("model-facing forbidden/extra fields = {}  -> {}".format(sorted(leak), verdict))
    print("example_id duplicates = {}  (MUST be 0)".format(dup))
    print("files: train.jsonl test.jsonl train_model.jsonl test_model.jsonl mapping.jsonl")


if __name__ == "__main__":
    main()
