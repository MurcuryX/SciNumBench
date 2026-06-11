"""Build the +787 NEW train examples, mirroring build_splits.py exactly.
Outputs:
  data/splits/new_train_model.jsonl   {example_id, text}  (all TRAIN, new src only)
  data/splits/new_train_full.jsonl     full fields for bookkeeping
  data/splits/mapping_v2.jsonl         old mapping + new rows (split="train")
Does NOT touch original splits/mapping.jsonl.
"""
import os, sys, json, hashlib, sqlite3
from collections import Counter
ROOT = "/home/pengminjie/Backup/paper/ICDE26"
sys.path.insert(0, ROOT + "/code/bench")
from clean_serialize import serialize
DB = ROOT + "/data/arxiv_data.db"
OUTDIR = ROOT + "/data/splits"
SEED = 42
STATUS_OK = "ok"

def make_example_id(salt, src_id, label):
    h = hashlib.sha1("{}:{}:{}".format(salt, src_id, label).encode()).hexdigest()
    return "ex_" + h[:16]

# old mapping src ids + collect rows for rewrite
old_rows = []
old_src = set()
old_test_src = set()
with open(OUTDIR + "/mapping.jsonl") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        old_rows.append(r)
        old_src.add(int(r["src_table_id"]))
        if r["split"] == "test":
            old_test_src.add(int(r["src_table_id"]))

con = sqlite3.connect(DB)
rows = con.execute(
    "SELECT src_table_id, arxiv_id, caption, original_grid, fabricated_grid, provenance "
    "FROM llm_fraud WHERE status=? ORDER BY src_table_id", (STATUS_OK,)).fetchall()
con.close()

new_ids = sorted({int(r[0]) for r in rows} - old_src)
assert all(i not in old_test_src for i in new_ids), "NEW id leaked into OLD test!"
print("n_new_src =", len(new_ids))

model_recs = []
full_recs = []
new_map = []
parse_drops = []
seen_eid = set()
for (src_id, arxiv_id, caption, orig_g, fab_g, prov) in rows:
    src_id = int(src_id)
    if src_id not in set(new_ids):
        continue
    try:
        og = json.loads(orig_g)
        fg = json.loads(fab_g)
        provenance = json.loads(prov) if prov else []
    except Exception as e:
        parse_drops.append((src_id, str(e)))
        continue
    for label, grid, prov_use in ((1, fg, provenance), (0, og, None)):
        ex_id = make_example_id(SEED, src_id, label)
        assert ex_id not in seen_eid, "dup example_id " + ex_id
        seen_eid.add(ex_id)
        text = serialize(grid, caption or "", provenance=prov_use)
        model_recs.append({"example_id": ex_id, "text": text})
        full_recs.append({"example_id": ex_id, "label": label, "text": text,
                          "src_table_id": src_id, "arxiv_id": arxiv_id,
                          "provenance": prov_use})
        new_map.append({"example_id": ex_id, "split": "train", "label": label,
                        "src_table_id": src_id, "arxiv_id": arxiv_id})

def dump(path, recs):
    with open(path, "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

dump(OUTDIR + "/new_train_model.jsonl", model_recs)
dump(OUTDIR + "/new_train_full.jsonl", full_recs)
# mapping_v2 = old + new
dump(OUTDIR + "/mapping_v2.jsonl", old_rows + new_map)

# verify no eid collision between old and new mappings
old_eids = {r["example_id"] for r in old_rows}
collide = old_eids & seen_eid
print("new model rows =", len(model_recs), "(expect", 2 * len(new_ids), "minus drops)")
print("parse_drops =", len(parse_drops), parse_drops[:5])
print("eid collisions old∩new =", len(collide))
lc = Counter(r["label"] for r in new_map)
print("new label counts:", dict(lc))
print("mapping_v2 total rows =", len(old_rows) + len(new_map),
      "unique src =", len({r['src_table_id'] for r in old_rows+new_map}))
print("files: new_train_model.jsonl new_train_full.jsonl mapping_v2.jsonl")
