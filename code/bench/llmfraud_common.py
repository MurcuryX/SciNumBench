"""
llmfraud_common.py — shared utilities: load llmfraud_eval.jsonl, rebuild the grid
from `text`, and validate parsing correctness against the DB (llm_fraud). Reused by
the objective detectors.

Design: the eval row's `text` is the product of clean_serialize.serialize (a markdown
pipe table plus a one-line Caption). Detectors should only see `text` (the judge's
view), but this module also exposes the DB grid for offline self-check alignment.
"""
import os
import re
import sys
import json
import sqlite3

ROOT = "/home/pengminjie/Backup/paper/ICDE26"
sys.path.insert(0, ROOT + "/code/bench")

DB = ROOT + "/data/arxiv_data.db"
EVAL = ROOT + "/results/llmfraud_eval.jsonl"


def load_eval(path=EVAL):
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def parse_text_to_grid(text):
    """Reverse-parse the markdown table produced by serialize() into
    (caption, grid[list-of-lists]). grid[0] is the header. Skip markdown
    separator rows (| --- | --- |) and the [...truncated...] footnote."""
    caption = ""
    rows = []
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith("Caption:"):
            caption = s[len("Caption:"):].strip()
            continue
        if not s.startswith("|"):
            continue
        # Separator row: cells consisting only of ---
        inner = s.strip("|")
        cells = [c.strip() for c in inner.split("|")]
        if cells and all(set(c) <= set("-: ") and c for c in cells):
            continue  # markdown separator line
        rows.append(cells)
    return caption, rows


def from_db():
    """Return {src_table_id: dict(original_grid, fabricated_grid, provenance, caption,
    method_axis, our_index, strategy)}. Used for self-check alignment."""
    conn = sqlite3.connect(DB)
    out = {}
    for r in conn.execute(
            """SELECT src_table_id, strategy, method_axis, our_index, caption,
                      original_grid, fabricated_grid, provenance
               FROM llm_fraud WHERE status='ok'"""):
        (sid, strat, axis, oi, cap, og, fg, prov) = r
        out[int(sid)] = dict(strategy=strat, method_axis=axis, our_index=int(oi),
                             caption=cap or "",
                             original_grid=json.loads(og),
                             fabricated_grid=json.loads(fg),
                             provenance=json.loads(prov) if prov else [])
    conn.close()
    return out
