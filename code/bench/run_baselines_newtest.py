"""
run_baselines_newtest.py — Step 4 (Plan B): re-run training-free baselines on the
NEW src-disjoint test split (2000 rows), comparable to the hidden-state probe.

REUSES the exact detector implementations:
  - detectors.py        (5-rule scanner: GRIM/DIST/CI/PCT/PVAL)  -> "rules"
  - detect_llmfraud.py  (benford / terminal_digit / rounding_bias / ours_dominance)
We only ADAPT THE INPUT: load data/splits/test_model.jsonl (text, by example_id),
reconstruct StructuredTable via llmfraud_common.parse_text_to_grid (same as before),
and pull method_axis/our_index from the llm_fraud DB keyed by src_table_id
(structural "which row is ours" metadata; available for pos & neg alike, NOT a label).

Outputs per-example predictions keyed by example_id:
  results/baseline_preds_<name>.json   (binary verdict 1=FABRICATED / 0 + a continuous score)
Each baseline emits a BINARY verdict (the paper's earlier decision rule, unsupervised
threshold p<0.05 / fixed quantile) AND a continuous SCORE so AUROC can be computed.
"""
import os, sys, json, math
import numpy as np
from scipy import stats

ROOT = "/home/pengminjie/Backup/paper/ICDE26"
sys.path.insert(0, ROOT + "/code/bench")
from parser import StructuredTable                       # noqa
import detectors as DET                                  # noqa
from llmfraud_common import parse_text_to_grid           # noqa
# reuse the EXACT numeric-extraction + ours_dominance feature code from detect_llmfraud
import detect_llmfraud as DLF                            # noqa

import sqlite3
DB = ROOT + "/data/arxiv_data.db"
SPLITS = ROOT + "/data/splits"
RESULTS = ROOT + "/results"
BENFORD = np.array([math.log10(1 + 1.0 / d) for d in range(1, 10)])


def load_test():
    rows = []
    for line in open(f"{SPLITS}/test_model.jsonl"):
        line = line.strip()
        if line:
            rows.append(json.loads(line))   # {example_id, text}
    return rows


def load_src_map():
    """example_id -> src_table_id (from test.jsonl)."""
    m = {}
    for line in open(f"{SPLITS}/test.jsonl"):
        r = json.loads(line)
        m[r["example_id"]] = r["src_table_id"]
    return m


def load_db_axis():
    """src_table_id -> (method_axis, our_index) from llm_fraud (structural metadata)."""
    conn = sqlite3.connect(DB)
    out = {}
    for sid, axis, oi in conn.execute(
            "SELECT src_table_id, method_axis, our_index FROM llm_fraud WHERE status='ok'"):
        out[int(sid)] = (axis, int(oi))
    conn.close()
    return out


def st_from_text(text):
    cap, grid = parse_text_to_grid(text)
    if not grid:
        return None
    return StructuredTable(grid[0], grid[1:], caption=cap)


# ---- continuous-score variants of each detector (same statistic the binary rule uses) ----
def benford_score(st):
    """1 - chi2_p  (higher = more anomalous). Binary rule: p<0.05."""
    if st is None:
        return 0.0, "PLAUSIBLE"
    nums, _ = DLF.numbers_and_lastdigits(st)
    fd = [DLF.first_digit(x) for x in nums]
    fd = [d for d in fd if d is not None and 1 <= d <= 9]
    n = len(fd)
    if n < 20:
        return 0.0, "PLAUSIBLE"
    obs = np.zeros(9)
    for d in fd:
        obs[d - 1] += 1
    exp = BENFORD * n
    chi2 = float(((obs - exp) ** 2 / exp).sum())
    p = 1.0 - stats.chi2.cdf(chi2, df=8)
    return 1.0 - p, ("FABRICATED" if p < 0.05 else "PLAUSIBLE")


def terminal_score(st):
    if st is None:
        return 0.0, "PLAUSIBLE"
    _, digs = DLF.numbers_and_lastdigits(st)
    n = len(digs)
    if n < 20:
        return 0.0, "PLAUSIBLE"
    counts = np.zeros(10)
    for d in digs:
        counts[d] += 1
    exp = n / 10.0
    chi2 = float(((counts - exp) ** 2 / exp).sum())
    p = 1.0 - stats.chi2.cdf(chi2, df=9)
    return 1.0 - p, ("FABRICATED" if p < 0.05 else "PLAUSIBLE")


def rounding_score(st):
    if st is None:
        return 0.0, "PLAUSIBLE"
    import re
    last = []
    for row in st.cells:
        for c in row:
            if c["type"] in ("meansd", "ci", "pct", "num", "countpct"):
                for tok in DLF._NUMTOK.findall(c.get("raw", "")):
                    if "." in tok:
                        d = tok[-1]
                        if d.isdigit():
                            last.append(int(d))
    n = len(last)
    if n < 20:
        return 0.0, "PLAUSIBLE"
    k = sum(1 for d in last if d in (0, 5))
    pval = stats.binomtest(k, n, 0.2, alternative="greater").pvalue
    return 1.0 - pval, ("FABRICATED" if pval < 0.05 else "PLAUSIBLE")


def rules_score(st):
    """5-rule scanner: count of flagged cells as score; any flag -> FABRICATED."""
    if st is None:
        return 0.0, "PLAUSIBLE"
    flags = DET.scan_all(st)
    return float(len(flags)), ("FABRICATED" if flags else "PLAUSIBLE")


def main():
    os.makedirs(RESULTS, exist_ok=True)
    rows = load_test()
    src_map = load_src_map()
    axis_map = load_db_axis()
    print(f"[DATA] {len(rows)} test rows", flush=True)

    sts = {}
    parse_fail = 0
    for r in rows:
        st = st_from_text(r["text"])
        if st is None:
            parse_fail += 1
        sts[r["example_id"]] = st
    print(f"[PARSE] grid-parse failures (st=None): {parse_fail}", flush=True)

    # ---- simple detectors with score+verdict ----
    simple = {
        "rules": rules_score,
        "benford": benford_score,
        "terminal_digit": terminal_score,
        "rounding": rounding_score,
    }
    for name, fn in simple.items():
        preds = {}
        for r in rows:
            eid = r["example_id"]
            score, verdict = fn(sts[eid])
            preds[eid] = {"score": float(score), "verdict": verdict,
                          "pred": 1 if verdict == "FABRICATED" else 0}
        path = f"{RESULTS}/baseline_preds_{name}.json"
        json.dump(preds, open(path, "w"), ensure_ascii=False, indent=0)
        nfab = sum(1 for v in preds.values() if v["pred"] == 1)
        print(f"[WRITE] {path}  FAB={nfab}/{len(preds)}", flush=True)

    # ---- ours_dominance: build feats using EXACT DLF.ours_dominance_features ----
    # need a record-like obj carrying method_axis & our_index
    feats = {}
    n_no_axis = 0
    for r in rows:
        eid = r["example_id"]
        sid = src_map.get(eid)
        ax = axis_map.get(sid)
        if ax is None:
            n_no_axis += 1
            feats[eid] = None
            continue
        rec = {"method_axis": ax[0], "our_index": ax[1]}
        feats[eid] = DLF.ours_dominance_features(rec, sts[eid])
    n_usable = sum(1 for f in feats.values() if f)
    print(f"[ours_dominance] usable (>=2 metrics): {n_usable}/{len(rows)}  no_axis={n_no_axis}", flush=True)

    # allwin: frac_win==1 & nmet>=2 -> FABRICATED ; score = frac_win
    # margin : mean_margin > 75th pct (unsupervised) -> FABRICATED ; score = mean_margin
    vals = [f["mean_margin"] for f in feats.values() if f]
    thr = float(np.percentile(vals, 75)) if vals else 0.0
    print(f"[ours_dominance] margin 75pct thr (unsup) = {thr:.4f}", flush=True)

    for name in ("ours_dominance_allwin", "ours_dominance_margin"):
        preds = {}
        for r in rows:
            eid = r["example_id"]
            f = feats[eid]
            if f is None:
                preds[eid] = {"score": 0.0, "verdict": "PLAUSIBLE", "pred": 0}
                continue
            if name.endswith("allwin"):
                score = f["frac_win"]
                verdict = "FABRICATED" if (f["frac_win"] >= 0.999 and f["nmet"] >= 2) else "PLAUSIBLE"
            else:
                score = f["mean_margin"]
                verdict = "FABRICATED" if f["mean_margin"] > thr else "PLAUSIBLE"
            preds[eid] = {"score": float(score), "verdict": verdict,
                          "pred": 1 if verdict == "FABRICATED" else 0}
        path = f"{RESULTS}/baseline_preds_{name}.json"
        json.dump(preds, open(path, "w"), ensure_ascii=False, indent=0)
        nfab = sum(1 for v in preds.values() if v["pred"] == 1)
        print(f"[WRITE] {path}  FAB={nfab}/{len(preds)}", flush=True)

    print("[DONE] non-LLM baselines", flush=True)


if __name__ == "__main__":
    main()
