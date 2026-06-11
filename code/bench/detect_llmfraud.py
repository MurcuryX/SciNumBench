"""
detect_llmfraud.py — objective (training-free) detector suite for the boost-ours fabrication eval set.

For each sample in results/llmfraud_eval.jsonl, rebuild the grid using only text (judge view),
run the following detectors, each producing results/verdicts_llmfraud_<name>.json:
  rule_control   : detectors.py's 5-rule scanner (GRIM/CI/PVAL/PCT/DIST); any hit -> FABRICATED.
                   Expected ~zero recall (boost-ours breaks no in-table consistency); used as control.
  benford        : first-digit Benford chi-square p<0.05 -> FABRICATED.
  terminal_digit : terminal-digit uniformity chi-square p<0.05 -> FABRICATED.
  rounding_bias  : significant preference for integer/half-integer/last-digit 0 or 5 (binomial test p<0.05) -> FABRICATED.
  ours_dominance : use method_axis/our_index to judge whether the authors' method is (nearly) best on all metrics / has an anomalous lead margin.
                   Two versions: dominance_allwin (all wins) and dominance_margin (lead-margin z).

All thresholds use unsupervised defaults (p<0.05 / fixed quantile); never tuned against labels.
ours_dominance needs our_index/method_axis (structural meta-info, not leakage: the table itself reveals "which row is ours");
the other detectors use only text numbers and do not touch label/provenance.
"""
import os
import re
import sys
import json
import math
import sqlite3
from collections import Counter

import numpy as np
from scipy import stats

ROOT = "/home/pengminjie/Backup/paper/ICDE26"
sys.path.insert(0, ROOT + "/code/bench")
from parser import StructuredTable          # noqa: E402
import detectors as DET                      # noqa: E402
from llmfraud_common import load_eval, parse_text_to_grid  # noqa: E402

RESULTS = ROOT + "/results"
_NUMTOK = re.compile(r"-?\d+(?:\.\d+)?")
BENFORD = np.array([math.log10(1 + 1.0 / d) for d in range(1, 10)])


# ───────────────────────── number extraction ─────────────────────────
def st_from_record(rec):
    cap, grid = parse_text_to_grid(rec["text"])
    if not grid:
        return None
    return StructuredTable(grid[0], grid[1:], caption=cap)


def numbers_and_lastdigits(st):
    """Return (list of floats for first digit, list of last digits). Reuses the baseline_benford view:
    numeric cells take values by type; last digit takes the last decimal/integer digit of the raw token."""
    nums, digs = [], []
    for row in st.cells:
        for c in row:
            t = c["type"]
            if t == "meansd":
                vals = [c["mean"], c["sd"]]
            elif t == "ci":
                vals = [c["point"], c["lo"], c["hi"]]
            elif t in ("pct", "num", "count", "pval"):
                vals = [c["value"]]
            elif t == "countpct":
                vals = [c["count"], c["pct"]] + ([c["denom"]] if c.get("denom") else [])
            else:
                vals = []
                for m in _NUMTOK.findall(c.get("raw", "")):
                    try:
                        vals.append(float(m))
                    except Exception:
                        pass
            nums.extend(vals)
            if t in ("meansd", "ci", "pct", "num", "count", "countpct", "pval"):
                for tok in _NUMTOK.findall(c.get("raw", "")):
                    d = tok[-1]
                    if d.isdigit():
                        digs.append(int(d))
    return nums, digs


def first_digit(x):
    x = abs(float(x))
    if x == 0:
        return None
    while x < 1:
        x *= 10
    while x >= 10:
        x /= 10
    return int(x)


# ───────────────────────── detectors ─────────────────────────
def det_rule_control(rec, st):
    """5-rule scan; any hit -> FABRICATED."""
    if st is None:
        return "PLAUSIBLE"
    flags = DET.scan_all(st)
    return "FABRICATED" if flags else "PLAUSIBLE"


def det_benford(rec, st):
    """first-digit Benford chi-square p<0.05 -> FABRICATED (PLAUSIBLE if too few numbers)."""
    if st is None:
        return "PLAUSIBLE"
    nums, _ = numbers_and_lastdigits(st)
    fd = [first_digit(x) for x in nums]
    fd = [d for d in fd if d is not None and 1 <= d <= 9]
    n = len(fd)
    if n < 20:                       # insufficient power, do not flag
        return "PLAUSIBLE"
    obs = np.zeros(9)
    for d in fd:
        obs[d - 1] += 1
    exp = BENFORD * n
    chi2 = float(((obs - exp) ** 2 / exp).sum())
    p = 1.0 - stats.chi2.cdf(chi2, df=8)
    return "FABRICATED" if p < 0.05 else "PLAUSIBLE"


def det_terminal(rec, st):
    """terminal-digit uniformity chi-square p<0.05 -> FABRICATED."""
    if st is None:
        return "PLAUSIBLE"
    _, digs = numbers_and_lastdigits(st)
    n = len(digs)
    if n < 20:
        return "PLAUSIBLE"
    counts = np.zeros(10)
    for d in digs:
        counts[d] += 1
    exp = n / 10.0
    chi2 = float(((counts - exp) ** 2 / exp).sum())
    p = 1.0 - stats.chi2.cdf(chi2, df=9)
    return "FABRICATED" if p < 0.05 else "PLAUSIBLE"


def det_rounding(rec, st):
    """Rounding fingerprint: proportion of last digit 0 or 5 significantly above the uninformative baseline (0.2).
    For tokens with decimals, look at the last decimal digit; one-sided binomial test p<0.05 -> FABRICATED."""
    if st is None:
        return "PLAUSIBLE"
    last = []
    for row in st.cells:
        for c in row:
            if c["type"] in ("meansd", "ci", "pct", "num", "countpct"):
                for tok in _NUMTOK.findall(c.get("raw", "")):
                    if "." in tok:            # only decimal tokens; rounding = last digit 0/5
                        d = tok[-1]
                        if d.isdigit():
                            last.append(int(d))
    n = len(last)
    if n < 20:
        return "PLAUSIBLE"
    k = sum(1 for d in last if d in (0, 5))
    # one-sided binomial test: H0 p=0.2 ({0,5} is 2/10 of random last digits)
    pval = stats.binomtest(k, n, 0.2, alternative="greater").pvalue
    return "FABRICATED" if pval < 0.05 else "PLAUSIBLE"


# ───── ours_dominance: needs structural meta-info our_index / method_axis ─────
def _numeric_matrix(st, axis, our_index):
    """Return (ours_vec, others_matrix) values aligned along the "metric" dimension.
    axis='row': each method is a row, our_index is the grid row number (0=header) -> data row = our_index-1.
                metrics = numeric columns.
    axis='col': each method is a column, our_index is the grid column number -> metrics = numeric rows.
    Skip missing/non-numeric metric positions. Return list[(ours_val, [other_vals])] aligned by metric."""
    def cellnum(i, j):
        if i < 0 or i >= st.nrows or j < 0 or j >= st.ncols:
            return None
        c = st.cells[i][j]
        if c["type"] == "meansd":
            return c["mean"]
        if c["type"] == "ci":
            return c["point"]
        if c["type"] in ("pct", "num", "count", "pval"):
            return c["value"]
        if c["type"] == "countpct":
            return c["pct"]
        # text fallback: leading number
        m = _NUMTOK.search(c.get("raw", ""))
        return float(m.group(0)) if m else None

    pairs = []
    if axis == "row":
        ours_i = our_index - 1                       # data row number
        if ours_i < 0 or ours_i >= st.nrows:
            return []
        method_rows = [i for i in range(st.nrows) if i != ours_i]
        for j in range(st.ncols):                    # each metric column
            ov = cellnum(ours_i, j)
            if ov is None:
                continue
            others = [cellnum(i, j) for i in method_rows]
            others = [v for v in others if v is not None]
            if len(others) >= 1:
                pairs.append((ov, others))
    else:  # col
        ours_j = our_index
        if ours_j < 0 or ours_j >= st.ncols:
            return []
        method_cols = [j for j in range(st.ncols) if j != ours_j]
        for i in range(st.nrows):                    # each metric row
            ov = cellnum(i, ours_j)
            if ov is None:
                continue
            others = [cellnum(i, j) for j in method_cols]
            others = [v for v in others if v is not None]
            if len(others) >= 1:
                pairs.append((ov, others))
    return pairs


def ours_dominance_features(rec, st):
    """Return the ours-dominance feature dict for this table (no decision; used by both versions).
    The polarity of each metric (higher/lower better) is unknown, so compute metric wins under both
    max-better and min-better hypotheses, and pick the one "more likely to be ours' home turf" (where ours wins more) as this table's direction."""
    if st is None:
        return None
    axis = rec["method_axis"]
    pairs = _numeric_matrix(st, axis, rec["our_index"])
    if len(pairs) < 2:
        return None
    # assume higher-better / lower-better separately
    win_hi = win_lo = 0
    margins_hi, margins_lo = [], []
    for ov, others in pairs:
        best_other_hi = max(others)
        best_other_lo = min(others)
        # higher-better: ours wins = ov >= all others
        if ov >= best_other_hi - 1e-9:
            win_hi += 1
        # lower-better: ours wins = ov <= all others
        if ov <= best_other_lo + 1e-9:
            win_lo += 1
        rng = (max(others + [ov]) - min(others + [ov])) or 1.0
        margins_hi.append((ov - best_other_hi) / rng)   # >0 means higher than the runner-up
        margins_lo.append((best_other_lo - ov) / rng)
    nmet = len(pairs)
    # choose the direction where ours wins more as this table's direction (matching the real-paper setup where ours is the protagonist)
    if win_hi >= win_lo:
        wins, margins = win_hi, margins_hi
    else:
        wins, margins = win_lo, margins_lo
    return dict(nmet=nmet, wins=wins, frac_win=wins / nmet,
                mean_margin=float(np.mean(margins)),
                max_margin=float(np.max(margins)))


def make_ours_allwin(feats_by_id):
    """Version A: ours is best on (nearly) all metrics -> FABRICATED. Threshold: frac_win==1 and nmet>=2."""
    def fn(eid):
        f = feats_by_id.get(eid)
        if f is None:
            return "PLAUSIBLE"
        return "FABRICATED" if (f["frac_win"] >= 0.999 and f["nmet"] >= 2) else "PLAUSIBLE"
    return fn


def make_ours_margin(feats_by_id):
    """Version B: ours' lead margin is anomalously large -> FABRICATED.
    Threshold is the unsupervised upper quantile (75%) of the margin distribution over the whole eval set; no labels used."""
    vals = [f["mean_margin"] for f in feats_by_id.values() if f]
    thr = float(np.percentile(vals, 75)) if vals else 0.0
    def fn(eid):
        f = feats_by_id.get(eid)
        if f is None:
            return "PLAUSIBLE"
        return "FABRICATED" if f["mean_margin"] > thr else "PLAUSIBLE"
    return fn, thr


# ───────────────────────── main flow ─────────────────────────
def main():
    recs = load_eval()
    sts = {}
    for r in recs:
        sts[r["eval_id"]] = st_from_record(r)

    simple = {
        "rule_control": det_rule_control,
        "benford": det_benford,
        "terminal_digit": det_terminal,
        "rounding_bias": det_rounding,
    }
    written = []
    for name, fn in simple.items():
        verdicts = {}
        for r in recs:
            verdicts[str(r["eval_id"])] = fn(r, sts[r["eval_id"]])
        path = f"{RESULTS}/verdicts_llmfraud_{name}.json"
        json.dump(verdicts, open(path, "w"), ensure_ascii=False, indent=0)
        written.append(path)
        print(f"[WRITE] {path}  FAB={sum(1 for v in verdicts.values() if v=='FABRICATED')}/{len(verdicts)}")

    # ours_dominance: compute features first
    feats = {}
    for r in recs:
        feats[r["eval_id"]] = ours_dominance_features(r, sts[r["eval_id"]])
    n_usable = sum(1 for f in feats.values() if f)
    print(f"[ours_dominance] usable tables (>=2 metrics): {n_usable}/{len(recs)}")

    fn_allwin = make_ours_allwin(feats)
    fn_margin, thr = make_ours_margin(feats)
    print(f"[ours_dominance] margin threshold (75pct, unsup) = {thr:.4f}")

    for name, fn in [("ours_dominance_allwin", fn_allwin),
                     ("ours_dominance_margin", fn_margin)]:
        verdicts = {str(r["eval_id"]): fn(r["eval_id"]) for r in recs}
        path = f"{RESULTS}/verdicts_llmfraud_{name}.json"
        json.dump(verdicts, open(path, "w"), ensure_ascii=False, indent=0)
        written.append(path)
        print(f"[WRITE] {path}  FAB={sum(1 for v in verdicts.values() if v=='FABRICATED')}/{len(verdicts)}")

    print("\n[DONE] verdict files:")
    for p in written:
        print("  ", p)


if __name__ == "__main__":
    main()
