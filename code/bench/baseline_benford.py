"""
baseline_benford.py — Benford / terminal-digit negative control baseline.

For each table, extract digits from all numeric cells and compute:
  (a) first-digit Benford chi-square + MAD
  (b) terminal-digit (last digit) uniformity chi-square
A table is flagged fake by threshold, evaluated as a table-level detector
(same P/R/F1/FPR + per-family recall). Threshold selected on val (max F1).
Expected to be near-random — that is the purpose of a negative control.

Additional report: median/distribution of numeric-cell counts per table, to
support the argument "median<<50 → terminal-digit test underpowered".

Loading mirrors evaluate.py: corrupted_grid is a JSON list, g[0]=columns,
g[1:]=data, caption from join with paper_tables, constructing
StructuredTable(g[0], g[1:], caption=cap). Output results/baseline_benford.json.
"""
import os, sys, json, time, argparse, sqlite3, re, math
import numpy as np
from collections import Counter

ROOT = "/home/pengminjie/Backup/paper/ICDE26"
sys.path.insert(0, ROOT + "/code/bench")
from parser import StructuredTable  # noqa: E402

DB = ROOT + "/data/arxiv_data.db"
OUT = ROOT + "/results/baseline_benford.json"
FAMILIES = ["GRIM", "PVAL", "CI", "PCT", "DIST"]
# Benford expected first-digit distribution
BENFORD = np.array([math.log10(1 + 1.0 / dd) for dd in range(1, 10)])  # digits 1..9
_NUMTOK = re.compile(r"-?\d+(?:\.\d+)?")


def extract_numbers(st):
    """Extract numbers from all numeric cells of a StructuredTable (by parsed
    type, falling back to raw tokens). Returns a list of floats."""
    nums = []
    for row in st.cells:
        for c in row:
            t = c["type"]
            vals = []
            if t == "meansd":
                vals = [c["mean"], c["sd"]]
            elif t == "ci":
                vals = [c["point"], c["lo"], c["hi"]]
            elif t in ("pct", "num", "count", "pval"):
                vals = [c["value"]]
            elif t == "countpct":
                vals = [c["count"], c["pct"]]
                if c.get("denom"):
                    vals.append(c["denom"])
            else:
                # text/empty: scan raw for embedded numeric tokens
                for m in _NUMTOK.findall(c.get("raw", "")):
                    try:
                        vals.append(float(m))
                    except Exception:
                        pass
            nums.extend(vals)
    return nums


def first_digit(x):
    x = abs(float(x))
    if x == 0:
        return None
    while x < 1:
        x *= 10
    while x >= 10:
        x /= 10
    return int(x)


def last_digit_from_raw(st):
    """Terminal digit: take the last digit of each numeric token in the raw
    string of every numeric cell, following the forensics-literature
    terminal-digit test which focuses on the final (decimal) digit."""
    digs = []
    for row in st.cells:
        for c in row:
            if c["type"] in ("meansd", "ci", "pct", "num", "count", "countpct", "pval"):
                # find numeric tokens in raw, take terminal digit of each
                for tok in _NUMTOK.findall(c.get("raw", "")):
                    d = tok[-1]
                    if d.isdigit():
                        digs.append(int(d))
    return digs


def chi2_uniform(counts, k):
    """Chi-square statistic vs a uniform distribution over k categories."""
    n = counts.sum()
    if n == 0:
        return 0.0
    exp = n / k
    return float(((counts - exp) ** 2 / exp).sum())


def benford_stats(nums):
    """Returns (chi2_first, mad_first, n_first_digits)."""
    fd = [first_digit(x) for x in nums]
    fd = [d for d in fd if d is not None and 1 <= d <= 9]
    n = len(fd)
    if n == 0:
        return 0.0, 0.0, 0
    obs = np.zeros(9)
    for d in fd:
        obs[d - 1] += 1
    p_obs = obs / n
    exp = BENFORD
    chi2 = float((((p_obs - exp) ** 2 / exp) * n).sum())
    mad = float(np.abs(p_obs - exp).mean())
    return chi2, mad, n


def terminal_stats(digs):
    """Last-digit uniformity chi-square (10 categories) and sample size."""
    if not digs:
        return 0.0, 0
    counts = np.zeros(10)
    for d in digs:
        counts[d] += 1
    return chi2_uniform(counts, 10), len(digs)


def metrics(y, pred):
    y = np.asarray(y); pred = np.asarray(pred)
    TP = int(((pred == 1) & (y == 1)).sum()); FP = int(((pred == 1) & (y == 0)).sum())
    FN = int(((pred == 0) & (y == 1)).sum()); TN = int(((pred == 0) & (y == 0)).sum())
    P = TP / (TP + FP) if TP + FP else 0.0
    R = TP / (TP + FN) if TP + FN else 0.0
    F1 = 2 * P * R / (P + R) if P + R else 0.0
    Acc = (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) else 0.0
    FPR = FP / (FP + TN) if (FP + TN) else 0.0
    return dict(TP=TP, FP=FP, FN=FN, TN=TN, P=P, R=R, F1=F1, Acc=Acc, FPR=FPR)


def per_family_recall(fam, y, pred):
    fam = np.asarray(fam); y = np.asarray(y); pred = np.asarray(pred)
    out = {}
    for f in FAMILIES:
        m = (y == 1) & (fam == f)
        n = int(m.sum())
        out[f] = dict(n=n, recall=(float((pred[m] == 1).mean()) if n else None))
    return out


def load_rows(conn, splits, limit=0):
    q = f"""SELECT sb.bench_id, sb.dataset_split, sb.label, sb.corruption_family,
                   sb.corrupted_grid, pt.caption
            FROM scinum_bench sb LEFT JOIN paper_tables pt ON sb.src_table_id=pt.id
            WHERE sb.dataset_split IN ({','.join('?'*len(splits))})"""
    if limit:
        q += f" LIMIT {limit}"
    return conn.execute(q, splits).fetchall()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()

    conn = sqlite3.connect(DB)
    splits = ["val", "test", "cross_test"]
    rows = load_rows(conn, splits)
    conn.close()

    # group rows by split
    data = {s: [] for s in splits}
    ncell_all = []
    for b, sp, lab, fam, cg, cap in rows:
        try:
            g = json.loads(cg)
        except Exception:
            continue
        if len(g) < 2:
            st = None
            nums = []; digs = []
        else:
            st = StructuredTable(g[0], g[1:], caption=cap or "")
            nums = extract_numbers(st)
            digs = last_digit_from_raw(st)
        c1, mad, nfd = benford_stats(nums)
        ct, ntd = terminal_stats(digs)
        n_numeric_cells = 0
        if st is not None:
            tc = st.type_counts()
            n_numeric_cells = sum(tc.get(t, 0) for t in
                                  ("meansd", "ci", "pct", "num", "count", "countpct", "pval"))
        rec = dict(bench_id=int(b), label=int(lab), family=(fam or ""),
                   chi2_first=c1, mad_first=mad, n_first=nfd,
                   chi2_term=ct, n_term=ntd, n_numeric_cells=n_numeric_cells)
        data[sp].append(rec)
        ncell_all.append(n_numeric_cells)

    if args.smoke:
        for s in data:
            data[s] = data[s][:args.smoke]

    # numeric-cell-count distribution
    nc = np.array([r["n_numeric_cells"] for r in data["test"]])
    cell_dist = dict(
        median=float(np.median(nc)), mean=float(np.mean(nc)),
        p25=float(np.percentile(nc, 25)), p75=float(np.percentile(nc, 75)),
        p90=float(np.percentile(nc, 90)), max=int(nc.max()),
        frac_ge_50=float((nc >= 50).mean()), frac_ge_30=float((nc >= 30).mean()),
        n_tables=len(nc),
        note="terminal-digit uniformity test empirically needs >=50 numeric points; median<<50 -> underpowered (scope-out argument)")
    print(f"[CELL-DIST] test numeric-cells: median={cell_dist['median']} "
          f"mean={cell_dist['mean']:.1f} p90={cell_dist['p90']} "
          f"frac>=50={cell_dist['frac_ge_50']:.3f}", flush=True)

    results = {"reference": {
        "M1_rules": {"F1": 0.881, "FPR": 0.190},
        "random_baseline_note": "fake is 60% of test; always-fake F1=0.75 is the trivial upper bound"},
        "numeric_cell_distribution_test": cell_dist,
        "detectors": {}}

    def yfam(recs):
        return (np.array([r["label"] for r in recs]),
                np.array([r["family"] for r in recs]))

    yval, _ = yfam(data["val"])
    yte, fam_te = yfam(data["test"])
    ycross = np.array([r["label"] for r in data["cross_test"]])

    def eval_detector(name, key, higher_is_fake=True):
        """key in {chi2_first, mad_first, chi2_term}. Threshold selected on val (max F1)."""
        sval = np.array([r[key] for r in data["val"]], dtype=float)
        ste = np.array([r[key] for r in data["test"]], dtype=float)
        scross = np.array([r[key] for r in data["cross_test"]], dtype=float)
        # candidate thresholds from val percentiles
        cand = np.unique(np.percentile(sval, np.linspace(1, 99, 99)))
        best_thr, best_f1 = None, -1
        for thr in cand:
            pred = (sval >= thr).astype(int) if higher_is_fake else (sval <= thr).astype(int)
            f1 = metrics(yval, pred)["F1"]
            if f1 > best_f1:
                best_f1, best_thr = f1, float(thr)
        pred_te = (ste >= best_thr).astype(int) if higher_is_fake else (ste <= best_thr).astype(int)
        pred_cross = (scross >= best_thr).astype(int) if higher_is_fake else (scross <= best_thr).astype(int)
        m_test = metrics(yte, pred_te)
        m_cross = metrics(ycross, pred_cross)
        pfr = per_family_recall(fam_te, yte, pred_te)
        out = dict(key=key, thr=best_thr, val_F1=best_f1, test=m_test,
                   cross_test_FPR=m_cross["FPR"], per_family_recall=pfr)
        print(f"[DET] {name}: thr={best_thr:.4g} val_F1={best_f1:.3f} | "
              f"test P={m_test['P']:.3f} R={m_test['R']:.3f} F1={m_test['F1']:.3f} "
              f"Acc={m_test['Acc']:.3f} FPR={m_test['FPR']:.3f} | cross_FPR={m_cross['FPR']:.3f}",
              flush=True)
        results["detectors"][name] = out

    eval_detector("benford_first_chi2", "chi2_first", higher_is_fake=True)
    eval_detector("benford_first_mad", "mad_first", higher_is_fake=True)
    eval_detector("terminal_digit_chi2", "chi2_term", higher_is_fake=True)

    # trivial all-fake reference on test
    results["trivial_all_fake_test"] = metrics(yte, np.ones_like(yte))

    results["meta"] = dict(db=DB, seed=42, smoke=args.smoke,
                           total_secs=round(time.time() - t0, 1))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
