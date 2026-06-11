"""
detectors.py — SciNumBench rule-based detector suite (no training).

Each detector is an independent scanner for its forensic rule: it scans the
whole table for violating cells without looking at ground truth.
Returns flags = [{r,c,rule,detail}]. Table-level prediction = any rule fires -> fake.
Logically the inverse of the corruptor, but implemented independently (does not call verify_fake).
"""
import re
import sys
sys.path.insert(0, "/home/pengminjie/Backup/paper/ICDE26/code/bench")
from parser import StructuredTable


def _grim_consistent(mean, n, dec):
    if n <= 0:
        return True
    x = round(mean * n)
    return round(x / n, dec) == round(mean, dec)


def scan_grim(st):
    """Mean of integer-type measurements must be reproducible from an integer sum given n and decimal places. Integer means (dec=0) are skipped."""
    flags = []
    for (i, j) in st.positions_of("meansd"):
        n = st.n_for(j)
        if not n:
            continue
        cell = st.cells[i][j]
        m = re.match(r"^\s*(-?\d+(?:\.\d+)?)", cell["raw"])
        mt = m.group(1)
        dec = len(mt.split(".")[1]) if "." in mt else 0
        if dec == 0:
            continue
        if not _grim_consistent(cell["mean"], n, dec):
            flags.append({"r": i + 1, "c": j, "rule": "GRIM",
                          "detail": f"mean={cell['mean']} n={n} dec={dec}"})
    return flags


# Non-negative quantity names (for DIST: mean-2SD<0 is impossible only when the quantity should be non-negative)
_NONNEG = re.compile(r"(?i)\b(age|year|weight|height|bmi|score|time|dose|count|number|"
                     r"rate|level|concentration|duration|length|distance|mass|volume|"
                     r"pressure|temperature|index|days?|weeks?|months?|min(ute)?s?|"
                     r"seconds?|cm|kg|mg|mm|ml|years?)\b")


def scan_dist(st):
    """mean-2SD<0 and row/column label implies a non-negative quantity -> impossible distribution."""
    flags = []
    for (i, j) in st.positions_of("meansd"):
        cell = st.cells[i][j]
        if cell["sd"] <= 0 or cell["mean"] <= 0:
            continue
        if cell["mean"] - 2 * cell["sd"] >= 0:
            continue
        # Context non-negativity check: row label or column header
        rowlab = st.data[i][0] if st.ncols and st.data[i] else ""
        collab = st.columns[j] if j < st.ncols else ""
        ctx = f"{rowlab} {collab} {st.caption}"
        if _NONNEG.search(ctx):
            flags.append({"r": i + 1, "c": j, "rule": "DIST",
                          "detail": f"mean={cell['mean']} sd={cell['sd']} -> mean-2sd<0, non-negative quantity"})
    return flags


def scan_ci(st):
    """Point estimate must lie within [lo,hi]."""
    flags = []
    for (i, j) in st.positions_of("ci"):
        cell = st.cells[i][j]
        if cell["hi"] <= cell["lo"]:
            continue
        if not (cell["lo"] <= cell["point"] <= cell["hi"]):
            flags.append({"r": i + 1, "c": j, "rule": "CI",
                          "detail": f"point={cell['point']} ∉ [{cell['lo']},{cell['hi']}]"})
    return flags


def scan_pct(st):
    """count(%) cells: count/denominator should approx equal the shown % (denominator from cell denom or column n)."""
    flags = []
    for (i, j) in st.positions_of("countpct"):
        cell = st.cells[i][j]
        denom = cell["denom"] or st.n_for(j)
        if not denom or denom <= 0 or cell["count"] > denom:
            continue
        expected = 100.0 * cell["count"] / denom
        if abs(expected - cell["pct"]) > 8.0:
            flags.append({"r": i + 1, "c": j, "rule": "PCT",
                          "detail": f"count={cell['count']}/{denom}->{expected:.1f}% vs shown {cell['pct']}%"})
    return flags


def scan_pval(st):
    """p-value significance must agree with the same-row CI (CI excludes null=1 iff p<0.05)."""
    flags = []
    for (i, j) in st.positions_of("pval"):
        cell = st.cells[i][j]
        ci = [st.cells[i][jj] for jj in range(st.ncols) if st.cells[i][jj]["type"] == "ci"]
        if not ci:
            continue
        cc = ci[0]
        if cc["hi"] <= cc["lo"]:
            continue
        excludes = not (cc["lo"] <= 1.0 <= cc["hi"])
        sig = cell["value"] < 0.05
        if excludes != sig:
            flags.append({"r": i + 1, "c": j, "rule": "PVAL",
                          "detail": f"p={cell['value']} vs CI[{cc['lo']},{cc['hi']}]"})
    return flags


DETECTORS = {"GRIM": scan_grim, "DIST": scan_dist, "CI": scan_ci,
             "PCT": scan_pct, "PVAL": scan_pval}


def scan_all(st):
    """Run all detectors and return all flags."""
    flags = []
    for fn in DETECTORS.values():
        flags.extend(fn(st))
    return flags
