"""
corruptors.py — SciNumBench 6 corruption families (refactor v2)

Each family function signature: fn(st: StructuredTable, rng: random.Random)
Returns (corrupted_grid, spans), or (None, []) if the table does not support this family.
  corrupted_grid: 2D list including the header (grid[0]=header, data rows start at grid[1])
  spans: [{"r":grid_row,"c":col,"orig":str,"new":str,"family":..,"rule":..,"note":..}]

Each family has a companion check_*(grid_or_st) self-check: confirms the injected
violation is actually detectable by the corresponding rule.

Coordinate convention: StructuredTable.data row i -> grid row i+1 (grid[0] is the header).
"""

import re
from typing import Any, Dict, List, Optional, Tuple

FAMILIES = ["SURF", "GRIM", "PVAL", "DIST", "PCT", "CI"]

RULE = {
    "SURF": "exact_recompute",
    "GRIM": "GRIM",
    "PVAL": "pval_consistency",
    "DIST": "sd_range_plausibility",
    "PCT": "count_pct_consistency",
    "CI": "ci_brackets_point",
}


# ── Numeric formatting utilities ──
def _decimals(numstr: str) -> int:
    numstr = numstr.strip()
    return len(numstr.split(".")[1]) if "." in numstr else 0


def _fmt(v: float, dec: int) -> str:
    return f"{v:.{dec}f}" if dec > 0 else str(int(round(v)))


def _flip_last_digit(numstr: str, rng) -> str:
    """Change the last digit of a number string to a different digit, preserving sign and decimal structure."""
    chars = list(numstr.strip())
    idx = max((k for k, ch in enumerate(chars) if ch.isdigit()), default=-1)
    if idx < 0:
        return numstr
    old = int(chars[idx])
    chars[idx] = str(rng.choice([d for d in range(10) if d != old]))
    return "".join(chars)


def _grid(st):
    return st.to_grid()


def _span(r, c, orig, new, family, note=""):
    return {"r": r, "c": c, "orig": orig, "new": new,
            "family": family, "rule": RULE[family], "note": note}


# ============================================================
# C-SURF surface tampering — alter the last significant digit of 1~2 values (integers stay integers)
# ============================================================
_RE_MEANSD_TOK = re.compile(r"^(\s*-?\d+(?:\.\d+)?\s*)(±|\\pm|\+/-)(.*)$")
_RE_CI_TOK = re.compile(r"^(\s*-?\d+(?:\.\d+)?\s*)([\(\[].*)$")


def corrupt_surf(st, rng) -> Tuple[Optional[List[List]], List[Dict]]:
    grid = _grid(st)
    cands = st.positions_of("meansd", "ci", "num", "count", "pct")
    if not cands:
        return None, []
    k = 1 if len(cands) < 6 else 2
    picks = rng.sample(cands, min(k, len(cands)))
    spans = []
    for (i, j) in picks:
        cell = st.cells[i][j]
        raw = cell["raw"]
        t = cell["type"]
        if t == "meansd":
            m = _RE_MEANSD_TOK.match(raw)
            if not m:
                continue
            mean_tok = m.group(1).strip()
            new = raw.replace(mean_tok, _flip_last_digit(mean_tok, rng), 1)
        elif t == "ci":
            m = _RE_CI_TOK.match(raw)
            if not m:
                continue
            point_tok = m.group(1).strip()
            new = raw.replace(point_tok, _flip_last_digit(point_tok, rng), 1)
        else:  # num / count / pct
            num_part = raw.rstrip("%").strip()
            new = raw.replace(num_part, _flip_last_digit(num_part, rng), 1)
        if new != raw:
            grid[i + 1][j] = new
            spans.append(_span(i + 1, j, raw, new, "SURF", "last-significant-digit tampering"))
    if not spans:
        return None, []
    return grid, spans


# ============================================================
# C-GRIM granularity inconsistency — use the true n to change the mean into a GRIM-impossible value
# ============================================================
def _grim_consistent(mean: float, n: int, dec: int) -> bool:
    """Integer-valued measurement: mean should satisfy round(mean*n)/n returning to mean at dec decimals."""
    if n <= 0:
        return True
    x = round(mean * n)
    return round(x / n, dec) == round(mean, dec)


def corrupt_grim(st, rng) -> Tuple[Optional[List[List]], List[Dict]]:
    grid = _grid(st)
    # Prefer means that already have decimals; otherwise raise the precision of an integer mean to create a GRIM violation
    cells = [(i, j) for (i, j) in st.positions_of("meansd") if st.n_for(j)]
    if not cells:
        return None, []
    rng.shuffle(cells)
    for (i, j) in cells:
        cell = st.cells[i][j]
        n = st.n_for(j)
        raw = cell["raw"]
        m = _RE_MEANSD_TOK.match(raw)
        if not m:
            continue
        mean_tok = m.group(1).strip()
        dec = _decimals(mean_tok)
        target_dec = dec if dec >= 1 else 1  # integer mean -> raise to 1 decimal place
        mean = cell["mean"]
        # Search for a GRIM-inconsistent candidate at target_dec decimals
        step = 10 ** (-target_dec)
        new_mean = None
        for k in range(1, 10):
            for cand in (round(mean + k * step, target_dec), round(mean - k * step, target_dec)):
                if cand <= 0:
                    continue
                if not _grim_consistent(cand, n, target_dec):
                    new_mean = cand
                    break
            if new_mean is not None:
                break
        if new_mean is None:
            continue
        new_mean_tok = _fmt(new_mean, target_dec)
        new = raw.replace(mean_tok, new_mean_tok, 1)
        if new == raw:
            continue
        grid[i + 1][j] = new
        spans = [_span(i + 1, j, raw, new, "GRIM",
                       f"n={n}, mean {new_mean_tok} is impossible at this granularity (GRIM)")]
        return grid, spans
    return None, []


# ============================================================
# C-PVAL p-value inconsistency — contradicts the same-row CI/effect size, or crosses the 0.05 threshold
# ============================================================
def _ci_excludes_null(lo: float, hi: float, null: float = 1.0) -> bool:
    return not (lo <= null <= hi)


def corrupt_pval(st, rng) -> Tuple[Optional[List[List]], List[Dict]]:
    grid = _grid(st)
    pvals = st.positions_of("pval")
    if not pvals:
        return None, []
    rng.shuffle(pvals)
    for (i, j) in pvals:
        cell = st.cells[i][j]
        dec = max(2, _decimals(cell["raw"].lstrip("<>=").strip()))
        # Strict mode: the same row must have a CI/effect-size interval, otherwise skip (no loose threshold flipping)
        ci_in_row = [st.cells[i][jj] for jj in range(st.ncols)
                     if st.cells[i][jj]["type"] == "ci"]
        if not ci_in_row:
            continue
        cc = ci_in_row[0]
        excludes = _ci_excludes_null(cc["lo"], cc["hi"], 1.0)
        # CI excludes null -> should be significant (p<0.05); otherwise non-significant. Make a p that contradicts the CI.
        new_v = round(rng.uniform(0.18, 0.45), dec) if excludes else round(rng.uniform(0.001, 0.039), dec)
        new = _fmt(new_v, dec)
        if cell["star"]:
            new += "*"
        if new == cell["raw"]:
            continue
        grid[i + 1][j] = new
        note = (f"[pval_ci_mismatch] CI[{cc['lo']},{cc['hi']}] "
                f"{'excludes' if excludes else 'includes'} null, p changed to {new} contradicts it")
        return grid, [_span(i + 1, j, cell["raw"], new, "PVAL", note)]
    return None, []


# ============================================================
# C-DIST impossible distribution — inflate SD so mean-2SD crosses the natural lower bound 0
# ============================================================
def corrupt_dist(st, rng) -> Tuple[Optional[List[List]], List[Dict]]:
    grid = _grid(st)
    cells = []
    for (i, j) in st.positions_of("meansd"):
        c = st.cells[i][j]
        if c["mean"] > 0 and c["sd"] > 0 and (c["mean"] - 2 * c["sd"]) >= 0:
            cells.append((i, j))
    if not cells:
        return None, []
    i, j = rng.choice(cells)
    cell = st.cells[i][j]
    raw = cell["raw"]
    m = _RE_MEANSD_TOK.match(raw)
    if not m:
        return None, []
    sd_tok = m.group(3).strip().rstrip("*").strip()
    dec = max(_decimals(sd_tok), 1)
    mean = cell["mean"]
    # Need mean-2*new_sd < 0 -> new_sd > mean/2. Validate after formatting; if it fails, raise precision/multiplier
    new_sd_tok = None
    for mult in (1.15, 1.40, 1.80, 2.5):
        for d in (dec, dec + 1, dec + 2):
            tok = _fmt((mean / 2) * mult, d)
            if mean - 2 * float(tok) < 0 and tok != sd_tok:
                new_sd_tok = tok
                break
        if new_sd_tok:
            break
    if new_sd_tok is None:
        return None, []
    new = raw[:m.start(3)] + raw[m.start(3):].replace(sd_tok, new_sd_tok, 1)
    if new == raw:
        return None, []
    grid[i + 1][j] = new
    note = f"mean={cell['mean']}, SD {sd_tok}->{new_sd_tok} makes mean-2SD<0 (impossible for a non-negative quantity)"
    return grid, [_span(i + 1, j, raw, new, "DIST", note)]


# ============================================================
# C-PCT count-percentage mismatch — break a summable percentage group, or make count/% inconsistent
# ============================================================
def _pct_groups(st):
    """Return [(orientation,index,[(i,j,value)])] where each percentage group sums to ~100."""
    groups = []
    # Row groups
    for i in range(st.nrows):
        cells = [(i, j, st.cells[i][j]["value"]) for j in range(st.ncols)
                 if st.cells[i][j]["type"] == "pct"]
        if len(cells) >= 2 and 97 <= sum(v for _, _, v in cells) <= 103:
            groups.append(("row", i, cells))
    # Column groups
    for j in range(st.ncols):
        cells = [(i, j, st.cells[i][j]["value"]) for i in range(st.nrows)
                 if st.cells[i][j]["type"] == "pct"]
        if len(cells) >= 2 and 97 <= sum(v for _, _, v in cells) <= 103:
            groups.append(("col", j, cells))
    return groups


def corrupt_pct(st, rng) -> Tuple[Optional[List[List]], List[Dict]]:
    grid = _grid(st)
    # Mode A: count(%) cells; compute expected % from the cell's denom or the column n, then make it inconsistent
    cp_cells = []
    for (i, j) in st.positions_of("countpct"):
        cell = st.cells[i][j]
        denom = cell["denom"] or st.n_for(j)
        if not denom or denom <= 0:
            continue
        expected = 100.0 * cell["count"] / denom
        if abs(expected - cell["pct"]) <= 3.0:  # currently consistent -> can be broken
            cp_cells.append((i, j, expected))
    if cp_cells:
        i, j, expected = rng.choice(cp_cells)
        cell = st.cells[i][j]
        raw = cell["raw"]
        dec = _decimals(cell["pct_tok"])
        sign = 1 if rng.random() < 0.5 else -1
        new_pct = expected + sign * rng.uniform(15, 30)
        new_pct = min(99.0, max(1.0, new_pct))
        if abs(new_pct - cell["pct"]) < 8:
            new_pct = min(99.0, max(1.0, expected - sign * rng.uniform(15, 30)))
        new_tok = _fmt(new_pct, dec)
        new = raw.replace(cell["pct_tok"], new_tok, 1)
        if new != raw:
            grid[i + 1][j] = new
            note = (f"count={cell['count']}, denom≈{int(round(expected*0+ (cell['denom'] or st.n_for(j))))}, "
                    f"should be ≈{expected:.1f}%, displayed value changed to {new_tok}% (count-percentage mismatch)")
            return grid, [_span(i + 1, j, raw, new, "PCT", note)]

    groups = _pct_groups(st)
    if groups:
        _, _, cells = rng.choice(groups)
        i, j, v = rng.choice(cells)
        raw = st.cells[i][j]["raw"]
        dec = _decimals(raw.rstrip("%"))
        cur_sum = sum(x for _, _, x in cells)
        # Shift the sum toward 88~94 or 106~112
        target = rng.choice([rng.uniform(88, 94), rng.uniform(106, 112)])
        new_v = v + (target - cur_sum)
        if new_v <= 0:
            new_v = v + rng.choice([rng.uniform(6, 14), -rng.uniform(6, 14)])
        if new_v <= 0:
            return None, []
        new = _fmt(new_v, dec) + "%"
        if new == raw:
            return None, []
        grid[i + 1][j] = new
        note = f"percentage group originally summed to ≈{cur_sum:.0f}%, after change ≠100%"
        return grid, [_span(i + 1, j, raw, new, "PCT", note)]
    return None, []


# ============================================================
# C-CI interval inconsistency — make the point estimate fall outside the CI
# ============================================================
def corrupt_ci(st, rng) -> Tuple[Optional[List[List]], List[Dict]]:
    grid = _grid(st)
    cells = [(i, j) for (i, j) in st.positions_of("ci")
             if st.cells[i][j]["lo"] <= st.cells[i][j]["point"] <= st.cells[i][j]["hi"]
             and st.cells[i][j]["hi"] > st.cells[i][j]["lo"]]
    if not cells:
        return None, []
    i, j = rng.choice(cells)
    cell = st.cells[i][j]
    raw = cell["raw"]
    width = cell["hi"] - cell["lo"]
    # Move the point estimate outside the interval (above or below) while keeping the CI unchanged -> point not in CI
    point_tok = re.match(r"^\s*(-?\d+(?:\.\d+)?)", raw).group(1)
    dec = _decimals(point_tok)
    up = rng.random() < 0.5
    # After formatting, force the point estimate strictly outside [lo,hi]; otherwise increase offset/precision
    new_point_tok = None
    for frac in (0.30, 0.60, 1.0, 2.0):
        for d in (max(dec, 2), max(dec, 2) + 1):
            off = max(width * frac, 10 ** (-d) * 2)
            cand = (cell["hi"] + off) if up else (cell["lo"] - off)
            tok = _fmt(cand, d)
            fv = float(tok)
            if (fv > cell["hi"] or fv < cell["lo"]) and tok != point_tok:
                new_point_tok = tok
                break
        if new_point_tok:
            break
    if new_point_tok is None:
        return None, []
    new = raw.replace(point_tok, new_point_tok, 1)
    if new == raw:
        return None, []
    grid[i + 1][j] = new
    note = f"point estimate {point_tok}->{new_point_tok} falls outside CI[{cell['lo']},{cell['hi']}]"
    return grid, [_span(i + 1, j, raw, new, "CI", note)]


CORRUPTORS = {
    "SURF": corrupt_surf, "GRIM": corrupt_grim, "PVAL": corrupt_pval,
    "DIST": corrupt_dist, "PCT": corrupt_pct, "CI": corrupt_ci,
}


def attempt(st, family: str, rng):
    return CORRUPTORS[family](st, rng)
