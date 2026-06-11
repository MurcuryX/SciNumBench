"""
detector_engine.py — SciNumDSL expert rule engine (Stage 1)

Detects fabricated tabular data using classic statistical laws:
  - GRIM test: mean granularity consistency
  - GRIMMER test: SD granularity consistency
  - Benford's law: leading-digit distribution
  - P-value self-consistency: back-computed t test
  - Boundary check: whether the mean lies within [Min, Max]
  - Percentage sum: whether category proportions close to 100%
  - Distribution plausibility: whether Mean - 2SD is non-negative

Input: 2D list (table grid)
Output: detection result (rule pass/fail + confidence + overall score)
"""

import re
import math
import json
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass, field

# ── numeric parsing helpers ──

def _try_float(s: Any) -> bool:
    if s is None:
        return False
    s = str(s).strip()
    if not s:
        return False
    s = s.rstrip('%').replace(',', '').replace(' ', '')
    s = re.sub(r'±.*$', '', s)
    s = re.sub(r'[×x]\s*10\^?', 'e', s)
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def _to_float(s: Any) -> float:
    s = str(s).strip().rstrip('%').replace(',', '').replace(' ', '')
    s = re.sub(r'±.*$', '', s)
    s = re.sub(r'[×x]\s*10\^?', 'e', s)
    return float(s)


def _has_percent(s: Any) -> bool:
    return str(s).strip().endswith('%')


# ── keyword detection ──

MEAN_KW = re.compile(
    r"(?i)^(mean|average|avg|m|μ|mu|x[\s_-]?bar|expectation)$"
)
SD_KW = re.compile(
    r"(?i)^(sd|std|std[\s._]dev|standard[\s_-]deviation|σ|sigma|se|sem|s\.d\.)$"
)
PVALUE_KW = re.compile(
    r"(?i)(p[\s._-]?(value|val|sig|level)|significance|p\s*[<>=]|pr\s*[><=])"
)
MIN_KW = re.compile(r"(?i)^(min|minimum|lower)$")
MAX_KW = re.compile(r"(?i)^(max|maximum|upper)$")
N_KW = re.compile(r"(?i)^(\bn\b|n\s*=|sample[\s_-]?size|count|freq|frequency)$")


# ============================================================
# Detection result data structures
# ============================================================

@dataclass
class RuleResult:
    """Detection result of a single rule."""
    name: str
    passed: bool
    confidence: float  # 0~1, confidence in this rule's judgement
    detail: str = ""
    corruption_type: str = ""  # corruption type: L1/L2/L3/L6-A/L6-B/L6-C

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "confidence": round(self.confidence, 3),
            "detail": self.detail,
            "corruption_type": self.corruption_type,
        }


@dataclass
class DetectionResult:
    """Overall detection result."""
    grid_id: str = ""
    score: float = 0.0  # overall anomaly score (0~1, higher = more suspicious)
    verdict: str = "Clean"  # Clean / Corrupted
    corruption_type: str = ""  # corruption type: L1/L2/L3/L6-A/L6-B/L6-C
    confidence: float = 0.0  # confidence in the verdict
    rules: List[RuleResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "grid_id": self.grid_id,
            "score": round(self.score, 3),
            "verdict": self.verdict,
            "corruption_type": self.corruption_type,
            "confidence": round(self.confidence, 3),
            "rules": [r.to_dict() for r in self.rules],
        }


# ============================================================
# Table structure parsing
# ============================================================

def _parse_table_structure(grid: List[List]) -> Dict[str, Any]:
    """
    Parse the table structure, extracting positions and values of statistics.
    Returns:
        {
            "mean_positions": [(row, col, value), ...],
            "sd_positions": [(row, col, value), ...],
            "min_positions": [(row, col, value), ...],
            "max_positions": [(row, col, value), ...],
            "n_positions": [(row, col, value), ...],
            "pvalue_positions": [(row, col, value), ...],
            "percentage_positions": [(row, col, value), ...],
            "all_numerics": [(row, col, value), ...],
            "header": [str, ...],
        }
    """
    result = {
        "mean_positions": [],
        "sd_positions": [],
        "min_positions": [],
        "max_positions": [],
        "n_positions": [],
        "pvalue_positions": [],
        "percentage_positions": [],
        "all_numerics": [],
        "header": [],
    }

    if not grid or not grid[0]:
        return result

    result["header"] = [str(c).strip() for c in grid[0]]

    for i, row in enumerate(grid):
        for j, cell in enumerate(row):
            cell_str = str(cell).strip()
            if not cell_str:
                continue

            # record all numeric values
            if _try_float(cell_str):
                val = _to_float(cell_str)
                result["all_numerics"].append((i, j, val))

            # record percentages (detected independently of _try_float)
            if _has_percent(cell_str):
                # extract the numeric part
                pct_str = cell_str.rstrip('%').strip()
                try:
                    pct_val = float(pct_str)
                    result["percentage_positions"].append((i, j, pct_val))
                except (ValueError, TypeError):
                    pass

            # check label columns (usually the first column or header)
            if j == 0 or (i == 0):
                if MEAN_KW.match(cell_str):
                    # find the numeric value to the right in the same row
                    for k in range(j + 1, len(row)):
                        if _try_float(str(row[k])):
                            result["mean_positions"].append((i, k, _to_float(row[k])))
                            break
                elif SD_KW.match(cell_str):
                    for k in range(j + 1, len(row)):
                        if _try_float(str(row[k])):
                            result["sd_positions"].append((i, k, _to_float(row[k])))
                            break
                elif MIN_KW.match(cell_str):
                    for k in range(j + 1, len(row)):
                        if _try_float(str(row[k])):
                            result["min_positions"].append((i, k, _to_float(row[k])))
                            break
                elif MAX_KW.match(cell_str):
                    for k in range(j + 1, len(row)):
                        if _try_float(str(row[k])):
                            result["max_positions"].append((i, k, _to_float(row[k])))
                            break
                elif N_KW.match(cell_str):
                    for k in range(j + 1, len(row)):
                        if _try_float(str(row[k])):
                            n_val = _to_float(row[k])
                            if n_val > 0 and n_val == int(n_val):
                                result["n_positions"].append((i, k, int(n_val)))
                            break

            # P-value detection (any position)
            if PVALUE_KW.search(cell_str):
                for k in range(j + 1, len(row)):
                    if _try_float(str(row[k])):
                        result["pvalue_positions"].append((i, k, _to_float(row[k])))
                        break

    # column-wise detection: header in the first row
    if not result["mean_positions"] and len(grid) >= 2:
        header = [str(c).strip() for c in grid[0]]
        for j, h in enumerate(header):
            if MEAN_KW.match(h):
                for i in range(1, len(grid)):
                    if j < len(grid[i]) and _try_float(str(grid[i][j])):
                        result["mean_positions"].append((i, j, _to_float(grid[i][j])))
                        break
            elif SD_KW.match(h):
                for i in range(1, len(grid)):
                    if j < len(grid[i]) and _try_float(str(grid[i][j])):
                        result["sd_positions"].append((i, j, _to_float(grid[i][j])))
                        break
            elif MIN_KW.match(h):
                for i in range(1, len(grid)):
                    if j < len(grid[i]) and _try_float(str(grid[i][j])):
                        result["min_positions"].append((i, j, _to_float(grid[i][j])))
                        break
            elif MAX_KW.match(h):
                for i in range(1, len(grid)):
                    if j < len(grid[i]) and _try_float(str(grid[i][j])):
                        result["max_positions"].append((i, j, _to_float(grid[i][j])))
                        break
            elif N_KW.match(h):
                for i in range(1, len(grid)):
                    if j < len(grid[i]) and _try_float(str(grid[i][j])):
                        n_val = _to_float(grid[i][j])
                        if n_val > 0 and n_val == int(n_val):
                            result["n_positions"].append((i, j, int(n_val)))
                        break
            elif PVALUE_KW.search(h):
                for i in range(1, len(grid)):
                    if j < len(grid[i]) and _try_float(str(grid[i][j])):
                        result["pvalue_positions"].append((i, j, _to_float(grid[i][j])))
                        break

    return result


# ============================================================
# Rule implementations
# ============================================================

def rule_grim(struct: Dict[str, Any], default_n: int = 30) -> RuleResult:
    """
    GRIM test: mean granularity consistency.
    Given N and scale precision (integer or one decimal), Mean x N must be an
    integer (or a multiple matching the precision).
    """
    means = struct["mean_positions"]
    ns = struct["n_positions"]
    n = ns[0][2] if ns else default_n

    if not means:
        return RuleResult("GRIM", True, 0.0, "Mean not found, skipped")

    failures = []
    for row, col, mean_val in means:
        # infer scale precision from the number of decimals in the mean
        # for an integer scale, Mean x N must be an integer
        product = mean_val * n
        # allow small error (floating-point precision)
        nearest_int = round(product)
        if abs(product - nearest_int) > 0.01:
            failures.append(f"Mean={mean_val} x N={n} = {product:.2f} (not integer)")

    if failures:
        return RuleResult("GRIM", False, 0.9,
                          detail="; ".join(failures),
                          corruption_type="L3")  # GRIM failure -> mean fabricated
    return RuleResult("GRIM", True, 0.8, detail=f"Mean x N integer check passed (N={n})")


def rule_grimmer(struct: Dict[str, Any], default_n: int = 30) -> RuleResult:
    """
    GRIMMER test: SD granularity consistency.
    Given N, Mean and scale precision, the SD must satisfy mathematical
    constraints. Simplified version: check whether SD^2 x (N-1) is close to an
    integer (i.e. the sum of squares).
    """
    means = struct["mean_positions"]
    sds = struct["sd_positions"]
    ns = struct["n_positions"]
    n = ns[0][2] if ns else default_n

    if not means or not sds:
        return RuleResult("GRIMMER", True, 0.0, "Mean/SD not found, skipped")

    failures = []
    for (_, _, mean_val), (_, _, sd_val) in zip(means, sds):
        if sd_val <= 0:
            continue
        # SS = SD^2 x (N-1), should be positive
        ss = (sd_val ** 2) * (n - 1)
        if ss < 0:
            failures.append(f"SD={sd_val}, SS={ss:.2f} < 0")
            continue
        # SS should be close to an integer (if the original data are integers)
        nearest_int = round(ss)
        # be lenient: allow 5% error
        if abs(ss - nearest_int) > 0.05 * max(abs(ss), 1):
            failures.append(f"SD={sd_val}^2x(N-1)={ss:.2f} (not integer)")

    if failures:
        # decide L3 vs L6-C: if Mean-2SD < 0, more likely L6-C
        has_negative = any("Mean-2SD" in f or "< 0" in f for f in failures)
        corruption_type = "L6-C" if has_negative else "L3"
        return RuleResult("GRIMMER", False, 0.85,
                          detail="; ".join(failures),
                          corruption_type=corruption_type)
    return RuleResult("GRIMMER", True, 0.7, detail="SD granularity check passed")


def rule_benford(struct: Dict[str, Any]) -> RuleResult:
    """
    Benford's law: leading-digit distribution.
    In natural data the leading digit d occurs with probability
    ~ log10(1 + 1/d). A chi-square test decides whether the deviation is
    significant.
    """
    numerics = struct["all_numerics"]

    if len(numerics) < 20:
        return RuleResult("Benford", True, 0.0, f"too few values ({len(numerics)}<20), skipped")

    # extract leading digits
    first_digits = []
    for _, _, val in numerics:
        if val == 0:
            continue
        abs_val = abs(val)
        first_digit = int(str(abs_val).lstrip('0').replace('.', '')[0])
        if 1 <= first_digit <= 9:
            first_digits.append(first_digit)

    if len(first_digits) < 20:
        return RuleResult("Benford", True, 0.0, "too few valid leading digits, skipped")

    # Benford expected distribution
    expected_probs = {d: math.log10(1 + 1 / d) for d in range(1, 10)}
    n = len(first_digits)

    # observed frequencies
    observed = {d: first_digits.count(d) for d in range(1, 10)}

    # chi-square statistic
    chi2 = 0
    for d in range(1, 10):
        expected = expected_probs[d] * n
        if expected > 0:
            chi2 += (observed[d] - expected) ** 2 / expected

    # df 8, critical values: alpha=0.05 -> 15.51, alpha=0.01 -> 20.09
    if chi2 > 20.09:
        # Benford anomaly -> L1 (surface tampering) or L6 (self-contradiction)
        # default to L1, since Benford mainly detects manually altered digits
        return RuleResult("Benford", False, 0.85,
                          detail=f"chi2={chi2:.1f} > 20.09 (p<0.01)",
                          corruption_type="L1")
    elif chi2 > 15.51:
        return RuleResult("Benford", False, 0.6,
                          detail=f"chi2={chi2:.1f} > 15.51 (p<0.05)",
                          corruption_type="L1")
    else:
        return RuleResult("Benford", True, 0.7,
                          detail=f"chi2={chi2:.1f} (normal)")


def rule_pvalue_consistency(struct: Dict[str, Any], default_n: int = 30) -> RuleResult:
    """
    P-value self-consistency: given Mean, SD, N, back-compute the t and p values
    and check for consistency. Hypothesis test: H0: mu = 0 (or compared with the
    adjacent row).
    """
    means = struct["mean_positions"]
    sds = struct["sd_positions"]
    pvals = struct["pvalue_positions"]
    ns = struct["n_positions"]
    n = ns[0][2] if ns else default_n

    if not means or not sds or not pvals:
        return RuleResult("P-value-consistency", True, 0.0, "incomplete statistics, skipped")

    # try to pair Mean/SD/P-value
    # simplified: pair by row
    failures = []
    for (m_row, _, m_val), (s_row, _, s_val) in zip(means, sds):
        if s_val <= 0:
            continue
        # find the P-value in the same row
        p_val = None
        for p_row, _, pv in pvals:
            if abs(p_row - m_row) <= 1:  # allow a row difference of 1
                p_val = pv
                break

        if p_val is None:
            continue

        # back-compute the t value
        # t = Mean / (SD / sqrt(N))
        t_stat = abs(m_val) / (s_val / math.sqrt(n))

        # compute p from t (two-sided test, df = N-1)
        # approximation: p ~ 2 * (1 - Phi(|t|)) for large N
        # more precise: use the t distribution
        from scipy import stats as sp_stats
        df = n - 1
        computed_p = 2 * (1 - sp_stats.t.cdf(abs(t_stat), df))

        # check consistency (allow 0.05 error)
        if abs(computed_p - p_val) > 0.1 and (computed_p < 0.05) != (p_val < 0.05):
            failures.append(
                f"Mean={m_val:.2f}, SD={s_val:.2f}, "
                f"reported_p={p_val:.4f}, computed_p={computed_p:.4f}"
            )

    if failures:
        # P-value inconsistent with Mean/SD -> P-value tampered (L2)
        # note: if GRIM/GRIMMER also fail, Mean/SD were tampered too, that is L3
        # but here we only judge P-value consistency, so classify as L2
        return RuleResult("P-value-consistency", False, 0.85,
                          detail="; ".join(failures[:3]),
                          corruption_type="L2")
    return RuleResult("P-value-consistency", True, 0.6, detail="P-value consistent with statistics")


def rule_boundary_check(struct: Dict[str, Any]) -> RuleResult:
    """
    Boundary check: whether the Mean lies within [Min, Max].
    """
    means = struct["mean_positions"]
    mins = struct["min_positions"]
    maxs = struct["max_positions"]

    if not means:
        return RuleResult("boundary-check", True, 0.0, "Mean not found, skipped")

    failures = []
    for (m_row, _, m_val) in means:
        # find Min/Max in the same or adjacent row
        row_min = None
        row_max = None

        for r, _, v in mins:
            if abs(r - m_row) <= 1:
                row_min = v
                break
        for r, _, v in maxs:
            if abs(r - m_row) <= 1:
                row_max = v
                break

        if row_min is not None and m_val < row_min:
            failures.append(f"Mean={m_val:.2f} < Min={row_min:.2f}")
        if row_max is not None and m_val > row_max:
            failures.append(f"Mean={m_val:.2f} > Max={row_max:.2f}")

    if failures:
        return RuleResult("boundary-check", False, 0.95,
                          detail="; ".join(failures),
                          corruption_type="L6-A")
    return RuleResult("boundary-check", True, 0.5, detail="mean within range")


def rule_percentage_sum(struct: Dict[str, Any]) -> RuleResult:
    """
    Percentage sum: percentages in the same group should add up to 100%.
    Checks both row sums and column sums.
    """
    pct_positions = struct["percentage_positions"]

    if len(pct_positions) < 2:
        return RuleResult("percentage-sum", True, 0.0, "fewer than 2 percentages, skipped")

    failures = []

    # check row sums
    row_groups = {}
    for row, col, val in pct_positions:
        if row not in row_groups:
            row_groups[row] = []
        row_groups[row].append((col, val))

    for row, cells in row_groups.items():
        if len(cells) >= 2:
            total = sum(v for _, v in cells)
            if abs(total - 100) > 2:
                vals_str = ", ".join(f"{v:.1f}%" for _, v in cells)
                failures.append(f"row{row}: {vals_str} = {total:.1f}%")

    # check column sums (percentages in the same column should add to 100%)
    col_groups = {}
    for row, col, val in pct_positions:
        if col not in col_groups:
            col_groups[col] = []
        col_groups[col].append((row, val))

    for col, cells in col_groups.items():
        if len(cells) >= 2:
            total = sum(v for _, v in cells)
            if abs(total - 100) > 2:
                vals_str = ", ".join(f"{v:.1f}%" for _, v in cells)
                failures.append(f"col{col}: {vals_str} = {total:.1f}%")

    if failures:
        return RuleResult("percentage-sum", False, 0.9,
                          detail="; ".join(failures),
                          corruption_type="L6-B")
    return RuleResult("percentage-sum", True, 0.6, detail="percentage sums normal")


def rule_distribution_possible(struct: Dict[str, Any]) -> RuleResult:
    """
    Distribution plausibility: Mean - 2SD >= Min (non-negative data).
    If Mean - 2SD < 0 it implies the distribution has negative values, which the
    scenario does not allow.
    """
    means = struct["mean_positions"]
    sds = struct["sd_positions"]
    mins = struct["min_positions"]

    if not means or not sds:
        return RuleResult("distribution-possible", True, 0.0, "Mean/SD not found, skipped")

    failures = []
    for (m_row, _, m_val), (s_row, _, s_val) in zip(means, sds):
        lower_bound = m_val - 2 * s_val

        # find Min in the same row
        row_min = None
        for r, _, v in mins:
            if abs(r - m_row) <= 1:
                row_min = v
                break

        # if Min exists and Min >= 0, check Mean-2SD
        if row_min is not None and row_min >= 0 and lower_bound < 0:
            failures.append(
                f"Mean={m_val:.2f}, SD={s_val:.2f}, "
                f"Mean-2SD={lower_bound:.2f} < 0 (Min={row_min:.2f})"
            )
        # if no Min but Mean-2SD is far below 0 (e.g. -50%), also suspicious
        elif lower_bound < -abs(m_val) * 0.5:
            failures.append(
                f"Mean={m_val:.2f}, SD={s_val:.2f}, "
                f"Mean-2SD={lower_bound:.2f} (extreme negative)"
            )

    if failures:
        return RuleResult("distribution-possible", False, 0.8,
                          detail="; ".join(failures),
                          corruption_type="L6-C")
    return RuleResult("distribution-possible", True, 0.5, detail="distribution reasonable")


# ============================================================
# Main detector
# ============================================================

class SciNumDSL:
    """
    SciNumDSL expert rule engine.
    Combines multiple statistical rules and outputs an overall verdict.
    """

    # rule weights (used to compute the overall score)
    RULE_WEIGHTS = {
        "GRIM": 0.20,
        "GRIMMER": 0.15,
        "Benford": 0.15,
        "P-value-consistency": 0.20,
        "boundary-check": 0.10,
        "percentage-sum": 0.10,
        "distribution-possible": 0.10,
    }

    # thresholds
    THRESHOLD_CORRUPTED = 0.5  # overall score > this value -> Corrupted
    THRESHOLD_SUSPECT = 0.2    # overall score > this value -> Suspect

    def __init__(self, default_n: int = 30):
        self.default_n = default_n

    def detect(self, grid: List[List], grid_id: str = "") -> DetectionResult:
        """
        Detect a single table.

        Args:
            grid: 2D list table data
            grid_id: optional identifier

        Returns:
            DetectionResult
        """
        result = DetectionResult(grid_id=grid_id)

        # parse the table structure
        struct = _parse_table_structure(grid)

        # run all rules
        rules = [
            rule_grim(struct, self.default_n),
            rule_grimmer(struct, self.default_n),
            rule_benford(struct),
            rule_pvalue_consistency(struct, self.default_n),
            rule_boundary_check(struct),
            rule_percentage_sum(struct),
            rule_distribution_possible(struct),
        ]

        result.rules = rules

        # compute the overall anomaly score
        # strategy: any failed rule should raise the score significantly
        failed_scores = []
        total_weight = 0.0

        for rule in rules:
            weight = self.RULE_WEIGHTS.get(rule.name, 0.1)
            if not rule.passed:
                # base score of a single failure = weight x confidence
                # then multiplied by an amplification factor so a single failure
                # can also exceed the threshold
                base_score = weight * rule.confidence
                failed_scores.append(base_score)
            total_weight += weight

        if failed_scores:
            # take the highest as the main score, other failures accumulate (decayed)
            max_score = max(failed_scores)
            other_sum = sum(failed_scores) - max_score
            # main score + 30% contribution from other failures
            result.score = min(1.0, max_score * 2.5 + other_sum * 0.3)
        else:
            result.score = 0.0

        # verdict
        if result.score > self.THRESHOLD_SUSPECT:
            result.verdict = "Corrupted"
        else:
            result.verdict = "Clean"

        # determine corruption type: take the type of the highest-confidence failed rule
        failed_rules = [r for r in rules if not r.passed and r.corruption_type]
        if failed_rules:
            # sort by confidence, take the highest
            best_rule = max(failed_rules, key=lambda r: r.confidence)
            result.corruption_type = best_rule.corruption_type
            result.confidence = best_rule.confidence
        else:
            result.corruption_type = ""
            result.confidence = 0.0

        return result

    def detect_batch(self, grids: List[Tuple[str, List[List]]]) -> List[DetectionResult]:
        """
        Batch detection.

        Args:
            grids: [(grid_id, grid), ...]

        Returns:
            [DetectionResult, ...]
        """
        return [self.detect(grid, gid) for gid, grid in grids]


# ============================================================
# Test / demo
# ============================================================

def _demo():
    """Demonstrate usage."""
    engine = SciNumDSL(default_n=30)

    # Test 1: normal table
    grid_clean = [
        ["Variable", "Min", "Max", "Mean", "SD", "p-value", "Category", "Pct"],
        ["Score", "10", "50", "30", "8", "0.12", "TypeA", "45%"],
        ["Age", "18", "65", "35", "12", "0.08", "TypeB", "55%"],
        ["BMI", "15", "40", "22", "3", "0.45", "", ""],
    ]
    result = engine.detect(grid_clean, "clean_001")
    _print_result(result)

    # Test 2: L6-A (mean out of range)
    grid_l6a = [
        ["Variable", "Min", "Max", "Mean", "SD"],
        ["Score", "10", "50", "65", "8"],  # Mean > Max
    ]
    result = engine.detect(grid_l6a, "l6a_001")
    _print_result(result)

    # Test 3: L6-B (percentages do not sum to 100)
    grid_l6b = [
        ["Category", "Pct"],
        ["Male", "45%"],
        ["Female", "60%"],  # sum 105%
    ]
    result = engine.detect(grid_l6b, "l6b_001")
    _print_result(result)

    # Test 4: L6-C (impossible distribution)
    grid_l6c = [
        ["Variable", "Min", "Mean", "SD"],
        ["Height", "0", "170", "100"],  # Mean-2SD = -30 < 0
    ]
    result = engine.detect(grid_l6c, "l6c_001")
    _print_result(result)

    # Test 5: Benford anomaly (unnatural leading digits)
    # synthetically construct data whose leading digit is always 9
    grid_benford = [["Data"]] + [[str(9 * 10 ** (i % 3) + random_int())]
                                  for i in range(30)]
    result = engine.detect(grid_benford, "benford_001")
    _print_result(result)


def random_int():
    import random
    return random.randint(1, 99)


def _print_result(result: DetectionResult):
    """Print the detection result in a formatted way."""
    print(f"  verdict: {result.verdict} (score: {result.score:.3f})")

    if result.flags:
        print(f"  flags: {', '.join(result.flags)}")

    print("  rule details:")
    for rule in result.rules:
        status = "[PASS]" if rule.passed else "[FAIL]"
        conf = f"{rule.confidence:.2f}"
        print(f"    {status} {rule.name:<12} [{conf}] {rule.detail}")


if __name__ == "__main__":
    _demo()
