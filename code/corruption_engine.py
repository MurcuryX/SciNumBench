"""
corruption_engine.py — SciNumBench benchmark dataset construction engine

Selects numeric tables from extracted_tables in arxiv_data.db,
injects corruptions at multiple difficulty levels, and splits into Train/Val/Test sets.

Usage:
  D:\Python\Python\Python\python.exe corruption_engine.py
"""

import json
import random
import sqlite3
import re
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

# ── Path configuration ──
DB_PATH = r"D:\coding package\electric detect\arxiv_data.db"
PYTHON_EXE = r"D:\Python\Python\Python\python.exe"

# ── Sampling configuration ──
TOTAL_BENCH = 6000
SPLIT_RATIO = {"train": 3400, "val": 600, "test": 2000}
CORRUPTION_DIST = {
    "train": {"clean": 1700, "L1": 510, "L2": 680, "L3": 340, "L6": 170},
    "val":   {"clean": 300, "L1": 90, "L2": 120, "L3": 60, "L6": 30},
    "test":  {"clean": 1800, "L1": 60, "L2": 80, "L3": 40, "L6": 20},
}
# Total: Train=3400, Val=600, Test=2000
# Fake ratio: L1=30%, L2=40%, L3=20%, L6=10%

# ── Numeric filter threshold ──
NUMERIC_THRESHOLD = 0.30

# ── P-value detection regex (broad) ──
RE_PVALUE_LABEL = re.compile(
    r"(?i)(p[\s._-]?(value|val|sig|level)|significance|p\s*[<>=]|pr\s*[><=])"
)

# ── Mean/SD detection keywords (covers various spellings) ──
MEAN_KEYWORDS = re.compile(
    r"(?i)^(mean|average|avg|m|μ|mu|x[\s_-]?bar|expectation)$"
)
SD_KEYWORDS = re.compile(
    r"(?i)^(sd|std|std[\s._]dev|standard[\s_-]deviation|σ|sigma|se|sem|s\.d\.)$"
)


# ============================================================
# Database table creation
# ============================================================

def create_bench_table(conn: sqlite3.Connection):
    """Create the scinum_bench table if it does not exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scinum_bench (
            bench_id         INTEGER PRIMARY KEY AUTOINCREMENT,
            arxiv_id         TEXT NOT NULL,
            table_id         INTEGER NOT NULL,
            dataset_split    TEXT NOT NULL,
            original_grid    TEXT NOT NULL,
            corrupted_grid   TEXT NOT NULL,
            corruption_level TEXT NOT NULL,
            is_human_modified INTEGER DEFAULT 0,
            UNIQUE(arxiv_id, table_id, dataset_split, corruption_level)
        )
    """)
    conn.commit()


# ============================================================
# Numeric parsing utilities
# ============================================================

def _try_float(s: Any) -> bool:
    """Try to convert a value to float, supporting various formats."""
    if s is None:
        return False
    s = str(s).strip()
    if not s:
        return False
    # Strip percent sign, commas, spaces, and ± suffix
    s = s.rstrip('%').replace(',', '').replace(' ', '')
    # Strip ± and everything after it (e.g. "3.14 ± 0.02" -> "3.14")
    s = re.sub(r'±.*$', '', s)
    # Handle scientific notation (e.g. "1.2x10^3" -> "1.2e3")
    s = re.sub(r'[×x]\s*10\^?', 'e', s)
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def _to_float(s: Any) -> float:
    """Safely convert to float."""
    s = str(s).strip().rstrip('%').replace(',', '').replace(' ', '')
    s = re.sub(r'±.*$', '', s)
    s = re.sub(r'[×x]\s*10\^?', 'e', s)
    return float(s)


def _float_to_str(f: float, original: Any) -> str:
    """Convert a float back to a string, preserving the original formatting style."""
    orig = str(original).strip()
    # Handle special values
    if not np.isfinite(f):
        return orig  # do not write NaN/Inf
    # Preserve percent sign
    if orig.endswith('%'):
        return f"{f:.2f}%"
    # Preserve original number of decimal places
    if '.' in orig:
        clean = orig.rstrip('%').split('.')[-1]
        decimals = len(clean)
        return f"{f:.{decimals}f}"
    return str(int(f)) if f == int(f) else str(f)


# ============================================================
# Numeric table filtering
# ============================================================

def _count_numeric_cells(grid: List[List]) -> Tuple[int, int]:
    """Count the number of cells convertible to float."""
    total = 0
    numeric = 0
    for row in grid:
        for cell in row:
            cell_str = str(cell).strip()
            if not cell_str:
                continue
            total += 1
            if _try_float(cell_str):
                numeric += 1
    return numeric, total


def is_numeric_table(grid: List[List], threshold: float = NUMERIC_THRESHOLD) -> bool:
    """Determine whether this is a numeric statistics table."""
    numeric, total = _count_numeric_cells(grid)
    if total == 0:
        return False
    return (numeric / total) > threshold


def _parse_grid_from_json(table_json: str) -> List[List] | None:
    """Parse a 2D grid from table_json, supporting multiple formats."""
    raw = json.loads(table_json)
    # orient="split" -> {"columns": [...], "data": [...]}
    if isinstance(raw, dict):
        if "data" in raw:
            grid = raw["data"]
        elif "values" in raw:
            grid = raw["values"]
        else:
            return None
    elif isinstance(raw, list):
        grid = raw
    else:
        return None

    if not isinstance(grid, list) or len(grid) < 2:
        return None

    # Ensure each row is a list
    return [list(row) if isinstance(row, (list, tuple)) else [row] for row in grid]


def load_and_filter_tables(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load and filter numeric tables from extracted_tables."""
    df = pd.read_sql_query("""
        SELECT t.table_id, t.arxiv_id, t.table_json, t.caption, t.label, t.section,
               r.primary_category
        FROM extracted_tables t
        JOIN raw_papers r ON t.arxiv_id = r.arxiv_id
    """, conn)

    valid_rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Filtering numeric tables"):
        try:
            grid = _parse_grid_from_json(row["table_json"])
            if grid is not None and is_numeric_table(grid):
                valid_rows.append(row)
        except (json.JSONDecodeError, TypeError, KeyError):
            continue

    result = pd.DataFrame(valid_rows)

    if len(result) < TOTAL_BENCH:
        print(f"[WARN] Fewer than {TOTAL_BENCH} valid tables; sampling will be scaled down proportionally")
    return result


# ============================================================
# Corruption engine
# ============================================================

class CorruptionEngine:
    """Multi-level table data corruption engine."""

    @staticmethod
    def _clone(grid: List[List]) -> List[List]:
        return [list(row) for row in grid]

    @staticmethod
    def _find_numeric_positions(grid: List[List]) -> List[Tuple[int, int]]:
        """Find the coordinates of all cells convertible to float."""
        positions = []
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                if _try_float(str(cell)):
                    positions.append((i, j))
        return positions

    # ── Level 1: surface tampering ──

    def level1_surface_tampering(self, grid: List[List]) -> List[List]:
        """
        Randomly pick 30% of the floats and change the last decimal digit to a value
        different from the original. Ensures a change actually occurs (excludes cases
        where the last digit already equals the target value).
        """
        result = self._clone(grid)
        positions = self._find_numeric_positions(result)

        if not positions:
            return result

        n_tamper = max(1, int(len(positions) * 0.30))
        tamper_set = set(random.sample(positions, min(n_tamper, len(positions))))

        for i, j in tamper_set:
            try:
                val = _to_float(result[i][j])
                original = str(result[i][j]).strip()

                # Decide how many decimal places to keep
                if '.' in original:
                    decimals = len(original.rstrip('%').split('.')[-1])
                    if decimals == 0:
                        decimals = 2
                else:
                    decimals = 2

                # Format to target precision
                formatted = f"{val:.{decimals}f}"
                parts = formatted.split('.')

                if len(parts) == 2 and decimals > 0:
                    decimal_digits = list(parts[1])
                    last_digit = int(decimal_digits[-1])

                    # Pick a last digit different from the original
                    candidates = [d for d in range(10) if d != last_digit]
                    new_digit = random.choice(candidates)
                    decimal_digits[-1] = str(new_digit)

                    new_val = float(parts[0] + '.' + ''.join(decimal_digits))
                    result[i][j] = _float_to_str(new_val, original)
            except (ValueError, IndexError, TypeError):
                continue

        return result

    # ── Level 2: statistical contradiction ──

    def level2_statistical_contradiction(self, grid: List[List]) -> List[List]:
        """
        Find suspected P-value rows (label contains p/sig/significance and a value in 0.05~0.9).
        Change them to a random number < 0.05 without changing the same row's mean/variance,
        creating a mathematical inconsistency.
        Detection scope: labels in all columns, not just the first 3.
        """
        result = self._clone(grid)

        for i, row in enumerate(result):
            if len(row) < 2:
                continue

            # Check whether any position in this row contains a P-value label
            has_p_label = False
            for j, cell in enumerate(row):
                if RE_PVALUE_LABEL.search(str(cell)):
                    has_p_label = True
                    break

            if not has_p_label:
                continue

            # Find values in 0.05~0.9 within this row
            for j, cell in enumerate(row):
                try:
                    val = _to_float(str(cell))
                    if 0.05 <= val <= 0.9:
                        new_val = random.uniform(0.001, 0.049)
                        result[i][j] = _float_to_str(new_val, str(grid[i][j]))
                        break
                except (ValueError, TypeError):
                    continue

        return result

    # ── Level 3: advanced camouflage ──

    def _find_mean_sd_pairs(self, grid: List[List]) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Find (Mean position, SD position) pairs in the table.
        Supports multiple layouts:
          - Row layout: each row has label+value, Mean and SD on different rows
          - Column layout: header contains Mean/SD, data in rows below
          - Same row: a row contains both a Mean column and an SD column
        Returns a list of all pairs found.
        """
        mean_positions = []  # (row, col)
        sd_positions = []

        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                cell_str = str(cell).strip()
                if MEAN_KEYWORDS.match(cell_str):
                    # Find the value to the right in the same row
                    for k in range(j + 1, len(row)):
                        if _try_float(str(row[k])):
                            mean_positions.append((i, k))
                            break
                elif SD_KEYWORDS.match(cell_str):
                    for k in range(j + 1, len(row)):
                        if _try_float(str(row[k])):
                            sd_positions.append((i, k))
                            break

        # If row layout found nothing, try column layout (header in first row)
        if not mean_positions or not sd_positions:
            if len(grid) >= 2:
                header = [str(c).strip().lower() for c in grid[0]]
                for j, h in enumerate(header):
                    if MEAN_KEYWORDS.match(h) and not mean_positions:
                        for i in range(1, len(grid)):
                            if _try_float(str(grid[i][j])):
                                mean_positions.append((i, j))
                                break
                    elif SD_KEYWORDS.match(h) and not sd_positions:
                        for i in range(1, len(grid)):
                            if _try_float(str(grid[i][j])):
                                sd_positions.append((i, j))
                                break

        # Pair one-to-one; ignore extras
        pairs = []
        for m, s in zip(mean_positions, sd_positions):
            if m != s:  # Mean and SD cannot be the same cell
                pairs.append((m, s))

        return pairs

    def level3_advanced_camouflage(self, grid: List[List]) -> List[List]:
        """
        Advanced camouflage:
        1. Find Mean and SD
        2. Mean *= 1.1 (10% offset)
        3. Back-solve SD so that the t-test p ≈ 0.02 (assuming N=30, μ0 = original Mean)
        Mathematically self-consistent but the data is fabricated. Falls back to L1 if no Mean/SD found.
        """
        result = self._clone(grid)
        pairs = self._find_mean_sd_pairs(result)

        if not pairs:
            return self.level1_surface_tampering(grid)

        N = 30
        target_p = 0.02
        t_crit = stats.t.ppf(1 - target_p / 2, df=N - 1)

        modified = False
        for (mi, mj), (si, sj) in pairs:
            try:
                original_mean = _to_float(result[mi][mj])
                original_sd = _to_float(result[si][sj])

                if original_sd <= 0 or original_mean == 0:
                    continue

                new_mean = original_mean * 1.1
                mean_diff = abs(new_mean - original_mean)

                if mean_diff == 0:
                    continue

                # t = |new_mean - μ0| / (new_sd / sqrt(N))
                # new_sd = |new_mean - μ0| * sqrt(N) / t_crit
                new_sd = mean_diff * np.sqrt(N) / t_crit

                result[mi][mj] = _float_to_str(new_mean, grid[mi][mj])
                result[si][sj] = _float_to_str(new_sd, grid[si][sj])
                modified = True

            except (ValueError, IndexError, ZeroDivisionError):
                continue

        if not modified:
            return self.level1_surface_tampering(grid)

        return result

    # ── Level 6: self-contradiction (combines three sub-types) ──

    def _find_numeric_columns(self, grid: List[List]) -> Dict[int, List[Tuple[int, float]]]:
        """Collect values by column, returning {col_idx: [(row_idx, value), ...]}."""
        col_data = {}
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                try:
                    val = _to_float(str(cell))
                    if j not in col_data:
                        col_data[j] = []
                    col_data[j].append((i, val))
                except (ValueError, TypeError):
                    continue
        return col_data

    def _find_percentage_rows(self, grid: List[List]) -> List[Tuple[int, List[Tuple[int, float]]]]:
        """Find rows containing percentage values, returning [(row_idx, [(col_idx, value), ...])].
        Percentage trait: value in 0~100 and original string contains %, or values sum to ~100."""
        pct_rows = []
        for i, row in enumerate(grid):
            pct_cells = []
            for j, cell in enumerate(row):
                cell_str = str(cell).strip()
                if cell_str.endswith('%'):
                    try:
                        val = _to_float(cell_str)
                        pct_cells.append((j, val))
                    except (ValueError, TypeError):
                        continue
            if len(pct_cells) >= 2:
                pct_rows.append((i, pct_cells))
        return pct_rows

    def level6_A_mean_out_of_range(self, grid: List[List]) -> List[List]:
        """
        L6-A: mean out of data range
        Find a column with >=4 values, compute the true min/max, and change the Mean to beyond max.
        """
        result = self._clone(grid)

        # Strategy 1: find a column whose header contains "Mean" and modify one of its values
        if len(grid) >= 3:
            header = [str(c).strip() for c in grid[0]]
            mean_col = -1
            for j, h in enumerate(header):
                if MEAN_KEYWORDS.match(h):
                    mean_col = j
                    break

            if mean_col >= 0 and mean_col < len(grid[0]):
                # Collect all values in this column
                col_values = []
                for i in range(1, len(grid)):
                    if mean_col < len(grid[i]):
                        try:
                            val = _to_float(grid[i][mean_col])
                            col_values.append((i, val))
                        except (ValueError, TypeError):
                            continue

                if len(col_values) >= 1:
                    # Use the range of other numeric columns as a reference
                    all_numeric = []
                    for j in range(len(grid[0])):
                        if j == mean_col:
                            continue
                        for i in range(1, len(grid)):
                            if j < len(grid[i]):
                                try:
                                    all_numeric.append(_to_float(grid[i][j]))
                                except (ValueError, TypeError):
                                    continue

                    if all_numeric:
                        ref_max = max(all_numeric)
                    else:
                        ref_max = max(v for _, v in col_values)

                    # Randomly pick a row's Mean value and change it out of range
                    target_row, original_val = random.choice(col_values)
                    overshoot = abs(ref_max) * random.uniform(0.10, 0.20)
                    new_val = ref_max + overshoot
                    result[target_row][mean_col] = _float_to_str(new_val, grid[target_row][mean_col])
                    return result

        # Strategy 2: fallback -- find a column with >=4 values and change one of them
        col_data = self._find_numeric_columns(result)
        candidates = []
        for col_idx, values in col_data.items():
            if len(values) >= 4:
                vals = [v for _, v in values]
                candidates.append((col_idx, values, min(vals), max(vals)))

        if not candidates:
            return self.level1_surface_tampering(grid)

        col_idx, values, col_min, col_max = random.choice(candidates)
        target_row, original_val = random.choice(values)

        overshoot = abs(col_max) * random.uniform(0.10, 0.20)
        new_val = col_max + overshoot
        result[target_row][col_idx] = _float_to_str(new_val, grid[target_row][col_idx])

        return result

    def level6_B_percentage_sum(self, grid: List[List]) -> List[List]:
        """
        L6-B: percentages do not sum
        Find a row with >=2 percentages and change one value so the sum != 100%.
        """
        result = self._clone(grid)
        pct_rows = self._find_percentage_rows(result)

        if not pct_rows:
            return self.level1_surface_tampering(grid)

        # Randomly pick a row
        row_idx, pct_cells = random.choice(pct_rows)

        # Compute the current sum
        current_sum = sum(v for _, v in pct_cells)

        # Randomly pick a cell to change
        target_col, original_val = random.choice(pct_cells)

        # Compute how much to change so the sum deviates from 100%
        # Strategy: make the sum 90~95% or 105~110%
        if current_sum > 0:
            desired_sum = random.choice([
                random.uniform(88, 95),   # under
                random.uniform(105, 112),  # over
            ])
            # Apply the difference to the changed cell
            new_val = original_val + (desired_sum - current_sum)
            # Ensure the changed value is still reasonable (>0)
            if new_val > 0:
                result[row_idx][target_col] = _float_to_str(new_val, grid[row_idx][target_col])
            else:
                # If the result is negative, just add a random offset
                offset = random.choice([random.uniform(5, 15), random.uniform(-15, -5)])
                new_val = original_val + offset
                if new_val > 0:
                    result[row_idx][target_col] = _float_to_str(new_val, grid[row_idx][target_col])

        return result

    def level6_C_impossible_distribution(self, grid: List[List]) -> List[List]:
        """
        L6-C: impossible distribution
        Find cells with Mean+SD where Mean-2SD > 0, and increase SD so Mean-2SD < 0.
        This implies the data contains negative values, which the scenario disallows
        (e.g. heights, percentages).
        """
        result = self._clone(grid)
        pairs = self._find_mean_sd_pairs(result)

        if not pairs:
            return self.level1_surface_tampering(grid)

        # Find pairs where Mean - 2*SD > 0
        valid_pairs = []
        for (mi, mj), (si, sj) in pairs:
            try:
                mean_val = _to_float(result[mi][mj])
                sd_val = _to_float(result[si][sj])
                if sd_val > 0 and mean_val - 2 * sd_val > 0:
                    valid_pairs.append(((mi, mj), (si, sj), mean_val, sd_val))
            except (ValueError, TypeError):
                continue

        if not valid_pairs:
            return self.level1_surface_tampering(grid)

        # Randomly pick a pair
        (mi, mj), (si, sj), mean_val, sd_val = random.choice(valid_pairs)

        # Compute how large SD must be so that Mean - 2*SD < 0
        # Requires SD > Mean / 2
        min_sd = mean_val / 2
        # Add some margin so Mean - 2*SD lands in the -5%~-20% range
        target_sd = min_sd * random.uniform(1.05, 1.25)

        result[si][sj] = _float_to_str(target_sd, grid[si][sj])

        return result

    def level6_self_contradiction(self, grid: List[List]) -> List[List]:
        """
        Level 6 entry point: randomly choose one of L6-A / L6-B / L6-C to execute.
        """
        sub_type = random.choice(["A", "B", "C"])
        if sub_type == "A":
            return self.level6_A_mean_out_of_range(grid)
        elif sub_type == "B":
            return self.level6_B_percentage_sum(grid)
        else:
            return self.level6_C_impossible_distribution(grid)


# ============================================================
# Grid post-validation
# ============================================================

def validate_corrupted(original: List[List], corrupted: List[List]) -> bool:
    """Validate that the corrupted table is still valid."""
    # Structure must match
    if len(corrupted) != len(original):
        return False
    if any(len(c) != len(o) for c, o in zip(corrupted, original)):
        return False
    # Cannot be entirely empty
    flat = [str(c).strip() for row in corrupted for c in row if str(c).strip()]
    if not flat:
        return False
    # Must not introduce NaN/Inf literals
    for row in corrupted:
        for cell in row:
            s = str(cell).lower().strip()
            if s in ('nan', 'inf', '-inf', 'infinity', '-infinity'):
                return False
    return True


# ============================================================
# Dataset splitting and persistence (split at paper level to prevent data leakage)
# ============================================================

def build_bench_dataset(conn: sqlite3.Connection, valid_df: pd.DataFrame, cross_domain: bool = False):
    """Build the SciNumBench dataset and write it to the scinum_bench table.

    Key: split by arxiv_id so all tables of one paper appear in only one split.
    When cross_domain=True: Train/Val use CS papers, Test uses non-CS papers.
    """

    total_available = len(valid_df)
    actual_total = min(TOTAL_BENCH, total_available)

    # Deduplicate at the paper level
    unique_papers = valid_df["arxiv_id"].unique()
    n_papers = len(unique_papers)

    if actual_total < TOTAL_BENCH:
        scale = actual_total / TOTAL_BENCH
        for split in CORRUPTION_DIST:
            for key in CORRUPTION_DIST[split]:
                CORRUPTION_DIST[split][key] = max(1, int(CORRUPTION_DIST[split][key] * scale))
        print(f"[WARN] Scaled down proportionally to about {actual_total} samples")

    rng = random.Random(42)

    # ── Paper-level split ──
    if cross_domain and "primary_category" in valid_df.columns:
        # Cross-domain mode: CS -> Train/Val, non-CS -> Test
        paper_cat = valid_df.groupby("arxiv_id")["primary_category"].first()
        cs_papers = [p for p, cat in paper_cat.items() if str(cat).startswith("cs.")]
        non_cs_papers = [p for p, cat in paper_cat.items() if not str(cat).startswith("cs.")]

        rng.shuffle(cs_papers)
        rng.shuffle(non_cs_papers)

        train_need = sum(CORRUPTION_DIST["train"].values())
        val_need = sum(CORRUPTION_DIST["val"].values())
        test_need = sum(CORRUPTION_DIST["test"].values())

        # Assign CS papers to Train/Val
        train_n = max(1, int(len(cs_papers) * train_need / (train_need + val_need)))
        val_n = len(cs_papers) - train_n

        train_papers = set(cs_papers[:train_n])
        val_papers = set(cs_papers[train_n:train_n + val_n])
        test_papers = set(non_cs_papers)

        # Subject distribution statistics
        test_cats = paper_cat[paper_cat.index.isin(test_papers)].value_counts().head(5)
        print(f"[SPLIT] Test subject distribution: {dict(test_cats)}")
    else:
        # Random mode
        shuffled_papers = list(unique_papers)
        rng.shuffle(shuffled_papers)

        train_need = sum(CORRUPTION_DIST["train"].values())
        val_need = sum(CORRUPTION_DIST["val"].values())
        test_need = sum(CORRUPTION_DIST["test"].values())
        total_need = train_need + val_need + test_need

        train_n = max(1, int(n_papers * train_need / total_need))
        val_n = max(1, int(n_papers * val_need / total_need))
        test_n = n_papers - train_n - val_n

        train_papers = set(shuffled_papers[:train_n])
        val_papers = set(shuffled_papers[train_n:train_n + val_n])
        test_papers = set(shuffled_papers[train_n + val_n:])

    # Group by paper
    paper_tables = valid_df.groupby("arxiv_id")

    engine = CorruptionEngine()
    batch_buffer = []

    def flush_batch():
        nonlocal batch_buffer
        if batch_buffer:
            conn.executemany("""
                INSERT OR IGNORE INTO scinum_bench
                (arxiv_id, table_id, dataset_split, original_grid, corrupted_grid,
                 corruption_level, is_human_modified)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, batch_buffer)
            batch_buffer = []

    for split, paper_set, dist in [
        ("train", train_papers, CORRUPTION_DIST["train"]),
        ("val", val_papers, CORRUPTION_DIST["val"]),
        ("test", test_papers, CORRUPTION_DIST["test"]),
    ]:
        # Collect all available tables for this split
        split_tables = []
        for pid in paper_set:
            if pid in paper_tables.groups:
                group = paper_tables.get_group(pid)
                for _, row in group.iterrows():
                    split_tables.append(row)

        rng.shuffle(split_tables)

        # Allocate corruption levels
        clean_n = dist["clean"]
        l1_n = dist["L1"]
        l2_n = dist["L2"]
        l3_n = dist["L3"]
        l6_n = dist.get("L6", 0)
        total_split = clean_n + l1_n + l2_n + l3_n + l6_n

        if len(split_tables) < total_split:
            ratio = len(split_tables) / total_split
            clean_n = max(1, int(clean_n * ratio))
            l1_n = max(1, int(l1_n * ratio))
            l2_n = max(1, int(l2_n * ratio))
            l3_n = max(1, int(l3_n * ratio))
            l6_n = max(1, int(l6_n * ratio))
            total_split = clean_n + l1_n + l2_n + l3_n + l6_n
            print(f"[WARN] Not enough {split} tables; scaled down to {total_split}")

        cursor = 0

        # Clean
        for row in tqdm(split_tables[cursor:cursor + clean_n],
                        desc=f"{split}/Clean", leave=False):
            try:
                grid = _parse_grid_from_json(row["table_json"])
                if grid is None:
                    continue
                batch_buffer.append((
                    row["arxiv_id"], row["table_id"], split,
                    json.dumps(grid), json.dumps(grid), "0_Clean", 0
                ))
            except Exception as e:
                tqdm.write(f"[ERR] Clean: {e}")
        cursor += clean_n
        flush_batch()

        # Fake levels
        level_map = [
            ("L1", l1_n, "1_Surface", engine.level1_surface_tampering),
            ("L2", l2_n, "2_Contradiction", engine.level2_statistical_contradiction),
            ("L3", l3_n, "3_Camouflage", engine.level3_advanced_camouflage),
            ("L6", l6_n, "6_SelfContradiction", engine.level6_self_contradiction),
        ]

        for level_key, count, level_name, corrupt_fn in level_map:
            if count <= 0:
                continue
            for row in tqdm(split_tables[cursor:cursor + count],
                            desc=f"{split}/{level_key}", leave=False):
                try:
                    grid = _parse_grid_from_json(row["table_json"])
                    if grid is None:
                        continue
                    corrupted = corrupt_fn(grid)
                    if not validate_corrupted(grid, corrupted):
                        corrupted = grid  # fall back to clean
                    batch_buffer.append((
                        row["arxiv_id"], row["table_id"], split,
                        json.dumps(grid), json.dumps(corrupted), level_name, 1
                    ))
                except Exception as e:
                    tqdm.write(f"[ERR] {level_key}: {e}")
            cursor += count
            flush_batch()

        conn.commit()

    # Final statistics
    _print_stats(conn)


def _print_stats(conn: sqlite3.Connection):
    """Print dataset statistics."""
    rows = conn.execute("""
        SELECT dataset_split, corruption_level, COUNT(*)
        FROM scinum_bench
        GROUP BY dataset_split, corruption_level
        ORDER BY dataset_split, corruption_level
    """).fetchall()

    print(f"{'Split':<8} {'Level':<20} {'Count':>6}")
    print("-" * 40)

    current_split = ""
    split_total = 0
    for split, level, count in rows:
        if split != current_split:
            if current_split and split_total:
                print(f"  {'':8} {'Subtotal':<20} {split_total:>6}")
                print()
            current_split = split
            split_total = 0
        print(f"  {split:<8} {level:<20} {count:>6}")
        split_total += count

    if split_total:
        print(f"  {'':8} {'Subtotal':<20} {split_total:>6}")

    total = conn.execute("SELECT COUNT(*) FROM scinum_bench").fetchone()[0]
    papers = conn.execute("SELECT COUNT(DISTINCT arxiv_id) FROM scinum_bench").fetchone()[0]
    print("-" * 40)
    print(f"  {'TOTAL':<8} {'':20} {total:>6}")
    print(f"  Papers: {papers}")


# ============================================================
# Main entry point
# ============================================================

def main():
    conn = sqlite3.connect(DB_PATH)

    create_bench_table(conn)

    # Check whether data already exists
    existing = conn.execute("SELECT COUNT(*) FROM scinum_bench").fetchone()[0]
    if existing > 0:
        print(f"[WARN] scinum_bench already contains {existing} records")
        ans = input("Clear and rebuild? [y/N] ").strip().lower()
        if ans == "y":
            conn.execute("DELETE FROM scinum_bench")
            conn.commit()
        else:
            conn.close()
            return

    # Load and filter
    valid_df = load_and_filter_tables(conn)
    if len(valid_df) == 0:
        print("[ERR] No valid numeric tables, exiting")
        conn.close()
        return

    # Choose split mode
    actual = min(TOTAL_BENCH, len(valid_df))
    print(f"\nSplit modes:")
    print(f"  1. Random split (Train/Val/Test assigned randomly)")
    print(f"  2. Cross-domain generalization (Train/Val=CS, Test=non-CS)")

    mode = input("Choose mode [1/2] (default 1): ").strip()
    cross_domain = (mode == "2")

    ans = input("\nConfirm to start building? [y/N] ").strip().lower()
    if ans != "y":
        print("Cancelled.")
        conn.close()
        return

    build_bench_dataset(conn, valid_df, cross_domain=cross_domain)

    conn.close()


if __name__ == "__main__":
    main()
