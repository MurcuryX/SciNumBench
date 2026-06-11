"""
parser.py — SciNumBench structured table parsing (refactor v2 foundation)

table_json (pandas orient="split": {columns, index, data}) -> StructuredTable
Keeps the header; classifies each cell by type; extracts per-group sample size n from header/caption.

Cell types:
  meansd  "21±2" / "1.19 ± 0.23"        -> {mean, sd}
  ci      "0.61 (0.43-0.84)"            -> {point, lo, hi}   (includes median(IQR))
  pval    "<0.01" / "0.037*" in p-value columns -> {op, value, star}
  pct     "74%"                          -> {value}
  count   "13"   (pure integer)         -> {value}
  num     "0.92" (with decimal)         -> {value}
  text / empty
"""

import json
import re
from typing import Any, Dict, List, Optional

# Type-detection regexes (applied to the raw cell string)
_RE_MEANSD = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*(?:±|\\pm|\+/-)\s*(\d+(?:\.\d+)?)\s*\*?\s*$")
_RE_CI = re.compile(
    r"^\s*(-?\d+(?:\.\d+)?)\s*[\(\[]\s*(-?\d+(?:\.\d+)?)\s*[-–—~,;]\s*(-?\d+(?:\.\d+)?)\s*[\)\]]\s*$"
)
_RE_PVAL = re.compile(r"^\s*([<>]=?|=)?\s*(\d*\.\d+|\d+)\s*(\*+)?\s*$")
_RE_PCT = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*%\s*$")
# "28 (52%)" / "28 (52)" / "28/54 (52%)"  → count [/denom] (pct%)
_RE_COUNTPCT = re.compile(
    r"^\s*(\d+)\s*(?:/\s*(\d+))?\s*[\(\[]\s*(\d+(?:\.\d+)?)\s*%?\s*[\)\]]\s*$"
)
_RE_INT = re.compile(r"^\s*-?\d+\s*$")
_RE_NUM = re.compile(r"^\s*-?\d+\.\d+\s*$")

# Extract n from header
_RE_N = re.compile(r"\b[nN]\s*=\s*(\d+)")
# Detect p-value column headers
_RE_PVAL_HEADER = re.compile(r"(?i)(p[\s._-]?(value|val|sig|level)?\b|significance|p\s*[<>=])")
# Zero-width characters / non-breaking spaces, for cleaning
_ZW = dict.fromkeys(map(ord, "​‌‍﻿  "), None)


def _clean(s: Any) -> str:
    if s is None:
        return ""
    return str(s).translate(_ZW).strip()


def classify_cell(raw: Any, in_pval_col: bool = False) -> Dict[str, Any]:
    """Return {type, ...parsed}. in_pval_col=the cell's column header is a p-value column."""
    s = _clean(raw)
    if s == "":
        return {"type": "empty", "raw": ""}

    m = _RE_MEANSD.match(s)
    if m:
        return {"type": "meansd", "raw": s,
                "mean": float(m.group(1)), "sd": float(m.group(2))}

    m = _RE_CI.match(s)
    if m:
        return {"type": "ci", "raw": s,
                "point": float(m.group(1)), "lo": float(m.group(2)), "hi": float(m.group(3))}

    if in_pval_col:
        m = _RE_PVAL.match(s)
        if m:
            return {"type": "pval", "raw": s,
                    "op": m.group(1) or "=", "value": float(m.group(2)),
                    "star": bool(m.group(3))}

    m = _RE_PCT.match(s)
    if m:
        return {"type": "pct", "raw": s, "value": float(m.group(1))}

    m = _RE_COUNTPCT.match(s)
    if m:
        pct = float(m.group(3))
        if 0.0 <= pct <= 100.0:  # Filter out years / non-percentages like "5 (2008)"
            return {"type": "countpct", "raw": s,
                    "count": int(m.group(1)),
                    "denom": int(m.group(2)) if m.group(2) else None,
                    "pct": pct, "pct_tok": m.group(3)}

    if _RE_INT.match(s):
        return {"type": "count", "raw": s, "value": int(s)}

    if _RE_NUM.match(s):
        return {"type": "num", "raw": s, "value": float(s)}

    return {"type": "text", "raw": s}


class StructuredTable:
    """Typed table including its header."""

    def __init__(self, columns: List[str], data: List[List[str]],
                 caption: str = "", src_table_id: Optional[int] = None,
                 arxiv_id: str = "", forensic_tags: str = ""):
        self.columns = [_clean(c) for c in columns]
        self.data = [[_clean(c) for c in row] for row in data]
        self.caption = caption or ""
        self.src_table_id = src_table_id
        self.arxiv_id = arxiv_id
        self.tags = set(t for t in (forensic_tags or "").split(",") if t)
        self.ncols = len(self.columns)
        self.nrows = len(self.data)

        # Which columns are p-value columns (matched on header)
        self.pval_cols = {j for j, h in enumerate(self.columns) if _RE_PVAL_HEADER.search(h)}
        # Per-column n (extracted from header); fall back to caption-level n if absent
        self.col_n: Dict[int, int] = {}
        for j, h in enumerate(self.columns):
            mm = _RE_N.search(h)
            if mm:
                self.col_n[j] = int(mm.group(1))
        cm = _RE_N.search(self.caption)
        self.table_n: Optional[int] = int(cm.group(1)) if cm else None

        # Type each cell (align data with columns; data rows may be short/long, truncate/pad by column)
        self.cells: List[List[Dict[str, Any]]] = []
        for row in self.data:
            crow = []
            for j in range(self.ncols):
                raw = row[j] if j < len(row) else ""
                crow.append(classify_cell(raw, in_pval_col=(j in self.pval_cols)))
            self.cells.append(crow)

    def n_for(self, col: int) -> Optional[int]:
        """Sample size for a column: column n takes priority, else table-level n."""
        return self.col_n.get(col, self.table_n)

    def positions_of(self, *types) -> List[tuple]:
        """(row, col) coordinates of all cells of the specified types."""
        want = set(types)
        return [(i, j) for i in range(self.nrows) for j in range(self.ncols)
                if self.cells[i][j]["type"] in want]

    def type_counts(self) -> Dict[str, int]:
        c: Dict[str, int] = {}
        for row in self.cells:
            for cell in row:
                c[cell["type"]] = c.get(cell["type"], 0) + 1
        return c

    def to_grid(self) -> List[List[str]]:
        """Rebuild the 2D grid including header (row 0 = header), for storing original_grid."""
        return [list(self.columns)] + [list(r) for r in self.data]

    @classmethod
    def from_json(cls, table_json: str, **kw) -> Optional["StructuredTable"]:
        try:
            d = json.loads(table_json)
        except Exception:
            return None
        if not isinstance(d, dict):
            return None
        cols = d.get("columns")
        data = d.get("data")
        if cols is None or data is None or not isinstance(data, list):
            return None
        return cls(cols, data, **kw)


def grid_from_structured(columns: List[str], data: List[List[str]]) -> List[List[str]]:
    return [list(columns)] + [list(r) for r in data]
