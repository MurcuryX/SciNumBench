"""
clean_serialize.py — shared cleaning + serialization layer for the SciNumBench LLM-judge

Provides:
  clean_grid(grid)        -> grid          cell-by-cell removal of LaTeX residue
                                            (inline comments / \\\\ / \\cmd), conservative;
                                            never drops valid percentages (74% / 52.1%) or breaks numerics.
  serialize(grid, caption, provenance=None, max_tokens=8000)
                          -> str            human-readable pipe (markdown-style) table + one-line caption;
                                            truncates by chars/4 token estimate, keeping the header and the
                                            fake table's provenance rows.

Design basis (from measured data): grid is a list-of-lists, row 0 = header; ~2-4% of tables have LaTeX
residue (leaked `% ...` comments, `\\\\` row joins, overlong cells, sparse `\\cmd`); only 2 tables in the
whole DB serialize to >~8k tokens.

Conventions align with the table_text in detectors.py / cot_detect.py: each row joined by " | "; upgraded
here to a markdown table with a header separator, but cell text is cleaned. __main__ self-tests bench_id 81
and several PCT tables.
"""
import re

# Inline LaTeX comment: from `%` to end of cell.
# Only treated as a comment when % is immediately followed by space / letter (a real comment `% Ensured ...`);
# pure numeric percentages `74%` / `52.1%` / `(%)` are always kept.
# Match condition: % followed by whitespace or letter/backslash/`#` (typical comment start), and the % is not
# a percent sign right after a digit.
_INLINE_COMMENT = re.compile(r"(?<!\d)%(?=\s|[A-Za-z\\#])[^\n]*$")
# Fallback: even when % is preceded by a digit, if followed by " Ensured"/a word and the cell has more than one
# token it may still be a comment. To be safe, also handle "digit% <space><letters...>" like "0.0114 \\ % Ensured"
# where the prefix has already been split off by \\.
_COMMENT_AFTER_SPACE = re.compile(r"\s+%\s+[A-Za-z][^\n]*$")

# LaTeX row join `\\` (table line break)
_ROW_BREAK = re.compile(r"\\\\")
# LaTeX control sequence `\cmd` / `\cmd{...}` (keep text inside braces)
_LATEX_CMD = re.compile(r"\\[A-Za-z@]+\s*")
# Residual stray backslashes
_STRAY_BS = re.compile(r"\\+")
# Excess whitespace
_WS = re.compile(r"[ \t]{2,}")

MAX_CELL_LEN = 80  # Overlong cells (known noise): if still long after whitespace normalization, single cells
                   # are not extra-truncated during serialization; the table-level token budget is the backstop.
                   # Used here only for the whitespace-normalization decision.


def clean_cell(s):
    """Clean a single cell string: strip inline comments, \\\\, \\cmd, stray backslashes; keep numerics and valid percentages."""
    if s is None:
        return ""
    t = str(s)

    # 1) First turn `\\` (row join) into a gentle separator to avoid gluing multi-line numbers together.
    #    Use " ; " as a soft separator; later whitespace normalization converges it; no duplication if spaces already present.
    t = _ROW_BREAK.sub(" ; ", t)

    # 2) Strip inline comments (before stripping \cmd: `\\` is now ` ; ` and the comment `%` is still in place).
    #    First handle "digit % letters..." (common after \\ split: "0.0114 ; % Ensured ...")
    t = _COMMENT_AFTER_SPACE.sub("", t)
    #    Then handle general inline comments (% immediately followed by whitespace/letter/backslash/#)
    t = _INLINE_COMMENT.sub("", t)

    # 3) Strip LaTeX control sequences \cmd (comments already removed, so the \ inside comments is not mis-eaten);
    #    the cmd of \cmd{text} is removed, then step 4 removes the bare braces and keeps text.
    t = _LATEX_CMD.sub(" ", t)

    # 4) Strip LaTeX grouping braces (keep inner text) and residual stray backslashes
    t = t.replace("{", " ").replace("}", " ")
    t = _STRAY_BS.sub(" ", t)

    # 5) Finish: drop soft separator " ; " at the ends, normalize whitespace
    t = _WS.sub(" ", t).strip()
    t = re.sub(r"^[;\s]+|[;\s]+$", "", t)        # strip leading/trailing semicolons/whitespace
    t = re.sub(r"\s*;\s*;+\s*", " ; ", t)        # merge consecutive semicolons
    return t.strip()


def clean_grid(grid):
    """Clean an entire list-of-lists table cell by cell; structure (rows/columns) is unchanged."""
    out = []
    for row in grid:
        out.append([clean_cell(c) for c in row])
    return out


def _row_to_md(row):
    return "| " + " | ".join(c if c != "" else " " for c in row) + " |"


def serialize(grid, caption, provenance=None, max_tokens=8000):
    """Clean + serialize into a human-readable pipe table + one-line caption.

    grid: list-of-lists, row 0 = header (caller may pass pre- or post-clean; this function cleans internally).
    caption: string.
    provenance: JSON-decoded list for the fake table (each item has r,c); used during overlong truncation to
                ensure modified rows are not cut. provenance.r already uses grid-row numbering (0 = header),
                consistent with detectors flags (flag r = i+1) and evaluate.py truth=(p["r"],p["c"]).
    max_tokens: chars/4 estimated upper bound; if exceeded, truncate keeping header + provenance rows + first N rows.
    """
    g = clean_grid(grid)
    if not g:
        return (caption or "").strip()
    header = g[0]
    body = g[1:]

    # Set of (grid) row numbers where provenance lives; must be kept
    keep_rows = set()
    if provenance:
        for p in provenance:
            try:
                keep_rows.add(int(p["r"]))   # grid row numbering (0 = header)
            except Exception:
                pass

    def render(rows_body):
        lines = [_row_to_md(header),
                 "| " + " | ".join("---" for _ in header) + " |"]
        lines.extend(_row_to_md(r) for r in rows_body)
        cap = (caption or "").strip()
        head = f"Caption: {cap}" if cap else "Caption: (none)"
        return head + "\n\n" + "\n".join(lines)

    text = render(body)
    if len(text) <= max_tokens * 4:
        return text

    # Overlong: shrink body step by step. Guarantee (a) the header (b) provenance rows (grid row in keep_rows).
    # provenance row r maps to body index = r-1 (grid row - 1, since body=g[1:])
    keep_body_idx = sorted({r - 1 for r in keep_rows if 1 <= r - 1 < len(body) + 0 or r - 1 >= 0})
    keep_body_idx = [i for i in keep_body_idx if 0 <= i < len(body)]

    # Find the largest keepable prefix N (linear) so render(first N rows ∪ must-keep rows) fits the budget
    N = len(body)
    while N > 0:
        sel_idx = sorted(set(range(N)) | set(keep_body_idx))
        sel = [body[i] for i in sel_idx]
        dropped = len(body) - len(sel)
        trial = render(sel)
        if dropped > 0:
            trial = trial + f"\n[...truncated {dropped} rows...]"
        if len(trial) <= max_tokens * 4:
            return trial
        N -= max(1, len(body) // 50)  # step shrink
    # Extreme fallback: header only + must-keep rows
    sel_idx = sorted(set(keep_body_idx))
    sel = [body[i] for i in sel_idx]
    dropped = len(body) - len(sel)
    trial = render(sel)
    if dropped > 0:
        trial = trial + f"\n[...truncated {dropped} rows...]"
    return trial


# Self-test
if __name__ == "__main__":
    import sqlite3, json, sys
    sys.path.insert(0, "/home/pengminjie/Backup/paper/ICDE26/code/bench")
    DB = "/home/pengminjie/Backup/paper/ICDE26/data/arxiv_data.db"

    cases = [
        ("74%", "74%"),
        ("52.1%", "52.1%"),
        ("(%)", "(%)"),
        ("ASR (%)", "ASR (%)"),
        ("23 / 100 (47.7%)", "23 / 100 (47.7%)"),
        ("0.0114 ± 0.000720 \\\\ % Ensured 6 decimal places", "0.0114 ± 0.000720"),
        ("12.3 % Ensured consistency", "12.3"),
        ("% full line comment", ""),
        ("value \\textbf{bold}", "value bold") ,
        ("a \\\\ b \\\\ c", "a ; b ; c"),
        ("3.14 \\pm 0.5", "3.14 0.5"),
    ]
    ok = True
    for inp, exp in cases:
        got = clean_cell(inp)
        flag = "OK " if got == exp else "FAIL"
        if got != exp:
            ok = False
        print(f"  [{flag}] {inp!r:50s} -> {got!r:30s} (exp {exp!r})")
    print("  ALL PASS" if ok else "  !!! SOME FAILED")

    conn = sqlite3.connect(DB)

    print("bench_id 81 (known % Ensured residue + \\\\ residue) before/after cleaning")
    r = conn.execute("""SELECT sb.corrupted_grid, sb.provenance, pt.caption
                        FROM scinum_bench sb LEFT JOIN paper_tables pt ON sb.src_table_id=pt.id
                        WHERE sb.bench_id=81""").fetchone()
    g = json.loads(r[0]); prov = json.loads(r[1]) if r[1] else None
    cg = clean_grid(g)
    for ri, (orow, crow) in enumerate(zip(g, cg)):
        for ci, (o, c) in enumerate(zip(orow, crow)):
            if str(o) != str(c):
                print(f"  cell[{ri}][{ci}]:\n     BEFORE: {o!r}\n     AFTER : {c!r}")
    print("\n  --- serialized (clean) ---")
    print(serialize(g, r[2], provenance=prov))

    print("PCT-table percentage not-mis-dropped spot check (bench 78/88/89)")
    for bid in (78, 88, 89):
        row = conn.execute("""SELECT corrupted_grid FROM scinum_bench WHERE bench_id=?""", (bid,)).fetchone()
        if not row:
            continue
        g = json.loads(row[0]); cg = clean_grid(g)
        pcts_before = [c for row_ in g for c in row_ if "%" in str(c)]
        pcts_after = [c for row_ in cg for c in row_ if "%" in str(c)]
        print(f"  bench {bid}: %cells before={len(pcts_before)} after={len(pcts_after)} "
              f"{'(kept OK)' if len(pcts_before)==len(pcts_after) else '(!!changed)'}")
        for b_, a_ in list(zip(pcts_before, pcts_after))[:4]:
            print(f"     {b_!r} -> {a_!r}")
    conn.close()
