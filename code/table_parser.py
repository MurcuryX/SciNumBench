"""
table_parser.py — LaTeX table parser (with location info)

Extracts tables from a .tar.gz source archive, recording:
  - table_index: index of the table within the paper
  - caption: table caption
  - label: LaTeX label (e.g. tab:xxx)
  - section: section the table belongs to
  - rows/cols: number of rows and columns
  - table_json: table data (DataFrame.to_json)
"""

import re
import json
import tarfile
import pandas as pd
from typing import List, Dict

BS = chr(92)  # backslash

# Precompiled regexes
RE_TABLE_ENV = re.compile(
    BS + BS + r'begin\{table\*?\}([\s\S]*?)' + BS + BS + r'end\{table\*?\}'
)
RE_TABULAR = re.compile(
    BS + BS + r'begin\{tabular\*?\}(\{[^}]*\})?([\s\S]*?)' + BS + BS + r'end\{tabular\*?\}'
)
RE_CAPTION = re.compile(BS + BS + r'caption\{([^}]*)\}')
RE_LABEL = re.compile(BS + BS + r'label\{([^}]*)\}')
RE_SECTION = re.compile(BS + BS + r'(?:sub)*section\{([^}]*)\}')
RE_RULE = re.compile(BS + BS + r'(toprule|midrule|bottomrule|hline|cline\{[^}]*\})')
RE_COL_SPEC = re.compile(r'^[@lrcpmb\{\}\|!\s\d\\]+$')
RE_PAGEBREAK = re.compile(BS + BS + r'(?:newpage|clearpage|pagebreak)')


def clean_cell(text: str) -> str:
    """Strip LaTeX commands, leave plain text."""
    text = text.strip()
    text = re.sub(BS + BS + r'textbf\{([^}]*)\}', r'\1', text)
    text = re.sub(BS + BS + r'textit\{([^}]*)\}', r'\1', text)
    text = re.sub(BS + BS + r'emph\{([^}]*)\}', r'\1', text)
    text = re.sub(BS + BS + r'multicolumn\{\d+\}\{[^}]*\}\{([^}]*)\}', r'\1', text)
    text = re.sub(BS + BS + r'multirow\{\d+\}\{[^}]*\}\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\$([^$]*)\$', r'\1', text)
    text = text.replace(BS + '%', '%').replace(BS + '#', '#')
    text = text.replace(BS + '&', '&').replace(BS + '_', '_')
    text = text.replace(BS + '~', ' ').replace(BS + ' ', ' ')
    text = re.sub(BS + BS + r'[a-zA-Z]+\*?(?:\{[^}]*\})*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.replace('{', '').replace('}', '')
    return text


def parse_tabular(block: str) -> pd.DataFrame | None:
    """Parse tabular content into DataFrame."""
    # Remove column spec
    stripped = block.lstrip()
    if stripped.startswith('{'):
        depth, end = 0, 0
        for i, ch in enumerate(stripped):
            if ch == '{': depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0: end = i + 1; break
        if end > 0 and RE_COL_SPEC.match(stripped[1:end-1]):
            block = stripped[end:]

    # Remove rules
    block = RE_RULE.sub('', block)

    rows = []
    current_row = []
    for line in block.split('\n'):
        line = line.strip()
        if not line or line.startswith('%'):
            continue
        # Split by & respecting braces
        parts, depth, buf = [], 0, []
        for ch in line:
            if ch == '{': depth += 1; buf.append(ch)
            elif ch == '}': depth -= 1; buf.append(ch)
            elif ch == '&' and depth == 0: parts.append(''.join(buf)); buf = []
            else: buf.append(ch)
        parts.append(''.join(buf))

        for part in parts:
            part = part.strip()
            if part.endswith('\\\\') or part.endswith(BS + BS):
                cell = part.rstrip(BS).rstrip()
                if cell: current_row.append(clean_cell(cell))
                if current_row: rows.append(current_row)
                current_row = []
            elif part:
                current_row.append(clean_cell(part))

        stripped = line.rstrip()
        if stripped.endswith(BS + BS) and not stripped.endswith(BS + BS + BS + BS):
            if current_row: rows.append(current_row)
            current_row = []

    if current_row:
        rows.append(current_row)

    if len(rows) < 2:
        return None

    # Normalize
    max_cols = max(len(r) for r in rows)
    for r in rows:
        while len(r) < max_cols: r.append("")

    header = rows[0]
    data = rows[1:]
    try:
        df = pd.DataFrame(data, columns=header)
    except Exception:
        df = pd.DataFrame(data)

    df.columns = [str(c).strip() or f"col_{i}" for i, c in enumerate(df.columns)]
    df = df[~(df == '').all(axis=1)]

    if df.empty or df.shape[0] < 1 or df.shape[1] < 2:
        return None

    return df


def extract_tables_from_tex(content: str) -> List[Dict]:
    """Extract tables with location info from .tex content.
    Returns list of dicts: {table_index, caption, label, section, page, rows, cols, table_json}
    """
    results = []
    idx = 0

    # Precompute pagebreak positions for page estimation
    pagebreak_positions = [m.start() for m in RE_PAGEBREAK.finditer(content)]

    # Track current section
    current_section = ""
    for m in RE_SECTION.finditer(content):
        current_section = m.group(1).strip()

    # Find all \begin{table}...\end{table}
    for table_match in RE_TABLE_ENV.finditer(content):
        block = table_match.group(1)
        table_pos = table_match.start()

        # Estimate page number: count how many pagebreaks occurred before this table
        page = 1
        for pb_pos in pagebreak_positions:
            if pb_pos < table_pos:
                page += 1
            else:
                break

        caption = ""
        cap_m = RE_CAPTION.search(block)
        if cap_m:
            caption = clean_cell(cap_m.group(1))

        label = ""
        lab_m = RE_LABEL.search(block)
        if lab_m:
            label = lab_m.group(1).strip()

        # Find tabular inside this table block
        for _, tab_content in RE_TABULAR.findall(block):
            df = parse_tabular(tab_content)
            if df is not None and not df.empty:
                results.append({
                    "table_index": idx,
                    "caption": caption,
                    "label": label,
                    "section": current_section,
                    "page": page,
                    "rows": len(df),
                    "cols": len(df.columns),
                    "table_json": df.to_json(orient="split"),
                })
                idx += 1

    return results


def extract_from_tarball(tar_path: str) -> List[Dict]:
    """Extract tables from a .tar.gz LaTeX source archive."""
    results = []
    try:
        with tarfile.open(tar_path, 'r:gz') as tar:
            for member in tar.getmembers():
                if member.name.endswith('.tex'):
                    f = tar.extractfile(member)
                    if f:
                        content = f.read().decode('utf-8', errors='ignore')
                        results.extend(extract_tables_from_tex(content))
    except Exception:
        pass
    return results
