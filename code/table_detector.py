"""
table_detector.py — lightweight LaTeX table probe

Counts only, no parsing. Much faster than table_parser.py.
Takes a .tar.gz path -> returns the number of tables (int).
"""

import re
import tarfile
from typing import List

BS = chr(92)  # backslash

# Matches \begin{table} or \begin{table*}
RE_TABLE_BEGIN = re.compile(
    BS + BS + r'begin\{table\*?\}',
    re.IGNORECASE,
)


def _count_tables_in_tex(content: str) -> int:
    """Count the number of table environments in a single .tex file."""
    return len(RE_TABLE_BEGIN.findall(content))


def detect_table_count(tar_path: str) -> int:
    """Probe entry point: read all .tex files in the .tar.gz, return total table count.

    - corrupt archive -> 0
    - no .tex files -> 0
    - files that fail to decode are silently skipped
    """
    count = 0
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            for member in tar.getmembers():
                if not member.name.lower().endswith(".tex"):
                    continue
                # Skip oversized single files (>10MB), likely not paper body
                if member.size > 10_000_000:
                    continue
                f = tar.extractfile(member)
                if f is None:
                    continue
                try:
                    content = f.read().decode("utf-8", errors="ignore")
                except Exception:
                    continue
                count += _count_tables_in_tex(content)
    except (tarfile.TarError, EOFError, OSError):
        return 0
    return count


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python table_detector.py <file.tar.gz>")
        sys.exit(1)
    n = detect_table_count(sys.argv[1])
    print(f"{sys.argv[1]}: {n} tables")
