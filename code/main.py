"""
main.py — SciNumGuard data ingestion layer, one-click run

Pipeline:
  Step 1: Fetch multi-discipline arXiv paper metadata (with filtering)
  Step 2: Concurrently download LaTeX sources -> parse tables (with position info) -> store in database -> delete source files

Usage:
  python main.py              # default 10000 papers
  python main.py 100          # test with 100 papers
"""

import sys
from ingestion_db import IngestionDB
from fetcher_arxiv import ArxivFetcher
from download_scheduler import run_scheduler


def main():
    max_papers = int(sys.argv[1]) if len(sys.argv) > 1 else 10000

    # Step 1: fetch metadata
    db = IngestionDB()
    fetcher = ArxivFetcher(db)
    fetcher.fetch(max_results=max_papers)
    db.print_stats()

    # Step 2: download + parse tables
    run_scheduler()

    import sqlite3
    conn = sqlite3.connect("arxiv_data.db")

    # Paper statistics
    total = conn.execute("SELECT COUNT(*) FROM raw_papers").fetchone()[0]
    ok = conn.execute("SELECT COUNT(*) FROM raw_papers WHERE status=1").fetchone()[0]
    fail = conn.execute("SELECT COUNT(*) FROM raw_papers WHERE status=-1").fetchone()[0]
    print(f"\nPapers: {total} (success {ok} / failed {fail})")

    # Table statistics
    t_total = conn.execute("SELECT COUNT(*) FROM paper_tables").fetchone()[0]
    t_papers = conn.execute("SELECT COUNT(DISTINCT arxiv_id) FROM paper_tables").fetchone()[0]
    print(f"Tables: {t_total} (from {t_papers} papers)")

    # Category distribution
    print("\nCategory distribution:")
    cats = conn.execute("""
        SELECT primary_category, COUNT(*) FROM raw_papers
        GROUP BY primary_category ORDER BY COUNT(*) DESC LIMIT 10
    """).fetchall()
    for cat, cnt in cats:
        print(f"  {cat:15s} {cnt} papers")

    # Table samples (with position)
    print("\nTable samples:")
    samples = conn.execute("""
        SELECT t.arxiv_id, t.table_index, t.caption, t.label, t.section, t.rows, t.cols
        FROM paper_tables t LIMIT 5
    """).fetchall()
    for s in samples:
        loc = f"#{s[1]}"
        if s[3]: loc += f" [{s[3]}]"
        if s[4]: loc += f" in {s[4]}"
        print(f"  {s[0]:20s} {loc:30s} {s[2][:40]:40s} ({s[5]}x{s[6]})")

    conn.close()
    print()


if __name__ == "__main__":
    main()
