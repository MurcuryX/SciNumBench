"""
auto_replenish.py — SciNumGuard automatic cleaning and dynamic replenishment controller

Loop: fetch -> download (keep source files) -> probe -> clean junk data -> parse valid papers -> ...
until valid papers (status=1 and table_count>0) reach the target watermark.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from ingestion_db import IngestionDB
from fetcher_arxiv import ArxivFetcher
from download_scheduler import _download_source, RAW_DATA_DIR, MAX_WORKERS
from table_detector import detect_table_count
from table_parser import extract_tables_from_tex, extract_from_tarball

TARGET_VALID_PAPERS = 10_000
DB_PATH = "arxiv_data.db"
RAW_DIR = "./raw_data/arxiv"


# Stage 1: download (keep source files, no parsing)

def download_batch(db: IngestionDB):
    """Batch download source for status=0 papers, keeping files for later probing."""
    os.makedirs(RAW_DIR, exist_ok=True)
    total_ok, total_skip = 0, 0
    round_num = 0

    while True:
        pending = db.get_pending_downloads(limit=200)
        if not pending:
            break

        round_num += 1
        pbar = tqdm(total=len(pending), desc=f"[download] round {round_num}", unit="paper")

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {}
            for p in pending:
                futures[executor.submit(_download_source, p["arxiv_id"])] = p

            for future in as_completed(futures):
                p = futures[future]
                ok, err, fpath = future.result()
                pbar.update(1)
                if ok:
                    db.mark_success(p["arxiv_id"], fpath)
                    total_ok += 1
                else:
                    db.mark_failed(p["arxiv_id"])
                    total_skip += 1

        pbar.close()

    print(f"[download] success {total_ok} | failed {total_skip}")


# Stage 2: probe

def probe_all(db: IngestionDB):
    """Probe all papers with status=1 and table_count=-1, updating table counts."""
    with db._get_conn() as conn:
        rows = conn.execute(
            "SELECT arxiv_id, file_path FROM raw_papers WHERE status=1 AND table_count=-1"
        ).fetchall()

    if not rows:
        print("[PROBE] no new papers to probe")
        return 0

    print(f"[PROBE] probing {len(rows)} papers...")
    probed, has_tables = 0, 0

    for arxiv_id, fpath in rows:
        # construct file path
        if not fpath or not os.path.exists(fpath):
            safe_id = arxiv_id.replace("/", "_").replace(":", "_")
            fpath = os.path.join(RAW_DIR, f"{safe_id}.tar.gz")

        if not os.path.exists(fpath):
            # file does not exist (deleted by old flow); skip without marking 0,
            # let backfill_table_count refill from paper_tables
            continue

        tc = detect_table_count(fpath)
        db.update_table_count(arxiv_id, tc)
        probed += 1
        if tc > 0:
            has_tables += 1

    print(f"[PROBE] probe done: {probed} papers, {has_tables} contain tables")
    return has_tables


# Stage 3: parse valid papers

def parse_valid_papers(db: IngestionDB):
    """Fully parse papers with table_count>0 that are not yet stored in paper_tables."""
    with db._get_conn() as conn:
        # find papers that have tables but no record in paper_tables
        rows = conn.execute("""
            SELECT r.arxiv_id, r.file_path
            FROM raw_papers r
            WHERE r.status=1 AND r.table_count>0
              AND r.arxiv_id NOT IN (SELECT DISTINCT arxiv_id FROM paper_tables)
        """).fetchall()

    if not rows:
        print("[PARSE] no new papers to parse")
        return

    print(f"[PARSE] parsing tables of {len(rows)} valid papers...")
    parsed, table_total = 0, 0

    for arxiv_id, fpath in rows:
        if not fpath or not os.path.exists(fpath):
            safe_id = arxiv_id.replace("/", "_").replace(":", "_")
            fpath = os.path.join(RAW_DIR, f"{safe_id}.tar.gz")

        if not os.path.exists(fpath):
            continue

        try:
            tables = extract_from_tarball(fpath)
            if tables:
                db.store_tables(arxiv_id, tables)
                table_total += len(tables)
            parsed += 1
        except Exception:
            pass

    print(f"[PARSE] parse done: {parsed} papers, {table_total} tables total")


# Stage 4: cleanup

def cleanup(db: IngestionDB):
    """Delete junk data with table_count=0 to free space."""
    db.clean_zero_table_papers(raw_dir=RAW_DIR)


# Single replenishment cycle

def replenish_cycle(db: IngestionDB, fetcher: ArxivFetcher, gap: int):
    """Single cycle: (fetch a small batch on demand) -> download -> probe -> parse -> clean.

    Each cycle only tops up the download queue to BATCH before starting downloads,
    so valid keeps growing; this avoids the original logic of fetching thousands of
    metadata records at once before downloading, which delayed output.
    """
    BATCH = int(os.environ.get("SCINUM_BATCH", "500"))
    with db._get_conn() as conn:
        pending = conn.execute("SELECT COUNT(*) FROM raw_papers WHERE status=0").fetchone()[0]

    if pending < BATCH:
        fetch_count = BATCH - pending
        print(f"\n[FETCH] download queue has {pending} papers, fetching {fetch_count} more (target batch {BATCH})...")
        fetcher.fetch(max_results=fetch_count)
    else:
        print(f"\n[FETCH] download queue already has {pending} papers (>={BATCH}), downloading without fetching")

    # download source (keep files)
    download_batch(db)

    # probe table count
    probe_all(db)

    # parse tables of valid papers
    parse_valid_papers(db)

    # clean junk data + free space
    cleanup(db)


# Main entry

def main():
    db = IngestionDB(DB_PATH)
    fetcher = ArxivFetcher(db)
    os.makedirs(RAW_DIR, exist_ok=True)

    # read-only display of current status
    db.print_stats()
    valid = db.count_valid_papers()
    gap = TARGET_VALID_PAPERS - valid

    if gap <= 0:
        print(f"\nAlready have {valid} valid papers, target reached, no action needed.")
        return

    print(f"\nCurrent valid papers: {valid} | target: {TARGET_VALID_PAPERS} | gap: {gap}")
    print("Steps: backfill table_count -> probe new papers -> parse tables -> clean junk -> loop fetch to replenish")
    ans = input("Confirm start? [y/N] ").strip().lower()
    if ans != "y":
        print("Cancelled.")
        return

    # executed only after user confirmation

    # backfill table_count of old data
    db.backfill_table_count()

    # probe + parse + clean
    probe_all(db)
    parse_valid_papers(db)
    cleanup(db)
    db.print_stats()

    cycle = 0
    while True:
        valid = db.count_valid_papers()
        gap = TARGET_VALID_PAPERS - valid

        if gap <= 0:
            break

        cycle += 1
        print(f"[INFO] cycle {cycle} | valid papers: {valid} | gap to target: {gap}")

        replenish_cycle(db, fetcher, gap)
        db.print_stats()

    # target reached
    final = db.count_valid_papers()
    print(f"Target reached. Valid papers: {final}")
    db.print_stats()


if __name__ == "__main__":
    main()
