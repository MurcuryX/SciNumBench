"""
ingestion_db.py — SciNumGuard data ingestion layer: metadata and experiment bus

Database: arxiv_data.db
Tables:   raw_papers (paper metadata)
          paper_tables (table data + location info)
"""

import sqlite3
from typing import List, Dict, Optional, Tuple


class IngestionDB:
    """SQLite store for arXiv paper metadata, tables, and download scheduling."""

    def __init__(self, db_path: str = "arxiv_data.db"):
        self.db_path = db_path
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _init_db(self):
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS raw_papers (
                    arxiv_id          TEXT PRIMARY KEY,
                    title             TEXT NOT NULL,
                    primary_category  TEXT,
                    authors           TEXT,
                    publish_date      TEXT,
                    abstract          TEXT,
                    file_path         TEXT,
                    status            INTEGER DEFAULT 0,
                    table_count       INTEGER DEFAULT -1,
                    is_human_modified INTEGER DEFAULT 0
                )
            """)
            # Migration: add missing column to old tables
            try:
                conn.execute("ALTER TABLE raw_papers ADD COLUMN table_count INTEGER DEFAULT -1")
            except sqlite3.OperationalError:
                pass  # column already exists
            conn.execute("""
                CREATE TABLE IF NOT EXISTS paper_tables (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    arxiv_id        TEXT NOT NULL,
                    table_index     INTEGER NOT NULL,
                    caption         TEXT,
                    label           TEXT,
                    section         TEXT,
                    page            INTEGER,
                    rows            INTEGER,
                    cols            INTEGER,
                    table_json      TEXT NOT NULL,
                    FOREIGN KEY (arxiv_id) REFERENCES raw_papers(arxiv_id)
                )
            """)
            conn.commit()

    def reset_db(self):
        """Drop and recreate tables. Use for fresh start."""
        with self._get_conn() as conn:
            conn.execute("DROP TABLE IF EXISTS paper_tables")
            conn.execute("DROP TABLE IF EXISTS raw_papers")
            conn.commit()
        self._init_db()

    # ── Paper writes ──────────────────────────────────────────────────────

    def upsert_papers(self, papers: List[Dict]):
        """Batch upsert paper metadata. Dedup by arxiv_id."""
        if not papers:
            return
        with self._get_conn() as conn:
            conn.executemany(
                """INSERT INTO raw_papers
                   (arxiv_id, title, primary_category, authors, publish_date, abstract)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(arxiv_id) DO UPDATE SET
                       title=excluded.title,
                       abstract=excluded.abstract
                """,
                [
                    (
                        p["arxiv_id"],
                        p["title"],
                        p.get("primary_category", ""),
                        p.get("authors", ""),
                        p.get("publish_date", ""),
                        p.get("abstract", ""),
                    )
                    for p in papers
                ],
            )
            conn.commit()

    def mark_success(self, arxiv_id: str, file_path: str):
        """Mark paper as successfully downloaded (status=1)."""
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE raw_papers SET status=1, file_path=? WHERE arxiv_id=?",
                (file_path, arxiv_id),
            )
            conn.commit()

    def mark_failed(self, arxiv_id: str):
        """Mark paper as failed / no source (status=-1)."""
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE raw_papers SET status=-1 WHERE arxiv_id=?",
                (arxiv_id,),
            )
            conn.commit()

    # ── Table writes ──────────────────────────────────────────────────────

    def store_tables(self, arxiv_id: str, tables: List[Dict]):
        """Store extracted tables with location info.
        Each dict: {table_index, caption, label, section, page, rows, cols, table_json}
        """
        if not tables:
            return
        with self._get_conn() as conn:
            conn.executemany(
                """INSERT INTO paper_tables
                   (arxiv_id, table_index, caption, label, section, page, rows, cols, table_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        arxiv_id,
                        t["table_index"],
                        t.get("caption", ""),
                        t.get("label", ""),
                        t.get("section", ""),
                        t.get("page", 1),
                        t.get("rows", 0),
                        t.get("cols", 0),
                        t["table_json"],
                    )
                    for t in tables
                ],
            )
            conn.commit()

    # ── Queries ───────────────────────────────────────────────────────────

    def get_pending_downloads(self, limit: int = 50) -> List[Dict]:
        """Get papers with status=0 (pending download)."""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT arxiv_id, title FROM raw_papers WHERE status=0 LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def count(self) -> Tuple[int, int, int, int]:
        """Returns (total, pending, success, failed)."""
        with self._get_conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM raw_papers").fetchone()[0]
            pending = conn.execute("SELECT COUNT(*) FROM raw_papers WHERE status=0").fetchone()[0]
            success = conn.execute("SELECT COUNT(*) FROM raw_papers WHERE status=1").fetchone()[0]
            failed = conn.execute("SELECT COUNT(*) FROM raw_papers WHERE status=-1").fetchone()[0]
        return total, pending, success, failed

    def table_count(self) -> Tuple[int, int]:
        """Returns (total_tables, papers_with_tables)."""
        with self._get_conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM paper_tables").fetchone()[0]
            papers = conn.execute("SELECT COUNT(DISTINCT arxiv_id) FROM paper_tables").fetchone()[0]
        return total, papers

    def backfill_table_count(self):
        """Backfill table_count from existing paper_tables data, for migrating old data.

        Only processes downloaded papers with status=1; leaves status=0 (pending) records untouched.
        """
        with self._get_conn() as conn:
            # Count tables per arxiv_id from paper_tables
            table_map = dict(conn.execute(
                "SELECT arxiv_id, COUNT(*) FROM paper_tables GROUP BY arxiv_id"
            ).fetchall())

            # Only backfill records with status=1 and table_count=-1
            to_update = []
            for r in conn.execute(
                "SELECT arxiv_id FROM raw_papers WHERE status=1 AND table_count=-1"
            ).fetchall():
                to_update.append((table_map.get(r[0], 0), r[0]))

            if to_update:
                conn.executemany(
                    "UPDATE raw_papers SET table_count=? WHERE arxiv_id=?", to_update
                )

            # Force status=1 for papers that have tables
            conn.execute(
                "UPDATE raw_papers SET status=1 WHERE table_count>0 AND status!=1"
            )
            conn.commit()

            valid = conn.execute(
                "SELECT COUNT(*) FROM raw_papers WHERE status=1 AND table_count>0"
            ).fetchone()[0]
            total = conn.execute(
                "SELECT COUNT(*) FROM raw_papers"
            ).fetchone()[0]
            pending = conn.execute(
                "SELECT COUNT(*) FROM raw_papers WHERE status=0"
            ).fetchone()[0]
            print(f"[BACKFILL] done, total={total} | valid={valid} | pending={pending}")
            return valid

    def count_valid_papers(self) -> int:
        """Return the count of valid papers with status=1 and table_count>0."""
        with self._get_conn() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM raw_papers WHERE status=1 AND table_count>0"
            ).fetchone()[0]

    def update_table_count(self, arxiv_id: str, count: int):
        """Update the detected table count for the given paper."""
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE raw_papers SET table_count=? WHERE arxiv_id=?",
                (count, arxiv_id),
            )
            conn.commit()

    def clean_zero_table_papers(self, raw_dir: str = "./raw_data/arxiv") -> int:
        """Delete junk data with table_count=0 (DB records + local files); return cleanup count."""
        import os, glob
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT arxiv_id, file_path FROM raw_papers WHERE table_count=0"
            ).fetchall()

            deleted = 0
            for arxiv_id, fpath in rows:
                # Delete local files (prefer file_path, fall back to glob match)
                targets = []
                if fpath and os.path.exists(fpath):
                    targets.append(fpath)
                safe_id = arxiv_id.replace("/", "_").replace(":", "_")
                targets.extend(glob.glob(os.path.join(raw_dir, f"{safe_id}.*")))
                for t in set(targets):
                    try:
                        os.remove(t)
                        deleted += 1
                    except OSError:
                        pass

            conn.execute("DELETE FROM raw_papers WHERE table_count=0")
            conn.commit()

        print(f"[CLEAN] removed {len(rows)} papers with no tables (deleted {deleted} local files)")
        return len(rows)

    def print_stats(self):
        total, pending, success, failed = self.count()
        t_total, t_papers = self.table_count()
        valid = self.count_valid_papers()
        print(f"Database: {self.db_path}")
        print(f"  Papers: {total} (success={success} / pending={pending} / failed={failed})")
        print(f"  Valid: {valid} (with tables)")
        print(f"  Tables: {t_total} (from {t_papers} papers)")
