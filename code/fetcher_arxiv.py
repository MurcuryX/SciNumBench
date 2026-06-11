"""
fetcher_arxiv.py — SciNumGuard data ingestion layer · multi-discipline fetcher with filters

Metadata ingestion only; never downloads files.
Supports parallel crawling via a Tor proxy pool.
"""

import os
import re
import json
import time
import threading
import xml.etree.ElementTree as ET
import requests
from typing import List, Dict, Optional
from tqdm import tqdm
from ingestion_db import IngestionDB

# Proxy pool (optional)
try:
    from tor_pool import get_tor_proxy, get_tor_pool
    TOR_AVAILABLE = True
except ImportError:
    TOR_AVAILABLE = False

# ── Filter configuration ──

ALLOWED_PREFIXES = ("cs.", "math.", "physics.", "q-bio.", "q-fin.", "stat.", "eess.", "econ.")

TITLE_BLACKLIST = [
    r"(?i)^(editorial|erratum|corrigendum|addendum|retraction)\b",
    r"(?i)^(call for papers|conference|workshop|proceedings)\b",
    r"(?i)^(comment on|reply to|response to)\b",
]
RE_TITLE_BLACKLIST = [re.compile(p) for p in TITLE_BLACKLIST]

# Discipline query list + table-richness weight (higher weight = larger per-round fetch budget).
# Use specific subcategories rather than wildcard cat:cs.* — heavy wildcard queries via a Tor
# exit get hammered with 429s; specific subcategories avoid 429 and precisely target
# table-dense fields (ML/CV/NLP/IR/DB, etc.).
DISCIPLINE_QUERIES = [
    ("cs.LG",    "cat:cs.LG",    6),   # machine learning — densest tables
    ("cs.CV",    "cat:cs.CV",    5),   # computer vision
    ("cs.CL",    "cat:cs.CL",    5),   # NLP
    ("cs.AI",    "cat:cs.AI",    3),
    ("cs.IR",    "cat:cs.IR",    2),   # information retrieval
    ("cs.DB",    "cat:cs.DB",    2),   # databases (ICDE-related)
    ("cs.SE",    "cat:cs.SE",    2),   # software engineering
    ("cs.NI",    "cat:cs.NI",    2),   # networking
    ("cs.CR",    "cat:cs.CR",    2),   # security
    ("stat.ML",  "cat:stat.ML",  3),
    ("stat.ME",  "cat:stat.ME",  2),
    ("eess.IV",  "cat:eess.IV",  2),   # image/video processing
    ("eess.SP",  "cat:eess.SP",  2),   # signal processing
    ("q-bio.QM", "cat:q-bio.QM", 1),   # quantitative biology (many experiment tables)
    ("econ.EM",  "cat:econ.EM",  1),   # econometrics (many regression tables)
]
_TOTAL_WEIGHT = sum(w for _, _, w in DISCIPLINE_QUERIES)

# Cross-round persistent "paging depth": avoid rescanning the latest batch from start=0 each round (root cause of churn)
OFFSET_FILE = os.environ.get("SCINUM_OFFSET_FILE", "fetch_offsets.json")


def _load_offsets() -> dict:
    try:
        with open(OFFSET_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_offsets(offsets: dict):
    try:
        with open(OFFSET_FILE, "w", encoding="utf-8") as f:
            json.dump(offsets, f)
    except Exception:
        pass

ARXIV_API = "https://export.arxiv.org/api/query"
NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
SESSION = requests.Session()
SESSION.trust_env = False

# ── Rate-limit configuration ──
# When fetching metadata directly, arXiv returns 429 on sustained per-IP crawling. Slow down
# request intervals to avoid triggering it; keep the backoff penalty small so that when stuck
# we quickly abandon the batch and resume next round via the persisted offset, instead of idling for minutes.
REQUEST_DELAY = int(os.environ.get("SCINUM_REQ_DELAY", "10"))   # interval between requests (seconds)
BACKOFF_BASE = int(os.environ.get("SCINUM_BACKOFF_BASE", "20")) # initial 429 backoff (seconds)
BACKOFF_MAX = int(os.environ.get("SCINUM_BACKOFF_MAX", "90"))   # backoff cap (seconds)
MAX_RETRIES = int(os.environ.get("SCINUM_MAX_RETRIES", "4"))    # max retries on 429

# ── Proxy configuration (Tor onion) ──
# Toggled via environment variables, disabled by default (keeps bare connection / Windows behavior unchanged):
#   SCINUM_USE_TOR=1         enable Tor proxy pool
#   SCINUM_TOR_INSTANCES=5   number of instances
# External SOCKS ports (pointing at server0's Tor via SSH tunnel). When set, use "tunnel mode"; no local Tor is started.
#   SCINUM_SOCKS_PORTS=9050,9051,...,9057
SOCKS_PORTS = [p.strip() for p in os.environ.get("SCINUM_SOCKS_PORTS", "").split(",") if p.strip()]
_EXT_PROXIES = [{"http": f"socks5h://127.0.0.1:{p}", "https": f"socks5h://127.0.0.1:{p}"} for p in SOCKS_PORTS]

# Hybrid mode: metadata queries use a bare connection (export.arxiv.org public API throttles Tor exits very hard → switch to direct).
# The download side (download_scheduler) still goes over onion; they do not interfere.
FETCH_DIRECT = os.environ.get("SCINUM_FETCH_DIRECT", "0") == "1"

USE_TOR = os.environ.get("SCINUM_USE_TOR", "0") == "1" or bool(_EXT_PROXIES)
TOR_INSTANCES = int(os.environ.get("SCINUM_TOR_INSTANCES", "5"))

_pool = None
_ext_idx = [0]
_ext_lock = threading.Lock()


def _get_pool():
    """Lazily initialize the local Tor proxy pool (only when USE_TOR and not using external tunnel ports)."""
    global _pool
    if _pool is None and USE_TOR and not _EXT_PROXIES and TOR_AVAILABLE:
        _pool = get_tor_pool(num_instances=TOR_INSTANCES)
    return _pool


def _proxy():
    """Get the next exit proxy (round-robin). Prefer external tunnel ports (server0 Tor); else local Tor pool; else bare connection (None)."""
    if FETCH_DIRECT:          # hybrid mode: metadata via bare connection
        return None
    if _EXT_PROXIES:
        with _ext_lock:
            pr = _EXT_PROXIES[_ext_idx[0] % len(_EXT_PROXIES)]
            _ext_idx[0] += 1
        return pr
    pool = _get_pool()
    return pool.get_proxy() if pool else None


def _is_research_paper(categories: list) -> bool:
    return any(c.startswith(ALLOWED_PREFIXES) for c in categories)


def _is_valid_title(title: str) -> bool:
    for pat in RE_TITLE_BLACKLIST:
        if pat.search(title):
            return False
    return len(title.strip()) >= 10


def _fetch_batch(query: str, start: int, batch_size: int) -> List[Dict]:
    """Fetch one page from arXiv API. Returns list of paper dicts.
    Automatically retries 429 errors with exponential backoff.
    """
    params = {
        "search_query": query,
        "start": start,
        "max_results": batch_size,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }

    proxy = _proxy()          # Tor exit (None = bare connection when not enabled)
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = SESSION.get(ARXIV_API, params=params, timeout=60, proxies=proxy)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            # Occasional Tor circuit jitter: switch to another exit and retry instead of giving up the whole batch
            last_err = e
            proxy = _proxy()
            tqdm.write(f"[NET] Connection error {type(e).__name__}, switching exit and retrying ({attempt+1}/{MAX_RETRIES})")
            time.sleep(3)
            continue

        if resp.status_code == 429:
            # Prefer the server-provided Retry-After
            retry_after = resp.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                wait = int(retry_after)
            else:
                wait = min(BACKOFF_BASE * (2 ** attempt), BACKOFF_MAX)
            # When using Tor, switching exit IP is more effective than waiting it out, so shorten the wait
            if proxy:
                proxy = _proxy()
                wait = min(wait, 15)
            tqdm.write(f"[429] Rate limited, waiting {wait}s before retry ({attempt+1}/{MAX_RETRIES})...")
            time.sleep(wait)
            continue

        resp.raise_for_status()
        break
    else:
        # All retries failed
        raise Exception(f"Still failing after {MAX_RETRIES} rate-limit/network retries: {last_err or '429'}")

    root = ET.fromstring(resp.text)
    papers = []

    for entry in root.findall("atom:entry", NS):
        # Extract arxiv_id
        entry_id = entry.find("atom:id", NS)
        if entry_id is None:
            continue
        arxiv_id = entry_id.text.strip().split("/abs/")[-1]
        # Strip version suffix v1, v2, etc.
        arxiv_id = re.sub(r'v\d+$', '', arxiv_id)

        # Title
        title_el = entry.find("atom:title", NS)
        title = title_el.text.strip().replace("\n", " ") if title_el is not None else ""
        title = re.sub(r'\s+', ' ', title)

        # Abstract
        summary_el = entry.find("atom:summary", NS)
        abstract = summary_el.text.strip().replace("\n", " ") if summary_el is not None else ""
        abstract = re.sub(r'\s+', ' ', abstract)

        # Authors
        authors = []
        for author in entry.findall("atom:author", NS):
            name = author.find("atom:name", NS)
            if name is not None:
                authors.append(name.text.strip())
        authors_str = ", ".join(authors[:15])
        if len(authors) > 15:
            authors_str += f" et al. ({len(authors)})"

        # Categories
        categories = []
        primary_category = ""
        for cat in entry.findall("atom:category", NS):
            term = cat.get("term", "")
            if term:
                categories.append(term)
        # arxiv:primary_category
        pc = entry.find("arxiv:primary_category", NS)
        if pc is not None:
            primary_category = pc.get("term", "")
        if not primary_category and categories:
            primary_category = categories[0]

        # Date
        published_el = entry.find("atom:published", NS)
        publish_date = published_el.text[:10] if published_el is not None else ""

        papers.append({
            "arxiv_id": arxiv_id,
            "title": title,
            "primary_category": primary_category,
            "authors": authors_str,
            "publish_date": publish_date,
            "abstract": abstract,
            "_categories": categories,
        })

    return papers


class ArxivFetcher:
    """Fetch multi-discipline paper metadata from arXiv with filtering. No downloads."""

    def __init__(self, db: IngestionDB):
        self.db = db

    def fetch(self, max_results: int = 10000) -> int:
        """Fetch metadata across disciplines by weight, persisting paging depth across rounds (page deeper in time rather than rescanning the latest batch).

        - Bias toward table-rich disciplines: cs/stat/eess have high weight, math/physics/econ low.
        - Per-discipline budget this round = max_results × weight share; continue paging deeper from last round's persisted offset.
        - Skip already-ingested papers (dedup); offset keeps advancing, fundamentally eliminating "rescan latest → churn".
        """
        offsets = _load_offsets()
        batch_insert: List[Dict] = []
        total_fetched = 0
        total_accepted = 0
        total_skipped = 0

        # Load the set of existing arxiv_id values for deduplication
        with self.db._get_conn() as conn:
            existing_ids = set(
                r[0] for r in conn.execute("SELECT arxiv_id FROM raw_papers").fetchall()
            )

        pbar = tqdm(total=max_results, desc="Fetching arXiv multi-discipline metadata", unit="paper")

        for disc_name, query, weight in DISCIPLINE_QUERIES:
            if total_accepted >= max_results:
                break

            # Per-discipline budget this round (allocated by weight, at least 50)
            budget = max(50, int(max_results * weight / _TOTAL_WEIGHT))
            start = int(offsets.get(disc_name, 0))
            end = start + budget

            while start < end and total_accepted < max_results:
                batch_size = min(100, end - start)

                try:
                    papers = _fetch_batch(query, start, batch_size)
                except Exception as e:
                    tqdm.write(f"[WARN] {disc_name} start={start}: {e}")
                    break

                if not papers:
                    # This discipline has been paged to the end by submittedDate (or arxiv limited depth) → skip this round
                    tqdm.write(f"  [{disc_name}] start={start} no more results, skipping this round")
                    break

                for p in papers:
                    total_fetched += 1
                    pbar.update(1)

                    # Already ingested → skip (does not affect offset advancement)
                    if p["arxiv_id"] in existing_ids:
                        total_skipped += 1
                        continue

                    # Filter
                    if not _is_research_paper(p["_categories"]):
                        continue
                    if not _is_valid_title(p["title"]):
                        continue

                    # Ingest
                    del p["_categories"]
                    existing_ids.add(p["arxiv_id"])
                    batch_insert.append(p)
                    total_accepted += 1

                    if total_accepted >= max_results:
                        break

                if len(batch_insert) >= 100:
                    self.db.upsert_papers(batch_insert)
                    batch_insert = []

                start += batch_size
                time.sleep(REQUEST_DELAY)

            # Persist this discipline's new paging depth (next round continues deeper from here, no longer returning to the latest batch)
            offsets[disc_name] = start
            _save_offsets(offsets)

            time.sleep(4)  # discipline-switch interval (specific subcategories no longer 429, can be shortened)

        if batch_insert:
            self.db.upsert_papers(batch_insert)

        pbar.close()
        _save_offsets(offsets)

        print(f"\n[Filter stats] fetched {total_fetched} papers → {total_accepted} new (skipped {total_skipped} duplicates)")
        print("[Paging depth] " + ", ".join(f"{d}={offsets.get(d, 0)}" for d, _, _ in DISCIPLINE_QUERIES))
        total, pending, success, failed = self.db.count()
        print(f"[Database] {total} papers total, {pending} pending download")
        return total_accepted


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    db = IngestionDB()
    fetcher = ArxivFetcher(db)
    fetcher.fetch(max_results=n)
    db.print_stats()
