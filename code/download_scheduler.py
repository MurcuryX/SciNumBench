"""
download_scheduler.py — SciNumGuard data ingestion layer (concurrent downloader)

Flow: download LaTeX source -> parse tables (with position info) -> store in DB -> delete source file
success -> status=1
failure / no source -> status=-1, discarded
"""

import os
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from ingestion_db import IngestionDB
from table_parser import extract_from_tarball

RAW_DATA_DIR = "./raw_data/arxiv"
MAX_WORKERS = 6
CONNECT_TIMEOUT = 10
READ_TIMEOUT = 30

SESSION = requests.Session()
SESSION.trust_env = False

# Tor onion proxy (toggled by env var, off by default)
try:
    from tor_pool import get_tor_pool
    TOR_AVAILABLE = True
except ImportError:
    TOR_AVAILABLE = False

import threading

# External SOCKS ports (via SSH tunnel to remote Tor); when set, use tunnel mode and do not start a local Tor
SOCKS_PORTS = [p.strip() for p in os.environ.get("SCINUM_SOCKS_PORTS", "").split(",") if p.strip()]
_EXT_PROXIES = [{"http": f"socks5h://127.0.0.1:{p}", "https": f"socks5h://127.0.0.1:{p}"} for p in SOCKS_PORTS]

USE_TOR = os.environ.get("SCINUM_USE_TOR", "0") == "1" or bool(_EXT_PROXIES)
TOR_INSTANCES = int(os.environ.get("SCINUM_TOR_INSTANCES", "5"))
MAX_DL_RETRIES = int(os.environ.get("SCINUM_DL_RETRIES", "3"))   # retries for Tor circuit jitter

_pool = None
_ext_idx = [0]
_ext_lock = threading.Lock()


def _get_pool():
    global _pool
    if _pool is None and USE_TOR and not _EXT_PROXIES and TOR_AVAILABLE:
        _pool = get_tor_pool(num_instances=TOR_INSTANCES)
    return _pool


def _proxy():
    """Prefer external tunnel ports (remote Tor, thread-safe round-robin); else local Tor pool; else direct connection."""
    if _EXT_PROXIES:
        with _ext_lock:
            pr = _EXT_PROXIES[_ext_idx[0] % len(_EXT_PROXIES)]
            _ext_idx[0] += 1
        return pr
    pool = _get_pool()
    return pool.get_proxy() if pool else None


def _download_source(arxiv_id: str) -> tuple[bool, str, str]:
    """Download LaTeX source. Returns (success, error_msg, file_path).

    When using Tor, retry on timeout/connection failure with a different exit to avoid
    circuit jitter being mistaken for a permanent failure.
    Real HTTP errors (e.g. 404 no source) are not retried.
    """
    url = f"https://arxiv.org/src/{arxiv_id}"
    safe_id = arxiv_id.replace("/", "_").replace(":", "_")
    file_path = os.path.join(RAW_DATA_DIR, f"{safe_id}.tar.gz")

    attempts = MAX_DL_RETRIES if USE_TOR else 1
    last_err = "unknown"
    for attempt in range(attempts):
        proxy = _proxy()        # None when Tor is not enabled (direct connection)
        try:
            resp = SESSION.get(url, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT), stream=True,
                               headers={"User-Agent": "SciNumGuard/1.0"}, proxies=proxy)
            resp.raise_for_status()

            ct = resp.headers.get("content-type", "")
            if "html" in ct.lower():
                return False, "no source (HTML)", ""

            with open(file_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)

            if os.path.getsize(file_path) < 512:
                os.remove(file_path)
                return False, "file too small", ""

            with open(file_path, "rb") as f:
                if f.read(2) != b"\x1f\x8b":
                    os.remove(file_path)
                    return False, "not gzip", ""

            return True, "", file_path

        except requests.exceptions.HTTPError as e:
            return False, f"HTTP {e.response.status_code}", ""
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_err = "timeout" if isinstance(e, requests.exceptions.Timeout) else "connection failed"
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except OSError:
                    pass
            time.sleep(2)
            continue        # switch Tor exit on next attempt
        except Exception as e:
            last_err = type(e).__name__
            continue

    return False, last_err, ""


def download_one(arxiv_id: str, db_path: str) -> str:
    """Download → parse tables → store → delete source. Returns status message."""
    db = IngestionDB(db_path)

    safe_id = arxiv_id.replace("/", "_").replace(":", "_")
    file_path = os.path.join(RAW_DATA_DIR, f"{safe_id}.tar.gz")

    # if already present, skip download but still parse
    already_exists = os.path.exists(file_path) and os.path.getsize(file_path) > 512

    if not already_exists:
        t0 = time.perf_counter()
        ok, err, fpath = _download_source(arxiv_id)
        elapsed = time.perf_counter() - t0

        if not ok:
            db.mark_failed(arxiv_id)
            return f"[SKIP] {arxiv_id} | {err} | {elapsed:.1f}s"

        db.mark_success(arxiv_id, fpath)
    else:
        fpath = file_path
        db.mark_success(arxiv_id, fpath)

    # parse tables
    try:
        tables = extract_from_tarball(fpath)
        if tables:
            db.store_tables(arxiv_id, tables)
            table_info = f" -> {len(tables)} tables"
        else:
            table_info = " -> 0 tables"
    except Exception:
        table_info = " -> parse failed"

    # delete source file
    try:
        if fpath and os.path.exists(fpath):
            os.remove(fpath)
    except Exception:
        pass

    if already_exists:
        return f"[SKIP] {arxiv_id} | already exists{table_info}"
    return f"[OK]   {arxiv_id} | {elapsed:.1f}s{table_info}"


def run_scheduler(db_path: str = "arxiv_data.db"):
    """Main loop: fetch pending → concurrent download → parse → repeat."""
    db = IngestionDB(db_path)
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    total_ok = 0
    total_skip = 0
    round_num = 0

    while True:
        pending = db.get_pending_downloads(limit=50)
        if not pending:
            break

        round_num += 1
        pbar = tqdm(total=len(pending), desc=f"round {round_num}", unit="paper")

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(download_one, p["arxiv_id"], db_path): p
                for p in pending
            }
            for future in as_completed(futures):
                msg = future.result()
                pbar.update(1)
                if msg.startswith("[OK]"):
                    total_ok += 1
                else:
                    total_skip += 1
                tqdm.write(msg)

        pbar.close()

    # clean up empty directories
    if os.path.exists(RAW_DATA_DIR):
        try:
            os.rmdir(RAW_DATA_DIR)
            os.rmdir(os.path.dirname(RAW_DATA_DIR))
        except OSError:
            pass

    print(f"Download complete: success {total_ok} | failed/discarded {total_skip}")
    db.print_stats()


if __name__ == "__main__":
    run_scheduler()
