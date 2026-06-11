"""
pmc_fetcher.py — fetch RCT/clinical-trial full text (JATS) from PMC OA, parse tables into the DB.

Unified with the arXiv pipeline: writes into the same arxiv_data.db raw_papers / paper_tables,
but with source='pmc', paper primary key 'PMCxxxx', and table_json using the same
orient=split structure {columns, index, data} as arXiv.

env:
  SCINUM_DB              DB path (default arxiv_data.db, relative to cwd)
  PMC_TERM              esearch query
  PMC_TARGET_ARTICLES   target number of articles (default 3000)
  PMC_EFETCH_BATCH      IDs per efetch call (default 40)
  PMC_REQ_DELAY         delay between requests in seconds (default 0.4, i.e. <3 req/s)
  NCBI_API_KEY          optional; if set, 10 req/s
"""
import os, time, json, re, sqlite3, socket, ssl, http.client, urllib.request, urllib.parse
from lxml import etree
from tqdm import tqdm

DB = os.environ.get("SCINUM_DB", "arxiv_data.db")
EUTILS_HOST = "eutils.ncbi.nlm.nih.gov"
EUTILS = f"https://{EUTILS_HOST}/entrez/eutils"
UA = {"User-Agent": "SciNumBench/1.0 (mailto:lera_accusamuscp@mail.com)"}
API_KEY = os.environ.get("NCBI_API_KEY", "")
REQ_DELAY = float(os.environ.get("PMC_REQ_DELAY", "0.4"))
TERM = os.environ.get("PMC_TERM", '"randomized controlled trial"[Publication Type] AND "open access"[filter]')
TARGET = int(os.environ.get("PMC_TARGET_ARTICLES", "3000"))
BATCH = int(os.environ.get("PMC_EFETCH_BATCH", "40"))
# tunnel: "127.0.0.1:9200" connects directly to eutils:443 via server0 (data travels over an ssh reverse tunnel)
PROXY = os.environ.get("PMC_PROXY_HOSTPORT", "")
_CURSOR_FILE = os.environ.get("PMC_CURSOR_FILE", "pmc_cursor.json")
_SSL = ssl.create_default_context()

P = etree.XMLParser(recover=True, resolve_entities=False, no_network=True, huge_tree=True)


def _get(url):
    if API_KEY:
        url += ("&" if "?" in url else "?") + "api_key=" + urllib.parse.quote(API_KEY)
    last = None
    for attempt in range(4):
        try:
            if PROXY:
                host, port = PROXY.split(":")
                sp = urllib.parse.urlsplit(url)
                path = sp.path + ("?" + sp.query if sp.query else "")
                raw = socket.create_connection((host, int(port)), timeout=120)
                sock = _SSL.wrap_socket(raw, server_hostname=EUTILS_HOST)
                c = http.client.HTTPSConnection(EUTILS_HOST, timeout=120)
                c.sock = sock
                c.request("GET", path, headers={"Host": EUTILS_HOST, **UA})
                r = c.getresponse()
                data = r.read()
                c.close()
                if r.status != 200:
                    raise RuntimeError(f"HTTP {r.status}")
                return data
            return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120).read()
        except Exception as e:
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"GET failed ({last}): {url[:120]}")


def migrate(conn):
    for t in ("raw_papers", "paper_tables"):
        cols = [r[1] for r in conn.execute(f"pragma table_info({t})")]
        if "source" not in cols:
            conn.execute(f"ALTER TABLE {t} ADD COLUMN source TEXT DEFAULT 'arxiv'")
    conn.commit()


def esearch_history(term):
    """usehistory=y -> (WebEnv, query_key, Count). Bypasses the retstart<=10000 limit to page through the full set."""
    url = f"{EUTILS}/esearch.fcgi?db=pmc&usehistory=y&retmax=0&term=" + urllib.parse.quote(term)
    xml = _get(url).decode("utf-8", "ignore")
    we = re.search(r"<WebEnv>(\S+?)</WebEnv>", xml)
    qk = re.search(r"<QueryKey>(\d+)</QueryKey>", xml)
    cnt = re.search(r"<Count>(\d+)</Count>", xml)
    if not (we and qk and cnt):
        raise RuntimeError("esearch history parse failed")
    return we.group(1), qk.group(1), int(cnt.group(1))


def efetch_history(webenv, qk, retstart, retmax):
    url = (f"{EUTILS}/efetch.fcgi?db=pmc&query_key={qk}&WebEnv={webenv}"
           f"&retstart={retstart}&retmax={retmax}")
    return _get(url)


def _load_cursor():
    try:
        with open(_CURSOR_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_cursor(d):
    tmp = _CURSOR_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(d, f)
    os.replace(tmp, _CURSOR_FILE)


def lt(el):
    return etree.QName(el).localname


def _cells(tr):
    out = []
    for c in tr:
        if lt(c) in ("td", "th"):
            txt = re.sub(r"\s+", " ", "".join(c.itertext()).strip())
            out.extend([txt] * int(c.get("colspan", "1") or "1"))
    return out


def parse_table(tbl):
    rows = [_cells(tr) for tr in tbl.iter() if lt(tr) == "tr"]
    rows = [r for r in rows if r]
    if len(rows) < 2:
        return None
    w = max(len(r) for r in rows)
    rows = [(r + [""] * w)[:w] for r in rows]
    cols, data = rows[0], rows[1:]
    return {"columns": cols, "index": list(range(len(data))), "data": data}


def article_records(art):
    """Parse one <article> into (paper_dict, [table_dicts])."""
    pmcid = ""
    pmcaid = ""
    for e in art.iter():
        if lt(e) == "article-id":
            t = e.get("pub-id-type")
            val = (e.text or "").strip()
            if t in ("pmcid", "pmc") and val:
                pmcid = val if val.upper().startswith("PMC") else "PMC" + val
                break
            if t in ("pmcaid", "pmc-uid") and val and not pmcaid:
                pmcaid = "PMC" + val
    if not pmcid:
        pmcid = pmcaid
    if not pmcid:
        return None, []

    def first_text(tag):
        for e in art.iter():
            if lt(e) == tag:
                return re.sub(r"\s+", " ", "".join(e.itertext()).strip())
        return ""

    title = first_text("article-title")
    journal = first_text("journal-title")
    # publication year
    year = ""
    for e in art.iter():
        if lt(e) == "pub-date":
            y = e.find("{*}year") if False else None
            for ch in e:
                if lt(ch) == "year":
                    year = (ch.text or "").strip()
                    break
            if year:
                break
    # authors
    authors = []
    for e in art.iter():
        if lt(e) == "contrib" and e.get("contrib-type") == "author":
            sn = gn = ""
            for ch in e.iter():
                if lt(ch) == "surname":
                    sn = (ch.text or "").strip()
                elif lt(ch) == "given-names":
                    gn = (ch.text or "").strip()
            nm = (gn + " " + sn).strip()
            if nm:
                authors.append(nm)
        if len(authors) >= 15:
            break
    abstract = first_text("abstract")[:4000]

    tables = []
    for tw in art.iter():
        if lt(tw) != "table-wrap":
            continue
        label = ""
        cap = ""
        for ch in tw.iter():
            if lt(ch) == "label" and not label:
                label = "".join(ch.itertext()).strip()
            if lt(ch) == "caption" and not cap:
                cap = re.sub(r"\s+", " ", "".join(ch.itertext()).strip())
        tbl = next((e for e in tw.iter() if lt(e) == "table"), None)
        if tbl is None:
            continue
        grid = parse_table(tbl)
        if grid is None:
            continue
        tables.append({
            "label": label,
            "caption": cap,
            "rows": len(grid["data"]),
            "cols": len(grid["columns"]),
            "table_json": json.dumps(grid, ensure_ascii=False),
        })

    paper = {
        "arxiv_id": pmcid,
        "title": title,
        "primary_category": "pmc.rct",
        "authors": ", ".join(authors),
        "publish_date": year,
        "abstract": abstract,
        "journal": journal,
    }
    return paper, tables


def _insert(conn, paper, tables):
    conn.execute(
        """INSERT OR IGNORE INTO raw_papers
           (arxiv_id, title, primary_category, authors, publish_date,
            abstract, status, table_count, is_human_modified, source)
           VALUES (?, ?, ?, ?, ?, ?, 1, ?, 0, 'pmc')""",
        (paper["arxiv_id"], paper["title"], paper["primary_category"],
         paper["authors"], paper["publish_date"], paper["abstract"], len(tables)))
    for ti, t in enumerate(tables):
        conn.execute(
            """INSERT INTO paper_tables
               (arxiv_id, table_index, caption, label, section, page,
                rows, cols, table_json, source)
               VALUES (?, ?, ?, ?, '', 0, ?, ?, ?, 'pmc')""",
            (paper["arxiv_id"], ti, t["caption"], t["label"],
             t["rows"], t["cols"], t["table_json"]))


def main():
    conn = sqlite3.connect(DB)
    migrate(conn)
    existing = set(r[0] for r in conn.execute(
        "SELECT arxiv_id FROM raw_papers WHERE source='pmc'"))
    print(f"[PMC] DB already has {len(existing)} PMC papers | tunnel={PROXY or 'direct'} | target new={TARGET}", flush=True)
    print(f"[PMC] esearch: {TERM}", flush=True)

    cursors = _load_cursor()
    cursor = int(cursors.get(TERM, 0))
    webenv, qk, total = esearch_history(TERM)
    print(f"[PMC] history mode: pool of {total} articles, paging from retstart={cursor}", flush=True)
    n_paper = n_table = 0
    t0 = time.time()
    last_report = 0
    empty_streak = 0
    while n_paper < TARGET and cursor < total:
        try:
            root = etree.fromstring(efetch_history(webenv, qk, cursor, BATCH), P)
        except Exception as e:
            print(f"[WARN] efetch retstart={cursor}: {str(e)[:80]} (refreshing WebEnv and retrying)", flush=True)
            try:
                webenv, qk, total = esearch_history(TERM)
            except Exception:
                pass
            time.sleep(5)
            continue
        arts = [e for e in root.iter() if lt(e) == "article"]
        if not arts:
            empty_streak += 1
            if empty_streak >= 5:  # WebEnv may have expired; refresh
                try:
                    webenv, qk, total = esearch_history(TERM)
                except Exception:
                    pass
                empty_streak = 0
            cursor += BATCH
            cursors[TERM] = cursor
            _save_cursor(cursors)
            continue
        empty_streak = 0
        for art in arts:
            paper, tables = article_records(art)
            if not paper or not tables or paper["arxiv_id"] in existing:
                continue
            _insert(conn, paper, tables)
            existing.add(paper["arxiv_id"])
            n_paper += 1
            n_table += len(tables)
        conn.commit()
        cursor += BATCH
        cursors[TERM] = cursor
        _save_cursor(cursors)
        if n_paper - last_report >= 200:
            last_report = n_paper
            rate = n_paper / max(1e-6, time.time() - t0) * 60
            print(f"[PMC] +{n_paper} papers/{n_table} tables | retstart={cursor}/{total} | {rate:.0f} papers/min", flush=True)
        time.sleep(REQ_DELAY)

    tot_pmc_p = conn.execute("SELECT COUNT(*) FROM raw_papers WHERE source='pmc'").fetchone()[0]
    tot_pmc_t = conn.execute("SELECT COUNT(*) FROM paper_tables WHERE source='pmc'").fetchone()[0]
    print(f"\n[PMC] this run added {n_paper} papers / {n_table} tables (took {(time.time()-t0)/60:.1f} min)", flush=True)
    print(f"[PMC] DB totals: PMC papers {tot_pmc_p} | PMC tables {tot_pmc_t}", flush=True)
    conn.close()


if __name__ == "__main__":
    main()
