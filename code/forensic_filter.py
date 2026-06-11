import sqlite3, json, re
DB="/home/pengminjie/Backup/paper/ICDE26/data/arxiv_data.db"
c=sqlite3.connect(DB)

# Add columns (idempotent)
cols=[r[1] for r in c.execute("pragma table_info(paper_tables)")]
for col,decl in [("forensic_usable","INTEGER DEFAULT 0"),("forensic_tags","TEXT")]:
    if col not in cols:
        c.execute(f"ALTER TABLE paper_tables ADD COLUMN {col} {decl}")
c.commit()

RE_MEANSD=re.compile(r"-?\d+\.?\d*\s*(?:±|\\pm)\s*\d+\.?\d*")
RE_IQR=re.compile(r"-?\d+\.?\d*\s*[\(\[]\s*-?\d+\.?\d*\s*[,;\-–]\s*-?\d+\.?\d*\s*[\)\]]")
RE_PCT=re.compile(r"\d+\s*[/(]\s*\d+.*?%|\d+\.?\d*\s*%")
RE_NUM=re.compile(r"-?\d+\.?\d*"); RE_N=re.compile(r"[\(\s/,]([nN])\s*=\s*\d+")
RE_CI=re.compile(r"95\s*%\s*CI|\bCI\b")
RE_PVAL=re.compile(r"[pP]\s*[=<>]\s*0?\.\d|\bp[-\s]?value")

def tags(cap,tj):
    d=json.loads(tj); colh=[str(x) for x in d.get("columns",[])]; data=d.get("data",[])
    flat=["" if x is None else str(x) for r in data for x in r]; n=max(1,len(flat))
    ms=sum(1 for x in flat if RE_MEANSD.search(x))
    iq=sum(1 for x in flat if RE_IQR.search(x) and not RE_MEANSD.search(x))
    pc=sum(1 for x in flat if RE_PCT.search(x))
    nm=sum(1 for x in flat if RE_NUM.search(x)); nf=nm/n
    head=" ".join(colh)+" "+(cap or "")
    has_n=bool(RE_N.search(head)) or bool(RE_N.search(" ".join(flat[:40])))
    has_ci=bool(RE_CI.search(tj)); has_p=bool(RE_PVAL.search(tj))
    stat=ms+iq+pc
    usable = 1 if (stat>=3 and nf>=0.35) else 0
    t=[]
    if not usable: return 0,""
    if ms>=1 and has_n: t.append("grim")
    if ms>=3 and has_n and has_p: t.append("carlisle")
    if has_p: t.append("pval")
    if has_ci: t.append("ci")
    if iq>=1: t.append("iqr")
    if pc>=1: t.append("countpct")
    if nf>=0.5: t.append("benford")
    return 1,",".join(t)

rows=c.execute("SELECT id,caption,table_json FROM paper_tables WHERE source='pmc'").fetchall()
upd=[]
for rid,cap,tj in rows:
    try: u,t=tags(cap,tj)
    except Exception: u,t=0,""
    upd.append((u,t,rid))
c.executemany("UPDATE paper_tables SET forensic_usable=?, forensic_tags=? WHERE id=?",upd)
c.commit()

# Verify
print("marked pmc rows:",len(upd))
print("forensic_usable=1:",c.execute("SELECT COUNT(*) FROM paper_tables WHERE source='pmc' AND forensic_usable=1").fetchone()[0])
print("\ntier counts:")
from collections import Counter
cnt=Counter()
for (t,) in c.execute("SELECT forensic_tags FROM paper_tables WHERE source='pmc' AND forensic_usable=1"):
    for x in (t or "").split(","):
        if x: cnt[x]+=1
for k in ["benford","countpct","iqr","grim","ci","pval","carlisle"]:
    print("  %-10s: %d"%(k,cnt[k]))
