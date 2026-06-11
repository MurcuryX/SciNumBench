#!/bin/bash
cd ~/Backup/paper/ICDE26/data
export NCBI_API_KEY=$(cat ../.ncbi_key)
export PMC_PROXY_HOSTPORT=127.0.0.1:9200 PMC_REQ_DELAY=0.12 PMC_TARGET_ARTICLES=40000
PY=~/Backup/miniforge3/envs/scinum-crawl/bin/python
CAP=40000
while true; do
  tot=$($PY -c "import sqlite3;print(sqlite3.connect(\"arxiv_data.db\").execute(\"SELECT COUNT(*) FROM raw_papers WHERE source=\x27pmc\x27\").fetchone()[0])")
  if [ "$tot" -ge "$CAP" ]; then echo "[wrap] reached $tot >= $CAP, stopping"; break; fi
  echo "[wrap] starting crawler (current $tot)"
  $PY -u ../code/pmc_fetcher.py
  echo "[wrap] crawler exited, resuming in 3s"
  sleep 3
done
