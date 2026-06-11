#!/bin/bash
# Hybrid: metadata (fetcher) connects directly; source-code download (download_scheduler) goes through an SSH tunnel via server0 Tor. Code + DB all reside on server1.
cd "$HOME/Backup/paper/ICDE26/data" || exit 1
export PYTHONPATH="$HOME/Backup/paper/ICDE26/code"
export PYTHONUNBUFFERED=1
export SCINUM_SOCKS_PORTS="9050,9051,9052,9053,9054,9055,9056,9057"
export SCINUM_FETCH_DIRECT=1
export SCINUM_DL_RETRIES="4"
echo y | "$HOME/Backup/miniforge3/envs/scinum-crawl/bin/python" -u \
    "$HOME/Backup/paper/ICDE26/code/auto_replenish.py"
