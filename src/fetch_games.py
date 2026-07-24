"""Fetch historical GAME payloads (results + pool dividends) for backtesting
ticket allocators. Saves one json.gz per game to data/games_hist/.

Usage: python src/fetch_games.py 2026-01-01 2026-07-22
"""
import gzip
import json
import sys
import time
import urllib.request
from datetime import date, timedelta
from pathlib import Path

BASE = "https://www.atg.se/services/racinginfo/v1/api"
OUT = Path(__file__).resolve().parent.parent / "data" / "games_hist"
TYPES = ["V64", "V86", "V75", "GS75", "V65"]
HEADERS = {"User-Agent": "travmodel-hobby/0.1"}


def get(url, retries=3):
    for a in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=30) as r:
                return json.load(r)
        except Exception:
            if a == retries - 1:
                return None
            time.sleep(2)


def main():
    start = date.fromisoformat(sys.argv[1])
    end = date.fromisoformat(sys.argv[2])
    OUT.mkdir(parents=True, exist_ok=True)
    d = start
    n = 0
    while d <= end:
        cal = get(f"{BASE}/calendar/day/{d}")
        if cal:
            trot = {str(t["id"]) for t in cal.get("tracks", [])
                    if t.get("sport") in (None, "trot")}
            for gt, lst in (cal.get("games") or {}).items():
                if gt not in TYPES:
                    continue
                for g in lst:
                    gid = g["id"]
                    parts = gid.split("_")
                    if len(parts) > 2 and parts[2] not in trot:
                        continue
                    f = OUT / f"{gid}.json.gz"
                    if f.exists():
                        continue
                    payload = get(f"{BASE}/games/{gid}")
                    time.sleep(0.1)
                    if payload and payload.get("status") == "results":
                        with gzip.open(f, "wt") as fh:
                            json.dump(payload, fh, ensure_ascii=False)
                        n += 1
        d += timedelta(days=1)
        if d.day == 1:
            print(f"{d}: {n} games so far", flush=True)
    print(f"DONE: {n} games", flush=True)


if __name__ == "__main__":
    main()
