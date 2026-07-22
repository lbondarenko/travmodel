"""Scrape historical Swedish trot races from ATG's public racinginfo API.

Writes one gzipped JSONL file per day to data/raw/<date>.jsonl.gz
(one line per race, the full /extended payload). Resumable: existing
day files are skipped. Days are processed most-recent-first so a
partial run still yields the freshest training data.

Usage: python src/scrape.py 2026-01-01 2026-07-21 [countries_csv] [outdir]
       default countries SE, default outdir data/raw
       v2: python src/scrape.py 2025-01-01 2026-07-22 SE,NO,DK,FI data/raw2
"""
import gzip
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

BASE = "https://www.atg.se/services/racinginfo/v1/api"
RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
COUNTRIES = {"SE"}
HEADERS = {"User-Agent": "travmodel-hobby-project/0.1 (personal research)"}


def get(url, retries=3):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except Exception as e:
            if attempt == retries - 1:
                print(f"  FAIL {url}: {e}", flush=True)
                return None
            time.sleep(2 * (attempt + 1))


def fetch_race(rid):
    time.sleep(0.1)  # stay polite: ~3 threads * this sleep ≈ <10 req/s
    return get(f"{BASE}/races/{rid}/extended")


def scrape_day(d):
    out = RAW / f"{d}.jsonl.gz"
    if out.exists():
        return "skip"
    cal = get(f"{BASE}/calendar/day/{d}")
    if cal is None:
        return "cal-fail"
    rids = []
    for t in cal.get("tracks", []):
        if t.get("countryCode") not in COUNTRIES or t.get("sport") not in (None, "trot"):
            continue
        for r in t.get("races", []):
            rids.append(r["id"])
    races = []
    with ThreadPoolExecutor(max_workers=3) as ex:
        for payload in ex.map(fetch_race, rids):
            if payload and payload.get("status") == "results" and payload.get("sport") == "trot":
                races.append(payload)
    tmp = out.with_suffix(".tmp")
    with gzip.open(tmp, "wt") as f:
        for r in races:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.rename(out)
    return f"{len(races)} races"


def main():
    global RAW, COUNTRIES
    start = date.fromisoformat(sys.argv[1])
    end = date.fromisoformat(sys.argv[2])
    if len(sys.argv) > 3:
        COUNTRIES = set(sys.argv[3].split(","))
    if len(sys.argv) > 4:
        RAW = Path(__file__).resolve().parent.parent / sys.argv[4]
    RAW.mkdir(parents=True, exist_ok=True)
    days = []
    d = end
    while d >= start:
        days.append(d)
        d -= timedelta(days=1)
    t0 = time.time()
    for i, d in enumerate(days, 1):
        status = scrape_day(d)
        if status != "skip":
            print(f"[{i}/{len(days)}] {d}: {status} ({time.time()-t0:.0f}s)", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
