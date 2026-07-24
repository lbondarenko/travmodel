"""Fetch & parse expert rankings (Gratistravtips ABCD) per round.

Public archive, predictable URLs: gratistravtips.se/{track-slug}/{date}/{game}.
Parsed into {leg: [{nr, name, rank, pts}, ...]} with legs in card order.
Cached as JSON per game id. Used for the expert audit and the EXPERTS
drawer on race pages.
"""
import gzip
import json
import re
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "experts_hist"
HEADERS = {"User-Agent": "Mozilla/5.0 (travmodel hobby project)"}

SLUG_MAP = str.maketrans({"å": "a", "ä": "a", "ö": "o", "é": "e", " ": "-"})


def slugify(track):
    return track.lower().translate(SLUG_MAP)


def fetch_gt(track, date, gtype):
    url = f"https://www.gratistravtips.se/{slugify(track)}/{date}/{gtype.lower()}"
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=30) as r:
            h = r.read().decode("utf-8", "replace")
    except Exception:
        return None
    rows = re.findall(
        r'ay-td-name">\s*<strong>\s*(\d+)&nbsp;.*?title="([^"]+)".*?ay-rank-([abcd])".*?\((\d+)\)',
        h, re.S)
    if not rows:
        return None
    legs, cur, prev = [], [], 0
    for nr, name, rank, pts in rows:
        nr = int(nr)
        if nr <= prev and cur:
            legs.append(cur)
            cur = []
        cur.append({"nr": nr, "name": name.title(), "rank": rank.upper(), "pts": int(pts)})
        prev = nr
    if cur:
        legs.append(cur)
    return {"source": "Gratistravtips ABCD", "url": url,
            "legs": {str(i): leg for i, leg in enumerate(legs, 1)}}


def get_cached(gid, track, date, gtype, refresh=False):
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{gid}.json"
    if f.exists() and not refresh:
        return json.loads(f.read_text())
    data = fetch_gt(track, date, gtype)
    if data:
        f.write_text(json.dumps(data, ensure_ascii=False))
    return data


def crawl_games_hist():
    """Backfill expert data for every game in data/games_hist."""
    got = miss = 0
    for f in sorted((ROOT / "data" / "games_hist").glob("*.json.gz")):
        gid = f.stem.replace(".json", "")
        if (CACHE / f"{gid}.json").exists():
            got += 1
            continue
        game = json.load(gzip.open(f, "rt"))
        track = game["races"][0]["track"]["name"] if game.get("races") else None
        date = gid.split("_")[1]
        gtype = gid.split("_")[0]
        if not track:
            miss += 1
            continue
        data = get_cached(gid, track, date, gtype)
        time.sleep(0.4)
        if data:
            got += 1
        else:
            miss += 1
    print(f"crawl done: {got} with expert data, {miss} unavailable", flush=True)


if __name__ == "__main__":
    crawl_games_hist()
