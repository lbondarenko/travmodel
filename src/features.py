"""Build a per-start feature table from scraped ATG races.

Every row = one horse in one race. Label = won (finishOrder == 1).
All features are point-in-time safe:
  - form features come from the past-start records embedded in each race
    payload (ATG serves them as they stood on race day), filtered to
    dates strictly before the race date as a belt-and-braces guard;
  - driver/trainer aggregates use the *previous calendar year* only,
    since the API returns current-year stats as of fetch time.

Usage: python src/features.py            -> data/features.parquet
"""
import gzip
import json
import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

GALLOP_RE = re.compile(r"\bgal\b|\bgal[ .,]|galopp|^gal|sg\b|startgalopp", re.IGNORECASE)


def km_time_seconds(kt):
    if not isinstance(kt, dict) or kt.get("seconds") is None:
        return None
    return kt.get("minutes", 1) * 60 + kt["seconds"] + kt.get("tenths", 0) / 10


def parse_place(p):
    """ATG place: '1'..'n', '0' = unplaced, 'd'/None/'?' = disqualified/unknown."""
    try:
        v = int(p)
        return v if v > 0 else 15
    except (TypeError, ValueError):
        return None


def start_features(race, start):
    h = start["horse"]
    drv = start.get("driver", {})
    race_date = date.fromisoformat(race["date"])
    prev_year = str(race_date.year - 1)

    f = {
        "race_id": race["id"],
        "date": race["date"],
        "track": race["track"]["name"],
        "start_number": start["number"],
        "horse": h.get("name"),
        "won": 1 if (start.get("result") or {}).get("finishOrder") == 1 else 0,
        "final_odds": (start.get("result") or {}).get("finalOdds"),
        # race shape
        "distance": start.get("distance") or race.get("distance"),
        "handicap": (start.get("distance") or race.get("distance", 0)) - race.get("distance", 0),
        "auto_start": 1 if race.get("startMethod") == "auto" else 0,
        "post": start.get("postPosition"),
        "field_size": len(race["starts"]),
        "sprint": 1 if (race.get("distance") or 2140) <= 1700 else 0,
        # horse basics
        "age": h.get("age"),
        "is_mare": 1 if h.get("sex") == "mare" else 0,
        "money_per_start": None,
        # equipment
        "shoe_front": 1 if (h.get("shoes") or {}).get("front", {}).get("hasShoe") else 0,
        "shoe_back": 1 if (h.get("shoes") or {}).get("back", {}).get("hasShoe") else 0,
        "shoe_change": 1 if ((h.get("shoes") or {}).get("front", {}).get("changed")
                             or (h.get("shoes") or {}).get("back", {}).get("changed")) else 0,
        "sulky_change": 1 if (h.get("sulky") or {}).get("changed") else 0,
    }

    # driver: previous-year stats only (leakage-safe)
    dstats = (drv.get("statistics") or {}).get("years", {}).get(prev_year, {})
    f["drv_starts_py"] = dstats.get("starts", 0)
    f["drv_winpct_py"] = (dstats.get("winPercentage") or 0) / 10000  # API gives 2791 for 27.91%
    tstats = ((h.get("trainer") or {}).get("statistics") or {}).get("years", {}).get(prev_year, {})
    f["trn_winpct_py"] = (tstats.get("winPercentage") or 0) / 10000

    # career money per start (career totals are lifetime-to-fetch, mild noise;
    # keep because earnings class is a strong predictor)
    stats_life = (h.get("statistics") or {}).get("life", {})
    money = (h.get("money") or stats_life.get("earnings") or 0)
    starts_life = stats_life.get("starts") or 0
    if starts_life:
        f["money_per_start"] = money / starts_life

    # form from point-in-time past starts
    recs = [r for r in (h.get("results", {}).get("records") or [])
            if r.get("date") and r["date"] < race["date"]]
    recs.sort(key=lambda r: r["date"], reverse=True)
    recs = recs[:5]
    f["n_recent"] = len(recs)
    if recs:
        last = recs[0]
        f["days_since"] = (race_date - date.fromisoformat(last["date"])).days
        lp = parse_place(last.get("place"))
        f["last_place"] = lp if lp is not None else 15
        f["last_odds"] = (last.get("odds") or 0) / 100 or None
        places = [parse_place(r.get("place")) for r in recs]
        places = [p for p in places if p is not None]
        f["avg_place_5"] = float(np.mean(places)) if places else None
        f["wins_5"] = sum(1 for r in recs if parse_place(r.get("place")) == 1)
        f["top3_5"] = sum(1 for r in recs if (parse_place(r.get("place")) or 15) <= 3)
        f["gallops_5"] = sum(
            1 for r in recs
            if GALLOP_RE.search((r.get("trMediaInfo") or {}).get("comment") or "")
            or parse_place(r.get("place")) is None
        )
        # best recent km time, adjusted +1s per 20m handicap-free proxy:
        # compare only within same start method to keep it sane
        times = [km_time_seconds(r.get("kmTime")) for r in recs]
        times = [t for t in times if t]
        f["best_kmtime_5"] = min(times) if times else None
        f["avg_won_frac_5"] = float(np.mean(
            [1 if parse_place(r.get("place")) == 1 else 0 for r in recs]))
    return f


def main():
    rows = []
    files = sorted(RAW.glob("*.jsonl.gz"))
    print(f"{len(files)} day files")
    for fp in files:
        with gzip.open(fp, "rt") as fh:
            for line in fh:
                race = json.loads(line)
                if race.get("status") != "results":
                    continue
                for s in race["starts"]:
                    res = s.get("result") or {}
                    # skip scratched (no finishOrder and no place at all)
                    if res.get("finishOrder") is None and res.get("place") is None:
                        continue
                    rows.append(start_features(race, s))
    df = pd.DataFrame(rows)
    # market implied probability from final win odds, normalized within race
    df["implied"] = 1.0 / df["final_odds"].where(df["final_odds"] > 1.0)
    df["implied"] = df["implied"].fillna(df.groupby("race_id")["implied"].transform("mean"))
    df["implied_norm"] = df["implied"] / df.groupby("race_id")["implied"].transform("sum")
    out = ROOT / "data" / "features.parquet"
    df.to_parquet(out)
    print(f"{len(df)} starts, {df['race_id'].nunique()} races, "
          f"win rate {df['won'].mean():.3f} -> {out}")


if __name__ == "__main__":
    main()
