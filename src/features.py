"""Build a per-start feature table from scraped ATG races. (v2)

Every row = one horse in one race. Label = won (finishOrder == 1).
Point-in-time safety as in v1: form comes from the past-start records
embedded in each race payload, filtered to dates strictly before the
race date; driver/trainer aggregates use the previous calendar year.

v2 additions:
  - structured gallop/disqualification flags from records (archive strips
    the TR Media comments the v1 regex needed — flags are always there)
  - class features from past races' firstPrize (fixes the earnings trap)
  - track-adjusted speed figures: best recent km-time relative to the
    median winning time at that track/start-method/distance bucket
  - past-odds signal: how other markets priced the horse recently
  - foreign-form share + thin-footprint markers for the market-trust
    interaction (weights learned in training, never assumed)

Two passes: pass 1 builds the track baselines, pass 2 the feature rows.

Usage: python src/features.py [rawdir=data/raw2] [out=data/features.parquet]
"""
import gzip
import json
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def km_time_seconds(kt):
    if not isinstance(kt, dict) or kt.get("seconds") is None:
        return None
    return kt.get("minutes", 1) * 60 + kt["seconds"] + kt.get("tenths", 0) / 10


def dist_bucket(m):
    if not m:
        return "mid"
    return "short" if m <= 1900 else ("mid" if m <= 2400 else "long")


def parse_place(p):
    try:
        v = int(p)
        return v if v > 0 else 15
    except (TypeError, ValueError):
        return None


def iter_races(rawdir):
    for fp in sorted(rawdir.glob("*.jsonl.gz")):
        with gzip.open(fp, "rt") as fh:
            for line in fh:
                race = json.loads(line)
                if race.get("status") == "results":
                    yield race


def build_baselines(rawdir):
    """Median winning km-time per (track, startMethod, distance bucket).

    Sources: winners in scraped races AND place-1 past-start records, so
    tracks outside the scrape window (small Norwegian ovals) get covered.
    """
    times = defaultdict(list)
    for race in iter_races(rawdir):
        method = race.get("startMethod")
        for s in race["starts"]:
            res = s.get("result") or {}
            if res.get("finishOrder") == 1:
                t = km_time_seconds(res.get("kmTime"))
                if t:
                    times[(race["track"]["name"], method,
                           dist_bucket(s.get("distance") or race.get("distance")))].append(t)
            for r in (s["horse"].get("results", {}) or {}).get("records", []):
                if r.get("place") == "1":
                    t = km_time_seconds(r.get("kmTime"))
                    if t:
                        times[(r.get("track", {}).get("name"),
                               (r.get("race") or {}).get("startMethod"),
                               dist_bucket((r.get("start") or {}).get("distance")))].append(t)
    baselines = {k: float(np.median(v)) for k, v in times.items() if len(v) >= 5}
    fallback = defaultdict(list)
    for (track, method, bucket), med in baselines.items():
        fallback[(method, bucket)].append(med)
    fallback = {k: float(np.median(v)) for k, v in fallback.items()}
    return baselines, fallback


def start_features(race, start, baselines=None, fallback=None):
    h = start["horse"]
    drv = start.get("driver", {})
    race_date = date.fromisoformat(race["date"])
    prev_year = str(race_date.year - 1)

    f = {
        "race_id": race["id"],
        "date": race["date"],
        "track": race["track"]["name"],
        "country": race["track"].get("countryCode", "SE"),
        "start_number": start["number"],
        "horse": h.get("name"),
        "won": 1 if (start.get("result") or {}).get("finishOrder") == 1 else 0,
        "final_odds": (start.get("result") or {}).get("finalOdds"),
        "distance": start.get("distance") or race.get("distance"),
        "handicap": (start.get("distance") or race.get("distance", 0)) - race.get("distance", 0),
        "auto_start": 1 if race.get("startMethod") == "auto" else 0,
        "post": start.get("postPosition"),
        "field_size": len(race["starts"]),
        "sprint": 1 if (race.get("distance") or 2140) <= 1700 else 0,
        "age": h.get("age"),
        "is_mare": 1 if h.get("sex") == "mare" else 0,
        "money_per_start": None,
        "shoe_front": 1 if (h.get("shoes") or {}).get("front", {}).get("hasShoe") else 0,
        "shoe_back": 1 if (h.get("shoes") or {}).get("back", {}).get("hasShoe") else 0,
        "shoe_change": 1 if ((h.get("shoes") or {}).get("front", {}).get("changed")
                             or (h.get("shoes") or {}).get("back", {}).get("changed")) else 0,
        "sulky_change": 1 if (h.get("sulky") or {}).get("changed") else 0,
    }

    dstats = (drv.get("statistics") or {}).get("years", {}).get(prev_year, {})
    f["drv_starts_py"] = dstats.get("starts", 0)
    f["drv_winpct_py"] = (dstats.get("winPercentage") or 0) / 10000
    f["drv_stats_missing"] = 1 if not dstats else 0
    tstats = ((h.get("trainer") or {}).get("statistics") or {}).get("years", {}).get(prev_year, {})
    f["trn_winpct_py"] = (tstats.get("winPercentage") or 0) / 10000

    stats_life = (h.get("statistics") or {}).get("life", {})
    money = (h.get("money") or stats_life.get("earnings") or 0)
    starts_life = stats_life.get("starts") or 0
    if starts_life:
        f["money_per_start"] = money / starts_life

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
        places = [p for p in (parse_place(r.get("place")) for r in recs) if p is not None]
        f["avg_place_5"] = float(np.mean(places)) if places else None
        f["wins_5"] = sum(1 for r in recs if parse_place(r.get("place")) == 1)
        f["top3_5"] = sum(1 for r in recs if (parse_place(r.get("place")) or 15) <= 3)
        # v2: structured flags, not comment regex
        f["gallops_5"] = sum(1 for r in recs if r.get("galloped"))
        f["disq_5"] = sum(1 for r in recs if r.get("disqualified"))
        # v2: foreign form share
        f["foreign_share"] = float(np.mean(
            [1 if r.get("track", {}).get("countryCode") not in ("SE", None) else 0 for r in recs]))
        # v2: class from purses (firstPrize is in öre*100 units; log absorbs scale)
        purses = [(r.get("race") or {}).get("firstPrize") for r in recs]
        purses = [p for p in purses if p]
        f["log_avg_purse_5"] = float(np.mean(np.log1p(purses))) if purses else None
        # v2: what other markets thought of this horse recently
        odds5 = [(r.get("odds") or 0) / 100 for r in recs]
        odds5 = [o for o in odds5 if o > 1]
        f["avg_log_odds_5"] = float(np.mean(np.log(odds5))) if odds5 else None
        f["last_odds"] = odds5[0] if odds5 else None
        # v2: track-adjusted speed figure (negative = faster than typical winner)
        figs = []
        if baselines is not None:
            for r in recs:
                t = km_time_seconds(r.get("kmTime"))
                if not t:
                    continue
                key = (r.get("track", {}).get("name"),
                       (r.get("race") or {}).get("startMethod"),
                       dist_bucket((r.get("start") or {}).get("distance")))
                base = baselines.get(key) or fallback.get(key[1:])
                if base:
                    figs.append(t - base)
        f["best_speedfig_5"] = min(figs) if figs else None
    return f


def main():
    rawdir = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "data/raw2")
    out = ROOT / (sys.argv[2] if len(sys.argv) > 2 else "data/features.parquet")
    print("pass 1: track baselines ...")
    baselines, fallback = build_baselines(rawdir)
    print(f"  {len(baselines)} (track, method, bucket) baselines")
    (ROOT / "data" / "baselines.json").write_text(json.dumps(
        {"|".join(str(x) for x in k): v for k, v in baselines.items()}
        | {"__fb__" + "|".join(str(x) for x in k): v for k, v in fallback.items()}))
    print("pass 2: features ...")
    rows = []
    for race in iter_races(rawdir):
        for s in race["starts"]:
            res = s.get("result") or {}
            if res.get("finishOrder") is None and res.get("place") is None:
                continue
            rows.append(start_features(race, s, baselines, fallback))
    df = pd.DataFrame(rows)
    df["implied"] = 1.0 / df["final_odds"].where(df["final_odds"] > 1.0)
    df["implied"] = df["implied"].fillna(df.groupby("race_id")["implied"].transform("mean"))
    df["implied_norm"] = df["implied"] / df.groupby("race_id")["implied"].transform("sum")
    df.to_parquet(out)
    print(f"{len(df)} starts, {df['race_id'].nunique()} races, "
          f"win rate {df['won'].mean():.3f} -> {out}")


if __name__ == "__main__":
    main()
