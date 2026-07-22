"""Score a live (bettable) ATG game with the trained blended model.

Fetches the game, pulls each leg's /extended race payload for point-in-time
form, uses live vinnare odds as the market feature, and prints per-leg
win probabilities next to the market's view, flagging value.

Usage: python src/predict.py V86_2026-07-22_25_3
"""
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import start_features  # noqa: E402
from train import prepare  # noqa: E402

BASE = "https://www.atg.se/services/racinginfo/v1/api"
ROOT = Path(__file__).resolve().parent.parent


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "travmodel/0.1"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def load_baselines():
    raw = json.loads((ROOT / "data" / "baselines.json").read_text())
    baselines, fallback = {}, {}
    for k, v in raw.items():
        if k.startswith("__fb__"):
            m, b = k[6:].split("|")
            fallback[(m if m != "None" else None, b)] = v
        else:
            t, m, b = k.split("|")
            baselines[(t, m if m != "None" else None, b)] = v
    return baselines, fallback


def main():
    game_id = sys.argv[1]
    model = json.loads((ROOT / "data" / "model.json").read_text())
    baselines, fallback = load_baselines()
    game = get(f"{BASE}/games/{game_id}")

    all_rows = []
    for leg_no, race in enumerate(game["races"], 1):
        ext = get(f"{BASE}/races/{race['id']}/extended")
        # live market odds from the game payload (vinnare pool, odds x100)
        odds_by_num, dist_by_num = {}, {}
        for s in race["starts"]:
            pools = s.get("pools") or {}
            vo = (pools.get("vinnare") or {}).get("odds")
            if vo and vo > 100:
                odds_by_num[s["number"]] = vo / 100
            bd = (pools.get(game_id.split("_")[0]) or {}).get("betDistribution")
            if bd:
                dist_by_num[s["number"]] = bd / 10000
        scratched = {s["number"] for s in race["starts"]
                     if (s.get("scratched") or (s.get("horse") or {}).get("scratched"))}
        for s in ext["starts"]:
            if s["number"] in scratched:
                continue
            f = start_features(ext, s, baselines, fallback)
            f["leg"] = leg_no
            f["live_odds"] = odds_by_num.get(s["number"])
            f["bet_dist"] = dist_by_num.get(s["number"])
            all_rows.append(f)

    df = pd.DataFrame(all_rows)
    # market implied prob from live win odds; fall back to pool distribution
    df["implied"] = 1.0 / df["live_odds"]
    df["implied"] = df["implied"].fillna(df["bet_dist"])
    df["implied"] = df["implied"].fillna(df.groupby("leg")["implied"].transform("mean"))
    df["implied_norm"] = df["implied"] / df.groupby("leg")["implied"].transform("sum")

    df = prepare(df)
    cols = model["features"]
    mu = pd.Series(model["mu"])[cols]
    sd = pd.Series(model["sd"])[cols]
    beta = pd.Series(model["beta"])[cols]
    X = ((df[cols] - mu) / sd).values
    df["u"] = X @ beta.values
    df["u"] = df["u"] - df.groupby("leg")["u"].transform("max")
    df["e"] = np.exp(df["u"])
    df["p_model"] = df["e"] / df.groupby("leg")["e"].transform("sum")

    for leg in sorted(df["leg"].unique()):
        d = df[df["leg"] == leg].sort_values("p_model", ascending=False)
        print(f"\n=== Leg {leg} ===")
        print(f"{'nr':>3} {'horse':<24} {'model%':>7} {'market%':>8}  note")
        for _, r in d.iterrows():
            edge = r["p_model"] / r["implied_norm"] if r["implied_norm"] else 1
            note = "VALUE" if edge > 1.3 and r["p_model"] > 0.08 else (
                "overbet" if edge < 0.7 and r["implied_norm"] > 0.15 else "")
            print(f"{r['start_number']:>3} {r['horse']:<24} "
                  f"{100*r['p_model']:>6.1f}% {100*r['implied_norm']:>7.1f}%  {note}")
        top = d.iloc[0]
        tag = "SPIK candidate" if top["p_model"] > 0.45 else "open leg"
        print(f"  -> A-pick: {top['start_number']} {top['horse']} "
              f"({100*top['p_model']:.0f}%) [{tag}]")


if __name__ == "__main__":
    main()
