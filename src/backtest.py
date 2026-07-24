"""Backtest ticket allocators on historical rounds with real dividends.

Variants:
  A  current greedy (max win-probability per krona)         — v2, live today
  B  A + spik floor: a leg may stay single only if its top   — "weak banker" fix
     horse has >= 45% model chance; otherwise it must carry
     two horses before anything else is bought
  C  EV-greedy: each krona buys expected *payout*, weighting — "Jan's jackpot point"
     each horse by model chance x (model chance / streck)

Scoring uses each round's actual pool dividends (tiers that rolled to
jackpot pay 0, exactly as in reality). Caveat: model probabilities use
final odds as the market input (the only archived market), which flatters
all variants equally — comparisons stay fair, absolute ROI is optimistic.

Usage: python src/backtest.py
"""
import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import prepare  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
GAMES = ROOT / "data" / "games_hist"
ROW_PRICE = {"V64": 1.0, "V65": 1.0, "V75": 0.5, "V86": 0.25, "GS75": 0.25}
BUDGET = 50.0
MAX_PER_LEG = 5


def model_probs():
    model = json.loads((ROOT / "data" / "model.json").read_text())
    df = pd.read_parquet(ROOT / "data" / "features.parquet")
    df = prepare(df)
    cols = model["features"]
    mu = pd.Series(model["mu"])[cols]
    sd = pd.Series(model["sd"])[cols]
    beta = pd.Series(model["beta"])[cols]
    X = ((df[cols] - mu) / sd).values
    with np.errstate(all="ignore"):
        df["u"] = X @ beta.values
    df["u"] = df["u"] - df.groupby("race_id")["u"].transform("max")
    df["e"] = np.exp(df["u"])
    df["p"] = df["e"] / df.groupby("race_id")["e"].transform("sum")
    return {(r, int(n)): float(p) for r, n, p in zip(df["race_id"], df["start_number"], df["p"])}


def greedy(legs, price, floor=None, ev=False):
    """legs: list of lists of (nr, p, streck) sorted by p desc. Returns picks/leg."""
    sel = [1] * len(legs)
    if floor is not None:
        for i, L in enumerate(legs):
            if L[0][1] < floor and len(L) > 1:
                sel[i] = 2

    def rows():
        r = 1
        for k in sel:
            r *= k
        return r

    def unit(h):
        nr, p, st = h
        if not ev:
            return p
        vr = min(p / max(st, 0.005), 5.0)
        return p * vr

    def cov(i):
        return sum(unit(h) for h in legs[i][:sel[i]])

    if rows() * price > BUDGET:   # forced floor blew the budget: fall back
        sel = [1] * len(legs)
    while True:
        best, best_eff = None, 0.0
        base = rows()
        for i, L in enumerate(legs):
            k = sel[i]
            if k >= min(MAX_PER_LEG, len(L)):
                continue
            new_cost = price * base * (k + 1) / k
            if new_cost > BUDGET:
                continue
            gain = math.log((cov(i) + unit(L[k])) / cov(i)) if cov(i) > 0 else 1.0
            eff = gain / (new_cost - price * base)
            if eff > best_eff:
                best, best_eff = i, eff
        if best is None:
            break
        sel[best] += 1
    return [[h[0] for h in legs[i][:sel[i]]] for i in range(len(legs))]


def score(picks, winners, payouts, price):
    ways = {0: 1}
    for pk, w in zip(picks, winners):
        wc = 1 if w in pk else 0
        ww = len(pk) - wc
        new = {}
        for k, v in ways.items():
            if wc:
                new[k + 1] = new.get(k + 1, 0) + v * wc
            if ww:
                new[k] = new.get(k, 0) + v * ww
        ways = new
    rows = sum(v for v in ways.values())
    ret = 0.0
    for tier, info in (payouts or {}).items():
        pay = (info.get("payout", 0) if isinstance(info, dict) else 0) / 100
        ret += ways.get(int(tier), 0) * pay
    hits = sum(1 for pk, w in zip(picks, winners) if w in pk)
    allhit = 1 if hits == len(picks) else 0
    return rows * price, ret, hits, allhit


def main():
    probs = model_probs()
    stats = {v: dict(stake=0.0, ret=0.0, hits=0, allhit=0, n=0, legs=0)
             for v in ("A", "B", "C")}
    skipped = 0
    for f in sorted(GAMES.glob("*.json.gz")):
        game = json.load(gzip.open(f, "rt"))
        gid = f.stem.replace(".json", "")
        gtype = gid.split("_")[0]
        price = ROW_PRICE.get(gtype, 1.0)
        legs, winners = [], []
        ok = True
        for race in game.get("races", []):
            rid = race["id"]
            L, w = [], None
            for s in race["starts"]:
                res = s.get("result") or {}
                if res.get("finishOrder") == 1:
                    w = int(s["number"])
                p = probs.get((rid, int(s["number"])))
                bd = ((s.get("pools") or {}).get(gtype) or {}).get("betDistribution")
                if p is not None:
                    L.append((int(s["number"]), p, (bd or 100) / 10000))
            if w is None or len(L) < 4:
                ok = False
                break
            L.sort(key=lambda x: -x[1])
            legs.append(L)
            winners.append(w)
        if not ok or not legs:
            skipped += 1
            continue
        payouts = ((game.get("pools") or {}).get(gtype) or {}).get("result", {}).get("payouts")
        for v, kw in (("A", {}), ("B", {"floor": 0.45}), ("C", {"ev": True})):
            picks = greedy(legs, price, **kw)
            stake, ret, hits, allhit = score(picks, winners, payouts, price)
            st = stats[v]
            st["stake"] += stake
            st["ret"] += ret
            st["hits"] += hits
            st["legs"] += len(legs)
            st["allhit"] += allhit
            st["n"] += 1

    print(f"rounds tested: {stats['A']['n']}  (skipped {skipped})\n")
    print(f"{'variant':<28}{'stake':>9}{'return':>10}{'ROI':>8}{'all-hit':>9}{'legs hit':>10}")
    names = {"A": "A current (probability)", "B": "B + spik floor 45%",
             "C": "C EV / jackpot-weighted"}
    for v in ("A", "B", "C"):
        st = stats[v]
        roi = st["ret"] / st["stake"] - 1 if st["stake"] else 0
        print(f"{names[v]:<28}{st['stake']:>9.0f}{st['ret']:>10.0f}{roi:>8.1%}"
              f"{st['allhit']:>9}{st['hits']/st['legs']:>10.1%}")


if __name__ == "__main__":
    main()
