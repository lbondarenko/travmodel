"""Train a conditional logit (softmax-per-race) win model. (v2)

P(horse i wins race r) = exp(x_i . beta) / sum_j exp(x_j . beta)

Variants compared:
  fundamentals : form/driver/equipment/class/speed features only
  blended      : fundamentals + ln(market implied prob)
  blended+trust: blended + ln(market) x thin-footprint interaction —
                 lets the model lean on the market less for horses the
                 Swedish market knows little about (foreign form, unknown
                 driver). Kept only if it wins on holdout.

Evaluation on a chronological holdout: log loss per race vs the market,
top-pick hit rate, and a calibration report on the foreign-form segment.

Usage: python src/train.py [test_from=2026-05-01]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent.parent

FUNDAMENTALS = [
    "handicap", "auto_start", "post_x_auto", "post_x_sprint", "age", "is_mare",
    "log_money_per_start", "shoe_front", "shoe_back", "shoe_change", "sulky_change",
    "drv_winpct_py", "log_drv_starts", "trn_winpct_py",
    "days_since_c", "long_layoff", "last_place_c", "avg_place_5_c",
    "wins_5", "top3_5", "gallops_5", "disq_5", "n_recent",
    # v2
    "foreign_share", "drv_stats_missing", "log_avg_purse_c",
    "avg_log_odds_c", "best_speedfig_c",
]
MARKET = ["log_implied"]
TRUST = ["implied_x_thin"]


def prepare(df):
    df = df.copy()
    df["post_x_auto"] = df["post"].fillna(6) * df["auto_start"]
    df["post_x_sprint"] = df["post"].fillna(6) * df["sprint"]
    df["log_money_per_start"] = np.log1p(df["money_per_start"].fillna(0))
    df["log_drv_starts"] = np.log1p(df["drv_starts_py"].fillna(0))
    df["days_since_c"] = df["days_since"].fillna(60).clip(0, 180)
    df["long_layoff"] = (df["days_since_c"] > 45).astype(int)
    df["last_place_c"] = df["last_place"].fillna(8).clip(1, 15)
    df["avg_place_5_c"] = df["avg_place_5"].fillna(8).clip(1, 15)
    for c in ["wins_5", "top3_5", "gallops_5", "disq_5", "n_recent",
              "foreign_share", "drv_stats_missing"]:
        df[c] = df.get(c, pd.Series(0, index=df.index)).fillna(0)
    df["log_avg_purse_c"] = df["log_avg_purse_5"].fillna(df["log_avg_purse_5"].median())
    df["avg_log_odds_c"] = df["avg_log_odds_5"].fillna(np.log(12))
    df["best_speedfig_c"] = df["best_speedfig_5"].fillna(0).clip(-8, 8)
    df["log_implied"] = np.log(df["implied_norm"].clip(1e-4, 1))
    df["thin_foot"] = ((df["foreign_share"] >= 0.5) | (df["drv_stats_missing"] == 1)).astype(int)
    df["implied_x_thin"] = df["log_implied"] * df["thin_foot"]
    df["age"] = df["age"].fillna(df["age"].median())
    return df


def fit_clogit(X, race_idx, won, l2=1e-3):
    n_races = race_idx.max() + 1
    order = np.argsort(race_idx, kind="stable")
    X, race_idx, won = X[order], race_idx[order], won[order]
    race_starts = np.searchsorted(race_idx, np.arange(n_races))

    def nll_grad(beta):
        with np.errstate(all="ignore"):
            u = X @ beta
            u = u - np.maximum.reduceat(u, race_starts)[race_idx]
            e = np.exp(u)
            denom = np.add.reduceat(e, race_starts)
            p = e / denom[race_idx]
            ll = np.sum(np.log(np.clip(p[won == 1], 1e-300, None)))
            grad = X.T @ (p - won)
            val = -(ll) + l2 * beta @ beta
        if not np.isfinite(val) or not np.all(np.isfinite(grad)):
            return 1e12, np.zeros_like(beta)
        return val, grad + 2 * l2 * beta

    res = minimize(nll_grad, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B",
                   options={"maxiter": 800})
    return res.x


def predict_clogit(beta, X, race_ids):
    df = pd.DataFrame({"rid": race_ids, "u": X @ beta})
    df["u"] = df["u"] - df.groupby("rid")["u"].transform("max")
    df["e"] = np.exp(df["u"])
    return (df["e"] / df.groupby("rid")["e"].transform("sum")).values


def race_logloss(p, won, race_ids):
    d = pd.DataFrame({"p": p, "won": won, "rid": race_ids})
    return -np.log(d[d["won"] == 1]["p"].clip(1e-6)).mean()


def top_pick_hits(p, won, race_ids):
    d = pd.DataFrame({"p": p, "won": won, "rid": race_ids})
    return d.loc[d.groupby("rid")["p"].idxmax()]["won"].mean()


def main():
    test_from = sys.argv[1] if len(sys.argv) > 1 else "2026-05-01"
    df = pd.read_parquet(ROOT / "data" / "features.parquet")
    df = prepare(df)
    ok = df.groupby("race_id")["won"].transform("sum") == 1
    df = df[ok & df["final_odds"].notna()].reset_index(drop=True)

    train = df[df["date"] < test_from]
    test = df[df["date"] >= test_from]
    print(f"train: {train['race_id'].nunique()} races / {len(train)} starts   "
          f"test: {test['race_id'].nunique()} races / {len(test)} starts")

    variants = [("fundamentals", FUNDAMENTALS),
                ("blended", FUNDAMENTALS + MARKET),
                ("blended+trust", FUNDAMENTALS + MARKET + TRUST)]
    results = {}
    for name, cols in variants:
        mu, sd = train[cols].mean(), train[cols].std().replace(0, 1)
        Xtr = ((train[cols] - mu) / sd).values
        Xte = ((test[cols] - mu) / sd).values
        beta = fit_clogit(Xtr, pd.factorize(train["race_id"])[0],
                          train["won"].values.astype(float))
        p = predict_clogit(beta, Xte, test["race_id"].values)
        results[name] = {"cols": cols, "mu": mu, "sd": sd, "beta": beta, "p": p,
                         "logloss": race_logloss(p, test["won"].values, test["race_id"].values),
                         "top_hit": top_pick_hits(p, test["won"].values, test["race_id"].values)}

    market_ll = race_logloss(test["implied_norm"].values, test["won"].values, test["race_id"].values)
    fav_hit = top_pick_hits(test["implied_norm"].values, test["won"].values, test["race_id"].values)
    print(f"\n=== holdout ({test_from}+) ===")
    print(f"market baseline : logloss {market_ll:.4f}  favorite hit {fav_hit:.3f}")
    for name, _ in variants:
        r = results[name]
        print(f"{name:15s} : logloss {r['logloss']:.4f}  top-pick hit {r['top_hit']:.3f}")

    # segment calibration: the Edens Odin population
    best_name = min(("blended", "blended+trust"), key=lambda n: results[n]["logloss"])
    t = test.copy()
    t["p_model"] = results[best_name]["p"]
    seg = t[t["foreign_share"] >= 0.5]
    if len(seg):
        print(f"\nforeign-form segment (n={len(seg)}, {seg['won'].sum():.0f} winners):")
        print(f"  actual win rate {100*seg['won'].mean():.1f}%  "
              f"market says {100*seg['implied_norm'].mean():.1f}%  "
              f"{best_name} says {100*seg['p_model'].mean():.1f}%")
    se = t[t["country"] == "SE"]
    print(f"SE-only holdout logloss: market {race_logloss(se['implied_norm'].values, se['won'].values, se['race_id'].values):.4f}  "
          f"{best_name} {race_logloss(se['p_model'].values, se['won'].values, se['race_id'].values):.4f}")

    r = results[best_name]
    model = {
        "version": 2, "variant": best_name, "test_from": test_from,
        "features": r["cols"],
        "mu": {c: float(r["mu"][c]) for c in r["cols"]},
        "sd": {c: float(r["sd"][c] or 1) for c in r["cols"]},
        "beta": dict(zip(r["cols"], (float(b) for b in r["beta"]))),
        "holdout_logloss": {"market": float(market_ll), best_name: float(r["logloss"])},
    }
    out = ROOT / "data" / "model.json"
    out.write_text(json.dumps(model, indent=2))
    print(f"\nsaved {best_name} model -> {out}")


if __name__ == "__main__":
    main()
