"""Train a conditional logit (softmax-per-race) win model — the classic
Bolton-Chapman / Benter architecture for race betting.

P(horse i wins race r) = exp(x_i . beta) / sum_j exp(x_j . beta)

Two models are fit and compared:
  fundamentals : form/driver/equipment features only
  blended      : fundamentals + ln(market implied prob)  <- the Benter blend

Evaluation on a chronological holdout:
  - log loss per race vs the market baseline (normalized implied probs)
  - top-pick hit rate vs betting favorite hit rate
  - flat-stake win-bet ROI when model sees >=20% edge over the market

Usage: python src/train.py [test_from=2026-06-15]
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
    "wins_5", "top3_5", "gallops_5", "n_recent", "log_last_odds",
]
MARKET = ["log_implied"]


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
    df["log_last_odds"] = np.log(df["last_odds"].clip(1.01, 200)).fillna(np.log(15))
    for c in ["wins_5", "top3_5", "gallops_5", "n_recent"]:
        df[c] = df[c].fillna(0)
    df["log_implied"] = np.log(df["implied_norm"].clip(1e-4, 1))
    df["age"] = df["age"].fillna(df["age"].median())
    return df


def fit_clogit(X, race_idx, won, l2=1e-3):
    n_feat = X.shape[1]
    n_races = race_idx.max() + 1

    def nll_grad(beta):
        with np.errstate(all="ignore"):
            u = X @ beta
            u = u - np.maximum.reduceat(u, race_starts)[race_idx]  # stabilize
            e = np.exp(u)
            denom = np.add.reduceat(e, race_starts)
            p = e / denom[race_idx]
            ll = np.sum(np.log(np.clip(p[won == 1], 1e-300, None)))
            grad = X.T @ (p - won)
            val = -(ll) + l2 * beta @ beta
        if not np.isfinite(val) or not np.all(np.isfinite(grad)):
            return 1e12, np.zeros_like(beta)  # reject line-search probe
        return val, grad + 2 * l2 * beta

    order = np.argsort(race_idx, kind="stable")
    X, race_idx, won = X[order], race_idx[order], won[order]
    race_starts = np.searchsorted(race_idx, np.arange(n_races))
    res = minimize(nll_grad, np.zeros(n_feat), jac=True, method="L-BFGS-B",
                   options={"maxiter": 500})
    return res.x


def predict_clogit(beta, X, race_ids):
    u = X @ beta
    df = pd.DataFrame({"rid": race_ids, "u": u})
    df["u"] = df["u"] - df.groupby("rid")["u"].transform("max")
    df["e"] = np.exp(df["u"])
    return (df["e"] / df.groupby("rid")["e"].transform("sum")).values


def race_logloss(p, won, race_ids):
    d = pd.DataFrame({"p": p, "won": won, "rid": race_ids})
    winners = d[d["won"] == 1]
    return -np.log(winners["p"].clip(1e-6)).mean()


def top_pick_hits(p, won, race_ids):
    d = pd.DataFrame({"p": p, "won": won, "rid": race_ids})
    top = d.loc[d.groupby("rid")["p"].idxmax()]
    return top["won"].mean()


def main():
    test_from = sys.argv[1] if len(sys.argv) > 1 else "2026-06-15"
    df = pd.read_parquet(ROOT / "data" / "features.parquet")
    df = prepare(df)
    # need a winner and a market price to be usable
    ok = df.groupby("race_id")["won"].transform("sum") == 1
    df = df[ok & df["final_odds"].notna()].reset_index(drop=True)

    train = df[df["date"] < test_from]
    test = df[df["date"] >= test_from]
    print(f"train: {train['race_id'].nunique()} races / {len(train)} starts   "
          f"test: {test['race_id'].nunique()} races / {len(test)} starts")

    results = {}
    for name, cols in [("fundamentals", FUNDAMENTALS),
                       ("blended", FUNDAMENTALS + MARKET)]:
        mu, sd = train[cols].mean(), train[cols].std().replace(0, 1)
        Xtr = ((train[cols] - mu) / sd).values
        Xte = ((test[cols] - mu) / sd).values
        rid_tr = pd.factorize(train["race_id"])[0]
        beta = fit_clogit(Xtr, rid_tr, train["won"].values.astype(float))
        p = predict_clogit(beta, Xte, test["race_id"].values)
        results[name] = {
            "logloss": race_logloss(p, test["won"].values, test["race_id"].values),
            "top_hit": top_pick_hits(p, test["won"].values, test["race_id"].values),
            "beta": dict(zip(cols, np.round(beta, 4))),
            "p": p,
        }

    market_ll = race_logloss(test["implied_norm"].values, test["won"].values,
                             test["race_id"].values)
    fav_hit = top_pick_hits(test["implied_norm"].values, test["won"].values,
                            test["race_id"].values)
    print(f"\n=== holdout ({test_from}+) ===")
    print(f"market baseline : logloss {market_ll:.4f}  favorite hit {fav_hit:.3f}")
    for name in ("fundamentals", "blended"):
        r = results[name]
        print(f"{name:15s} : logloss {r['logloss']:.4f}  top-pick hit {r['top_hit']:.3f}")

    # value-betting simulation with the blended model
    p = results["blended"]["p"]
    t = test.copy()
    t["p_model"] = p
    for edge in (1.2, 1.4):
        bets = t[(t["p_model"] > edge / t["final_odds"]) & (t["final_odds"] <= 20)]
        if len(bets):
            roi = (bets["won"] * bets["final_odds"]).sum() / len(bets) - 1
            print(f"value bets (edge>{edge:.1f}x, odds<=20): n={len(bets)}, "
                  f"hit {bets['won'].mean():.3f}, flat ROI {roi:+.1%}")

    model = {
        "test_from": test_from,
        "features": FUNDAMENTALS + MARKET,
        "mu": {c: float(train[FUNDAMENTALS + MARKET][c].mean()) for c in FUNDAMENTALS + MARKET},
        "sd": {c: float(train[FUNDAMENTALS + MARKET][c].std() or 1) for c in FUNDAMENTALS + MARKET},
        "beta": {k: float(v) for k, v in results["blended"]["beta"].items()},
        "holdout_logloss": {"market": float(market_ll),
                            "blended": float(results["blended"]["logloss"])},
    }
    out = ROOT / "data" / "model.json"
    out.write_text(json.dumps(model, indent=2))
    print(f"\nsaved blended model -> {out}")


if __name__ == "__main__":
    main()
