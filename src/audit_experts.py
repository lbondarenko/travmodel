"""Audit: how good are the experts' rankings, and do they add signal
beyond the market and Travmodel?

Joins data/experts_hist (Gratistravtips ABCD per round) with
data/games_hist (results + streck) and the model's probabilities.

Usage: python src/audit_experts.py
"""
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest import model_probs  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main():
    probs = model_probs()
    n_legs = 0
    hits = {"expert": 0, "model": 0, "streck": 0}
    rankdist = {"A": 0, "B": 0, "C": 0, "D": 0, "?": 0}
    div_cell = []      # expert-top but model < 10%
    inv_cell = []      # model-top but expert C/D
    for exf in sorted((ROOT / "data" / "experts_hist").glob("*.json")):
        gid = exf.stem
        gf = ROOT / "data" / "games_hist" / f"{gid}.json.gz"
        if not gf.exists():
            continue
        ex = json.loads(exf.read_text())
        game = json.load(gzip.open(gf, "rt"))
        gtype = gid.split("_")[0]
        if len(ex.get("legs", {})) != len(game.get("races", [])):
            continue
        for i, race in enumerate(game["races"], 1):
            exleg = ex["legs"].get(str(i))
            if not exleg:
                continue
            winner, best_st, best_st_nr = None, -1, None
            pm = {}
            for s in race["starts"]:
                nr = int(s["number"])
                res = s.get("result") or {}
                if res.get("finishOrder") == 1:
                    winner = nr
                bd = ((s.get("pools") or {}).get(gtype) or {}).get("betDistribution") or 0
                if bd > best_st:
                    best_st, best_st_nr = bd, nr
                p = probs.get((race["id"], nr))
                if p is not None:
                    pm[nr] = p
            if winner is None or not pm:
                continue
            ex_by_nr = {h["nr"]: h for h in exleg}
            if set(ex_by_nr) != set(pm):
                # scratched horses etc — require winner present at least
                if winner not in ex_by_nr:
                    continue
            ex_top = max(exleg, key=lambda h: h.get("pts", 0))["nr"]
            model_top = max(pm, key=pm.get)
            n_legs += 1
            hits["expert"] += 1 if ex_top == winner else 0
            hits["model"] += 1 if model_top == winner else 0
            hits["streck"] += 1 if best_st_nr == winner else 0
            rankdist[ex_by_nr.get(winner, {}).get("rank", "?")] += 1
            # divergence: expert-top horse that the model prices under 10%
            p_extop = pm.get(ex_top)
            if p_extop is not None and p_extop < 0.10:
                div_cell.append((p_extop, 1 if ex_top == winner else 0))
            # inverse: model-top horse the experts rank C or D
            if ex_by_nr.get(model_top, {}).get("rank") in ("C", "D"):
                inv_cell.append((pm[model_top], 1 if model_top == winner else 0))

    print(f"legs audited: {n_legs}\n")
    print("top-pick strike rate (same legs):")
    for k in ("expert", "model", "streck"):
        print(f"  {k:<8} {hits[k]/n_legs:.1%}")
    tot = sum(rankdist.values())
    print("\nwhere winners came from (expert rank of the winner):")
    for r in ("A", "B", "C", "D", "?"):
        print(f"  {r}: {rankdist[r]/tot:.1%}")
    if div_cell:
        n = len(div_cell)
        wr = sum(w for _, w in div_cell) / n
        ep = sum(p for p, _ in div_cell) / n
        print(f"\nDIVERGENCE CELL — expert top pick, model <10% (n={n}):")
        print(f"  model said {ep:.1%} on average · actually won {wr:.1%}")
        print("  -> experts DO add signal here" if wr > ep * 1.3 else
              "  -> no meaningful extra signal beyond the model")
    if inv_cell:
        n = len(inv_cell)
        wr = sum(w for _, w in inv_cell) / n
        ep = sum(p for p, _ in inv_cell) / n
        print(f"\nINVERSE CELL — model top pick, experts rank C/D (n={n}):")
        print(f"  model said {ep:.1%} on average · actually won {wr:.1%}")
        print("  -> trust the model when experts dismiss it" if wr > ep * 0.8 else
              "  -> expert dismissal is a valid warning")


if __name__ == "__main__":
    main()
