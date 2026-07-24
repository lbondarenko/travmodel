---
name: travmodel
phase: published    # concept | poc | mvp | beta | published | promoting | paused
created: 2026-07-22
updated: 2026-07-22
---

# travmodel
Benter-style win-probability model for Swedish trotting, trained on ATG's open
racinginfo API. Purpose: pure fun — the "4th contestant" in the weekly V86
showdown vs. brother-in-law's self-pick, ATG's Harry Boy, and Claude+Lillian's
manual analysis.

## Status
- **Phase:** poc
- **Next action:** run predictions on a live V86 Wednesday and log results vs the other contestants
- **Blocked on:** nothing

## Identity & accounts
> Hobby project. GitHub account: lbondarenko (credentials -> vault). No other accounts,
> no money at stake beyond fun-tickets bought manually on ATG.

## Tech
- Repos: ./src (scrape.py → features.py → train.py → predict.py)
- Stack: Python 3 venv (.venv): numpy, pandas, scipy, scikit-learn, pyarrow
- Data: ./data/raw/<date>.jsonl.gz (one line per race, ATG /races/{id}/extended payload);
  ./data/features.parquet; ./data/model.json (fitted coefficients + scaler)
- Model: conditional logit (softmax per race, L-BFGS MLE), fundamentals + ln(market prob)
  blend — the Bolton-Chapman/Benter architecture

## Key decisions
- 2026-07-22 — Conditional logit over fancier ML: interpretable coefficients, the
  canonical architecture for race betting, tiny data (~5k races) favors simple models.
- 2026-07-22 — Only previous-calendar-year driver/trainer stats as features: the API
  returns current-year aggregates as-of-fetch (leakage). Form comes from the embedded
  point-in-time past starts, filtered to date < race date.
- 2026-07-22 — Blend WITH the market (ln implied prob as a feature) instead of
  competing against it — the market is the strongest single predictor (Benter's insight).

## Links
- Live site: https://lbondarenko.github.io/travmodel/ (GitHub Pages, public)
- Repo: https://github.com/lbondarenko/travmodel (public — Actions cron */30 min regenerates docs/)
- Notes: ./notes/
- ATG API: https://www.atg.se/services/racinginfo/v1/api/{calendar/day/:date, games/:id, races/:id/extended}

## Open loops
- [ ] Score round 1 of the family contest (Skellefteå 2026-07-22) once results land
- [x] **V2 (BUILT 2026-07-22 night)** — trigger: leg-1 winner Edens Odin,
  32/1 Norwegian raider the model had at 3.4% because Swedish-only data made him a ghost.
  Principle: fix eyesight, not opinions (measured: foreign+strong horses are OVERbet ratio 0.85 —
  no blanket raider boost).
  - [x] Scrape foreign (NO/DK/FI) tracks from ATG calendar + extend back through 2025
  - [x] Class feature from past races' firstPrize (kills the earnings trap)
  - [x] Track-adjusted speed figures (km-time normalized by track/distance/startMethod)
  - [x] Avg past-odds signal over last 5 starts (learned weight)
  - [x] Gallop feature via structured disqualification/place codes (archive strips TR comments —
        current regex feature is blind in training, live-only at predict time)
  - [x] Validate: market-weight × data-richness interaction (trust market less on thin-footprint horses)
- [x] Ticket optimizer v1 (greedy budget allocator, shipped in site)
- [x] **Ticket engine v3 (SHIPPED 2026-07-24)**: EV/jackpot-weighted allocator (Jan's skräll economics).
  Backtested 208 rounds w/ real dividends: v2 prob-greedy −1.8% ROI · +45% spik-floor +1.1% ·
  **EV-weighted +14.2% (shipped)** · floor+EV combo −18% (rejected). Payout-aware allocation buys
  underbet horses; weak-banker problem dies naturally.

## Rejected feature hypotheses (measured, do NOT re-add without new evidence)
- [x] **Expert rank as model feature** — REJECTED. Audited 803 legs (src/audit_experts.py):
  Gratistravtips ABCD top pick won 25% vs model 40%. Divergence cell (expert-high/model-low):
  won 3.0% vs model's 5.0% — no incremental signal. Inverse cell (model-high/expert-C/D):
  won 33% ≈ model's 37% — trust the model when experts dismiss. (Scope: free-tier ABCD only,
  which is an ALGORITHM not human handicappers; says nothing about paid Guiden / trainer quotes.)
- [x] **Spik floor (<45% legs must carry 2 horses)** — REJECTED by backtest: floor+EV = −18% ROI.
  The floor strangles the lottery structure EV-weighting exploits.
- [x] **Volt × youth × gallop-history interaction** — REJECTED 2026-07-24. Calibration on 42k starts:
  green (≤3yo, ≤6 starts) in volt = model 13.6% vs actual 13.4% (ratio 0.98); the exact
  green+volt+prior-gallop cell = 11.3% vs 10.9% (0.97, noise). Existing gallops_5 + n_recent
  features already price volt-gallop risk correctly. No interaction term needed.

## Still open
- [ ] Score round 1 of the family contest (Skellefteå 2026-07-22) — DONE, on site as past race
- [ ] **Model coefficient retrain — DEFER to ~late August.** Blocker is DATA VOLUME (only ~6 new
  race-days since v2), not features. Three straight feature hypotheses died in validation → v2 is
  sound; the retrain's job is to ingest more races, not add cleverness. Re-run scrape (SE+NO+DK+FI,
  through retrain date) → features → train → publish; expect small movement.
- [ ] Divergence display flag (expert/streck-vs-model) — display only, not a model feature
