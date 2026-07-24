"""One-shot site publisher with full race lifecycle (GitHub Actions / cron).

Per round:
  - until start-30 min : normal refresh cadence (hourly, 30-min inside 4 h,
    and always-refresh inside the final hour so the last run lands ~30 min out)
  - at start-30 min    : LOCK — data, model and kupong frozen; page stamped
  - start .. +90 min   : shown as locked/underway, no updates
  - at start+90 min    : move to Past Races; fetch results and auto-generate
    the result page from the locked snapshot. If results aren't complete yet,
    show "Running Analysis" on the home card and race page and retry next run.
  - past page exists   : display only, never recompute.

State lives in docs/state.json (committed between runs).

Run: python src/publish.py [--force]
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent))
import webapp  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
STATE_FILE = DOCS / "state.json"
TZ = ZoneInfo("Europe/Stockholm")

FULL_MAX_AGE = 55 * 60
CLOSE_MAX_AGE = 25 * 60
CLOSE_WINDOW = 4 * 3600
LOCK_BEFORE = 30 * 60          # picks locked 30 min before start
PAST_AFTER = 90 * 60           # moved to Past Races 1.5 h after start
HARDCODED_PAST = {"V86_2026-07-22_25_3"}


def lock_page(gid):
    f = DOCS / "game" / f"{gid}.html"
    if not f.exists():
        return
    s = f.read_text()
    if "PICKS LOCKED" in s:
        return
    import re
    s = re.sub(r'<span class="stamp-note">.*?</span>',
               '<span class="stamp-note">PICKS LOCKED — final data before start; '
               'result page appears ~90 min after the race</span>', s, count=1, flags=re.S)
    s = s.replace('<span class="stamp-label">DATA SNAPSHOT</span>',
                  '<span class="stamp-label">PICKS LOCKED</span>')
    f.write_text(s)
    webapp.log(f"locked {gid}")


def mark_analyzing(gid):
    f = DOCS / "game" / f"{gid}.html"
    if not f.exists():
        return
    s = f.read_text()
    if "anlz-banner" in s:
        return
    banner = ('<div class="anlz-banner" style="margin:0 0 16px;padding:10px 14px;'
              'border:2px solid var(--exp);border-radius:10px;font-weight:600;color:var(--exp)">'
              '⏳ RUNNING ANALYSIS — the race is finished; results are being fetched and scored. '
              'The result page appears automatically.</div>')
    s = s.replace('<div class="grid">', banner + '<div class="grid">', 1)
    f.write_text(s)
    webapp.log(f"analyzing banner on {gid}")


def try_make_past(gid, st):
    snap_file = DOCS / "data" / f"{gid}.json"
    if not snap_file.exists():
        webapp.log(f"no snapshot for {gid} — cannot auto-score")
        return None
    snap = json.loads(snap_file.read_text())
    try:
        game = webapp.get(f"{webapp.BASE}/games/{gid}")
    except Exception as e:
        webapp.log(f"results fetch failed for {gid}: {e}")
        return False
    results = {}
    for i, race in enumerate(game.get("races", []), 1):
        places, winner, odds = {}, None, None
        for s_ in race["starts"]:
            r = s_.get("result") or {}
            if r.get("finishOrder"):
                places[int(s_["number"])] = r["finishOrder"]
                if r["finishOrder"] == 1:
                    winner = int(s_["number"])
                    odds = r.get("finalOdds")
        if winner is None:
            return False
        results[str(i)] = {"winner": winner, "odds": odds, "places": places}
    pool = ((game.get("pools") or {}).get(gid.split("_")[0]) or {}).get("result", {}).get("payouts")
    html = webapp.render_past(gid, snap, results, st["start"], pool)
    (DOCS / "past").mkdir(parents=True, exist_ok=True)
    (DOCS / "past" / f"{gid}.html").write_text(html)
    hits = sum(1 for l, r in results.items()
               if r["winner"] in snap["ticket"]["picks"].get(l, []))
    ways = {0: 1}
    for l, r in results.items():
        picks = snap["ticket"]["picks"].get(l, [])
        wc = 1 if r["winner"] in picks else 0
        ww = len(picks) - wc
        new = {}
        for k, v in ways.items():
            if wc: new[k+1] = new.get(k+1, 0) + v*wc
            if ww: new[k] = new.get(k, 0) + v*ww
        ways = new
    win_kr = sum(ways.get(int(t), 0) * ((i.get("payout",0) if isinstance(i,dict) else i)/100)
                 for t, i in (pool or {}).items())
    net = win_kr - snap["ticket"]["cost"]
    st["outcome"] = f"Ticket: {hits} of {len(results)} — net {net:+.0f} kr"
    biggest = max((r.get("odds") or 0) for r in results.values())
    st["note"] = f"Biggest winner odds: {biggest:.2f}" if biggest else "Auto-scored result"
    webapp.log(f"past page generated for {gid} ({hits}/{len(results)})")
    return True


def main():
    force = "--force" in sys.argv
    webapp.WEB = DOCS
    DOCS.mkdir(parents=True, exist_ok=True)
    state = {}
    if STATE_FILE.exists():
        state = json.loads(STATE_FILE.read_text())
    webapp.STATE.update(state)

    for g in webapp.upcoming_games(6):
        st = webapp.STATE.setdefault(g["id"], {})
        st.setdefault("start", g["start"])
        st.setdefault("type", g["type"])

    now = datetime.now(TZ)
    nowts = time.time()
    entries, extra_past = [], []

    for gid, st in sorted(webapp.STATE.items(), key=lambda kv: kv[1].get("start", "")):
        if not isinstance(st, dict) or "start" not in st:
            continue
        gtype = st.get("type", gid.split("_")[0])
        g = {"id": gid, "type": gtype, "start": st["start"]}
        start = datetime.fromisoformat(st["start"]).replace(tzinfo=TZ)
        to_post = (start - now).total_seconds()

        if st.get("past") or (DOCS / "past" / f"{gid}.html").exists():
            st["past"] = True
            st["analyzing"] = False
            if gid not in HARDCODED_PAST:
                extra_past.append({
                    "href": f"past/{gid}.html", "type": gtype,
                    "track": st.get("track", "…"),
                    "finished": start.strftime("%A %d %b"),
                    "outcome": st.get("outcome", "scored"),
                    "note": st.get("note", "auto-scored result"),
                    "_start": st["start"]})
            continue

        if to_post > LOCK_BEFORE:
            max_age = CLOSE_MAX_AGE if to_post <= CLOSE_WINDOW else FULL_MAX_AGE
            if to_post <= 3600:
                max_age = 0  # final hour: refresh on every run for a true last snapshot
            if force or nowts - st.get("last", 0) >= max_age:
                webapp.update_game(g)
            else:
                webapp.log(f"skip {gid} (fresh, {int((nowts-st.get('last',0))/60)} min old, "
                           f"{int(to_post/3600)}h to post)")
            entries.append({**g, **webapp.STATE.get(gid, {})})
        elif to_post > -PAST_AFTER:
            if not st.get("locked"):
                st["locked"] = True
                lock_page(gid)
            label = (" · <b style='color:var(--exp)'>PICKS LOCKED</b>" if to_post > 0
                     else " · <b style='color:var(--exp)'>RACE UNDERWAY · PICKS LOCKED</b>")
            entries.append({**g, **st, "phase_label": label})
        else:
            ok = try_make_past(gid, st)
            if ok:
                st["past"] = True
                st["analyzing"] = False
                extra_past.append({
                    "href": f"past/{gid}.html", "type": gtype,
                    "track": st.get("track", "…"),
                    "finished": start.strftime("%A %d %b"),
                    "outcome": st.get("outcome", "scored"),
                    "note": st.get("note", "auto-scored result"),
                    "_start": st["start"]})
            elif ok is False:
                st["analyzing"] = True
                mark_analyzing(gid)
                extra_past.append({
                    "href": f"game/{gid}.html", "type": gtype,
                    "track": st.get("track", "…"),
                    "finished": start.strftime("%A %d %b"),
                    "outcome": "⏳ Running Analysis",
                    "note": "race finished — fetching and scoring results",
                    "_start": st["start"]})
            else:  # no snapshot: nothing we can score
                st["past"] = True

    entries = entries[:3]
    extra_past.sort(key=lambda x: x.get("_start", ""), reverse=True)
    extra_past = extra_past[:3]
    for x in extra_past:
        x.pop("_start", None)

    (DOCS / "index.html").write_text(webapp.render_index(entries, extra_past))
    webapp.write_login(DOCS)
    (DOCS / ".nojekyll").write_text("")
    STATE_FILE.write_text(json.dumps(dict(webapp.STATE), default=str))
    webapp.log(f"index written: {len(entries)} upcoming, {len(extra_past)} dynamic past")


if __name__ == "__main__":
    main()
