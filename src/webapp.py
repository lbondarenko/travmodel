"""travmodel web — local site with auto-updating model sheets.

- Home page (/): the next 3 upcoming betting rounds (V86/V75/GS75/V64/V65),
  each linking to a detail page.
- Detail page (/game/<id>.html): per-leg tiles with V64/V86-streck vs the
  model's win probabilities, program comments for the model's top picks,
  VALUE/overbet flags, and a prominent data-snapshot stamp.
- Scheduler thread: refreshes each round every 60 min, tightening to every
  30 min once inside 4 hours to post. Pages are static files in web/,
  served by a tiny built-in HTTP server on 0.0.0.0:8030 (open from the
  iPad too: http://<mac-ip>:8030).

Run: .venv/bin/python src/webapp.py
"""
import json
import sys
import threading
import time
import urllib.request
from datetime import datetime, timedelta
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import start_features  # noqa: E402
from train import prepare  # noqa: E402

BASE = "https://www.atg.se/services/racinginfo/v1/api"
GAME_TYPES = ["V86", "V75", "GS75", "V64", "V65"]
PORT = 8030
FULL_INTERVAL = 3600          # >4h to post: hourly
CLOSE_INTERVAL = 1800         # <=4h to post: every 30 min
CLOSE_WINDOW = 4 * 3600

MODEL = json.loads((ROOT / "data" / "model.json").read_text())
_raw = json.loads((ROOT / "data" / "baselines.json").read_text())
BASELINES, FALLBACK = {}, {}
for k, v in _raw.items():
    if k.startswith("__fb__"):
        m, b = k[6:].split("|")
        FALLBACK[(m if m != "None" else None, b)] = v
    else:
        t, m, b = k.split("|")
        BASELINES[(t, m if m != "None" else None, b)] = v


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "travmodel-web/0.1"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def upcoming_games(n=3):
    games = []
    today = datetime.now().date()
    for d in range(4):
        day = today + timedelta(days=d)
        try:
            cal = get(f"{BASE}/calendar/day/{day}")
        except Exception:
            continue
        trot_tracks = {str(t["id"]) for t in cal.get("tracks", [])
                       if t.get("sport") in (None, "trot")}
        for gtype, lst in (cal.get("games") or {}).items():
            if gtype not in GAME_TYPES:
                continue
            for g in lst:
                if g.get("status") != "bettable":
                    continue
                start = g.get("startTime")
                track_id = g["id"].split("_")[2] if len(g["id"].split("_")) > 2 else ""
                if not start or track_id not in trot_tracks:
                    continue
                games.append({"id": g["id"], "type": gtype, "start": start})
        if len(games) >= n:
            break
    games.sort(key=lambda g: g["start"])
    return games[:n]


def score_game(game_id):
    """Fetch a game, run the model, return leg data for rendering."""
    game = get(f"{BASE}/games/{game_id}")
    gtype = game_id.split("_")[0]
    track = game["races"][0]["track"]["name"] if game.get("races") else "?"
    rows, comments, legmeta = [], {}, {}
    for leg, race in enumerate(game["races"], 1):
        ext = get(f"{BASE}/races/{race['id']}/extended")
        legmeta[leg] = f"{ext.get('distance','')}m {ext.get('startMethod','')}"
        scratched = {s["number"] for s in race["starts"]
                     if s.get("scratched") or (s.get("horse") or {}).get("scratched")}
        live, bd = {}, {}
        for s in race["starts"]:
            pools = s.get("pools") or {}
            vo = (pools.get("vinnare") or {}).get("odds")
            if vo and vo > 100:
                live[s["number"]] = vo / 100
            b = (pools.get(gtype) or {}).get("betDistribution")
            if b:
                bd[s["number"]] = b / 100
        for s in ext["starts"]:
            if s["number"] in scratched:
                continue
            f = start_features(ext, s, BASELINES, FALLBACK)
            f["leg"] = leg
            f["live_odds"] = live.get(s["number"])
            f["streck"] = bd.get(s["number"], 0.0)
            f["driver"] = f"{(s['driver'].get('firstName') or '')[:1]}.{s['driver'].get('lastName','')}"
            comments[(leg, s["number"])] = " ".join(
                c.get("commentText", "") for c in (s.get("comments") or []))
            rows.append(f)
    df = pd.DataFrame(rows)
    df["implied"] = 1.0 / df["live_odds"]
    df["implied"] = df["implied"].fillna(df["streck"].where(df["streck"] > 0) / 100)
    df["implied"] = df["implied"].fillna(df.groupby("leg")["implied"].transform("mean"))
    df["implied_norm"] = df["implied"] / df.groupby("leg")["implied"].transform("sum")
    df = prepare(df)
    cols = MODEL["features"]
    mu = pd.Series(MODEL["mu"])[cols]
    sd = pd.Series(MODEL["sd"])[cols]
    beta = pd.Series(MODEL["beta"])[cols]
    X = ((df[cols] - mu) / sd).values
    with np.errstate(all="ignore"):
        df["u"] = X @ beta.values
    df["u"] = df["u"] - df.groupby("leg")["u"].transform("max")
    df["e"] = np.exp(df["u"])
    df["p"] = df["e"] / df.groupby("leg")["e"].transform("sum")
    legs = {}
    for leg in sorted(df["leg"].unique()):
        d = df[df["leg"] == leg].sort_values("p", ascending=False)
        legs[int(leg)] = [{
            "nr": int(r["start_number"]), "horse": r["horse"], "driver": r["driver"],
            "streck": float(r["streck"]), "model": 100 * float(r["p"]),
            "comment": comments.get((leg, int(r["start_number"])), ""),
        } for _, r in d.iterrows()]
    return {"track": track, "type": gtype, "legs": legs, "legmeta": legmeta}


# ---------- rendering ----------

CSS = """
  :root{ --bg:#FAF8F3; --ink:#26241F; --muted:#867F73; --line:#E3DED4;
    --pick:#2E6B4A; --pick-bg:#EDF3EE; --head-bg:#26241F; --head-ink:#FAF8F3;
    --exp:#8A5A1E; --card:#FFFFFF; }
  @media (prefers-color-scheme: dark){ :root{ --bg:#181A16; --ink:#E9E6DD; --muted:#98937F;
    --line:#37392F; --pick:#8CC3A4; --pick-bg:#26312A; --head-bg:#E9E6DD; --head-ink:#1D1F1B;
    --exp:#D4A860; --card:#22241F; } }
  *{ box-sizing:border-box; } html,body{ margin:0; }
  body{ background:var(--bg); color:var(--ink); line-height:1.45; padding:36px 18px 60px;
    font-family:"Avenir Next","Seravek",Seravek,system-ui,-apple-system,sans-serif; }
  main{ max-width:960px; margin:0 auto; }
  h1{ font-size:clamp(20px,3.5vw,26px); font-weight:600; margin:0 0 3px; letter-spacing:-.01em; }
  .sub{ color:var(--muted); font-size:13px; margin:0; }
  a{ color:inherit; }
  .pagehead{ position:relative; margin-bottom:20px; padding-right:270px; }
  .stampbox{ position:absolute; top:0; right:0; display:flex; flex-direction:column; gap:1px;
    width:250px; background:var(--head-bg); color:var(--head-ink); border-radius:8px;
    padding:9px 13px; border-left:5px solid var(--exp); }
  @media (max-width:640px){ .pagehead{ padding-right:0; }
    .stampbox{ position:static; margin-top:10px; width:auto; max-width:280px; } }
  .stamp-label{ font-size:9px; font-weight:700; letter-spacing:.18em; opacity:.75; }
  .stamp-time{ font-size:19px; font-weight:700; font-variant-numeric:tabular-nums; line-height:1.25; }
  .stamp-note{ font-size:10px; opacity:.8; line-height:1.35; margin-top:2px; }
  .grid{ display:grid; grid-template-columns:repeat(2,1fr); gap:16px; align-items:start; }
  @media (max-width:640px){ .grid{ grid-template-columns:1fr; } }
  .tile{ background:var(--card); border:1px solid var(--line); border-radius:10px;
    padding:12px 14px 10px; break-inside:avoid; }
  .leghead{ display:flex; gap:10px; align-items:baseline; border-bottom:2px solid var(--ink);
    padding-bottom:4px; margin-bottom:2px; }
  .leghead h2{ font-size:15px; margin:0; }
  .leghead .meta{ color:var(--muted); font-size:11px; }
  table{ border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; }
  th,td{ padding:3.5px 6px; text-align:left; font-size:12px; }
  thead th{ background:var(--head-bg); color:var(--head-ink); font-size:9.5px; letter-spacing:.07em; }
  th:nth-child(3),td:nth-child(3),th:nth-child(4),td:nth-child(4){ text-align:right; }
  tbody tr{ border-bottom:1px solid var(--line); }
  td.nr{ color:var(--muted); width:22px; }
  td.num{ white-space:nowrap; } td.strong{ font-weight:600; }
  tr.top td{ background:var(--pick-bg); }
  tr.top td:nth-child(2){ color:var(--pick); font-weight:700; }
  .flag{ font-size:9px; font-weight:700; letter-spacing:.08em; border-radius:3px; padding:0 4px; }
  .flag.value{ color:var(--pick); border:1px solid currentColor; }
  .flag.over{ color:var(--exp); border:1px solid currentColor; }
  .infos{ margin-top:8px; }
  .info{ margin:0 0 5px; font-size:11px; color:var(--muted); line-height:1.5; }
  .info b{ color:var(--pick); }
  .cards{ display:flex; flex-direction:column; gap:14px; }
  .gamecard{ display:block; text-decoration:none; background:var(--card);
    border:1px solid var(--line); border-radius:12px; padding:16px 18px; }
  .gamecard:hover{ border-color:var(--pick); }
  .gamecard .row1{ display:flex; justify-content:space-between; align-items:baseline; gap:10px; flex-wrap:wrap; }
  .gamecard .gt{ font-size:20px; font-weight:700; }
  .gamecard .when{ font-size:14px; font-variant-numeric:tabular-nums; color:var(--muted); }
  .gamecard .row2{ display:flex; justify-content:space-between; gap:10px; flex-wrap:wrap;
    margin-top:4px; font-size:12.5px; color:var(--muted); }
  .fresh{ color:var(--pick); font-weight:600; }
  footer{ margin-top:24px; padding-top:10px; border-top:1px solid var(--line);
    font-size:10.5px; color:var(--muted); line-height:1.6; }
  .userchip{ text-align:right; font-size:12px; color:var(--muted); margin:-20px 0 8px; }
  .userchip a{ color:var(--pick); }
  @media print{ .userchip{ display:none; } }
  @media print{
    :root{ --muted:#3A3733; --pick:#1E4A33; --exp:#6B4413; --line:#999; }
    body{ background:#fff; color:#000; padding:0; }
    main{ max-width:none; } .grid{ gap:10px; } .tile{ border-color:#888; }
    thead th{ background:#000 !important; color:#fff !important; -webkit-print-color-adjust:exact; print-color-adjust:exact; }
    .stampbox{ background:#000 !important; color:#fff !important; -webkit-print-color-adjust:exact; print-color-adjust:exact; }
    tr.top td{ background:#EDF3EE !important; -webkit-print-color-adjust:exact; print-color-adjust:exact; }
    .info, .sub, footer, .leghead .meta, td.nr{ color:#3A3733; }
  }
"""


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ---------- simple login gate ----------
# NOTE: static-site gate, not real security — content is public files.
# Hashes are sha256("username:password").
AUTH_USERS = {
    "c83b5c64cb855009fc8ee591068416efb8b1611b8f2830988d25c38e80262a26": "Lillian",
    "32006fd4a0d2aae8144f86cd555e5f4207b0c5b47a4ee6110893b45d3b7c2423": "Jan",
}


SESSION_MS = 7 * 24 * 3600 * 1000  # login lasts a week


def auth_guard(login_rel):
    return (f"""<script>(function(){{var t=+localStorage.getItem("tm_ts")||0;"""
            f"""if(!localStorage.getItem("tm_user")||Date.now()-t>{SESSION_MS})"""
            f"""location.replace("{login_rel}");}})();</script>""")


USERCHIP = """<div class="userchip">👤 <span id="tmu"></span> · <a href="#" id="tmlo">log out</a></div>
<script>
document.getElementById('tmu').textContent=localStorage.getItem('tm_user')||'';
document.getElementById('tmlo').onclick=function(){localStorage.removeItem('tm_user');location.reload();return false;};
</script>"""

LOGIN_HTML = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Lillian's Model — log in</title>
<style>__CSS__
  .loginbox{ max-width:340px; margin:10vh auto 0; background:var(--card);
    border:1px solid var(--line); border-radius:12px; padding:26px 26px 22px; }
  .loginbox h1{ margin-bottom:14px; }
  .loginbox label{ display:block; font-size:12px; color:var(--muted); margin:12px 0 4px;
    letter-spacing:.06em; }
  .loginbox input{ width:100%; padding:9px 11px; font-size:15px; border:1px solid var(--line);
    border-radius:7px; background:var(--bg); color:var(--ink); font-family:inherit; }
  .loginbox button{ margin-top:18px; width:100%; padding:10px; font-size:15px; font-weight:600;
    background:var(--pick); color:#fff; border:none; border-radius:7px; cursor:pointer;
    font-family:inherit; }
  .err{ color:#C23B2E; font-size:13px; margin-top:10px; display:none; }
</style></head><body><main>
<form class="loginbox" id="f">
  <h1>Lillian's Model 🐴</h1>
  <label for="u">USERNAME</label><input id="u" autocomplete="username" autocapitalize="none">
  <label for="p">PASSWORD</label><input id="p" type="password" autocomplete="current-password">
  <button type="submit">Log in</button>
  <p class="err" id="e">Wrong username or password.</p>
</form>
<script>
var USERS = __USERS__;
async function h(s){
  var b = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(s));
  return Array.from(new Uint8Array(b)).map(x=>x.toString(16).padStart(2,"0")).join("");
}
document.getElementById("f").onsubmit = async function(ev){
  ev.preventDefault();
  var u = document.getElementById("u").value.trim().toLowerCase();
  var p = document.getElementById("p").value;
  var d = await h(u + ":" + p);
  if(USERS[d]){ localStorage.setItem("tm_user", USERS[d]); localStorage.setItem("tm_ts", Date.now()); location.replace("index.html"); }
  else document.getElementById("e").style.display = "block";
};
if(localStorage.getItem("tm_user") && Date.now()-(+localStorage.getItem("tm_ts")||0) < 604800000) location.replace("index.html");
</script></main></body></html>"""


def write_login(outdir):
    html = LOGIN_HTML.replace("__CSS__", CSS).replace("__USERS__", json.dumps(AUTH_USERS))
    (Path(outdir) / "login.html").write_text(html)


def stampbox(updated, start_str):
    return f"""<div class="stampbox"><span class="stamp-label">DATA SNAPSHOT</span>
<span class="stamp-time">{updated}</span>
<span class="stamp-note">streck &amp; odds move until post {start_str} — auto-updates hourly, every 30 min in the last 4 h</span></div>"""


def render_game(game, data, updated):
    start_dt = datetime.fromisoformat(game["start"])
    tiles = []
    for leg, horses in data["legs"].items():
        rows, infos = [], []
        for i, h in enumerate(horses):
            edge = h["model"] / h["streck"] if h["streck"] else None
            flag = ""
            if edge and edge > 1.3 and h["model"] > 8:
                flag = " <span class='flag value'>VALUE</span>"
            elif edge and edge < 0.7 and h["streck"] > 15:
                flag = " <span class='flag over'>OVERBET</span>"
            cls = "top" if i < (1 if horses[0]["model"] > 45 else 2) else ""
            rows.append(f"<tr class='{cls}'><td class='nr'>{h['nr']}</td>"
                        f"<td>{esc(h['horse'])}{flag}</td>"
                        f"<td class='num'>{h['streck']:.1f}%</td>"
                        f"<td class='num strong'>{h['model']:.1f}%</td></tr>")
        for h in horses[:3]:
            if h["comment"]:
                infos.append(f"<p class='info'><b>#{h['nr']} {esc(h['horse'])}</b> "
                             f"({esc(h['driver'])}) — {esc(h['comment'])}</p>")
        spik = " · ★ spik candidate" if horses and horses[0]["model"] > 45 else ""
        tiles.append(f"""<article class="tile">
<div class="leghead"><h2>Leg {leg}</h2><span class="meta">{esc(data['legmeta'].get(leg,''))}{spik}</span></div>
<table><thead><tr><th>#</th><th>Horse</th><th>Streck</th><th>Model</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<div class="infos">{''.join(infos)}</div></article>""")
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="600">
{auth_guard("../login.html")}
<title>{data['type']} {esc(data['track'])} {start_dt.strftime('%a %d %b %H:%M')} — Lillian's Model</title>
<style>{CSS}</style></head><body><main>
{USERCHIP}
<div class="pagehead"><div>
<h1><a href="../index.html">←</a> {data['type']} {esc(data['track'])} · {start_dt.strftime('%A %d %B %Y')}</h1>
<p class="sub">First start {start_dt.strftime('%H:%M')} · streck (share of tickets) vs Lillian's Model
(win probability) · green rows = model's top of the leg · program comments under each table</p>
</div>{stampbox(updated, start_dt.strftime('%H:%M'))}</div>
<div class="grid">{''.join(tiles)}</div>
<footer>Lillian's Model = travmodel v2 · conditional logit on 17,356 Nordic races · market-blended ·
data from ATG's open API · page auto-reloads every 10 min · not betting advice, not a valid bet.</footer>
</main></body></html>"""


def render_index(entries):
    cards = []
    for e in entries:
        start_dt = datetime.fromisoformat(e["start"])
        upd = e.get("updated", "not yet")
        cards.append(f"""<a class="gamecard" href="game/{e['id']}.html">
<div class="row1"><span class="gt">{e['type']} · {esc(e.get('track','...'))}</span>
<span class="when">{start_dt.strftime('%A %d %b · %H:%M')}</span></div>
<div class="row2"><span>{e.get('nlegs','?')} legs · model + streck sheet</span>
<span class="fresh">data: {upd}</span></div></a>""")
    now = datetime.now().strftime("%a %d %b · %H:%M")
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="300">
{auth_guard("login.html")}
<title>Lillian's Model — next races</title><style>{CSS}</style></head><body><main>
{USERCHIP}
<div class="pagehead"><div>
<h1>Lillian's Model 🐴</h1>
<p class="sub">The next {len(entries)} betting rounds, scored by travmodel v2. Pages update hourly —
every 30 minutes inside the final 4 hours before post.</p>
</div><div class="stampbox"><span class="stamp-label">PAGE GENERATED</span>
<span class="stamp-time">{now}</span>
<span class="stamp-note">each race card shows its own data snapshot</span></div></div>
<div class="cards">{''.join(cards)}</div>
<footer>travmodel v2 · ATG open data · family duel edition · not betting advice.</footer>
</main></body></html>"""


# ---------- scheduler ----------

STATE = {}          # game_id -> {"last": ts, "track":..., "nlegs":..., "updated": str}
STATE_LOCK = threading.Lock()


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def update_game(game):
    gid = game["id"]
    try:
        data = score_game(gid)
    except Exception as e:
        log(f"update {gid} FAILED: {e}")
        return
    updated = datetime.now().strftime("%a %d %b · %H:%M")
    (WEB / "game").mkdir(parents=True, exist_ok=True)
    (WEB / "game" / f"{gid}.html").write_text(render_game(game, data, updated))
    with STATE_LOCK:
        STATE[gid] = {"last": time.time(), "track": data["track"],
                      "nlegs": len(data["legs"]), "updated": updated}
    log(f"updated {gid} ({data['type']} {data['track']}, {len(data['legs'])} legs)")


def scheduler():
    while True:
        try:
            games = upcoming_games(3)
            for g in games:
                start_ts = datetime.fromisoformat(g["start"]).timestamp()
                to_post = start_ts - time.time()
                interval = CLOSE_INTERVAL if to_post <= CLOSE_WINDOW else FULL_INTERVAL
                with STATE_LOCK:
                    last = STATE.get(g["id"], {}).get("last", 0)
                if time.time() - last >= interval:
                    update_game(g)
            entries = []
            for g in games:
                with STATE_LOCK:
                    st = STATE.get(g["id"], {})
                entries.append({**g, **st})
            WEB.mkdir(parents=True, exist_ok=True)
            (WEB / "index.html").write_text(render_index(entries))
            write_login(WEB)
        except Exception as e:
            log(f"scheduler error: {e}")
        time.sleep(120)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(WEB), **kw)

    def log_message(self, *a):
        pass


def main():
    WEB.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=scheduler, daemon=True).start()
    log(f"serving http://localhost:{PORT} (LAN: http://<mac-ip>:{PORT})")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
