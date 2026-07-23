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
from sv_en import translate  # noqa: E402

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
    contrib = X * beta.values
    df["_ci"] = range(len(df))
    legs = {}
    for leg in sorted(df["leg"].unique()):
        d = df[df["leg"] == leg].sort_values("p", ascending=False)
        entries = []
        for i, (_, r) in enumerate(d.iterrows()):
            e = {
                "nr": int(r["start_number"]), "horse": r["horse"], "driver": r["driver"],
                "streck": float(r["streck"]), "model": 100 * float(r["p"]),
                "comment": comments.get((leg, int(r["start_number"])), ""),
                "family": str(r["horse"]).strip().lower() in FAMILY_SPIKS,
            }
            if i < 2:
                e["reasons"] = gen_reasons(r, contrib[int(r["_ci"])], cols)
            entries.append(e)
        legs[int(leg)] = entries
    fam_legs = [leg for leg, hs in legs.items() if any(h.get("family") for h in hs)]
    return {"track": track, "type": gtype, "legs": legs, "legmeta": legmeta,
            "fam_legs": fam_legs}


# ---------- past races (hand-curated result pages in docs/past/) ----------
PAST_RACES = [
    {"href": "past/V86_2026-07-22_25_3.html", "type": "V86", "track": "Skellefteå",
     "finished": "Wednesday 22 Jul", "outcome": "Our ticket: 5 of 8 legs — no payout (6+ needed)",
     "note": "Two spik-killing skrälls: 32/1 Edens Odin & 85/1 Grace Kelly"},
]

# ---------- ticket builder ----------

ROW_PRICE = {"V64": 1.0, "V65": 1.0, "V75": 0.5, "V86": 0.25, "GS75": 0.25}
TICKET_BUDGET = 50.0   # kronor
MAX_PER_LEG = 5

# FAMILY RULE — non-negotiable, overrides all statistics:
# Pralines is Jan's own horse. When she races, she IS the spik. Always.
FAMILY_SPIKS = {"pralines": "Jans häst"}


def build_ticket(legs, gtype):
    """Greedy budget allocator: repeatedly add the horse that buys the most
    log-coverage per krona, until the budget is exhausted. Spiks emerge on
    legs where the top horse is strong (additions there are inefficient)."""
    import math
    price = ROW_PRICE.get(gtype, 1.0)
    forced = {}
    for leg, horses in legs.items():
        for h in horses:
            if h["horse"].strip().lower() in FAMILY_SPIKS:
                forced[leg] = h
    sel = {leg: 1 for leg in legs}            # horses taken from the top of each leg
    cov = {leg: (forced[leg]["model"] / 100 if leg in forced
                 else legs[leg][0]["model"] / 100) for leg in legs}

    def rows():
        r = 1
        for n in sel.values():
            r *= n
        return r

    while True:
        best, best_eff = None, 0.0
        base_rows = rows()
        for leg, horses in legs.items():
            if leg in forced:
                continue  # family rule: spik stays a spik, no additions
            k = sel[leg]
            if k >= min(MAX_PER_LEG, len(horses)):
                continue
            p_next = horses[k]["model"] / 100
            new_cost = price * base_rows * (k + 1) / k
            if new_cost > TICKET_BUDGET:
                continue
            gain = math.log((cov[leg] + p_next) / cov[leg])
            added_cost = new_cost - price * base_rows
            eff = gain / added_cost if added_cost > 0 else 0
            if eff > best_eff:
                best, best_eff = leg, eff
        if best is None:
            break
        cov[best] += legs[best][sel[best]]["model"] / 100
        sel[best] += 1

    hit_all = 1.0
    for leg in legs:
        hit_all *= min(cov[leg], 0.99)
    picks, spiks = {}, {}
    for leg in legs:
        if leg in forced:
            picks[leg] = [forced[leg]["nr"]]
            spiks[leg] = {**forced[leg], "family": True}
        else:
            picks[leg] = [h["nr"] for h in legs[leg][:sel[leg]]]
            if sel[leg] == 1:
                spiks[leg] = {**legs[leg][0], "family": False}
    return {"picks": picks, "spiks": spiks,
            "rows": rows(), "price": price, "cost": rows() * price,
            "hit_all": hit_all}




# ---------- user ticket upload & comparison (client-side, localStorage) ----------
TIX_HTML = """
<div class="modalback" id="tixmodal">
<div class="modal">
<h3>Upload my ticket</h3>
<p class="mnote">Attach a photo of the receipt for reference, then type the picks —
20 seconds, and it never leaves your browser.</p>
<label>Photo of ticket (optional — gallery, camera, or paste)</label>
<div class="mrow photorow">
<button type="button" class="mbtn" onclick="document.getElementById('tixphoto').click()">📁 Gallery</button>
<button type="button" class="mbtn" onclick="document.getElementById('tixcam').click()">📷 Camera</button>
<button type="button" class="mbtn" onclick="tixPaste()">📋 Paste</button>
</div>
<input type="file" id="tixphoto" accept="image/*" style="display:none">
<input type="file" id="tixcam" accept="image/*" capture="environment" style="display:none">
<img id="tixpreview" alt="" style="display:none">
<p class="mnote">Tip: Cmd/Ctrl+V anywhere in this dialog pastes a screenshot.</p>
<label>Label</label>
<input type="text" id="tixlabel" placeholder="e.g. Harry Boy / Min egen">
<label>Ticket number (for dedupe — the long number or rättningskod on the receipt)</label>
<input type="text" id="tixno" placeholder="e.g. D2F6 4828 7800 2027">
<div id="tixlegs"></div>
<p class="mnote err" id="tixerr" style="display:none"></p>
<div class="mrow"><button class="mbtn save" onclick="tixSave()">Save ticket</button>
<button class="mbtn" onclick="tixClose()">Cancel</button></div>
</div></div>
<section class="cmp" id="cmpsec"></section>
"""

TIX_JS = """
<script>
(function(){
var TM = __TMDATA__;
var KEY = "tm_tix_v1";
function user(){ return localStorage.getItem("tm_user")||"?"; }
function store(){ try{return JSON.parse(localStorage.getItem(KEY))||{};}catch(e){return {};} }
function mytix(){ var s=store(); return (s[user()]||{})[TM.game]||{}; }
function saveTix(no,obj){ var s=store(); s[user()]=s[user()]||{}; s[user()][TM.game]=s[user()][TM.game]||{};
  var existed = !!s[user()][TM.game][no]; s[user()][TM.game][no]=obj;
  localStorage.setItem(KEY, JSON.stringify(s)); return existed; }
function delTix(no){ var s=store(); if(s[user()]&&s[user()][TM.game]){ delete s[user()][TM.game][no];
  localStorage.setItem(KEY, JSON.stringify(s)); } renderCmp(); }
document.addEventListener("click",function(ev){
  var a=(ev.target.closest?ev.target.closest(".tixdel"):null);
  if(a){ ev.preventDefault(); delTix(a.getAttribute("data-no")); }
});
window.tixOpen=function(){
  var box=document.getElementById("tixlegs"); box.innerHTML="";
  Object.keys(TM.legs).sort(function(a,b){return a-b;}).forEach(function(l){
    box.innerHTML += "<label>Leg "+l+" — horses (e.g. 2, 3, 5)</label>"+
      "<input type='text' class='tixleg' data-leg='"+l+"' inputmode='numeric'>";
  });
  document.getElementById("tixmodal").style.display="flex";
};
window.tixClose=function(){ document.getElementById("tixmodal").style.display="none"; };
function showPhoto(blob){ var img=document.getElementById("tixpreview");
  img.src=URL.createObjectURL(blob); img.style.display="block"; }
document.addEventListener("change",function(ev){
  if(ev.target && (ev.target.id==="tixphoto"||ev.target.id==="tixcam") && ev.target.files[0]){
    showPhoto(ev.target.files[0]); }});
document.addEventListener("paste",function(ev){
  var m=document.getElementById("tixmodal");
  if(!m || m.style.display!=="flex" || !ev.clipboardData) return;
  for(var i=0;i<ev.clipboardData.items.length;i++){
    var it=ev.clipboardData.items[i];
    if(it.type && it.type.indexOf("image")===0){ showPhoto(it.getAsFile()); ev.preventDefault(); return; }
  }});
window.tixPaste=function(){
  if(navigator.clipboard && navigator.clipboard.read){
    navigator.clipboard.read().then(function(items){
      var found=false;
      for(var i=0;i<items.length;i++){
        var t=(items[i].types||[]).filter(function(x){return x.indexOf("image")===0;})[0];
        if(t){ found=true; items[i].getType(t).then(showPhoto); break; } }
      if(!found) alert("No image on the clipboard - copy a screenshot first.");
    }).catch(function(){ alert("Clipboard blocked by the browser - press Cmd/Ctrl+V instead."); });
  } else alert("Press Cmd/Ctrl+V to paste the screenshot.");
};
window.tixSave=function(){
  var err=document.getElementById("tixerr"); err.style.display="none";
  var no=(document.getElementById("tixno").value||"").replace(/\s+/g," ").trim();
  var label=(document.getElementById("tixlabel").value||"Ticket").trim();
  if(!no){ err.textContent="Ticket number is required (it is how duplicates are caught).";
    err.style.display="block"; return; }
  var picks={}, bad=null;
  document.querySelectorAll(".tixleg").forEach(function(inp){
    var l=inp.dataset.leg;
    var nums=(inp.value||"").split(/[^0-9]+/).filter(Boolean).map(Number);
    var valid=TM.legs[l]||[];
    nums.forEach(function(n){ if(valid.indexOf(n)<0) bad="Leg "+l+": horse "+n+" is not in that leg."; });
    if(!nums.length) bad=bad||("Leg "+l+" is empty.");
    picks[l]=nums;
  });
  if(bad){ err.textContent=bad; err.style.display="block"; return; }
  var existed=saveTix(no,{label:label,picks:picks,ts:Date.now()});
  tixClose(); renderCmp();
  if(existed) alert("That ticket number was already uploaded — it has been updated (no duplicate created).");
};
function deco(n,l){
  var w=TM.winners[l];
  if(w===undefined||w===null) return String(n);
  if(Number(n)===Number(w)) return "<span class='hitnum'>"+n+"</span>";
  return "<s class='lostnum'>"+n+"</s>";
}
function cell(nums,l,label){
  var w=TM.winners[l];
  var html=(label&&nums.length===1)?deco(nums[0],l)+" "+label:nums.map(function(n){return deco(n,l);}).join(", ");
  var miss=(w!==undefined&&w!==null&&nums.map(Number).indexOf(Number(w))<0);
  return "<td class='"+(miss?"misscell":"")+"'>"+html+(miss?" ✗":"")+"</td>";
}
window.renderCmp=function(){
  var sec=document.getElementById("cmpsec"); if(!sec) return;
  var tix=mytix(); var nos=Object.keys(tix);
  sec.classList.toggle("hasusr", nos.length>0);
  var legs=Object.keys(TM.legs).sort(function(a,b){return a-b;});
  var scored=Object.keys(TM.winners).length>0;
  var h="<div class='leghead'><h2>Tickets — the model vs "+user()+"</h2></div>";
  h+="<table><thead><tr><th>Leg</th><th>The Model"+(TM.modelName?"<small>"+TM.modelName+"</small>":"")+"</th>";
  nos.forEach(function(no){ h+="<th>"+tix[no].label+"<small>"+no+
    " <a href='#' class='tixdel' data-no='"+no+"'>remove</a></small></th>"; });
  h+="</tr></thead><tbody>";
  legs.forEach(function(l){
    h+="<tr><th>"+l+"</th>"+cell(TM.model[l]||[],l,TM.spikNames[l]||null);
    nos.forEach(function(no){ h+=cell(tix[no].picks[l]||[],l,null); });
    h+="</tr>";
  });
  h+="</tbody>";
  if(scored){
    h+="<tfoot><tr><td>Legs hit</td>";
    var cols=[TM.model].concat(nos.map(function(no){return tix[no].picks;}));
    cols.forEach(function(pk){ var hit=0;
      legs.forEach(function(l){ var w=TM.winners[l];
        if(w!==undefined&&(pk[l]||[]).map(Number).indexOf(Number(w))>=0) hit++; });
      h+="<td><b>"+hit+" of "+legs.length+"</b></td>"; });
    h+="</tr></tfoot>";
  }
  h+="</table>";
  sec.innerHTML=h;
};
document.addEventListener("DOMContentLoaded", renderCmp);
if(document.readyState!=="loading") renderCmp();
})();
</script>
"""

TIX_CSS = """
  .upbtn{ margin-top:10px; margin-left:8px; background:none; border:1px solid var(--line);
    border-radius:7px; padding:5px 12px; font:600 12px/1.2 "Avenir Next","Seravek",system-ui,sans-serif;
    color:var(--ink); cursor:pointer; }
  .upbtn:hover{ border-color:var(--pick); color:var(--pick); }
  .modalback{ display:none; position:fixed; inset:0; background:rgba(0,0,0,.45); z-index:80;
    align-items:flex-start; justify-content:center; overflow-y:auto; padding:5vh 16px; }
  .modal{ background:var(--card); color:var(--ink); border-radius:12px; padding:20px 22px;
    width:min(430px,94vw); }
  .modal h3{ margin:0 0 4px; }
  .modal label{ display:block; font-size:10.5px; letter-spacing:.08em; color:var(--muted);
    margin:10px 0 3px; text-transform:uppercase; }
  .modal input[type=text]{ width:100%; padding:7px 10px; border:1px solid var(--line);
    border-radius:6px; background:var(--bg); color:var(--ink); font:inherit; }
  .modal .mnote{ font-size:11.5px; color:var(--muted); margin:2px 0 0; }
  .modal .mnote.err{ color:#C23B2E; }
  #tixpreview{ max-width:100%; max-height:180px; margin-top:6px; border-radius:6px; }
  .mrow{ display:flex; gap:10px; margin-top:16px; }
  .mbtn{ flex:1; padding:9px; border-radius:7px; border:1px solid var(--line);
    background:none; color:var(--ink); font:600 13px/1 "Avenir Next",system-ui,sans-serif; cursor:pointer; }
  .mbtn.save{ background:var(--pick); border-color:var(--pick); color:#fff; }
  .cmp{ display:none; margin-top:40px; }
  .cmp.hasusr{ display:block; }
  .cmp table{ width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; }
  .cmp th,.cmp td{ padding:7px 10px; text-align:left; font-size:13.5px; border-bottom:1px solid var(--line); }
  .cmp thead th{ background:var(--head-bg); color:var(--head-ink); font-size:11px; letter-spacing:.05em; }
  .cmp thead th small{ display:block; font-weight:400; opacity:.7; font-size:9.5px; }
  .cmp thead th small a{ color:inherit; }
  .cmp tbody th{ width:40px; font-weight:700; }
  .cmp tfoot td{ font-size:13px; border-top:2px solid var(--ink); }
  .hitnum{ display:inline-block; min-width:1.5em; text-align:center; border:2px solid var(--pick);
    border-radius:50%; color:var(--pick); font-weight:700; padding:0 3px; }
  .lostnum{ color:var(--muted); text-decoration-color:#C23B2E; }
  .misscell{ color:#C23B2E; }
  footer{ margin-top:64px !important; }
"""

# ---------- model reasoning ----------

def _phrase(name, r):
    import math
    if name == "log_implied":
        return None  # market view already visible in the streck/odds columns
    if name == "avg_log_odds_c":
        avg = math.exp(r["avg_log_odds_c"])
        return f"consistently short odds recently (~{avg:.1f})" if avg <= 6 else None
    if name == "drv_winpct_py":
        v = 100 * r["drv_winpct_py"]
        return f"driver wins {v:.0f}% of his races" if v >= 12 else None
    if name == "trn_winpct_py":
        v = 100 * r["trn_winpct_py"]
        return f"trainer wins {v:.0f}%" if v >= 12 else None
    if name == "best_speedfig_c":
        v = r["best_speedfig_c"]
        return f"top speed figure ({v:+.1f}s vs a typical winner)" if v <= -1 else None
    if name == "wins_5":
        n = int(r["wins_5"])
        return f"{n} wins in last 5 starts" if n >= 2 else None
    if name == "top3_5":
        n = int(r["top3_5"])
        return f"{n} top-3 finishes in last 5" if n >= 3 else None
    if name == "avg_place_5_c":
        v = r["avg_place_5_c"]
        return f"average finish {v:.1f} in last 5" if v <= 3.5 else None
    if name == "log_money_per_start":
        return "high career earnings per start"
    if name == "log_avg_purse_c":
        return "has raced at higher purse levels (class edge)"
    if name == "shoe_change":
        return "shoe change tonight" if r.get("shoe_change") else None
    if name == "sulky_change":
        return "cart change tonight" if r.get("sulky_change") else None
    return None


def gen_reasons(r, contrib_row, cols):
    """Top positive feature contributions for this horse, phrased; plus cautions."""
    out = []
    for name, c in sorted(zip(cols, contrib_row), key=lambda x: -x[1]):
        if c <= 0.02 or len(out) == 3:
            break
        ph = _phrase(name, r)
        if ph:
            out.append(ph)
    cautions = []
    if r.get("gallops_5", 0) >= 2:
        cautions.append(f"{int(r['gallops_5'])} gallops in last 5")
    if r.get("days_since") and r["days_since"] > 45:
        cautions.append(f"{int(r['days_since'])}-day layoff")
    txt = " · ".join(out)
    if cautions:
        txt += (" — caution: " if txt else "caution: ") + ", ".join(cautions)
    return txt


# ---------- rendering ----------

CSS = """
  :root{ --bg:#FAF8F3; --ink:#26241F; --muted:#867F73; --line:#E3DED4;
    --pick:#2E6B4A; --pick-bg:#EDF3EE; --head-bg:#26241F; --head-ink:#FAF8F3;
    --exp:#8A5A1E; --card:#FFFFFF; --acc2:#46647F; --fam:#B45062; --fam-bg:#F7E7EA; }
  @media (prefers-color-scheme: dark){ :root{ --bg:#181A16; --ink:#E9E6DD; --muted:#98937F;
    --line:#37392F; --pick:#8CC3A4; --pick-bg:#26312A; --head-bg:#E9E6DD; --head-ink:#1D1F1B;
    --exp:#D4A860; --card:#22241F; --acc2:#93B4CE; --fam:#E39AAB; --fam-bg:#3A2A2E; } }
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
  .legrow{ display:contents; }
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
  th:nth-child(4),td:nth-child(4),th:nth-child(5),td:nth-child(5){ text-align:right; }
  td.drv{ color:var(--muted); font-size:11px; }
  tbody tr{ border-bottom:1px solid var(--line); }
  td.nr{ color:var(--muted); width:22px; }
  td.num{ white-space:nowrap; } td.strong{ font-weight:600; }
  tr.top td{ background:var(--pick-bg); }
  tr.top td:nth-child(2){ color:var(--pick); font-weight:700; }
  .flag{ font-size:9px; font-weight:700; letter-spacing:.08em; border-radius:3px; padding:0 4px; }
  .flag.value{ color:var(--pick); border:1px solid currentColor; }
  .flag.over{ color:var(--exp); border:1px solid currentColor; }
  .infos{ margin-top:8px; }
  p.sechead{ font-size:9.5px; line-height:1.3; font-weight:700; letter-spacing:.16em;
    margin:9px 0 3px; padding-bottom:2px; border-bottom:1px solid var(--line);
    text-transform:uppercase; }
  p.sechead.tr, p.sechead.model{ color:var(--exp); }
  .info{ margin:0 0 5px; font-size:11px; color:var(--muted); line-height:1.5; }
  .info b{ color:var(--pick); }
  .tdrawer{ position:fixed; top:0; right:0; height:100vh; z-index:60;
    transform:translateX(100%); transition:transform .25s ease; }
  .tdrawer.open{ transform:translateX(0); }
  @media (prefers-reduced-motion: reduce){ .tdrawer{ transition:none; } }
  .ttab{ position:absolute; left:-40px; top:36%; width:40px;
    writing-mode:vertical-rl; background:var(--pick); color:#fff; border:none;
    border-radius:8px 0 0 8px; padding:14px 10px; cursor:pointer;
    font:700 11px/1.2 "Avenir Next","Seravek",system-ui,sans-serif; letter-spacing:.16em; }
  .slipd{ width:min(300px, 86vw); height:100%; overflow-y:auto;
    background:#FCFAF7; color:#211E1B; padding:24px 18px 30px;
    font-family:ui-monospace,'SF Mono',Menlo,Consolas,'Courier New',monospace;
    font-size:12.5px; line-height:1.5; font-variant-numeric:tabular-nums;
    box-shadow:-14px 0 34px rgba(0,0,0,.3); }
  .dlogo{ font-size:18px; font-weight:700; letter-spacing:.16em; text-align:center; margin:0; }
  .dlogo span{ font-weight:400; }
  .dtag{ font-size:8.5px; color:#8A837A; letter-spacing:.06em; text-align:center; margin:3px 0 0; }
  .drule{ border:none; border-top:1.5px dashed #8A837A; opacity:.65; margin:10px 0; }
  .drule.solid{ border-top:2px solid #211E1B; opacity:1; }
  .drow{ display:flex; justify-content:space-between; gap:10px; padding:1.5px 0; }
  .drow span{ color:#8A837A; }
  .dlegs{ width:100%; border-collapse:collapse; }
  .dlegs th{ font-size:8.5px; letter-spacing:.14em; color:#8A837A; text-align:left;
    font-weight:400; padding-bottom:4px; }
  .dlegs td{ padding:2.5px 0; }
  .dlegs td.avd{ font-weight:700; width:30px; }
  .dlegs td.hst{ font-weight:700; }
  .dtotal{ display:flex; justify-content:space-between; border-top:2px solid #211E1B;
    margin-top:6px; padding-top:6px; font-weight:700; font-size:15px; }
  .dnote{ font-size:9.5px; color:#8A837A; line-height:1.55; margin:8px 0 0; }
  .dstamp{ width:fit-content; margin:14px auto 0; border:2.5px solid #C23B2E; color:#C23B2E;
    padding:5px 12px; font-weight:700; font-size:11px; letter-spacing:.12em;
    transform:rotate(-5deg); border-radius:4px; text-align:center; opacity:.85; }
  .dstamp small{ display:block; font-size:7.5px; letter-spacing:.16em; font-weight:400; }
  @media print{ .tdrawer{ display:none; } }
  .fambadge{ background:var(--fam-bg); color:var(--fam); border-radius:99px;
    font-size:10px; font-weight:700; letter-spacing:.1em; padding:2px 9px;
    margin-left:auto; white-space:nowrap; }
  .famtag{ color:var(--fam); font-size:9px; font-weight:700; letter-spacing:.08em;
    border:1px solid currentColor; border-radius:3px; padding:0 4px; }
  .tile.famtile{ border-color:var(--fam); border-width:2px; }
  tr.fam td{ background:var(--fam-bg); }
  tr.fam td:nth-child(2){ color:var(--fam); font-weight:700; }
  .gamecard.famcard{ border-color:var(--fam); border-width:2px; }
  .secbar{ font-size:11px; font-weight:700; letter-spacing:.18em; color:var(--muted);
    margin:56px 0 10px; border-bottom:1px solid var(--line); padding-bottom:5px; }
  .gamecard.pastcard{ opacity:.85; }
  .gamecard.pastcard .fresh{ color:var(--exp); }
  .gamecard.famcard:hover{ border-color:var(--fam); }
  .printbtn{ margin-top:10px; background:none; border:1px solid var(--line); border-radius:7px;
    padding:5px 12px; font:600 12px/1.2 "Avenir Next","Seravek",system-ui,sans-serif;
    color:var(--ink); cursor:pointer; }
  .printbtn:hover{ border-color:var(--pick); color:var(--pick); }
  .printslip{ display:none; }
  .cards{ display:flex; flex-direction:column; gap:14px; }
  .gamecard{ display:block; text-decoration:none; background:var(--card);
    border:1px solid var(--line); border-radius:12px; padding:16px 18px; }
  .gamecard:hover{ border-color:var(--pick); }
  .gamecard .row1{ display:flex; justify-content:space-between; align-items:baseline; gap:10px;
    flex-wrap:wrap; position:relative; }
  .gamecard .row1 .fambadge{ position:absolute; left:50%; top:50%;
    transform:translate(-50%,-50%); margin-left:0; }
  @media (max-width:640px){
    .gamecard .row1 .fambadge{ position:static; transform:none; } }
  .gamecard .gt{ font-size:20px; font-weight:700; }
  .gamecard .when{ font-size:14px; font-variant-numeric:tabular-nums; color:var(--muted); }
  .gamecard .row2{ display:flex; justify-content:space-between; gap:10px; flex-wrap:wrap;
    margin-top:4px; font-size:12.5px; color:var(--muted); }
  .fresh{ color:var(--pick); font-weight:600; }
  .tl{ font-size:9px; font-weight:700; letter-spacing:.14em; color:var(--muted);
    margin-right:6px; vertical-align:1px; }
  footer{ margin-top:24px; padding-top:10px; border-top:1px solid var(--line);
    font-size:10.5px; color:var(--muted); line-height:1.6; }
  .legend-title{ font-size:9.5px; font-weight:700; letter-spacing:.16em; color:var(--exp);
    margin:0 0 6px; }
  dl.legend{ margin:0 0 12px; display:grid; grid-template-columns:auto 1fr; gap:5px 12px;
    font-size:11px; }
  dl.legend dt{ font-weight:700; white-space:nowrap; color:var(--ink); }
  dl.legend dd{ margin:0; line-height:1.55; }
  .userchip{ text-align:right; font-size:12px; color:var(--muted); margin:-20px 0 8px; }
  .userchip a{ color:var(--pick); }
  @media print{ .userchip{ display:none; } }
  @media print{
    :root{ --muted:#3A3733; --pick:#1E4A33; --exp:#6B4413; --line:#999; }
    body{ background:#fff; color:#000; padding:0; }
    main{ max-width:none; } .grid{ display:block; }
    .legrow{ display:grid; grid-template-columns:1fr 1fr; gap:10px; align-items:start;
      page-break-after:always; break-inside:avoid; page-break-inside:avoid; margin-bottom:0; }
    .tile{ border-color:#888; break-inside:avoid; page-break-inside:avoid; }
    .printbtn,.upbtn,.modalback{ display:none; } footer{ display:none; } body{ font-size:10px; }
    .cmp{ display:block !important; break-before:page; padding-top:24px; }
    .cmp thead th{ background:#000 !important; color:#fff !important;
      -webkit-print-color-adjust:exact; print-color-adjust:exact; }
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
            if h.get("family"):
                cls += " fam"
                flag = " <span class='famtag'>♥ JANS HÄST</span>" + flag
            rows.append(f"<tr class='{cls}'><td class='nr'>{h['nr']}</td>"
                        f"<td>{esc(h['horse'])}{flag}</td>"
                        f"<td class='drv'>{esc(h['driver'])}</td>"
                        f"<td class='num'>{h['streck']:.1f}%</td>"
                        f"<td class='num strong'>{h['model']:.1f}%</td></tr>")
        tr_notes = [f"<p class='info'><b>#{h['nr']} {esc(h['horse'])}</b> "
                    f"({esc(h['driver'])}) — {esc(translate(h['comment']))}</p>"
                    for h in horses[:3] if h["comment"]]
        mr_notes = [f"<p class='info'><b>#{h['nr']} {esc(h['horse'])}</b> "
                    f"({100*0+h['model']:.0f}%) — {esc(h['reasons'])}</p>"
                    for h in horses[:2] if h.get("reasons")]
        if tr_notes:
            infos.append("<p class='sechead tr'>TR MEDIA</p>" + "".join(tr_notes))
        if mr_notes:
            infos.append("<p class='sechead model'>MODEL REASONING</p>" + "".join(mr_notes))
        spik = " · ★ spik candidate" if horses and horses[0]["model"] > 45 else ""
        fam_in_leg = any(h.get("family") for h in horses)
        fambadge = "<span class='fambadge'>♥ PRALINES</span>" if fam_in_leg else ""
        tiles.append(f"""<article class="tile{' famtile' if fam_in_leg else ''}">
<div class="leghead"><h2>Leg {leg}</h2><span class="meta">{esc(data['legmeta'].get(leg,''))}{spik}</span>{fambadge}</div>
<table><thead><tr><th>#</th><th>Horse</th><th>Driver</th><th>Streck</th><th>Model</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<div class="infos">{''.join(infos)}</div></article>""")
    ticket = build_ticket(data["legs"], data["type"])
    tro = []
    for leg in sorted(ticket["picks"]):
        if leg in ticket["spiks"]:
            sp = ticket["spiks"][leg]
            heart = " ♥" if sp.get("family") else ""
            val = f"{sp['nr']} {esc(sp['horse'])} ★{heart}"
        else:
            val = ", ".join(str(n) for n in ticket["picks"][leg])
        tro.append(f"<tr><td class='avd'>{leg}</td><td class='hst'>{val}</td></tr>")
    mult = "×".join(str(len(ticket["picks"][leg])) for leg in sorted(ticket["picks"]))
    price_str = f"{ticket['price']:.2f}".rstrip("0").rstrip(".")
    fam_note = "".join(f"♥ FAMILJEREGEL: {esc(h['horse'])} är Jans häst — alltid spik, oavsett statistik. "
                       for h in ticket["spiks"].values() if h.get("family"))
    slip_inner = f"""
  <p class="dlogo">TRAVMODEL<span>™</span></p>
  <p class="dtag">TEAM LILLIAN × CLAUDE · FAMILJENS EGET SPELBOLAG</p>
  <hr class="drule solid">
  <div class="drow"><span>Spel</span><b>{data['type']}</b></div>
  <div class="drow"><span>Bana</span><b>{esc(data['track'])}</b></div>
  <div class="drow"><span>Start</span><b>{start_dt.strftime('%a %d %b · %H:%M')}</b></div>
  <div class="drow"><span>Data</span><b>{updated}</b></div>
  <hr class="drule">
  <table class="dlegs"><tr><th>AVD</th><th>HÄSTAR</th></tr>{''.join(tro)}</table>
  <hr class="drule">
  <div class="drow"><span>{mult}</span><b>= {ticket['rows']} rader</b></div>
  <div class="drow"><span>{ticket['rows']} × {price_str} kr</span><b>{ticket['cost']:.2f} kr</b></div>
  <div class="dtotal"><span>TOTALT</span><b>{ticket['cost']:.2f} kr</b></div>
  <p class="dnote">{fam_note}Modellens egen chans: ~{100*ticket['hit_all']:.0f}% att pricka alla {len(ticket['picks'])}.
  Byggs om vid varje datauppdatering — kupongen kan ändras. ★ = spik.
  Inget giltigt spel — spela hos atg.se om du vill.</p>
  <div class="dstamp">EJ GILTIGT SPEL<small>ENDAST SKRYTRÄTTIGHETER</small></div>
"""
    ticket_html = ("""<aside class="tdrawer" id="tdrawer">\n<button class="ttab" onclick="tdT()">🎟️ KUPONG</button>\n<div class="slipd">""" + slip_inner + '</div></aside>') + """<script>
function tdT(){var d=document.getElementById('tdrawer');var o=d.classList.toggle('open');
try{localStorage.setItem('tm_drawer',o?'1':'0');}catch(e){}}
try{if(localStorage.getItem('tm_drawer')==='1')document.getElementById('tdrawer').classList.add('open');}catch(e){}
</script>"""
    import json as _json
    spik_names = {str(leg): f"{sp['nr']} {esc(sp['horse'])} \u2605" + (" \u2665" if sp.get("family") else "")
                  for leg, sp in ticket["spiks"].items()}
    tmdata = _json.dumps({
        "game": game["id"], "modelName": f"{ticket['rows']} rader · {ticket['cost']:.0f} kr",
        "legs": {str(leg): [h["nr"] for h in data["legs"][leg]] for leg in data["legs"]},
        "model": {str(leg): ticket["picks"][leg] for leg in ticket["picks"]},
        "spikNames": spik_names, "winners": {},
    }, ensure_ascii=False)
    tix_block = TIX_HTML + TIX_JS.replace("__TMDATA__", tmdata)
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="600">
{auth_guard("../login.html")}
<title>{data['type']} {esc(data['track'])} {start_dt.strftime('%a %d %b %H:%M')} — Lillian's Model</title>
<style>{CSS}{TIX_CSS}</style></head><body><main>
{USERCHIP}
<div class="pagehead"><div>
<h1><a href="../index.html">←</a> {data['type']} {esc(data['track'])} · {start_dt.strftime('%A %d %B %Y')}</h1>
<p class="sub">First start {start_dt.strftime('%H:%M')} · streck (share of tickets) vs Lillian's Model
(win probability) · green rows = model's top of the leg · program comments under each table</p>
<button class="printbtn" onclick="window.print()">🖨️&nbsp; Skriv ut / Print</button><button class="upbtn" onclick="tixOpen()">📷&nbsp; Upload My Ticket</button>
</div>{stampbox(updated, start_dt.strftime('%H:%M'))}</div>
{ticket_html}
<div class="grid">{''.join('<div class="legrow">' + ''.join(tiles[i:i+2]) + '</div>' for i in range(0, len(tiles), 2))}</div>
{tix_block}
<footer>
<p class="legend-title">WHAT THE LABELS MEAN</p>
<dl class="legend">
<dt><span class='flag value'>VALUE</span></dt>
<dd>The crowd likes this horse <em>less</em> than it deserves — the model's win chance is at least
1.3× its share of the betting pool. A better deal than its price, not a guaranteed winner.</dd>
<dt><span class='flag over'>OVERBET</span></dt>
<dd>The crowd likes this horse <em>more</em> than it deserves — the model's chance is under 0.7×
its betting share (flagged only when 15%+ of the money is on it). Often a decent horse at a bad price.</dd>
<dt>★ spik</dt>
<dd>Banker candidate: the leg's top horse is strong enough (45%+ model chance) to carry the leg alone
— passes the classic rule "only bank a horse whose real chance ≥ its betting percentage".</dd>
<dt>♥</dt>
<dd>Family rule: Pralines is Jan's own horse. When she races she is always the spik on the kupong,
whatever the statistics say.</dd>
<dt>Green row</dt>
<dd>The model's top of the leg (top two when the leg is open).</dd>
<dt>Streck vs Model</dt>
<dd>Streck = share of all tickets that include the horse. Model = win probability from Lillian's Model.
The gap between them is where the flags come from.</dd>
</dl>
<p>Lillian's Model = travmodel v2 · conditional logit on 17,356 Nordic races · market-blended ·
data from ATG's open API · page auto-reloads every 10 min · not betting advice, not a valid bet.</p>
</footer>
</main></body></html>"""


def render_index(entries):
    past_cards = []
    for pr in PAST_RACES:
        past_cards.append(f"""<a class="gamecard pastcard" href="{pr['href']}">
<div class="row1"><span class="gt">{pr['type']} · {esc(pr['track'])}</span>
<span class="when"><span class="tl">FINISHED</span>{pr['finished']}</span></div>
<div class="row2"><span>{esc(pr['note'])}</span>
<span class="fresh">{esc(pr['outcome'])}</span></div></a>""")
    cards = []
    for e in entries:
        start_dt = datetime.fromisoformat(e["start"])
        upd = e.get("updated", "not yet")
        fam = e.get("fam")
        fambadge = "<span class='fambadge'>♥ PRALINES STARTAR</span>" if fam else ""
        cards.append(f"""<a class="gamecard{' famcard' if fam else ''}" href="game/{e['id']}.html">
<div class="row1"><span class="gt">{e['type']} · {esc(e.get('track','...'))}</span>{fambadge}
<span class="when"><span class="tl">RACE STARTS</span>{start_dt.strftime('%A %d %b · %H:%M')}</span></div>
<div class="row2"><span>{e.get('nlegs','?')} legs · model + streck sheet</span>
<span class="fresh"><span class="tl">DATA UPDATED</span>{upd}</span></div></a>""")
    now = datetime.now().strftime("%a %d %b · %H:%M")
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="300">
{auth_guard("login.html")}
<title>Lillian's Model — next races</title><style>{CSS}{TIX_CSS}</style></head><body><main>
{USERCHIP}
<div class="pagehead"><div>
<h1>Lillian's Model 🐴</h1>
<p class="sub">The next {len(entries)} betting rounds, scored by travmodel v2. Pages update hourly —
every 30 minutes inside the final 4 hours before post.</p>
</div><div class="stampbox"><span class="stamp-label">PAGE GENERATED</span>
<span class="stamp-time">{now}</span>
<span class="stamp-note">each race card shows its own data snapshot</span></div></div>
<div class="cards">{''.join(cards)}</div>
<p class="secbar">PAST RACES</p>
<div class="cards">{''.join(past_cards)}</div>
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
                      "nlegs": len(data["legs"]), "updated": updated,
                      "fam": bool(data.get("fam_legs"))}
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
