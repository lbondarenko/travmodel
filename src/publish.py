"""One-shot site publisher for GitHub Actions (or any cron).

Regenerates docs/ (GitHub Pages source):
  - fetches the next 3 upcoming rounds
  - re-scores a round if its data is older than the allowed interval:
      > 4 h to post  -> refresh if older than ~55 min
      <= 4 h to post -> refresh if older than ~25 min
    (margins absorb cron jitter; the workflow itself runs every 30 min)
  - always rewrites the index

State (last update time per game) lives in docs/state.json so it
persists between workflow runs via the git commit.

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


def main():
    force = "--force" in sys.argv
    webapp.WEB = DOCS  # reroute output
    DOCS.mkdir(parents=True, exist_ok=True)
    state = {}
    if STATE_FILE.exists():
        state = json.loads(STATE_FILE.read_text())
    webapp.STATE.update(state)

    games = webapp.upcoming_games(3)
    now = datetime.now(TZ)
    for g in games:
        start = datetime.fromisoformat(g["start"]).replace(tzinfo=TZ)
        to_post = (start - now).total_seconds()
        max_age = CLOSE_MAX_AGE if to_post <= CLOSE_WINDOW else FULL_MAX_AGE
        last = webapp.STATE.get(g["id"], {}).get("last", 0)
        if force or time.time() - last >= max_age:
            webapp.update_game(g)
        else:
            webapp.log(f"skip {g['id']} (fresh, {int((time.time()-last)/60)} min old, "
                       f"{int(to_post/3600)}h to post)")

    entries = []
    for g in games:
        entries.append({**g, **webapp.STATE.get(g["id"], {})})
    (DOCS / "index.html").write_text(webapp.render_index(entries))
    (DOCS / ".nojekyll").write_text("")
    STATE_FILE.write_text(json.dumps(
        {k: v for k, v in webapp.STATE.items()}, default=str))
    webapp.log(f"index written with {len(entries)} rounds")


if __name__ == "__main__":
    main()
