"""Translate TR Media's formulaic Swedish program comments to English.

These day-comments are generated from a small set of stat templates, so an
ordered phrase glossary translates them faithfully. Anything unmatched is
left as-is (better a Swedish fragment than a wrong translation).
"""
import re

# ordered: longest / most specific first
PHRASES = [
    ("har snabbaste rekordtiden i fältet i Sverige över aktuell distans med bilstart",
     "has the fastest record in the field in Sweden over tonight's distance with auto start"),
    ("har snabbaste rekordtiden i fältet i Sverige över aktuell distans med voltstart",
     "has the fastest record in the field in Sweden over tonight's distance with volt start"),
    ("har högst segerprocent bland tränarna i fältet senaste året",
     "has the highest trainer win-rate in the field this past year"),
    ("har lägst segerprocent i Sverige bland kuskarna i fältet senaste året",
     "has the lowest driver win-rate in Sweden in this field over the past year"),
    ("har tjänat mest pengar i startfältet", "has earned the most money in the field"),
    ("har tjänat minst pengar i startfältet", "has earned the least money in the field"),
    ("gånger på sommaren och varit bland de tre främsta i",
     "times this summer and finished top-3 in"),
    ("gånger på sommaren och vunnit", "times this summer and won"),
    ("har varit bland de tre främsta i", "has finished top-3 in"),
    ("och varit bland de tre främsta i", "and finished top-3 in"),
    ("som favoritspelad på de större streckspelen",
     "as the bet-down favorite in the big pool games"),
    ("uppsittningar den senaste månaden i Sverige", "drives in Sweden in the last month"),
    ("uppsittningar senaste året i Sverige", "drives in Sweden in the last year"),
    ("uppsittningar", "drives"),
    ("tränarsegrar på", "trainer wins from"),
    ("när det som nu gått minst en månad mellan loppen",
     "when, as now, at least a month between races"),
    ("när det som nu gått", "when, as now,"),
    ("dagar mellan loppen", "days between races"),
    ("i lopp med som nu högst", "in races worth, as tonight, at most"),
    ("kronor i förstapris", "kronor in first prize"),
    ("har galopperat i", "has galloped in"),
    ("är obesegrad på", "is unbeaten in"),
    ("har startat", "has started"),
    ("i nuvarande regi", "for the current stable"),
    ("den senaste månaden", "in the last month"),
    ("senaste året", "in the last year"),
    ("segrar på", "wins from"),
    ("seger på", "win from"),
    ("försök i spets", "attempts in the lead"),
    ("starter i spets", "starts in the lead"),
    ("starter från tillägg", "starts from a handicap mark"),
    ("försök från tillägg", "attempts from a handicap mark"),
    ("från tillägg", "from a handicap mark"),
    ("bakom bilen", "behind the starting gate"),
    ("i sulkyn", "in the sulky"),
    ("barfota runt om", "barefoot all around"),
    ("barfota fram", "barefoot in front"),
    ("barfota bak", "barefoot behind"),
    ("över aktuell distans", "over tonight's distance"),
    ("i Sverige", "in Sweden"),
    ("i spets", "in the lead"),
    ("av dessa", "of them"),
    ("från spår", "from post"),
    ("starter på", "starts at"),
    ("starter", "starts"),
    ("försök", "attempts"),
]

WORDS = {
    "har": "has", "är": "is", "och": "and", "med": "with", "utan": "without",
    "av": "of", "noll": "zero", "gånger": "times",
    "en": "one", "ett": "one", "två": "two", "tre": "three", "fyra": "four",
    "fem": "five", "sex": "six", "sju": "seven", "åtta": "eight", "nio": "nine",
    "tio": "ten", "elva": "eleven", "tolv": "twelve", "tretton": "thirteen",
    "fjorton": "fourteen", "femton": "fifteen", "sexton": "sixteen",
    "sjutton": "seventeen", "arton": "eighteen", "nitton": "nineteen",
    "tjugo": "twenty",
}


def translate(text):
    if not text:
        return text
    out = text
    for sv, en in PHRASES:
        out = out.replace(sv, en)
    def repl(m):
        w = m.group(0)
        t = WORDS.get(w.lower())
        return (t.capitalize() if w[0].isupper() else t) if t else w
    out = re.sub(r"\b[a-zA-ZåäöÅÄÖ]+\b", repl, out)
    return out
