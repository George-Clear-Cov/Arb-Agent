from __future__ import annotations

"""
Cross-source market matching — production rewrite.

Improvements over previous version:
  1. rapidfuzz (C++ SIMD) replaces fuzzywuzzy — 10-100x faster, drop-in API
  2. Candidate-name fast path: extract short name from yes_subtitle / colon-split /
     Polymarket outcome label, match on "Gavin Newsom" vs "Gavin Newsom" instead of
     matching full 80-char question strings
  3. Filler stripping: "Will X win the 2028 Presidential Election?" → "x"
     so platform boilerplate doesn't tank similarity scores
  4. Structured election key (year / office / party / state) as hard gate before
     any fuzzy scoring — eliminates false positives cheaply
  5. Inverted proper-noun index for prediction buckets: O(n) average case instead of
     O(n²) full pairwise scan
  6. Correct bucket routing: "binary" prediction markets (Kalshi/Polymarket/PredictIt)
     stay in the "prediction" bucket — not mistakenly routed to "game" bucket where
     team-name expansion corrupts political titles ("Arizona" → "arizona diamondbacks")
  7. In-memory match cache with 2-hour TTL: zero re-work within polling cycle

Sources: taetaehoho/poly-kalshi-arb, ImMike/polymarket-arbitrage,
         realfishsam/prediction-market-arbitrage-bot
"""

import re
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from rapidfuzz import fuzz

from src.models import Market, Source

# ── Thresholds ───────────────────────────────────────────────────────────────
_THRESHOLD_CANDIDATE  = 82   # short name comparison ("Gavin Newsom" vs "Gavin Newsom")
_THRESHOLD_GAME       = 70   # team/game name comparison — fallback when no LLM
_THRESHOLD_GAME_HIGH  = 85   # auto-match without LLM (clear hit)
_THRESHOLD_GAME_BORDER = 52  # LLM verify band: [BORDER, HIGH)
_THRESHOLD_PREDICTION      = 78   # fuzzy fallback when LLM is unavailable
_THRESHOLD_PRED_HIGH       = 90   # auto-match without LLM (near-identical wording)
_THRESHOLD_PRED_BORDER     = 60   # LLM verify band: [BORDER, HIGH)

# ── In-memory match-pair cache (2-hour TTL) ───────────────────────────────────
_CACHE: dict[tuple[str, str], bool] = {}
_CACHE_TS: float = 0.0
_CACHE_TTL = 7200.0

# ── LLM market matcher (optional — enabled when ANTHROPIC_API_KEY is set) ─────
_llm_matcher: "Optional[LLMMarketMatcher]" = None


def init_llm_matcher(api_key: str) -> None:
    """Call once at startup to enable Claude-powered borderline pair verification."""
    global _llm_matcher
    from src.engine.llm_matcher import LLMMarketMatcher
    _llm_matcher = LLMMarketMatcher(api_key=api_key)
    import logging as _log
    _log.getLogger(__name__).info(
        "LLM market matcher enabled (Claude Sonnet) — borderline pairs [%d–%d] sent to Claude",
        _THRESHOLD_GAME_BORDER, _THRESHOLD_GAME_HIGH,
    )


def _cache_key(a: Market, b: Market) -> tuple[str, str]:
    x, y = a.market_id, b.market_id
    return (x, y) if x <= y else (y, x)


def _maybe_expire_cache() -> None:
    global _CACHE_TS
    if time.monotonic() - _CACHE_TS > _CACHE_TTL:
        _CACHE.clear()
        _CACHE_TS = time.monotonic()


# ── Source sets ───────────────────────────────────────────────────────────────
PREDICTION_SOURCES = {
    Source.KALSHI,
    Source.POLYMARKET,
    Source.PREDICTIT,
    Source.OPINION,
    Source.PREDICTFUN,
    Source.MANIFOLD,
}
SPORTS_SOURCES = {Source.KALSHI_SPORTS}

# Market types that represent concrete game outcomes (h2h / totals / spreads).
# NOTE: "binary" is intentionally excluded here — prediction exchanges use "binary"
# for all yes/no markets including political questions.  It must not force-route
# political markets into the "game" bucket.  See _bucket_key for the full logic.
_SPORTS_GAME_TYPES: frozenset[str] = frozenset({
    "h2h", "match_odds", "match odds", "moneyline", "1x2", "winner", "to win",
    "totals", "over_under", "total_goals", "over/under",
    "spreads", "asian_handicap", "handicap", "run_line", "puck_line",
})

# All market types that can appear in a game bucket (binary allowed only for sports sources
# or prediction sources with a non-prediction sport tag)
_GAME_MARKET_TYPES_SET: frozenset[str] = _SPORTS_GAME_TYPES | frozenset({"binary"})

_SPORTS_ALIASES: list[set[str]] = [
    {"h2h", "match_odds", "match odds", "moneyline", "1x2", "winner", "to win", "binary"},
    {"totals", "over_under", "total_goals", "over/under"},
    {"spreads", "asian_handicap", "handicap", "run_line", "puck_line"},
    {"h2h_lay"},
]


# ── Bucket key ────────────────────────────────────────────────────────────────

def _alias(market_type: str) -> str:
    mt = market_type.lower()
    for group in _SPORTS_ALIASES:
        if mt in group:
            return min(group)
    return mt


def _bucket_key(m: Market) -> tuple:
    """Compute the (category, sport, market_type_alias, sub_market) bucket for a market.

    Critical rule: prediction-source markets with type "binary" are ALWAYS
    prediction markets even though "binary" is technically in _GAME_MARKET_TYPES_SET.
    Only route to "game" bucket when:
      - Source is a sports source (ESPN/DK, OddsAPI, KalshiSports, Betfair), OR
      - Source is a prediction exchange but sport is NOT "prediction"
        (e.g. Polymarket MLB game, Kalshi sports binary)

    Sub-market tag: isolates period/set/half markets so "match winner" is never
    compared against "2nd set winner" or "1st half" odds even when they share
    identical player/team names.
    """
    alias = _alias(m.market_type)
    mt = m.market_type.lower()

    if m.source in PREDICTION_SOURCES:
        # For prediction exchanges: only game-bucket if sport tag indicates a real sport
        # "prediction" sport = political/economic/general — stays in prediction bucket
        is_game = mt in _SPORTS_GAME_TYPES or (mt == "binary" and m.sport != "prediction")
        cat = "game" if is_game else "prediction"
    elif m.source in SPORTS_SOURCES:
        cat = "game" if mt in _GAME_MARKET_TYPES_SET else "sportsbook"
    else:
        cat = "sportsbook"

    sub = _sub_market_tag(m.event_name) if cat == "game" else "full"
    return (cat, m.sport, alias, sub)


# ── Structured election key ───────────────────────────────────────────────────

_OFFICE_MAP: list[tuple[str, str]] = sorted([
    ("vice president",    "vp"),
    ("vice presidential", "vp"),
    ("presidential",      "president"),
    ("presidency",        "president"),
    ("president",         "president"),
    ("senatorial",        "senator"),
    ("senator",           "senator"),
    ("senate",            "senator"),
    ("gubernatorial",     "governor"),
    ("governorship",      "governor"),
    ("governor",          "governor"),
    ("congressional",     "representative"),
    ("representative",    "representative"),
    ("house",             "representative"),
    ("prime minister",    "pm"),
    ("mayor",             "mayor"),
], key=lambda x: -len(x[0]))  # longest first to avoid partial matches

_PARTY_MAP: dict[str, str] = {
    "democratic": "dem", "democrat": "dem", "dnc": "dem",
    "republican": "rep", "gop": "rep",    "rnc": "rep",
}

_YEAR_RE = re.compile(r'\b(20[2-4]\d)\b')

_US_STATES = {
    "alabama","alaska","arizona","arkansas","california","colorado","connecticut",
    "delaware","florida","georgia","hawaii","idaho","illinois","indiana","iowa",
    "kansas","kentucky","louisiana","maine","maryland","massachusetts","michigan",
    "minnesota","mississippi","missouri","montana","nebraska","nevada",
    "new hampshire","new jersey","new mexico","new york","north carolina",
    "north dakota","ohio","oklahoma","oregon","pennsylvania","rhode island",
    "south carolina","south dakota","tennessee","texas","utah","vermont",
    "virginia","washington","west virginia","wisconsin","wyoming",
}


@dataclass
class _ElectionKey:
    year:   Optional[int] = None
    office: Optional[str] = None
    party:  Optional[str] = None
    state:  Optional[str] = None

    def compatible(self, other: "_ElectionKey") -> bool:
        if self.year   and other.year   and self.year   != other.year:   return False
        if self.office and other.office and self.office != other.office: return False
        if self.party  and other.party  and self.party  != other.party:  return False
        if self.state  and other.state  and self.state  != other.state:  return False
        return True


def _parse_election_key(text: str) -> _ElectionKey:
    tl = text.lower()
    years = _YEAR_RE.findall(text)
    year = int(years[0]) if years else None

    office: Optional[str] = None
    for kw, off in _OFFICE_MAP:
        if kw in tl:
            office = off
            break

    party: Optional[str] = None
    for kw, p in _PARTY_MAP.items():
        if kw in tl:
            party = p
            break

    state: Optional[str] = None
    for s in sorted(_US_STATES, key=len, reverse=True):
        if s in tl:
            state = s
            break

    return _ElectionKey(year=year, office=office, party=party, state=state)


# ── Candidate name extraction ─────────────────────────────────────────────────

# Words that signal a temporal qualifier ("Before 2027", "By January") rather than
# a candidate name.  These appear as PredictIt / Kalshi subtitle suffixes but must
# not be used as candidate names or they cause false positive cross-market matches.
_TEMPORAL_STARTS = frozenset({
    "before", "after", "by", "until", "prior", "within", "during",
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    "q1", "q2", "q3", "q4",
})


def _extract_candidate(m: Market) -> Optional[str]:
    """Return the short canonical candidate name from a prediction market, or None.

    Kalshi:     title is "Question: Candidate Name" (yes_sub_title appended in feed)
    PredictIt:  title is "Will X win?: Candidate Name"
    Polymarket: binary — outcome name if it's not a generic Yes/No
    """
    # Colon-split works for Kalshi and PredictIt
    if ": " in m.event_name:
        after = m.event_name.rsplit(": ", 1)[-1].strip()
        # Reject temporal qualifiers like "Before 2027" or "By January 2026"
        first_word = after.split()[0].lower() if after.split() else ""
        if first_word in _TEMPORAL_STARTS:
            return None
        # Valid candidate: starts uppercase, 2–60 chars, 1–5 words, no question marks
        if (2 <= len(after) <= 60
                and after[0].isupper()
                and "?" not in after
                and len(after.split()) <= 5):
            return after.lower()

    # Polymarket: outcome name when it's not a generic Yes/No label
    if m.source == Source.POLYMARKET and m.outcomes:
        name = m.outcomes[0].name.strip()
        if name.lower() not in ("yes", "no", "yep", "nope") and len(name) >= 2:
            return name.lower()

    return None


# ── Filler word stripping ─────────────────────────────────────────────────────

# ── Crypto / numeric threshold matching ──────────────────────────────────────
# Matches "Will BTC be above $110,000 by Dec 31?" against
# "Bitcoin above $110k end of 2026?" — same asset, same level, close date.

_CRYPTO_ASSETS: dict[str, str] = {
    "btc": "btc", "bitcoin": "btc",
    "eth": "eth", "ethereum": "eth",
    "sol": "sol", "solana": "sol",
    "xrp": "xrp", "ripple": "xrp",
    "doge": "doge", "dogecoin": "doge",
    "bnb": "bnb",
    "ada": "ada", "cardano": "ada",
    "avax": "avax", "avalanche": "avax",
    "matic": "matic", "polygon": "matic",
    "link": "link", "chainlink": "link",
}

_ECON_ASSETS: dict[str, str] = {
    "cpi": "cpi", "inflation": "cpi",
    "fed rate": "fed_rate", "federal funds rate": "fed_rate", "ffr": "fed_rate",
    "fed funds": "fed_rate",
    "gdp": "gdp",
    "unemployment": "unemployment", "unemployment rate": "unemployment",
    "s&p 500": "sp500", "s&p500": "sp500", "spx": "sp500",
    "nasdaq": "nasdaq",
    "dow jones": "dow", "djia": "dow",
}

_NUM_RE = re.compile(r'[\$]?([\d,]+(?:\.\d+)?)([kmb])?', re.IGNORECASE)
_YEAR_MONTH_RE = re.compile(
    r'\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|'
    r'jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|'
    r'dec(?:ember)?)\b.*?\b(20\d\d)\b'
    r'|'
    r'\b(q[1-4])\s*(20\d\d)\b'
    r'|'
    r'\b(20\d\d)\b',
    re.IGNORECASE,
)


def _parse_num(s: str, suffix: str) -> float:
    """Parse "110,000" → 110000.0, "110k" → 110000.0, "1.1m" → 1100000.0."""
    try:
        v = float(s.replace(",", ""))
        m = suffix.lower() if suffix else ""
        if m == "k": v *= 1_000
        elif m == "m": v *= 1_000_000
        elif m == "b": v *= 1_000_000_000
        return v
    except (ValueError, AttributeError):
        return 0.0


def _extract_crypto_key(text: str) -> tuple[str, float] | None:
    """Return (asset_canonical, threshold_usd) if text is a crypto price question."""
    tl = text.lower()
    asset = None
    for kw, canon in _CRYPTO_ASSETS.items():
        if kw in tl:
            asset = canon
            break
    if not asset:
        return None
    # Extract the largest dollar/numeric value as threshold
    best = 0.0
    for m in _NUM_RE.finditer(tl):
        v = _parse_num(m.group(1), m.group(2) or "")
        if v > best:
            best = v
    if best < 100:   # sanity — crypto prices are > $100
        return None
    return (asset, round(best, -2))   # round to nearest 100 for fuzzy match


def _extract_econ_key(text: str) -> tuple[str, float] | None:
    """Return (indicator, threshold) for economics questions (CPI, Fed rate, etc.)."""
    tl = text.lower()
    indicator = None
    for kw, canon in sorted(_ECON_ASSETS.items(), key=lambda x: -len(x[0])):
        if kw in tl:
            indicator = canon
            break
    if not indicator:
        return None
    best = 0.0
    for m in _NUM_RE.finditer(tl):
        v = _parse_num(m.group(1), m.group(2) or "")
        if 0 < v < 100:   # rates / percentages are in this range
            if v > best:
                best = v
    return (indicator, round(best, 1)) if best > 0 else (indicator, -1.0)


_FILLER_RE = re.compile(
    # True grammatical filler only — keep election/nomination/primary/general
    # because they distinguish different contest types.
    r'\b(will|the|a|an|be|to|is|are|was|were|'
    r'win|wins|winning|lose|loses|become|named|selected|'
    r'who|what|which|that|this|for|of|in|on|at|by)\b',
    re.IGNORECASE,
)


def _strip_filler(text: str) -> str:
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    text = _FILLER_RE.sub(' ', text)
    return re.sub(r'\s+', ' ', text).strip()


# ── Inverted proper-noun index ────────────────────────────────────────────────

def _proper_nouns(text: str) -> list[str]:
    """Extract multi-char capitalized word tokens (likely proper nouns/names).

    Uses strict 3+ char lowercase requirement to avoid single-word filler
    like "Will" or "The" polluting the index with every market.
    """
    # [A-Z] + at least 2 lowercase chars → minimum "Jon", excludes "I", "US", "GOP"
    return [t.lower() for t in re.findall(r'\b[A-Z][a-z]{2,}\b', text)]


def _build_name_index(bucket: list[tuple[int, Market]]) -> dict[str, list[int]]:
    """Map lower-cased proper-noun tokens → bucket POSITIONS (0-based, not global idx)."""
    idx: dict[str, list[int]] = defaultdict(list)
    for bucket_pos, (_global_idx, m) in enumerate(bucket):
        cand = _extract_candidate(m)
        tokens = cand.split() if cand else _proper_nouns(m.event_name)
        for tok in tokens:
            idx[tok].append(bucket_pos)
    return dict(idx)


def _build_candidate_map(bucket: list[tuple[int, Market]]) -> dict[int, set[int]]:
    """For each bucket position, return the set of other positions sharing a token.

    Limits pairwise comparisons to markets that share at least one proper-noun token,
    reducing average complexity from O(n²) to O(n).
    """
    name_idx = _build_name_index(bucket)
    cmap: dict[int, set[int]] = defaultdict(set)
    for positions in name_idx.values():
        if len(positions) < 2:
            continue
        for bp in positions:
            cmap[bp].update(p for p in positions if p != bp)
    return dict(cmap)


# ── Team / game name normalisation ────────────────────────────────────────────

_STOP = {
    "vs", "v", "at", "the", "and", "&", "fc", "afc", "sc", "city",
    "united", "utd", "town", "county", "athletic", "rovers", "wanderers",
    "hotspur", "albion", "wednesday", "tuesday", "monday",
    "total", "totals", "points", "goals", "game", "games",
    "over", "under", "spread", "winner", "win",
    "runs", "scored",
}

_TEAM_EXPANSIONS: dict[str, str] = {
    "new york y":    "new york yankees",
    "new york m":    "new york mets",
    "new york j":    "new york jets",
    "chicago ws":    "chicago white sox",
    "chicago c":     "chicago cubs",
    "chicago b":     "chicago bulls",
    "los angeles a": "los angeles angels",
    "los angeles d": "los angeles dodgers",
    "los angeles r": "los angeles rams",
    "los angeles c": "los angeles chargers",
    "los angeles l": "los angeles lakers",
    "los angeles k": "los angeles kings",
    "mtl canadiens": "montreal canadiens",
    "car hurricanes":"carolina hurricanes",
    # MLB
    "nyy": "new york yankees",   "nym": "new york mets",
    "bos": "boston",             "lad": "los angeles dodgers",
    "laa": "los angeles angels", "chc": "chicago cubs",
    "chw": "chicago white sox",  "atl": "atlanta",
    "stl": "st louis",           "min": "minnesota",
    "cle": "cleveland",          "det": "detroit",
    "cin": "cincinnati",         "mil": "milwaukee",
    "pit": "pittsburgh",         "col": "colorado",
    "ari": "arizona",            "sdp": "san diego padres",
    "tex": "texas rangers",      "kcr": "kansas city royals",
    "tor": "toronto",            "bal": "baltimore",
    "tbr": "tampa bay rays",     "mia": "miami",
    "was": "washington",         "wsh": "washington",
    # NBA
    "gsw": "golden state warriors",  "lal": "los angeles lakers",
    "lac": "los angeles clippers",   "bkn": "brooklyn nets",
    "phx": "phoenix suns",           "dal": "dallas",
    "chi": "chicago",                "nyk": "new york knicks",
    "den": "denver",                 "mem": "memphis grizzlies",
    "ind": "indiana",                "por": "portland trail blazers",
    "sac": "sacramento kings",       "hou": "houston",
    "orl": "orlando magic",          "nop": "new orleans pelicans",
    "cha": "charlotte hornets",
    # NHL
    "nyr": "new york rangers",    "nyi": "new york islanders",
    "njd": "new jersey devils",   "phi": "philadelphia",
    "car": "carolina",            "fla": "florida panthers",
    "tbl": "tampa bay lightning", "ott": "ottawa senators",
    "mtl": "montreal canadiens",  "buf": "buffalo",
    "cbj": "columbus blue jackets","nsh": "nashville predators",
    "wpg": "winnipeg jets",       "vgs": "vegas golden knights",
    "edm": "edmonton oilers",     "cgy": "calgary flames",
    "van": "vancouver canucks",   "sea": "seattle",
    "sjs": "san jose sharks",     "ana": "anaheim ducks",
    "lak": "los angeles kings",
    # NFL
    "kc":  "kansas city chiefs",  "sf":  "san francisco 49ers",
    "nyg": "new york giants",     "nyj": "new york jets",
    "ne":  "new england patriots","no":  "new orleans saints",
    "tb":  "tampa bay buccaneers","lar": "los angeles rams",
    "las": "las vegas raiders",   "ten": "tennessee titans",
    "jax": "jacksonville jaguars","gb":  "green bay packers",
    # short forms / nicknames
    "a's":           "athletics",
    "okc":           "oklahoma city thunder",
    "sas":           "san antonio spurs",
    "sd":            "san diego",
    "oak":           "oakland",
    "spurs":         "san antonio spurs",
    "thunder":       "oklahoma city thunder",
    "oklahoma city": "oklahoma city thunder",
    "san antonio":   "san antonio spurs",
    "cavaliers":     "cleveland cavaliers",
    "pacers":        "indiana pacers",
    "nuggets":       "denver nuggets",
    "timberwolves":  "minnesota timberwolves",
    "celtics":       "boston celtics",
    "heat":          "miami heat",
    "knicks":        "new york knicks",
    "lakers":        "los angeles lakers",
    "warriors":      "golden state warriors",
    # MLB city expansions (for Kalshi KXMLBTOTAL "Arizona vs Seattle: Total Runs")
    "seattle":        "seattle mariners",
    "colorado":       "colorado rockies",
    "philadelphia":   "philadelphia phillies",
    "texas":          "texas rangers",
    "milwaukee":      "milwaukee brewers",
    "cincinnati":     "cincinnati reds",
    "detroit":        "detroit tigers",
    "cleveland":      "cleveland guardians",
    # NHL city/nickname
    "oilers":         "edmonton oilers",
    "jets":           "winnipeg jets",
    "golden knights": "vegas golden knights",
    "panthers":       "florida panthers",
    "hurricanes":     "carolina hurricanes",
    "rangers":        "new york rangers",
    # Soccer — FIFA country name variations (World Cup / international)
    "usa":                   "united states",
    "u.s.a.":                "united states",
    "usmnt":                 "united states",
    "uswnt":                 "united states",
    "cote d'ivoire":         "ivory coast",
    "côte d'ivoire":         "ivory coast",
    "republic of ireland":   "ireland",
    "republic of korea":     "south korea",
    "korea republic":        "south korea",
    "dprk":                  "north korea",
    "korea dpr":             "north korea",
    "bosnia":                "bosnia and herzegovina",
    "czechia":               "czech republic",
    "trinidad":              "trinidad and tobago",
    "t&t":                   "trinidad and tobago",
    "china pr":              "china",
    "chinese taipei":        "taiwan",
    "democratic republic of congo": "congo dr",
    "dr congo":              "congo dr",
    "drc":                   "congo dr",
}

# NOTE: "arizona" and "az" intentionally removed from _TEAM_EXPANSIONS because
# "Arizona" appears in political market titles (Arizona Senate race) and would
# be incorrectly expanded to "arizona diamondbacks" there.  MLB Arizona markets
# use the "ari" abbreviation which safely expands.


def _expand_team_names(s: str) -> str:
    sl = s.lower()
    for abbr, full in _TEAM_EXPANSIONS.items():
        if full in sl:
            continue
        sl = re.sub(r'(?<!\w)' + re.escape(abbr) + r'(?!\w)', full, sl)
    return sl


_KALSHI_TICKER_PREFIXES = (
    "KXMLBGAME", "KXMLBTOTAL",
    "KXNBAGAME", "KXNBATOTAL",
    "KXNHLGAME", "KXNHLTOTAL",
    "KXNFLGAME", "KXNFLTOTAL",
    "KXSOCCERGAME", "KXSOCCERTOTAL",
    "KXATPGWINNER", "KXWTASETWINNER",
    "KXUFCFIGHT", "KXBOXINGFIGHT",
)
_KALSHI_DATE_TOKEN = re.compile(r'^\d{2}[A-Z]{3}\d{0,2}$')

# ── Sub-market period detection ───────────────────────────────────────────────
# Prevents "match winner" from being compared to "2nd set winner" or "1st half"
# markets — those share player/team names but resolve on completely different events.

_SET_RE     = re.compile(
    r'\b(set\s*[1-5]|[1-5](?:st|nd|rd|th)\s+set'
    r'|first\s+set|second\s+set|third\s+set|fourth\s+set|fifth\s+set)\b',
    re.IGNORECASE,
)
_HALF_RE    = re.compile(
    r'\b([1-2](?:st|nd)\s+half|first\s+half|second\s+half)\b',
    re.IGNORECASE,
)
_PERIOD_RE  = re.compile(
    r'\b([1-3](?:st|nd|rd)\s+period|first\s+period|second\s+period|third\s+period)\b',
    re.IGNORECASE,
)
_QUARTER_RE = re.compile(
    r'\b([1-4](?:st|nd|rd|th)\s+quarter|first\s+quarter|second\s+quarter'
    r'|third\s+quarter|fourth\s+quarter)\b',
    re.IGNORECASE,
)
_INNING_RE  = re.compile(
    r'\b([1-9](?:st|nd|rd|th)\s+inning|first\s+inning)\b',
    re.IGNORECASE,
)


def _sub_market_tag(event_name: str) -> str:
    """Return a period discriminator for game sub-markets.

    'full' = outright match winner (the default).
    Any other return value means the market resolves on a specific period, set,
    half, etc. — these must never be compared against full-match odds.
    """
    # Kalshi tickers that encode "set winner" in the prefix
    name_u = event_name.upper()
    if "SETWINNER" in name_u:
        return "set"

    name = event_name.lower()
    if _SET_RE.search(name):
        return "set"
    if _HALF_RE.search(name):
        return "half"
    if _PERIOD_RE.search(name):
        return "period"
    if _QUARTER_RE.search(name):
        return "quarter"
    if _INNING_RE.search(name):
        return "inning"
    return "full"


def _expand_kalshi_ticker(s: str) -> str:
    su = s.upper()
    for prefix in _KALSHI_TICKER_PREFIXES:
        if su.startswith(prefix + "-"):
            parts = s[len(prefix) + 1:].split("-")
            names = [
                p.lower() for p in parts
                if p and not _KALSHI_DATE_TOKEN.match(p.upper()) and len(p) > 1
            ]
            return " ".join(names) if names else s
    return s


_NORM_VS = (" vs. ", " vs ", " v. ", " v ", " at ")


def _normalize_game(s: str) -> str:
    """Normalize a game/sports market title for fuzzy comparison."""
    s = _expand_kalshi_ticker(s)
    s = _expand_team_names(s)
    s_lower = s.lower()
    if ": " in s:
        colon_pos = s.index(": ")
        for sep in _NORM_VS:
            sep_pos = s_lower.find(sep)
            if sep_pos != -1 and colon_pos < sep_pos:
                s = s[colon_pos + 2:]
                break
    tokens = s.lower().strip().split()
    tokens = [t.strip(".,:-()") for t in tokens]
    tokens = [t for t in tokens if t not in _STOP and len(t) > 1]
    return " ".join(tokens)


# backward-compat alias
def _normalize(s: str) -> str:
    return _normalize_game(s)


# ── Scoring ───────────────────────────────────────────────────────────────────

def _jaccard(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    union = sa | sb
    return (len(sa & sb) / len(union) * 100) if union else 100.0


def _score_game(na: str, nb: str) -> float:
    return (fuzz.token_sort_ratio(na, nb) * 0.5
            + fuzz.token_set_ratio(na, nb) * 0.3
            + _jaccard(na, nb) * 0.2)


def _score_prediction(na: str, nb: str) -> float:
    fa, fb = _strip_filler(na), _strip_filler(nb)
    if not fa or not fb:
        return 0.0
    return (fuzz.token_sort_ratio(fa, fb) * 0.5
            + fuzz.token_set_ratio(fa, fb) * 0.3
            + _jaccard(fa, fb) * 0.2)


# ── Market type compatibility ─────────────────────────────────────────────────

def _market_types_compatible(a: str, b: str) -> bool:
    a, b = a.lower(), b.lower()
    if a == b:
        return True
    for group in _SPORTS_ALIASES:
        if a in group and b in group:
            return True
    return False


# ── Per-type same-event logic ─────────────────────────────────────────────────

def _prediction_same(a: Market, b: Market) -> bool:
    """True if two prediction markets represent the same event."""
    if a.source == b.source:
        return False

    # ── Crypto price fast path ────────────────────────────────────────────────
    # "Will BTC be above $110,000 by end of 2026?" on Kalshi vs Polymarket.
    # If both questions name the same asset and threshold we call it a match
    # without fuzzy scoring (number formats vary wildly: "$110k" vs "110,000").
    ka = _extract_crypto_key(a.event_name)
    kb = _extract_crypto_key(b.event_name)
    if ka and kb:
        if ka[0] == kb[0]:                        # same asset
            ratio = min(ka[1], kb[1]) / max(ka[1], kb[1]) if max(ka[1], kb[1]) else 1
            return ratio >= 0.90                  # within 10% → same threshold
        return False                              # different assets → never match

    # ── Economics indicator fast path ─────────────────────────────────────────
    ea = _extract_econ_key(a.event_name)
    eb = _extract_econ_key(b.event_name)
    if ea and eb:
        if ea[0] == eb[0]:                        # same indicator (CPI, fed_rate…)
            if ea[1] < 0 or eb[1] < 0:           # no numeric threshold — fall through
                pass
            else:
                return abs(ea[1] - eb[1]) <= 0.25  # within 0.25pp
        else:
            return False

    # ── Hard veto: incompatible structured election keys ─────────────────────
    if not _parse_election_key(a.event_name).compatible(_parse_election_key(b.event_name)):
        return False

    # Hard veto: different US states
    sa, sb = _us_state(a.event_name), _us_state(b.event_name)
    if sa and sb and sa != sb:
        return False

    # Candidate fast path: extract short name from subtitle / colon-split
    cand_a = _extract_candidate(a)
    cand_b = _extract_candidate(b)

    if cand_a and cand_b:
        return fuzz.token_sort_ratio(cand_a, cand_b) >= _THRESHOLD_CANDIDATE

    if cand_a:
        return cand_a in b.event_name.lower()

    if cand_b:
        return cand_b in a.event_name.lower()

    # Fallback: filler-stripped fuzzy on full question strings
    return _score_prediction(a.event_name, b.event_name) >= _THRESHOLD_PREDICTION


def _same_game_date(a: Market, b: Market) -> bool:
    """True if both markets close within 20 hours of each other.

    Game markets on different platforms use close_time / endDate as commence_time.
    Same-day games on Kalshi vs Polymarket differ by ≤ a few hours; different-day
    games (same teams, next series game) differ by ≥ 20 hours.  20h is the safe
    midpoint: max same-day spread is ~12h (noon vs 10pm game), min cross-day spread
    is ~20h (back-to-back night games in the same series).
    """
    if not a.commence_time or not b.commence_time:
        return True  # unknown date — don't reject
    diff_secs = abs((a.commence_time - b.commence_time).total_seconds())
    return diff_secs <= 64_800  # 18 hours — same-day games differ ≤12h; back-to-back series games ≥20h


def _game_same_guards(a: Market, b: Market) -> bool:
    """Fast early-exit checks before any string scoring. Return False to reject."""
    a_pred = a.source in PREDICTION_SOURCES
    b_pred = b.source in PREDICTION_SOURCES
    if a_pred != b_pred:
        if not (a.market_type.lower() in _GAME_MARKET_TYPES_SET
                and b.market_type.lower() in _GAME_MARKET_TYPES_SET):
            return False
    if a.source == b.source:
        if a.source not in SPORTS_SOURCES:
            return False
        a_book = a.raw.get("bookmaker", "")
        b_book = b.raw.get("bookmaker", "")
        if not a_book or not b_book or a_book == b_book:
            return False
    if not _market_types_compatible(a.market_type, b.market_type):
        return False
    if a.sport != b.sport:
        return False
    if not _same_game_date(a, b):
        return False
    # Hard veto: different sub-market periods (match winner ≠ set 2 winner ≠ 1st half)
    # This fires even when the bucket key already separates them — guards against
    # cross-bucket comparisons that bypass the bucket step (e.g. OddsAPI fast path).
    if _sub_market_tag(a.event_name) != _sub_market_tag(b.event_name):
        return False
    return True


def _game_same(a: Market, b: Market) -> bool:
    """True if two game markets represent the same matchup (fuzzy fallback)."""
    if not _game_same_guards(a, b):
        return False
    return _score_game(_normalize_game(a.event_name), _normalize_game(b.event_name)) >= _THRESHOLD_GAME


# ── Bucket matching ───────────────────────────────────────────────────────────

def _match_prediction_bucket(
    bucket: list[tuple[int, Market]],
    groups: list[list[Market]],
    used: set[int],
) -> None:
    """Match prediction markets using inverted index + optional LLM verification.

    Three-zone decision (mirrors _match_game_bucket):
      score ≥ HIGH (90)           → auto-match
      BORDER (60) ≤ score < HIGH  → LLM verifies; fallback = threshold 78
      score < BORDER              → auto-reject
    """
    if len(bucket) < 2:
        return

    candidate_map = _build_candidate_map(bucket)

    # ── LLM pre-pass for borderline pairs ────────────────────────────────────
    if _llm_matcher is not None:
        borderline_pairs: list[tuple[Market, Market]] = []
        borderline_keys:  list[tuple[str, str]] = []

        for ii, (i, a) in enumerate(bucket):
            for bpj in sorted(candidate_map.get(ii, set())):
                if bpj <= ii:
                    continue
                j, b = bucket[bpj]
                if a.source == b.source:
                    continue
                ck = _cache_key(a, b)
                if ck in _CACHE:
                    continue
                # Score using filler-stripped prediction similarity
                score = _score_prediction(a.event_name, b.event_name)
                # Also check candidate-level score if candidates extractable
                ca, cb = _extract_candidate(a), _extract_candidate(b)
                if ca and cb:
                    cand_score = fuzz.token_sort_ratio(ca, cb)
                    score = max(score, cand_score)  # take best signal

                if score >= _THRESHOLD_PRED_HIGH:
                    _CACHE[ck] = True
                elif score >= _THRESHOLD_PRED_BORDER:
                    borderline_pairs.append((a, b))
                    borderline_keys.append(ck)
                else:
                    _CACHE[ck] = False

        if borderline_pairs:
            llm_results = _llm_matcher.verify_pairs(borderline_pairs, category="prediction")
            for ck, is_match in zip(borderline_keys, llm_results):
                _CACHE[ck] = is_match

    # ── Main grouping loop ────────────────────────────────────────────────────
    for ii, (gi, a) in enumerate(bucket):
        if gi in used:
            continue
        group: list[Market] = [a]

        for bpj in sorted(candidate_map.get(ii, set())):
            if bpj <= ii:
                continue
            gj, b = bucket[bpj]
            if gj in used:
                continue
            if a.source == b.source:
                continue

            ck = _cache_key(a, b)
            if ck not in _CACHE:
                _CACHE[ck] = _prediction_same(a, b)
            if _CACHE[ck]:
                group.append(b)
                used.add(gj)

        if len(group) > 1:
            used.add(gi)
            groups.append(group)


def _match_game_bucket(
    bucket: list[tuple[int, Market]],
    groups: list[list[Market]],
    used: set[int],
) -> None:
    """Match game/sports markets using fuzzy scoring + optional LLM verification.

    Three-zone decision:
      score ≥ HIGH (85)      → auto-match, no LLM call
      BORDER (52) ≤ score < HIGH → LLM verifies (batched); fallback = threshold 70
      score < BORDER         → auto-reject
    """
    if _llm_matcher is not None and len(bucket) >= 2:
        # Pre-pass: compute scores for all uncached pairs, collect borderline ones.
        borderline_pairs: list[tuple[Market, Market]] = []
        borderline_keys:  list[tuple[str, str]] = []

        for ii, (i, a) in enumerate(bucket):
            for j, b in bucket[ii + 1:]:
                ck = _cache_key(a, b)
                if ck in _CACHE:
                    continue  # already decided
                # Run the fast guards first (same source, sport mismatch, type mismatch)
                if not _game_same_guards(a, b):
                    _CACHE[ck] = False
                    continue
                na = _normalize_game(a.event_name)
                nb = _normalize_game(b.event_name)
                score = _score_game(na, nb)
                if score >= _THRESHOLD_GAME_HIGH:
                    _CACHE[ck] = True
                elif score >= _THRESHOLD_GAME_BORDER:
                    borderline_pairs.append((a, b))
                    borderline_keys.append(ck)
                else:
                    _CACHE[ck] = False

        if borderline_pairs:
            llm_results = _llm_matcher.verify_pairs(borderline_pairs)
            for ck, is_match in zip(borderline_keys, llm_results):
                _CACHE[ck] = is_match

    # Main grouping loop — all pairs either cached or decided by _game_same fallback.
    for ii, (i, a) in enumerate(bucket):
        if i in used:
            continue
        group: list[Market] = [a]
        for j, b in bucket[ii + 1:]:
            if j in used:
                continue
            ck = _cache_key(a, b)
            if ck not in _CACHE:
                _CACHE[ck] = _game_same(a, b)  # fallback: fuzzy threshold 70
            if _CACHE[ck]:
                group.append(b)
                used.add(j)
        if len(group) > 1:
            used.add(i)
            groups.append(group)


# ── Main entry point ──────────────────────────────────────────────────────────

def group_matching_markets(markets: list[Market]) -> list[list[Market]]:
    """Group markets from different sources that represent the same real-world event."""
    _maybe_expire_cache()

    groups: list[list[Market]] = []
    used: set[int] = set()

    # ── Fuzzy-matching buckets for cross-source matching ─────────────────────
    buckets: dict[tuple, list[tuple[int, Market]]] = defaultdict(list)
    for idx, m in enumerate(markets):
        if idx in used:
            continue
        buckets[_bucket_key(m)].append((idx, m))

    for bucket_key, bucket in buckets.items():
        cat = bucket_key[0]
        if cat == "prediction":
            _match_prediction_bucket(bucket, groups, used)
        else:
            _match_game_bucket(bucket, groups, used)

    return groups


# ── Shared helpers ────────────────────────────────────────────────────────────

def _us_state(text: str) -> Optional[str]:
    lower = text.lower()
    for state in sorted(_US_STATES, key=len, reverse=True):
        if state in lower:
            return state
    return None
