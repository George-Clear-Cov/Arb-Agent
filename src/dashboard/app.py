from __future__ import annotations

"""
FastAPI web application with WebSocket push.

Routes:
  GET  /                   → redirect to /arb
  GET  /odds-api           → Odds API dashboard
  GET  /betfair            → Betfair dashboard
  GET  /arb                → cross-platform arb opportunities
  GET  /portfolio          → paper trading P&L + equity chart
  WS   /ws                 → real-time state push (replaces 15s page reload)
  GET  /api/state          → JSON snapshot
  GET  /api/pnl-history    → time-series for TradingView equity chart
  POST /api/kill-switch    → toggle execution on/off
  POST /api/paper-trade    → manually paper-trade an arb by ID
"""
import asyncio
import json
from datetime import date as _date, datetime
from pathlib import Path
from typing import Optional

# Lifestyle-agent data files — resolved relative to this file so the path
# works on both dev (macOS, /Users/…) and prod (Ubuntu, /home/ubuntu/…).
_LIFESTYLE_ROOT = Path(__file__).parent.parent.parent.parent / "lifestyle-agent"
_JOBS_FILE    = _LIFESTYLE_ROOT / "jobs" / "jobs_db.json"
_FLIGHTS_FILE = _LIFESTYLE_ROOT / "flights" / "price_history.json"

_AIRPORT_LABELS = {
    "MIA":("Miami","Miami"), "FLL":("Fort Lauderdale","Miami"),
    "LAX":("Los Angeles","Los Angeles"), "BUR":("Burbank","Los Angeles"),
    "LGB":("Long Beach","Los Angeles"), "SNA":("Orange County","Los Angeles"),
    "SJU":("San Juan","San Juan"), "CUN":("Cancun","Cancun"),
    "AUA":("Aruba","Aruba"), "MBJ":("Montego Bay","Montego Bay"),
    "SXM":("St Martin","St Martin"), "UVF":("St Lucia","St Lucia"),
    "IBZ":("Ibiza","Ibiza"), "NCE":("Nice, France","Nice"),
    "JMK":("Mykonos, Greece","Mykonos"), "DBV":("Dubrovnik, Croatia","Dubrovnik"),
    "SPU":("Split, Croatia","Split"), "BCN":("Barcelona, Spain","Barcelona"),
    "JTR":("Santorini, Greece","Santorini"), "ATH":("Athens, Greece","Athens"),
    "NAP":("Naples (Amalfi Coast)","Amalfi Coast"), "CAG":("Cagliari, Sardinia","Sardinia"),
    "OLB":("Olbia, Sardinia","Sardinia"), "LIS":("Lisbon, Portugal","Lisbon"),
    "OPO":("Porto, Portugal","Porto"), "PMI":("Mallorca, Spain","Mallorca"),
    "AGP":("Malaga (Marbella)","Marbella"), "TIV":("Tivat, Montenegro","Montenegro"),
    "TGD":("Podgorica, Montenegro","Montenegro"), "BJV":("Bodrum, Turkey","Bodrum"),
    "JFK":("New York (JFK)","New York"), "LGA":("New York (LGA)","New York"),
    "EWR":("Newark (EWR)","New York"),
}

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.engine.feed_monitor import feed_monitor
from src.models import ArbOpportunity, Market, Source

# Sources that are pure prediction markets (no sportsbook odds)
_PREDICTION_SOURCES = {Source.KALSHI, Source.POLYMARKET, Source.PREDICTIT,
                       Source.OPINION, Source.PREDICTFUN, Source.MANIFOLD}
# Sources that are traditional sportsbooks
_SPORTSBOOK_SOURCES = {Source.ODDS_API, Source.BETFAIR, Source.ESPN_DK, Source.BOVADA, Source.KALSHI_SPORTS, Source.ACTION_NETWORK}

TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

app = FastAPI(title="Arbitrage Agent")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── Shared state ───────────────────────────────────────────────────────────

_state: dict = {
    "markets": [],
    "arbs": [],
    "stats": {},
    "last_updated": None,
    "kill_switch": False,
    "cycle_count": 0,
}

# Injected by agent.py after DB is ready
_store = None
_trader = None


def set_store(store, trader) -> None:
    global _store, _trader
    _store = store
    _trader = trader


# ── WebSocket connection manager ───────────────────────────────────────────

class _ConnectionManager:
    def __init__(self):
        self._active: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._active.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        try:
            self._active.remove(ws)
        except ValueError:
            pass

    async def broadcast(self, payload: dict) -> None:
        msg = json.dumps(payload)
        dead: list[WebSocket] = []
        for ws in list(self._active):
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


_manager = _ConnectionManager()


async def update_state(
    markets: list[Market],
    arbs: list[ArbOpportunity],
    stats: dict,
) -> None:
    _state["markets"] = markets
    _state["arbs"] = arbs
    _state["stats"] = stats
    _state["last_updated"] = datetime.utcnow().isoformat()
    _state["cycle_count"] += 1
    await _manager.broadcast(_build_ws_payload())


def is_kill_switch_active() -> bool:
    return _state["kill_switch"]


# ── WebSocket endpoint ─────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await _manager.connect(websocket)
    try:
        # Send full state immediately on connect
        await websocket.send_text(json.dumps(_build_ws_payload()))
        while True:
            # Keep alive — client sends pings
            await asyncio.wait_for(websocket.receive_text(), timeout=60)
    except (WebSocketDisconnect, asyncio.TimeoutError, Exception):
        _manager.disconnect(websocket)


# ── Pages ──────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse("/arb")


@app.get("/prediction-markets", response_class=HTMLResponse)
async def prediction_markets_dashboard(request: Request):
    return templates.TemplateResponse("prediction_markets.html", {
        "request": request,
        "last_updated": _state["last_updated"],
    })


@app.get("/sportsbooks", response_class=HTMLResponse)
async def sportsbooks_dashboard(request: Request):
    return templates.TemplateResponse("sportsbooks.html", {
        "request": request,
        "last_updated": _state["last_updated"],
    })


# Keep old URLs working as redirects
@app.get("/odds-api", include_in_schema=False)
async def odds_api_redirect():
    return RedirectResponse("/sportsbooks")


@app.get("/betfair", include_in_schema=False)
async def betfair_redirect():
    return RedirectResponse("/sportsbooks")


@app.get("/markets", include_in_schema=False)
async def markets_redirect():
    return RedirectResponse("/prediction-markets")


@app.get("/arb", response_class=HTMLResponse)
async def arb_dashboard(request: Request):
    return templates.TemplateResponse("arb.html", {
        "request": request,
        "arbs": _state["arbs"],
        "kill_switch": _state["kill_switch"],
        "last_updated": _state["last_updated"],
    })


@app.get("/portfolio", response_class=HTMLResponse)
async def portfolio_dashboard(request: Request):
    return templates.TemplateResponse("portfolio.html", {
        "request": request,
        "stats": _state["stats"],
        "last_updated": _state["last_updated"],
    })


@app.get("/debug", response_class=HTMLResponse)
async def debug_dashboard(request: Request):
    return templates.TemplateResponse("debug.html", {
        "request": request,
        "last_updated": _state["last_updated"],
    })


@app.get("/jobs", response_class=HTMLResponse)
async def jobs_dashboard(request: Request):
    return templates.TemplateResponse("jobs.html", {
        "request": request,
        "last_updated": _state["last_updated"],
    })


@app.get("/flights", response_class=HTMLResponse)
async def flights_dashboard(request: Request):
    return templates.TemplateResponse("flights.html", {
        "request": request,
        "last_updated": _state["last_updated"],
    })


# ── API ────────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def api_health():
    return JSONResponse({"status": "ok", "cycle_count": _state["cycle_count"]})


@app.get("/api/state")
async def api_state():
    return JSONResponse(_build_ws_payload())


@app.get("/api/pnl-history")
async def api_pnl_history():
    if _store is None:
        return JSONResponse({"history": []})
    history = await _store.get_pnl_history()
    return JSONResponse({"history": history})


@app.post("/api/kill-switch")
async def toggle_kill_switch():
    _state["kill_switch"] = not _state["kill_switch"]
    await _manager.broadcast({"type": "kill_switch", "active": _state["kill_switch"]})
    return JSONResponse({"kill_switch": _state["kill_switch"]})


@app.post("/api/paper-trade/{arb_id}")
async def manual_paper_trade(arb_id: str):
    if _trader is None:
        return JSONResponse({"error": "trader not ready"}, status_code=503)
    arb = next((a for a in _state["arbs"] if a.id == arb_id), None)
    if not arb:
        return JSONResponse({"error": "arb not found"}, status_code=404)
    pos = await _trader.open_position(arb)
    if pos:
        return JSONResponse({"position_id": pos.id, "stake": pos.total_stake})
    return JSONResponse({"error": "insufficient balance"}, status_code=400)


@app.get("/api/markets/prediction")
async def api_prediction_markets():
    markets = [m for m in _state["markets"] if m.source in _PREDICTION_SOURCES]
    return JSONResponse({"markets": [_market_to_dict(m) for m in markets]})


@app.get("/api/markets/sportsbooks")
async def api_sportsbook_markets():
    markets = [m for m in _state["markets"] if m.source in _SPORTSBOOK_SOURCES]
    return JSONResponse({"markets": [_market_to_dict(m) for m in markets]})


# Legacy endpoints kept for any external callers
@app.get("/api/markets/odds-api")
async def api_odds_api_markets():
    markets = [m for m in _state["markets"] if m.source == Source.ODDS_API]
    return JSONResponse({"markets": [_market_to_dict(m) for m in markets]})


@app.get("/api/markets/betfair")
async def api_betfair_markets():
    markets = [m for m in _state["markets"] if m.source == Source.BETFAIR]
    return JSONResponse({"markets": [_market_to_dict(m) for m in markets]})


@app.get("/api/markets/all")
async def api_all_markets():
    """All loaded markets across every source, for the Markets dashboard page."""
    return JSONResponse({"markets": [_market_to_dict(m) for m in _state["markets"]]})


@app.get("/api/arbs/sportsbooks")
async def api_arbs_sportsbooks():
    """Cross-bookmaker arbs from OddsAPI only (FanDuel vs DraftKings etc.)."""
    sb_source_values = {s.value for s in _SPORTSBOOK_SOURCES}
    arbs = [
        a for a in _state["arbs"]
        if a.sources and all(s in sb_source_values for s in a.sources)
    ]
    return JSONResponse({"arbs": [_arb_to_dict(a) for a in arbs]})


@app.get("/api/arbs/prediction")
async def api_arbs_prediction():
    """Cross-platform prediction market arbs (Kalshi vs PredictIt vs Polymarket)."""
    pred_source_values = {s.value for s in _PREDICTION_SOURCES}
    arbs = [
        a for a in _state["arbs"]
        if a.sources and any(s in pred_source_values for s in a.sources)
        and all(s in pred_source_values for s in a.sources)
    ]
    return JSONResponse({"arbs": [_arb_to_dict(a) for a in arbs]})


@app.get("/api/debug")
async def api_debug():
    """Feed diagnostics: per-source counts, sport breakdown, and feed health."""
    markets = _state["markets"]
    by_source: dict = {}
    for m in markets:
        src = m.source.value
        if src not in by_source:
            by_source[src] = {"total": 0, "by_sport": {}, "by_type": {}, "samples": []}
        entry = by_source[src]
        entry["total"] += 1
        entry["by_sport"][m.sport] = entry["by_sport"].get(m.sport, 0) + 1
        entry["by_type"][m.market_type] = entry["by_type"].get(m.market_type, 0) + 1
        if len(entry["samples"]) < 5:
            entry["samples"].append({
                "event": m.event_name[:80],
                "sport": m.sport,
                "type": m.market_type,
                "outcomes": [o.name for o in m.outcomes[:3]],
            })

    # Feed health from the monitor (last fetch time, error counts, etc.)
    feed_health = feed_monitor.all_statuses()
    unhealthy = feed_monitor.unhealthy_feeds()

    return JSONResponse({
        "total_markets": len(markets),
        "by_source": by_source,
        "feed_health": feed_health,
        "unhealthy_feeds": unhealthy,
        "cycle_count": _state["cycle_count"],
        "last_updated": _state["last_updated"],
    })


@app.get("/api/lifestyle/jobs")
async def api_jobs():
    return JSONResponse(_load_jobs())


@app.post("/api/lifestyle/jobs/{job_id}/status")
async def api_update_job_status(job_id: str, request: Request):
    body = await request.json()
    jobs = _load_jobs()
    for job in jobs:
        if job.get("job_id") == job_id:
            job["status"] = body.get("status", job["status"])
            break
    if _JOBS_FILE.exists():
        _JOBS_FILE.write_text(json.dumps(jobs, indent=2))
    return JSONResponse({"ok": True})


@app.get("/api/lifestyle/flights")
async def api_flights():
    return JSONResponse(_load_flights())


# ── Lifestyle helpers ──────────────────────────────────────────────────────

def _load_jobs() -> list:
    if not _JOBS_FILE.exists():
        return []
    return json.loads(_JOBS_FILE.read_text())


def _load_flights() -> list:
    if not _FLIGHTS_FILE.exists():
        return []
    ph = json.loads(_FLIGHTS_FILE.read_text())
    today_str = _date.today().isoformat()
    routes = []
    for route_key, history in ph.items():
        if not history:
            continue
        prices = [h["price"] for h in history if h.get("price")]
        if not prices:
            continue
        avg = sum(prices) / len(prices)
        low = min(prices)
        latest = history[-1]
        current = latest["price"]
        pct = round((avg - current) / avg * 100) if avg else 0
        dated = {}
        for h in history:
            dep = h.get("depart", "")
            if dep and dep >= today_str:
                if dep not in dated or h["price"] < dated[dep]["price"]:
                    dated[dep] = h
        deal_dates = []
        for dep, h in sorted(dated.items(), key=lambda x: dated[x[0]]["price"]):
            p = h["price"]
            pct_off = round((avg - p) / avg * 100) if avg else 0
            if pct_off >= 20:
                deal_dates.append({"depart": dep, "ret": h.get("ret",""), "price": p, "pct": pct_off})
        if not deal_dates and pct >= 20:
            deal_dates.append({"depart": None, "ret": None, "price": current, "pct": pct})
        parts = route_key.split("-", 1)
        origin = parts[0] if len(parts) == 2 else route_key
        dest   = parts[1] if len(parts) == 2 else ""
        dest_city, route_city = _AIRPORT_LABELS.get(dest, (dest, dest))
        origin_city = _AIRPORT_LABELS.get(origin, (origin,))[0]
        routes.append({
            "route": route_key, "origin": origin, "origin_city": origin_city,
            "dest": dest, "dest_city": dest_city, "route_city": route_city,
            "current": current, "avg": round(avg), "low": low,
            "pct_below": pct, "deal_dates": deal_dates,
            "scans": len(history),
            "last_seen": latest.get("ts","")[:16].replace("T"," "),
            "is_deal": pct >= 20,
        })
    routes.sort(key=lambda r: (-r["is_deal"], -r["pct_below"]))
    return routes


# ── Helpers ────────────────────────────────────────────────────────────────

def _build_ws_payload() -> dict:
    markets = _state["markets"]
    # per-source counts
    source_counts = {
        src.value: sum(1 for m in markets if m.source == src)
        for src in Source
        if any(m.source == src for m in markets)
    }
    # per-sport counts (across all sources)
    sport_counts: dict[str, int] = {}
    for m in markets:
        sport_counts[m.sport] = sport_counts.get(m.sport, 0) + 1

    return {
        "type": "state",
        "arbs": [_arb_to_dict(a) for a in _state["arbs"]],
        "stats": _state["stats"],
        "kill_switch": _state["kill_switch"],
        "cycle_count": _state["cycle_count"],
        "market_counts": source_counts,
        "sport_counts": sport_counts,
        "total_markets": len(markets),
        "last_updated": _state["last_updated"],
    }


def _group_by_bookmaker(markets: list[Market]) -> dict[str, list[Market]]:
    grouped: dict[str, list[Market]] = {}
    for m in markets:
        key = m.raw.get("bookmaker", "unknown")
        grouped.setdefault(key, []).append(m)
    return grouped


def _group_by_sport(markets: list[Market]) -> dict[str, list[Market]]:
    grouped: dict[str, list[Market]] = {}
    for m in markets:
        grouped.setdefault(m.sport, []).append(m)
    return grouped


def _arb_to_dict(a: ArbOpportunity) -> dict:
    # Build a lookup so we can enrich each leg with description + URL
    market_lookup: dict[tuple[str, str], Market] = {
        (m.source.value, m.market_id): m
        for m in _state["markets"]
    }
    # kalshi_sports legs use outcome tickers (e.g. KXMLBGAME-...-DET) but
    # Markets are keyed by event ticker (e.g. KXMLBGAME-...).  Index by each
    # outcome's market_id so legs can find their parent Market and its raw data.
    for _m in _state["markets"]:
        if _m.source.value == "kalshi_sports":
            for _o in _m.outcomes:
                if _o.market_id != _m.market_id:
                    market_lookup[("kalshi_sports", _o.market_id)] = _m

    # Earliest non-null expiry across all legs (determines arb time horizon)
    # Use expires_at from the arb itself (set by detector) if available;
    # otherwise fall back to scanning legs via market_lookup.
    if a.expires_at is not None:
        earliest_expiry = a.expires_at.isoformat()
    else:
        expiry_times = []
        for _l in a.legs:
            _m = market_lookup.get((_l.source.value, _l.market_id))
            ct = _m.commence_time if _m else None
            if ct:
                if ct.tzinfo is None:
                    ct = ct.replace(tzinfo=__import__("datetime").timezone.utc)
                expiry_times.append(ct)
        earliest_expiry = min(expiry_times).isoformat() if expiry_times else None

    return {
        "id": a.id,
        "sport": a.sport,
        "event_name": a.event_name,
        "market_type": a.market_type,
        "gross_margin_pct": round(a.gross_margin * 100, 2),
        "margin_pct": round(a.margin * 100, 2),
        "annualized_margin_pct": round(a.annualized_margin * 100, 1),
        "total_stake": a.total_stake,
        "profit": a.profit,
        "gross_profit": a.gross_profit,
        "sources": a.sources,
        "detected_at": a.detected_at.isoformat(),
        "earliest_expiry": earliest_expiry,
        "legs": [
            _leg_to_dict(l, market_lookup)
            for l in a.legs
        ],
        # Flat {source: url} map for quick badge links in the main table row
        "market_urls": {
            l.source.value: _market_url(
                l.source.value, l.market_id,
                (market_lookup.get((l.source.value, l.market_id)) or type("", (), {"raw": {}})()).raw
            )
            for l in a.legs
        },
    }


def _leg_to_dict(l: "ArbLeg", market_lookup: dict) -> dict:
    mkt = market_lookup.get((l.source.value, l.market_id))
    raw = mkt.raw if mkt else {}

    # Build a direct link to the market on the source platform
    url = _market_url(l.source.value, l.market_id, raw)

    # Best available description: raw description field → raw title → market event_name
    description = (
        raw.get("description")
        or raw.get("title")
        or (mkt.event_name if mkt else None)
        or ""
    )

    # Volume / liquidity shown alongside the leg
    volume = mkt.total_volume if mkt else None
    expire = mkt.commence_time.isoformat() if (mkt and mkt.commence_time) else None

    return {
        "source": l.source.value,
        "bookmaker": l.bookmaker,
        "outcome": l.outcome_name,
        "price": l.price,
        "effective_price": l.effective_price,
        "stake": l.stake,
        "side": l.side.value,
        "market_id": l.market_id,
        "market_url": url,
        "description": description,
        "volume": volume,
        "expires": expire,
    }


def _market_url(source: str, market_id: str, raw: dict) -> Optional[str]:
    """Build a direct URL to the market on its source platform."""
    # Prefer the canonical source_url the API returns (present for all PH platforms)
    if raw.get("source_url"):
        return raw["source_url"]
    # Fallbacks per source
    if source == "polymarket":
        slug = raw.get("slug")
        return f"https://polymarket.com/event/{slug}" if slug else f"https://polymarket.com/event/{market_id}"
    if source == "predictit":
        return f"https://www.predictit.org/markets/detail/{market_id}"
    if source == "opinion":
        return f"https://opinionmarkets.com/market/{market_id}"
    if source == "predictfun":
        return f"https://predict.fun/markets/{market_id}"
    if source == "kalshi_sports":
        # Use event_ticker from raw when present; it's the valid Kalshi URL target.
        # The leg's market_id is an outcome ticker (e.g. ...SEADET-DET) which may 404.
        ticker = raw.get("event_ticker") or market_id
        series = raw.get("series")
        if series:
            return f"https://kalshi.com/markets/{series}/{ticker}"
        return f"https://kalshi.com/markets/{ticker}"
    if source == "kalshi":
        return f"https://kalshi.com/markets/{market_id}"
    if source == "betfair":
        return f"https://www.betfair.com/exchange/plus/market/{market_id}"
    return None


def _market_to_dict(m: Market) -> dict:
    return {
        "source": m.source.value,
        "market_id": m.market_id,
        "sport": m.sport,
        "event_name": m.event_name,
        "market_type": m.market_type,
        "total_volume": m.total_volume,
        "commence_time": m.commence_time.isoformat() if m.commence_time else None,
        "outcomes": [
            {
                "name": o.name,
                "price": o.price,
                "implied_prob": o.implied_prob,
                "bookmaker": o.bookmaker,
                "side": o.side.value,
                "available_volume": o.available_volume,
            }
            for o in m.outcomes
        ],
    }
