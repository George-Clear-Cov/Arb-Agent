from __future__ import annotations

"""
Arbitrage agent — main entry point.

Starts all feeds, runs the fee-aware detector each cycle, updates the
WebSocket dashboard, triggers paper trades, and enforces risk limits.

Usage:
    python agent.py

Environment:
    Copy .env.example → .env and fill in your API keys.
"""
import asyncio
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, date, timedelta, timezone
from pathlib import Path

import uvicorn
import yaml
from dotenv import load_dotenv

from src.alerts.notifier import Notifier
from src.models import BetSide, Market, Outcome, Source
from src.dashboard.app import app, is_kill_switch_active, set_store, update_state
from src.engine.detector import detect_arbs_with_groups
from src.engine.pair_monitor import ConfirmedPairMonitor
from src.engine.feed_monitor import feed_monitor
from src.engine.matcher import init_llm_matcher
from src.engine.sizer import kelly_stake
from src.engine.threshold_tuner import compute_and_save as _tune_thresholds
from src.execution.kalshi_exec import KalshiExecutor
from src.execution.paper_trader import PaperTrader
from src.execution.polymarket_exec import PolymarketExecutor
from src.feeds.kalshi import KalshiFeed
from src.feeds.kalshi_sports import KalshiSportsFeed
from src.feeds.kalshi_ws import KalshiWSFeed
from src.feeds.polymarket_clob import PolymarketCLOBFeed
from src.feeds.polymarket_ws import PolymarketWSFeed
from src.feeds.predictit import PredictItFeed
from src.feeds.opinion import OpinionFeed
from src.feeds.gemini import GeminiFeed
from src.feeds.hyperliquid import HyperliquidFeed
from src.feeds.prophetx import ProphetXFeed
from src.storage.db import Store

load_dotenv()

# DATA_DIR: cloud volumes mount here; local dev falls back to ./state
_DATA_DIR = Path(os.environ.get("DATA_DIR", "."))
_STATE_DIR = _DATA_DIR / "state"
_STATE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Feed disk cache — persist expensive API responses across restarts
# ---------------------------------------------------------------------------
_FEED_CACHE_DIR = _STATE_DIR / "feed_cache"
_FEED_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path(feed_name: str) -> Path:
    safe = feed_name.replace("/", "_").replace(" ", "_")
    return _FEED_CACHE_DIR / f"{safe}.json"


def _save_feed_cache(feed_name: str, markets: list[Market]) -> None:
    try:
        payload = []
        for m in markets:
            payload.append({
                "source": m.source.value,
                "market_id": m.market_id,
                "sport": m.sport,
                "event_name": m.event_name,
                "commence_time": m.commence_time.isoformat() if m.commence_time else None,
                "home_team": m.home_team,
                "away_team": m.away_team,
                "market_type": m.market_type,
                "total_volume": m.total_volume,
                "raw": m.raw,
                "outcomes": [
                    {
                        "name": o.name,
                        "price": o.price,
                        "implied_prob": o.implied_prob,
                        "source": o.source.value,
                        "market_id": o.market_id,
                        "bookmaker": o.bookmaker,
                        "side": o.side.value,
                        "available_volume": o.available_volume,
                        "is_maker": o.is_maker,
                    }
                    for o in m.outcomes
                ],
            })
        data = {"ts": time.time(), "markets": payload}
        _cache_path(feed_name).write_text(json.dumps(data))
    except Exception as exc:
        log.warning("Could not save feed cache for %s: %s", feed_name, exc)


def _load_feed_cache(feed_name: str, max_age_seconds: int) -> list[Market] | None:
    """Return cached markets if the cache exists and is fresher than max_age_seconds."""
    path = _cache_path(feed_name)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        age = time.time() - data["ts"]
        if age > max_age_seconds:
            return None
        markets = []
        for m in data["markets"]:
            outcomes = [
                Outcome(
                    name=o["name"],
                    price=o["price"],
                    implied_prob=o["implied_prob"],
                    source=Source(o["source"]),
                    market_id=o["market_id"],
                    bookmaker=o.get("bookmaker"),
                    side=BetSide(o.get("side", "back")),
                    available_volume=o.get("available_volume"),
                    is_maker=o.get("is_maker", False),
                )
                for o in m["outcomes"]
            ]
            ct = m.get("commence_time")
            markets.append(Market(
                source=Source(m["source"]),
                market_id=m["market_id"],
                sport=m["sport"],
                event_name=m["event_name"],
                commence_time=datetime.fromisoformat(ct) if ct else None,
                home_team=m.get("home_team"),
                away_team=m.get("away_team"),
                market_type=m["market_type"],
                total_volume=m.get("total_volume"),
                raw=m.get("raw", {}),
                outcomes=outcomes,
            ))
        log.info("Loaded %d %s markets from disk cache (age %.0fs)", len(markets), feed_name, age)
        return markets
    except Exception as exc:
        log.warning("Could not load feed cache for %s: %s", feed_name, exc)
        return None


# ---------------------------------------------------------------------------
# Single-instance lock — prevents multiple agents hammering the same APIs
# ---------------------------------------------------------------------------
_LOCK_FILE = _STATE_DIR / "agent.lock"


def _acquire_lock() -> None:
    """Write our PID to the lock file.  Exit if another process is already running."""
    if _LOCK_FILE.exists():
        try:
            pid = int(_LOCK_FILE.read_text().strip())
            # Check if that process is still alive
            os.kill(pid, 0)
            # Still alive — refuse to start
            print(f"ERROR: agent already running (PID {pid}). "
                  f"Kill it first: kill {pid}", file=sys.stderr)
            sys.exit(1)
        except (ProcessLookupError, PermissionError):
            pass  # stale lock — process gone, safe to overwrite
        except ValueError:
            pass  # corrupted lock file — overwrite
    _LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    _LOCK_FILE.write_text(str(os.getpid()))


def _release_lock() -> None:
    try:
        _LOCK_FILE.unlink(missing_ok=True)
    except Exception:
        pass


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("agent")


def _expand_env(val) -> str:
    if isinstance(val, str) and val.startswith("${") and val.endswith("}"):
        return os.environ.get(val[2:-1], "")
    return val


def load_config(path: str = "config.yaml") -> dict:
    raw = Path(path).read_text()
    cfg = yaml.safe_load(raw)
    for section in cfg.values():
        if isinstance(section, dict):
            for k, v in section.items():
                section[k] = _expand_env(v)
    return cfg


class Agent:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._markets: list[Market] = []
        self._store: Store | None = None
        self._trader: PaperTrader | None = None

        # Fires when any WS feed updates a price — wakes detect_loop immediately
        self._price_event: asyncio.Event = asyncio.Event()

        pt = cfg["paper_trading"]
        self._min_arb = pt["min_arb_margin"]
        self._stake = pt["starting_balance"] * pt["max_position_pct"]
        self._daily_loss_limit = pt.get("daily_loss_limit", 0.05)  # fraction of balance

        alerts_cfg = cfg.get("alerts", {})
        self._notifier = Notifier(
            min_margin=alerts_cfg.get("min_margin_to_alert", 0.03),
            slack_webhook=os.environ.get("SLACK_WEBHOOK_URL") or None,
        )

        # Real-money execution
        exec_cfg = cfg.get("execution", {})
        self._exec_enabled         = os.environ.get("KALSHI_EXECUTE", "").lower() == "true"
        self._exec_min_ann_margin  = exec_cfg.get("min_annualized_margin", 0.50)  # 50% annualised
        self._exec_max_days        = exec_cfg.get("max_days_to_expiry", 30)
        self._exec_max_stake       = exec_cfg.get("max_stake_per_arb", 50.0)
        self._exec_dry_run         = exec_cfg.get("dry_run", False)
        self._executor: KalshiExecutor | None = None
        self._poly_executor: PolymarketExecutor | None = None
        self._executed_arb_ids: set[str] = set()  # persisted to disk; prevents re-fire on restart
        self._exec_ids_file = _STATE_DIR / "executed_ids.json"

        # Daily loss tracking
        self._day_start_balance: float | None = None
        self._current_day: date | None = None

        # PredictIt slow-polling state (free API but rate-limited ~1 req/5min safe)
        self._last_pi_fetch: datetime | None = None
        self._pi_markets: list[Market] = []

        # Kalshi slow-polling state (~28s fetch, keep to every 10 min)
        self._last_kalshi_fetch: datetime | None = None
        self._kalshi_markets: list[Market] = []

        # Kalshi sports live-game fast-polling state (every 30s)
        self._last_kalshi_sports_fetch: datetime | None = None
        self._kalshi_sports_markets: list[Market] = []

        # Polymarket cache — pre-seeded from disk on startup so WS can start
        # subscribing immediately without waiting for the 3-min CLOB fetch.
        self._poly_markets: list[Market] = []

    async def setup(self) -> None:
        self._store = Store()
        await self._store.connect()
        self._notifier._store = self._store  # persist notification dedup across restarts

        pt = self.cfg["paper_trading"]
        self._trader = PaperTrader(
            store=self._store,
            starting_balance=pt["starting_balance"],
            max_position_pct=pt["max_position_pct"],
        )

        set_store(self._store, self._trader)

        # Load persisted executed arb IDs so we don't re-fire on restart
        try:
            if self._exec_ids_file.exists():
                self._executed_arb_ids = set(json.loads(self._exec_ids_file.read_text()))
                log.info("Loaded %d executed arb IDs from disk", len(self._executed_arb_ids))
        except Exception as exc:
            log.warning("Could not load executed_ids.json: %s", exc)

        # Dedicated ThreadPoolExecutor for arb detection.  The OddsAPI fast path
        # (event_id grouping) reduces detection time to ~100ms, so GIL contention
        # is minimal.  rapidfuzz releases the GIL during string comparisons,
        # giving uvicorn ~80ms of window per cycle.
        self._detect_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="arb-detect")

        ka = self.cfg["kalshi"]
        self._kalshi_feed = None
        self._kalshi_sports_feed = None
        if ka.get("api_key"):
            self._kalshi_feed = KalshiFeed(
                api_key=ka["api_key"],
                base_url=ka["base_url"],
            )
            if self._kalshi_feed._disk_markets:
                self._kalshi_markets = self._kalshi_feed._disk_markets
                log.info("Kalshi: pre-seeded %d markets from disk cache", len(self._kalshi_markets))
            # Separate fast-polling feed for live game markets
            self._kalshi_sports_feed = KalshiSportsFeed(
                api_key=ka["api_key"],
                base_url=ka["base_url"],
            )

        # Real-money execution — only if KALSHI_EXECUTE=true and private key exists
        # KALSHI_EXEC_API_KEY is the RSA-signing UUID (separate from read-only KALSHI_API_KEY)
        exec_api_key = os.environ.get("KALSHI_EXEC_API_KEY") or ka.get("api_key")
        if self._exec_enabled and exec_api_key:
            key_path = os.environ.get("KALSHI_PRIVATE_KEY_PATH", "~/.kalshi/private_key.pem")
            try:
                self._executor = KalshiExecutor(
                    api_key=exec_api_key,
                    private_key_path=key_path,
                    base_url=ka["base_url"],
                )
                log.info(
                    "Kalshi executor ready — min ann. margin %.0f%% | max stake $%.0f | dry_run=%s",
                    self._exec_min_ann_margin * 100,
                    self._exec_max_stake,
                    self._exec_dry_run,
                )
            except FileNotFoundError as exc:
                log.error("Kalshi executor disabled: %s", exc)
                self._exec_enabled = False
        elif self._exec_enabled:
            log.warning("KALSHI_EXECUTE=true but no KALSHI_EXEC_API_KEY — execution disabled")

        # Polymarket execution — enabled when POLYMARKET_EXECUTE=true + POLY_PRIVATE_KEY set
        poly_priv = os.environ.get("POLY_PRIVATE_KEY", "")
        if os.environ.get("POLYMARKET_EXECUTE", "").lower() == "true" and poly_priv:
            try:
                self._poly_executor = PolymarketExecutor(private_key=poly_priv)
                log.info("Polymarket executor ready — wallet %s", self._poly_executor._address[:10])
            except Exception as exc:
                log.error("Polymarket executor disabled: %s", exc)

        po = self.cfg["polymarket"]
        self._poly_feed = PolymarketCLOBFeed(
            clob_url=po.get("clob_url", "https://clob.polymarket.com"),
            gamma_url=po.get("gamma_url", "https://gamma-api.polymarket.com"),
        )
        # Seed from disk cache immediately so detection/WS don't wait 3 min
        if self._poly_feed._disk_markets:
            self._poly_markets = self._poly_feed._disk_markets
            log.info("Polymarket: pre-seeded %d markets from disk cache", len(self._poly_markets))

        # Dedicated PredictIt feed — free public API, no auth, no quota concerns
        self._predictit_feed = PredictItFeed()
        if self._predictit_feed._disk_markets:
            self._pi_markets = self._predictit_feed._disk_markets
            log.info("PredictIt: pre-seeded %d markets from disk cache", len(self._pi_markets))

        # Opinion — BNB Chain CLOB (requires OPINION_API_KEY env var)
        self._opinion_feed = OpinionFeed()
        self._opinion_markets: list[Market] = []

        # Gemini prediction markets — CFTC-regulated, no auth needed for reads
        self._gemini_feed = GeminiFeed()
        self._gemini_markets: list[Market] = self._gemini_feed._disk_markets or []
        if self._gemini_markets:
            log.info("Gemini: pre-seeded %d markets from disk cache", len(self._gemini_markets))

        # Hyperliquid HIP-4 outcome markets — no auth needed
        self._hyperliquid_feed = HyperliquidFeed()
        self._hyperliquid_markets: list[Market] = self._hyperliquid_feed._disk_markets or []
        if self._hyperliquid_markets:
            log.info("Hyperliquid: pre-seeded %d markets from disk cache", len(self._hyperliquid_markets))

        # ProphetX sports exchange via PolyRouter (requires POLYROUTER_API_KEY)
        self._prophetx_feed = ProphetXFeed()
        self._prophetx_markets: list[Market] = []

        # LLM matcher disabled — matching handled by matcher_agent.py (embeddings, free)

    async def _check_daily_loss_limit(self) -> bool:
        """Return True if we should halt trading due to daily loss limit."""
        today = date.today()
        if self._current_day != today:
            self._current_day = today
            self._day_start_balance = await self._trader.balance()
            return False

        if self._day_start_balance is None:
            return False

        current = await self._trader.balance()
        if self._day_start_balance <= 0:
            return False
        loss_pct = (self._day_start_balance - current) / self._day_start_balance
        if loss_pct >= self._daily_loss_limit:
            log.warning(
                "Daily loss limit hit: lost %.1f%% today (limit %.1f%%)",
                loss_pct * 100, self._daily_loss_limit * 100,
            )
            return True
        return False

    # ------------------------------------------------------------------
    # Independent feed loops — each runs forever at its own interval.
    # Arb detection never blocks on or triggers an API call.
    # ------------------------------------------------------------------

    async def _feed_loop(self, name: str, fetch_fn, cache_attr: str, interval: int,
                         disk_cache: bool = False) -> None:
        """Generic feed loop: fetch → cache → sleep → repeat.

        If disk_cache=True, on startup we check for a recent disk cache
        (fresher than `interval` seconds) and skip the API call if found.
        After every successful fetch the result is persisted to disk.
        This prevents restarts from burning expensive API quota.

        Adaptive backoff: consecutive errors multiply the sleep by 2x up to 8x,
        so a broken feed backs off gracefully instead of hammering a down API.
        """
        log.info("Feed loop started: %s (every %ds)", name, interval)
        backoff_mult = 1.0

        # On startup, try disk cache first to avoid burning API quota
        if disk_cache:
            cached = _load_feed_cache(name, max_age_seconds=interval)
            if cached:
                setattr(self, cache_attr, cached)
                feed_monitor.record_success(name, len(cached))
                cache_age = time.time() - _cache_path(name).stat().st_mtime
                wait = max(0, interval - cache_age)
                log.info("%s: using disk cache, next fetch in %.0fs", name, wait)
                await asyncio.sleep(wait)

        while True:
            try:
                fresh = await fetch_fn()
                if fresh:
                    setattr(self, cache_attr, fresh)
                    feed_monitor.record_success(name, len(fresh))
                    log.info("%s: refreshed %d markets", name, len(fresh))
                    if disk_cache:
                        _save_feed_cache(name, fresh)
                    backoff_mult = 1.0  # reset only on successful non-empty fetch
                else:
                    feed_monitor.record_success(name, 0)
                    log.warning("%s: returned 0 markets — keeping cache (%d)",
                                name, len(getattr(self, cache_attr)))
            except Exception as exc:
                feed_monitor.record_error(name, str(exc))
                log.error("%s feed error: %s", name, exc)
                backoff_mult = min(backoff_mult * 2.0, 8.0)
                if backoff_mult > 1.0:
                    log.warning("%s: backing off — next fetch in %.0fs", name,
                                interval * backoff_mult)
            await asyncio.sleep(interval * backoff_mult)

    async def _auto_execute(self, arbs: list) -> None:
        """Fire real Kalshi orders for qualifying arbs.

        Only runs when KALSHI_EXECUTE=true, executor is ready, and the arb
        passes margin + expiry filters.  Each arb ID is tracked in-memory so
        we don't re-fire the same opportunity on the next detection cycle.
        """
        if not self._executor:
            return
        if not self._store:
            return

        now    = datetime.now(tz=timezone.utc)
        cutoff = now + timedelta(days=self._exec_max_days)

        for arb in arbs:
            if arb.id in self._executed_arb_ids:
                continue
            # Gate on annualised return — raw margin alone is meaningless without
            # knowing the time horizon.  16% in 2 years is worse than T-bills.
            if arb.annualized_margin < self._exec_min_ann_margin:
                continue

            # Expiry must exist and be within the configured window
            if arb.expires_at:
                exp = arb.expires_at
                if exp.tzinfo is None:
                    exp = exp.replace(tzinfo=timezone.utc)
                if exp > cutoff:
                    continue
            else:
                continue  # skip if no expiry known — can't assess urgency

            # Must have at least one Kalshi leg
            kalshi_sources = {Source.KALSHI, Source.KALSHI_SPORTS}
            if not any(leg.source in kalshi_sources for leg in arb.legs):
                continue

            # Mark as seen before firing — persisted to disk to survive restarts
            self._executed_arb_ids.add(arb.id)
            try:
                self._exec_ids_file.write_text(json.dumps(list(self._executed_arb_ids)))
            except Exception:
                pass

            log.info(
                "AUTO-EXEC: firing arb %s | %.1f%% margin | $%.2f stake | expires %s",
                arb.id, arb.margin * 100, arb.total_stake,
                arb.expires_at.strftime("%Y-%m-%d") if arb.expires_at else "?",
            )

            # Kelly-sized stake — scales with edge and current bankroll
            pt = self.cfg["paper_trading"]
            bankroll = await self._store.get_balance(pt["starting_balance"])
            # Game arbs (fast resolution) get higher Kelly; prediction arbs get lower
            # due to longer horizon and higher model/execution uncertainty
            kelly_fraction = 0.25 if arb.sport not in ("prediction",) else 0.15
            exec_stake = kelly_stake(
                margin=arb.margin,
                bankroll=bankroll,
                kelly_fraction=kelly_fraction,
                max_stake=self._exec_max_stake,
            )
            log.info(
                "AUTO-EXEC Kelly stake: $%.2f (margin=%.1f%% bankroll=$%.0f k=%.2f)",
                exec_stake, arb.margin * 100, bankroll, kelly_fraction,
            )

            # Execute each platform's legs with the right executor
            results = []
            try:
                if self._executor:
                    r = await self._executor.execute_arb(
                        arb, max_stake=exec_stake, dry_run=self._exec_dry_run,
                    )
                    results.append(r)
                if self._poly_executor:
                    r = await self._poly_executor.execute_arb(
                        arb, max_stake=exec_stake, dry_run=self._exec_dry_run,
                    )
                    results.append(r)
            except Exception as exc:
                log.exception("AUTO-EXEC exception for arb %s: %s", arb.id, exc)
                continue

            fully_filled = results and all(r.fully_filled for r in results if r.error != "dry_run")
            hedged       = any(getattr(r, "hedged", False) for r in results)

            if fully_filled:
                log.info(
                    "AUTO-EXEC SUCCESS arb=%s | %.1f%% margin | legs=%d",
                    arb.id, arb.margin * 100, sum(len(r.leg_results) for r in results),
                )
                await self._notifier.notify_arbs([arb])

            elif hedged:
                msg = (
                    f"ONE-LEGGED POSITION — MANUAL HEDGE NEEDED: arb {arb.id} "
                    f"({arb.event_name[:50]}) filled one side but the other failed. "
                    f"Check Kalshi and Polymarket dashboards immediately."
                )
                log.error("AUTO-EXEC HEDGE ALERT: %s", msg)
                await self._notifier.notify_arbs([arb])

            elif results and results[0].partial:
                log.warning(
                    "AUTO-EXEC PARTIAL (rolled back) arb=%s: positions cancelled", arb.id,
                )

            else:
                err = results[0].error if results else "no executor ran"
                log.warning("AUTO-EXEC FAILED arb=%s: %s", arb.id, err)

    async def _dismiss_arb(self, arb_id_prefix: str) -> str:
        """Handle /dismiss <id> — permanently suppress a false-positive arb pair."""
        if len(arb_id_prefix) < 6:
            return f"ID prefix too short (need ≥6 chars, got {len(arb_id_prefix)})"
        arb = await self._store.get_opportunity_by_id_prefix(arb_id_prefix)
        if not arb:
            return f"No arb found with ID prefix '{arb_id_prefix}'"
        # Update in-memory cache on the LLM matcher (file write deferred to next _save_brain)
        from src.engine.matcher import _llm_matcher as _lm
        if _lm and len(arb.legs) >= 2:
            x, y = arb.legs[0].market_id, arb.legs[1].market_id
            key = (x, y) if x <= y else (y, x)
            _lm._cache[key] = False  # in-memory only; next _save_brain() persists it
        await self._store.permanently_suppress(arb.id)
        log.info("Dismissed arb %s — %s", arb.id, arb.event_name)
        return f"Dismissed '{arb.event_name[:50]}' — will never re-alert"

    async def detect_loop(self) -> None:
        """Hybrid arb detection:
          - Fast path (<100ms): check confirmed pairs on every WS price event
          - Slow path (60-90s): full O(n²) matching scan every 5 minutes

        The slow path runs in a thread executor. The fast path runs inline
        since it's pure arithmetic on in-memory Market objects.
        """
        from src.engine.matcher import _llm_matcher as _lm

        log.info("Waiting for initial feed data...")
        while not self._kalshi_markets or not self._poly_markets:
            await asyncio.sleep(2)

        pair_monitor = ConfirmedPairMonitor()
        if _lm:
            pair_monitor.load_from_llm_cache(_lm)

        # If brain has confirmed pairs, skip the expensive startup scan.
        # The fast path runs immediately; full scan runs in 5 minutes for new pairs.
        if pair_monitor.pair_count > 0:
            pair_monitor.last_full_scan_ts = time.time()

        log.info("Hybrid detection ready — %d confirmed pairs pre-loaded (fast path)",
                 pair_monitor.pair_count)

        # Pre-populate dashboard with last known arbs from DB so page isn't blank
        # during the window between startup and the first detection cycle.
        if self._store:
            _cutoff = datetime.utcnow() - timedelta(hours=24)
            _recent = await self._store.get_recent_opportunities(limit=50)
            _live_recent = [a for a in _recent if a.detected_at >= _cutoff]
            if _live_recent:
                stats = await self._trader.get_stats()
                markets = (self._poly_markets + self._kalshi_markets +
                           self._kalshi_sports_markets + self._pi_markets)
                await update_state(markets, _live_recent, stats)
                log.info("Dashboard pre-populated with %d recent arbs from DB", len(_live_recent))

        _last_tuner_run: float = 0.0
        _TUNER_INTERVAL = 86400
        _last_fast_path_ts: float = 0.0
        _FAST_PATH_DEBOUNCE = 2.0  # min seconds between fast path runs

        # Track arbs across cycles for dashboard continuity
        _last_arbs: list = []

        while True:
            try:
                if is_kill_switch_active() or await self._check_daily_loss_limit():
                    self._price_event.clear()
                    await asyncio.sleep(30)
                    continue

                now_ts = time.time()

                # Threshold tuner — once per day
                if now_ts - _last_tuner_run > _TUNER_INTERVAL:
                    loop = asyncio.get_event_loop()
                    new_thresholds = await loop.run_in_executor(
                        self._detect_executor, _tune_thresholds
                    )
                    if new_thresholds:
                        self._notifier._thresholds = new_thresholds
                    _last_tuner_run = now_ts

                markets = (
                    self._poly_markets
                    + self._kalshi_markets
                    + self._kalshi_sports_markets
                    + self._pi_markets
                    + self._opinion_markets
                    + self._gemini_markets
                    + self._hyperliquid_markets
                    + self._prophetx_markets
                )

                pair_monitor.update_market_index(markets)

                if pair_monitor.should_run_full_scan():
                    # ── SLOW PATH: full matching + pair discovery (every 5 min) ──
                    if _lm:
                        _lm.reset_cycle()
                    loop = asyncio.get_event_loop()
                    arbs, groups = await loop.run_in_executor(
                        self._detect_executor,
                        detect_arbs_with_groups,
                        markets, self._min_arb, self._stake,
                    )
                    pair_monitor.register_groups(groups)
                    pair_monitor.last_full_scan_ts = now_ts
                    log.info("Full scan: %d arbs | %d confirmed pairs",
                             len(arbs), pair_monitor.pair_count)
                    _last_arbs = arbs

                    for arb in arbs:
                        await self._store.save_opportunity(arb)
                    await self._notifier.notify_arbs(arbs)
                    await self._auto_execute(arbs)
                    stats = await self._trader.get_stats()
                    await update_state(markets, arbs, stats)

                elif pair_monitor.pair_count > 0 and (now_ts - _last_fast_path_ts) >= _FAST_PATH_DEBOUNCE:
                    # ── FAST PATH: check confirmed pairs only (<100ms) ──
                    _last_fast_path_ts = now_ts
                    arbs = pair_monitor.check_confirmed_pairs(self._min_arb, self._stake)
                    if arbs:
                        log.info("Fast path: %d arbs on confirmed pairs", len(arbs))
                        _last_arbs = arbs
                        for arb in arbs:
                            await self._store.save_opportunity(arb)
                        await self._notifier.notify_arbs(arbs)
                        await self._auto_execute(arbs)
                        stats = await self._trader.get_stats()
                        await update_state(markets, arbs, stats)

            except Exception:
                log.exception("Detection loop error")

            # Wake on WS price change OR 30s timeout
            self._price_event.clear()
            try:
                await asyncio.wait_for(self._price_event.wait(), timeout=30)
                log.debug("Detection triggered by WS price change")
            except asyncio.TimeoutError:
                pass

    async def poll_loop(self) -> None:
        ka = self.cfg.get("kalshi", {})
        po = self.cfg.get("polymarket", {})
        pi = self.cfg.get("predictit", {})

        tasks = [self.detect_loop()]

        # Polymarket real-time WebSocket
        _poly_ws = PolymarketWSFeed(self._price_event)
        tasks.append(_poly_ws.run(lambda: self._poly_markets))

        # Kalshi real-time WebSocket — requires RSA key (same as executor)
        _kalshi_key  = os.environ.get("KALSHI_API_KEY", "")
        _kalshi_pkey = os.environ.get("KALSHI_PRIVATE_KEY_PATH", "")
        if _kalshi_key and _kalshi_pkey:
            _kalshi_ws = KalshiWSFeed(_kalshi_key, _kalshi_pkey, self._price_event)
            tasks.append(_kalshi_ws.run(lambda: self._kalshi_markets))

        if self._kalshi_feed:
            tasks.append(self._feed_loop(
                "Kalshi", self._kalshi_feed.fetch, "_kalshi_markets",
                ka.get("poll_interval_seconds", 900),
            ))
        if self._kalshi_sports_feed:
            tasks.append(self._feed_loop(
                "KalshiSports", self._kalshi_sports_feed.fetch, "_kalshi_sports_markets",
                ka.get("sports_poll_interval_seconds", 10),  # fast poll for live games
            ))

        tasks.append(self._feed_loop(
            "Polymarket", self._poly_feed.fetch, "_poly_markets",
            po.get("poll_interval_seconds", 300),
        ))
        tasks.append(self._feed_loop(
            "PredictIt", self._predictit_feed.fetch, "_pi_markets",
            pi.get("poll_interval_seconds", 300),
        ))
        tasks.append(self._feed_loop(
            "Opinion", self._opinion_feed.fetch, "_opinion_markets",
            self.cfg.get("opinion", {}).get("poll_interval_seconds", 600),
        ))
        tasks.append(self._feed_loop(
            "Gemini", self._gemini_feed.fetch, "_gemini_markets",
            self.cfg.get("gemini", {}).get("poll_interval_seconds", 120),
        ))
        tasks.append(self._feed_loop(
            "Hyperliquid", self._hyperliquid_feed.fetch, "_hyperliquid_markets",
            self.cfg.get("hyperliquid", {}).get("poll_interval_seconds", 60),
        ))
        tasks.append(self._feed_loop(
            "ProphetX", self._prophetx_feed.fetch, "_prophetx_markets",
            self.cfg.get("prophetx", {}).get("poll_interval_seconds", 300),
        ))
        tasks.append(self._notifier.poll_telegram_commands(self._dismiss_arb))
        log.info("Cycle interval: 30s (detection) | feeds run independently")
        await asyncio.gather(*tasks)

    async def teardown(self) -> None:
        for feed in [self._kalshi_feed, self._kalshi_sports_feed,
                     self._poly_feed, self._predictit_feed,
                     self._opinion_feed, self._gemini_feed,
                     self._hyperliquid_feed, self._prophetx_feed]:
            if feed:
                try:
                    await feed.close()
                except Exception:
                    pass
        if self._executor:
            try:
                await self._executor.close()
            except Exception:
                pass
        if self._store:
            await self._store.close()
        if hasattr(self, "_detect_executor"):
            self._detect_executor.shutdown(wait=False, cancel_futures=True)


async def _demo_loop() -> None:
    """Inject synthetic data every 8s for UI preview without real API keys."""
    from src.demo import generate_demo_state
    log.info("DEMO MODE — generating synthetic arbs every 8s")
    while True:
        markets, arbs, stats = generate_demo_state()
        await update_state(markets, arbs, stats)
        await asyncio.sleep(8)


async def main(demo: bool = False) -> None:
    cfg = load_config()
    agent = Agent(cfg)
    await agent.setup()

    dash = cfg["dashboard"]
    server = uvicorn.Server(uvicorn.Config(
        app=app,
        host=dash["host"],
        port=dash["port"],
        log_level="warning",
    ))

    log.info("Dashboard → http://localhost:%d", dash["port"])

    if demo:
        await asyncio.gather(server.serve(), _demo_loop())
    else:
        active_feeds = [
            name for name, ok in [
                ("Kalshi (prediction)", bool(cfg["kalshi"].get("api_key"))),
                ("Kalshi (live sports)", bool(cfg["kalshi"].get("api_key"))),
                ("Polymarket-CLOB", True),
                ("PredictIt (direct)", True),
            ] if ok
        ]
        log.info("Active feeds: %s", ", ".join(active_feeds))
        log.info("Min arb margin: %.1f%%  |  Daily loss limit: %.1f%%",
                 cfg["paper_trading"]["min_arb_margin"] * 100,
                 cfg["paper_trading"].get("daily_loss_limit", 0.05) * 100)
        await asyncio.gather(server.serve(), agent.poll_loop())


if __name__ == "__main__":
    demo_mode = "--demo" in sys.argv
    _acquire_lock()
    try:
        asyncio.run(main(demo=demo_mode))
    except KeyboardInterrupt:
        log.info("Shutting down.")
    finally:
        _release_lock()
