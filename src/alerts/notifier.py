from __future__ import annotations

import json
import logging
import os
import subprocess
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from src.models import ArbOpportunity, ArbLeg, Source

if TYPE_CHECKING:
    from src.storage.db import Store

_ARB_LOG = Path(__file__).parent.parent.parent / "brain" / "arb_log.jsonl"

log = logging.getLogger(__name__)

_TELEGRAM_TOKEN   = "8305669215:AAF4yquHVqTnoJ9XoSX0vZCZjCiQcxxJBLo"
_TELEGRAM_CHAT_ID = "7960884508"

# GitHub Pages URL for the Mini App redirect page
_MINI_APP_BASE = "https://george-clear-cov.github.io/Arb-Agent/poly-open.html"

MIN_MARGIN_PCT      = 5.0   # net ROI threshold — lowered from 10% to catch game arbs
MIN_MARGIN_PCT_SLOW = 10.0  # higher bar for outright/long-dated prediction markets
MAX_DAYS            = 30    # only alert on arbs expiring within 30 days


def _days_left(expires_at: datetime) -> float:
    now = datetime.now(tz=timezone.utc)
    exp = expires_at if expires_at.tzinfo else expires_at.replace(tzinfo=timezone.utc)
    return (exp - now).total_seconds() / 86400


def _market_url(leg: ArbLeg) -> str | None:
    mid = leg.market_id
    if leg.source in (Source.KALSHI, Source.KALSHI_SPORTS):
        return f"https://kalshi.com/markets/{mid}"
    if leg.source == Source.POLYMARKET:
        if mid and not mid.isdigit():
            return f"__poly__{mid}"
        return None
    if leg.source == Source.PREDICTIT:
        return f"https://www.predictit.org/markets/detail/{mid}"
    return None


async def _resolve_poly_urls(slug: str, client) -> tuple[str, str]:
    """Return (web_url, app_url) for a Polymarket market slug.

    web_url — canonical /event/... URL, opens in browser at the right market
    app_url — polymarket.us/events/{event-slug} universal link opened via
              Telegram Mini App so iOS handles it natively
    """
    fallback = f"https://polymarket.com/market/{slug}"
    web_url  = fallback
    try:
        resp     = await client.get(fallback, follow_redirects=False, timeout=5)
        location = resp.headers.get("location", "")
        if "/event/" in location:
            if location.startswith("http"):
                web_url = location
            else:
                web_url = f"https://polymarket.com{location}"
    except Exception:
        pass
    # Extract event slug: /event/{event-slug}/{market-slug} → events/{event-slug}
    path  = web_url.split("polymarket.com", 1)[-1]
    parts = path.strip("/").split("/")
    if len(parts) >= 2 and parts[0] in ("event", "events"):
        app_url = f"https://polymarket.us/events/{parts[1]}"
    else:
        app_url = f"https://polymarket.us{path}"
    return web_url, app_url


def _mini_app_url(app_url: str) -> str:
    """Wrap a polymarket.us deep link in the GitHub Pages Mini App redirect URL."""
    return f"{_MINI_APP_BASE}?link={urllib.parse.quote(app_url, safe='')}"


def _mac_notify(title: str, message: str) -> None:
    try:
        subprocess.Popen(["afplay", "/System/Library/Sounds/Glass.aiff"])
        subprocess.Popen([
            "osascript", "-e",
            f'display notification "{message}" with title "{title}"',
        ])
    except Exception:
        pass


class Notifier:
    def __init__(self, min_margin: float, slack_webhook: str | None = None,
                 store: "Store | None" = None):
        self.min_margin = min_margin
        self._slack_webhook = slack_webhook
        self._store = store

    async def notify_arbs(self, arbs: list[ArbOpportunity]) -> None:
        for arb in arbs:
            if arb.expires_at is None:
                continue
            days = _days_left(arb.expires_at)
            if days <= 0 or days > MAX_DAYS:
                continue
            threshold = MIN_MARGIN_PCT if days <= 1 else MIN_MARGIN_PCT_SLOW
            if arb.margin * 100 < threshold:
                continue

            if self._store and await self._store.has_notified(arb.id):
                continue
            if self._store:
                await self._store.mark_notified(arb.id)

            self._log_arb(arb)
            _mac_notify(
                title=f"Arb {arb.margin*100:.1f}% — {arb.sport.upper()}",
                message=f"{arb.event_name[:60]}  +${arb.profit:.0f}  {days:.0f}d left",
            )
            if self._slack_webhook:
                await self._slack(arb)
            await self._telegram(arb)

    def _log_arb(self, arb: ArbOpportunity) -> None:
        legs = "  |  ".join(
            f"{l.outcome_name} @ {l.price:.3f} [{l.source.value}/{l.bookmaker}]"
            for l in arb.legs
        )
        log.info(
            "ARB DETECTED  id=%s  margin=%.2f%%  profit=$%.2f  event=%r  legs=[%s]",
            arb.id, arb.margin * 100, arb.profit, arb.event_name, legs,
        )
        try:
            _ARB_LOG.parent.mkdir(parents=True, exist_ok=True)
            entry = {
                "id": arb.id,
                "ts": datetime.now(tz=timezone.utc).isoformat(),
                "sport": arb.sport,
                "event": arb.event_name,
                "margin_pct": round(arb.margin * 100, 3),
                "profit_usd": round(arb.profit, 2),
                "expires_at": arb.expires_at.isoformat() if arb.expires_at else None,
                "legs": [
                    {"source": l.source.value, "market_id": l.market_id,
                     "outcome": l.outcome_name, "price": l.price}
                    for l in arb.legs
                ],
            }
            with open(_ARB_LOG, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as exc:
            log.debug("arb_log write failed: %s", exc)

    async def _slack(self, arb: ArbOpportunity) -> None:
        try:
            import httpx
            days = _days_left(arb.expires_at) if arb.expires_at else 0
            text = (
                f":money_with_wings: *Arb {arb.margin*100:.1f}%* — {arb.event_name}  ({days:.0f}d left)\n"
                + "\n".join(
                    f"  • {l.outcome_name} @ {l.price:.3f}  bet ${l.stake:.0f}  [{l.bookmaker or l.source.value}]"
                    for l in arb.legs
                )
                + f"\n  Total: ${arb.total_stake:.0f}  →  Profit: *${arb.profit:.0f}*"
            )
            async with httpx.AsyncClient() as client:
                await client.post(self._slack_webhook, json={"text": text}, timeout=5)
        except Exception:
            log.exception("Slack notification failed")

    async def _telegram(self, arb: ArbOpportunity) -> None:
        try:
            import httpx

            margin_pct = arb.margin * 100
            days = _days_left(arb.expires_at) if arb.expires_at else 0
            sport_emoji = {
                "baseball": "⚾", "basketball": "🏀", "hockey": "🏒",
                "football": "🏈", "soccer": "⚽", "tennis": "🎾",
                "mma": "🥊", "boxing": "🥊",
            }.get(arb.sport, "🎯")

            if days < 1:
                expiry_str = f"{days*24:.0f}h left"
            else:
                expiry_str = f"{days:.0f}d left"

            api_url = f"https://api.telegram.org/bot{_TELEGRAM_TOKEN}/sendMessage"
            async with httpx.AsyncClient(timeout=5.0) as client:
                raw_urls = {l: _market_url(l) for l in arb.legs}
                poly_urls: dict = {}
                for l, u in raw_urls.items():
                    if u and u.startswith("__poly__"):
                        poly_urls[l] = await _resolve_poly_urls(u[len("__poly__"):], client)

                leg_lines = []
                for l in arb.legs:
                    raw = raw_urls.get(l)
                    bookmaker = l.bookmaker or l.source.value
                    if l in poly_urls:
                        web, _ = poly_urls[l]
                        link = f'<a href="{web}">{bookmaker}</a>'
                    elif raw:
                        link = f'<a href="{raw}">{bookmaker}</a>'
                    else:
                        link = bookmaker
                    leg_lines.append(
                        f"  └ <b>{l.outcome_name}</b> @ {link}  {l.price:.3f}  → <b>bet ${l.stake:.0f}</b>"
                    )

                sources = " × ".join(sorted({l.source.value for l in arb.legs}))
                text = (
                    f"{sport_emoji} <b>ARB {margin_pct:.1f}%</b>  ⏱ {expiry_str}\n"
                    f"<b>{arb.event_name}</b>\n"
                    f"Platforms: {sources}\n\n"
                    + "\n".join(leg_lines)
                    + f"\n\n💰 Put in <b>${arb.total_stake:.0f}</b>  →  lock in <b>${arb.profit:.0f}</b>"
                )

                # Build web_app inline buttons for each Polymarket leg.
                # Telegram Mini App context allows openLink() → Safari → universal link.
                keyboard_rows = []
                seen_app_urls: set[str] = set()
                for l, (web, app) in poly_urls.items():
                    if app in seen_app_urls:
                        continue
                    seen_app_urls.add(app)
                    bookmaker = l.bookmaker or l.source.value
                    keyboard_rows.append([{
                        "text": f"📱 Open {bookmaker} in app",
                        "web_app": {"url": _mini_app_url(app)},
                    }])

                payload: dict = {
                    "chat_id": _TELEGRAM_CHAT_ID,
                    "text": text,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                }
                if keyboard_rows:
                    payload["reply_markup"] = {"inline_keyboard": keyboard_rows}

                await client.post(api_url, json=payload)
        except Exception:
            log.warning("Telegram notification failed", exc_info=True)
