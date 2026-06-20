from __future__ import annotations

import json
import logging
import os
import subprocess
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
            # Store slug so _resolve_poly_url can build the deep link async
            return f"__poly__{mid}"
        return None
    if leg.source == Source.PREDICTIT:
        return f"https://www.predictit.org/markets/detail/{mid}"
    return None


async def _resolve_poly_url(slug: str, client) -> str:
    """Resolve /market/{slug} → canonical /event/... path, then prepend /us/ for iOS deep link."""
    try:
        resp = await client.get(
            f"https://polymarket.com/market/{slug}",
            follow_redirects=False, timeout=5,
        )
        location = resp.headers.get("location", "")
        # location is like https://polymarket.com/event/world-cup-winner/will-iraq-...
        if "/event/" in location:
            path = location.split("polymarket.com", 1)[-1]  # /event/slug/market-slug
            return f"https://polymarket.com/us{path}"
    except Exception:
        pass
    # Fallback: /us/market/ still opens the app, just not to the right market
    return f"https://polymarket.com/us/market/{slug}"


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
            # Game arbs expiring within 24h: lower threshold (prices move fast)
            # Long-dated / outright markets: require higher margin
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
                # Resolve Polymarket URLs (needs redirect follow to get canonical event path)
                raw_urls = {l: _market_url(l) for l in arb.legs}
                resolved_urls: dict = {}
                for l, u in raw_urls.items():
                    if u and u.startswith("__poly__"):
                        resolved_urls[l] = await _resolve_poly_url(u[len("__poly__"):], client)
                    else:
                        resolved_urls[l] = u

                leg_lines = []
                for l in arb.legs:
                    url = resolved_urls.get(l)
                    bookmaker = l.bookmaker or l.source.value
                    link = f'<a href="{url}">{bookmaker}</a>' if url else bookmaker
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
                await client.post(api_url, json={
                    "chat_id": _TELEGRAM_CHAT_ID,
                    "text": text,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": False,
                })
        except Exception:
            log.warning("Telegram notification failed", exc_info=True)
