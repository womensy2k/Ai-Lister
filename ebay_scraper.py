from __future__ import annotations

import base64
import os
import re
import threading
import time

import requests
from dotenv import load_dotenv

# ============================================================
# SELF-CONTAINED EBAY BROWSE API CLIENT
# Mirrors market_scraper.py's self-contained-module pattern so either
# source can be swapped/disabled without touching the other.
# ============================================================

load_dotenv()

EBAY_CLIENT_ID = os.getenv("EBAY_CLIENT_ID")
EBAY_CLIENT_SECRET = os.getenv("EBAY_CLIENT_SECRET")

# Production by default. Set EBAY_ENV=sandbox to point at eBay's sandbox
# endpoints instead (different keys are required for sandbox — the
# production Client ID/Secret will not work there).
_EBAY_ENV = os.getenv("EBAY_ENV", "production").strip().lower()
_API_ROOT = (
    "https://api.sandbox.ebay.com"
    if _EBAY_ENV == "sandbox"
    else "https://api.ebay.com"
)
_TOKEN_URL = f"{_API_ROOT}/identity/v1/oauth2/token"
_SEARCH_URL = f"{_API_ROOT}/buy/browse/v1/item_summary/search"

# eBay marketplace to search — override with EBAY_MARKETPLACE for a
# different country's catalog (e.g. EBAY_GB, EBAY_AU).
_MARKETPLACE_ID = os.getenv("EBAY_MARKETPLACE", "EBAY_US")

_token_lock = threading.Lock()
_cached_token = None  # (access_token, expires_at_epoch_seconds)


def ebay_configured() -> bool:
    """True once real Client ID/Secret are present in the environment."""
    return bool(EBAY_CLIENT_ID and EBAY_CLIENT_SECRET)


def _get_access_token() -> str:
    """
    Client-credentials OAuth2 token, cached in-process until ~60s before
    it actually expires. Thread-safe: concurrent callers block on the
    same lock rather than each firing their own token request.
    """
    global _cached_token

    with _token_lock:
        if _cached_token is not None:
            token, expires_at = _cached_token
            if time.time() < expires_at - 60:
                return token

        if not ebay_configured():
            raise RuntimeError(
                "EBAY_CLIENT_ID / EBAY_CLIENT_SECRET were not found. "
                "Add them to your .env file."
            )

        basic = base64.b64encode(
            f"{EBAY_CLIENT_ID}:{EBAY_CLIENT_SECRET}".encode("utf-8")
        ).decode("ascii")

        response = requests.post(
            _TOKEN_URL,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Authorization": f"Basic {basic}",
            },
            data={
                "grant_type": "client_credentials",
                "scope": "https://api.ebay.com/oauth/api_scope",
            },
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()

        token = payload["access_token"]
        expires_in = int(payload.get("expires_in", 7200))
        _cached_token = (token, time.time() + expires_in)

        return token


def _clean(value) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def ebay_item_web_url(item_web_url: str) -> str:
    return _clean(item_web_url)


def search_ebay_comps(query: str, limit: int = 24) -> dict:
    """
    Search eBay's Browse API for active listings matching `query`.

    Returns comps in the same shape market_scraper.py's MarketComp uses
    (as plain dicts, so this module doesn't need to import
    market_scraper.py — avoids any circular-import risk):

        {"source": "eBay", "title": str, "price": float|None,
         "url": str, "sold": False}

    Like the Depop scraper, this reads ACTIVE listing asking prices, not
    confirmed sold prices — the Browse API doesn't expose sold data
    without eBay's separately-gated Marketplace Insights API access.
    Matching Depop's own methodology keeps the two sources comparable.

    On any failure (missing keys, network error, bad response) this
    returns an empty comps list with a non-empty "errors" list rather
    than raising — callers merge multiple sources and one failing
    should not take the others down with it.
    """
    query = _clean(query)

    if not query:
        return {"comps": [], "errors": []}

    if not ebay_configured():
        return {
            "comps": [],
            "errors": ["EBAY_CLIENT_ID / EBAY_CLIENT_SECRET not configured."],
        }

    try:
        token = _get_access_token()

        response = requests.get(
            _SEARCH_URL,
            headers={
                "Authorization": f"Bearer {token}",
                "X-EBAY-C-MARKETPLACE-ID": _MARKETPLACE_ID,
            },
            params={
                "q": query,
                "limit": min(max(int(limit), 1), 50),
            },
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()

    except Exception as exc:
        return {"comps": [], "errors": [str(exc)]}

    comps = []

    for item in payload.get("itemSummaries", []) or []:
        price_block = item.get("price") or {}

        try:
            price = float(price_block.get("value"))
        except (TypeError, ValueError):
            price = None

        if price is not None and not (0 < price < 10000):
            price = None

        title = _clean(item.get("title", "")) or "eBay listing"

        comps.append(
            {
                "source": "eBay",
                "title": title[:300],
                "price": price,
                "url": ebay_item_web_url(item.get("itemWebUrl", "")),
                "sold": False,
            }
        )

    return {"comps": comps, "errors": []}
