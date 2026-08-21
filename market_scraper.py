from __future__ import annotations

import re
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from urllib.parse import quote_plus

from ebay_scraper import search_ebay_comps

def _get_sync_playwright():
    """Load Playwright only when a scrape is actually requested."""
    try:
        import importlib
        playwright_sync = importlib.import_module(
            "playwright.sync_api"
        )
        return playwright_sync.sync_playwright
    except Exception as exc:
        raise RuntimeError(
            "Playwright is required for Depop scraping. "
            "Install it with: pip install playwright && "
            "playwright install chromium"
        ) from exc


# ============================================================
# SELF-CONTAINED DEPOP SCRAPER
# IMPORTANT: this module intentionally does NOT import
# market_scraper.py. That prevents the circular-import problem
# caused by market_scraper importing itself.
# ============================================================

@dataclass
class MarketComp:
    source: str
    title: str
    price: float | None
    url: str
    sold: bool = False


def _clean(value) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _money(value):
    if value is None:
        return None

    text = _clean(value)
    if not text:
        return None

    matches = re.findall(
        r"(?:US\$|\$)\s*([0-9]{1,4}(?:\.[0-9]{1,2})?)",
        text,
    )

    if not matches:
        if re.fullmatch(r"\d{1,4}(?:\.\d{1,2})?", text):
            try:
                number = float(text)
                return number if 0 < number < 10000 else None
            except Exception:
                return None
        return None

    try:
        number = float(matches[0])
        return number if 0 < number < 10000 else None
    except Exception:
        return None


def _dedupe(comps):
    seen = set()
    output = []

    for comp in comps:
        key = comp.url or (
            comp.title.lower(),
            comp.price,
        )

        if key in seen:
            continue

        seen.add(key)
        output.append(comp)

    return output


def build_market_query(listing: dict) -> str:
    """
    Build a focused Depop search from the generated listing.
    """
    if not isinstance(listing, dict):
        return ""

    title = _clean(listing.get("title", ""))
    brand = _clean(listing.get("brand", ""))
    garment = _clean(
        listing.get(
            "garment_type",
            listing.get("type", ""),
        )
    )

    tokens = re.findall(r"[A-Za-z0-9']+", title)

    stop = {
        "vintage", "y2k", "fairycore", "coquette",
        "cute", "womens", "women", "woman", "girls",
        "girl", "size", "new", "rare", "authentic",
        "excellent", "condition", "with", "the", "and",
        "for", "this", "that", "top", "shirt",
        "bottom", "clothing",
    }

    useful = []
    for token in tokens:
        low = token.lower()
        if len(low) >= 3 and low not in stop and low not in {
            x.lower() for x in useful
        }:
            useful.append(token)

    parts = []

    if brand and brand.lower() not in {
        "unknown", "-", "n/a", "none"
    }:
        parts.append(brand)

    parts.extend(useful[:3])

    if garment and garment.lower() not in {
        "unknown", "-", "n/a", "none"
    }:
        parts.append(garment)

    if not parts:
        parts = [title] if title else ["womens clothing"]

    return " ".join(parts[:6]).strip()


def depop_search_url(query: str) -> str:
    return (
        "https://www.depop.com/search/"
        "?q="
        + quote_plus(_clean(query))
    )


# ============================================================
# PERFORMANCE FIX: BROWSER REUSE PER WORKER THREAD
# ============================================================
# The old version called p.chromium.launch(headless=True) inside a fresh
# `with sync_playwright() as p:` block for EVERY single search — a cold
# Chromium launch (typically 1-3s) on every item, run automatically after
# every listing batch. Playwright's sync API requires all calls for a
# given Playwright/Browser instance to happen on the thread that created
# them, so a browser can't simply be shared across the whole
# ThreadPoolExecutor — but it CAN be created once per worker thread and
# reused for every job that thread picks up afterward. ThreadPoolExecutor
# reuses its worker threads across submitted tasks, so once
# scrape_depop_all is processing more items than there are workers, this
# now saves a full browser cold-start on every job after the first one
# each thread handles.
#
# Browsers are kept alive for the lifetime of the worker thread (which,
# in this app, lives for the lifetime of the Streamlit server process).

_thread_local = threading.local()


def _get_thread_browser():
    browser = getattr(_thread_local, "browser", None)
    playwright = getattr(_thread_local, "playwright", None)

    if browser is not None:
        try:
            # Cheap liveness check; a crashed/closed browser raises here.
            _ = browser.contexts
            return browser
        except Exception:
            browser = None

    sync_playwright = _get_sync_playwright()
    playwright = sync_playwright().start()
    browser = playwright.chromium.launch(
        headless=True,
        args=["--disable-gpu", "--disable-dev-shm-usage"],
    )

    _thread_local.playwright = playwright
    _thread_local.browser = browser

    return browser


def _scrape_one_depop(query: str, limit: int = 24, start_delay: float = 0.0) -> dict:
    comps = []

    if start_delay > 0:
        time.sleep(start_delay)

    try:
        browser = _get_thread_browser()
    except Exception as exc:
        return {
            "query": query,
            "depop_url": depop_search_url(query),
            "comps": [],
            "summary": {},
            "errors": [str(exc)],
        }

    context = None

    try:
        context = browser.new_context(
            viewport={"width": 1280, "height": 900},
            locale="en-US",
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/151.0.0.0 Safari/537.36"
            ),
        )

        page = context.new_page()

        try:
            page.goto(
                depop_search_url(query),
                wait_until="domcontentloaded",
                timeout=12000,
            )

            page.wait_for_timeout(700)

            body = _clean(
                page.locator("body").inner_text(timeout=6000)
            )

            if any(
                phrase in body.lower()
                for phrase in (
                    "verify you are human",
                    "access denied",
                    "captcha",
                )
            ):
                return {
                    "query": query,
                    "depop_url": depop_search_url(query),
                    "comps": [],
                    "summary": {},
                    "errors": [
                        "Depop returned an access/security page."
                    ],
                }

            anchors = page.locator("a[href*='/products/']")
            count = min(anchors.count(), limit * 3)
            seen_urls = set()

            for i in range(count):
                if len(comps) >= limit:
                    break

                a = anchors.nth(i)

                try:
                    href = a.get_attribute("href") or ""

                    if not href:
                        continue

                    if href.startswith("/"):
                        href = "https://www.depop.com" + href

                    if href in seen_urls:
                        continue

                    seen_urls.add(href)

                except Exception:
                    continue

                price = None
                card_text = ""

                node = a

                for _ in range(8):
                    try:
                        candidate = _clean(
                            node.inner_text(timeout=350)
                        )
                    except Exception:
                        candidate = ""

                    if candidate and len(candidate) <= 900:
                        candidate_price = _money(candidate)

                        if candidate_price is not None:
                            price = candidate_price
                            card_text = candidate
                            break

                    for selector in (
                        "[itemprop='price']",
                        "[data-testid*='price']",
                        "[class*='price']",
                        "meta[itemprop='price']",
                    ):
                        try:
                            loc = node.locator(selector)

                            for j in range(
                                min(loc.count(), 2)
                            ):
                                raw = (
                                    loc.nth(j).get_attribute(
                                        "content"
                                    )
                                    or loc.nth(j).get_attribute(
                                        "aria-label"
                                    )
                                    or _clean(
                                        loc.nth(j).inner_text(
                                            timeout=250
                                        )
                                    )
                                )

                                candidate_price = _money(raw)

                                if candidate_price is not None:
                                    price = candidate_price
                                    card_text = candidate
                                    break

                            if price is not None:
                                break

                        except Exception:
                            pass

                    if price is not None:
                        break

                    try:
                        node = node.locator(
                            "xpath=.."
                        )
                    except Exception:
                        break

                title = ""

                for selector in (
                    "[data-testid*='title']",
                    "[class*='title']",
                    "h3",
                    "h2",
                    "h1",
                ):
                    try:
                        loc = a.locator(selector)

                        if loc.count():
                            candidate = _clean(
                                loc.first.inner_text(
                                    timeout=350
                                )
                            )

                            if len(candidate) > 3:
                                title = candidate
                                break

                    except Exception:
                        pass

                if not title:
                    try:
                        title = _clean(
                            a.inner_text(timeout=450)
                        )
                    except Exception:
                        title = ""

                if not title:
                    title = "Depop listing"

                comps.append(
                    MarketComp(
                        source="Depop",
                        title=title[:300],
                        price=price,
                        url=href,
                        sold=False,
                    )
                )

            comps = _dedupe(comps)[:limit]

            priced = sorted(
                c.price
                for c in comps
                if c.price is not None
                and c.price > 0
            )

            if priced:
                median = statistics.median(priced)
                low = min(priced)
                high = max(priced)

                recommended = round(
                    median * 1.50,
                    2,
                )
            else:
                median = None
                low = None
                high = None
                recommended = None

            return {
                "query": query,
                "depop_url": depop_search_url(query),
                "comps": [
                    asdict(comp)
                    for comp in comps
                ],
                "summary": {
                    "count": len(comps),
                    "priced_count": len(priced),
                    "median": median,
                    "low": low,
                    "high": high,
                    "recommended": recommended,
                    "pricing_multiplier": 1.50,
                },
                "errors": [],
            }

        except Exception as exc:
            return {
                "query": query,
                "depop_url": depop_search_url(query),
                "comps": [],
                "summary": {},
                "errors": [str(exc)],
            }

    finally:
        # Only the context is closed per-scrape — the browser itself is
        # kept alive on the thread-local for reuse by the next job this
        # worker thread picks up.
        if context is not None:
            try:
                context.close()
            except Exception:
                pass


def _summarize(priced: list[float]) -> dict:
    if priced:
        median = statistics.median(priced)
        low = min(priced)
        high = max(priced)
        recommended = round(median * 1.50, 2)
    else:
        median = None
        low = None
        high = None
        recommended = None

    return {
        "count": len(priced),
        "priced_count": len(priced),
        "median": median,
        "low": low,
        "high": high,
        "recommended": recommended,
        "pricing_multiplier": 1.50,
    }


def _scrape_one_market(query: str, limit: int, start_delay: float = 0.0) -> dict:
    """
    One item's combined market comps: Depop (scraped) + eBay (Browse
    API), merged into a single deduped comp list with ONE combined
    median/low/high/recommended computed across both sources — plus a
    per-source breakdown (still handy for a "DEPOP: $x / EBAY: $y"
    style detail view) in case that's ever wanted alongside the
    combined numbers. Each source's own comps are independently error-
    tolerant: eBay failing (bad/missing keys, network error) does not
    blank out Depop's comps, and vice versa — errors from both are
    collected, not just the first one hit.
    """
    depop_result = _scrape_one_depop(query, limit, start_delay)

    try:
        ebay_result = search_ebay_comps(query, limit)
    except Exception as exc:
        ebay_result = {"comps": [], "errors": [str(exc)]}

    depop_comps = [
        MarketComp(**comp) for comp in depop_result.get("comps", [])
    ]
    ebay_comps = [
        MarketComp(**comp) for comp in ebay_result.get("comps", [])
    ]

    combined_comps = _dedupe(depop_comps + ebay_comps)[: limit * 2]

    combined_priced = sorted(
        c.price for c in combined_comps if c.price is not None and c.price > 0
    )
    depop_priced = sorted(
        c.price for c in depop_comps if c.price is not None and c.price > 0
    )
    ebay_priced = sorted(
        c.price for c in ebay_comps if c.price is not None and c.price > 0
    )

    errors = list(depop_result.get("errors", [])) + list(
        ebay_result.get("errors", [])
    )

    return {
        "query": query,
        "depop_url": depop_search_url(query),
        "comps": [asdict(comp) for comp in combined_comps],
        "summary": _summarize(combined_priced),
        "sources": {
            "depop": _summarize(depop_priced),
            "ebay": _summarize(ebay_priced),
        },
        "errors": errors,
    }


def scrape_market_all(
    listings: list[dict],
    limit: int = 24,
    workers: int = 2,
    stagger_seconds: float = 1.5,
) -> dict[int, dict]:
    """
    Scrape ALL generated listings concurrently across BOTH Depop and
    eBay, combined into one median/low/high per item.

    All Depop jobs share one browser pool on the same outbound IP (see
    _get_thread_browser above) — firing every search at once looks like
    a burst from a single source and Depop's bot detection can flag it
    after the very first request goes through fine, leaving every
    later item with an "access denied"/"captcha" error and no median
    (confirmed live: only item 1 got a price, items 2+ all got the
    access-denied error). `workers` defaults lower (2, not 4) and each
    job additionally sleeps `stagger_seconds * job_position` before its
    actual Depop request, so requests land spread out in time even
    within the smaller worker pool. eBay's Browse API is an
    authenticated call, not scraped, so it isn't staggered — it just
    rides along inside the same per-item job right after the (staggered)
    Depop call.

    Returns:
        {
            0: {...Item 1 combined market data...},
            1: {...Item 2 combined market data...},
            ...
        }
    """
    jobs = []

    for index, listing in enumerate(listings):
        query = build_market_query(listing)

        if query:
            jobs.append(
                (index, query)
            )

    results = {}

    if not jobs:
        return results

    max_workers = max(
        1,
        min(
            int(workers),
            len(jobs),
        ),
    )

    with ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:

        future_map = {
            executor.submit(
                _scrape_one_market,
                query,
                limit,
                job_position * stagger_seconds,
            ): (index, query)
            for job_position, (index, query) in enumerate(jobs)
        }

        for future in as_completed(
            future_map
        ):
            index, query = future_map[
                future
            ]

            try:
                results[index] = future.result()
            except Exception as exc:
                results[index] = {
                    "query": query,
                    "depop_url": depop_search_url(
                        query
                    ),
                    "comps": [],
                    "summary": {},
                    "sources": {},
                    "errors": [str(exc)],
                }

    # Real fallback for a genuine scrape failure (access-denied, captcha,
    # timeout — `errors` non-empty), as distinct from an item that
    # legitimately has zero comps (errors empty, summary just {} with no
    # median). Retry those ONE time, sequentially and after a pause —
    # not concurrently, since firing right back at whatever just got
    # flagged would likely repeat the same block.
    job_by_index = dict(jobs)
    failed_indices = [
        index for index in results
        if results[index].get("errors")
    ]

    for index in failed_indices:
        time.sleep(stagger_seconds)
        try:
            retried = _scrape_one_market(
                job_by_index[index],
                limit,
            )
        except Exception:
            continue

        if not retried.get("errors"):
            results[index] = retried

    return results


# Backwards-compatible alias — scrape_market_all is the combined
# Depop+eBay entry point; scrape_depop_all now just points at it so
# any existing caller keeps working without a rename.
scrape_depop_all = scrape_market_all
