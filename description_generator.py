"""
AI Description Generator — a manual copy/paste tool, NOT an autopost
tool. Bulk-ingests a batch of items, generates title + description
(brand/size/condition baked into the text) for each, and lets you
copy each field individually to paste into Depop yourself.

This exists specifically to sidestep Depop's shipping/package-size
autopost bug — nothing in this module ever submits or posts anything
to any marketplace. It is completely separate from the main "AI
Listing Generator" app (app.py) — it does not import from app.py and
app.py does not need to import from here except to call
render_description_generator() from behind the page router.
"""

import base64
import html as html_module
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps

from ai_listing import analyze_item, build_depop_description, image_to_data_url


UPLOAD_DIR = Path("uploads") / "description_generator"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

SLOTS_PER_BATCH = 10


# ============================================================
# STATE
# ============================================================

def _init_state():
    st.session_state.setdefault("descgen_items", [])
    st.session_state.setdefault("descgen_current_index", 0)
    st.session_state.setdefault("descgen_upload_counter", 0)
    # The starting item number for the batch of slots currently on
    # screen (1, then 11, then 21, ...) — same numbered-batch pattern
    # as the main app's upload deck, just without needing to replicate
    # its whole custom drag-and-drop component: plain
    # st.file_uploader per slot is already fast and, since each slot
    # IS the grouping (no AI auto-grouping step at all), skips a
    # whole round of vision-API calls the old bulk-upload flow paid
    # for — a real, direct speed win, not just a UI change.
    st.session_state.setdefault("descgen_batch_start", 1)
    # Bumped every time a batch is confirmed — st.file_uploader has no
    # programmatic "clear" method, so getting fresh empty slots for
    # the next batch means giving them a new key, forcing Streamlit to
    # mount brand-new (empty) uploader widgets.
    st.session_state.setdefault("descgen_slot_generation", 0)


# ============================================================
# INGESTION
# ============================================================

def _save_uploads(uploaded_files):
    """Saves raw uploaded bytes as-is (no EXIF-stripping rewrite) —
    grouping depends on DateTimeOriginal surviving the save, same
    reasoning as the main app's own bulk-upload saver."""
    saved_paths = []

    for uploaded_file in uploaded_files:
        safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", uploaded_file.name)
        counter = st.session_state["descgen_upload_counter"]
        st.session_state["descgen_upload_counter"] = counter + 1

        file_path = UPLOAD_DIR / f"batch_{counter}_{safe_name}"
        uploaded_file.seek(0)
        file_path.write_bytes(uploaded_file.getvalue())
        saved_paths.append(str(file_path))

    return saved_paths


def _split_brand_from_bullets(listing):
    """The generation prompt asks for brand as description_bullets[0]
    — split it back out into its own tracked field so brand edits can
    be re-spliced into the rebuilt description later without guessing
    which bullet was the brand."""
    bullets = list(listing.get("description_bullets", []) or [])
    brand = str(listing.get("brand", "") or "").strip()

    if bullets and brand and bullets[0].strip().lower() == brand.lower():
        other_bullets = bullets[1:]
    else:
        other_bullets = bullets

    return brand, other_bullets


def _item_from_listing(photos, listing):
    brand, other_bullets = _split_brand_from_bullets(listing)
    size = str(listing.get("size", "") or "")
    condition = str(listing.get("condition", "") or "")
    hashtags = listing.get("hashtags", []) or []

    return {
        "photos": photos,
        "title": str(listing.get("title", "") or ""),
        "brand": brand,
        "size": size,
        "condition": condition,
        "other_bullets": other_bullets,
        "hashtags": hashtags,
        "description": build_depop_description(condition, size, [brand] + other_bullets if brand else other_bullets, hashtags),
        "error": None,
        "ebay_comps": None,
    }


def _item_from_error(photos, error):
    return {
        "photos": photos,
        "title": "",
        "brand": "",
        "size": "",
        "condition": "",
        "other_bullets": [],
        "hashtags": [],
        "description": "",
        "error": str(error),
        "ebay_comps": None,
    }


def _generate_batch(slot_photo_lists):
    """slot_photo_lists: list of (item_number, [saved photo paths]) for
    every FILLED slot in the current batch — each slot already IS one
    item's grouping (no AI auto-grouping call needed at all, which is
    both simpler and meaningfully faster than the old bulk-upload +
    AI-grouping flow)."""
    total = len(slot_photo_lists)
    worker_count = min(3, max(1, total))
    progress = st.progress(0)
    status = st.empty()
    status.write(f"Generating {total} listing(s) in batches of {worker_count}...")

    def generate_one(entry):
        """Rate-limit-aware retries — same pattern as the main app's
        own generate_one (4 attempts, exponential backoff, longer
        delay specifically for rate-limit errors). This was
        previously a bare try/except with no retry at all, meaning a
        single transient rate-limit hit failed that item outright
        instead of recovering — a real gap under load, not just a
        missing nicety."""
        item_number, photos = entry
        last_error = None

        for attempt in range(4):
            try:
                listing = analyze_item(photos)
                if not listing:
                    raise RuntimeError("AI returned an empty listing.")
                return item_number, photos, listing, None
            except Exception as error:
                last_error = error
                if attempt < 3:
                    error_text = str(error).lower()
                    is_rate_limit = (
                        "rate_limit_exceeded" in error_text
                        or "429" in error_text
                        or "tokens per min" in error_text
                    )
                    delay = (10 if is_rate_limit else 3) * (2 ** attempt)
                    time.sleep(delay)

        return item_number, photos, None, last_error

    results_by_number = {}
    completed = 0

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(generate_one, entry) for entry in slot_photo_lists]
        for future in as_completed(futures):
            item_number, photos, listing, error = future.result()
            if listing is not None:
                results_by_number[item_number] = _item_from_listing(photos, listing)
            else:
                results_by_number[item_number] = _item_from_error(photos, error)

            completed += 1
            status.write(f"Generated {completed} of {total}...")
            progress.progress(completed / total)

    new_items = [results_by_number[num] for num, _ in slot_photo_lists]
    st.session_state["descgen_items"].extend(new_items)
    st.session_state["descgen_current_index"] = len(st.session_state["descgen_items"]) - total

    failed = sum(1 for item in new_items if item["error"])
    if failed:
        status.warning(f"Generated {total - failed} of {total} — {failed} failed (see item errors).")
    else:
        status.success(f"Generated {total} item(s).")


_SLOT_CARD_CSS = """
<style>
div[class*="st-key-descgen_slot_card_"] {
    border: 1px solid rgba(43,33,48,0.14);
    border-radius: 12px;
    background: #FFFFFF;
    padding: 12px;
    box-shadow: 0 4px 14px rgba(0,0,0,.05);
}
div[class*="st-key-descgen_slot_card_"] div[data-testid="stFileUploaderDropzone"] {
    background: #FBF3E3;
    border: 1px dashed rgba(43,33,48,0.25);
    border-radius: 8px;
    padding: 8px;
    min-height: 76px;
}
div[class*="st-key-descgen_slot_card_"] div[data-testid="stFileUploaderDropzoneInstructions"] span,
div[class*="st-key-descgen_slot_card_"] div[data-testid="stFileUploaderDropzoneInstructions"] small {
    font-size: 11px;
}
div[class*="st-key-descgen_slot_card_"] button {
    padding: 4px 10px !important;
    font-size: 12px !important;
}
div[class*="st-key-descgen_slot_card_"] div[data-testid="stFileUploaderFileName"] {
    font-size: 11px;
}
.descgen-slot-thumbs {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    margin-top: 6px;
}
.descgen-slot-thumbs img {
    width: 34px;
    height: 34px;
    object-fit: cover;
    border-radius: 4px;
    border: 1px solid rgba(43,33,48,0.18);
}
</style>
"""


@st.cache_data(show_spinner=False, max_entries=500)
def _thumb_data_url_from_bytes(file_bytes, max_dimension=120):
    """The actual PIL decode/resize/encode work, cached by the file's
    own bytes. Without this, every rerun (which Streamlit triggers on
    ANY widget interaction anywhere on the page) was redoing this for
    every already-uploaded photo in all 10 slots, not just whatever
    just changed — real, repeated full-resolution image processing on
    every click. This was very likely the direct cause of the iPhone
    upload "freeze": each additional photo dropped in re-processed
    every photo already sitting in every other slot too."""
    try:
        with Image.open(BytesIO(file_bytes)) as image:
            image = ImageOps.exif_transpose(image)
            if image.mode != "RGB":
                image = image.convert("RGB")
            image.thumbnail((max_dimension, max_dimension))
            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=70)
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            return f"data:image/jpeg;base64,{encoded}"
    except Exception:
        return None


def _thumb_data_url_from_upload(uploaded_file, max_dimension=120):
    """Quick thumbnail straight from the in-memory uploaded bytes —
    no disk write yet. Files only get saved once the batch is
    actually generated, so a slot can be filled/cleared/re-picked
    freely without piling up throwaway files on disk."""
    uploaded_file.seek(0)
    return _thumb_data_url_from_bytes(uploaded_file.getvalue(), max_dimension)


@st.fragment
def _render_ingestion():
    # Scopes reruns caused by interacting with THESE widgets (e.g.
    # dropping a photo into one slot) to just this fragment, instead
    # of re-executing the whole page — same pattern the main app uses
    # for its own upload deck, for the same reason: without it, every
    # single photo drop was re-rendering the entire Description
    # Generator page (including the item viewer below, once items
    # exist), not just the upload section.
    batch_start = st.session_state["descgen_batch_start"]
    generation = st.session_state["descgen_slot_generation"]
    batch_end = batch_start + SLOTS_PER_BATCH - 1

    st.markdown(f"#### Upload Items {batch_start}–{batch_end}")
    st.caption(
        "Drop each item's photos into its own slot below — same "
        "numbered-batch pattern as the main app. Once you generate "
        f"this batch, you'll get a fresh set of empty slots for "
        f"items {batch_end + 1}–{batch_end + SLOTS_PER_BATCH}."
    )
    st.html(_SLOT_CARD_CSS)

    slot_files = [None] * SLOTS_PER_BATCH
    cols_row1 = st.columns(5, gap="small")
    cols_row2 = st.columns(5, gap="small")
    all_cols = cols_row1 + cols_row2

    for slot_index in range(SLOTS_PER_BATCH):
        item_number = batch_start + slot_index
        with all_cols[slot_index]:
            with st.container(key=f"descgen_slot_card_{slot_index}"):
                st.markdown(f"**Item {item_number}**")
                files = st.file_uploader(
                    f"Item {item_number}",
                    type=["jpg", "jpeg", "png", "webp"],
                    accept_multiple_files=True,
                    key=f"descgen_slot_{slot_index}_{generation}",
                    label_visibility="collapsed",
                )
                slot_files[slot_index] = files
                if files:
                    thumbs_html = "".join(
                        f'<img src="{url}">'
                        for url in (
                            _thumb_data_url_from_upload(f) for f in files[:8]
                        )
                        if url
                    )
                    if thumbs_html:
                        st.markdown(
                            f'<div class="descgen-slot-thumbs">{thumbs_html}</div>',
                            unsafe_allow_html=True,
                        )
                    st.caption(f"{len(files)} photo(s)")

    filled_count = sum(1 for files in slot_files if files)

    if st.button(
        f"Generate This Batch ({filled_count} item(s)) →",
        type="primary",
        key="descgen_generate_batch",
        disabled=filled_count == 0,
        width="stretch",
    ):
        slot_photo_lists = []
        for slot_index in range(SLOTS_PER_BATCH):
            files = slot_files[slot_index]
            if not files:
                continue
            item_number = batch_start + slot_index
            saved_paths = _save_uploads(files)
            slot_photo_lists.append((item_number, saved_paths))

        _generate_batch(slot_photo_lists)

        # Advance to the next batch of 10 and force fresh empty
        # uploader widgets for it.
        st.session_state["descgen_batch_start"] = batch_start + SLOTS_PER_BATCH
        st.session_state["descgen_slot_generation"] = generation + 1
        st.rerun()


# ============================================================
# COPY BUTTON — real clipboard copy via st.html(unsafe_allow_javascript=True).
# st.markdown's HTML sanitizer strips both <script> tags AND inline
# onclick="" attributes, so a plain st.markdown button can't do this;
# st.html with that flag executes real JS in the actual page (not an
# iframe), which is what makes navigator.clipboard.writeText() work
# here. Text is passed via a hidden element's escaped text content
# (read back with .textContent in JS) rather than inlined into the
# script/attribute directly — sidesteps any quote/apostrophe escaping
# bugs entirely, since arbitrary listing text (descriptions, titles)
# can contain anything.
# ============================================================

_copy_button_counter = {"n": 0}


def _copy_button(text_value, label="Copy", big=False):
    _copy_button_counter["n"] += 1
    unique_id = f"descgen_copy_{_copy_button_counter['n']}"
    safe_text = html_module.escape(str(text_value or ""), quote=True)
    safe_label = html_module.escape(label)

    button_style = (
        """
            display: block;
            width: 100%;
            padding: 16px 18px;
            min-height: 52px;
            border-radius: 10px;
            border: 1px solid rgba(43,33,48,0.20);
            background: #2B2130;
            color: #FBF3E3;
            font-weight: 800;
            font-size: 16px;
            cursor: pointer;
            white-space: nowrap;
            margin: 8px 0 16px;
        """
        if big
        else """
            padding: 10px 16px;
            min-height: 44px;
            min-width: 44px;
            border-radius: 8px;
            border: 1px solid rgba(240,42,160,0.35);
            background: #F02AA0;
            color: #FFFFFF;
            font-weight: 700;
            font-size: 14px;
            cursor: pointer;
            white-space: nowrap;
        """
    )
    wrapper_style = "display:block; margin: 2px 0;" if big else "display:inline-block; margin: 2px 0;"

    st.html(
        f"""
        <div style="{wrapper_style}">
        <span id="{unique_id}_src" style="display:none">{safe_text}</span>
        <button id="{unique_id}_btn" style="{button_style}">{safe_label}</button>
        </div>
        <script>
        (function() {{
            var btn = document.getElementById("{unique_id}_btn");
            var src = document.getElementById("{unique_id}_src");
            if (!btn || !src || btn.dataset.bound) return;
            btn.dataset.bound = "1";
            btn.addEventListener("click", function() {{
                var text = src.textContent;
                var original = "{safe_label}";
                navigator.clipboard.writeText(text).then(function() {{
                    btn.innerText = "\\u2713 Copied";
                    setTimeout(function() {{ btn.innerText = original; }}, 1200);
                }}).catch(function() {{
                    btn.innerText = "Copy failed";
                    setTimeout(function() {{ btn.innerText = original; }}, 1200);
                }});
            }});
        }})();
        </script>
        """,
        unsafe_allow_javascript=True,
    )


# ============================================================
# LIVE-SYNC: editing Brand/Size/Condition rebuilds the description
# from the same deterministic template used everywhere else in the
# app (build_depop_description) — instant, no AI call, and it only
# ever changes the condition line / size line / brand bullet, never
# any other wording, since it's template assembly, not regeneration.
# ============================================================

def _sync_description(item, index):
    bullets = ([item["brand"]] if item["brand"] else []) + item["other_bullets"]
    item["description"] = build_depop_description(
        item["condition"], item["size"], bullets, item["hashtags"]
    )
    # The description text_area's widget key already has an entry in
    # session_state from its previous render — passing a new value=
    # on the next rerun does NOT override an existing keyed widget's
    # stored state in Streamlit (confirmed live: without this, the
    # box kept showing the stale pre-edit text). Writing it directly
    # here happens BEFORE the widget re-instantiates on the next run,
    # which is the safe order (same pattern used elsewhere in this
    # app for the same class of bug).
    st.session_state[f"descgen_desc_{index}"] = item["description"]


# ============================================================
# ONE-ITEM-AT-A-TIME VIEWER
# ============================================================

def _render_photo_strip(photos):
    if not photos:
        st.caption("No photos.")
        return
    cols = st.columns(min(len(photos), 5))
    for index, photo_path in enumerate(photos[:5]):
        with cols[index % len(cols)]:
            try:
                data_url = image_to_data_url(photo_path, max_dimension=700, quality=78)
                st.image(data_url, width="stretch")
            except Exception:
                st.caption("(photo failed to load)")
    if len(photos) > 5:
        st.caption(f"+ {len(photos) - 5} more photo(s) for this item")


# ============================================================
# PRICE COMPS — eBay only, on demand per item.
#
# The main app's comp pricing also scrapes Depop, which is
# deliberately throttled to ~1.5s PER ITEM (staggered) because
# scraping it faster has actually gotten it blocked before — for a
# batch of 10 that's a real wait, and repeating it here would bring
# back exactly the slowness this tab is meant to avoid. eBay's side
# is a plain authenticated REST call (Browse API, no scraping, no
# anti-bot throttling needed at all) — fast and safe either way, so
# it's the only source used here. Fetched lazily per item (a button
# on whichever item you're viewing), not for the whole batch
# automatically, and cached on the item once fetched so flipping
# Prev/Next doesn't lose it or re-fetch it.
# ============================================================

def _fetch_ebay_comps(item):
    # Both imports deferred to first actual use — same reasoning as
    # the OpenAI client: don't pay for something not every session
    # needs just to render the page.
    from ebay_scraper import search_ebay_comps, ebay_configured
    from market_scraper import build_market_query

    if not ebay_configured():
        return {"error": "eBay isn't configured (EBAY_CLIENT_ID / EBAY_CLIENT_SECRET missing)."}

    listing_like = {
        "title": item["title"],
        "brand": item["brand"],
        "garment_type": "",
    }
    query = build_market_query(listing_like)
    if not query:
        return {"error": "Not enough info to build a search query yet."}

    result = search_ebay_comps(query, limit=24)
    comps = result.get("comps", []) or []
    errors = result.get("errors", []) or []

    priced = sorted(
        c["price"] for c in comps
        if c.get("price") is not None and c["price"] > 0
    )

    if not priced:
        return {
            "query": query,
            "count": 0,
            "error": errors[0] if errors else "No priced eBay comps found for this search.",
        }

    import statistics
    median = statistics.median(priced)

    return {
        "query": query,
        "count": len(priced),
        "median": median,
        "low": min(priced),
        "high": max(priced),
        "recommended": round(median * 1.5, 2),
        "error": None,
    }


def _render_price_comps(item):
    st.markdown("#### Price Comps (eBay)")

    if item.get("ebay_comps") is None:
        if st.button("💰 Get Price Comps", key=f"descgen_get_comps_{item['title'][:20]}_{id(item)}"):
            with st.spinner("Checking eBay for comparable listings..."):
                item["ebay_comps"] = _fetch_ebay_comps(item)
            st.rerun()
        return

    comps = item["ebay_comps"]

    if comps.get("error"):
        st.caption(f"⚠️ {comps['error']}")
        if st.button("Retry", key=f"descgen_retry_comps_{id(item)}"):
            item["ebay_comps"] = None
            st.rerun()
        return

    price_col1, price_col2, price_col3 = st.columns(3)
    with price_col1:
        st.metric("Recommended", f"${comps['recommended']:.0f}")
    with price_col2:
        st.metric("Median", f"${comps['median']:.0f}")
    with price_col3:
        st.metric("Range", f"${comps['low']:.0f}–${comps['high']:.0f}")
    st.caption(f"Based on {comps['count']} active eBay listing(s) for \"{comps['query']}\"")


def _render_item_viewer():
    items = st.session_state["descgen_items"]
    total = len(items)
    index = st.session_state["descgen_current_index"]
    index = max(0, min(total - 1, index))
    st.session_state["descgen_current_index"] = index

    item = items[index]

    # ---- Detect a pending Brand/Size/Condition edit from the widget
    # interaction that triggered THIS rerun, and sync the description
    # BEFORE any widget below is instantiated. Streamlit does not
    # apply a keyed widget's new session_state value if that same
    # widget already rendered earlier in the current script run — the
    # Description text_area renders before the Details fields further
    # down, so the sync has to happen up here, not interleaved with
    # rendering (confirmed live: doing it inline down there left the
    # description showing stale text after an edit). ----
    field_changed = False
    for field_key in ("brand", "size", "condition"):
        widget_key = f"descgen_{field_key}_{index}"
        if widget_key in st.session_state:
            widget_value = st.session_state[widget_key]
            if widget_value != item[field_key]:
                item[field_key] = widget_value
                field_changed = True
    if field_changed:
        _sync_description(item, index)

    nav_prev, nav_label, nav_next = st.columns([1, 3, 1])
    with nav_prev:
        if st.button("← Prev", key="descgen_prev", disabled=index <= 0, width="stretch"):
            st.session_state["descgen_current_index"] = index - 1
            st.rerun()
    with nav_label:
        st.markdown(
            f"<div style='text-align:center; font-weight:800; padding-top:8px;'>"
            f"Item {index + 1} of {total}</div>",
            unsafe_allow_html=True,
        )
    with nav_next:
        if st.button("Next →", key="descgen_next", disabled=index >= total - 1, width="stretch"):
            st.session_state["descgen_current_index"] = index + 1
            st.rerun()

    st.markdown("---")

    if item["error"]:
        st.error(f"This item failed to generate: {item['error']}")
        return

    _render_photo_strip(item["photos"])

    # Depop has no separate title field on its own listing form — the
    # title has to be the first line of the one description box,
    # blank line, then the rest. This is the single button that
    # actually matches what you'd paste into Depop; Title/Description
    # below stay separately copyable too, for anything with real
    # separate fields (Shopify, eBay, etc.).
    depop_text = f"{item['title']}\n\n{item['description']}"
    _copy_button(depop_text, "📋 Copy for Depop (Title + Description)", big=True)

    st.markdown("#### Title")
    title_col, title_copy_col = st.columns([5, 1])
    with title_col:
        new_title = st.text_input(
            "Title", value=item["title"], key=f"descgen_title_{index}",
            label_visibility="collapsed",
        )
        item["title"] = new_title
    with title_copy_col:
        _copy_button(item["title"], "📋 Copy")

    st.markdown("#### Description")
    new_description = st.text_area(
        "Description", value=item["description"], key=f"descgen_desc_{index}",
        height=260, label_visibility="collapsed",
    )
    # Manual edits to the description itself are respected — only
    # brand/size/condition edits below trigger an automatic rebuild.
    item["description"] = new_description
    _copy_button(item["description"], "📋 Copy Description")

    st.markdown("#### Details")
    st.caption(
        "Editing Brand, Size, or Condition automatically updates the "
        "description above to match."
    )

    detail_cols = st.columns(3)
    field_specs = [
        ("brand", "Brand", detail_cols[0]),
        ("size", "Size", detail_cols[1]),
        ("condition", "Condition", detail_cols[2]),
    ]

    for field_key, field_label, col in field_specs:
        with col:
            # Change detection already happened above, before any
            # widget rendered this run — this just displays the
            # (possibly just-synced) current value.
            st.text_input(
                field_label, value=item[field_key], key=f"descgen_{field_key}_{index}",
            )
            _copy_button(item[field_key], f"📋 Copy {field_label}")

    st.markdown("---")
    _render_price_comps(item)


# ============================================================
# ENTRY POINT
# ============================================================

def render_description_generator():
    _init_state()

    st.markdown(
        """
        <div class="brand-step">
            <span class="brand-step-num">📋</span>
            <span class="brand-step-text">AI Description Generator</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Bulk-generate title + description text to copy-paste into Depop "
        "yourself — this tab never posts or submits anything automatically. "
        "Built specifically to route around Depop's shipping-field autopost bug."
    )

    batch_start = st.session_state["descgen_batch_start"]
    expander_label = (
        f"➕ Upload Items {batch_start}–{batch_start + SLOTS_PER_BATCH - 1}"
    )
    with st.expander(expander_label, expanded=not st.session_state["descgen_items"]):
        _render_ingestion()

    if not st.session_state["descgen_items"]:
        st.info("Upload photos into the slots above to get started.")
        return

    _render_item_viewer()
