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

import html as html_module
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import streamlit as st

from ai_listing import analyze_item, build_depop_description, image_to_data_url
from grouping import group_photos_with_ai, sort_paths_by_capture_time


UPLOAD_DIR = Path("uploads") / "description_generator"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# STATE
# ============================================================

def _init_state():
    st.session_state.setdefault("descgen_items", [])
    st.session_state.setdefault("descgen_current_index", 0)
    st.session_state.setdefault("descgen_upload_counter", 0)


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
    }


def _group_and_generate(uploaded_files):
    saved_paths = _save_uploads(uploaded_files)

    # group_photos_with_ai numbers photos 1-based against ITS OWN
    # internal capture-time sort — sort here first so the numbers it
    # hands back can be mapped to real paths correctly.
    records = sort_paths_by_capture_time(saved_paths)
    sorted_paths = [str(record["path"]) for record in records]

    with st.spinner(f"Grouping {len(sorted_paths)} photo(s) into items..."):
        try:
            result = group_photos_with_ai(sorted_paths)
        except Exception as error:
            st.error(f"Grouping failed: {error}")
            return
    groups = result.get("groups", []) if result else []

    if not groups:
        st.warning("No items could be grouped from these photos.")
        return

    jobs = []
    for group in groups:
        numbers = group.get("photo_numbers", []) or []
        photos = [
            sorted_paths[number - 1]
            for number in numbers
            if 0 < number <= len(sorted_paths)
        ]
        if photos:
            jobs.append(photos)

    if not jobs:
        st.warning("Grouping returned no usable items.")
        return

    total = len(jobs)
    worker_count = min(3, max(1, total))
    progress = st.progress(0)
    status = st.empty()
    status.write(f"Generating {total} listing(s) in batches of {worker_count}...")

    def generate_one(photos):
        try:
            listing = analyze_item(photos)
            if not listing:
                raise RuntimeError("AI returned an empty listing.")
            return photos, listing, None
        except Exception as error:
            return photos, None, error

    results_by_index = {}
    completed = 0

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(generate_one, photos): index
            for index, photos in enumerate(jobs)
        }
        for future in as_completed(futures):
            index = futures[future]
            photos, listing, error = future.result()
            if listing is not None:
                results_by_index[index] = _item_from_listing(photos, listing)
            else:
                results_by_index[index] = _item_from_error(photos, error)

            completed += 1
            status.write(f"Generated {completed} of {total}...")
            progress.progress(completed / total)

    new_items = [results_by_index[i] for i in range(total)]
    st.session_state["descgen_items"].extend(new_items)
    st.session_state["descgen_current_index"] = len(st.session_state["descgen_items"]) - total

    failed = sum(1 for item in new_items if item["error"])
    if failed:
        status.warning(f"Generated {total - failed} of {total} — {failed} failed (see item errors).")
    else:
        status.success(f"Generated {total} item(s).")


def _render_ingestion():
    st.markdown("#### Upload photos")
    st.caption(
        "Drop in every photo for this whole batch at once — items are "
        "grouped automatically, same as the main app's grouping."
    )

    uploaded_files = st.file_uploader(
        "Photos",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        key="descgen_uploader",
        label_visibility="collapsed",
    )

    if uploaded_files and st.button(
        f"Group & Generate ({len(uploaded_files)} photo(s))",
        type="primary",
        key="descgen_group_generate",
    ):
        _group_and_generate(uploaded_files)
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


def _copy_button(text_value, label="Copy"):
    _copy_button_counter["n"] += 1
    unique_id = f"descgen_copy_{_copy_button_counter['n']}"
    safe_text = html_module.escape(str(text_value or ""), quote=True)
    safe_label = html_module.escape(label)

    st.html(
        f"""
        <div style="display:inline-block; margin: 2px 0;">
        <span id="{unique_id}_src" style="display:none">{safe_text}</span>
        <button id="{unique_id}_btn" style="
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
        ">{safe_label}</button>
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

    with st.expander("➕ Add more items", expanded=not st.session_state["descgen_items"]):
        _render_ingestion()

    if not st.session_state["descgen_items"]:
        st.info("Upload photos above to get started.")
        return

    _render_item_viewer()
