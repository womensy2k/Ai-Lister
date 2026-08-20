import streamlit as st
import streamlit.components.v1 as components
import base64
import io
import json
from PIL import Image, ImageOps
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import re
from pathlib import Path
import os
import base64

# PUBLIC MARKET SCRAPING
# Requires: pip install playwright && playwright install chromium

from ai_listing import (
    analyze_item,
    regenerate_field,
    SHOPIFY_CATEGORY_PATHS,
    SHOPIFY_CATEGORY_GIDS,
)
from market_scraper import (
    build_market_query,
    scrape_market,
    ebay_search_url,
    depop_search_url,
    get_ebay_debug,
)


# ============================================================
# ADAPTIVE USER PRICING PROFILE
# ============================================================

PRICING_PROFILE_FILE = Path(__file__).with_name("pricing_profile.json")
BASE_PRICING_MULTIPLIER = 1.50  # 50% above market median


def _load_pricing_profile():
    if not PRICING_PROFILE_FILE.exists():
        return {}
    try:
        data = json.loads(PRICING_PROFILE_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_pricing_profile(profile):
    try:
        PRICING_PROFILE_FILE.write_text(
            json.dumps(profile, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    except Exception:
        # Pricing learning should never break listing generation.
        pass


def _pricing_style_bucket(listing):
    title = str(listing.get("title", "") or "").lower()
    return "vintage" if "vintage" in title else "non_vintage"


def _pricing_title_signature(listing):
    """
    Keep only useful title/style signals. Do not key on the entire title,
    because every item title is unique.
    """
    title = str(listing.get("title", "") or "").lower()
    stop = {
        "vintage", "y2k", "fairycore", "coquette", "cute", "womens",
        "women", "woman", "girls", "girl", "size", "new", "rare",
        "authentic", "excellent", "condition", "with", "the", "and",
        "for", "top", "shirt", "bottom", "clothing",
    }
    tokens = re.findall(r"[a-z0-9]+", title)
    useful = [t for t in tokens if len(t) >= 4 and t not in stop]
    return "|".join(useful[:3])


def _pricing_bucket_keys(listing):
    brand = str(listing.get("brand", "") or "").strip().lower() or "unknown"
    garment = str(
        listing.get("garment_type", listing.get("type", "")) or ""
    ).strip().lower() or "unknown"
    condition = str(listing.get("condition", "") or "").strip().lower() or "unknown"
    style = _pricing_style_bucket(listing)
    title_sig = _pricing_title_signature(listing)

    # Most-specific -> progressively broader fallbacks.
    keys = [
        f"brand={brand}|garment={garment}|condition={condition}|style={style}",
        f"brand={brand}|garment={garment}|style={style}",
        f"garment={garment}|condition={condition}|style={style}",
        f"garment={garment}|style={style}",
        f"style={style}",
    ]

    if title_sig:
        keys.insert(
            0,
            f"brand={brand}|garment={garment}|style={style}|title={title_sig}",
        )

    return keys


def _get_learned_multiplier(listing):
    """
    Return the user's learned markup multiplier.

    With no history, both vintage and non-vintage start at 1.50 (50% over
    median). As the user manually changes prices, the profile learns the
    ratio of their chosen price to the observed market median.
    """
    profile = _load_pricing_profile()

    for key in _pricing_bucket_keys(listing):
        record = profile.get(key)
        if isinstance(record, dict):
            try:
                multiplier = float(record.get("multiplier"))
                count = int(record.get("count", 0))
            except (TypeError, ValueError):
                continue

            if count >= 1:
                return max(0.75, min(multiplier, 3.00)), key, count

    return BASE_PRICING_MULTIPLIER, _pricing_bucket_keys(listing)[0], 0


def _record_manual_price(listing, manual_price, median):
    """
    Learn from the user's actual price change.

    The target is not "50% forever". It starts at 50%, then moves toward the
    user's observed manual median/market ratio. A small EWMA prevents one
    unusually expensive or cheap item from completely changing the strategy.
    """
    if median is None or median <= 0 or manual_price <= 0:
        return

    target_multiplier = float(manual_price) / float(median)
    target_multiplier = max(0.75, min(target_multiplier, 3.00))

    profile = _load_pricing_profile()
    key = _pricing_bucket_keys(listing)[0]
    existing = profile.get(key, {})

    try:
        old_multiplier = float(existing.get("multiplier", BASE_PRICING_MULTIPLIER))
        count = int(existing.get("count", 0))
    except (TypeError, ValueError):
        old_multiplier = BASE_PRICING_MULTIPLIER
        count = 0

    # First manual price gets meaningful influence; later edits smooth out.
    alpha = 0.50 if count == 0 else 0.25
    learned = (
        target_multiplier
        if count == 0
        else old_multiplier * (1 - alpha) + target_multiplier * alpha
    )
    learned = max(0.75, min(learned, 3.00))

    profile[key] = {
        "multiplier": round(learned, 4),
        "count": count + 1,
        "last_manual_price": round(float(manual_price), 2),
        "last_market_median": round(float(median), 2),
        "style": _pricing_style_bucket(listing),
        "brand": str(listing.get("brand", "") or ""),
        "garment_type": str(
            listing.get("garment_type", listing.get("type", "")) or ""
        ),
        "condition": str(listing.get("condition", "") or ""),
    }
    _save_pricing_profile(profile)
    return learned

def format_depop_description(description, title, size, condition, hashtags):
    """
    Build the final Depop description in the exact structure requested.

    Controlled fields:
      - title
      - Size X (omitted completely when size == "-")
      - condition wording
      - required styling line
      - the exact same five generated hashtags

    The AI-generated middle copy is preserved, but duplicated title/size/
    condition/styling/hashtag lines are removed so they cannot conflict.
    """
    description = str(description or "").strip()
    title = str(title or "").strip()
    size = str(size or "").strip()
    condition = str(condition or "").strip()

    # Normalize hashtags to exactly the generated listing values.
    if isinstance(hashtags, str):
        tags = [x.strip() for x in hashtags.split() if x.strip()]
    else:
        tags = [str(x).strip() for x in (hashtags or []) if str(x).strip()]

    tags = [
        tag if tag.startswith("#") else "#" + tag
        for tag in tags
    ]

    # Remove the title if AI already put it as the first line.
    body = description
    if title and body.lower().startswith(title.lower()):
        body = body[len(title):].lstrip(" \n\r-:")

    # Remove old controlled size/condition/styling/hashtag lines.
    body = re.sub(
        r"(?im)^[ \t]*(?:tagged\s+size|fits\s+like)\b.*(?:\n|$)",
        "",
        body,
    )
    body = re.sub(
        r"(?im)^[ \t]*(?:size)\s*[:\-]?\s*(?:[A-Za-z0-9/]+)\s*$",
        "",
        body,
    )
    body = re.sub(
        r"(?im)^[ \t]*excellent condition(?:[^\n]*)?(?:\n|$)",
        "",
        body,
    )
    body = re.sub(
        r"(?im)^[ \t]*shirts may be styled, pinned, tucked, or tied to better show off the fit\.\s*$",
        "",
        body,
    )
    body = re.sub(
        r"(?im)^[ \t]*#[A-Za-z0-9_+-]+(?:\s+#[A-Za-z0-9_+-]+)*\s*$",
        "",
        body,
    )

    # Strip accidental markdown links around hashtags if the AI returned them.
    body = re.sub(
        r"\[([^\]]*#[A-Za-z0-9_+-]+[^\]]*)\]\([^)]+\)",
        r"\1",
        body,
    )

    # Remove excessive whitespace while preserving the natural body paragraph.
    body = re.sub(r"\n{3,}", "\n\n", body).strip()

    sections = []

    # User requested title first.
    if title:
        sections.append(title)

    # AI's natural item copy comes next.
    if body:
        sections.append(body)

    # Size is a dedicated line and disappears completely when unknown/manual.
    if size and size != "-":
        sections.append(f"Size {size}")

    # Keep the requested condition wording.
    condition_text = (
        "Excellent condition with no stains or flaws that I'm aware of."
        if condition.lower().startswith("excellent")
        else (
            f"{condition} with no stains or flaws that I'm aware of."
            if condition
            else "Excellent condition with no stains or flaws that I'm aware of."
        )
    )
    sections.append(condition_text)

    # Keep this exact seller line.
    sections.append(
        "Shirts may be styled, pinned, tucked, or tied to better show off the fit."
    )

    # The exact same generated hashtags used by the listing.
    if tags:
        sections.append(" ".join(tags[:5]))

    return "\n\n".join(sections).strip()


def sync_description_size(description, title, size, condition, hashtags):
    """Rebuild controlled description fields when the Size dropdown changes."""
    return format_depop_description(
        description,
        title,
        size,
        condition,
        hashtags,
    )


# ============================================================
# SHOPIFY CONTROLLED LISTING FIELDS
# ============================================================

# Keep these JSON files beside app.py.
# They are the source of truth for the selectable Shopify values.
TOPS_TAXONOMY_FILE = Path(__file__).with_name("shopify_taxonomy_options.json")
JEANS_TAXONOMY_FILE = Path(__file__).with_name("shopify_taxonomy_options_jeans.json")


def _load_taxonomy(path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


SHOPIFY_TAXONOMY = _load_taxonomy(TOPS_TAXONOMY_FILE)
SHOPIFY_JEANS_TAXONOMY = _load_taxonomy(JEANS_TAXONOMY_FILE)

SHOPIFY_FIELD_NAMES = {
    "color": "Color",
    "pattern": "Pattern",
    "sleeve_length_type": "Sleeve length type",
    "age_group": "Age group",
    "neckline": "Neckline",
    "target_gender": "Target gender",
    "top_length_type": "Top length type",
    "care_instructions": "Care instructions",
    "fabric": "Fabric",
    "size": "Size",
    "clothing_features": "Clothing features",
    "sleeve_style": "Sleeve style",
    "stretch_level": "Stretch level",
    "top_fit": "Top fit",
}


def _shopify_values(field, taxonomy=None, attribute_name=None):
    taxonomy = taxonomy if taxonomy is not None else SHOPIFY_TAXONOMY
    name = attribute_name or SHOPIFY_FIELD_NAMES.get(field)
    if not name:
        return []

    values = (
        taxonomy.get("attributes", {})
        .get(name, {})
        .get("values", [])
    )

    out = []
    for value in values:
        if isinstance(value, dict):
            value = value.get("name")
        if value and str(value).strip():
            out.append(str(value).strip())
    return out


def _controlled_select(
    label,
    field,
    value,
    key,
    custom=True,
    taxonomy=None,
    attribute_name=None,
):
    options = _shopify_values(
        field,
        taxonomy=taxonomy,
        attribute_name=attribute_name,
    )

    if not options:
        options = ["Other"]

    current = str(value or "").strip()
    if field == "size" and current.lower() in {"unknown", "none", "n/a", "not visible", "not readable"}:
        current = "-"
    size_detected = field == "size" and bool(current) and current != "-"

    choices = ["— Select —"] + options
    if field == "size" and "-" not in choices:
        choices.insert(1, "-")

    # Preserve an actual AI-detected size even when Shopify's list does not
    # contain a composite value such as 32/34 or 32x34.
    if size_detected and current not in choices:
        choices.insert(1, current)

    if custom and "Custom" not in choices:
        choices.append("Custom")

    if current in choices:
        index = choices.index(current)
    elif field == "size":
        index = choices.index("-") if "-" in choices else 0
    elif current:
        index = choices.index("Custom") if custom else (
            choices.index("Other") if "Other" in choices else 0
        )
    else:
        index = 0

    selected = st.selectbox(
        label,
        choices,
        index=index,
        key=f"{key}_select",
        help=(
            "Select from Shopify's allowed values. "
            "Choose Custom only when needed."
        ),
    )

    if selected == "Custom":
        return st.text_input(
            f"{label} — Custom",
            value=current if current and current not in options else "",
            key=f"{key}_custom",
        ).strip()

    if selected == "— Select —":
        return "-" if field == "size" else ""

    return selected


def _controlled_multiselect(
    label,
    field,
    value,
    key,
    taxonomy=None,
    attribute_name=None,
):
    options = _shopify_values(
        field,
        taxonomy=taxonomy,
        attribute_name=attribute_name,
    )

    current = value if isinstance(value, list) else []
    current = [x for x in current if x in options]

    return st.multiselect(
        label,
        options=options,
        default=current,
        key=key,
        help="Select Shopify-approved values.",
    )


def _taxonomy_key(label):
    return re.sub(
        r"[^a-z0-9]+",
        "_",
        str(label).lower(),
    ).strip("_")


def _is_jeans_garment(value):
    normalized = str(value or "").strip().lower()
    return normalized in {
        "jeans",
        "jean",
        "denim jeans",
        "denim jean",
    }


def _garment_type_select(value, key):
    options = [
        "Tank Top", "Cami", "Crop Top", "Baby Tee", "T-Shirt",
        "Long Sleeve Top", "Blouse", "Button Up", "Polo",
        "Halter Top", "Tube Top", "Corset Top", "Bodysuit",
        "Cardigan", "Sweater", "Hoodie", "Sweatshirt", "Tunic",
        "Jacket", "Coat", "Vest", "Dress", "Skirt", "Pants",
        "Jeans", "Shorts", "Leggings", "Jumpsuit", "Romper", "Other",
    ]

    aliases = {
        "tank": "Tank Top",
        "tank top": "Tank Top",
        "camisole": "Cami",
        "cami top": "Cami",
        "crop": "Crop Top",
        "cropped top": "Crop Top",
        "tee": "T-Shirt",
        "t shirt": "T-Shirt",
        "t-shirt": "T-Shirt",
        "long sleeve": "Long Sleeve Top",
        "long-sleeve": "Long Sleeve Top",
        "long sleeve shirt": "Long Sleeve Top",
        "long sleeve top": "Long Sleeve Top",
        "hooded sweatshirt": "Hoodie",
        "pullover hoodie": "Hoodie",
        "crewneck": "Sweatshirt",
        "crewneck sweatshirt": "Sweatshirt",
        "crew neck sweatshirt": "Sweatshirt",
        "cardigan sweater": "Cardigan",
        "corset": "Corset Top",
        "button down": "Button Up",
        "button-down": "Button Up",
        "polo shirt": "Polo",
    }

    current = str(value or "").strip()
    canonical = next(
        (x for x in options if x.lower() == current.lower()),
        None,
    )

    if not canonical:
        canonical = aliases.get(current.lower(), current)

    choices = ["— Select —"] + options + ["Custom"]

    if canonical in choices:
        index = choices.index(canonical)
    elif canonical:
        index = choices.index("Custom")
    else:
        index = 0

    selected = st.selectbox(
        "Garment Type",
        choices,
        index=index,
        key=f"{key}_select",
        help="Choose the most accurate garment type.",
    )

    if selected == "Custom":
        return st.text_input(
            "Garment Type — Custom",
            value=canonical if canonical not in options else "",
            key=f"{key}_custom",
        ).strip()

    if selected == "— Select —":
        return "Other"

    return selected


def _render_shopify_attributes(listing, item_index, garment_type):
    """
    Render Shopify-controlled attributes.

    Jeans use shopify_taxonomy_options_jeans.json.
    Everything else uses shopify_taxonomy_options.json.
    """
    taxonomy = (
        SHOPIFY_JEANS_TAXONOMY
        if _is_jeans_garment(garment_type)
        else SHOPIFY_TAXONOMY
    )

    attributes = taxonomy.get("attributes", {})

    if not attributes:
        st.warning(
            "Shopify taxonomy file was not found or contains no attributes. "
            "Keep the appropriate JSON file beside app.py."
        )
        return {}

    # Shopify Clothing Features is a multi-value attribute.
    multiselect_fields = {"Clothing features"}

    if _is_jeans_garment(garment_type):
        preferred_order = [
            "Size", "Color", "Pattern", "Fabric", "Target gender",
            "Age group", "Clothing features", "Distressing style",
            "Pocket style", "Waistband style",
        ]
    else:
        preferred_order = [
            "Size", "Color", "Pattern", "Target gender", "Age group",
            "Fabric", "Clothing features", "Sleeve length type",
            "Sleeve style", "Neckline", "Top length type",
            "Stretch level", "Top fit", "Care instructions",
        ]

    ordered_names = (
        [name for name in preferred_order if name in attributes]
        + [name for name in attributes if name not in preferred_order]
    )

    values_out = {}

    for attribute_name in ordered_names:
        field = _taxonomy_key(attribute_name)
        key = f"generated_shopify_{field}_{item_index}"
        current = listing.get(field, "")

        if attribute_name in multiselect_fields:
            value = _controlled_multiselect(
                attribute_name,
                field,
                current,
                key,
                taxonomy=taxonomy,
                attribute_name=attribute_name,
            )
        else:
            value = _controlled_select(
                attribute_name,
                field,
                current,
                key,
                taxonomy=taxonomy,
                attribute_name=attribute_name,
            )

        values_out[field] = value

    return values_out



UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

st.set_page_config(
    page_title="Depop AI",
    page_icon="👕",
    layout="wide"
)


# ============================================================
# OPTIONAL DRAG/DROP
# ============================================================

try:
    from streamlit_dnd import dnd
    DND_AVAILABLE = True
except ImportError:
    DND_AVAILABLE = False


# ============================================================
# WHOLE ITEM DROP / REORDER COMPONENT
# ============================================================
#
# Streamlit V2 component:
# - no iframe
# - no declare_component()
# - no separate frontend server
# - no CDN / network dependency
#
# This is deliberate. The previous V1 iframe was the source of the
# blank/flashing item card.

try:
    from streamlit.components.v2 import component as _streamlit_v2_component
    STREAMLIT_V2_AVAILABLE = True
except ImportError:
    _streamlit_v2_component = None
    STREAMLIT_V2_AVAILABLE = False


ITEM_CARD_HTML = """
<div class="item-card">
  <div class="upload-overlay" data-upload-overlay>
    <div class="spinner"></div>
    <div>Uploading photos… please wait</div>
  </div>

  <div class="item-header">
    <img class="item-thumb" data-thumb alt="">
    <div class="item-heading">
      <div class="item-title" data-title>Item 1</div>
      <div class="item-meta" data-meta>0 photos</div>
    </div>
    <span class="status-dot" title="Ready"></span>
    <button class="delete-btn" type="button" data-delete>Delete Item</button>
  </div>

  <div class="photos-heading">Photos in this item</div>
  <div class="photos-grid" data-photos></div>
</div>
"""


ITEM_CARD_CSS = """
.item-card {
  position: relative;
  width: 100%;
  min-height: 300px;
  padding: 16px;
  border: 1px solid #343944;
  border-radius: 10px;
  background: #0f1218;
  color: #f4f5f7;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  overflow: hidden;
  transition: border-color .12s ease, background .12s ease, box-shadow .12s ease;
}

.item-card.file-over {
  border: 2px solid #22c55e;
  background: #13251a;
  box-shadow: 0 0 0 4px rgba(34,197,94,.12), 0 0 28px rgba(34,197,94,.18);
}

.item-card.uploading .upload-overlay {
  display: flex;
}

.upload-overlay {
  display: none;
  position: absolute;
  inset: 0;
  z-index: 100;
  background: rgba(10,13,18,.78);
  backdrop-filter: blur(7px);
  align-items: center;
  justify-content: center;
  flex-direction: column;
  gap: 10px;
  font-size: 18px;
  font-weight: 700;
}

.spinner {
  width: 30px;
  height: 30px;
  border: 3px solid rgba(255,255,255,.18);
  border-top-color: #22c55e;
  border-radius: 50%;
  animation: spin .7s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.item-header {
  display: flex;
  align-items: center;
  gap: 14px;
  margin-bottom: 16px;
}

.item-thumb {
  width: 86px;
  height: 64px;
  border-radius: 8px;
  object-fit: cover;
  background: #242833;
  border: 1px solid #3a404c;
  flex: 0 0 auto;
  display: none;
}

.item-heading {
  flex: 1;
  min-width: 0;
}

.item-title {
  font-size: 24px;
  font-weight: 750;
  margin-bottom: 5px;
}

.item-meta {
  font-size: 14px;
  color: #a6adb9;
}

.status-dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: #22c55e;
  box-shadow: 0 0 10px rgba(34,197,94,.55);
  flex: 0 0 auto;
}

.delete-btn {
  border: 1px solid #4b5563;
  color: #fca5a5;
  background: #1d212a;
  border-radius: 8px;
  padding: 9px 13px;
  cursor: pointer;
  font-size: 14px;
}

.delete-btn:hover {
  border-color: #ef4444;
  background: #281c20;
}

.photos-heading {
  font-weight: 700;
  margin: 0 0 12px;
}

.photos-grid {
  display: grid;
  grid-template-columns: repeat(5, minmax(100px, 1fr));
  gap: 14px;
  min-height: 125px;
}

.item-card.item-dragging {
  opacity: .42;
  transform: scale(.985);
  border-color: #8a929f;
  box-shadow: 0 8px 28px rgba(0,0,0,.28);
}
.item-card.item-drag-target {
  border-color: #858e9d;
  background: #191e26;
  box-shadow: 0 0 0 2px rgba(255,255,255,.06);
  animation: itemWiggle .42s ease-in-out infinite;
}
.item-card.item-drop-before { box-shadow: inset 0 3px 0 #8a929f; }
.item-card.item-drop-after { box-shadow: inset 0 -3px 0 #8a929f; }
@keyframes itemWiggle {
  0%,100% { transform: rotate(0deg); }
  25% { transform: rotate(-.35deg); }
  75% { transform: rotate(.35deg); }
}

.photo-card {
  position: relative;
  border: 1px solid #414754;
  border-radius: 9px;
  padding: 5px;
  background: #171b22;
  cursor: grab;
  user-select: none;
  touch-action: none;
  transition: transform .18s cubic-bezier(.2,.8,.2,1),
              opacity .18s ease,
              border-color .18s ease,
              box-shadow .18s ease;
  will-change: transform;
}

.photo-card:hover {
  border-color: #5b6472;
}

.photo-card.dragging {
  opacity: .28;
  transform: scale(.96);
  border-color: #7a8390;
}

.photo-card.reorder-nearby {
  animation: tinyShake .16s ease;
}

.photo-card.photo-selected {
  border-color: #22c55e;
  box-shadow: 0 0 0 3px rgba(34,197,94,.18);
  transform: translateY(-2px);
}

.photo-card.photo-move-target {
  border-color: #60a5fa;
  box-shadow: inset 0 0 0 2px rgba(96,165,250,.22);
}

.photo-card {
  touch-action: none;
  user-select: none;
  -webkit-user-select: none;
  transition: transform 180ms cubic-bezier(.2,.8,.2,1), box-shadow 160ms ease;
}

.photo-card.pointer-dragging {
  cursor: grabbing;
  z-index: 99999;
  opacity: .96;
  box-shadow: 0 20px 42px rgba(0,0,0,.35);
}

.photo-card.pointer-dragging img {
  pointer-events: none;
}

.photo-card.reorder-placeholder {
  opacity: .28;
  border: 2px dashed #7a8390;
  background: rgba(255,255,255,.04);
  box-shadow: none;
}

.photo-card.reorder-placeholder > * {
  visibility: hidden;
}

@keyframes tinyShake {
  0%   { transform: translateX(0); }
  35%  { transform: translateX(-1px); }
  70%  { transform: translateX(1px); }
  100% { transform: translateX(0); }
}

.drop-placeholder {
  min-height: 145px;
  border: 1px solid #6b7280;
  border-radius: 9px;
  background: #292e37;
  box-shadow: inset 0 0 0 1px rgba(255,255,255,.035);
  transition: transform .16s cubic-bezier(.2,.8,.2,1),
              opacity .16s ease;
  animation: placeholderPulse .55s ease-in-out infinite alternate;
}

@keyframes placeholderPulse {
  from { opacity: .62; }
  to { opacity: 1; }
}

.photo-card img {
  width: 100%;
  aspect-ratio: 1.3;
  object-fit: cover;
  display: block;
  border-radius: 8px;
  background: #20242d;
}

.photo-actions {
  position: absolute;
  top: 7px;
  left: 7px;
  right: 7px;
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  z-index: 5;
  pointer-events: none;
}

.photo-action {
  width: 26px;
  height: 26px;
  border: 1px solid rgba(255,255,255,.22);
  border-radius: 7px;
  background: rgba(10,13,18,.78);
  color: #f4f5f7;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 15px;
  line-height: 1;
  cursor: pointer;
  pointer-events: auto;
  backdrop-filter: blur(4px);
  box-shadow: 0 2px 8px rgba(0,0,0,.25);
}

.photo-action:hover {
  background: rgba(30,35,44,.95);
}

.photo-action.delete-photo {
  color: #fca5a5;
}

.photo-action.rotate-photo {
  color: #d1d5db;
}

.photo-name {
  color: #8e96a3;
  font-size: 11px;
  margin-top: 6px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.photo-number {
  color: #9ca3af;
  font-size: 12px;
  font-weight: 700;
  margin-top: 3px;
}

.empty-drop {
  min-height: 125px;
  border: 1px dashed #414754;
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #8f98a7;
  grid-column: 1 / -1;
}
"""


ITEM_CARD_JS = """
export default function(component) {
  const { data, parentElement, setTriggerValue } = component;

  const root = parentElement.querySelector(".item-card");
  const title = parentElement.querySelector("[data-title]");
  const meta = parentElement.querySelector("[data-meta]");
  const thumb = parentElement.querySelector("[data-thumb]");
  const photosEl = parentElement.querySelector("[data-photos]");
  const deleteBtn = parentElement.querySelector("[data-delete]");

  let photos = [];
  let draggedId = null;
  let draggedSourceItem = null;
  let dropPlaceholder = null;
  let selectedPhotoId = null;
  let pointerDrag = null;
  let draggedItemNumber = null;
  let itemGhost = null;
  let externalDepth = 0;

  function emit(action) {
    setTriggerValue("action", action);
  }

  function render() {
    // A successful trigger causes Streamlit to rerun Python and send the
    // updated item back. Always clear the temporary upload overlay first.
    root.classList.remove("uploading");

    const itemNumber = Number(data?.item_number || 1);

    try {
      photos = JSON.parse(data?.photos_json || "[]");
      if (!Array.isArray(photos)) photos = [];
    } catch {
      photos = [];
    }

    title.textContent = "Item " + itemNumber;
    meta.textContent = photos.length + " photos";
    // The card itself can be dragged, but photo cards must be allowed to
    // win the drag gesture when the user grabs an individual photo.
    root.draggable = true;
    root.dataset.itemNumber = String(itemNumber);

    if (photos.length && photos[0].src) {
      thumb.src = photos[0].src;
      thumb.style.display = "block";
    } else {
      thumb.removeAttribute("src");
      thumb.style.display = "none";
    }

    photosEl.innerHTML = "";

    if (!photos.length) {
      const empty = document.createElement("div");
      empty.className = "empty-drop";
      empty.textContent = "Drop photos anywhere on this item";
      photosEl.appendChild(empty);
      return;
    }

    photos.forEach((photo, index) => {
      const card = document.createElement("div");
      card.className = "photo-card";
      card.draggable = true;
      card.dataset.id = String(photo.id);

      const image = document.createElement("img");
      image.src = photo.src || "";
      image.alt = photo.name || ("Photo " + (index + 1));
      image.draggable = false;
      // Let the photo-card own the reorder drag; don't let the browser
      // start its native image drag instead.
      image.draggable = false;

      // Editing controls live on top of the image:
      // rotate on the upper-left, delete on the upper-right.
      const actions = document.createElement("div");
      actions.className = "photo-actions";

      const rotateBtn = document.createElement("button");
      rotateBtn.type = "button";
      rotateBtn.className = "photo-action rotate-photo";
      rotateBtn.title = "Rotate photo 90°";
      rotateBtn.textContent = "↻";
      rotateBtn.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        emit({
          type: "rotate",
          id: String(photo.id)
        });
      });

      const deletePhotoBtn = document.createElement("button");
      deletePhotoBtn.type = "button";
      deletePhotoBtn.className = "photo-action delete-photo";
      deletePhotoBtn.title = "Remove photo from this item";
      deletePhotoBtn.textContent = "×";
      deletePhotoBtn.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        emit({
          type: "remove",
          id: String(photo.id)
        });
      });

      actions.appendChild(rotateBtn);
      actions.appendChild(deletePhotoBtn);

      const number = document.createElement("div");
      number.className = "photo-number";
      number.textContent = "#" + (index + 1);

      const name = document.createElement("div");
      name.className = "photo-name";
      name.textContent = photo.name || "";

      card.appendChild(image);
      card.appendChild(actions);
      card.appendChild(number);
      card.appendChild(name);

      card.addEventListener("mousedown", () => {
        // Prevent the parent item card from stealing this gesture.
        root.draggable = false;
      });

      // iPhone-style pointer reorder: hold a photo, lift it, move the
      // pointer across another photo's center, and the other photos animate
      // into the newly opened slot.
      card.addEventListener("pointerdown", (event) => {
        if (event.button !== 0) return;
        if (event.target.closest(".photo-action")) return;

        event.preventDefault();
        event.stopPropagation();

        const rect = card.getBoundingClientRect();
        pointerDrag = {
          id: String(photo.id),
          sourceItem: itemNumber,
          card,
          pointerId: event.pointerId,
          startX: event.clientX,
          startY: event.clientY,
          offsetX: event.clientX - rect.left,
          offsetY: event.clientY - rect.top,
          active: false,
          placeholder: null
        };

        try { card.setPointerCapture(event.pointerId); } catch {}
      });

      function gridCards() {
        return Array.from(photosEl.querySelectorAll(".photo-card"))
          .filter(el => el !== pointerDrag?.card && !el.classList.contains("reorder-placeholder"));
      }

      function flip(mutate) {
        const cards = gridCards();
        const first = new Map(cards.map(el => [el, el.getBoundingClientRect()]));
        mutate();
        cards.forEach(el => {
          const a = first.get(el);
          const b = el.getBoundingClientRect();
          const dx = a.left - b.left;
          const dy = a.top - b.top;
          if (Math.abs(dx) < 1 && Math.abs(dy) < 1) return;
          el.style.transition = "none";
          el.style.transform = `translate(${dx}px, ${dy}px)`;
          requestAnimationFrame(() => {
            el.style.transition = "transform 180ms cubic-bezier(.2,.8,.2,1)";
            el.style.transform = "translate(0,0)";
          });
        });
      }

      function movePlaceholder(event) {
        const drag = pointerDrag;
        if (!drag?.placeholder) return;
        const cards = gridCards();
        if (!cards.length) return;

        let closest = null;
        let best = Infinity;
        for (const el of cards) {
          const r = el.getBoundingClientRect();
          const d = Math.hypot(
            event.clientX - (r.left + r.width / 2),
            event.clientY - (r.top + r.height / 2)
          );
          if (d < best) {
            best = d;
            closest = { el, r };
          }
        }
        if (!closest) return;

        const r = closest.r;
        const before =
          event.clientY < r.top + r.height / 2 ||
          (Math.abs(event.clientY - (r.top + r.height / 2)) < r.height * .45 &&
           event.clientX < r.left + r.width / 2);

        const current = Array.from(photosEl.children).indexOf(drag.placeholder);
        const target = Array.from(photosEl.children).indexOf(closest.el);
        const desired = before ? target : target + 1;

        if (desired === current || desired === current + 1) return;

        flip(() => {
          if (before) photosEl.insertBefore(drag.placeholder, closest.el);
          else photosEl.insertBefore(drag.placeholder, closest.el.nextSibling);
        });
      }

      function startRealDrag(event) {
        const drag = pointerDrag;
        if (!drag || drag.active) return;
        drag.active = true;
        root.draggable = false;

        const rect = drag.card.getBoundingClientRect();
        const ph = document.createElement("div");
        ph.className = "photo-card reorder-placeholder";
        ph.dataset.id = "__placeholder__";
        ph.style.width = rect.width + "px";
        ph.style.height = rect.height + "px";
        drag.placeholder = ph;
        photosEl.insertBefore(ph, drag.card);

        // The real card becomes the lifted/floating card.
        drag.card.classList.add("pointer-dragging");
        drag.card.style.position = "fixed";
        drag.card.style.left = (event.clientX - drag.offsetX) + "px";
        drag.card.style.top = (event.clientY - drag.offsetY) + "px";
        drag.card.style.width = rect.width + "px";
        drag.card.style.height = rect.height + "px";
        drag.card.style.margin = "0";
        drag.card.style.transition = "none";
        drag.card.style.transform = "scale(1.06) rotate(1.2deg)";
        drag.card.style.pointerEvents = "none";
        drag.card.style.zIndex = "99999";
      }

      function pointerMove(event) {
        const drag = pointerDrag;
        if (!drag || drag.card !== card) return;

        const distance = Math.hypot(
          event.clientX - drag.startX,
          event.clientY - drag.startY
        );
        if (!drag.active) {
          if (distance < 6) return;
          startRealDrag(event);
        }

        event.preventDefault();
        event.stopPropagation();
        drag.card.style.left = (event.clientX - drag.offsetX) + "px";
        drag.card.style.top = (event.clientY - drag.offsetY) + "px";
        movePlaceholder(event);
      }

      function pointerUp(event) {
        const drag = pointerDrag;
        if (!drag || drag.card !== card) return;
        pointerDrag = null;
        if (!drag.active) return;

        event.preventDefault();
        event.stopPropagation();

        if (drag.placeholder?.parentNode) {
          drag.placeholder.parentNode.insertBefore(card, drag.placeholder);
          drag.placeholder.remove();
        }

        card.classList.remove("pointer-dragging");
        card.style.position = "";
        card.style.left = "";
        card.style.top = "";
        card.style.width = "";
        card.style.height = "";
        card.style.margin = "";
        card.style.transition = "";
        card.style.transform = "";
        card.style.pointerEvents = "";
        card.style.zIndex = "";

        const ids = Array.from(photosEl.querySelectorAll(".photo-card"))
          .map(el => String(el.dataset.id));
        emit({ type: "reorder", ids });

        card.dataset.justDragged = "true";
        setTimeout(() => delete card.dataset.justDragged, 300);
        root.draggable = true;
      }

      card.addEventListener("pointermove", pointerMove);
      card.addEventListener("pointerup", pointerUp);
      card.addEventListener("pointercancel", pointerUp);

      // Click-to-reorder: click one photo, then click the destination photo.
      // The first photo is moved BEFORE the second photo.
      card.addEventListener("click", (event) => {
        if (event.target.closest(".photo-action")) return;
        if (card.dataset.justDragged === "true") {
          event.preventDefault();
          event.stopPropagation();
          return;
        }

        event.preventDefault();
        event.stopPropagation();

        const id = String(photo.id);

        if (!selectedPhotoId) {
          selectedPhotoId = id;
          card.classList.add("photo-selected");
          return;
        }

        if (selectedPhotoId === id) {
          selectedPhotoId = null;
          card.classList.remove("photo-selected");
          return;
        }

        const targetCards = Array.from(
          photosEl.querySelectorAll(".photo-card:not(.dragging)")
        );
        const targetIndex = targetCards.indexOf(card);

        emit({
          type: "move_photo",
          id: selectedPhotoId,
          source_item: Number(itemNumber),
          target_index: Math.max(0, targetIndex)
        });

        selectedPhotoId = null;
      });

      photosEl.appendChild(card);
    });
  }

  photosEl.addEventListener("dragover", (event) => {
    const types = Array.from(event.dataTransfer?.types || []);
    if (!types.includes("application/x-depop-photo")) return;
    event.preventDefault();
    event.stopPropagation();

    if (!draggedId) return;

    if (!dropPlaceholder) {
      dropPlaceholder = document.createElement("div");
      dropPlaceholder.className = "drop-placeholder";
      dropPlaceholder.dataset.placeholder = "true";
      dropPlaceholder.style.height = "145px";
    }

    // Empty space / bottom of the item becomes the final landing slot.
    photosEl.appendChild(dropPlaceholder);
  });

  photosEl.addEventListener("drop", (event) => {
    const types = Array.from(event.dataTransfer?.types || []);
    if (!types.includes("application/x-depop-photo")) return;
    if (!draggedId) return;

    event.preventDefault();
    event.stopPropagation();

    let sourceItem = draggedSourceItem;
    let photoId = draggedId;

    try {
      const raw = event.dataTransfer.getData(
        "application/x-depop-photo"
      );
      if (raw) {
        const parsed = JSON.parse(raw);
        photoId = String(parsed.id || photoId);
        sourceItem = Number(parsed.source_item || sourceItem);
      }
    } catch {}

    let targetIndex =
      photosEl.querySelectorAll(".photo-card:not(.dragging)").length;

    if (dropPlaceholder && dropPlaceholder.parentNode) {
      const placeholderIndex =
        Array.from(photosEl.children).indexOf(dropPlaceholder);

      targetIndex = Array.from(photosEl.children)
        .slice(0, placeholderIndex)
        .filter(
          el =>
            el.classList.contains("photo-card") &&
            !el.classList.contains("dragging")
        )
        .length;
    }

    emit({
      type: "move_photo",
      id: photoId,
      source_item: Number(sourceItem),
      target_index: targetIndex
    });
  });

  function isExternalFileDrag(event) {
    return !!(
      event.dataTransfer &&
      Array.from(event.dataTransfer.types || []).includes("Files")
    );
  }

  root.addEventListener("dragenter", (event) => {
    if (!isExternalFileDrag(event)) return;
    event.preventDefault();
    externalDepth += 1;
    root.classList.add("file-over");
  });

  root.addEventListener("dragover", (event) => {
    if (!isExternalFileDrag(event)) return;
    event.preventDefault();
    event.stopPropagation();
    event.dataTransfer.dropEffect = "copy";
    root.classList.add("file-over");
  });

  root.addEventListener("dragleave", (event) => {
    if (!isExternalFileDrag(event)) return;
    externalDepth -= 1;
    if (externalDepth <= 0) {
      externalDepth = 0;
      root.classList.remove("file-over");
    }
  });

  root.addEventListener("drop", async (event) => {
    if (!isExternalFileDrag(event)) return;

    event.preventDefault();
    event.stopPropagation();

    externalDepth = 0;
    root.classList.remove("file-over");
    root.classList.add("uploading");

    const files = Array.from(event.dataTransfer.files || []).filter(
      file => /\\.(jpe?g|png|webp)$/i.test(file.name)
    );

    if (!files.length) {
      root.classList.remove("uploading");
      return;
    }

    // This workflow is intentionally optimized for adding one missing
    // photo at a time. Multiple dropped photos are still supported, but
    // each one is compressed before crossing the Streamlit boundary.
    try {
      const payload = [];

      // Do NOT send the original 4–8 MB phone image through the component
      // trigger. That can make the Streamlit websocket appear frozen.
      // Resize/compress in the browser first; Python still receives the
      // actual image bytes, just at a practical listing-photo size.
      async function prepareFile(file) {
        const bitmap = await createImageBitmap(file);

        const MAX_SIDE = 1800;
        const scale = Math.min(
          1,
          MAX_SIDE / Math.max(bitmap.width, bitmap.height)
        );

        const width = Math.max(1, Math.round(bitmap.width * scale));
        const height = Math.max(1, Math.round(bitmap.height * scale));

        const canvas = document.createElement("canvas");
        canvas.width = width;
        canvas.height = height;

        const ctx = canvas.getContext("2d", { alpha: false });
        ctx.drawImage(bitmap, 0, 0, width, height);
        bitmap.close();

        const dataUrl = canvas.toDataURL("image/jpeg", 0.82);

        return {
          name: file.name.replace(/\.[^.]+$/, "") + ".jpg",
          type: "image/jpeg",
          data: dataUrl
        };
      }

      for (const file of files) {
        payload.push(await prepareFile(file));
      }

      emit({
        type: "add_files",
        files: payload
      });
    } catch (error) {
      console.error("Upload failed", error);
      root.classList.remove("uploading");
    }
  });

  deleteBtn.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    emit({ type: "delete_item" });
  });

  // Whole item-card dragging. Photo cards remain independently draggable.
  function clearItemDragVisuals() {
    root.classList.remove(
      "item-dragging","item-drag-target",
      "item-drop-before","item-drop-after"
    );
    if (itemGhost && itemGhost.parentNode) {
      itemGhost.parentNode.removeChild(itemGhost);
    }
    itemGhost = null;
    document.querySelectorAll(".item-card").forEach(el => {
      el.classList.remove("item-drag-target","item-drop-before","item-drop-after");
    });
  }

  // The card stays draggable. Photo cards are also draggable and explicitly
  // stop their dragstart from bubbling, so grabbing a photo moves the photo;
  // grabbing the header/empty area moves the whole item.
  root.addEventListener("dragstart", (event) => {
    // A photo drag is handled by the photo-card listener and is allowed to
    // stop here. This listener only handles the item-card drag.
    if (draggedId) {
      event.preventDefault();
      return;
    }

    if (
      event.target.closest &&
      event.target.closest(".photo-card")
    ) {
      return;
    }

    draggedItemNumber = itemNumber;
    root.classList.add("item-dragging");

    event.dataTransfer.effectAllowed = "move";
    event.dataTransfer.setData(
      "application/x-depop-item",
      JSON.stringify({source_item: itemNumber})
    );
    event.dataTransfer.setData("text/plain", "item:" + itemNumber);

    // Full-card drag image: border + title + thumbnail + photos.
    itemGhost = root.cloneNode(true);
    itemGhost.classList.remove("item-dragging","item-drag-target");
    itemGhost.style.position = "fixed";
    itemGhost.style.left = "-10000px";
    itemGhost.style.top = "0";
    itemGhost.style.width = root.getBoundingClientRect().width + "px";
    itemGhost.style.height = root.getBoundingClientRect().height + "px";
    itemGhost.style.opacity = ".94";
    itemGhost.style.transform = "rotate(-1deg) scale(.98)";
    itemGhost.style.pointerEvents = "none";
    document.body.appendChild(itemGhost);

    try {
      event.dataTransfer.setDragImage(
        itemGhost,
        Math.min(120, root.getBoundingClientRect().width / 2),
        35
      );
    } catch {}

    requestAnimationFrame(() => {
      root.classList.add("item-drag-target");
    });
  });

  root.addEventListener("dragover", (event) => {
    const types = Array.from(event.dataTransfer?.types || []);
    if (!types.includes("application/x-depop-item")) return;

    event.preventDefault();
    event.stopPropagation();
    event.dataTransfer.dropEffect = "move";

    let source = draggedItemNumber;
    try {
      const raw = event.dataTransfer.getData("application/x-depop-item");
      if (raw) source = Number(JSON.parse(raw).source_item);
    } catch {}

    if (!source || Number(source) === Number(itemNumber)) return;

    const rect = root.getBoundingClientRect();
    const before = event.clientY < rect.top + rect.height / 2;

    root.classList.add("item-drag-target");
    root.classList.toggle("item-drop-before", before);
    root.classList.toggle("item-drop-after", !before);
  });

  root.addEventListener("drop", (event) => {
    const types = Array.from(event.dataTransfer?.types || []);
    if (!types.includes("application/x-depop-item")) return;

    event.preventDefault();
    event.stopPropagation();

    let source = draggedItemNumber;
    try {
      const raw = event.dataTransfer.getData("application/x-depop-item");
      if (raw) source = Number(JSON.parse(raw).source_item);
    } catch {}

    if (!source || Number(source) === Number(itemNumber)) {
      clearItemDragVisuals();
      return;
    }

    const rect = root.getBoundingClientRect();
    const before = event.clientY < rect.top + rect.height / 2;

    emit({
      type: "move_item",
      source_item: Number(source),
      target_item: Number(itemNumber),
      before: before
    });

    clearItemDragVisuals();
    draggedItemNumber = null;
  });

  root.addEventListener("dragend", () => {
    clearItemDragVisuals();
    draggedItemNumber = null;
  });

  render();

  return () => {
    // No timers or global listeners to clean up.
  };
}
"""


if STREAMLIT_V2_AVAILABLE:
    item_drop_card = _streamlit_v2_component(
        "manual_item_card_v3",
        html=ITEM_CARD_HTML,
        css=ITEM_CARD_CSS,
        js=ITEM_CARD_JS,
        isolate_styles=True,
    )
else:
    item_drop_card = None


def make_item_thumbnail_data_url(path, max_size=260):
    """
    Create a compact browser-safe thumbnail for the custom item card.
    The browser component only needs a preview; the original file remains
    on disk for the actual listing pipeline.
    """
    path = Path(path)

    try:
        with Image.open(path) as image:
            # Respect the camera/phone EXIF orientation so portrait photos
            # don't appear sideways in the item card.
            image = ImageOps.exif_transpose(image)
            image = image.convert("RGB")
            image.thumbnail(
                (max_size, max_size)
            )

            buffer = io.BytesIO()

            image.save(
                buffer,
                format="JPEG",
                quality=78,
                optimize=True,
            )

            encoded = base64.b64encode(
                buffer.getvalue()
            ).decode("ascii")

            return (
                "data:image/jpeg;base64,"
                + encoded
            )

    except Exception:
        return ""


def build_item_component_photos(paths):
    photos = []

    for index, path in enumerate(paths):

        path = Path(path)

        photos.append({
            "id": str(
                path.resolve()
            ),
            "name": path.name,
            "src": make_item_thumbnail_data_url(
                path
            ),
            "index": index,
        })

    return photos


def save_component_files(
    file_payloads,
    item_number,
):
    """
    Save files returned by the browser component.

    Each payload is:
      {name, type, data}

    where data is a base64 data URL.
    """
    saved = []

    counter = st.session_state.get(
        "manual_upload_counter",
        0,
    )

    for payload in file_payloads or []:

        if not isinstance(
            payload,
            dict,
        ):
            continue

        name = Path(
            str(
                payload.get(
                    "name",
                    "photo.jpg",
                )
            )
        ).name

        data_url = str(
            payload.get(
                "data",
                "",
            )
        )

        if "," not in data_url:
            continue

        encoded = data_url.split(
            ",",
            1,
        )[1]

        try:
            raw = base64.b64decode(
                encoded,
                validate=True,
            )
        except Exception:
            continue

        if not raw:
            continue

        counter += 1

        output = (
            UPLOAD_DIR
            /
            (
                f"manual_item_{item_number}_"
                f"drop_{counter}_{name}"
            )
        )

        with open(
            output,
            "wb",
        ) as handle:
            handle.write(raw)

        saved.append(output)

    st.session_state[
        "manual_upload_counter"
    ] = counter

    return saved


def handle_item_component_action(
    group,
    action,
):
    """
    Apply only explicit user actions from the custom item card.

    There is NO grouping logic here.
    """
    if not isinstance(
        action,
        dict,
    ):
        return False

    action_type = action.get(
        "action"
    )

    if action_type == "add_files":

        saved_paths = save_component_files(
            action.get(
                "files",
                [],
            ),
            group.get(
                "item_number",
                0,
            ),
        )

        if not saved_paths:
            return False

        existing = {
            str(
                Path(path).resolve()
            )
            for path in group.get(
                "paths",
                [],
            )
        }

        for path in saved_paths:

            key = str(
                Path(path).resolve()
            )

            if key in existing:
                continue

            group.setdefault(
                "paths",
                []
            ).append(
                path
            )

            existing.add(
                key
            )

        manual_sync_bulk_paths()
        clear_qa_state()
        return True

    if action_type == "move_item":

        source_item = int(action.get("source_item", 0) or 0)
        target_item = int(action.get("target_item", 0) or 0)
        before = bool(action.get("before", True))

        if not source_item or not target_item or source_item == target_item:
            return False

        groups = st.session_state.get("manual_groups", [])
        source_index = next(
            (i for i,g in enumerate(groups)
             if int(g.get("item_number",0) or 0) == source_item),
            None
        )
        target_index = next(
            (i for i,g in enumerate(groups)
             if int(g.get("item_number",0) or 0) == target_item),
            None
        )

        if source_index is None or target_index is None:
            return False

        moved = groups.pop(source_index)
        if source_index < target_index:
            target_index -= 1

        insert_at = target_index if before else target_index + 1
        insert_at = max(0, min(len(groups), insert_at))
        groups.insert(insert_at, moved)

        st.session_state["manual_groups"] = groups
        manual_sync_bulk_paths()
        clear_qa_state()
        return True

    if action_type == "move_photo":

        photo_id = str(action.get("id", ""))
        source_item = int(action.get("source_item", 0) or 0)
        target_index = int(action.get("target_index", len(group.get("paths", []))) or 0)

        # Locate the source group. Moving across item cards is explicit:
        # there is no AI or similarity check involved.
        groups = st.session_state.get("manual_groups", [])
        source_group = None

        for candidate in groups:
            if int(candidate.get("item_number", 0) or 0) == source_item:
                source_group = candidate
                break

        if source_group is None:
            return False

        source_paths = list(source_group.get("paths", []))
        target_paths = list(group.get("paths", []))

        source_pos = None
        source_path = None

        for pos, path in enumerate(source_paths):
            if str(Path(path).resolve()) == photo_id:
                source_pos = pos
                source_path = path
                break

        if source_path is None:
            return False

        # Same-item moves are ordered against the list AFTER the dragged photo
        # has been removed. This makes the browser target slot deterministic.
        if source_group is group:
            remaining = [
                p for p in source_paths
                if str(Path(p).resolve()) != photo_id
            ]
            target_index = max(0, min(len(remaining), target_index))
            remaining.insert(target_index, source_path)
            group["paths"] = remaining
        else:
            # Remove from source, then insert at the exact visible landing slot.
            source_group["paths"] = [
                p for p in source_paths
                if str(Path(p).resolve()) != photo_id
            ]

            # Don't duplicate a file if a stale browser event arrives.
            target_paths = [
                p for p in target_paths
                if str(Path(p).resolve()) != photo_id
            ]

            target_index = max(0, min(len(target_paths), target_index))
            target_paths.insert(target_index, source_path)
            group["paths"] = target_paths

        manual_sync_bulk_paths()
        clear_qa_state()
        return True

    if action_type == "reorder":

        ids = action.get(
            "ids",
            [],
        )

        if not isinstance(
            ids,
            list,
        ):
            return False

        current = {
            str(
                Path(path).resolve()
            ): path
            for path in group.get(
                "paths",
                [],
            )
        }

        # Only accept an exact permutation of the current item.
        # This prevents the browser from adding/removing ownership through
        # a reorder event.
        if (
            len(ids)
            != len(current)
            or set(ids)
            != set(current)
        ):
            return False

        group[
            "paths"
        ] = [
            current[path_id]
            for path_id in ids
        ]

        manual_sync_bulk_paths()
        clear_qa_state()
        return True

    if action_type == "rotate":

        photo_id = str(
            action.get(
                "id",
                "",
            )
        )

        for path in group.get(
            "paths",
            [],
        ):
            if str(
                Path(path).resolve()
            ) != photo_id:
                continue

            try:
                image_path = Path(path)

                with Image.open(
                    image_path
                ) as image:

                    # Bake any existing EXIF orientation into the pixels
                    # first, then apply exactly one 90° clockwise manual
                    # rotation. The resulting pixels are the permanent source
                    # of truth for every later step, including Shopify.
                    corrected = ImageOps.exif_transpose(
                        image
                    )

                    rotated = corrected.rotate(
                        -90,
                        expand=True,
                    )

                    # JPEG cannot preserve arbitrary source modes.
                    if rotated.mode not in (
                        "RGB",
                        "L",
                    ):
                        rotated = rotated.convert(
                            "RGB"
                        )

                    if image_path.suffix.lower() in {
                        ".jpg",
                        ".jpeg",
                    }:
                        rotated.save(
                            image_path,
                            format="JPEG",
                            quality=95,
                            optimize=True,
                            exif=b"",
                        )
                    elif image_path.suffix.lower() == ".png":
                        rotated.save(
                            image_path,
                            format="PNG",
                        )
                    elif image_path.suffix.lower() == ".webp":
                        rotated.save(
                            image_path,
                            format="WEBP",
                            quality=95,
                        )
                    else:
                        rotated = rotated.convert("RGB")
                        rotated.save(
                            image_path,
                            format="JPEG",
                            quality=95,
                            optimize=True,
                            exif=b"",
                        )

                manual_sync_bulk_paths()
                clear_qa_state()
                return True

            except Exception as error:
                st.error(
                    f"Could not rotate photo: {error}"
                )
                return False

        return False

    if action_type == "remove":

        photo_id = str(
            action.get(
                "id",
                "",
            )
        )

        old_paths = group.get(
            "paths",
            []
        )

        new_paths = [
            path
            for path in old_paths
            if str(
                Path(path).resolve()
            ) != photo_id
        ]

        if len(new_paths) == len(
            old_paths
        ):
            return False

        group[
            "paths"
        ] = new_paths

        manual_sync_bulk_paths()
        clear_qa_state()
        return True

    if action_type == "delete_item":
        return "DELETE_ITEM"

    return False



# ============================================================
# HELPERS
# ============================================================

def clear_qa_state():

    for key in [
        "qa_results",
        "qa_approved",
        "qa_red_selected",
        "qa_green_selected",
        "qa_ai_edit_queue",
        "qa_complete"
    ]:

        st.session_state.pop(
            key,
            None
        )


def get_photo_path(
    photo_number
):

    paths = st.session_state.get(
        "bulk_paths",
        []
    )

    index = photo_number - 1

    if (
        0 <= index
        < len(paths)
    ):

        return paths[index]

    return None


def normalize_groups(
    groups,
    recover_missing=True,
    total_photos_override=None
):
    """
    Normalize group metadata WITHOUT splitting a multi-photo AI group.

    If the AI returned [1,2,3] as one group, this function keeps
    [1,2,3] as one group.

    Missing photos are only recovered as singleton groups when
    recover_missing=True. During the initial AI result, the app
    uses strict validation first so we can see the real AI output.
    """

    normalized = []
    seen = set()

    if total_photos_override is not None:
        try:
            total_photos = int(total_photos_override)
        except (TypeError, ValueError):
            total_photos = 0
    else:
        total_photos = len(
            st.session_state.get(
                "bulk_paths",
                []
            )
        )

    for group in groups:

        if not isinstance(
            group,
            dict
        ):
            continue

        photo_numbers = []

        for value in group.get(
            "photo_numbers",
            []
        ):

            try:
                number = int(value)
            except (
                ValueError,
                TypeError
            ):
                continue

            if not (
                1 <= number <= total_photos
            ):
                continue

            if number in seen:
                continue

            photo_numbers.append(
                number
            )

        if not photo_numbers:
            continue

        label = str(
            group.get(
                "label",
                ""
            )
        ).strip()

        if not label:
            label = str(
                group.get(
                    "title",
                    ""
                )
            ).strip()

        if not label:
            label = (
                f"Item "
                f"{len(normalized) + 1}"
            )

        normalized.append(
            {
                "item_number":
                    len(normalized) + 1,

                "photo_numbers":
                    sorted(
                        photo_numbers
                    ),

                "anchor_photo":
                    group.get(
                        "anchor_photo",
                        photo_numbers[0]
                    ),

                "reason":
                    str(
                        group.get(
                            "reason",
                            ""
                        )
                    ).strip(),

                "label":
                    label,

                "title":
                    label
            }
        )

        seen.update(
            photo_numbers
        )

    if recover_missing:

        for number in range(
            1,
            total_photos + 1
        ):

            if number in seen:
                continue

            normalized.append(
                {
                    "item_number":
                        len(normalized) + 1,

                    "photo_numbers":
                        [number],

                    "anchor_photo":
                        number,

                    "reason":
                        "Photo was not assigned by the grouping AI.",

                    "label":
                        "Unidentified Garment",

                    "title":
                        "Unidentified Garment"
                }
            )

            seen.add(
                number
            )

    for index, group in enumerate(
        normalized,
        start=1
    ):
        group[
            "item_number"
        ] = index

    return normalized


def strict_initial_groups(
    grouping_result
):
    """
    Preserve exactly what grouping.py returned.

    This is used immediately after AI grouping so the DND layer
    cannot be blamed for or alter the initial grouping.
    """

    raw_groups = grouping_result.get(
        "groups",
        []
    )

    if not isinstance(
        raw_groups,
        list
    ):
        raise ValueError(
            "Grouping AI did not return a groups list."
        )

    total_photos = len(
        st.session_state.get(
            "bulk_paths",
            []
        )
    )

    groups = normalize_groups(
        raw_groups,
        recover_missing=False,
        total_photos_override=total_photos
    )

    seen = []

    for group in groups:
        seen.extend(
            group.get(
                "photo_numbers",
                []
            )
        )

    duplicates = sorted(
        {
            number
            for number in seen
            if seen.count(number) > 1
        }
    )

    missing = sorted(
        set(
            range(
                1,
                total_photos + 1
            )
        )
        - set(seen)
    )

    invalid = sorted(
        {
            number
            for number in seen
            if not (
                1 <= number <= total_photos
            )
        }
    )

    if duplicates:
        raise ValueError(
            "Grouping AI returned duplicate photos: "
            + str(duplicates)
        )

    if invalid:
        raise ValueError(
            "Grouping AI returned invalid photos: "
            + str(invalid)
        )

    # We deliberately recover omitted photos here instead of letting
    # the later DND layer silently create them.
    if missing:

        for number in missing:

            groups.append(
                {
                    "item_number":
                        len(groups) + 1,

                    "photo_numbers":
                        [number],

                    "anchor_photo":
                        number,

                    "reason":
                        "Grouping AI omitted this photo.",

                    "label":
                        "Unidentified Garment",

                    "title":
                        "Unidentified Garment"
                }
            )

    for index, group in enumerate(
        groups,
        start=1
    ):
        group[
            "item_number"
        ] = index

    return groups


def validate_groups(
    groups
):

    total_photos = len(
        st.session_state.get(
            "bulk_paths",
            []
        )
    )

    expected = set(
        range(
            1,
            total_photos + 1
        )
    )

    seen = []

    for group in groups:

        seen.extend(
            group.get(
                "photo_numbers",
                []
            )
        )

    duplicates = sorted(
        {
            number
            for number in seen
            if seen.count(number) > 1
        }
    )

    missing = sorted(
        expected -
        set(seen)
    )

    invalid = sorted(
        {
            number
            for number in seen
            if number not in expected
        }
    )

    return {
        "valid":
            not duplicates
            and not missing
            and not invalid,

        "duplicates":
            duplicates,

        "missing":
            missing,

        "invalid":
            invalid
    }


def group_label(
    group,
    index,
    generated_listings=None
):

    # Once listings exist, use the real title.
    if generated_listings:

        item_number = group.get(
            "item_number",
            index + 1
        )

        for item in generated_listings:

            if item.get(
                "item_number"
            ) != item_number:

                continue

            listing = item.get(
                "listing"
            )

            if listing:

                title = str(
                    listing.get(
                        "title",
                        ""
                    )
                ).strip()

                if title:

                    return title

    label = str(
        group.get(
            "label",
            ""
        )
    ).strip()

    if label:
        return label

    return (
        f"Item {index + 1}"
    )


def rebuild_groups_from_dnd(
    groups,
    dnd_state
):

    new_groups = []

    for index, group in enumerate(
        groups
    ):

        key = (
            f"group_{index}"
        )

        photo_numbers = dnd_state.get(
            key,
            []
        )

        new_groups.append(
            {
                **group,
                "photo_numbers":
                    sorted(
                        [
                            int(number)
                            for number in photo_numbers
                        ]
                    )
            }
        )

    # Remove empty groups.
    new_groups = [
        group
        for group in new_groups
        if group.get(
            "photo_numbers",
            []
        )
    ]

    # Renumber.
    for index, group in enumerate(
        new_groups,
        start=1
    ):

        group[
            "item_number"
        ] = index

    return new_groups


# ============================================================
# PAGE
# ============================================================

st.title(
    "Depop AI Listing Machine"
)

st.write(
    "Upload clothing photos and let AI group, "
    "generate, QA, and prepare listings for Shopify."
)


# ============================================================

# ============================================================
# ============================================================
# PERMANENT UPLOAD ORIENTATION
# Manual rotation is the only orientation change after upload.
# ============================================================

def normalize_uploaded_image_to_path(uploaded_file, file_path):
    """
    Bake EXIF orientation into pixels before the file enters the app.

    This prevents Streamlit/browser previews from disagreeing with Shopify
    about phone-camera orientation.
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    try:
        uploaded_file.seek(0)

        with Image.open(uploaded_file) as source:
            image = ImageOps.exif_transpose(source).copy()

        if suffix in {".jpg", ".jpeg"}:
            if image.mode not in ("RGB", "L"):
                image = image.convert("RGB")
            image.save(
                file_path,
                format="JPEG",
                quality=95,
                optimize=True,
                exif=b"",
            )

        elif suffix == ".png":
            image.save(file_path, format="PNG")

        elif suffix == ".webp":
            if image.mode not in ("RGB", "RGBA"):
                image = image.convert("RGBA")
            image.save(
                file_path,
                format="WEBP",
                quality=95,
            )

        else:
            image = image.convert("RGB")
            file_path = file_path.with_suffix(".jpg")
            image.save(
                file_path,
                format="JPEG",
                quality=95,
                optimize=True,
                exif=b"",
            )

        return file_path

    except Exception:
        uploaded_file.seek(0)
        file_path.write_bytes(uploaded_file.getbuffer())
        return file_path

# SINGLE ITEM
# ============================================================

st.header(
    "Single Item"
)

uploaded_files = st.file_uploader(
    "Upload clothing photos",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp"
    ],
    accept_multiple_files=True,
    key="single_item_uploader"
)

if uploaded_files:

    st.success(
        f"{len(uploaded_files)} photos uploaded."
    )

    saved_paths = []

    cols = st.columns(4)

    for i, uploaded_file in enumerate(
        uploaded_files
    ):

        file_path = (
            UPLOAD_DIR
            /
            uploaded_file.name
        )

        file_path = normalize_uploaded_image_to_path(
            uploaded_file,
            file_path
        )

        saved_paths.append(
            file_path
        )

        with cols[
            i % 4
        ]:

            st.image(
                str(file_path),
                width="stretch"
            )

            st.caption(
                uploaded_file.name
            )

    if st.button(
        "Generate Depop Listing",
        type="primary",
        key="single_generate"
    ):

        with st.spinner(
            "AI is analyzing the clothing..."
        ):

            try:

                listing = analyze_item(
                    saved_paths
                )

                st.success(
                    "Listing generated!"
                )

                st.subheader(
                    listing.get(
                        "title",
                        "Untitled Clothing Item"
                    )
                )

                st.write(
                    f"**Brand:** "
                    f"{listing.get('brand', 'Unknown')}"
                )

                st.write(
                    f"**Category:** "
                    f"{listing.get('category', '-')}"
                )

                st.write(
                    f"**Size:** "
                    f"{listing.get('size', '-')}"
                )

                st.write(
                    f"**Color:** "
                    f"{listing.get('color', 'Unknown')}"
                )

                st.write(
                    f"**Suggested price:** "
                    f"${listing.get('suggested_price', 10)}"
                )

                st.text_area(
                    "Ready-to-copy listing",
                    listing.get(
                        "description",
                        ""
                    ),
                    height=300,
                    key="single_copy"
                )

                hashtags = listing.get(
                    "hashtags",
                    []
                )

                st.write(
                    "**Hashtags:** "
                    +
                    (
                        " ".join(
                            hashtags
                        )
                        if isinstance(
                            hashtags,
                            list
                        )
                        else str(
                            hashtags
                        )
                    )
                )

            except Exception as e:

                st.error(
                    f"Something went wrong: {e}"
                )


# ============================================================
# BULK INVENTORY — MANUAL GROUPING MODE
# ============================================================

st.divider()

st.header("Bulk Inventory — Manual Grouping")

st.write(
    "AI grouping is OFF for this workflow. "
    "Upload ONLY the photos for the current garment, "
    "click Make Item, then upload the next garment. "
    "Nothing is automatically grouped or moved between items."
)

# ------------------------------------------------------------
# WORKFLOW STATE
# ------------------------------------------------------------

# Version the workflow so an old staging/selection state from v1
# cannot leak into this new one-item-at-a-time workflow.
if st.session_state.get(
    "manual_workflow_version"
) != 5:

    for key in [
        "manual_staging_paths",
        "manual_selected_staging",
        "manual_groups",
        "manual_next_item_number",
        "manual_upload_counter",
        "manual_current_uploader_nonce",
        "manual_item_uploader_nonce",
        "manual_grouping_ready",
        "manual_component_events",
    ]:
        st.session_state.pop(
            key,
            None
        )

    st.session_state[
        "manual_workflow_version"
    ] = 5


if "manual_groups" not in st.session_state:
    st.session_state[
        "manual_groups"
    ] = []

if "manual_next_item_number" not in st.session_state:
    st.session_state[
        "manual_next_item_number"
    ] = 1


if "manual_upload_counter" not in st.session_state:
    st.session_state[
        "manual_upload_counter"
    ] = 0

if "manual_current_uploader_nonce" not in st.session_state:
    st.session_state[
        "manual_current_uploader_nonce"
    ] = 0

if "manual_item_uploader_nonce" not in st.session_state:
    st.session_state[
        "manual_item_uploader_nonce"
    ] = {}


def manual_save_uploaded_files(
    uploaded_files,
    destination_prefix
):
    """
    Save uploaded files with unique names.

    Every upload is copied to the uploads directory immediately.
    The saved path becomes the permanent identity of that photo.
    """
    saved = []

    counter = st.session_state.get(
        "manual_upload_counter",
        0
    )

    for uploaded_file in uploaded_files or []:

        counter += 1

        safe_name = Path(
            uploaded_file.name
        ).name

        file_path = (
            UPLOAD_DIR
            /
            (
                f"manual_{destination_prefix}_"
                f"{counter}_{safe_name}"
            )
        )

        file_path = normalize_uploaded_image_to_path(
            uploaded_file,
            file_path
        )

        saved.append(
            file_path
        )

    st.session_state[
        "manual_upload_counter"
    ] = counter

    return saved


def manual_make_group(
    paths,
    item_number
):
    return {
        "item_number": item_number,
        "paths": list(paths),
        "photo_numbers": [],
        "anchor_photo": None,
        "label": f"Item {item_number}",
        "title": f"Item {item_number}",
        "reason": "Manually grouped by user.",
    }


def manual_all_assigned_paths():
    """
    Return absolute paths already owned by an item.

    This is used only as a safety check. It NEVER decides grouping.
    """
    assigned = set()

    for group in st.session_state.get(
        "manual_groups",
        []
    ):

        for path in group.get(
            "paths",
            []
        ):

            assigned.add(
                str(
                    Path(path).resolve()
                )
            )

    return assigned


def manual_sync_bulk_paths():
    """
    Convert the manually-created item groups into the exact
    bulk_paths/bulk_groups structure expected by the existing
    listing-generation pipeline.

    IMPORTANT:
    The item grouping itself is never changed here.
    """
    groups = st.session_state.get(
        "manual_groups",
        []
    )

    ordered = []
    seen = set()

    # Item order is authoritative.
    for group in groups:

        for path in group.get(
            "paths",
            []
        ):

            path = Path(path)
            key = str(
                path.resolve()
            )

            if key not in seen:

                ordered.append(
                    path
                )

                seen.add(
                    key
                )

    st.session_state[
        "bulk_paths"
    ] = ordered

    path_to_number = {
        str(
            path.resolve()
        ): index + 1
        for index, path in enumerate(
            ordered
        )
    }

    for group in groups:

        numbers = []

        for path in group.get(
            "paths",
            []
        ):

            number = path_to_number.get(
                str(
                    Path(path).resolve()
                )
            )

            if number is not None:
                numbers.append(
                    number
                )

        group[
            "photo_numbers"
        ] = numbers

        group[
            "anchor_photo"
        ] = (
            numbers[0]
            if numbers
            else None
        )

    st.session_state[
        "manual_groups"
    ] = groups

    st.session_state[
        "bulk_groups"
    ] = [
        {
            **group
        }
        for group in groups
    ]


# ------------------------------------------------------------
# CURRENT ITEM UPLOAD
# ------------------------------------------------------------

st.subheader(
    f"1. Upload Item {st.session_state['manual_next_item_number']}"
)

st.caption(
    "Drag ONLY the photos belonging to this one garment into "
    "this box. For example: front, back, details, tag. "
    "When all photos for that garment are here, make the item."
)

current_nonce = st.session_state[
    "manual_current_uploader_nonce"
]

current_upload_key = (
    "manual_current_item_uploader_"
    + str(current_nonce)
)

current_files = st.file_uploader(
    "Drop this item's photos here",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp"
    ],
    accept_multiple_files=True,
    key=current_upload_key,
    help=(
        "Upload the complete photo set for ONE garment."
    ),
)

if current_files:

    st.write(
        f"**Current item: {len(current_files)} photos**"
    )

    preview_cols = st.columns(
        min(
            6,
            max(
                1,
                len(current_files)
            )
        )
    )

    for index, uploaded_file in enumerate(
        current_files
    ):

        with preview_cols[
            index % len(preview_cols)
        ]:

            try:
                uploaded_file.seek(0)
                with Image.open(uploaded_file) as preview_source:
                    preview_image = ImageOps.exif_transpose(
                        preview_source
                    ).copy()

                st.image(
                    preview_image,
                    width="stretch"
                )
            except Exception:
                st.image(
                    uploaded_file,
                    width="stretch"
                )

            st.caption(
                uploaded_file.name
            )

    st.success(
        "These photos are still the CURRENT item. "
        "Nothing has been grouped yet. "
        "After you make the item, use Rotate on any photo you need to fix. "
        "That rotation is saved permanently to the image file."
    )

    if st.button(
        (
            "Make Item "
            + str(
                st.session_state[
                    "manual_next_item_number"
                ]
            )
            + " With These Photos"
        ),
        type="primary",
        width="stretch",
        key=(
            "manual_make_current_item_"
            + str(current_nonce)
        )
    ):

        item_number = st.session_state[
            "manual_next_item_number"
        ]

        saved_paths = manual_save_uploaded_files(
            current_files,
            "item_"
            + str(item_number)
        )

        if not saved_paths:

            st.warning(
                "No photos were uploaded for this item."
            )

        else:

            new_group = manual_make_group(
                saved_paths,
                item_number
            )

            groups_now = list(
                st.session_state.get(
                    "manual_groups",
                    []
                )
            )

            groups_now.append(
                new_group
            )

            st.session_state[
                "manual_groups"
            ] = groups_now

            st.session_state[
                "manual_next_item_number"
            ] = item_number + 1

            # Change the uploader key so the previous upload box is
            # genuinely reset and ready for the next garment.
            st.session_state[
                "manual_current_uploader_nonce"
            ] = current_nonce + 1

            manual_sync_bulk_paths()

            clear_qa_state()

            st.success(
                f"Item {item_number} created with "
                f"{len(saved_paths)} photos."
            )

            st.rerun()

else:

    st.info(
        (
            "Drop the photos for Item "
            + str(
                st.session_state[
                    "manual_next_item_number"
                ]
            )
            + " here."
        )
    )


# ------------------------------------------------------------
# CREATED ITEMS — WHOLE CARD IS THE DROP / REORDER SURFACE
# ------------------------------------------------------------

groups = st.session_state.get(
    "manual_groups",
    []
)

if groups:

    st.divider()

    st.subheader(
        "2. Your Items"
    )

    st.caption(
        "The entire item card is now the drop zone. "
        "Drag a new File Explorer photo over an item and the whole card "
        "turns green. Drop it to add it directly to that item. "
        "Drag photos within an item to reorder them, or drag a photo "
        "to another item card to move it there. A gray landing slot shows "
        "exactly where it will be placed."
    )

    for index, group in enumerate(
        groups
    ):

        item_number = group.get(
            "item_number",
            index + 1,
        )

        paths = list(
            group.get(
                "paths",
                []
            )
        )

        if item_drop_card is None:
            st.error(
                "This version of Streamlit does not have "
                "st.components.v2. Upgrade Streamlit, then restart the app."
            )
            st.stop()

        component_result = item_drop_card(
            key="manual_item_card_" + str(item_number),
            data={
                "item_number": item_number,
                "photos_json": json.dumps(
                    build_item_component_photos(paths),
                    ensure_ascii=False,
                ),
            },
            on_action_change=lambda: None,
        )

        component_value = getattr(
            component_result,
            "action",
            None,
        )

        if component_value:
            # V2 trigger payloads arrive as the exact object emitted by JS.
            action = component_value

            if isinstance(action, dict) and action.get("type"):
                action = {
                    "action": action.get("type"),
                    **{
                        key: value
                        for key, value in action.items()
                        if key != "type"
                    },
                }

            result = handle_item_component_action(
                group,
                action,
            )

            if result == "DELETE_ITEM":

                st.session_state[
                    "manual_groups"
                ] = [
                    existing
                    for existing in st.session_state[
                        "manual_groups"
                    ]
                    if existing is not group
                ]

                manual_sync_bulk_paths()
                clear_qa_state()
                st.rerun()

            if result is True:
                st.rerun()

else:

    st.info(
        "No items created yet. Upload Item 1's photos above and "
        "click Make Item 1 With These Photos."
    )


# STATUS
# ENABLE THE EXISTING LISTING PIPELINE ONLY AFTER MANUAL GROUPING
# ------------------------------------------------------------

groups = st.session_state.get(
    "manual_groups",
    []
)

manual_sync_bulk_paths()

if groups:

    st.session_state[
        "bulk_groups"
    ] = [
        {
            **group
        }
        for group in groups
    ]

    all_paths = []

    for group in groups:
        all_paths.extend(
            str(
                Path(path).resolve()
            )
            for path in group.get(
                "paths",
                []
            )
        )

    duplicate_paths = sorted(
        {
            path
            for path in all_paths
            if all_paths.count(path) > 1
        }
    )

    if duplicate_paths:

        st.session_state[
            "manual_grouping_ready"
        ] = False

        st.error(
            "A photo is assigned to more than one item. "
            "Remove the duplicate before generating listings."
        )

    else:

        st.session_state[
            "manual_grouping_ready"
        ] = True

else:

    st.session_state[
        "manual_grouping_ready"
    ] = False


# ============================================================
# GENERATE LISTINGS
# ============================================================

if (
    "bulk_groups" in st.session_state
    and
    "bulk_paths" in st.session_state
    and
    st.session_state.get(
        "manual_grouping_ready",
        False
    )
):

    groups = st.session_state[
        "bulk_groups"
    ]

    bulk_paths = st.session_state[
        "bulk_paths"
    ]

    validation = validate_groups(
        groups
    )

    st.divider()

    st.subheader(
        "Generate All Depop Listings"
    )

    if not validation["valid"]:

        st.warning(
            "Fix the grouping before generating listings."
        )

    elif st.button(
        "Generate All Listings",
        type="primary",
        key="generate_all_listings"
    ):
        # ------------------------------------------------------------
        # FAST BATCHED LISTING GENERATION
        # ------------------------------------------------------------
        # We keep the exact same analyze_item() function and listing
        # format. The only change is that several independent garments
        # can be generated at the same time.
        #
        # This is controlled concurrency, not 30+ requests at once.
        # Four workers gives a good speed increase while avoiding the
        # request burst that was causing repeated 429s.
        # ------------------------------------------------------------

        generated_listings = []

        total_items = len(groups)

        progress_bar = st.progress(
            0,
            text="Preparing listing generation..."
        )

        status_text = st.empty()

        # Build every item's photo list before starting threads.
        jobs = []

        for index, group in enumerate(groups):

            item_number = group.get(
                "item_number",
                index + 1
            )

            item_paths = []

            for photo_number in group.get(
                "photo_numbers",
                []
            ):

                photo_index = (
                    photo_number - 1
                )

                if (
                    0 <= photo_index
                    < len(bulk_paths)
                ):
                    item_paths.append(
                        bulk_paths[
                            photo_index
                        ]
                    )

            if item_paths:
                jobs.append(
                    {
                        "index":
                            index,

                        "item_number":
                            item_number,

                        "photos":
                            item_paths
                    }
                )

        completed = 0
        worker_count = min(
            4,
            max(
                1,
                len(jobs)
            )
        )

        def generate_one(job):
            """
            Generate one listing using the existing analyze_item()
            function. Errors stay attached to that item so one failed
            request never kills the entire batch.
            """

            last_error = None

            for attempt in range(3):

                try:

                    listing = analyze_item(
                        job["photos"]
                    )

                    if listing is None:
                        raise RuntimeError(
                            "AI returned an empty listing."
                        )

                    return {
                        "index":
                            job["index"],

                        "item_number":
                            job["item_number"],

                        "photos":
                            job["photos"],

                        "listing":
                            listing
                    }

                except Exception as error:

                    last_error = error

                    # Controlled backoff for temporary API limits.
                    # The retry happens inside the worker, so one
                    # rate-limited request doesn't kill the whole batch.
                    if attempt < 2:

                        time.sleep(
                            2 ** (
                                attempt + 1
                            )
                        )

            return {
                "index":
                    job["index"],

                "item_number":
                    job["item_number"],

                "photos":
                    job["photos"],

                "listing":
                    None,

                "error":
                    str(last_error)
            }

        status_text.write(
            f"Generating {len(jobs)} listings "
            f"in batches of {worker_count}..."
        )

        # ------------------------------------------------------------
        # Controlled parallel generation.
        # ------------------------------------------------------------

        results_by_index = {}

        with ThreadPoolExecutor(
            max_workers=worker_count
        ) as executor:

            futures = {
                executor.submit(
                    generate_one,
                    job
                ):
                    job["index"]
                for job in jobs
            }

            for future in as_completed(
                futures
            ):

                result = future.result()

                results_by_index[
                    result["index"]
                ] = result

                completed += 1

                percent = (
                    completed
                    / max(
                        1,
                        len(jobs)
                    )
                )

                status_text.write(
                    f"Generated {completed} "
                    f"of {len(jobs)} listings "
                    f"({percent:.0%})"
                )

                progress_bar.progress(
                    percent,
                    text=(
                        f"Generating listings — "
                        f"{completed}/{len(jobs)}"
                    )
                )

        # Keep the original item order for the review screen.
        generated_listings = [
            results_by_index[index]
            for index in sorted(
                results_by_index
            )
        ]

        progress_bar.progress(
            1.0,
            text="All listings generated"
        )

        status_text.success(
            "All listings finished!"
        )

        st.session_state[
            "generated_listings"
        ] = generated_listings

        st.session_state[
            "listing_review_approved"
        ] = False

        st.session_state[
            "show_final_qa"
        ] = False

        clear_qa_state()

        st.rerun()


# ============================================================
# PRE-QA LISTING VALIDATION
# ============================================================

def validate_listing_for_qa(listing):
    """
    Hard requirements before Final AI QA.
    Missing/invalid hashtags are a QA failure.
    """
    failures = []

    if not isinstance(listing, dict):
        return [
            "Listing data is missing or invalid."
        ]

    title = str(
        listing.get("title", "")
    ).strip()

    if not title:
        failures.append(
            "Missing title."
        )

    description = str(
        listing.get("description", "")
    ).strip()

    if not description:
        failures.append(
            "Missing description."
        )

    size = str(
        listing.get("size", "")
    ).strip()

    if not size or size.lower() in {"-", "unknown", "none", "n/a", "not visible", "not readable"}:
        failures.append(
            "Size requires manual review — the tag could not be read confidently."
        )

    category = str(listing.get("category_path", listing.get("category", ""))).strip()
    if not category or category == "-" or category not in SHOPIFY_CATEGORY_PATHS:
        failures.append(
            "Shopify category requires manual review."
        )

    hashtags = listing.get(
        "hashtags",
        []
    )

    if isinstance(
        hashtags,
        str
    ):
        hashtags = hashtags.split()

    if not isinstance(
        hashtags,
        list
    ):
        hashtags = []

    clean_tags = []

    for tag in hashtags:

        tag = str(
            tag
        ).strip()

        if tag:
            if not tag.startswith("#"):
                tag = "#" + tag

            clean_tags.append(
                tag
            )

    # EXACTLY five hashtags is required.
    if len(clean_tags) != 5:
        failures.append(
            f"Exactly 5 hashtags are required; "
            f"this listing has {len(clean_tags)}."
        )

    # Remove duplicates while preserving order.
    deduped = []

    for tag in clean_tags:

        if tag.lower() not in {
            existing.lower()
            for existing in deduped
        }:
            deduped.append(
                tag
            )

    if len(deduped) != 5:
        failures.append(
            "Hashtags contain duplicates."
        )

    listing["hashtags"] = deduped

    return failures


# ============================================================
# GENERATED LISTING REVIEW
# ============================================================

if "generated_listings" in st.session_state:

    generated_listings = st.session_state[
        "generated_listings"
    ]

    st.divider()

    st.header(
        "Generated Listings Review"
    )

    st.caption(
        "Review and edit the AI-generated listings before Final QA."
    )

    failed_count = sum(
        1
        for item in generated_listings
        if item.get("listing") is None
    )

    if failed_count:
        st.warning(
            f"{failed_count} listing(s) failed to generate."
        )

    listing_failures = {}

    for item_index, item in enumerate(
        generated_listings
    ):

        item_number = item.get(
            "item_number",
            item_index + 1
        )

        listing = item.get(
            "listing"
        )

        photos = item.get(
            "photos",
            []
        )

        with st.expander(
            f"Item {item_number}",
            expanded=True
        ):

            if photos:

                photo_cols = st.columns(
                    min(
                        len(photos),
                        4
                    )
                )

                for photo_index, photo in enumerate(
                    photos
                ):

                    with photo_cols[
                        photo_index
                        % len(photo_cols)
                    ]:

                        st.image(
                            photo,
                            width="stretch"
                        )

            if listing is None:

                st.error(
                    "This listing did not generate."
                )

                st.code(
                    item.get(
                        "error",
                        "Unknown generation error."
                    )
                )

                listing_failures[
                    item_index
                ] = [
                    "Listing generation failed."
                ]

                continue

            title = st.text_input(
                "Title",
                value=str(
                    listing.get(
                        "title",
                        ""
                    )
                ),
                key=f"generated_title_{item_index}"
            )

            brand = st.text_input(
                "Brand",
                value=str(
                    listing.get(
                        "brand",
                        ""
                    )
                ),
                key=f"generated_brand_{item_index}"
            )

            # --------------------------------------------------------
            # GARMENT TYPE + SHOPIFY CATEGORY
            # --------------------------------------------------------
            garment_key = f"generated_garment_type_{item_index}"
            desired_garment = str(
                listing.get(
                    "garment_type",
                    listing.get("type", "")
                )
                or ""
            ).strip()

            # If this is the first render for this listing, seed the widget
            # state from the AI result. This prevents a stale Streamlit selectbox
            # value from overriding the newly generated garment type.
            if garment_key + "_select" not in st.session_state:
                st.session_state[garment_key + "_select"] = (
                    desired_garment
                    if desired_garment
                    else "— Select —"
                )

            garment_type = _garment_type_select(
                desired_garment,
                garment_key,
            )

            listing["garment_type"] = garment_type
            listing["type"] = garment_type

            # Full Shopify clothing taxonomy choices. The AI supplies the
            # exact path + GID, and the review UI keeps that exact value selected.
            # There is intentionally NO Clothing Tops fallback for bottoms.
            category_choices = ["— Manual Review —"] + SHOPIFY_CATEGORY_PATHS
            existing_category = str(
                listing.get("category_path", listing.get("category", ""))
            ).strip()

            category_index = (
                category_choices.index(existing_category)
                if existing_category in category_choices
                else 0
            )

            category = st.selectbox(
                "Shopify Category",
                category_choices,
                index=category_index,
                key=f"generated_category_{item_index}",
                help="Exact Shopify Standard Product Taxonomy category. Choose Manual Review if the photos do not support a confident category.",
            )

            if category == "— Manual Review —":
                category = "-"
                listing["category"] = "-"
                listing["category_path"] = ""
                listing["category_gid"] = ""
            else:
                listing["category"] = category
                listing["category_path"] = category
                listing["category_gid"] = SHOPIFY_CATEGORY_GIDS.get(category, "")

            st.markdown("#### Shopify Attributes")

            shopify_values = _render_shopify_attributes(
                listing,
                item_index,
                garment_type,
            )

            # Keep the Shopify-controlled values in the same listing
            # dictionary used by the existing QA/export pipeline.
            for field_name, field_value in shopify_values.items():
                listing[field_name] = field_value

            size = shopify_values.get(
                "size",
                listing.get("size", "-"),
            ) or "-"

            color = shopify_values.get(
                "color",
                listing.get("color", ""),
            )

            pattern = shopify_values.get(
                "pattern",
                listing.get("pattern", ""),
            )

            condition = st.text_input(
                "Condition",
                value=str(
                    listing.get(
                        "condition",
                        ""
                    )
                ),
                key=f"generated_condition_{item_index}"
            )

            # The Size dropdown is the single source of truth for the size
            # wording inside the description. Streamlit remembers widget state,
            # so update the widget's session-state value when Size changes.
            description_key = f"generated_description_{item_index}"
            description_size_key = f"description_size_sync_{item_index}"

            raw_description = str(
                listing.get(
                    "description",
                    ""
                )
            )

            previous_size = st.session_state.get(
                description_size_key
            )

            hashtags_value = listing.get(
                "hashtags",
                []
            )

            if isinstance(
                hashtags_value,
                list
            ):
                hashtags_text = " ".join(
                    str(tag)
                    for tag in hashtags_value
                )
            else:
                hashtags_text = str(
                    hashtags_value
                )

            # Keep the description's controlled fields synchronized. This is
            # intentionally done before the hashtags widget so the description
            # uses the exact same generated hashtags as the listing.
            if (
                description_key not in st.session_state
                or previous_size != size
            ):
                st.session_state[description_key] = format_depop_description(
                    raw_description,
                    title,
                    size,
                    condition,
                    hashtags_value,
                )
                st.session_state[description_size_key] = size

            description = st.text_area(
                "Description",
                height=180,
                key=description_key
            )

            hashtags = st.text_input(
                "Hashtags — EXACTLY 5 REQUIRED",
                value=hashtags_text,
                key=f"generated_hashtags_{item_index}"
            )

            # --------------------------------------------------------
            # MARKET RESEARCH — AUTO SCRAPE + SOURCE VIEWER
            # --------------------------------------------------------
            market_state_key = f"market_results_{item_index}"
            market_auto_key = f"market_auto_scraped_{item_index}"
            market_query = build_market_query(listing)

            def run_market_scrape():
                with st.spinner(
                    "Scraping eBay sold listings + Depop listings…"
                ):
                    try:
                        learned_multiplier, learned_key, learned_count = (
                            _get_learned_multiplier(listing)
                        )

                        result = scrape_market(
                            market_query,
                            ebay_limit=24,
                            depop_limit=24,
                            listing=listing,
                            pricing_multiplier=learned_multiplier,
                        )

                        # Keep the pricing profile visible with the market result.
                        result.setdefault("summary", {})[
                            "pricing_multiplier"
                        ] = learned_multiplier
                        result["summary"]["pricing_profile_key"] = learned_key
                        result["summary"]["pricing_learning_count"] = learned_count

                        st.session_state[market_state_key] = result
                        st.session_state[market_auto_key] = True

                        recommended = (
                            result.get("summary", {}).get("recommended")
                        )

                        price_key = f"generated_price_{item_index}"
                        manual_key = f"pricing_manual_{item_index}"
                        baseline_key = f"pricing_auto_baseline_{item_index}"
                        context_key = f"pricing_context_{item_index}"

                        # A rescrape must not erase a price the user already
                        # manually chose.
                        if (
                            recommended is not None
                            and not st.session_state.get(manual_key, False)
                        ):
                            st.session_state[price_key] = float(recommended)
                            listing["suggested_price"] = float(recommended)
                            st.session_state[baseline_key] = float(recommended)

                        median = result.get("summary", {}).get("median")
                        if median is not None:
                            st.session_state[context_key] = {
                                "median": float(median),
                                "multiplier": float(learned_multiplier),
                            }

                    except Exception as exc:
                        st.session_state[market_state_key] = {
                            "query": market_query,
                            "ebay_url": ebay_search_url(market_query)
                            if "ebay_search_url" in globals()
                            else "",
                            "depop_url": depop_search_url(market_query)
                            if "depop_search_url" in globals()
                            else "",
                            "comps": [],
                            "summary": {},
                            "errors": [str(exc)],
                        }
                        st.session_state[market_auto_key] = True

            # Automatically scrape once for each generated item.
            # The state flag prevents every Streamlit rerun from scraping again.
            if not st.session_state.get(market_auto_key, False):
                run_market_scrape()

            market_result = st.session_state.get(
                market_state_key
            )

            if market_result:
                summary = market_result.get(
                    "summary",
                    {}
                )

                comps = market_result.get("comps", [])

                ebay_comps = [
                    c for c in comps
                    if c.get("source") in (
                        "eBay",
                        "eBay via Google",
                        "eBay via Bing",
                    )
                ]
                depop_comps = [
                    c for c in comps
                    if c.get("source") == "Depop"
                ]
                sold_count = sum(
                    1 for c in comps
                    if c.get("sold")
                )

                st.caption(
                    f"Market search: `{market_result.get('query', market_query)}`"
                )
                st.caption(
                    "eBay: direct SOLD/COMPLETED search first, then Google-indexed "
                    "eBay sold-listing discovery if direct eBay is blocked"
                )

                if summary.get("priced_count"):
                    pricing_multiplier = float(
                        summary.get(
                            "pricing_multiplier",
                            BASE_PRICING_MULTIPLIER,
                        )
                        or BASE_PRICING_MULTIPLIER
                    )
                    pricing_learning_count = int(
                        summary.get("pricing_learning_count", 0) or 0
                    )

                    if pricing_learning_count:
                        pricing_rule = (
                            f"Learned from your pricing: "
                            f"{(pricing_multiplier - 1) * 100:.0f}% above median"
                        )
                    else:
                        pricing_rule = "Starting strategy: 50% above median"

                    st.caption(
                        f"Pricing strategy: {pricing_rule}"
                    )

                    cols = st.columns(4)
                    cols[0].metric(
                        "Priced comps",
                        summary.get("priced_count", 0),
                    )
                    cols[1].metric(
                        "Median",
                        f"${summary['median']:.0f}",
                    )
                    cols[2].metric(
                        "Low",
                        f"${summary['low']:.0f}",
                    )
                    cols[3].metric(
                        "High",
                        f"${summary['high']:.0f}",
                    )

                    recommended = summary.get("recommended")

                    if recommended:
                        st.success(
                            f"Scraped — suggested price: ${recommended}"
                        )

                ebay_priced = sum(
                    1 for c in ebay_comps
                    if c.get("price") is not None
                )
                depop_priced = sum(
                    1 for c in depop_comps
                    if c.get("price") is not None
                )

                st.info(
                    f"Scraped: eBay {len(ebay_comps)} comps "
                    f"({ebay_priced} priced / {sold_count} sold) • "
                    f"Depop {len(depop_comps)} active comps "
                    f"({depop_priced} priced)"
                )

                # Two controls: manually run the scraper again, or inspect
                # exactly where every price/listing came from.
                rescrape_col, sources_col = st.columns(2)

                with rescrape_col:
                    if st.button(
                        "↻ Rescrape eBay + Depop",
                        key=f"rescrape_market_{item_index}",
                        help="Run fresh public searches again and replace the current market results.",
                    ):
                        run_market_scrape()
                        st.rerun()

                with sources_col:
                    source_label = (
                        f"🔗 View sources ({len(comps)})"
                        if comps
                        else "🔗 View search sources"
                    )

                    with st.popover(
                        source_label,
                        use_container_width=True,
                    ):
                        st.markdown("### Where the market data came from")
                        st.caption(
                            "These are the actual public search/listing URLs "
                            "used by the scraper."
                        )

                        ebay_search = market_result.get(
                            "ebay_url",
                            ""
                        )
                        depop_search = market_result.get(
                            "depop_url",
                            ""
                        )

                        if ebay_search:
                            st.markdown(
                                f"**eBay search:** [{market_query}]({ebay_search})"
                            )

                        if depop_search:
                            st.markdown(
                                f"**Depop search:** [{market_query}]({depop_search})"
                            )

                        st.divider()

                        if ebay_comps:
                            st.markdown(
                                f"#### eBay sold sources ({len(ebay_comps)})"
                            )

                            for number, comp in enumerate(
                                ebay_comps,
                                1,
                            ):
                                url = comp.get("url", "")
                                title_text = (
                                    comp.get("title", "")
                                    or "eBay listing"
                                )[:110]
                                price = comp.get("price")

                                if url:
                                    price_text = (
                                        f" — ${price:.2f}"
                                        if price is not None
                                        else ""
                                    )
                                    st.markdown(
                                        f"**${price:.2f}** — [{title_text}]({url})"
                                        if price is not None
                                        else f"**Price unavailable** — [{title_text}]({url})"
                                    )

                        if depop_comps:
                            st.markdown(
                                f"#### Depop active sources ({len(depop_comps)})"
                            )

                            for number, comp in enumerate(
                                depop_comps,
                                1,
                            ):
                                url = comp.get("url", "")
                                title_text = (
                                    comp.get("title", "")
                                    or "Depop listing"
                                )[:110]
                                price = comp.get("price")

                                if url:
                                    price_text = (
                                        f" — ${price:.2f}"
                                        if price is not None
                                        else ""
                                    )
                                    st.markdown(
                                        f"**${price:.2f}** — [{title_text}]({url})"
                                        if price is not None
                                        else f"**Price unavailable** — [{title_text}]({url})"
                                    )

                errors = market_result.get(
                    "errors",
                    []
                )

                for error in errors:
                    st.warning(error)

                if not comps and not errors:
                    st.warning(
                        "The scraper completed but found no readable comps. "
                        "Use Rescrape to try again."
                    )

                if not ebay_comps:
                    with st.expander("eBay scraper diagnostics"):
                        debug = get_ebay_debug() or {}
                        st.write("Primary query:", debug.get("query", market_query))

                        for attempt in debug.get("attempts", []):
                            st.markdown(
                                f"**Attempt: `{attempt.get('query', '')}`**"
                            )
                            st.write(
                                {
                                    "URL": attempt.get("url"),
                                    "Page title": attempt.get("page_title"),
                                    "HTTP/browser page body length": attempt.get("body_length"),
                                    "Sold text present": attempt.get("contains_sold"),
                                    "Completed text present": attempt.get("contains_completed"),
                                    "CAPTCHA text present": attempt.get("contains_captcha"),
                                    "Verify-human text present": attempt.get("contains_verify"),
                                    "/itm/ links found": attempt.get("itm_links"),
                                    "s-card elements": attempt.get("s_cards"),
                                    "s-item elements": attempt.get("s_items"),
                                    "Comps extracted": attempt.get("comps_found"),
                                    "Prices extracted": attempt.get("priced_found"),
                                    "Exception": attempt.get("exception"),
                                }
                            )

                        if debug.get("search_fallbacks"):
                            st.markdown("**Search-engine → eBay fallbacks**")
                            st.write(debug["search_fallbacks"])

                        if debug.get("http_fallback"):
                            st.markdown("**Direct eBay HTTP fallback**")
                            st.write(debug["http_fallback"])

                        st.caption(
                            "This diagnostic is intentionally visible so we can see "
                            "whether eBay is returning results, a security page, or "
                            "HTML with a different structure."
                        )

                if depop_comps and not any(
                    c.get("price") is not None for c in depop_comps
                ):
                    st.warning(
                        "Depop listings were found, but their prices were not "
                        "read from the public page. The scraper needs a fresh "
                        "price-field pattern for the page version being served."
                    )

                if ebay_comps and not any(
                    c.get("price") is not None for c in ebay_comps
                ):
                    st.warning(
                        "eBay listings were found, but their prices were not "
                        "read from the public page."
                    )

                st.caption(
                    "eBay = public sold/completed results. "
                    "Depop = public active search results. "
                    "No API credentials are being used."
                )


            price_widget_key = f"generated_price_{item_index}"
            manual_price_key = f"pricing_manual_{item_index}"
            auto_baseline_key = f"pricing_auto_baseline_{item_index}"
            pricing_context_key = f"pricing_context_{item_index}"

            if price_widget_key not in st.session_state:
                initial_price = float(
                    listing.get(
                        "suggested_price",
                        10
                    )
                    or 0
                )
                st.session_state[price_widget_key] = initial_price

            # Detect a real user edit on the previous Streamlit rerun.
            # The scraper writes the automatic baseline into its own state key,
            # so we can distinguish "AI/scraper changed it" from "you changed it".
            current_widget_price = st.session_state.get(price_widget_key)
            previous_auto_price = st.session_state.get(auto_baseline_key)

            if (
                previous_auto_price is not None
                and current_widget_price is not None
                and abs(
                    float(current_widget_price)
                    - float(previous_auto_price)
                ) > 0.001
            ):
                context = st.session_state.get(
                    pricing_context_key,
                    {}
                )
                median_for_learning = context.get("median")

                if median_for_learning:
                    learned = _record_manual_price(
                        listing,
                        float(current_widget_price),
                        float(median_for_learning),
                    )
                    st.session_state[manual_price_key] = True
                    st.session_state["pricing_last_learning"] = {
                        "item_index": item_index,
                        "manual_price": float(current_widget_price),
                        "median": float(median_for_learning),
                        "multiplier": float(learned or BASE_PRICING_MULTIPLIER),
                    }

                # Move the baseline to the user's chosen price so the same
                # edit is never learned repeatedly on every rerun.
                st.session_state[auto_baseline_key] = float(current_widget_price)

            suggested_price = st.number_input(
                "Suggested Price",
                min_value=0.0,
                max_value=1000.0,
                step=1.0,
                key=price_widget_key,
            )

            # Keep the listing dictionary synchronized with the actual widget.
            listing["suggested_price"] = float(suggested_price)

            if st.session_state.get(manual_price_key, False):
                st.caption(
                    "✓ Your manual price is being used. Future similar items "
                    "will learn from this pricing decision."
                )

            listing["title"] = title
            listing["brand"] = brand
            listing["category"] = category
            listing["category_path"] = category if category in SHOPIFY_CATEGORY_PATHS else ""
            listing["category_gid"] = SHOPIFY_CATEGORY_GIDS.get(category, "")
            listing["garment_type"] = garment_type
            listing["size"] = size
            listing["size_reading"] = size
            listing["color"] = color
            listing["pattern"] = pattern
            listing["condition"] = condition
            listing["description"] = description
            listing["hashtags"] = [
                tag.strip()
                if tag.strip().startswith("#")
                else "#" + tag.strip()
                for tag in hashtags.split()
                if tag.strip()
            ]
            listing["suggested_price"] = suggested_price

            failures = validate_listing_for_qa(
                listing
            )

            if failures:

                listing_failures[
                    item_index
                ] = failures

                st.error(
                    "QA rejection — " +
                    " ".join(
                        failures
                    )
                )

                # Individual repair button. It only touches THIS
                # listing and only regenerates fields that failed.
                if st.button(
                    "Fix Issues with AI",
                    key=f"fix_listing_{item_index}",
                    type="secondary"
                ):

                    repair_fields = []

                    if any(
                        "title" in failure.lower()
                        for failure in failures
                    ):
                        repair_fields.append(
                            "title"
                        )

                    if any(
                        "description" in failure.lower()
                        for failure in failures
                    ):
                        repair_fields.append(
                            "description"
                        )

                    if any(
                        "hashtag" in failure.lower()
                        for failure in failures
                    ):
                        repair_fields.append(
                            "hashtags"
                        )

                    if any(
                        "size" in failure.lower()
                        for failure in failures
                    ):
                        # Size is generated/verified as part of the
                        # main listing request, so regenerate the
                        # whole listing rather than inventing a size.
                        repair_fields.append(
                            "FULL_LISTING"
                        )

                    try:

                        with st.spinner(
                            f"AI fixing Item {item_number}..."
                        ):

                            if "FULL_LISTING" in repair_fields:

                                repaired = analyze_item(
                                    photos
                                )

                                if repaired is None:
                                    raise RuntimeError(
                                        "AI returned an empty listing."
                                    )

                                item["listing"] = repaired

                            else:

                                for field in repair_fields:

                                    repaired_value = (
                                        regenerate_field(
                                            field,
                                            listing,
                                            photos
                                        )
                                    )

                                    if field == "hashtags":

                                        repaired_value = [
                                            tag if str(tag).startswith("#")
                                            else "#" + str(tag).strip()
                                            for tag in (
                                                repaired_value or []
                                            )
                                            if str(tag).strip()
                                        ]

                                        if len(
                                            repaired_value
                                        ) != 5:
                                            raise RuntimeError(
                                                "AI did not return exactly 5 hashtags."
                                            )

                                    listing[field] = (
                                        repaired_value
                                    )

                                item["listing"] = listing

                        st.session_state[
                            "generated_listings"
                        ] = generated_listings

                        st.success(
                            f"Item {item_number} fixed. Review the updated fields below."
                        )

                        st.rerun()

                    except Exception as repair_error:

                        st.error(
                            "AI could not fix this item yet: "
                            + str(repair_error)
                        )

            else:

                st.success(
                    "Ready for Final AI QA"
                )

    st.session_state[
        "generated_listings"
    ] = generated_listings

    st.divider()

    # ------------------------------------------------------------
    # ONE BUTTON ONLY.
    #
    # Clicking this saves edits and immediately runs/render QA
    # in the same Streamlit execution. There is no second button.
    # ------------------------------------------------------------

    all_valid = (
        len(generated_listings) > 0
        and failed_count == 0
        and not listing_failures
    )

    if st.button(
        "Approve Listings & Run Final AI QA",
        type="primary",
        key="approve_and_run_final_qa"
    ):

        if not all_valid:

            st.error(
                "Cannot run Final AI QA yet. "
                "Fix every red listing above first. "
                "Exactly 5 hashtags are required for every item."
            )

        else:

            st.session_state[
                "listing_review_approved"
            ] = True

            clear_qa_state()

            # Continue directly to QA below on this same run.
            st.session_state[
                "show_final_qa"
            ] = True


# ============================================================
# FINAL AI QA
# ============================================================

if (
    "generated_listings" in st.session_state
    and
    st.session_state.get(
        "show_final_qa",
        False
    )
):

    generated_listings = st.session_state[
        "generated_listings"
    ]

    valid_for_qa = all(
        item.get(
            "listing"
        ) is not None
        and not validate_listing_for_qa(
            item.get(
                "listing"
            )
        )
        for item in generated_listings
    )

    if valid_for_qa:

        from qa_review import (
            render_qa_review
        )

        render_qa_review(
            generated_listings
        )
