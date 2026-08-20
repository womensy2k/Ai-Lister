import streamlit as st
import streamlit.components.v1 as components
from streamlit.errors import StreamlitAPIException
import base64
import hashlib
import io
import json
import os
from PIL import Image, ImageOps
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import re
from pathlib import Path
import html

# ============================================================
# STREAMLIT CLOUD SECRETS -> ENVIRONMENT VARIABLES
# Locally, python-dotenv loads .env (gitignored) into os.environ.
# On Streamlit Cloud there's no .env file at all — secrets are set via
# the dashboard's "Secrets" panel instead, which only populates
# st.secrets, NOT os.environ. Every module in this app (ai_listing,
# qa_review, shopify_integration, ebay_scraper) reads credentials via
# plain os.getenv(...) at import time, so without this, those imports
# fail on Cloud even with secrets configured correctly in the
# dashboard. Copying them into os.environ here — before any of those
# modules are imported below — makes the exact same os.getenv() code
# work in both places. A no-op locally (.env already populated
# os.environ by the time this runs... actually before, but the
# "already set" check makes the order irrelevant either way).
_SECRET_ENV_KEYS = (
    "OPENAI_API_KEY",
    "SHOPIFY_CLIENT_ID",
    "SHOPIFY_CLIENT_SECRET",
    "SHOPIFY_SHOP",
    "EBAY_CLIENT_ID",
    "EBAY_CLIENT_SECRET",
    "EBAY_ENV",
    "EBAY_MARKETPLACE",
)
try:
    for _secret_key in _SECRET_ENV_KEYS:
        if _secret_key not in os.environ and _secret_key in st.secrets:
            os.environ[_secret_key] = str(st.secrets[_secret_key])
except Exception:
    # No secrets.toml at all (plain local dev via .env) — st.secrets
    # itself can raise on first access in that case. Never let this
    # opportunistic sync break a working local setup.
    pass

# PUBLIC MARKET SCRAPING
# Requires: pip install playwright && playwright install chromium

from ai_listing import (
    analyze_item,
    regenerate_field,
    detect_photo_rotations,
    rotate_photo_in_place,
    apply_vintage_flag,
    record_vintage_correction,
    SHOPIFY_CATEGORY_PATHS,
    SHOPIFY_CATEGORY_GIDS,
)
from grouping import order_group_photos
from market_scraper import (
    build_market_query,
    depop_search_url,
    scrape_market_all,
)
import bulk_sort


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
    # Prefer the dedicated tag-evidence vintage check (is_vintage /
    # vintage_classification from _verify_vintage_from_tags() — RN
    # number, retired logo era, union label, etc.) when it ran: it's a
    # much more rigorous, evidence-backed signal than the main pass's
    # free-form aesthetic guess. That dedicated check only runs when
    # the size/brand tag pass runs or the aesthetic guess already
    # suspects vintage (see analyze_item()), so "is_vintage" not being
    # present just means it never ran for this item — fall back to the
    # aesthetic "style" list in that case, instead of scanning the
    # title text (which just reflects whatever words made it into the
    # title, not real evidence).
    if "is_vintage" in listing:
        return "vintage" if listing.get("is_vintage") else "non_vintage"

    style_tags = listing.get("style") or []
    if not isinstance(style_tags, (list, tuple, set)):
        style_tags = [style_tags]
    is_vintage = any(
        "vintage" in str(tag or "").lower() for tag in style_tags
    )
    return "vintage" if is_vintage else "non_vintage"


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


# ============================================================
# SHOPIFY CONTROLLED LISTING FIELDS
# ============================================================

# Keep these JSON files beside app.py.
# They are the source of truth for the selectable Shopify values.
TOPS_TAXONOMY_FILE = Path(__file__).with_name("shopify_taxonomy_options.json")
JEANS_TAXONOMY_FILE = Path(__file__).with_name("shopify_taxonomy_options_jeans.json")


@st.cache_data(show_spinner=False)
def _load_taxonomy(path):
    # Streamlit re-executes this whole script top-to-bottom on every
    # single widget interaction anywhere in the app — without caching,
    # these two on-disk JSON files (tens of KB each) were being
    # re-read and re-parsed on every rerun for the entire session,
    # even though their content never changes after the app starts.
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
    label_visibility="visible",
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
        label_visibility=label_visibility,
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


def _evidence_hover_label_html(label, photos, listing):
    """Underlined field label that previews the AI's actual evidence
    for that field on hover — the single zoomed-in tag crop it read
    size/brand from (evidence_crop_data_url, set by
    _verify_size_from_tags() in ai_listing.py), not a whole uncropped
    photo the user has to squint at to even find the tag in. Shared by
    Size and Brand since both are read from the same tag crop.

    Falls back to the identified tag photo (size_tag_photo_indexes),
    then the item's first photo, only when no crop was captured —
    e.g. an older listing generated before this existed, or one whose
    size/brand were confident enough on the first pass that the
    dedicated tag-crop reader never ran.
    """
    crop_data_url = listing.get("evidence_crop_data_url")

    if crop_data_url:
        images_html = [f'<img src="{crop_data_url}" alt="">']
        caption = "Tag crop the AI read this from"
    else:
        tag_indexes = listing.get("size_tag_photo_indexes") or []
        source_indexes = [
            index for index in tag_indexes if 0 <= index < len(photos)
        ][:1]
        caption = "Tag photo"

        if not source_indexes and photos:
            source_indexes = [0]
            caption = "No tag photo identified — showing photo 1"

        images_html = []
        for photo_index in source_indexes:
            try:
                data_url = make_item_thumbnail_data_url(
                    photos[photo_index], max_size=600
                )
            except Exception:
                data_url = ""
            if data_url:
                images_html.append(f'<img src="{data_url}" alt="">')

    if not images_html:
        return f'<div class="evidence-hover-label">{html.escape(label)}</div>'

    return (
        f'<div class="evidence-hover-label">{html.escape(label)}'
        '<div class="evidence-hover-popover">'
        + "".join(images_html)
        + f'<div class="evidence-hover-caption">{html.escape(caption)}</div>'
        "</div></div>"
    )


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


def _normalize_category_string(value):
    """Collapse case/whitespace so a near-miss AI category string
    (extra spaces, different capitalization, "a>b" vs "a > b") still
    matches the canonical Shopify taxonomy path instead of forcing
    Manual Review."""
    return re.sub(
        r"\s*>\s*",
        ">",
        re.sub(r"\s+", " ", str(value or "").strip().lower()),
    )


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


UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

st.set_page_config(
    page_title="Depop AI",
    page_icon="👕",
    layout="wide"
)


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
  border: 1px solid rgba(43,33,48,0.14);
  border-radius: 8px;
  background: #FFFFFF;
  color: #2B2130;
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
  border: 1px solid rgba(43,33,48,0.14);
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

/* Same shape/geometry as .deck-toggle (border-radius, padding, weight) —
   only the color semantics differ: destructive red vs. neutral toggle. */
.delete-btn {
  border: 1px solid rgba(220,38,38,0.30);
  color: #DC2626;
  background: #FFF5F5;
  border-radius: 8px;
  padding: 8px 12px;
  cursor: pointer;
  font-size: 14px;
  font-weight: 700;
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
  border: 1px solid rgba(43,33,48,0.14);
  border-radius: 8px;
  padding: 5px;
  background: #FFFFFF;
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
  border-color: #F02AA0;
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
  border: 1px solid rgba(43,33,48,0.22);
  border-radius: 8px;
  background: #FBF3E3;
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
  background: #FBF3E3;
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
  border-radius: 8px;
  background: rgba(10,13,18,.78);
  color: #2B2130;
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
  color: #DC2626;
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
  border: 1px dashed rgba(43,33,48,0.14);
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
          name: file.name.replace(/\\.[^.]+$/, "") + ".jpg",
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



# ============================================================
# INVENTORY DECK + REVIEW CAROUSEL
# ============================================================

INVENTORY_DECK_HTML = """
<div class="deck-shell" data-deck>
  <div class="deck-topbar">
    <div>
      <div class="deck-kicker" data-kicker>ITEM DECK</div>
      <div class="deck-count" data-count>0 items</div>
    </div>
    <button class="deck-toggle" data-toggle type="button">Spread</button>
  </div>
  <div class="deck-stage" data-stage></div>
</div>
"""

INVENTORY_DECK_CSS = """
.deck-shell {
  width:100%;
  min-height:0;
  padding:10px 2px 18px;
  box-sizing:border-box;
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  color:#2B2130;
}
.deck-topbar {
  display:flex;
  justify-content:space-between;
  align-items:center;
  margin-bottom:12px;
}
.deck-kicker { font-size:11px; letter-spacing:.14em; opacity:.55; font-weight:700; }
.deck-count { font-size:18px; font-weight:800; }
.deck-toggle {
  border:1px solid rgba(43,33,48,0.16);
  background:#FFFFFF;
  color:#2B2130;
  border-radius:8px;
  padding:8px 12px;
  cursor:pointer;
  font-weight:700;
}
.deck-stage {
  position:relative;
  height:350px;
  min-height:350px;
  display:flex;
  align-items:center;
  justify-content:center;
  overflow:visible;
  padding:18px 4px;
}
.deck-card {
  position:absolute;
  left:50%;
  top:50%;
  width:248px;
  height:305px;
  margin-left:-124px;
  margin-top:-152px;
  border:1px solid rgba(43,33,48,0.14);
  border-radius:16px;
  background:#FFFFFF;
  box-shadow:0 18px 45px rgba(0,0,0,.32);
  overflow:hidden;
  cursor:pointer;
  transition:transform .38s cubic-bezier(.2,.8,.2,1),
             left .38s cubic-bezier(.2,.8,.2,1),
             top .38s cubic-bezier(.2,.8,.2,1),
             opacity .25s ease,
             box-shadow .25s ease;
  user-select:none;
}
.deck-card:hover {
  box-shadow:0 22px 55px rgba(0,0,0,.48);
}
.deck-card.active {
  outline:2px solid rgba(255,255,255,.28);
}
.deck-gallery {
  padding:9px 9px 0;
  height:192px;
  box-sizing:border-box;
  display:grid;
  grid-template-columns:minmax(0, 1fr) 105px;
  gap:8px;
  background:#FFFFFF;
}
.deck-main-photo {
  width:100%;
  height:170px;
  object-fit:cover;
  object-position:center;
  display:block;
  border-radius:8px;
  background:#FBF3E3;
}
.deck-extra-photos {
  display:grid;
  grid-template-columns:repeat(2, 46px);
  grid-auto-rows:46px;
  gap:6px;
  align-content:start;
  justify-content:center;
  padding-top:1px;
}
.deck-extra-photo {
  width:46px;
  height:46px;
  object-fit:cover;
  object-position:center;
  display:block;
  border-radius:8px;
  background:#FBF3E3;
  border:1px solid rgba(255,255,255,.14);
}
.deck-extra-empty {
  width:46px;
  height:46px;
  border-radius:8px;
  box-sizing:border-box;
  background:rgba(255,255,255,.04);
  border:1px dashed rgba(255,255,255,.10);
}
.deck-body { padding:7px 12px 9px; }
.deck-num { font-size:11px; opacity:.55; font-weight:800; }
.deck-title { font-size:15px; font-weight:800; margin-top:2px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.deck-meta { font-size:12px; opacity:.65; margin-top:3px; }
.deck-status { position:absolute; right:10px; top:10px; width:9px; height:9px; border-radius:50%; background:#39d98a; box-shadow:0 0 0 4px rgba(57,217,138,.12); }
.deck-card[data-status="review"] .deck-status { background:#f5c451; box-shadow:0 0 0 4px rgba(245,196,81,.12); }
.deck-card[data-status="processing"] .deck-status { background:#B39DDB; animation:pulse 1s infinite; }
@keyframes pulse { 50% { opacity:.35; } }

 .deck-shell.is-spread .deck-card {
  position:relative;
  left:auto !important;
  top:auto !important;
  margin:0 !important;
  transform:none !important;
  flex:0 0 248px;
}
.deck-shell.is-spread .deck-stage {
  justify-content:flex-start;
  overflow-x:auto;
  overflow-y:hidden;
  gap:14px;
  padding:16px 8px 24px;
}
.deck-shell:not(.is-spread) .deck-card:nth-child(1) {
  z-index:50;
  transform:translate(0,0) rotate(-1deg);
}
.deck-shell:not(.is-spread) .deck-card:nth-child(2) {
  z-index:49;
  transform:translate(16px,-8px) rotate(2deg);
}
.deck-shell:not(.is-spread) .deck-card:nth-child(3) {
  z-index:48;
  transform:translate(32px,-16px) rotate(-3deg);
}
.deck-shell:not(.is-spread) .deck-card:nth-child(4) {
  z-index:47;
  transform:translate(48px,-24px) rotate(4deg);
}
.deck-shell:not(.is-spread) .deck-card:nth-child(n+5) {
  z-index:46;
  opacity:0;
  pointer-events:none;
  transform:translate(58px,-24px) scale(.94);
}

.deck-shell.is-spread .deck-card {
  scroll-snap-align:start;
}
.deck-shell.is-spread .deck-card:hover {
  transform:translateY(-8px) !important;
}
.deck-empty {
  width:100%;
  border:1px dashed rgba(43,33,48,0.16);
  border-radius:16px;
  min-height:190px;
  display:flex;
  align-items:center;
  justify-content:center;
  opacity:.55;
}
"""

INVENTORY_DECK_JS = """
export default function({data, parentElement, setTriggerValue}) {
  // V2 components are mounted in an isolated shadow root. Using
  // document.querySelector() cannot see the component's own DOM.
  const root = parentElement.querySelector('[data-deck]');
  if (!root) return;
  const stage = root.querySelector('[data-stage]');
  const count = root.querySelector('[data-count]');
  const toggle = root.querySelector('[data-toggle]');
  let items = [];
  try {
    items = Array.isArray(data?.items)
      ? data.items
      : JSON.parse(data?.items_json || '[]');
  } catch (e) {
    items = [];
  }

  const spread = !!data?.spread;

  count.textContent = `${items.length} item${items.length === 1 ? '' : 's'}`;
  root.classList.toggle('is-spread', spread);
  toggle.textContent = spread ? 'Stack' : 'Spread';

  stage.innerHTML = '';

  if (!items.length) {
    stage.innerHTML = '<div class="deck-empty">Your items will appear here as a deck.</div>';
  }

  items.forEach((item, index) => {
    const card = document.createElement('div');
    card.className = 'deck-card';
    card.dataset.status = item.status || 'ready';

    const gallery = document.createElement('div');
    gallery.className = 'deck-gallery';

    const main = document.createElement('img');
    main.className = 'deck-main-photo';
    main.src = (item.photo_thumbs && item.photo_thumbs[0])
      || item.thumb
      || '';
    main.alt = '';
    gallery.appendChild(main);

    const extras = document.createElement('div');
    extras.className = 'deck-extra-photos';

    const photos = Array.isArray(item.photo_thumbs)
      ? item.photo_thumbs.slice(1, 7)
      : [];

    for (let slot = 0; slot < 6; slot += 1) {
      if (photos[slot]) {
        const extra = document.createElement('img');
        extra.className = 'deck-extra-photo';
        extra.src = photos[slot];
        extra.alt = '';
        extras.appendChild(extra);
      } else {
        const empty = document.createElement('span');
        empty.className = 'deck-extra-empty';
        extras.appendChild(empty);
      }
    }

    gallery.appendChild(extras);
    card.appendChild(gallery);

    const status = document.createElement('span');
    status.className = 'deck-status';
    card.appendChild(status);

    const body = document.createElement('div');
    body.className = 'deck-body';
    body.innerHTML =
      `<div class="deck-num">ITEM ${item.number}</div>` +
      `<div class="deck-title">${escapeHtml(item.title || ('Item ' + item.number))}</div>` +
      `<div class="deck-meta">${item.photos || 0} photos${item.price ? ' · $' + item.price : ''}</div>`;
    card.appendChild(body);

    card.addEventListener('click', (event) => {
      if (event.target.closest('button')) return;
      setTriggerValue('action', {type:'open_item', index, item_number:Number(item.number)});
    });

    stage.appendChild(card);
  });

  toggle.onclick = (event) => {
    event.stopPropagation();
    setTriggerValue('action', {type:'toggle_spread', spread:!spread});
  };

  function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, c => ({
      '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'
    }[c]));
  }
}
"""

if STREAMLIT_V2_AVAILABLE:
    inventory_deck = _streamlit_v2_component(
        "inventory_deck_v2",
        html=INVENTORY_DECK_HTML,
        css=INVENTORY_DECK_CSS,
        js=INVENTORY_DECK_JS,
        isolate_styles=True,
    )
else:
    inventory_deck = None


@st.cache_data(
    show_spinner=False,
    max_entries=1000,
)
def _cached_thumbnail_data_url(path_string, modified_ns, file_size, max_size):
    """Cached browser thumbnail.

    modified_ns + file_size are part of the cache key, so if a photo is
    permanently rotated/re-written, Streamlit automatically generates a
    fresh thumbnail instead of showing a stale one.
    """
    path = Path(path_string)

    try:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image)
            image = image.convert("RGB")
            image.thumbnail((max_size, max_size))

            buffer = io.BytesIO()
            image.save(
                buffer,
                format="JPEG",
                quality=72,
                optimize=False,
            )

            return (
                "data:image/jpeg;base64,"
                + base64.b64encode(buffer.getvalue()).decode("ascii")
            )
    except Exception:
        return ""


def make_item_thumbnail_data_url(path, max_size=260):
    """Return a cached compact preview for a saved item photo."""
    path = Path(path)

    try:
        stat = path.stat()
    except OSError:
        return ""

    return _cached_thumbnail_data_url(
        str(path.resolve()),
        int(stat.st_mtime_ns),
        int(stat.st_size),
        int(max_size),
    )

def build_item_component_photos(paths, max_size=900):
    """max_size defaults much larger than make_item_thumbnail_data_url's
    own 260px default — this feeds the OPENED/expanded item card view
    (the only caller), where grid cells commonly render well past
    260px wide. Serving the same tiny grid-thumbnail asset there was
    upscaling a 260px/quality-72 JPEG, which is exactly what was
    causing the "images are blurry" complaint. 900px is a deliberate
    middle ground: sharp at typical card-grid sizes without paying
    the full 1400px cost (used for the single dedicated full-screen
    zoom dialog elsewhere) across every photo in the grid at once.
    """
    photos = []

    for index, path in enumerate(paths):

        path = Path(path)

        photos.append({
            "id": str(
                path.resolve()
            ),
            "name": path.name,
            "src": make_item_thumbnail_data_url(
                path, max_size=max_size
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
                # Bake any existing EXIF orientation into the pixels
                # first, then apply exactly one 90° clockwise manual
                # rotation. The resulting pixels are the permanent source
                # of truth for every later step, including Shopify.
                _rotate_photo_in_place(path, 90)

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


# Per-item review-widget keys are plain positional item_index, not a
# stable per-item id, so they must be dropped explicitly whenever
# generated_listings is rebuilt — otherwise a stale generated_brand_3 /
# generated_shopify_size_3_select from a PRIOR batch survives and
# silently shows on whatever new item lands at index 3.
_ITEM_WIDGET_STATE_PREFIXES = (
    "generated_",
    "pricing_",
    "review_saved_",
    "save_review_",
    "fix_listing_",
    "toggle_optional_",
    "ai_fill_optional_",
    "description_size_sync_",
    "rescrape_depop_",
)


def clear_generated_item_widget_state(count):
    """Pop every per-item review-widget key for indices [0, count)."""

    if count <= 0:
        return

    stale_suffixes = tuple(
        f"_{i}" for i in range(count)
    )

    for key in list(st.session_state.keys()):
        if not key.startswith(_ITEM_WIDGET_STATE_PREFIXES):
            continue

        base = key
        if base.endswith("_select"):
            base = base[: -len("_select")]
        elif base.endswith("_custom"):
            base = base[: -len("_custom")]

        if base.endswith(stale_suffixes):
            st.session_state.pop(key, None)


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


# ============================================================
# PAGE
# ============================================================

@st.cache_data(show_spinner=False)
def _brand_logo_data_url():
    """Base64-embeds the brand logo (background already removed,
    resized/palette-quantized to ~12KB) so the header can use it as a
    plain <img src="data:..."> with no separate file request — cached
    so the file isn't re-read/re-encoded on every rerun (Streamlit
    reruns this whole script on every interaction)."""
    path = Path(__file__).with_name("logo_transparent.png")
    if not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


st.markdown(
    """
    <style>
    .brand-header {
        background: linear-gradient(135deg, #FBF3E3 0%, #FFFFFF 65%);
        border: 1px solid rgba(240, 42, 160, .16);
        border-radius: 22px;
        padding: 30px 34px;
        margin-bottom: 20px;
        box-shadow: 0 12px 32px rgba(240, 42, 160, .10);
        position: relative;
        overflow: hidden;
    }
    .brand-header::before {
        content: "";
        position: absolute;
        top: -50px;
        right: -50px;
        width: 190px;
        height: 190px;
        background: radial-gradient(
            circle, rgba(240, 42, 160, .16), transparent 70%
        );
        border-radius: 50%;
        pointer-events: none;
    }
    .brand-header::after {
        content: "";
        position: absolute;
        bottom: -60px;
        left: 20%;
        width: 160px;
        height: 160px;
        background: radial-gradient(
            circle, rgba(251, 243, 227, .9), transparent 70%
        );
        border-radius: 50%;
        pointer-events: none;
    }
    /* The rotating shimmer — a real element, not a pseudo-element,
       since ::before/::after above are already spoken for by the two
       static glow blobs. A slow-spinning conic gradient sits behind
       the title/subtitle (z-index below them, clipped by the card's
       own overflow:hidden) for a genuinely moving light sweep instead
       of a static glow. */
    .brand-header-shimmer {
        position: absolute;
        inset: -60%;
        background: conic-gradient(
            from 0deg,
            transparent 0deg,
            rgba(240, 42, 160, .22) 45deg,
            transparent 110deg,
            transparent 220deg,
            rgba(255, 110, 199, .18) 285deg,
            transparent 360deg
        );
        animation: brandShimmerSpin 14s linear infinite;
        pointer-events: none;
        z-index: 0;
    }
    @keyframes brandShimmerSpin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    .brand-header-title {
        position: relative;
        z-index: 1;
        font-size: 36px;
        font-weight: 900;
        color: #2B2130;
        display: flex;
        align-items: center;
        gap: 12px;
        letter-spacing: -.02em;
        line-height: 1.15;
    }
    .brand-header-title .brand-accent {
        background: linear-gradient(
            100deg,
            #F02AA0 30%,
            #FFC1E9 45%,
            #FF6EC7 55%,
            #F02AA0 70%
        );
        background-size: 250% 100%;
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        animation: brandTextShimmer 4s ease-in-out infinite;
    }
    @keyframes brandTextShimmer {
        0% { background-position: 200% 0; }
        100% { background-position: -100% 0; }
    }
    .brand-star {
        font-size: 30px;
        filter: drop-shadow(0 2px 6px rgba(240, 42, 160, .35));
    }
    .brand-header-sub {
        position: relative;
        z-index: 1;
        margin-top: 9px;
        font-size: 14.5px;
        color: #6B5B66;
        max-width: 640px;
    }
    .brand-step {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 4px 0 10px 0;
    }
    .brand-step-num {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 28px;
        height: 28px;
        border-radius: 50%;
        background: linear-gradient(135deg, #F02AA0, #FF6EC7);
        color: #fff;
        font-weight: 900;
        font-size: 14px;
        box-shadow: 0 4px 10px rgba(240, 42, 160, .35);
        flex-shrink: 0;
    }
    .brand-step-text {
        font-size: 22px;
        font-weight: 800;
        color: #2B2130;
    }
    .brand-logo-img {
        display: block;
        height: 76px;
        width: auto;
        max-width: 100%;
        object-fit: contain;
        filter: drop-shadow(0 4px 10px rgba(240, 42, 160, .25));
    }
    /* Closest reasonable match to the logo's own chunky, rounded Y2K
       lettering — there's no image-generation tool available to paint
       new lettering directly into the logo artwork itself, so this
       pairs it with real, crisp, same-colored text next to it instead
       of trying to fake it as a raster edit. */
    @import url('https://fonts.googleapis.com/css2?family=Baloo+2:wght@700;800&display=swap');
    .brand-logo-row {
        position: relative;
        z-index: 1;
        display: flex;
        align-items: center;
        gap: 18px;
        flex-wrap: wrap;
    }
    .brand-wordmark-text {
        font-family: 'Baloo 2', sans-serif;
        font-weight: 800;
        font-size: 28px;
        line-height: 1.05;
        color: #F02AA0;
        letter-spacing: .01em;
        text-shadow: 0 2px 0 rgba(255, 255, 255, .55);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

_logo_data_url = _brand_logo_data_url()

st.markdown(
    f"""
    <div class="brand-header">
        <div class="brand-header-shimmer"></div>
        <div class="brand-header-title">
            {
                f'''<div class="brand-logo-row">
                    <img class="brand-logo-img" src="{_logo_data_url}" alt="Womens Y2K">
                    <span class="brand-wordmark-text">AI Listing Generator</span>
                </div>'''
                if _logo_data_url else
                '<span class="brand-star">✨</span> Depop '
                '<span class="brand-accent">AI Listing Machine</span>'
            }
        </div>
        <div class="brand-header-sub">
            Upload clothing photos and let AI group, generate, QA, and
            prepare listings for Shopify.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


def _brand_step_header(number, text):
    st.markdown(
        f"""
        <div class="brand-step">
            <span class="brand-step-num">{number}</span>
            <span class="brand-step-text">{html.escape(text)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================

# ============================================================
# ============================================================
# PERMANENT UPLOAD ORIENTATION
# Manual rotation is the only orientation change after upload.
# ============================================================

def normalize_uploaded_image_to_path(uploaded_file, file_path):
    """Bake EXIF orientation into pixels only when necessary.

    Most phone photos already have normal orientation. In that common case
    we copy the uploaded bytes directly instead of opening, decoding,
    recompressing, and rewriting the image.
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    try:
        uploaded_file.seek(0)
        raw = uploaded_file.getvalue()

        # Fast path: inspect orientation without doing a full pixel rewrite.
        try:
            with Image.open(io.BytesIO(raw)) as source:
                orientation = int(source.getexif().get(274, 1) or 1)
        except Exception:
            orientation = 1

        if orientation == 1:
            file_path.write_bytes(raw)
            return file_path

        # Only rotated/EXIF-oriented files take the expensive PIL path.
        with Image.open(io.BytesIO(raw)) as source:
            image = ImageOps.exif_transpose(source).copy()

        if suffix in {".jpg", ".jpeg"}:
            if image.mode not in ("RGB", "L"):
                image = image.convert("RGB")
            image.save(
                file_path,
                format="JPEG",
                quality=92,
                optimize=False,
                exif=b"",
            )
        elif suffix == ".png":
            image.save(file_path, format="PNG")
        elif suffix == ".webp":
            image.save(
                file_path,
                format="WEBP",
                quality=90,
                method=0,
            )
        else:
            image = image.convert("RGB")
            image.save(
                file_path,
                format="JPEG",
                quality=92,
                optimize=False,
                exif=b"",
            )

        return file_path

    except Exception:
        # Fall back to the raw upload so one bad image cannot break a batch.
        try:
            uploaded_file.seek(0)
            file_path.write_bytes(uploaded_file.getvalue())
            return file_path
        except Exception:
            raise

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



def manual_upload_signature(uploaded_files):
    """Stable signature so Streamlit reruns don't overwrite rotated files."""
    return [
        (
            str(getattr(file, "name", "")),
            int(getattr(file, "size", 0) or 0),
        )
        for file in (uploaded_files or [])
    ]


# rotate_photo_in_place() moved to ai_listing.py — it has no Streamlit
# dependency and qa_review.py needs it too (see the Final QA rotation
# safety net). Keep the historical local name as an alias so none of
# the existing call sites below need to change.
_rotate_photo_in_place = rotate_photo_in_place


# Small circular icon buttons overlaid on each review-gallery photo's
# corners (🔍 top-left, ↻ bottom-right, ✕ top-right). Each button lives
# in its OWN st.container(key=...) nested inside the photo's own
# st.container(key=...) — Streamlit gives every st.container(key=...)
# a matching st-key-<key> class, which is what these selectors target.
#
# Same fix as the upload deck's thumbnail-centering bug: Streamlit puts
# position:relative on every stElementContainer wrapper by default, so
# left alone, each wrapper becomes its OWN positioning root instead of
# the photo box being the one shared anchor. Forcing every direct-child
# wrapper back to static is what lets the three button containers'
# position:absolute correctly resolve against the photo box itself.
_REVIEW_PHOTO_BOX_CSS = """
<style>
div[class*="st-key-review_pbox_"] {
    position: relative !important;
    margin-bottom: 8px !important;
    /* The box's flex-column children are the image PLUS 3 nested
       containers for the corner buttons — those render at 0 height
       (confirmed live) since their content is pulled out via
       position:absolute, but Streamlit's default gap between flex
       children still reserved space for each of those 3 gaps,
       inflating the box ~48px taller than the image itself. That
       left "bottom:8px" anchoring below the image's true edge
       instead of hugging it. Zero the gap so the box's height
       matches the image exactly. */
    gap: 0 !important;
}
div[class*="st-key-review_pbox_"] > div[data-testid="stElementContainer"] {
    position: static !important;
}
/* st.image's real DOM chain (confirmed live via devtools) is FIVE
   flex divs deep before the <img> itself: stFullScreenFrame > an
   unnamed wrapper div > stImage > stImageContainer > img. Every one
   of those is a flex container sized to the image's INTRINSIC pixel
   size, not the box — width="stretch" alone, and even forcing just
   stImage or just the <img>, left the real width untouched (confirmed:
   img carried an inline width style that still computed to the
   intrinsic ~184px because its flex ANCESTORS never grew past that).
   Force width AND height through the whole chain so the percentage
   cascade actually reaches the <img>. */
div[class*="st-key-review_pbox_"] div[data-testid="stFullScreenFrame"],
div[class*="st-key-review_pbox_"] div[data-testid="stFullScreenFrame"] > div,
div[class*="st-key-review_pbox_"] div[data-testid="stImage"],
div[class*="st-key-review_pbox_"] div[data-testid="stImageContainer"] {
    width: 100% !important;
    height: 100% !important;
    max-width: none !important;
    max-height: none !important;
}
/* Plain "div[class*=...] img" ties Streamlit's own object-fit rule on
   specificity (both are one attribute/class + one type selector), and
   Streamlit's wins the tie by coming later in the cascade — img stayed
   "contain" (letterboxed) despite this !important. Routing through
   stImageContainer adds a selector segment, which wins outright. */
div[class*="st-key-review_pbox_"] div[data-testid="stImageContainer"] img {
    width: 100% !important;
    height: 100% !important;
    max-height: none !important;
    object-fit: cover !important;
    border-radius: 10px !important;
    display: block !important;
    background: #FBF3E3 !important;
}
/* Streamlit's own native "view fullscreen" hover control on st.image
   is a SEPARATE mechanism from the 🔍/↻/✕ corner buttons this photo
   box already builds — left enabled, it's a second, unexpected way
   to pop a fullscreen image overlay (with Streamlit's own close
   button, nothing to do with review_photo_zoom_state) just from a
   click landing anywhere near the image instead of squarely on one
   of the three custom buttons. Disable it outright; the 🔍 button is
   the only zoom entry point this box needs. */
div[class*="st-key-review_pbox_"] div[data-testid="stElementToolbar"],
div[class*="st-key-review_pbox_"] [data-testid="StyledFullScreenButton"],
div[class*="st-key-review_pbox_"] button[title*="fullscreen" i],
div[class*="st-key-review_pbox_"] button[aria-label*="fullscreen" i] {
    display: none !important;
    visibility: hidden !important;
    pointer-events: none !important;
}
/* Stretch to match the extras grid's actual height (2 or 3 rows,
   whatever it ends up being) instead of a fixed guessed ratio — the
   outer main_col/extras_col row is a flex row with default
   align-items:stretch, so both columns already share the same
   height; this just makes the photo box/frame fill that height all
   the way down instead of sizing to its own content. */
div[class*="st-key-review_pbox_main_"] {
    height: 100% !important;
}
/* Streamlit wraps each column's content in an stLayoutWrapper that
   sizes to its own content (auto) rather than inheriting the
   stretched stColumn height above it (confirmed live: stColumn was
   a definite 571px, stLayoutWrapper still only 387px) — bridge that
   gap explicitly, or the 100% below it has nothing definite to
   resolve against. */
div[data-testid="stLayoutWrapper"]:has(> div[class*="st-key-review_pbox_main_"]) {
    height: 100% !important;
}
div[class*="st-key-review_pbox_main_"] > div[data-testid="stElementContainer"] {
    height: 100% !important;
}
div[class*="st-key-review_pbox_main_"] div[data-testid="stFullScreenFrame"] {
    height: 100% !important;
}
/* Extras are perfect squares matching the inventory deck's
   .deck-extra-photo tiles — aspect-ratio instead of a fixed pixel
   height so each tile stays square at whatever width the 2-column
   grid below gives it. */
div[class*="st-key-review_pbox_extra_"] div[data-testid="stFullScreenFrame"] {
    aspect-ratio: 1 / 1 !important;
    height: auto !important;
}
/* Extras grid: two columns of small squares, same shape as the
   inventory deck's .deck-extra-photos grid, instead of a single
   horizontal-scrolling row. */
div[class*="st-key-review_extras_grid_"] div[data-testid="stHorizontalBlock"] {
    gap: 8px !important;
    margin-bottom: 8px !important;
}
div[class*="st-key-review_extras_grid_"] div[data-testid="stColumn"] {
    gap: 0 !important;
}
/* Matches the loaded photo boxes' own placeholder color (#FBF3E3,
   set on the <img> background rule above). Border switched from a
   semi-transparent white (invisible on this card's light background —
   the actual bug behind "still grey/looks broken" reports) to a
   visible plum-tinted dashed line. */
.review-extra-empty {
    width: 100%;
    aspect-ratio: 1 / 1;
    border-radius: 10px;
    box-sizing: border-box;
    background: #FBF3E3;
    border: 1px dashed rgba(43, 33, 48, .25);
    margin-bottom: 8px;
}
div[class*="st-key-review_pbox_zoom_"],
div[class*="st-key-review_pbox_rotate_"],
div[class*="st-key-review_pbox_del_"] {
    position: absolute !important;
    z-index: 30 !important;
    width: 28px !important;
    height: 28px !important;
    margin: 0 !important;
    padding: 0 !important;
}
div[class*="st-key-review_pbox_zoom_"] {
    top: 8px !important;
    left: 8px !important;
}
div[class*="st-key-review_pbox_rotate_"] {
    bottom: 8px !important;
    right: 8px !important;
}
div[class*="st-key-review_pbox_del_"] {
    top: 8px !important;
    right: 8px !important;
    width: 24px !important;
    height: 24px !important;
}
div[class*="st-key-review_pbox_zoom_"] div[data-testid="stButton"],
div[class*="st-key-review_pbox_rotate_"] div[data-testid="stButton"],
div[class*="st-key-review_pbox_del_"] div[data-testid="stButton"] {
    width: 100% !important;
    height: 100% !important;
}
div[class*="st-key-review_pbox_zoom_"] button,
div[class*="st-key-review_pbox_rotate_"] button,
div[class*="st-key-review_pbox_del_"] button {
    width: 100% !important;
    height: 100% !important;
    min-width: 0 !important;
    min-height: 0 !important;
    border-radius: 50% !important;
    padding: 0 !important;
    background: rgba(10,13,18,.82) !important;
    border: 1px solid rgba(255,255,255,.28) !important;
    color: #fff !important;
    font-size: 13px !important;
    line-height: 1 !important;
    box-shadow: 0 2px 8px rgba(0,0,0,.35) !important;
}
div[class*="st-key-review_pbox_zoom_"] button:hover,
div[class*="st-key-review_pbox_rotate_"] button:hover {
    background: rgba(30,35,44,.95) !important;
}
/* Smaller buttons + tighter corner offset on the 88px extra squares —
   the 28/24px sizing was tuned for the much bigger main photo; at
   extra-square scale the same sizes left too little edge clearance
   between the top-left and top-right buttons ("clashing"). */
div[class*="st-key-review_pbox_extra_"] div[class*="st-key-review_pbox_zoom_"],
div[class*="st-key-review_pbox_extra_"] div[class*="st-key-review_pbox_rotate_"] {
    width: 22px !important;
    height: 22px !important;
}
div[class*="st-key-review_pbox_extra_"] div[class*="st-key-review_pbox_zoom_"] {
    top: 5px !important;
    left: 5px !important;
}
div[class*="st-key-review_pbox_extra_"] div[class*="st-key-review_pbox_rotate_"] {
    bottom: 5px !important;
    right: 5px !important;
}
div[class*="st-key-review_pbox_extra_"] div[class*="st-key-review_pbox_del_"] {
    width: 18px !important;
    height: 18px !important;
    top: 5px !important;
    right: 5px !important;
}
div[class*="st-key-review_pbox_extra_"] div[class*="st-key-review_pbox_zoom_"] button,
div[class*="st-key-review_pbox_extra_"] div[class*="st-key-review_pbox_rotate_"] button,
div[class*="st-key-review_pbox_extra_"] div[class*="st-key-review_pbox_del_"] button {
    font-size: 10px !important;
}
div[class*="st-key-review_pbox_del_"] button {
    color: #DC2626 !important;
    font-size: 12px !important;
    font-weight: 800 !important;
}
div[class*="st-key-review_pbox_del_"] button:hover {
    border-color: #ef4444 !important;
    background: rgba(40,20,20,.9) !important;
}
</style>
"""


def _render_review_photo_box(item_index, photo_index, photo_path, all_photos, size):
    """One review-gallery photo: the image plus its 3 corner buttons.

    size is "main" or "extra" — purely a CSS hook for the height rules
    above, so the hero photo renders bigger than the smaller ones
    beside it, matching the original layout.
    """
    with st.container(key=f"review_pbox_{size}_{item_index}_{photo_index}"):
        # Both bumped up from the previous 500/220 — the "main" box
        # stretches to fill a flexible column that's often noticeably
        # wider than 500px, and "extra" thumbnails looked soft at
        # 220px on typical/high-DPI displays. Still well short of the
        # 1400px used for the dedicated single-photo zoom dialog,
        # since several of these render at once per item.
        thumb_url = make_item_thumbnail_data_url(
            photo_path,
            max_size=1100 if size == "main" else 420,
        )
        if thumb_url:
            st.image(thumb_url, width="stretch")

        with st.container(
            key=f"review_pbox_zoom_{item_index}_{photo_index}"
        ):
            if st.button(
                "🔍",
                key=f"review_photo_zoom_{item_index}_{photo_index}",
                help="View full size",
            ):
                st.session_state["review_photo_zoom_state"] = {
                    "item_index": item_index,
                    "photos": all_photos,
                    "index": photo_index,
                }
                # "Armed" flag, consumed the instant it's acted on (see
                # the render-trigger site) — only an explicit click
                # inside the dialog (Prev/Next/Rotate) re-arms it before
                # its own rerun. Dismissing the dialog via Escape/
                # clicking outside it runs no app code, so this flag is
                # left un-armed — the next unrelated button click
                # elsewhere on the page won't resurrect this photo.
                st.session_state["review_photo_zoom_open"] = True
                st.rerun()

        with st.container(
            key=f"review_pbox_rotate_{item_index}_{photo_index}"
        ):
            if st.button(
                "↻",
                key=f"review_photo_rotate_{item_index}_{photo_index}",
                help="Rotate 90° clockwise",
            ):
                try:
                    _rotate_photo_in_place(photo_path, 90)
                    manual_sync_bulk_paths()
                    clear_qa_state()
                except Exception as exc:
                    st.error(f"Could not rotate: {exc}")
                st.rerun()

        with st.container(
            key=f"review_pbox_del_{item_index}_{photo_index}"
        ):
            if st.button(
                "✕",
                key=f"review_photo_delete_{item_index}_{photo_index}",
                help="Remove this photo (wrong item / duplicate)",
            ):
                current_item = st.session_state["generated_listings"][
                    item_index
                ]
                current_item["photos"] = [
                    p
                    for p in current_item.get("photos", [])
                    if str(p) != str(photo_path)
                ]
                clear_qa_state()
                st.rerun()


@st.dialog("Photo review", width="large")
def _review_photo_zoom_dialog():
    """Click-to-zoom photo viewer for the Generated Listings Review
    gallery — a real Streamlit modal (built-in X close) so a listing's
    size tag/details can be inspected at full size, with prev/next
    across the item's own photos and a rotate button for fixing a
    missed auto-straighten mistake on the spot."""
    state = st.session_state.get("review_photo_zoom_state")

    if not state:
        return

    photos = state.get("photos", [])

    if not photos:
        st.warning("No photos to show.")
        return

    index = max(
        0,
        min(len(photos) - 1, int(state.get("index", 0))),
    )
    state["index"] = index
    path = photos[index]

    data_url = make_item_thumbnail_data_url(path, max_size=1400)

    if data_url:
        st.image(data_url, width="stretch")
    else:
        st.warning("Could not load this photo.")

    st.caption(
        f"Photo {index + 1} of {len(photos)} — {Path(path).name}"
    )

    nav_cols = st.columns(4)

    with nav_cols[0]:
        if st.button(
            "← Prev",
            key="review_photo_zoom_prev",
            disabled=index <= 0,
            width="stretch",
        ):
            state["index"] = index - 1
            st.session_state["review_photo_zoom_open"] = True
            st.rerun()

    with nav_cols[1]:
        if st.button(
            "Next →",
            key="review_photo_zoom_next",
            disabled=index >= len(photos) - 1,
            width="stretch",
        ):
            state["index"] = index + 1
            st.session_state["review_photo_zoom_open"] = True
            st.rerun()

    with nav_cols[2]:
        if st.button(
            "↻ Rotate",
            key="review_photo_zoom_rotate",
            width="stretch",
            help="Fix a sideways/upside-down photo.",
        ):
            try:
                _rotate_photo_in_place(path, 90)
                manual_sync_bulk_paths()
                clear_qa_state()
            except Exception as exc:
                st.error(f"Could not rotate: {exc}")
            st.session_state["review_photo_zoom_open"] = True
            st.rerun()

    with nav_cols[3]:
        if st.button(
            "✕ Close",
            key="review_photo_zoom_close",
            width="stretch",
        ):
            st.session_state["review_photo_zoom_state"] = None
            st.session_state["review_photo_zoom_open"] = False
            st.rerun()


def manual_slot_sync_files(slot_index, uploaded_files):
    """Persist one card independently.

    IMPORTANT: Streamlit can temporarily return an empty list for a
    previously-used uploader when another uploader triggers a rerun.
    An empty transient value must NOT clear this card. Each card owns
    its saved paths independently.
    """
    slots = st.session_state["manual_upload_slots"]
    slot = slots[slot_index]

    incoming = list(uploaded_files or [])

    # CRITICAL: do not wipe this card when another card causes a rerun.
    # Empty is a transient Streamlit widget state, not a user deletion.
    if not incoming:
        return

    old_paths = list(slot.get("paths", []))

    # Match existing saved files by their original filename suffix.
    existing_by_name = {}
    for path in old_paths:
        p = Path(path)
        existing_by_name[p.name] = path

    new_paths = []

    for uploaded_file in incoming:
        filename = Path(uploaded_file.name).name

        existing = None
        for saved_name, saved_path in existing_by_name.items():
            if (
                saved_name == filename
                or saved_name.endswith("_" + filename)
            ):
                if Path(saved_path).exists():
                    existing = saved_path
                    break

        if existing:
            new_paths.append(existing)
        else:
            saved = manual_save_uploaded_files(
                [uploaded_file],
                f"slot_{slot_index + 1}",
            )
            new_paths.extend(saved)

    # Preserve the exact order returned by Streamlit.
    slot["paths"] = new_paths
    slot["signature"] = manual_upload_signature(incoming)

    st.session_state["manual_upload_slots"] = slots

def manual_make_group(
    paths,
    item_number,
    marked_vintage=False,
):
    return {
        "item_number": item_number,
        "paths": list(paths),
        "photo_numbers": [],
        "anchor_photo": None,
        "label": f"Item {item_number}",
        "title": f"Item {item_number}",
        "reason": "Manually grouped by user.",
        "marked_vintage": bool(marked_vintage),
    }


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
# ------------------------------------------------------------

# ------------------------------------------------------------

# ------------------------------------------------------------

# ------------------------------------------------------------

# ------------------------------------------------------------
# ------------------------------------------------------------
# 10-ITEM SKINNY UPLOAD DECK
# ------------------------------------------------------------
# Five skinny cards across, five underneath.
# The entire card is the actual Streamlit dropzone.
# Only four tiny photo previews are visible at once.
# Hover a preview to inspect it larger.
# An arrow appears ONLY when there are more than four photos.
st.markdown(
    """
    <style>
    /* =========================================================
       DECK GEOMETRY
       ========================================================= */

    .upload-deck-card-shell {
        position: absolute !important;
        inset: 0 !important;
        z-index: 20 !important;
        transform: none !important;
        width: 100% !important;
        height: 120px !important;
        min-height: 120px !important;
        margin: 0 !important;
        padding: 0 !important;
        border: 0 !important;
        background: transparent !important;
        box-shadow: none !important;
        pointer-events: none !important;
        overflow: visible !important;
    }

    /* The actual Streamlit column becomes the card. */
    div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .upload-deck-card-shell) {
        position: relative !important;
        height: 120px !important;
        min-height: 120px !important;
        margin-bottom: 8px !important;
        box-sizing: border-box !important;
        border: 1px solid rgba(43,33,48,0.14) !important;
        border-radius: 12px !important;
        background: #FFFFFF !important;
        box-shadow: 0 6px 14px rgba(0,0,0,.18) !important;
        overflow: visible !important;
    }

    div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .upload-deck-card-shell):hover {
        border-color: #F02AA0 !important;
        box-shadow: 0 9px 18px rgba(0,0,0,.24) !important;
    }

    /* Streamlit gives every widget/markdown call its own
       stElementContainer wrapper, and that wrapper itself gets
       position:relative from Streamlit's base CSS. Left alone, each
       wrapper becomes its OWN positioning root — so a later widget's
       inset:0/top:Npx (the file uploader, the photo strip, the title)
       resolves against wherever ITS wrapper happens to sit in the
       column's normal document flow, not against the card. That's why
       the dropzone hit-target and the photo strip were drifting
       downward: force every wrapper back to static so the column
       above is the only positioning root any of them see. */
    div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .upload-deck-card-shell) > div[data-testid="stElementContainer"] {
        position: static !important;
    }

    /* =========================================================
       ITEM TITLE
       ========================================================= */

    .upload-deck-card-title {
        position: absolute !important;
        top: 7px !important;
        left: 9px !important;
        right: 9px !important;
        z-index: 50 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: space-between !important;
        pointer-events: none !important;
    }

    .upload-deck-card-title .item-number {
        font-size: 10px !important;
        font-weight: 900 !important;
        letter-spacing: .05em !important;
        color: #2B2130 !important;
    }

    .upload-deck-card-title .item-count {
        font-size: 8px !important;
        font-weight: 700 !important;
        color: rgba(43,33,48,0.55) !important;
    }

    /* =========================================================
       WHOLE CARD = DROPZONE
       No visible Upload button.
       No "200MB per file" text.
       The native uploader remains underneath and still accepts
       drag/drop and click if the browser needs it.
       ========================================================= */

    div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .upload-deck-card-shell) div[data-testid="stFileUploader"] {
        position: absolute !important;
        inset: 0 !important;
        z-index: 5 !important;
        width: 100% !important;
        height: 120px !important;
        min-height: 120px !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .upload-deck-card-shell) div[data-testid="stFileUploader"] section,
    div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .upload-deck-card-shell) div[data-testid="stFileUploaderDropzone"] {
        position: absolute !important;
        inset: 0 !important;
        width: 100% !important;
        height: 120px !important;
        min-height: 120px !important;
        max-height: 120px !important;
        padding: 0 !important;
        box-sizing: border-box !important;
        border: 0 !important;
        border-radius: 12px !important;
        background: transparent !important;
        overflow: hidden !important;
    }

    /* COMPLETELY REMOVE THE NATIVE UPLOADER UI.
       The uploader remains full-card and transparent so the browser still
       accepts drag/drop and clicks anywhere on the Item card. */
    div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .upload-deck-card-shell) div[data-testid="stFileUploader"] {
        opacity: 1 !important;
        visibility: visible !important;
        pointer-events: auto !important;
    }

    /* The uploader is invisible chrome, but its actual file input spans
       the complete card so the whole card accepts drops/clicks. */
    div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .upload-deck-card-shell) div[data-testid="stFileUploader"] input[type="file"] {
        position: absolute !important;
        inset: 0 !important;
        width: 100% !important;
        height: 100% !important;
        opacity: 0 !important;
        cursor: pointer !important;
        z-index: 20 !important;
    }

    div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .upload-deck-card-shell) div[data-testid="stFileUploaderFile"] {
        display: none !important;
    }


    div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .upload-deck-card-shell)
    div[data-testid="stFileUploader"] button,
    div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .upload-deck-card-shell)
    div[data-testid="stFileUploader"] label,
    div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .upload-deck-card-shell)
    div[data-testid="stFileUploader"] small,
    div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .upload-deck-card-shell)
    div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"],
    div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .upload-deck-card-shell)
    div[data-testid="stFileUploader"] [data-testid="stFileUploaderFileData"] {
        opacity: 0 !important;
        visibility: hidden !important;
        pointer-events: none !important;
        display: none !important;
    }

    /* Kill every native selected-file chip / instruction block.
       Keep ONLY the actual input element alive. */
    div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .upload-deck-card-shell)
    div[data-testid="stFileUploader"] section > div:not(:has(input[type="file"])) {
        display: none !important;
    }

    div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .upload-deck-card-shell)
    div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"],
    div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .upload-deck-card-shell)
    div[data-testid="stFileUploader"] [data-testid="stFileUploaderFileData"],
    div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .upload-deck-card-shell)
    div[data-testid="stFileUploader"] [data-testid*="FileUploaderFile"],
    div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .upload-deck-card-shell)
    div[data-testid="stFileUploader"] ul,
    div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .upload-deck-card-shell)
    div[data-testid="stFileUploader"] li {
        display: none !important;
        visibility: hidden !important;
    }

    /* Hide the native label/button text, not the input. */
    div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .upload-deck-card-shell)
    div[data-testid="stFileUploader"] label {
        position: absolute !important;
        inset: 0 !important;
        width: 100% !important;
        height: 120px !important;
        padding: 0 !important;
        margin: 0 !important;
        border: 0 !important;
        background: transparent !important;
        color: transparent !important;
        font-size: 0 !important;
        box-shadow: none !important;
        overflow: hidden !important;
    }

    div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .upload-deck-card-shell)
    div[data-testid="stFileUploader"] label button,
    div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .upload-deck-card-shell)
    div[data-testid="stFileUploader"] label small,
    div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .upload-deck-card-shell)
    div[data-testid="stFileUploader"] label span {
        display: none !important;
        visibility: hidden !important;
    }

    div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .upload-deck-card-shell)
    div[data-testid="stFileUploader"] input[type="file"] {
        position: absolute !important;
        inset: 0 !important;
        width: 100% !important;
        height: 120px !important;
        max-width: none !important;
        max-height: none !important;
        opacity: 0 !important;
        cursor: pointer !important;
        z-index: 999 !important;
    }

    /* Exact card being dragged over turns green. */
    div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .upload-deck-card-shell).upload-card-drag-active {
        border: 2px solid #2fd879 !important;
        background: rgba(47,216,121,.10) !important;
        box-shadow:
            inset 0 0 0 2px rgba(47,216,121,.14),
            0 0 0 3px rgba(47,216,121,.14) !important;
    }

    div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .upload-deck-card-shell).upload-card-drag-active .upload-deck-card-shell {
        border-radius: 12px !important;
    }


    /* =========================================================
       FOUR TINY PHOTO SLOTS
       { 0  0  0  0  → }
       ========================================================= */

       .upload-deck-photo-strip {
        position: absolute !important;
        top: 60px !important;
        left: 50% !important;
        right: auto !important;
        bottom: auto !important;
        width: max-content !important;
        height: 42px !important;
        margin: 0 !important;
        padding: 0 !important;
        transform: translate(-50%, -50%) !important;
        z-index: 10050 !important;
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 6px !important;
        pointer-events: none !important;
        overflow: visible !important;
    }

      .upload-deck-thumb-wrap {
        position: relative !important;
        flex: 0 0 34px !important;
        width: 34px !important;
        height: 42px !important;
        margin: 0 !important;
        padding: 0 !important;
        overflow: visible !important;
        z-index: 10001 !important;
        pointer-events: auto !important;
    }

      .upload-deck-thumb {
        width: 34px !important;
        height: 42px !important;
        margin: 0 !important;
        padding: 0 !important;
        object-fit: cover !important;
        display: block !important;
        border-radius: 8px !important;
        border: 1px solid rgba(43,33,48,0.12) !important;
        background: #FBF3E3 !important;
        cursor: zoom-in !important;
        transition: transform .16s ease, box-shadow .16s ease !important;
        transform-origin: center center !important;
        position: relative !important;
        z-index: 10002 !important;
        pointer-events: auto !important;
    }

    .upload-deck-thumb-wrap:hover {
        z-index: 1000 !important;
    }

    .upload-deck-thumb-wrap:hover .upload-deck-thumb,
    .upload-deck-thumb.upload-thumb-hover {
        transform: scale(3.8) !important;
        z-index: 100000 !important;
        border-color: #F02AA0 !important;
        box-shadow: 0 18px 38px rgba(0,0,0,.72) !important;
    }

    .upload-deck-thumb.empty {
        opacity: .13 !important;
        cursor: default !important;
    }

    /* BUG FIX: the thumbnails need pointer-events:auto for their own
       hover-to-zoom effect above, but that same setting was exactly
       what made a drop landing ON an existing photo fail silently —
       at z-index 10001/10002 (needed so hover-zoom draws above
       everything else) they sat ABOVE the invisible file input
       (z-index 999) that actually accepts the drop, so the browser
       delivered the drop to the thumbnail instead, which does nothing
       with it. The script below toggles this class on <body> for the
       duration of an actual drag (dragenter -> drop/dragleave-off-
       window), so thumbnails briefly stop intercepting pointer events
       and the drop reaches the file input underneath — hover-zoom is
       completely unaffected outside of an active drag. */
    body.upload-drag-in-progress .upload-deck-thumb-wrap,
    body.upload-drag-in-progress .upload-deck-thumb {
        pointer-events: none !important;
    }

    /* Same problem, same fix: the arrow (next-page) and vintage-
       checkbox host divs are ALSO pinned above the file input's
       stacking context (z-index 60) so a drop landing on either one
       would silently fail to register too — disable them for the
       same drag window. */
    body.upload-drag-in-progress div[class*="st-key-upload_arrow_host_"],
    body.upload-drag-in-progress div[class*="st-key-upload_vintage_host_"] {
        pointer-events: none !important;
    }

    /* =========================================================
       ARROW — ONLY FOR >4 PHOTOS
       ========================================================= */

    /* Streamlit gives keyed containers a st-key-* class. */
    div[class*="st-key-upload_arrow_host_"] {
        position: absolute !important;
        top: 42px !important;
        right: 5px !important;
        z-index: 60 !important;
        width: 18px !important;
        height: 40px !important;
        margin: 0 !important;
        padding: 0 !important;
        pointer-events: auto !important;
    }

    div[class*="st-key-upload_arrow_host_"]
    div[data-testid="stButton"] {
        width: 18px !important;
        height: 40px !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    div[class*="st-key-upload_arrow_host_"]
    div[data-testid="stButton"] button {
        width: 18px !important;
        min-width: 18px !important;
        height: 40px !important;
        min-height: 40px !important;
        padding: 0 !important;
        margin: 0 !important;
        border: 0 !important;
        background: transparent !important;
        color: rgba(43,33,48,0.55) !important;
        font-size: 18px !important;
        line-height: 1 !important;
        box-shadow: none !important;
    }

    div[class*="st-key-upload_arrow_host_"]
    div[data-testid="stButton"] button:hover {
        color: #F02AA0 !important;
        background: transparent !important;
    }

    .upload-deck-page-indicator {
        display: none !important;
    }

    /* =========================================================
       VINTAGE CHECKBOX — same "own keyed host" trick as the arrow
       above, for the same reason (the invisible full-card file input
       otherwise intercepts the click).
       ========================================================= */

    div[class*="st-key-upload_vintage_host_"] {
        position: absolute !important;
        left: 6px !important;
        right: 6px !important;
        bottom: 3px !important;
        top: auto !important;
        height: 20px !important;
        z-index: 60 !important;
        pointer-events: auto !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    div[class*="st-key-upload_vintage_host_"] div[data-testid="stCheckbox"] {
        margin: 0 !important;
        min-height: 0 !important;
    }

    div[class*="st-key-upload_vintage_host_"] label {
        gap: 4px !important;
        min-height: 0 !important;
    }

    div[class*="st-key-upload_vintage_host_"] label p {
        font-size: 10px !important;
        font-weight: 700 !important;
        color: rgba(43,33,48,0.6) !important;
        white-space: nowrap !important;
    }

    div[class*="st-key-upload_vintage_host_"] label span[data-baseweb="checkbox"] {
        transform: scale(0.72) !important;
    }

    /* =========================================================
       ROW SPACING
       ========================================================= */

    div[data-testid="stHorizontalBlock"] {
        gap: 8px !important;
    }

    /* Keep the create/confirm button outside the cards. */
    .upload-deck-confirm {
        margin-top: 8px !important;
        margin-bottom: 8px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Browser drag tracking: highlight the EXACT card underneath the pointer.
# Uses st.html(unsafe_allow_javascript=True) rather than
# st.markdown(unsafe_allow_html=True) — markdown's HTML rendering
# strips <script> tags entirely (only raw HTML passes through), so
# this script was never actually running; st.html with that flag
# executes directly in the main document (not iframed), which is
# what a script that needs to add classes to document.body requires.
st.html(
    """
    <script>
    (() => {
        if (window.__uploadDeckDragV7) return;
        window.__uploadDeckDragV7 = true;

        function cards() {
            return Array.from(
                document.querySelectorAll('.upload-deck-card-shell')
            )
            .map(shell => shell.parentElement)
            .filter(Boolean);
        }

        function cardAt(x, y) {
            for (const card of cards()) {
                const r = card.getBoundingClientRect();
                if (
                    x >= r.left &&
                    x <= r.right &&
                    y >= r.top &&
                    y <= r.bottom
                ) {
                    return card;
                }
            }
            return null;
        }

        function clear() {
            cards().forEach(card =>
                card.classList.remove('upload-card-drag-active')
            );
        }

        document.addEventListener('dragover', event => {
            event.preventDefault();

            document.body.classList.add('upload-drag-in-progress');

            const target = cardAt(event.clientX, event.clientY);

            cards().forEach(card => {
                card.classList.toggle(
                    'upload-card-drag-active',
                    card === target
                );
            });
        }, true);

        document.addEventListener('dragenter', event => {
            document.body.classList.add('upload-drag-in-progress');

            const target = cardAt(event.clientX, event.clientY);

            cards().forEach(card => {
                card.classList.toggle(
                    'upload-card-drag-active',
                    card === target
                );
            });
        }, true);

        document.addEventListener('drop', () => {
            document.body.classList.remove('upload-drag-in-progress');
            setTimeout(clear, 150);
        }, true);

        document.addEventListener('dragleave', event => {
            if (
                event.clientX <= 0 ||
                event.clientY <= 0 ||
                event.clientX >= window.innerWidth ||
                event.clientY >= window.innerHeight
            ) {
                document.body.classList.remove('upload-drag-in-progress');
                clear();
            }
        }, true);
    })();
    </script>
    """,
    unsafe_allow_javascript=True,
)

# ------------------------------------------------------------
# MANUAL UPLOAD STATE — MUST EXIST BEFORE THE UPLOAD FRAGMENT
# ------------------------------------------------------------
if "manual_upload_slots" not in st.session_state:
    st.session_state["manual_upload_slots"] = [
        {
            "slot": index,
            "paths": [],
            "signature": [],
            "vintage": False,
        }
        for index in range(10)
    ]

if "manual_groups" not in st.session_state:
    st.session_state["manual_groups"] = []

if "manual_next_item_number" not in st.session_state:
    st.session_state["manual_next_item_number"] = 1

if "manual_upload_counter" not in st.session_state:
    st.session_state["manual_upload_counter"] = 0

# Separate from manual_upload_counter (which increments per SAVED FILE,
# for unique filenames only). This one increments once per "Confirm
# Items & Add More" click, and is what actually forces the upload-deck
# widgets to reset for a fresh batch of empty cards — using the
# per-file counter for that too meant every single photo drop reset
# every OTHER card's uploader as an unwanted side effect.
if "manual_upload_batch" not in st.session_state:
    st.session_state["manual_upload_batch"] = 0

@st.fragment
def render_upload_deck():
    _brand_step_header(1, "Upload Your Items")
    st.markdown(
        '<div style="color:#9ca3af;margin-bottom:10px;">'
        'Drag each garment directly onto its card. '
        '<b>Each card is the entire drop zone.</b> '
        'Hover a photo to inspect it larger. '
        'Use the arrow only when a garment has more than four photos.'
        '</div>',
        unsafe_allow_html=True,
    )

    slots = st.session_state["manual_upload_slots"]
    batch_start = int(
        st.session_state.get(
            "manual_next_item_number",
            1,
        )
        or 1
    )

    for row_start in (0, 5):
        row_cols = st.columns(5, gap="small")

        for col_offset in range(5):
            slot_index = row_start + col_offset
            slot = slots[slot_index]
            item_number = batch_start + slot_index

            saved_paths = list(slot.get("paths", []))

            page_key = f"manual_slot_photo_page_{slot_index}"
            page = int(
                st.session_state.get(page_key, 0) or 0
            )

            total_pages = max(
                1,
                (len(saved_paths) + 3) // 4,
            )

            page = min(page, total_pages - 1)
            st.session_state[page_key] = page

            visible_paths = saved_paths[
                page * 4:
                page * 4 + 4
            ]

            photo_count = len(saved_paths)

            with row_cols[col_offset]:

                st.markdown(
                    f"""
                    <div class="upload-deck-card-shell">
                        <div class="upload-deck-card-title">
                            <span class="item-number">
                                ITEM {item_number}
                            </span>
                            <span class="item-count">
                                {photo_count} photo
                                {"s" if photo_count != 1 else ""}
                            </span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Key is stable per slot for a given batch — it must NOT
                # include the per-FILE upload counter. Baking that counter
                # in meant every new file dropped anywhere on the page —
                # on ANY card — changed every OTHER card's widget key too,
                # which Streamlit treats as a brand new widget instance
                # with no memory of its previous file selection. The sync
                # below then overwrote slot["paths"] with only what that
                # reset widget currently reported, silently losing
                # whatever was uploaded before the reset. manual_upload_batch
                # only changes once per "Confirm Items & Add More" click,
                # which is the one time we DO want a genuine widget reset.
                uploaded_files = st.file_uploader(
                    f"Item {item_number} photos",
                    type=["jpg", "jpeg", "png", "webp"],
                    accept_multiple_files=True,
                    key=(
                        f"manual_upload_deck_v5_"
                        f"{st.session_state.get('manual_upload_batch', 0)}_"
                        f"{slot_index + 1}"
                    ),
                    label_visibility="collapsed",
                    help="Drag this garment's photos anywhere onto this card.",
                )

                if uploaded_files is not None:
                    manual_slot_sync_files(
                        slot_index,
                        uploaded_files,
                    )

                slot = st.session_state["manual_upload_slots"][slot_index]
                saved_paths = list(slot.get("paths", []))

                total_pages = max(
                    1,
                    (len(saved_paths) + 3) // 4,
                )
                page = min(
                    int(st.session_state.get(page_key, 0) or 0),
                    total_pages - 1,
                )
                st.session_state[page_key] = page

                visible_paths = saved_paths[
                    page * 4:
                    page * 4 + 4
                ]

                thumb_html = []

                for path in visible_paths:
                    try:
                        data_url = make_item_thumbnail_data_url(path)
                        thumb_html.append(
                            '<div class="upload-deck-thumb-wrap">'
                            f'<img class="upload-deck-thumb" src="{data_url}" alt="">'
                            '</div>'
                        )
                    except Exception:
                        pass

                while len(thumb_html) < 4:
                    thumb_html.append(
                        '<div class="upload-deck-thumb-wrap">'
                        '<div class="upload-deck-thumb empty"></div>'
                        '</div>'
                    )

                st.markdown(
                    '<div class="upload-deck-photo-strip">'
                    + "".join(thumb_html)
                    + "</div>",
                    unsafe_allow_html=True,
                )

                # Its own keyed host, same reason as the arrow button below:
                # the card's invisible full-card file-input overlay
                # (z-index:999, needed so the whole card is a drop zone)
                # sits on top of anything in normal document flow. A
                # dedicated st-key-* container gets its own scoped
                # position:absolute + z-index rule (see
                # upload_vintage_host_ in the CSS above), which — because
                # that z-index is compared against the file-uploader's
                # OUTER wrapper (z-index:5), not the input's inner
                # z-index:999, which is scoped to its own stacking context
                # — reliably wins and stays clickable.
                with st.container(key=f"upload_vintage_host_{slot_index}"):
                    slot["vintage"] = st.checkbox(
                        "Vintage",
                        value=bool(slot.get("vintage", False)),
                        key=f"manual_slot_vintage_{slot_index}",
                        help=(
                            "Check if you know this item is vintage. The "
                            "AI still runs its own vintage check too — "
                            "this just guarantees it gets marked even if "
                            "that check misses it."
                        ),
                    )

                # The arrow lives in its own host so it can NEVER style or
                # collide with the big confirm button below the deck.
                if total_pages > 1:
                    with st.container(
                        key=f"upload_arrow_host_{slot_index}"
                    ):
                        if st.button(
                            "›",
                            key=f"slot_next_v5_{slot_index}",
                            help="Next four photos",
                        ):
                            st.session_state[page_key] = (
                                page + 1
                            ) % total_pages
                            st.rerun()

    # ------------------------------------------------------------
    # ONE CONFIRM BUTTON — FILLED CARDS ONLY
    # ------------------------------------------------------------
    filled_slots = [
        index
        for index, slot in enumerate(
            st.session_state["manual_upload_slots"]
        )
        if slot.get("paths")
    ]

    if filled_slots:
        st.markdown(
            '<div class="upload-deck-confirm">',
            unsafe_allow_html=True,
        )

        if st.button(
            f"Confirm {len(filled_slots)} Items & Add More",
            type="primary",
            width="stretch",
            key="confirm_upload_deck_v5",
        ):
            groups_now = list(
                st.session_state.get("manual_groups", [])
            )

            next_number = int(
                st.session_state.get(
                    "manual_next_item_number",
                    1,
                )
                or 1
            )

            for offset, slot_index in enumerate(filled_slots):
                confirming_slot = (
                    st.session_state["manual_upload_slots"]
                    [slot_index]
                )
                paths = list(confirming_slot.get("paths", []))

                if not paths:
                    continue

                item_number = next_number + offset

                groups_now.append(
                    manual_make_group(
                        paths,
                        item_number,
                        marked_vintage=confirming_slot.get(
                            "vintage", False
                        ),
                    )
                )

            groups_now.sort(
                key=lambda group: int(
                    group.get("item_number", 999)
                )
            )

            st.session_state["manual_groups"] = groups_now

            # Start a completely fresh 10-card batch.
            st.session_state["manual_next_item_number"] = (
                next_number + len(filled_slots)
            )
            st.session_state["manual_upload_counter"] = (
                int(
                    st.session_state.get(
                        "manual_upload_counter",
                        0,
                    )
                    or 0
                )
                + 1
            )

            # This is what actually resets the upload-deck widgets for
            # the fresh batch of empty cards below (see the file_uploader
            # key comment above) — manual_upload_counter alone no longer
            # does this, since it changes on every single file drop.
            st.session_state["manual_upload_batch"] = (
                int(
                    st.session_state.get(
                        "manual_upload_batch",
                        0,
                    )
                    or 0
                )
                + 1
            )

            st.session_state["manual_upload_slots"] = [
                {
                    "slot": index,
                    "paths": [],
                    "signature": [],
                    "vintage": False,
                }
                for index in range(10)
            ]

            for index in range(10):
                st.session_state.pop(
                    f"manual_slot_photo_page_{index}",
                    None,
                )

            manual_sync_bulk_paths()
            clear_qa_state()

            if groups_now:
                st.session_state["selected_item_index"] = 0

            st.rerun()

        st.caption(
            f"{len(filled_slots)} garment"
            + ("s" if len(filled_slots) != 1 else "")
            + " ready. Empty cards are fine — confirm only the cards you filled."
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )


# ============================================================
# BULK PHOTO AUTO-SORT — a second, additive upload flow
# Completely separate from the manual upload deck above: its own
# upload dir, its own session-state keys, its own bulk_sort.py
# pipeline. The ONLY integration point with the rest of the app is
# "Add These Items", which converts confirmed groups into normal
# manual_groups entries and calls the existing manual_sync_bulk_paths()
# — after that, an item built here is indistinguishable from one built
# by hand to every other part of the app.
# ============================================================

BULK_SORT_UPLOAD_DIR = UPLOAD_DIR / "bulk_sort"
BULK_SORT_UPLOAD_DIR.mkdir(exist_ok=True)

BULK_SORT_STATUS_CSS = """
<style>
.bulk-sort-dot {
    display:inline-block; width:10px; height:10px; border-radius:50%;
    margin-right:6px; vertical-align:middle;
}
.bulk-sort-dot[data-status="green"] { background:#39d98a; box-shadow:0 0 0 3px rgba(57,217,138,.15); }
.bulk-sort-dot[data-status="yellow"] { background:#f5c451; box-shadow:0 0 0 3px rgba(245,196,81,.15); }
.bulk-sort-dot[data-status="red"] { background:#ef4444; box-shadow:0 0 0 3px rgba(239,68,68,.15); }
.bulk-sort-group-title { font-weight:800; font-size:14px; }
.bulk-sort-group-meta { font-size:11px; opacity:.6; margin-top:2px; }
.bulk-sort-col-header {
    font-weight:900; font-size:13px; letter-spacing:.03em;
    padding:8px 12px; border-radius:8px; margin-bottom:10px;
    text-align:center;
}
.bulk-sort-col-header[data-status="green"] { background:rgba(57,217,138,.12); color:#1f9d5f; }
.bulk-sort-col-header[data-status="yellow"] { background:rgba(245,196,81,.15); color:#a67a1a; }
.bulk-sort-col-header[data-status="red"] { background:rgba(239,68,68,.12); color:#c0392b; }
.bulk-sort-match-hint {
    font-size:11px; font-weight:700; color:#a67a1a;
    background:rgba(245,196,81,.12); border:1px dashed rgba(245,196,81,.4);
    border-radius:6px; padding:4px 8px; margin:6px 0;
}
</style>
"""

# Fixed-size thumbnails for every bulk-sort photo grid, regardless of
# how many photos are in that particular group — without this,
# st.image(width="stretch") inside st.columns(N) scales inversely with
# N (1 photo fills the whole row = huge; 6 photos = tiny slivers).
# Forces the full st.image DOM chain (stFullScreenFrame > stImage >
# stImageContainer > img) to a fixed square, same technique proven
# earlier for the review-gallery thumbnails.
BULK_SORT_PHOTO_CSS = """
<style>
div[class*="st-key-bulk_sort_photo_grid_"] div[data-testid="stFullScreenFrame"],
div[class*="st-key-bulk_sort_photo_grid_"] div[data-testid="stFullScreenFrame"] > div,
div[class*="st-key-bulk_sort_photo_grid_"] div[data-testid="stImage"],
div[class*="st-key-bulk_sort_photo_grid_"] div[data-testid="stImageContainer"] {
    width: 130px !important;
    height: 130px !important;
    max-width: 130px !important;
    max-height: 130px !important;
}
div[class*="st-key-bulk_sort_photo_grid_"] div[data-testid="stImageContainer"] img {
    width: 130px !important;
    height: 130px !important;
    max-height: none !important;
    object-fit: cover !important;
    border-radius: 8px !important;
    display: block !important;
    background: #FBF3E3 !important;
}
div[class*="st-key-bulk_sort_photo_grid_"] div[data-testid="stColumn"] {
    flex: 0 0 auto !important;
    width: 140px !important;
    min-width: 140px !important;
}
</style>
"""

# The upload-deck's own CSS (search "upload-deck-card-shell" above)
# scopes itself via `div[data-testid="stVerticalBlock"]:has(.upload-
# deck-card-shell) div[data-testid="stFileUploader"] ...`, intending
# to only affect uploaders inside a deck card. But `:has()` matches
# ANY ancestor stVerticalBlock containing a card-shell ANYWHERE among
# its descendants — and Streamlit wraps large sections of the page in
# shared outer stVerticalBlocks, so that selector's descendant
# combinator (not `>`) reaches every stFileUploader in the whole
# shared ancestor, not just the ones actually inside a card. Confirmed
# live via Playwright: this bulk-sort uploader's <input type="file">
# was inheriting the deck's position:absolute/z-index:999/opacity:0
# treatment meant only for the 10 upload-deck cards, making it
# invisible and intercepting clicks on the "Save these photos" button
# below it. Rather than risk touching the deck's own (working, tested)
# CSS, this resets the same properties back to normal specifically for
# THIS uploader, scoped via its own container key at higher
# specificity so it wins regardless of source order.

def _bulk_sort_uploader_reset_selector(*parts):
    """The deck's own uploader CSS (":has()"-scoped, see the comment
    above) ties this scope's specificity exactly even with a doubled
    class-attribute selector (confirmed live — both landed at the same
    (0,4,3) count), so a plain higher-attribute-count trick isn't
    reliable here. ":not(#_bulk-sort-nonexistent)" is a no-op selector
    (that id never exists) that bumps specificity to ID-level, which
    unambiguously beats any class/attribute-only selector regardless
    of source order — sidesteps the whole fragile tie-counting problem
    instead of chasing it further.
    """
    scope = 'div[class*="st-key-bulk_sort_uploader_scope"]:not(#_bulk-sort-nonexistent)'
    return ",\n".join(f"{scope} {part}" for part in parts)


BULK_SORT_UPLOADER_RESET_CSS = f"""
<style>
{_bulk_sort_uploader_reset_selector(
    'div[data-testid="stFileUploader"]',
    'div[data-testid="stFileUploader"] section',
    'div[data-testid="stFileUploaderDropzone"]',
)} {{
    position: static !important;
    z-index: auto !important;
    opacity: 1 !important;
    visibility: visible !important;
    display: flex !important;
    height: auto !important;
    min-height: 0 !important;
    max-height: none !important;
}}
{_bulk_sort_uploader_reset_selector(
    'div[data-testid="stFileUploader"] input[type="file"]',
)} {{
    position: static !important;
    z-index: auto !important;
    opacity: 1 !important;
    width: auto !important;
    height: auto !important;
}}
{_bulk_sort_uploader_reset_selector(
    'div[data-testid="stFileUploader"] label',
    'div[data-testid="stFileUploader"] button',
    'div[data-testid="stFileUploader"] small',
    'div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"]',
    'div[data-testid="stFileUploader"] [data-testid="stFileUploaderFileData"]',
    'div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"]',
)} {{
    position: static !important;
    display: revert !important;
    opacity: 1 !important;
    visibility: visible !important;
    color: revert !important;
    font-size: revert !important;
}}
</style>
"""


def _bulk_sort_photo_key(path_str):
    return hashlib.md5(path_str.encode("utf-8")).hexdigest()[:12]


def _save_bulk_sort_uploads(uploaded_files):
    """Save raw uploaded bytes AS-IS — no EXIF-stripping rewrite.

    Unlike normalize_uploaded_image_to_path() (used by the manual
    upload deck), which intentionally re-saves a rotated photo with
    exif=b"" once it bakes orientation into pixels, bulk-sort's Step 1
    time-clustering depends on DateTimeOriginal surviving the save.
    Orientation is instead corrected on-the-fly at READ time —
    image_to_data_url() already applies ImageOps.exif_transpose()
    before sending a photo to the vision model — so there's no need to
    bake it into the saved file here, and no reason to risk losing the
    timestamp doing so.
    """
    saved_paths = []

    for uploaded_file in uploaded_files:
        safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", uploaded_file.name)
        counter = st.session_state.get("bulk_sort_upload_counter", 0)
        st.session_state["bulk_sort_upload_counter"] = counter + 1

        file_path = BULK_SORT_UPLOAD_DIR / f"batch_{counter}_{safe_name}"
        uploaded_file.seek(0)
        file_path.write_bytes(uploaded_file.getvalue())
        saved_paths.append(str(file_path))

    return saved_paths


def _bulk_sort_group_label(group):
    return f"Item {group['group_id']} ({len(group['photo_paths'])} photos)"


def _render_bulk_sort_photo_grid(state, photo_paths, grid_key, fit_scores=None, per_row=6):
    """Renders a row of photo thumbnails with a fit-confidence
    checkbox under each (the multi-select mechanism — shift/ctrl-click
    isn't available with plain Streamlit widgets, so checkboxes are
    the equivalent here). Returns the list of photo paths the user
    checked.

    Every thumbnail is a FIXED pixel size via BULK_SORT_PHOTO_CSS,
    regardless of how many photos are in this particular grid —
    st.image(width="stretch") inside st.columns(N) scales inversely
    with N (1 photo = full-width column = huge image, 6 photos = tiny
    slivers), which read as "inconsistent sizing" rather than a deck
    of same-size items. grid_key must be unique per call (e.g. the
    group id, or "attention") so each grid gets its own scoped
    container.
    """
    fit_scores = fit_scores or {}
    selected = []
    per_row = min(per_row, max(1, len(photo_paths)))

    with st.container(key=f"bulk_sort_photo_grid_{grid_key}"):
        cols = st.columns(per_row)

        for index, photo_path in enumerate(photo_paths):
            with cols[index % per_row]:
                thumb = make_item_thumbnail_data_url(Path(photo_path))
                if thumb:
                    st.image(thumb, width="stretch")

                fit = fit_scores.get(photo_path)
                label = f"fit {fit:.2f}" if fit is not None else "select"
                photo_key_id = _bulk_sort_photo_key(photo_path)

                if st.checkbox(label, key=f"bulk_sort_sel_{photo_key_id}"):
                    selected.append(photo_path)

    return selected


def _bulk_sort_move_to_dropdown(state, photo_keys, group_labels, key_suffix):
    """Shared 'Move selected photo(s) to <dropdown> [Go]' control used
    by every card (group cards and the stray/Needs Attention card
    alike) - factored out so the three-column layout doesn't repeat
    this block per card.
    """
    move_cols = st.columns([3, 1])

    options = ["+ Create new item"] + list(group_labels.keys())

    with move_cols[0]:
        choice = st.selectbox(
            "Move selected photo(s) to",
            options,
            format_func=lambda gid: (
                "+ Create new item" if gid == "+ Create new item"
                else group_labels[gid]
            ),
            key=f"bulk_sort_move_{key_suffix}",
            label_visibility="collapsed",
        )

    with move_cols[1]:
        if st.button("Go", key=f"bulk_sort_move_go_{key_suffix}"):
            if not photo_keys:
                st.warning("Select at least one photo first.")
            elif choice == "+ Create new item":
                state = bulk_sort.create_group_from_photos(state, photo_keys)
                st.session_state["bulk_sort_state"] = state
                st.rerun()
            else:
                state = bulk_sort.move_photos(state, photo_keys, choice)
                st.session_state["bulk_sort_state"] = state
                st.rerun()


def _render_bulk_sort_group_card(state, group, group_labels):
    """One group card - status dot, meta, an optional possible-match
    suggestion with a one-click merge button (for yellow groups), the
    reasoning expander, the photo grid, and the move-to control.
    """
    with st.container(border=True):
        status = group["status"]

        st.markdown(
            f'<span class="bulk-sort-dot" data-status="{status}"></span>'
            f'<span class="bulk-sort-group-title">'
            f'{_bulk_sort_group_label(group)} · {status.upper()}'
            f'</span>',
            unsafe_allow_html=True,
        )

        meta_bits = [
            f"confidence {group['group_confidence']:.2f}",
            group.get("dominant_garment_type") or "unknown type",
            "tag found" if group["has_tag_photo"] else "no tag photo",
        ]
        if group["has_rescued_photo"]:
            meta_bits.append("includes a rescued photo")

        st.markdown(
            f'<div class="bulk-sort-group-meta">{" · ".join(meta_bits)}</div>',
            unsafe_allow_html=True,
        )

        match_id = group.get("possible_match_group_id")
        if match_id is not None and match_id in group_labels:
            match_score = group.get("possible_match_score") or 0.0
            st.markdown(
                f'<div class="bulk-sort-match-hint">↔ Possible match: '
                f'{group_labels[match_id]} (score {match_score:.2f})</div>',
                unsafe_allow_html=True,
            )
            if st.button(
                f"🔗 Merge with {group_labels[match_id]}",
                key=f"bulk_sort_quickmerge_{group['group_id']}",
            ):
                state = bulk_sort.merge_groups(
                    state, match_id, group["group_id"]
                )
                st.session_state["bulk_sort_state"] = state
                st.rerun()

        with st.expander("Why this rating?"):
            relevant = [
                entry for entry in state["reasoning_log"]
                if entry.get("photo") in group["photo_paths"]
            ]
            if relevant:
                for entry in relevant[-10:]:
                    st.json(entry, expanded=False)
            else:
                st.caption(
                    "Placed directly from time + visual clustering — "
                    "no rescue decision was needed for this group's photos."
                )

        selected = _render_bulk_sort_photo_grid(
            state, group["photo_paths"], group["group_id"],
            fit_scores=group["image_fit_confidence"], per_row=3,
        )

        other_labels = {
            gid: label for gid, label in group_labels.items()
            if gid != group["group_id"]
        }
        _bulk_sort_move_to_dropdown(
            state, selected, other_labels, key_suffix=str(group["group_id"])
        )


def _render_bulk_sort_stray_card(state, photo_path, group_labels):
    """One Needs Attention photo - its own card matching the group
    card's visual language (red dot), with the same possible-match
    suggestion + one-click merge mechanism.
    """
    match = state.get("stray_matches", {}).get(photo_path, {})
    match_id = match.get("group_id")
    match_score = match.get("score") or 0.0

    with st.container(border=True):
        st.markdown(
            '<span class="bulk-sort-dot" data-status="red"></span>'
            '<span class="bulk-sort-group-title">Needs Attention</span>',
            unsafe_allow_html=True,
        )

        thumb = make_item_thumbnail_data_url(Path(photo_path))
        if thumb:
            st.image(thumb, width="stretch")

        if match_id is not None and match_id in group_labels:
            st.markdown(
                f'<div class="bulk-sort-match-hint">↔ Possible match: '
                f'{group_labels[match_id]} (score {match_score:.2f})</div>',
                unsafe_allow_html=True,
            )
            if st.button(
                f"🔗 Merge with {group_labels[match_id]}",
                key=f"bulk_sort_stray_quickmerge_{_bulk_sort_photo_key(photo_path)}",
            ):
                state = bulk_sort.move_photos(state, [photo_path], match_id)
                st.session_state["bulk_sort_state"] = state
                st.rerun()
        else:
            st.caption("No confident match found for this photo.")

        _bulk_sort_move_to_dropdown(
            state, [photo_path], group_labels,
            key_suffix=f"stray_{_bulk_sort_photo_key(photo_path)}",
        )


def render_bulk_sort_section():
    st.divider()
    st.markdown(
        '<h3 style="margin-bottom:4px;">🗂️ Bulk Photo Auto-Sort</h3>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Dump a big folder of mixed garment photos here and let AI figure "
        "out which photos belong to which item — a second, separate flow "
        "from the upload deck below. Confirmed items get added to Your "
        "Items once you're happy with how they're grouped."
    )

    st.markdown(BULK_SORT_STATUS_CSS, unsafe_allow_html=True)
    st.markdown(BULK_SORT_PHOTO_CSS, unsafe_allow_html=True)

    if "bulk_sort_state" not in st.session_state:
        st.session_state["bulk_sort_state"] = bulk_sort.load_bulk_sort_state()

    if "bulk_sort_paths" not in st.session_state:
        st.session_state["bulk_sort_paths"] = sorted(
            str(p) for p in BULK_SORT_UPLOAD_DIR.glob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )

    state = st.session_state["bulk_sort_state"]

    with st.expander(
        "Upload a batch",
        expanded=not state["groups"] and not state["needs_attention"],
    ):
        st.markdown(BULK_SORT_UPLOADER_RESET_CSS, unsafe_allow_html=True)

        with st.container(key="bulk_sort_uploader_scope"):
            uploaded_files = st.file_uploader(
                "Drop 20-100+ garment photos here",
                type=["jpg", "jpeg", "png", "webp"],
                accept_multiple_files=True,
                key="bulk_sort_uploader",
            )

        if uploaded_files and st.button(
            "Save these photos", key="bulk_sort_save_uploads"
        ):
            new_paths = _save_bulk_sort_uploads(uploaded_files)
            st.session_state["bulk_sort_paths"].extend(new_paths)
            st.success(f"Saved {len(new_paths)} photo(s). Now click Sort below.")
            st.rerun()

        st.caption(
            f"{len(st.session_state['bulk_sort_paths'])} photo(s) staged for sorting."
        )

        if st.session_state["bulk_sort_paths"] and st.button(
            "Sort My Photos", type="primary", key="bulk_sort_run"
        ):
            progress_bar = st.progress(0, text="Analyzing photos...")
            status_text = st.empty()

            def _bulk_sort_progress(done, total):
                if total <= 0:
                    return
                percent = done / total
                progress_bar.progress(
                    percent, text=f"Analyzing photos — {done}/{total}"
                )
                status_text.write(
                    f"Analyzed {done} of {total} photo(s) ({percent:.0%})"
                )

            status_text.write("Starting analysis...")

            try:
                state = bulk_sort.run_grouping_pipeline(
                    st.session_state["bulk_sort_paths"],
                    state,
                    on_progress=_bulk_sort_progress,
                )
                st.session_state["bulk_sort_state"] = state
                progress_bar.progress(1.0, text="Grouping complete")
                status_text.success(
                    f"Sorted into {len(state['groups'])} item(s), "
                    f"{len(state['needs_attention'])} photo(s) need attention."
                )
            except Exception as exc:
                status_text.error(f"Bulk-sort grouping failed: {exc}")

            st.rerun()

    if not state["groups"] and not state["needs_attention"]:
        return

    # ------------------------------------------------------------
    # TOP-LEVEL CONTROLS: undo, merge, clear batch
    # ------------------------------------------------------------
    control_cols = st.columns([1, 2, 2, 1])

    with control_cols[0]:
        if st.button(
            "↩ Undo",
            key="bulk_sort_undo",
            disabled=not state["undo_stack"],
        ):
            state = bulk_sort.undo(state)
            st.session_state["bulk_sort_state"] = state
            st.rerun()

    group_labels = {
        group["group_id"]: _bulk_sort_group_label(group)
        for group in state["groups"]
    }

    with control_cols[1]:
        merge_choices = st.multiselect(
            "Merge groups (pick 2+)",
            options=list(group_labels.keys()),
            format_func=lambda gid: group_labels[gid],
            key="bulk_sort_merge_choices",
        )

    with control_cols[2]:
        if st.button(
            "Merge Selected",
            key="bulk_sort_merge_go",
            disabled=len(merge_choices) < 2,
        ):
            target_id = merge_choices[0]
            for other_id in merge_choices[1:]:
                state = bulk_sort.merge_groups(state, target_id, other_id)
            st.session_state["bulk_sort_state"] = state
            st.rerun()

    with control_cols[3]:
        # This batch is saved to disk specifically so it survives a
        # page refresh (see bulk_sort.save_bulk_sort_state) — which
        # means, with no way to dismiss it, it also keeps reappearing
        # on every future launch even once you're done with it. A
        # two-step confirm since it deletes the staged photo copies in
        # uploads/bulk_sort/ (your original files elsewhere are
        # untouched — only the copies made for this review).
        if st.session_state.get("bulk_sort_confirm_clear"):
            st.warning(
                f"Delete all {len(state['groups']) + len(state['needs_attention'])} "
                f"staged photo(s) and start over?"
            )
            confirm_cols = st.columns(2)
            with confirm_cols[0]:
                if st.button("Yes, clear it", key="bulk_sort_clear_confirmed"):
                    for path in st.session_state.get("bulk_sort_paths", []):
                        try:
                            Path(path).unlink(missing_ok=True)
                        except Exception:
                            pass
                    st.session_state["bulk_sort_paths"] = []
                    fresh_state = bulk_sort._new_state()
                    bulk_sort.save_bulk_sort_state(fresh_state)
                    st.session_state["bulk_sort_state"] = fresh_state
                    st.session_state["bulk_sort_confirm_clear"] = False
                    st.rerun()
            with confirm_cols[1]:
                if st.button("Cancel", key="bulk_sort_clear_cancel"):
                    st.session_state["bulk_sort_confirm_clear"] = False
                    st.rerun()
        else:
            if st.button("🗑️ Clear batch", key="bulk_sort_clear_start"):
                st.session_state["bulk_sort_confirm_clear"] = True
                st.rerun()

    # ------------------------------------------------------------
    # THREE-COLUMN REVIEW
    # Green (confident) | Yellow (needs a look) | Red (needs
    # attention) — left to right, per the requested layout. Yellow
    # groups and red stray photos are sorted by where their suggested
    # match sits in the green column, so scanning across a row gives a
    # rough "this stray lines up with that candidate" visual pairing
    # without needing pixel-precise cross-column alignment. Each
    # yellow/red card also carries an explicit "Possible match" label
    # and a one-click merge button, so resolving the common case never
    # requires opening a dropdown.
    # ------------------------------------------------------------
    green_groups = [g for g in state["groups"] if g["status"] == "green"]
    yellow_groups = [g for g in state["groups"] if g["status"] != "green"]

    green_order = {gid: index for index, gid in enumerate(
        g["group_id"] for g in green_groups
    )}
    fallback_order = len(green_groups) + 1

    yellow_groups.sort(
        key=lambda g: green_order.get(
            g.get("possible_match_group_id"), fallback_order
        )
    )

    stray_photos = list(state["needs_attention"])
    stray_photos.sort(
        key=lambda p: green_order.get(
            state.get("stray_matches", {}).get(p, {}).get("group_id"),
            fallback_order,
        )
    )

    col_green, col_yellow, col_red = st.columns(3)

    with col_green:
        st.markdown(
            '<div class="bulk-sort-col-header" data-status="green">'
            f'🟢 CONFIDENT ({len(green_groups)})</div>',
            unsafe_allow_html=True,
        )
        if not green_groups:
            st.caption("No fully-confident groups yet.")
        for group in green_groups:
            _render_bulk_sort_group_card(state, group, group_labels)

    with col_yellow:
        st.markdown(
            '<div class="bulk-sort-col-header" data-status="yellow">'
            f'🟡 NEEDS A LOOK ({len(yellow_groups)})</div>',
            unsafe_allow_html=True,
        )
        if not yellow_groups:
            st.caption("Nothing uncertain — everything's confident or unassigned.")
        for group in yellow_groups:
            _render_bulk_sort_group_card(state, group, group_labels)

    with col_red:
        st.markdown(
            '<div class="bulk-sort-col-header" data-status="red">'
            f'🔴 NEEDS ATTENTION ({len(stray_photos)})</div>',
            unsafe_allow_html=True,
        )
        if not stray_photos:
            st.caption("Nothing here — every photo found a home.")
        for photo_path in stray_photos:
            _render_bulk_sort_stray_card(state, photo_path, group_labels)

    # ------------------------------------------------------------
    # FINALIZE — the only integration point with the rest of the app
    # ------------------------------------------------------------
    if state["groups"]:
        st.divider()

        if st.button(
            "✓ Add These Items to Your Items",
            type="primary",
            key="bulk_sort_finalize",
        ):
            manual_groups = st.session_state.setdefault("manual_groups", [])
            next_number = st.session_state.get("manual_next_item_number", 1)

            added = 0
            for group in state["groups"]:
                manual_groups.append({
                    "item_number": next_number,
                    "paths": [Path(p) for p in group["photo_paths"]],
                    "photo_numbers": [],
                    "anchor_photo": None,
                    "label": f"Item {next_number}",
                    "title": f"Item {next_number}",
                    "reason": "Bulk auto-sort",
                })
                next_number += 1
                added += 1

            st.session_state["manual_next_item_number"] = next_number
            manual_sync_bulk_paths()

            # Finalized groups are removed from the bulk-sort review
            # state (and its cache) so they don't show up again if the
            # user sorts another batch later — needs_attention photos
            # are untouched, still waiting for manual resolution.
            state["groups"] = []
            state["undo_stack"] = []
            st.session_state["bulk_sort_state"] = state
            bulk_sort.save_bulk_sort_state(state)

            st.success(f"Added {added} item(s) to Your Items.")
            st.rerun()


# Archived for now — the manual 10-slot deck below is the active flow.
# All bulk-sort code/state is untouched; uncomment this line to bring
# the "Bulk Photo Auto-Sort" section back.
# render_bulk_sort_section()

render_upload_deck()


def rerun_fragment_safe():
    """st.rerun(scope="fragment") only works while the current run IS a
    fragment-scoped rerun. A value change from a custom bidirectional
    component (inventory_deck, item_drop_card) isn't always recognized
    as one — it can land here during a full script pass instead, where
    the scoped call raises StreamlitAPIException. Fall back to a full
    rerun rather than crash; it's strictly less optimized, never wrong.
    """
    try:
        st.rerun(scope="fragment")
    except StreamlitAPIException:
        st.rerun()


@st.fragment
def render_items_deck_and_editor():
    # CREATED ITEMS — VISUAL INVENTORY DECK
    # ------------------------------------------------------------
    groups = st.session_state.get("manual_groups", [])

    if groups:
        st.divider()
        _brand_step_header(2, "Your Items")
        st.caption(
            "Your garments are now a deck. Click Spread to fan them out, "
            "then click any item to open it. The existing photo controls "
            "remain available inside the item."
        )

        deck_items = []
        for index, group in enumerate(groups):
            paths = list(group.get("paths", []))
            thumb = make_item_thumbnail_data_url(paths[0], max_size=128) if paths else ""
            photo_thumbs = [
                make_item_thumbnail_data_url(path)
                for path in paths[:7]
            ]
            deck_items.append({
                "number": group.get("item_number", index + 1),
                "title": group.get("title", f"Item {group.get('item_number', index + 1)}"),
                "photos": len(paths),
                "thumb": thumb,
                "photo_thumbs": photo_thumbs,
                "status": "ready",
            })

        if "inventory_deck_spread" not in st.session_state:
            st.session_state["inventory_deck_spread"] = False

        deck_result = None
        if inventory_deck is not None:
            deck_result = inventory_deck(
                key="inventory_deck_main",
                data={
                    "items_json": json.dumps(deck_items, ensure_ascii=False),
                    "spread": st.session_state["inventory_deck_spread"],
                },
                on_action_change=lambda: None,
            )

        deck_action = getattr(deck_result, "action", None) if deck_result else None
        if isinstance(deck_action, dict):
            action_type = deck_action.get("type")
            if action_type == "toggle_spread":
                st.session_state["inventory_deck_spread"] = bool(
                    deck_action.get("spread", False)
                )
                rerun_fragment_safe()
            elif action_type == "open_item":
                st.session_state["inventory_deck_spread"] = True
                st.session_state["selected_item_index"] = int(
                    deck_action.get("index", 0) or 0
                )
                rerun_fragment_safe()


    # ------------------------------------------------------------
    # ACTIVE ITEM — OPENED FROM THE DECK
    # ------------------------------------------------------------
    groups = st.session_state.get("manual_groups", [])
    if groups and "selected_item_index" in st.session_state:
        selected_item_index = int(
            st.session_state.get("selected_item_index", 0) or 0
        )
        selected_item_index = max(
            0,
            min(len(groups) - 1, selected_item_index),
        )
        selected_group = groups[selected_item_index]
        selected_item_number = selected_group.get(
            "item_number",
            selected_item_index + 1,
        )

        st.markdown(
            f"### Item {selected_item_number}"
        )
        st.caption(
            "This is the opened card. Rotate photos, reorder them, "
            "or drag photos between items exactly as before."
        )

        if item_drop_card is not None:
            detail_result = item_drop_card(
                key=f"manual_open_item_{selected_item_number}",
                data={
                    "item_number": selected_item_number,
                    "photos_json": json.dumps(
                        build_item_component_photos(
                            selected_group.get("paths", [])
                        ),
                        ensure_ascii=False,
                    ),
                },
                on_action_change=lambda: None,
            )

            detail_action = getattr(
                detail_result,
                "action",
                None,
            )

            if isinstance(detail_action, dict) and detail_action.get("type"):
                normalized_detail_action = {
                    "action": detail_action.get("type"),
                    **{
                        key: value
                        for key, value in detail_action.items()
                        if key != "type"
                    },
                }

                detail_result_value = handle_item_component_action(
                    selected_group,
                    normalized_detail_action,
                )

                if detail_result_value == "DELETE_ITEM":
                    st.session_state["manual_groups"] = [
                        existing
                        for existing in st.session_state["manual_groups"]
                        if existing is not selected_group
                    ]
                    st.session_state.pop("selected_item_index", None)
                    manual_sync_bulk_paths()
                    clear_qa_state()
                    rerun_fragment_safe()

                if detail_result_value is True:
                    manual_sync_bulk_paths()
                    clear_qa_state()
                    rerun_fragment_safe()

        if st.button(
            "Close Item",
            key=f"close_open_item_{selected_item_number}",
        ):
            st.session_state.pop("selected_item_index", None)
            st.session_state["inventory_deck_spread"] = True
            rerun_fragment_safe()



render_items_deck_and_editor()

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

    path_counts = {}
    for path in all_paths:
        path_counts[path] = path_counts.get(path, 0) + 1

    duplicate_paths = sorted(
        path
        for path, count in path_counts.items()
        if count > 1
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

    _brand_step_header(3, "Generate All Depop Listings")

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

        # Created FIRST, before rotation-checking/ordering even starts —
        # those phases used to run silently before this bar existed at
        # all, so the button looked hung for however long a big batch's
        # rotation check took (dozens of sequential photo checks). Now
        # there's visible progress from the very first click.
        progress_bar = st.progress(
            0,
            text="Checking photo orientation..."
        )

        status_text = st.empty()

        # ------------------------------------------------------------
        # AUTO-STRAIGHTEN + AUTO-ORDER PHOTOS
        # Straighten sideways/upside-down photos (bad/missing EXIF)
        # and reorder each group's photos front->back->details->
        # tag-last BEFORE analyzing — a sideways tag or garment view
        # would otherwise confuse both steps. Generation-only: this
        # reorders/dedupes the photo list analyze_item() sees for this
        # run, but does not write back into the manual deck, so your
        # groups stay exactly as you arranged them.
        #
        # A photo's orientation is checked AT MOST ONCE, ever, for the
        # life of this session: st.session_state["rotation_checked_paths"]
        # remembers every path already sent through detect_photo_rotations
        # (whether it needed correcting or not), so clicking "Generate
        # All Listings" again on the same photos — retries, adding more
        # items, re-running after a partial failure — never re-spends
        # tokens re-checking a photo that was already checked.
        # ------------------------------------------------------------
        try:
            rotation_checked_paths = st.session_state.setdefault(
                "rotation_checked_paths", set()
            )
            unchecked_paths = [
                path for path in bulk_paths
                if str(Path(path).resolve()) not in rotation_checked_paths
            ]

            if unchecked_paths:
                status_text.write(
                    f"Checking orientation of {len(unchecked_paths)} photo(s)…"
                )

                def _rotation_progress(done, total):
                    percent = done / max(1, total)
                    progress_bar.progress(
                        percent,
                        text=f"Checking photo orientation — {done}/{total}",
                    )
                    status_text.write(
                        f"Checked {done} of {total} photos for orientation "
                        f"({percent:.0%})"
                    )

                rotations = detect_photo_rotations(
                    unchecked_paths,
                    on_progress=_rotation_progress,
                )

                rotate_failures = 0
                for photo_index, degrees in rotations.items():
                    if 0 <= photo_index < len(unchecked_paths):
                        try:
                            _rotate_photo_in_place(
                                unchecked_paths[photo_index],
                                degrees,
                            )
                        except Exception as rotate_error:
                            rotate_failures += 1
                            print(
                                f"Auto-straighten skipped a photo: {rotate_error}"
                            )

                if rotate_failures:
                    st.warning(
                        f"Couldn't auto-straighten {rotate_failures} "
                        "photo(s) — check them manually with the ↻ "
                        "rotate button if they look sideways."
                    )

                for path in unchecked_paths:
                    rotation_checked_paths.add(str(Path(path).resolve()))

                if rotations:
                    manual_sync_bulk_paths()
        except Exception as rotation_error:
            print(
                f"Auto-straighten pass skipped: {rotation_error}"
            )
            st.warning(
                "Photo orientation check didn't run for this batch "
                f"({rotation_error}) — check your photos manually with "
                "the ↻ rotate button if any look sideways."
            )

        progress_bar.progress(0, text="Ordering photos...")
        status_text.write("Ordering photos within each item…")

        try:
            groups = order_group_photos(
                bulk_paths,
                groups,
                [],
            )
        except Exception as order_error:
            print(
                f"Auto photo ordering skipped: {order_error}"
            )
            st.warning(
                "Automatic photo ordering (front/back/detail/tag) didn't "
                f"run for this batch ({order_error}) — photos will use "
                "whatever order they were uploaded in."
            )

        generated_listings = []

        total_items = len(groups)

        progress_bar.progress(0, text="Preparing listing generation...")
        status_text.write("Preparing listing generation...")

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
                            item_paths,

                        "marked_vintage":
                            bool(group.get("marked_vintage", False)),
                    }
                )

        completed = 0
        # Keep image-generation concurrency at 2. The previous 4-worker
        # batch saturated the 200k TPM limit and caused cascading failures.
        worker_count = min(
            3,
            max(
                1,
                len(jobs)
            )
        )

        def generate_one(job):
            """Generate one listing with rate-limit-aware retries."""
            last_error = None

            for attempt in range(4):
                try:
                    listing = analyze_item(job["photos"])
                    if listing is None:
                        raise RuntimeError("AI returned an empty listing.")
                    # Preserved for the whole lifetime of this listing —
                    # the AI's own independent read, untouched by any
                    # manual override below. Used both to know whether a
                    # later manual toggle is a "correction" (disagrees
                    # with this) and to seed the correction-learning log.
                    listing["ai_auto_is_vintage"] = listing.get(
                        "is_vintage", False
                    )
                    # The upload-deck "Vintage" checkbox is a manual
                    # safety net on top of the AI's own tag-evidence
                    # check above — only forces vintage ON when checked,
                    # never forces it off (an unchecked box just means
                    # "not asserted," not "confirmed non-vintage").
                    if job.get("marked_vintage"):
                        if not listing["ai_auto_is_vintage"]:
                            record_vintage_correction(
                                listing,
                                from_state=False,
                                to_state=True,
                                ai_auto_verdict=False,
                            )
                        listing = apply_vintage_flag(listing, True)
                    return {
                        "index": job["index"],
                        "item_number": job["item_number"],
                        "photos": job["photos"],
                        "marked_vintage": job.get("marked_vintage", False),
                        "listing": listing,
                    }
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

            return {
                "index": job["index"],
                "item_number": job["item_number"],
                "photos": job["photos"],
                "marked_vintage": job.get("marked_vintage", False),
                "listing": None,
                "error": str(last_error),
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
            for index in sorted(results_by_index)
        ]

        # Rescue transient failures one at a time. The normal path remains
        # concurrent, but a 429 no longer permanently loses an item.
        failed_jobs = [
            job for job in jobs
            if not isinstance(
                results_by_index.get(job["index"], {}).get("listing"),
                dict,
            )
        ]
        if failed_jobs:
            status_text.write(
                f"Retrying {len(failed_jobs)} failed listing(s) one at a time..."
            )
            for retry_number, job in enumerate(failed_jobs, 1):
                rescued = None
                for retry_attempt in range(3):
                    try:
                        listing = analyze_item(job["photos"])
                        if listing is None:
                            raise RuntimeError("AI returned an empty listing.")
                        listing["ai_auto_is_vintage"] = listing.get(
                            "is_vintage", False
                        )
                        if job.get("marked_vintage"):
                            if not listing["ai_auto_is_vintage"]:
                                record_vintage_correction(
                                    listing,
                                    from_state=False,
                                    to_state=True,
                                    ai_auto_verdict=False,
                                )
                            listing = apply_vintage_flag(listing, True)
                        rescued = {
                            "index": job["index"],
                            "item_number": job["item_number"],
                            "photos": job["photos"],
                            "marked_vintage": job.get("marked_vintage", False),
                            "listing": listing,
                        }
                        break
                    except Exception as retry_error:
                        if retry_attempt == 2:
                            rescued = {
                                "index": job["index"],
                                "item_number": job["item_number"],
                                "photos": job["photos"],
                                "marked_vintage": job.get("marked_vintage", False),
                                "listing": None,
                                "error": str(retry_error),
                            }
                            break
                        error_text = str(retry_error).lower()
                        is_rate_limit = (
                            "rate_limit_exceeded" in error_text
                            or "429" in error_text
                            or "tokens per min" in error_text
                        )
                        delay = (15 if is_rate_limit else 5) * (2 ** retry_attempt)
                        time.sleep(delay)
                results_by_index[job["index"]] = rescued
                status_text.write(
                    f"Retrying failed listings — {retry_number}/{len(failed_jobs)}"
                )

        # Rebuild after rescue so recovered listings are reviewed and scraped.
        generated_listings = [
            results_by_index[index]
            for index in sorted(results_by_index)
        ]

        progress_bar.progress(
            1.0,
            text="All listings generated"
        )

        status_text.success(
            "All listings finished!"
        )

        clear_generated_item_widget_state(
            len(
                st.session_state.get(
                    "generated_listings",
                    [],
                )
            )
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

    _brand_step_header(4, "Generated Listings Review")

    st.caption(
        "Review and edit the AI-generated listings before Final QA."
    )

    st.markdown(
        """
        <style>
        .review-item-topline {
            display:flex;
            align-items:center;
            justify-content:space-between;
            gap:12px;
            width:100%;
            margin:0 0 10px 0;
            position:relative;
            z-index:20;
        }

        .review-item-label {
            font-size:13px;
            font-weight:800;
            letter-spacing:.02em;
            white-space:nowrap;
        }

        .review-item-label span {
            opacity:.55;
        }

        .review-market-row { margin: 2px 0 8px 0; }
        .depop-market-badge {
            position:relative;
            display:inline-flex;
            align-items:center;
            justify-content:center;
            gap:7px;
            padding:7px 11px;
            border:1px solid rgba(43,33,48,0.14);
            border-radius:8px;
            background:#FFFFFF;
            color:#2B2130;
            font-size:11px;
            line-height:1;
            cursor:help;
            white-space:nowrap;
            flex-shrink:0;
        }

        .depop-market-badge strong {
            font-size:12px;
        }

        .depop-market-label {
            font-size:9px;
            font-weight:900;
            letter-spacing:.08em;
            opacity:.55;
        }

        .depop-market-empty {
            opacity:.65;
        }

        .depop-market-tooltip {
            visibility:hidden;
            opacity:0;
            position:absolute;
            z-index:99999;
            top:calc(100% + 8px);
            right:0;
            width:480px;
            max-height:330px;
            overflow-y:auto;
            padding:12px;
            border:1px solid rgba(43,33,48,0.14);
            border-radius:12px;
            background:#FFFFFF;
            box-shadow:0 18px 50px rgba(0,0,0,.55);
            transition:opacity .14s ease, visibility .14s ease;
            white-space:normal;
        }

        .depop-market-badge:hover .depop-market-tooltip {
            visibility:visible;
            opacity:1;
        }

        .depop-tooltip-title {
            font-weight:800;
            margin-bottom:8px;
            color:#fff;
        }

        .depop-tooltip-columns {
            display:grid;
            grid-template-columns:1fr 1fr;
            gap:4px 12px;
        }

        .depop-tooltip-col {
            min-width:0;
        }

        .depop-tooltip-col:first-child {
            border-right:1px solid #262b35;
            padding-right:8px;
        }

        .depop-tooltip-col-title {
            font-size:9px;
            font-weight:900;
            letter-spacing:.08em;
            text-transform:uppercase;
            opacity:.55;
            margin-bottom:6px;
        }

        .depop-tooltip-col-empty {
            font-size:11px;
            opacity:.45;
            padding:6px 4px;
        }

        .depop-market-tooltip a {
            display:block;
            padding:6px 4px;
            color:#dfe5ef;
            text-decoration:none;
            border-radius:8px;
            font-size:11px;
            white-space:normal;
            word-break:break-word;
        }

        .depop-market-tooltip a:hover {
            background:#1b202b;
            color:#fff;
        }

        .evidence-hover-label {
            position:relative;
            display:inline-block;
            font-size:14px;
            font-weight:600;
            /* #e5e7eb (light gray) was meant for a dark card background
               like the tooltip/popover below, but this label sits bare
               on the page's own (light-theme) background, where it was
               nearly invisible. This works for light theme; dark theme
               can be revisited if this app ever ships one. */
            color:#31333F;
            text-decoration:underline;
            text-decoration-style:dotted;
            text-underline-offset:3px;
            cursor:help;
            margin-bottom:6px;
        }

        .evidence-hover-popover {
            visibility:hidden;
            opacity:0;
            position:absolute;
            z-index:99999;
            top:calc(100% + 8px);
            left:0;
            padding:10px;
            border:1px solid rgba(43,33,48,0.14);
            border-radius:10px;
            background:#FFFFFF;
            box-shadow:0 18px 50px rgba(0,0,0,.55);
            transition:opacity .14s ease, visibility .14s ease;
            white-space:normal;
        }

        .evidence-hover-label:hover .evidence-hover-popover {
            visibility:visible;
            opacity:1;
        }

        .evidence-hover-popover img {
            display:block;
            width:380px;
            max-width:min(380px, 90vw);
            height:auto;
            border-radius:7px;
            margin-bottom:5px;
        }

        .evidence-hover-popover .evidence-hover-caption {
            font-size:10px;
            opacity:.6;
            text-align:center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <style>
        .review-deck-card {
            width:100%;
            max-width:720px;
            margin:0 auto 18px auto;
            border:1px solid rgba(43,33,48,0.14);
            border-radius:16px;
            background:#FFFFFF;
            overflow:hidden;
            box-shadow:0 18px 45px rgba(0,0,0,.32);
        }
        .review-deck-gallery {
            padding:12px 12px 0;
            height:255px;
            box-sizing:border-box;
            display:grid;
            grid-template-columns:minmax(0,1fr) 132px;
            gap:10px;
        }
        .review-deck-main {
            width:100%;
            height:232px;
            object-fit:cover;
            object-position:center;
            display:block;
            border-radius:8px;
            background:#FBF3E3;
        }
        .review-deck-extras {
            display:grid;
            grid-template-columns:repeat(2,58px);
            grid-auto-rows:58px;
            gap:7px;
            align-content:start;
            justify-content:center;
        }
        .review-deck-extra,
        .review-deck-empty {
            width:58px;
            height:58px;
            box-sizing:border-box;
            border-radius:8px;
            display:block;
            background:#FBF3E3;
        }
        .review-deck-extra {
            object-fit:cover;
            object-position:center;
            border:1px solid rgba(255,255,255,.14);
        }
        .review-deck-empty {
            border:1px dashed rgba(255,255,255,.10);
            background:rgba(255,255,255,.04);
        }
        .review-deck-body {
            padding:10px 14px 13px;
        }
        .review-deck-num {
            font-size:10px;
            opacity:.55;
            font-weight:800;
            letter-spacing:.04em;
        }
        .review-deck-title {
            font-size:18px;
            line-height:1.2;
            font-weight:800;
            margin-top:3px;
            white-space:nowrap;
            overflow:hidden;
            text-overflow:ellipsis;
        }
        .review-deck-meta {
            font-size:12px;
            opacity:.65;
            margin-top:4px;
        }
        .review-deck-no-photo {
            height:255px;
            display:flex;
            align-items:center;
            justify-content:center;
            opacity:.5;
        }
        .review-section-label {
            font-size:11px;
            font-weight:900;
            letter-spacing:.08em;
            color:#F02AA0;
            margin:4px 0 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
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

    # The review page is a carousel: one listing is edited at a time.
    if "selected_review_index" not in st.session_state:
        st.session_state["selected_review_index"] = 0

    review_items = []
    for review_index, review_item in enumerate(generated_listings):
        review_listing = review_item.get("listing") or {}
        review_photos = review_item.get("photos") or []
        review_items.append({
            "number": review_item.get("item_number", review_index + 1),
            "title": review_listing.get("title", f"Item {review_index + 1}"),
            "photos": len(review_photos),
            "thumb": (
                make_item_thumbnail_data_url(review_photos[0])
                if review_photos
                else ""
            ),
            "price": review_listing.get("suggested_price", ""),
        })

    selected_review_index = max(
        0,
        min(
            len(generated_listings) - 1,
            int(st.session_state.get("selected_review_index", 0)),
        ),
    ) if generated_listings else 0

    # ================================================================
    # MARKET SCRAPE (DEPOP + EBAY) — ALL GENERATED ITEMS
    # ================================================================
    # The previous implementation lived inside the selected-item editor,
    # which meant Item 1 got scraped and Items 2..N never did.
    #
    # Runs automatically once for every generated item. Results persist
    # in session state, so normal Streamlit reruns do NOT scrape again.
    # The Depop leg is deliberately rate-limited (2 workers, 1.5s
    # stagger PER ITEM — see market_scraper.py's own comment on a real
    # past incident where firing all requests at once got every item
    # after the first "access denied" by Depop's bot detection), so a
    # large batch can take a while here before the review cards below
    # finish loading pricing — that wait is the tradeoff for not
    # getting the scraper blocked, not something safe to shortcut.
    if generated_listings:
        # Use the generated item's identity/photos for the scrape signature,
        # NOT editable title/brand fields. That way typing in a title later
        # does not unexpectedly launch another 24-result scrape.
        depop_signature = tuple(
            (
                index,
                item.get("item_number", index + 1),
                tuple(
                    str(path)
                    for path in (
                        item.get("photos") or []
                    )[:1]
                ),
            )
            for index, item in enumerate(
                generated_listings
            )
            if item.get("listing")
        )

        previous_signature = st.session_state.get(
            "depop_market_signature"
        )

        if previous_signature != depop_signature:
            # Only successful listings go to Depop. Failed AI outputs must
            # never be sent as empty dictionaries, and scraper indexes must be
            # mapped back to the real generated item indexes.
            valid_pairs = [
                (original_index, item)
                for original_index, item in enumerate(generated_listings)
                if isinstance(item.get("listing"), dict)
            ]
            valid_listings = [
                item.get("listing")
                for _, item in valid_pairs
            ]

            if valid_listings:
                with st.spinner(
                    f"Getting market comps (Depop + eBay) for {len(valid_listings)} items…"
                ):
                    try:
                        scraped = scrape_market_all(
                            valid_listings,
                            limit=24,
                        ) or {}

                        remapped_market = {}
                        for compact_index, result in scraped.items():
                            try:
                                original_index = valid_pairs[int(compact_index)][0]
                            except (IndexError, KeyError, TypeError, ValueError):
                                continue
                            remapped_market[original_index] = result

                        st.session_state["depop_market_results"] = remapped_market

                        # Pricing itself is no longer set here. This block
                        # used to ALSO directly write generated_price_{idx}
                        # etc. for every item with a median — a second,
                        # separate write site duplicating what the
                        # per-item price widget's own sync logic does,
                        # and a likely source of the two drifting out of
                        # sync. The per-item editor below is now the ONE
                        # place that decides and writes the displayed
                        # price, reading depop_market_results fresh every
                        # render — simpler to reason about, nothing here
                        # to get out of sync with it.

                        # Only mark the signature complete after a successful scrape.
                        st.session_state["depop_market_signature"] = depop_signature
                        st.session_state["depop_market_scrape_error"] = ""
                    except Exception as exc:
                        # Leave the signature untouched so the next rerun retries.
                        st.session_state["depop_market_scrape_error"] = str(exc)
                        st.warning(
                            f"Depop market scrape failed: {exc}. It will retry automatically."
                        )

    def _render_review_side_card(review_index, side_col):
        if not (0 <= review_index < len(generated_listings)):
            return

        side_item = generated_listings[review_index]
        side_listing = side_item.get("listing") or {}
        side_photos = side_item.get("photos") or []

        with side_col:
            side_thumb = (
                make_item_thumbnail_data_url(side_photos[0])
                if side_photos
                else ""
            )
            side_number = side_item.get(
                "item_number",
                review_index + 1,
            )
            side_title = str(
                side_listing.get(
                    "title",
                    f"Item {side_number}",
                )
            )

            if side_thumb:
                st.markdown(
                    f"""
                    <div style="width:100%;max-width:190px;margin:52px auto 0;
                                border:1px solid rgba(43,33,48,0.14);border-radius:16px;
                                overflow:hidden;background:#FFFFFF;
                                box-shadow:0 14px 34px rgba(0,0,0,.30);">
                        <img src="{side_thumb}"
                             style="width:100%;height:150px;object-fit:cover;
                                    display:block;background:#FBF3E3;">
                        <div style="padding:10px 11px 12px;">
                            <div style="font-size:10px;font-weight:800;
                                        opacity:.55;letter-spacing:.04em;">
                                ITEM {side_number}
                            </div>
                            <div style="font-size:14px;font-weight:800;
                                        margin-top:4px;white-space:nowrap;
                                        overflow:hidden;text-overflow:ellipsis;">
                                {side_title}
                            </div>
                            <div style="font-size:11px;opacity:.60;margin-top:5px;">
                                {len(side_photos)} photos
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    for item_index, item in enumerate(
        generated_listings
    ):
        # Only the active card gets the full editor. This is the key change
        # that removes the 50-item vertical scroll wall.
        if item_index != selected_review_index:
            continue

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

        # A failed generation is a real review state. Never call .get()
        # on None; render the failure card and let the user retry it.
        if not isinstance(listing, dict):
            st.error("This listing did not generate.")
            st.code(item.get("error", "Unknown generation error."))
            continue

        title = str(listing.get("title", ""))
        brand = str(listing.get("brand", ""))
        category = str(listing.get("category_path", listing.get("category", "")) or "")
        garment_type = str(listing.get("garment_type", listing.get("type", "")) or "")
        size = str(listing.get("size", "-") or "-")
        color = str(listing.get("color", "-") or "-")
        pattern = str(listing.get("pattern", "-") or "-")
        condition = str(listing.get("condition", ""))
        description = str(listing.get("description", ""))
        hashtags = str(listing.get("hashtags", "") or "")

        # ============================================================
        # REVIEW DECK STAGE
        # ============================================================
        review_stage = st.columns(
            [0.95, 4.8, 0.95],
            gap="medium",
        )

        if selected_review_index > 0:
            _render_review_side_card(
                selected_review_index - 1,
                review_stage[0],
            )

        with review_stage[1]:
            with st.container(border=True):
                # ========================================================
                # REVIEW CARD
                # ========================================================
                # REVIEW CARD HEADER
                # Previous/next navigation lives here. The old standalone
                # carousel above the card is gone.
                # ========================================================
                nav_left, nav_center, nav_right = st.columns(
                    [0.55, 3.9, 0.55],
                    gap="small",
                )

                with nav_left:
                    if st.button(
                        "‹",
                        key=f"review_prev_{item_index}",
                        disabled=(selected_review_index <= 0),
                        use_container_width=True,
                    ):
                        st.session_state["selected_review_index"] = (
                            selected_review_index - 1
                        )
                        st.rerun()

                with nav_center:
                    st.markdown(
                        f"<div style='text-align:center;font-weight:800;"
                        f"font-size:12px;opacity:.65;padding-top:6px;'>"
                        f"ITEM {item_number} · "
                        f"{selected_review_index + 1} / "
                        f"{len(generated_listings)}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                with nav_right:
                    if st.button(
                        "›",
                        key=f"review_next_{item_index}",
                        disabled=(
                            selected_review_index
                            >= len(generated_listings) - 1
                        ),
                        use_container_width=True,
                    ):
                        st.session_state["selected_review_index"] = (
                            selected_review_index + 1
                        )
                        st.rerun()

                # Photos first -> primary listing info -> specifics -> description
                # ========================================================
                st.markdown(
                    f"**ITEM {item_number}**  ·  "
                    f"**{selected_review_index + 1} / {len(generated_listings)}**"
                )

            # --------------------------------------------------------
                # DEPOP MARKET BADGE
                # --------------------------------------------------------
                # Market scraping is performed for ALL generated listings once,
                # outside the selected-card editor. This prevents the old bug
                # where only Item 1 was ever scraped.
                depop_market_all = st.session_state.get(
                    "depop_market_results",
                    {},
                )

            # DECK CARD — SAME PHOTO LAYOUT AS THE INVENTORY DECK
            # The photos appear once, in the card header.
            # Editable listing fields live directly underneath.
            #
            # Click-to-zoom + per-photo rotate/delete: this used to be a
            # block of plain <img> tags with no interactivity at all.
            # Main photo big + smaller extras beside it (same visual
            # layout as before), each with 3 small circular overlay
            # buttons — 🔍 top-left (open the full-size zoom dialog,
            # built-in X close), ↻ bottom-right (fix a sideways/upside-
            # down photo on the spot), ✕ top-right (drop a wrong/
            # duplicate photo from this listing).
            # ========================================================
            review_number = item.get(
                "item_number",
                item_index + 1,
            )
            review_title = str(
                listing.get(
                    "title",
                    f"Item {review_number}",
                )
            )

            review_photos = list(photos[:7])

            with st.container(border=True):
                if review_photos:
                    st.markdown(
                        _REVIEW_PHOTO_BOX_CSS,
                        unsafe_allow_html=True,
                    )

                    main_photo = review_photos[0]
                    extra_photos = review_photos[1:7]

                    main_col, extras_col = st.columns(
                        [3, 2], gap="small"
                    )

                    with main_col:
                        _render_review_photo_box(
                            item_index,
                            0,
                            main_photo,
                            review_photos,
                            "main",
                        )

                    with extras_col:
                        if extra_photos:
                            # 2-column grid of small squares, dashed
                            # placeholders for unfilled slots — matches
                            # the inventory deck's .deck-extra-photos
                            # card exactly instead of a scrolling row.
                            with st.container(
                                key=f"review_extras_grid_{item_index}"
                            ):
                                slot_count = len(extra_photos)
                                if slot_count % 2 == 1:
                                    slot_count += 1

                                for row_start in range(0, slot_count, 2):
                                    row_cols = st.columns(2, gap="small")
                                    for col_offset in range(2):
                                        slot_index = row_start + col_offset
                                        with row_cols[col_offset]:
                                            if slot_index < len(extra_photos):
                                                _render_review_photo_box(
                                                    item_index,
                                                    slot_index + 1,
                                                    extra_photos[slot_index],
                                                    review_photos,
                                                    "extra",
                                                )
                                            else:
                                                st.markdown(
                                                    '<div class="review-extra-empty"></div>',
                                                    unsafe_allow_html=True,
                                                )

                    st.markdown(
                        f"""
                        <div class="review-deck-body">
                            <div class="review-deck-num">
                                ITEM {review_number}
                            </div>
                            <div class="review-deck-title">
                                {review_title}
                            </div>
                            <div class="review-deck-meta">
                                {len(photos)} photos
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="review-deck-no-photo">
                            No photos attached
                        </div>
                        <div class="review-deck-body">
                            <div class="review-deck-num">
                                ITEM {review_number}
                            </div>
                            <div class="review-deck-title">
                                {review_title}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            if (
                st.session_state.get("review_photo_zoom_state")
                and st.session_state.get("review_photo_zoom_open")
                and st.session_state["review_photo_zoom_state"].get(
                    "item_index"
                ) == item_index
            ):
                # Consumed immediately — only re-armed by an explicit
                # click inside the dialog (Prev/Next/Rotate) or by
                # opening it fresh via 🔍. A dialog dismissed via
                # Escape/clicking outside it runs no app code, so this
                # stays un-armed and a later unrelated button click
                # (e.g. "Run Final QA") won't pop this photo back up.
                st.session_state["review_photo_zoom_open"] = False
                _review_photo_zoom_dialog()

            # Depop market pricing for THIS active item.
            depop_market_all = st.session_state.get(
                "depop_market_results",
                {},
            )
            depop_market = depop_market_all.get(
                item_index,
                {},
            )
            depop_summary = depop_market.get(
                "summary",
                {},
            )

            depop_median = depop_summary.get("median")
            depop_low = depop_summary.get("low")
            depop_high = depop_summary.get("high")

            # Split comps by source so the hover tooltip can show Depop
            # on one side and eBay on the other, instead of one
            # interleaved list — each side keeps its own comp links.
            depop_tooltip_lines = []
            ebay_tooltip_lines = []
            for comp in (
                depop_market.get("comps", [])
                if isinstance(depop_market, dict)
                else []
            )[:48]:
                if comp.get("price") is None:
                    continue

                comp_title = html.escape(
                    str(comp.get("title", "Market listing"))[:90]
                )
                comp_url = html.escape(
                    str(comp.get("url", "")),
                    quote=True,
                )
                if not comp_url:
                    continue

                line = (
                    f'<a href="{comp_url}" target="_blank">'
                    f'${float(comp["price"]):.0f} · {comp_title}'
                    f'</a>'
                )

                if str(comp.get("source", "")).lower() == "ebay":
                    ebay_tooltip_lines.append(line)
                else:
                    depop_tooltip_lines.append(line)

            if depop_median is not None:
                market_badge_html = (
                    '<div class="depop-market-badge">'
                    '<span class="depop-market-label">MARKET</span>'
                    f'<strong>${float(depop_median):.0f} median</strong>'
                    f'<span>· ${float(depop_low):.0f} low</span>'
                    f'<span>· ${float(depop_high):.0f} high</span>'
                )

                if depop_tooltip_lines or ebay_tooltip_lines:
                    market_badge_html += (
                        '<div class="depop-market-tooltip">'
                        '<div class="depop-tooltip-title">'
                        'Compared listings'
                        '</div>'
                        '<div class="depop-tooltip-columns">'
                        '<div class="depop-tooltip-col">'
                        '<div class="depop-tooltip-col-title">Depop</div>'
                        + (
                            "".join(depop_tooltip_lines)
                            if depop_tooltip_lines
                            else '<div class="depop-tooltip-col-empty">'
                            "No Depop comps</div>"
                        )
                        + '</div>'
                        '<div class="depop-tooltip-col">'
                        '<div class="depop-tooltip-col-title">eBay</div>'
                        + (
                            "".join(ebay_tooltip_lines)
                            if ebay_tooltip_lines
                            else '<div class="depop-tooltip-col-empty">'
                            "No eBay comps</div>"
                        )
                        + '</div>'
                        '</div>'
                        '</div>'
                    )

                market_badge_html += "</div>"
            elif depop_market.get("errors"):
                # A genuine scrape failure (e.g. Depop's bot detection,
                # or an eBay auth/network error) must not look identical
                # to "this item legitimately has no comps" — that
                # silently hid the failure before and left the AI's own
                # price estimate standing in with no indication anything
                # went wrong. Surface it and point at the retry button
                # below.
                market_badge_html = (
                    '<div class="depop-market-badge depop-market-empty" '
                    'title="'
                    + html.escape(
                        "; ".join(
                            str(e) for e in depop_market.get("errors", [])
                        )
                    )
                    + '">'
                    'MARKET · comps unavailable — tap ↻ Rescrape below'
                    '</div>'
                )
            else:
                market_badge_html = (
                    '<div class="depop-market-badge depop-market-empty">'
                    'MARKET · no priced comps yet'
                    '</div>'
                )

            st.markdown(
                f'<div class="review-market-row">{market_badge_html}</div>',
                unsafe_allow_html=True,
            )

            if st.button(
                "↻ Rescrape Market",
                key=f"rescrape_depop_{item_index}",
                help="Refresh this item's Depop + eBay market comps.",
            ):
                try:
                    fresh = scrape_market_all(
                        [listing],
                        limit=24,
                        workers=1,
                    )

                    if fresh:
                        depop_market_all[item_index] = fresh.get(
                            0,
                            {},
                        )
                        st.session_state[
                            "depop_market_results"
                        ] = depop_market_all

                        # Pricing itself is no longer set here — the
                        # per-item price widget below is the one place
                        # that computes and syncs the displayed price,
                        # reading depop_market_results fresh every
                        # render. It'll pick up this new median
                        # automatically (and correctly leaves a manual
                        # override alone, unlike the old code here which
                        # unconditionally reset it just from a rescrape).

                        st.rerun()
                except Exception as exc:
                    st.error(
                        f"Market scrape failed: {exc}"
                    )


            with st.container(border=True):
                st.markdown(
                    "<div class='review-section-label'>LISTING</div>",
                    unsafe_allow_html=True,
                )

                # ========================================================
                # LISTING FIELDS
                # ========================================================

                # Manual "Vintage" override, in case the item wasn't
                # marked vintage on import and the AI's own tag-evidence
                # check (analyze_item -> _verify_vintage_from_tags) also
                # missed it — or to correct a false positive. Toggling
                # this directly adds/removes "Vintage" from the title
                # below, via the same rules (never Vintage+Y2K, never
                # Fairycore+Coquette, max 2 style words) the AI itself
                # follows. The AI's own check still always runs on its
                # own — this is a manual safety net on top, not a
                # replacement.
                # Pending repairs from a "Fix Issues with AI" click on a
                # PRIOR run get consumed here — before ANY widget for
                # this item is instantiated. Setting a widget's
                # session_state value directly from the button's own
                # click handler (further below, same run) hits
                # Streamlit's "cannot modify a widget's value after
                # it's already been instantiated this run" restriction,
                # since by the time that handler runs, these widgets
                # already rendered earlier in the SAME script pass —
                # hence the two-step handoff: the button just stores
                # what changed and reruns; this consumes it fresh,
                # before instantiation, on the very next run.
                pending_repair = st.session_state.pop(
                    f"pending_repair_{item_index}", None
                )
                if pending_repair:
                    if "title" in pending_repair:
                        st.session_state[
                            f"generated_title_{item_index}"
                        ] = pending_repair["title"]
                    if "description" in pending_repair:
                        st.session_state[
                            f"generated_description_{item_index}"
                        ] = pending_repair["description"]
                    if "hashtags" in pending_repair:
                        st.session_state[
                            f"generated_hashtags_{item_index}"
                        ] = pending_repair["hashtags"]
                    if "size" in pending_repair:
                        st.session_state[
                            f"generated_shopify_size_{item_index}_select"
                        ] = pending_repair["size"]

                vintage_key = f"generated_vintage_{item_index}"
                vintage_prev_key = f"{vintage_key}_prev"
                title_key = f"generated_title_{item_index}"

                marked_vintage = st.checkbox(
                    "Vintage item",
                    value=bool(listing.get("is_vintage", False)),
                    key=vintage_key,
                    help=(
                        "Check if this item is vintage even if it wasn't "
                        "marked vintage on import — updates the title "
                        "automatically. Uncheck to remove \"Vintage\" "
                        "from the title."
                    ),
                )

                previously_marked_vintage = st.session_state.get(
                    vintage_prev_key, marked_vintage
                )

                if marked_vintage != previously_marked_vintage:
                    pre_toggle_title = str(
                        st.session_state.get(
                            title_key, listing.get("title", "")
                        )
                    )
                    listing["title"] = apply_vintage_flag(
                        {"title": pre_toggle_title},
                        marked_vintage,
                    )["title"]
                    st.session_state[title_key] = listing["title"]

                    record_vintage_correction(
                        listing,
                        from_state=previously_marked_vintage,
                        to_state=marked_vintage,
                        ai_auto_verdict=listing.get(
                            "ai_auto_is_vintage",
                            listing.get("is_vintage", False),
                        ),
                    )

                listing["is_vintage"] = marked_vintage
                st.session_state[vintage_prev_key] = marked_vintage

                title = st.text_input(
                    "Title",
                    value=str(listing.get("title", "")),
                    key=title_key,
                )

                top_cols = st.columns(2, gap="medium")

                with top_cols[0]:
                    st.markdown(
                        _evidence_hover_label_html(
                            "Brand", photos, listing
                        ),
                        unsafe_allow_html=True,
                    )
                    brand = st.text_input(
                        "Brand",
                        value=str(listing.get("brand", "")),
                        key=f"generated_brand_{item_index}",
                        label_visibility="collapsed",
                    )
                    if listing.get("brand_confidence"):
                        st.caption(
                            f"AI confidence: {listing['brand_confidence']}%"
                        )

                with top_cols[1]:
                    category_choices = (
                        ["— Manual Review —"]
                        + SHOPIFY_CATEGORY_PATHS
                    )

                    existing_category = str(
                        listing.get(
                            "category_path",
                            listing.get("category", ""),
                        )
                    ).strip()

                    if existing_category in category_choices:
                        category_index = category_choices.index(
                            existing_category
                        )
                    else:
                        # The AI's category can be a near-miss (case,
                        # stray whitespace, "  >  " spacing) rather than
                        # a genuinely different category — match those
                        # before giving up and forcing Manual Review.
                        normalized_existing = _normalize_category_string(
                            existing_category
                        )
                        matched_choice = next(
                            (
                                choice
                                for choice in SHOPIFY_CATEGORY_PATHS
                                if _normalize_category_string(choice)
                                == normalized_existing
                            ),
                            None,
                        ) if normalized_existing else None

                        category_index = (
                            category_choices.index(matched_choice)
                            if matched_choice
                            else 0
                        )

                    category = st.selectbox(
                        "Shopify Category",
                        category_choices,
                        index=category_index,
                        key=f"generated_category_{item_index}",
                        help="Exact Shopify Standard Product Taxonomy category.",
                    )

                    if category == "— Manual Review —":
                        category = "-"
                        listing["category"] = "-"
                        listing["category_path"] = ""
                        listing["category_gid"] = ""
                    else:
                        listing["category"] = category
                        listing["category_path"] = category
                        listing["category_gid"] = SHOPIFY_CATEGORY_GIDS.get(
                            category,
                            "",
                        )

                # Garment Type + Price share the same row and width.
                garment_price_cols = st.columns(2, gap="medium")

                with garment_price_cols[0]:
                    garment_key = f"generated_garment_type_{item_index}"
                    desired_garment = str(
                        listing.get(
                            "garment_type",
                            listing.get("type", ""),
                        ) or ""
                    ).strip()

                    # _garment_type_select() already computes and passes
                    # its own `index=` from desired_garment every render —
                    # pre-seeding session_state here too was redundant and
                    # raced with it (Streamlit warns/rejects a widget given
                    # both an explicit default AND a same-run session_state
                    # write for the same key).
                    garment_type = _garment_type_select(
                        desired_garment,
                        garment_key,
                    )

                    listing["garment_type"] = garment_type
                    listing["type"] = garment_type

                with garment_price_cols[1]:
                    price_widget_key = f"generated_price_{item_index}"
                    manual_price_key = f"pricing_manual_{item_index}"
                    auto_baseline_key = f"pricing_auto_baseline_{item_index}"
                    pricing_context_key = f"pricing_context_{item_index}"

                    depop_result_for_price = (
                        st.session_state
                        .get("depop_market_results", {})
                        .get(item_index, {})
                    ) or {}
                    market_for_price = depop_result_for_price.get("summary", {}) or {}
                    market_errors_for_price = depop_result_for_price.get("errors")
                    median_for_price = market_for_price.get("median")

                    # Single source of truth for "what should the price
                    # box show right now" — computed FRESH every render
                    # straight from depop_market_results (the same data
                    # the badge/caption below already read), instead of
                    # being tracked through several separate session-state
                    # mirrors (a stored median, an auto-baseline, a
                    # "changed" flag) that could silently drift out of
                    # sync with each other. That drift is what left items
                    # 2+ stuck on a stale price while item 1 — whichever
                    # one happened to stay in sync — looked fine.
                    # Whole dollars only — no cents. Market medians are
                    # rarely round numbers, so a straight percentage over
                    # median almost always lands on a cent value (e.g.
                    # $18.555) without this rounding step.
                    #
                    # Uses the learned multiplier for this brand/garment/
                    # style bucket when one exists (_record_manual_price
                    # below updates it every time the user overrides a
                    # price) — falls back to BASE_PRICING_MULTIPLIER
                    # (50%) for a bucket with no history yet, so behavior
                    # is unchanged until the user has actually taught it
                    # something.
                    if median_for_price is not None:
                        learned_multiplier, _learned_key, learned_count = (
                            _get_learned_multiplier(listing)
                        )
                        computed_auto_price = round(
                            float(median_for_price) * learned_multiplier
                        )
                    else:
                        learned_multiplier = BASE_PRICING_MULTIPLIER
                        learned_count = 0
                        computed_auto_price = round(
                            float(listing.get("suggested_price", 10) or 10)
                        )

                    is_manual_price = bool(
                        st.session_state.get(manual_price_key, False)
                    )
                    previous_auto_price = st.session_state.get(auto_baseline_key)

                    # Sync to the freshly-computed price whenever: this
                    # item's price has never rendered before, OR the
                    # computed price genuinely changed (new/updated
                    # market data) and the user hasn't manually typed
                    # their own number for this item.
                    needs_sync = (
                        price_widget_key not in st.session_state
                        or (
                            not is_manual_price
                            and (
                                previous_auto_price is None
                                or computed_auto_price != int(previous_auto_price)
                            )
                        )
                    )

                    if needs_sync:
                        st.session_state[price_widget_key] = computed_auto_price
                        st.session_state[auto_baseline_key] = computed_auto_price
                        st.session_state[pricing_context_key] = {
                            "median": median_for_price
                        }

                    # int min/max/step puts the widget in whole-dollar-only
                    # mode — Streamlit won't accept a typed decimal here.
                    suggested_price = st.number_input(
                        "Price",
                        min_value=0,
                        max_value=1000,
                        step=1,
                        key=price_widget_key,
                        help=(
                            "Starts at 50% above the combined Depop + "
                            "eBay median, rounded to the nearest dollar. "
                            "Change it manually to set your price."
                        ),
                    )

                    # The widget's returned value differs from the last
                    # auto price we synced -> the user typed their own
                    # number. Remember it as a manual override (feeds the
                    # personalized multiplier-learning below) so the next
                    # render's sync check doesn't stomp on it.
                    if int(suggested_price) != int(
                        st.session_state.get(
                            auto_baseline_key, suggested_price
                        )
                    ):
                        median_for_learning = st.session_state.get(
                            pricing_context_key, {}
                        ).get("median")

                        if median_for_learning:
                            learned = _record_manual_price(
                                listing,
                                float(suggested_price),
                                float(median_for_learning),
                            )
                            st.session_state["pricing_last_learning"] = {
                                "item_index": item_index,
                                "manual_price": float(suggested_price),
                                "median": float(median_for_learning),
                                "multiplier": float(
                                    learned or BASE_PRICING_MULTIPLIER
                                ),
                            }

                        st.session_state[manual_price_key] = True
                        st.session_state[auto_baseline_key] = int(suggested_price)

                    listing["suggested_price"] = int(
                        suggested_price
                    )

                    if median_for_price is not None:
                        markup_pct = round((learned_multiplier - 1) * 100)
                        learned_note = (
                            f" (learned from {learned_count} of your past prices)"
                            if learned_count > 0
                            else ""
                        )
                        st.caption(
                            f"{markup_pct}% above ${float(median_for_price):.2f} "
                            f"median → ${computed_auto_price:.0f}{learned_note}"
                        )
                    elif market_errors_for_price:
                        # A genuine scrape failure (Depop bot-detection,
                        # an eBay auth/network error, timeout, etc.) must
                        # not silently fall back to a generic price with
                        # no indication anything went wrong — this is the
                        # AI's own estimate standing in, not a real
                        # market comp.
                        st.caption(
                            "⚠️ Market price unavailable — using AI "
                            "estimate. Tap ↻ Rescrape Market above to retry."
                        )

                taxonomy = (
                    SHOPIFY_JEANS_TAXONOMY
                    if _is_jeans_garment(garment_type)
                    else SHOPIFY_TAXONOMY
                )

                front_attribute_names = [
                    "Size",
                    "Color",
                    "Pattern",
                ]

                shopify_values = {}
                attr_cols = st.columns(3, gap="small")

                for attr_index, attribute_name in enumerate(
                    front_attribute_names
                ):
                    if attribute_name not in taxonomy.get(
                        "attributes",
                        {},
                    ):
                        continue

                    field = _taxonomy_key(attribute_name)
                    key = (
                        f"generated_shopify_{field}_{item_index}"
                    )
                    current = listing.get(field, "")

                    with attr_cols[attr_index]:
                        if field == "size":
                            st.markdown(
                                _evidence_hover_label_html(
                                    "Size", photos, listing
                                ),
                                unsafe_allow_html=True,
                            )

                        value = _controlled_select(
                            attribute_name,
                            field,
                            current,
                            key,
                            taxonomy=taxonomy,
                            attribute_name=attribute_name,
                            label_visibility=(
                                "collapsed"
                                if field == "size"
                                else "visible"
                            ),
                        )

                    shopify_values[field] = value
                    listing[field] = value

                size = (
                    shopify_values.get(
                        "size",
                        listing.get("size", "-"),
                    ) or "-"
                )

                color = (
                    shopify_values.get(
                        "color",
                        listing.get("color", "-"),
                    ) or "-"
                )

                pattern = (
                    shopify_values.get(
                        "pattern",
                        listing.get("pattern", "-"),
                    ) or "-"
                )

                condition = st.text_input(
                    "Condition",
                    value=str(listing.get("condition", "")),
                    key=f"generated_condition_{item_index}",
                )


                # ========================================================
                # 3. SPECIFICS — COLLAPSED
                # ========================================================
                optional_key = (
                    f"generated_optional_open_{item_index}"
                )

                if optional_key not in st.session_state:
                    st.session_state[optional_key] = False

                optional_label = (
                    "Hide Specifics  ↑"
                    if st.session_state[optional_key]
                    else "Specifics  →"
                )

                if st.button(
                    optional_label,
                    key=f"toggle_optional_{item_index}",
                    use_container_width=True,
                ):
                    st.session_state[optional_key] = (
                        not st.session_state[optional_key]
                    )
                    st.rerun()

                if st.session_state[optional_key]:
                    st.caption(
                        "Optional Shopify attributes. "
                        "AI can fill these from the photos."
                    )

                    if st.button(
                        "✨ AI Fill Optional",
                        key=f"ai_fill_optional_{item_index}",
                    ):
                        with st.spinner(
                            "AI checking optional attributes from the photos..."
                        ):
                            try:
                                optional_ai = analyze_item(photos)

                                if optional_ai:
                                    primary_fields = {
                                        "title",
                                        "brand",
                                        "garment_type",
                                        "type",
                                        "category",
                                        "category_path",
                                        "category_gid",
                                        "size",
                                        "color",
                                        "pattern",
                                        "condition",
                                        "description",
                                        "hashtags",
                                        "suggested_price",
                                    }

                                    for attribute_name in taxonomy.get(
                                        "attributes",
                                        {},
                                    ):
                                        field = _taxonomy_key(
                                            attribute_name
                                        )

                                        if field in primary_fields:
                                            continue

                                        if field in optional_ai:
                                            listing[field] = optional_ai[field]

                                    st.success(
                                        "Optional attributes filled from photos."
                                    )
                                    st.rerun()

                            except Exception as exc:
                                st.error(
                                    f"AI optional attribute check failed: {exc}"
                                )

                    for attribute_name in taxonomy.get(
                        "attributes",
                        {},
                    ):
                        if attribute_name in front_attribute_names:
                            continue

                        field = _taxonomy_key(attribute_name)
                        key = (
                            f"generated_shopify_{field}_{item_index}"
                        )
                        current = listing.get(field, "")

                        if attribute_name == "Clothing features":
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

                        listing[field] = value

                # ========================================================
                # 4. DESCRIPTION UNDER SPECIFICS
                # ========================================================
                st.markdown("### Description")

                description_key = (
                    f"generated_description_{item_index}"
                )
                description_size_key = (
                    f"description_size_sync_{item_index}"
                )

                raw_description = str(
                    listing.get("description", "")
                )
                previous_size = st.session_state.get(
                    description_size_key
                )

                hashtags_value = listing.get(
                    "hashtags",
                    [],
                )

                if isinstance(hashtags_value, list):
                    hashtags_text = " ".join(
                        str(tag)
                        for tag in hashtags_value
                    )
                else:
                    hashtags_text = str(hashtags_value)

                if (
                    description_key not in st.session_state
                    or previous_size != size
                ):
                    st.session_state[
                        description_key
                    ] = format_depop_description(
                        raw_description,
                        title,
                        size,
                        condition,
                        hashtags_value,
                    )
                    st.session_state[
                        description_size_key
                    ] = size

                description = st.text_area(
                    "Description",
                    height=300,
                    key=description_key,
                    label_visibility="collapsed",
                )

                st.markdown("### Hashtags")

                hashtags = st.text_input(
                    "Hashtags — EXACTLY 5 REQUIRED",
                    value=hashtags_text,
                    key=f"generated_hashtags_{item_index}",
                    label_visibility="collapsed",
                )

                listing["description"] = description
                listing["hashtags"] = hashtags


                listing["title"] = title
                listing["is_vintage"] = marked_vintage
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

                # The widgets above are the source of truth for this item.
                # Save explicitly commits them to generated_listings so a
                # rerun cannot make the editor appear to lose the change.
                if st.button(
                    "Save Changes",
                    key=f"save_review_{item_index}",
                    type="primary",
                ):
                    st.session_state["generated_listings"][item_index]["listing"] = listing
                    st.session_state[f"review_saved_{item_index}"] = True
                    st.rerun()

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
                            # Repair only the size. Do NOT regenerate the
                            # entire listing and wipe the user's edits.
                            repair_fields.append(
                                "size"
                            )

                        # Added LAST, deliberately: description is now
                        # assembled from condition/size/hashtags (see
                        # build_depop_description in ai_listing.py), so
                        # if it's being fixed alongside any of those in
                        # this same pass, it must regenerate after them
                        # or it would embed the stale pre-repair values.
                        if any(
                            "description" in failure.lower()
                            for failure in failures
                        ):
                            repair_fields.append(
                                "description"
                            )

                        fixed_fields = []
                        failed_fields = []

                        with st.spinner(
                            f"AI fixing Item {item_number}..."
                        ):

                            for field in repair_fields:
                                # Each field gets its own try/except so one
                                # field failing (e.g. size, if the tag
                                # photos are unreadable) doesn't discard
                                # repairs already computed for the others
                                # in this same pass.
                                try:
                                    repaired_value = regenerate_field(
                                        field,
                                        listing,
                                        photos,
                                    )

                                    if field == "hashtags":
                                        repaired_value = [
                                            tag if str(tag).startswith("#")
                                            else "#" + str(tag).strip()
                                            for tag in (repaired_value or [])
                                            if str(tag).strip()
                                        ]
                                        if len(repaired_value) != 5:
                                            raise RuntimeError(
                                                "AI did not return exactly 5 hashtags."
                                            )

                                    if field == "size":
                                        if isinstance(repaired_value, dict):
                                            repaired_value = repaired_value.get("value", "Unknown")
                                        repaired_value = str(repaired_value or "Unknown").strip()

                                    if field == "title":
                                        # Re-apply the manual Vintage
                                        # checkbox state so a QA-triggered
                                        # title repair can't silently
                                        # undo it.
                                        repaired_value = apply_vintage_flag(
                                            {"title": repaired_value},
                                            listing.get("is_vintage", False),
                                        )["title"]

                                    listing[field] = repaired_value
                                    fixed_fields.append(field)

                                except Exception as field_error:
                                    failed_fields.append(
                                        f"{field} ({field_error})"
                                    )

                            item["listing"] = listing

                            # Can't write these widgets' session_state
                            # directly here — they already rendered
                            # earlier in this same script pass (see the
                            # matching comment up at "Pending repairs
                            # from a 'Fix Issues with AI' click"). Stage
                            # the new values and let next run's
                            # pre-instantiation sync apply them.
                            pending_repair = {}
                            if "size" in fixed_fields:
                                pending_repair["size"] = listing.get("size", "")
                            if "title" in fixed_fields:
                                pending_repair["title"] = listing.get("title", "")
                            if "description" in fixed_fields:
                                pending_repair["description"] = listing.get("description", "")
                            if "hashtags" in fixed_fields:
                                pending_repair["hashtags"] = " ".join(listing.get("hashtags", []))

                            if pending_repair:
                                st.session_state[
                                    f"pending_repair_{item_index}"
                                ] = pending_repair

                        st.session_state[
                            "generated_listings"
                        ] = generated_listings

                        if fixed_fields:
                            st.success(
                                f"Item {item_number}: fixed "
                                + ", ".join(fixed_fields)
                                + ". Review the updated fields below."
                            )

                        if failed_fields:
                            st.error(
                                "AI could not fix: "
                                + ", ".join(failed_fields)
                            )

                        if fixed_fields:
                            st.rerun()

                else:

                    st.success(
                        "Ready for Final AI QA"
                    )

            # --------------------------------------------------------
    if selected_review_index < len(generated_listings) - 1:
        _render_review_side_card(
            selected_review_index + 1,
            review_stage[2],
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

    # FAILED GENERATIONS — KEEP THEM VISIBLE AT THE BOTTOM
    failed_items = [
        (idx, item)
        for idx, item in enumerate(generated_listings)
        if not isinstance(item.get("listing"), dict)
    ]
    if failed_items:
        st.divider()
        st.subheader("Generation Issues")
        st.caption("Failed items stay here with their photos so nothing disappears.")
        for failed_index, failed_item in failed_items:
            failed_number = failed_item.get("item_number", failed_index + 1)
            with st.container(border=True):
                issue_col, retry_col = st.columns([4, 1])
                with issue_col:
                    st.markdown(
                        f"**ITEM {failed_number}** · {len(failed_item.get('photos') or [])} photos"
                    )
                    st.error(failed_item.get("error", "Unknown generation error."))
                with retry_col:
                    if st.button("Retry", key=f"retry_failed_listing_{failed_index}"):
                        try:
                            retry_listing = analyze_item(failed_item.get("photos") or [])
                            if retry_listing is None:
                                raise RuntimeError("AI returned an empty listing.")
                            retry_listing["ai_auto_is_vintage"] = retry_listing.get(
                                "is_vintage", False
                            )
                            if failed_item.get("marked_vintage"):
                                if not retry_listing["ai_auto_is_vintage"]:
                                    record_vintage_correction(
                                        retry_listing,
                                        from_state=False,
                                        to_state=True,
                                        ai_auto_verdict=False,
                                    )
                                retry_listing = apply_vintage_flag(retry_listing, True)
                            generated_listings[failed_index]["listing"] = retry_listing
                            generated_listings[failed_index].pop("error", None)
                            st.session_state["generated_listings"] = generated_listings
                            st.session_state.pop("depop_market_signature", None)
                            st.rerun()
                        except Exception as retry_error:
                            st.error(str(retry_error))

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

            # A single generic message here is easy to miss in a big
            # batch — "the button does nothing" reports have actually
            # been the gate correctly blocking, just without saying
            # WHICH item or WHY clearly enough to notice. List every
            # blocking item and its exact reason(s) right here.
            problem_lines = []

            if failed_count:
                problem_lines.append(
                    f"- {failed_count} listing(s) failed to generate — "
                    "see \"Generation Issues\" above."
                )

            for item_index in sorted(listing_failures):
                item_number = generated_listings[item_index].get(
                    "item_number", item_index + 1
                )
                reasons = " ".join(listing_failures[item_index])
                problem_lines.append(f"- Item {item_number}: {reasons}")

            st.error(
                "Cannot run Final AI QA yet — fix these first:\n\n"
                + "\n".join(problem_lines)
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
