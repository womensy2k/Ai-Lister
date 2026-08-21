import base64
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from io import BytesIO

from PIL import Image, ImageOps, ImageEnhance, ImageFilter

from dotenv import load_dotenv

load_dotenv()


# ============================================================
# OPENAI SETUP — lazy-loaded
# ============================================================
# Importing the `openai` package itself takes ~0.9s (confirmed by
# timing it directly) — a real, measurable chunk of "the app is slow
# to load", since this module is imported unconditionally by BOTH
# pages (app.py's Generator page and description_generator.py) before
# either can render anything, even just the upload screen. Deferring
# the import (and the client construction) to the first actual AI
# call means the page itself renders without paying that cost; it's
# only paid once, the first time someone actually clicks Generate —
# which is already expecting to take a few seconds for a real API
# call, so the added ~0.9s is far less noticeable there than it is
# sitting in front of an unresponsive page on load.
_openai_client = None
_openai_module = None


def _get_openai():
    """Returns (client, openai_module) — openai_module gives access to
    the exception classes (RateLimitError etc.) without a second
    top-level import. Constructed once per process and cached."""
    global _openai_client, _openai_module

    if _openai_client is None:
        import openai as openai_module

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY was not found. "
                "Make sure your .env file contains your API key."
            )

        _openai_module = openai_module
        _openai_client = openai_module.OpenAI(api_key=api_key)

    return _openai_client, _openai_module


# ============================================================
# MULTI-MODEL VISION CALLS
# ============================================================
# gpt-4o-mini alone was hitting the org's 200k TPM ceiling under load
# (every generation + regen + size-tag pass all draw from the same
# bucket). Each model on the list below has its OWN TPM bucket, so
# spreading calls across them means one model rate-limiting doesn't
# stall the whole app. Ordered by accuracy/capability first — gpt-4.1
# and gpt-4o read fine print (size tags, small text) far more reliably
# than the mini tiers, which are here only as a last-resort fallback if
# every stronger model is rate-limited at once. Override with
# AI_LISTING_MODELS, comma-separated, to change the chain without
# touching code.
VISION_MODEL_CHAIN = [
    model.strip()
    for model in os.getenv(
        "AI_LISTING_MODELS",
        "gpt-4.1,gpt-4o,gpt-4.1-mini,gpt-4o-mini",
    ).split(",")
    if model.strip()
]


def vision_chat_completion(content, temperature=0, models=None, retries_per_model=2, response_format=None):
    """Call chat.completions.create, falling back across models on rate limits.

    Tries each model in `models` (default VISION_MODEL_CHAIN) in order. A
    429 rate limit retries the same model with backoff a couple times, then
    moves on to the next model in the chain rather than failing outright.
    """
    client, openai_module = _get_openai()
    models_to_try = models or VISION_MODEL_CHAIN
    last_error = None

    for model in models_to_try:
        for attempt in range(retries_per_model):
            try:
                kwargs = {
                    "model": model,
                    "messages": [
                        {
                            "role": "user",
                            "content": content,
                        }
                    ],
                    "temperature": temperature,
                }
                if response_format is not None:
                    kwargs["response_format"] = response_format

                return client.chat.completions.create(**kwargs)
            except openai_module.RateLimitError as exc:
                last_error = exc
                if attempt < retries_per_model - 1:
                    time.sleep(1.5 * (attempt + 1))
                else:
                    print(f"{model} rate-limited, falling back to next model...")
            except (openai_module.APIConnectionError, openai_module.APITimeoutError) as exc:
                last_error = exc
                time.sleep(1.0 * (attempt + 1))

    raise last_error


# ============================================================
# SHOPIFY STANDARD PRODUCT TAXONOMY — WOMEN'S CLOTHING
# ============================================================
SHOPIFY_CLOTHING_CATEGORIES = [('Clothing Tops', 'gid://shopify/TaxonomyCategory/aa-1-13'), ('Clothing Tops > Blouses', 'gid://shopify/TaxonomyCategory/aa-1-13-1'), ('Clothing Tops > Bodysuits', 'gid://shopify/TaxonomyCategory/aa-1-13-2'), ('Clothing Tops > Cardigans', 'gid://shopify/TaxonomyCategory/aa-1-13-3'), ('Clothing Tops > Corset Tops', 'gid://shopify/TaxonomyCategory/aa-1-13-15'), ('Clothing Tops > Hoodies', 'gid://shopify/TaxonomyCategory/aa-1-13-13'), ('Clothing Tops > Overshirts', 'gid://shopify/TaxonomyCategory/aa-1-13-5'), ('Clothing Tops > Polos', 'gid://shopify/TaxonomyCategory/aa-1-13-6'), ('Clothing Tops > Shirts', 'gid://shopify/TaxonomyCategory/aa-1-13-7'), ('Clothing Tops > Shirts > Dress Shirts', 'gid://shopify/TaxonomyCategory/aa-1-13-7-2'), ('Clothing Tops > Shirts > Henley Shirts', 'gid://shopify/TaxonomyCategory/aa-1-13-7-1'), ('Clothing Tops > Sweaters', 'gid://shopify/TaxonomyCategory/aa-1-13-12'), ('Clothing Tops > Sweatshirts', 'gid://shopify/TaxonomyCategory/aa-1-13-14'), ('Clothing Tops > T-Shirts', 'gid://shopify/TaxonomyCategory/aa-1-13-8'), ('Clothing Tops > Tank Tops', 'gid://shopify/TaxonomyCategory/aa-1-13-9'), ('Clothing Tops > Tunics', 'gid://shopify/TaxonomyCategory/aa-1-13-11'), ('Dresses', 'gid://shopify/TaxonomyCategory/aa-1-4'), ('One-Pieces', 'gid://shopify/TaxonomyCategory/aa-1-9'), ('Outerwear', 'gid://shopify/TaxonomyCategory/aa-1-10'), ('Outerwear > Coats & Jackets', 'gid://shopify/TaxonomyCategory/aa-1-10-2'), ('Outerwear > Coats & Jackets > Blazers', 'gid://shopify/TaxonomyCategory/aa-1-10-2-18'), ('Outerwear > Coats & Jackets > Bolero Jackets', 'gid://shopify/TaxonomyCategory/aa-1-10-2-1'), ('Outerwear > Coats & Jackets > Bomber Jackets', 'gid://shopify/TaxonomyCategory/aa-1-10-2-2'), ('Outerwear > Coats & Jackets > Capes', 'gid://shopify/TaxonomyCategory/aa-1-10-2-3'), ('Outerwear > Coats & Jackets > Overcoats', 'gid://shopify/TaxonomyCategory/aa-1-10-2-5'), ('Outerwear > Coats & Jackets > Parkas', 'gid://shopify/TaxonomyCategory/aa-1-10-2-6'), ('Outerwear > Coats & Jackets > Pea Coats', 'gid://shopify/TaxonomyCategory/aa-1-10-2-7'), ('Outerwear > Coats & Jackets > Ponchos', 'gid://shopify/TaxonomyCategory/aa-1-10-2-8'), ('Outerwear > Coats & Jackets > Puffer Jackets', 'gid://shopify/TaxonomyCategory/aa-1-10-2-9'), ('Outerwear > Coats & Jackets > Rain Coats', 'gid://shopify/TaxonomyCategory/aa-1-10-2-10'), ('Outerwear > Coats & Jackets > Sport Jackets', 'gid://shopify/TaxonomyCategory/aa-1-10-2-11'), ('Outerwear > Coats & Jackets > Track Jackets', 'gid://shopify/TaxonomyCategory/aa-1-10-2-12'), ('Outerwear > Coats & Jackets > Trench Coats', 'gid://shopify/TaxonomyCategory/aa-1-10-2-13'), ('Outerwear > Coats & Jackets > Trucker Jackets', 'gid://shopify/TaxonomyCategory/aa-1-10-2-14'), ('Outerwear > Coats & Jackets > Varsity Jackets', 'gid://shopify/TaxonomyCategory/aa-1-10-2-15'), ('Outerwear > Coats & Jackets > Windbreakers', 'gid://shopify/TaxonomyCategory/aa-1-10-2-16'), ('Outerwear > Coats & Jackets > Wrap Coats', 'gid://shopify/TaxonomyCategory/aa-1-10-2-17'), ('Outerwear > Vests', 'gid://shopify/TaxonomyCategory/aa-1-10-6'), ('Pants', 'gid://shopify/TaxonomyCategory/aa-1-12'), ('Pants > Cargo Pants', 'gid://shopify/TaxonomyCategory/aa-1-12-2'), ('Pants > Chinos', 'gid://shopify/TaxonomyCategory/aa-1-12-3'), ('Pants > Harem Pants', 'gid://shopify/TaxonomyCategory/aa-1-12-12'), ('Pants > Jeans', 'gid://shopify/TaxonomyCategory/aa-1-12-4'), ('Pants > Jeggings', 'gid://shopify/TaxonomyCategory/aa-1-12-5'), ('Pants > Joggers', 'gid://shopify/TaxonomyCategory/aa-1-12-7'), ('Pants > Leggings', 'gid://shopify/TaxonomyCategory/aa-1-12-8'), ('Pants > Palazzo Pants', 'gid://shopify/TaxonomyCategory/aa-1-12-13'), ('Pants > Trousers', 'gid://shopify/TaxonomyCategory/aa-1-12-11'), ('Shorts', 'gid://shopify/TaxonomyCategory/aa-1-14'), ('Shorts > Bermudas', 'gid://shopify/TaxonomyCategory/aa-1-14-1'), ('Shorts > Cargo Shorts', 'gid://shopify/TaxonomyCategory/aa-1-14-2'), ('Shorts > Chino Shorts', 'gid://shopify/TaxonomyCategory/aa-1-14-3'), ('Shorts > Denim Shorts', 'gid://shopify/TaxonomyCategory/aa-1-14-5'), ('Shorts > Jegging Shorts', 'gid://shopify/TaxonomyCategory/aa-1-14-6'), ('Shorts > Jogger Shorts', 'gid://shopify/TaxonomyCategory/aa-1-14-7'), ('Shorts > Legging Shorts', 'gid://shopify/TaxonomyCategory/aa-1-14-8'), ('Shorts > Short Trousers', 'gid://shopify/TaxonomyCategory/aa-1-14-4'), ('Skirts', 'gid://shopify/TaxonomyCategory/aa-1-15'), ('Skorts', 'gid://shopify/TaxonomyCategory/aa-1-16'), ('Swimwear', 'gid://shopify/TaxonomyCategory/aa-1-20'), ('Swimwear > Classic Bikinis', 'gid://shopify/TaxonomyCategory/aa-1-20-6'), ('Swimwear > Cover Ups', 'gid://shopify/TaxonomyCategory/aa-1-20-7'), ('Swimwear > One-Piece Swimsuits', 'gid://shopify/TaxonomyCategory/aa-1-20-22'), ('Swimwear > Swim Dresses', 'gid://shopify/TaxonomyCategory/aa-1-20-17'), ('Swimwear > Swim Leggings', 'gid://shopify/TaxonomyCategory/aa-1-20-29'), ('Swimwear > Swim Shorts', 'gid://shopify/TaxonomyCategory/aa-1-20-30'), ('Swimwear > Swim Skirts', 'gid://shopify/TaxonomyCategory/aa-1-20-27'), ('Swimwear > Swimwear Tops', 'gid://shopify/TaxonomyCategory/aa-1-20-24'), ('Lingerie > Camisoles', 'gid://shopify/TaxonomyCategory/aa-1-6-4'), ('Lingerie > Corsets & Bustiers', 'gid://shopify/TaxonomyCategory/aa-1-6-15')]
SHOPIFY_CATEGORY_PATHS = [name for name, _gid in SHOPIFY_CLOTHING_CATEGORIES]
SHOPIFY_CATEGORY_GIDS = dict(SHOPIFY_CLOTHING_CATEGORIES)


# ============================================================
# IMAGE TO DATA URL
# ============================================================
# PERFORMANCE FIX: this used to read the raw file and base64 it with no
# resizing/compression at all — full 4-8MB phone photos went straight to
# OpenAI on every single vision call (main listing gen, every field regen,
# and every size-tag verification pass). That's the single biggest cause
# of slowness in this app: bigger upload, slower model processing, faster
# TPM throttling. This now resizes + recompresses once, with a small
# in-process cache so the same photo isn't re-encoded twice in one run.

_DATA_URL_CACHE = {}
_DATA_URL_CACHE_MAX = 300


def image_to_data_url(image_path, max_dimension=1500, quality=82):
    image_path = Path(image_path)

    try:
        stat = image_path.stat()
        cache_key = (str(image_path.resolve()), stat.st_mtime_ns, max_dimension, quality)
    except OSError:
        cache_key = None

    if cache_key is not None and cache_key in _DATA_URL_CACHE:
        return _DATA_URL_CACHE[cache_key]

    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source)

        if image.mode != "RGB":
            if "A" in image.getbands():
                background = Image.new("RGB", image.size, "white")
                background.paste(image, mask=image.getchannel("A"))
                image = background
            else:
                image = image.convert("RGB")

        if max(image.size) > max_dimension:
            scale = max_dimension / max(image.size)
            image = image.resize(
                (
                    max(1, int(image.width * scale)),
                    max(1, int(image.height * scale)),
                ),
                Image.Resampling.LANCZOS,
            )

        buffer = BytesIO()
        image.save(
            buffer,
            format="JPEG",
            quality=quality,
            optimize=True,
            progressive=True,
        )

    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    result = "data:image/jpeg;base64," + encoded

    if cache_key is not None:
        if len(_DATA_URL_CACHE) >= _DATA_URL_CACHE_MAX:
            # Cheap eviction: drop an arbitrary old entry rather than
            # tracking LRU order for what is just a same-run cache.
            _DATA_URL_CACHE.pop(next(iter(_DATA_URL_CACHE)))
        _DATA_URL_CACHE[cache_key] = result

    return result


def _image_variants_for_size(image_path, rotations=(0, 180), max_dim=1600):
    """Build orientation variants for the size-tag backup pass.

    PERFORMANCE FIX: this used to always render all 4 rotations (0/90/180/270)
    of every backup photo at high detail. Modern phones already bake EXIF
    orientation in correctly the vast majority of the time, so upright (0)
    and upside-down (180) catch nearly every real case. 90/270 are still
    available by passing rotations=(0,90,180,270) if you find sideways tags
    slipping through in practice.

    max_dim is overridable so callers that don't need fine print detail
    (e.g. rotation checking — judging overall garment orientation, not
    reading small tag text) can request a smaller/faster render without
    affecting the size-tag-reading caller, which keeps the full 1600px
    default it was tuned against.
    """
    path = Path(image_path)
    with Image.open(path) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGB")

        scale = min(1.0, max_dim / max(image.size))
        if scale < 1.0:
            image = image.resize(
                (max(1, int(image.width * scale)),
                 max(1, int(image.height * scale))),
                Image.Resampling.LANCZOS,
            )

        variants = []
        for angle in rotations:
            rotated = image.rotate(angle, expand=True, fillcolor="white") if angle else image
            enhanced = ImageEnhance.Contrast(rotated).enhance(1.12)
            enhanced = ImageEnhance.Sharpness(enhanced).enhance(1.35)
            enhanced = enhanced.filter(ImageFilter.SHARPEN)

            buffer = BytesIO()
            enhanced.save(
                buffer,
                format="JPEG",
                quality=85,
                optimize=True,
            )
            variants.append(
                "data:image/jpeg;base64,"
                + base64.b64encode(buffer.getvalue()).decode("utf-8")
            )

        return variants


def _locate_size_tag_regions(image_paths):
    """Find likely clothing-size-tag regions before the final size read.

    This is intentionally a separate vision pass. The model only locates
    candidate tag areas; it does not decide the size here. Coordinates are
    normalized 0-1000 so the result can be used to make a high-resolution
    crop locally.
    """
    content = [{
        "type": "text",
        "text": r"""
You are a CLOTHING TAG LOCATOR.

For each supplied clothing photo, find any visible area that could contain
an actual garment size marking.

Look for:
- sewn neck labels
- printed neck labels
- side-seam/care labels
- printed size directly on fabric
- heat-transfer size marks
- numeric or letter sizes
- kids/youth size markings
- woven/printed brand labels or logos (often the same physical label as
  the size marking)

Include a box for a label showing only a brand name with no visible size
too, not just labels that show a size.

Do NOT guess the size.
Do NOT guess the brand.
Do NOT infer anything from garment proportions.

Return ONLY JSON in this exact structure:
{
  "regions": [
    {
      "image": 0,
      "found": true,
      "boxes": [
        [left, top, right, bottom]
      ]
    }
  ]
}

Coordinates are normalized from 0 to 1000.
Include a box only when a tag/size-marking area is actually visible.
Use a generous box around the entire tag, including nearby characters.
If no tag is visible for an image, use:
{"image": 0, "found": false, "boxes": []}
"""
    }]

    # Use up to 8 photos for locating tags. Full-resolution EXIF-corrected
    # versions are enough for localization.
    for image_index, image_path in enumerate(list(image_paths)[:8]):
        try:
            content.append({
                "type": "text",
                "text": f"PHOTO_INDEX={image_index}",
            })
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": image_to_data_url(image_path),
                    "detail": "high",
                },
            })
        except Exception as exc:
            print(
                f"Size-tag locator skipped image {image_index}: {exc}"
            )

    response = vision_chat_completion(
        content,
        temperature=0,
    )

    result = clean_json_response(
        response.choices[0].message.content
    )

    regions = result.get("regions", [])
    if not isinstance(regions, list):
        return []

    cleaned = []

    for region in regions:
        try:
            image_index = int(region.get("image", -1))
        except (TypeError, ValueError):
            continue

        if image_index < 0 or image_index >= min(len(image_paths), 8):
            continue

        boxes = region.get("boxes", [])
        if not isinstance(boxes, list):
            continue

        valid_boxes = []

        for box in boxes[:3]:
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                continue

            try:
                left, top, right, bottom = [
                    float(value) for value in box
                ]
            except (TypeError, ValueError):
                continue

            left = max(0.0, min(1000.0, left))
            top = max(0.0, min(1000.0, top))
            right = max(0.0, min(1000.0, right))
            bottom = max(0.0, min(1000.0, bottom))

            if right <= left or bottom <= top:
                continue

            if (right - left) < 8 or (bottom - top) < 8:
                continue

            valid_boxes.append(
                [left, top, right, bottom]
            )

        if valid_boxes:
            cleaned.append({
                "image": image_index,
                "boxes": valid_boxes,
            })

    return cleaned


def _tag_crop_variants(image_path, box, rotations=(0, 180)):
    """Crop a located tag, enlarge it, and make orientation variants.

    PERFORMANCE FIX: dropped from always rendering 4 rotations to 2
    (upright + upside-down) by default — see _image_variants_for_size for
    the reasoning. This alone roughly halves the size of the reader pass.
    """
    path = Path(image_path)

    with Image.open(path) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGB")

        width, height = image.size

        left, top, right, bottom = box

        left = int(width * left / 1000.0)
        top = int(height * top / 1000.0)
        right = int(width * right / 1000.0)
        bottom = int(height * bottom / 1000.0)

        pad_x = max(30, int((right - left) * 0.35))
        pad_y = max(30, int((bottom - top) * 0.45))

        left = max(0, left - pad_x)
        top = max(0, top - pad_y)
        right = min(width, right + pad_x)
        bottom = min(height, bottom + pad_y)

        crop = image.crop(
            (left, top, right, bottom)
        )

        target_width = max(1200, crop.width * 2)
        target_height = max(900, crop.height * 2)

        scale = min(
            2.5,
            max(
                target_width / max(1, crop.width),
                target_height / max(1, crop.height),
            ),
        )

        crop = crop.resize(
            (
                max(1, int(crop.width * scale)),
                max(1, int(crop.height * scale)),
            ),
            Image.Resampling.LANCZOS,
        )

        variants = []

        for angle in rotations:
            rotated = crop.rotate(
                angle,
                expand=True,
                fillcolor="white",
            ) if angle else crop

            enhanced = ImageEnhance.Contrast(
                rotated
            ).enhance(1.18)

            enhanced = ImageEnhance.Sharpness(
                enhanced
            ).enhance(1.6)

            enhanced = enhanced.filter(
                ImageFilter.UnsharpMask(
                    radius=1.2,
                    percent=125,
                    threshold=3,
                )
            )

            buffer = BytesIO()

            enhanced.save(
                buffer,
                format="JPEG",
                quality=90,
                optimize=True,
            )

            variants.append(
                "data:image/jpeg;base64,"
                + base64.b64encode(
                    buffer.getvalue()
                ).decode("utf-8")
            )

        return variants


def _verify_size_from_tags(image_paths, located_regions=None):
    """Final size-tag reader using tag localization + enlarged rotations.

    Returns (size, confidence, evidence).
    If the actual marking cannot be read, returns ('-', 0, ...).

    PERFORMANCE FIX: previously sent up to 4 rotations per crop plus 4
    rotations of up to 6 full backup photos in ONE request — sometimes
    30+ images. Now sends 2 rotations per crop and caps backup photos at
    2 (also 2 rotations each), which is a large payload reduction while
    still catching upside-down tags, the realistic failure mode.

    located_regions: pass in an already-computed _locate_size_tag_regions()
    result (e.g. from analyze_item(), which also needs it for
    _verify_vintage_from_tags()) to skip a redundant locate call. Only
    computed here when not supplied.
    """
    if located_regions is None:
        located_regions = []

        try:
            located_regions = _locate_size_tag_regions(
                image_paths
            )
        except Exception as locator_error:
            print(
                "Size-tag locator failed: "
                + str(locator_error)
            )

    content = [{
        "type": "text",
        "text": r"""
You are the FINAL SIZE TAG READER for a clothing listing.

Your ONLY job is to read the EXACT SIZE CHARACTERS from the garment.

You will receive:
1. Enlarged/cropped candidate tag images.
2. Two rotations of each candidate crop (upright and upside-down).
3. A couple of full garment photos as backup.

The crop may be sideways or upside-down. Compare all rotations given.

READ THE ACTUAL CHARACTERS.
Examples:
M
L
XL
S
6
8
L/G
L/6
12
14
2T
etc.

STRICT RULES:
- NEVER infer size from garment appearance.
- NEVER infer size from measurements.
- NEVER infer size from brand conventions.
- NEVER infer size from model fit.
- NEVER choose a size because it is "probably" correct.
- NEVER convert a visible number into a letter size.
- NEVER confuse L/G with L/6.
- If the exact characters are not readable, return "-".
- A blurry, tiny, folded, partially hidden, or sideways tag is NOT permission
  to guess.
- If multiple actual size markings exist, use the clearest garment size marking.
- Confidence must reflect whether the exact characters can actually be read.

You will ALSO read the brand from the same crops, if a brand mark is on
them — brand and size are usually printed on the same physical label.

BRAND STRICT RULES:
- NEVER infer brand from garment style, cut, fabric, or aesthetic.
- NEVER guess a brand because it "looks like" that brand.
- Read only an actual printed, woven, or embroidered brand mark.
- If no brand mark is legible on any crop, return "-".
- A blurry, tiny, folded, partially hidden, or sideways label is NOT
  permission to guess the brand.
- If a brand IS legible, quote it exactly as printed.
- Confidence must reflect whether the brand text can actually be read.

CONFIDENCE SCALE:
size_confidence and brand_confidence are INTEGERS from 0 to 100 (a
percentage), NOT 0 to 1. If you clearly and exactly read the size or
brand text, that is high confidence — 90 to 100, NOT "1". Use 0 only
when you found genuinely nothing readable.

Return ONLY JSON:
{
  "size": "-",
  "size_confidence": 0,
  "size_evidence": "",
  "brand": "-",
  "brand_confidence": 0,
  "brand_evidence": ""
}
"""
    }]

    # The first successfully-cropped tag region's upright variant —
    # kept to hand back as the actual evidence image for the Size/
    # Brand hover preview in the review screen, instead of a whole
    # uncropped photo the user has to squint at to find the tag
    # themselves. Most items have exactly one real tag region, so the
    # first crop the model itself was given is, in practice, the tag.
    first_crop_data_url = None

    for region in located_regions:
        image_index = region["image"]

        for box in region["boxes"]:
            try:
                variants = _tag_crop_variants(
                    image_paths[image_index],
                    box,
                )

                if first_crop_data_url is None and variants:
                    first_crop_data_url = variants[0]

                for variant_index, data_url in enumerate(variants):
                    content.append({
                        "type": "text",
                        "text": (
                            f"TAG_CROP image={image_index} "
                            f"rotation={variant_index * 180}°"
                        ),
                    })
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": data_url,
                            "detail": "high",
                        },
                    })

            except Exception as crop_error:
                print(
                    "Tag crop failed: "
                    + str(crop_error)
                )

    # BACKUP: up to 2 full photos, 2 orientations each — catches tags the
    # locator missed without ballooning the request.
    for image_path in list(image_paths)[:2]:
        for variant_index, data_url in enumerate(
            _image_variants_for_size(image_path)
        ):
            content.append({
                "type": "text",
                "text": (
                    f"FULL_PHOTO rotation="
                    f"{variant_index * 180}°"
                ),
            })
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": data_url,
                    "detail": "high",
                },
            })

    response = vision_chat_completion(
        content,
        temperature=0,
    )

    result = clean_json_response(
        response.choices[0].message.content
    )

    size = str(
        result.get(
            "size",
            "-",
        )
        or "-"
    ).strip()

    evidence = str(
        result.get(
            "size_evidence",
            result.get("evidence", ""),
        )
        or ""
    ).strip()

    confidence = _normalize_confidence(
        result.get(
            "size_confidence",
            result.get("confidence", 0),
        )
    )

    if size.lower() in {
        "unknown",
        "none",
        "n/a",
        "not visible",
        "not readable",
        "null",
        "",
    }:
        size = "-"

    if confidence < 70:
        size = "-"
        confidence = 0

        if not evidence:
            evidence = (
                "The exact size marking could not be read "
                "confidently from the tag photos."
            )

    brand = str(
        result.get(
            "brand",
            "-",
        )
        or "-"
    ).strip()

    brand_evidence = str(
        result.get(
            "brand_evidence",
            "",
        )
        or ""
    ).strip()

    brand_confidence = _normalize_confidence(
        result.get(
            "brand_confidence",
            0,
        )
    )

    if brand.lower() in {
        "unknown",
        "none",
        "n/a",
        "not visible",
        "not readable",
        "null",
        "",
    }:
        brand = "-"

    if brand_confidence < 70:
        brand = "-"
        brand_confidence = 0

        if not brand_evidence:
            brand_evidence = (
                "No brand marking could be read confidently "
                "from the tag photos."
            )

    return (
        size,
        confidence,
        evidence,
        brand,
        brand_confidence,
        brand_evidence,
        first_crop_data_url,
    )


_VINTAGE_CLASSIFICATIONS = {
    "vintage",
    "discontinued",
    "vintage_and_discontinued",
    "none",
}


# Aesthetic/era words that may appear in a title. Order is the
# tie-breaker used when trimming down to _TITLE_STYLE_MAX — the
# earlier-listed word wins when one has to be dropped.
_TITLE_STYLE_KEYWORDS = [
    "vintage", "y2k", "fairycore", "coquette", "grunge", "boho",
    "preppy", "gothic", "emo", "punk", "western", "cottagecore",
    "athletic", "streetwear",
]

# Pairs that must never both appear in a title. When both are present,
# the SECOND word of the pair is the one dropped.
_TITLE_STYLE_CONFLICTS = [
    ("vintage", "y2k"),
    ("fairycore", "coquette"),
]

_TITLE_STYLE_MAX = 2


def _find_title_style_words(title):
    """Return [(word, position)] for each style keyword present in
    title as a whole word, in the order it first appears."""
    lower = title.lower()
    found = []

    for word in _TITLE_STYLE_KEYWORDS:
        match = re.search(r"\b" + re.escape(word) + r"\b", lower)
        if match:
            found.append((word, match.start()))

    found.sort(key=lambda pair: pair[1])
    return found


def _strip_title_word(title, word):
    """Remove one whole-word occurrence of `word` from title
    (case-insensitively) and clean up the resulting whitespace."""
    pattern = re.compile(r"\b" + re.escape(word) + r"\b\s*", re.IGNORECASE)
    stripped = pattern.sub("", title, count=1)
    return re.sub(r"\s{2,}", " ", stripped).strip(" -,")


def _enforce_title_style_rules(title):
    """Deterministically guarantee the title-level aesthetic/era rules
    hold, regardless of what the model produced: never "Vintage" +
    "Y2K" together, never "Fairycore" + "Coquette" together, and no
    more than _TITLE_STYLE_MAX style/era words total. Prompts alone
    only make the model likely to follow these rules — this makes it
    certain, since it's applied to every title before it's shown to
    the user (initial generation, the vintage-tag injection, and the
    "regenerate title" repair path all funnel through here).
    """
    title = str(title or "")

    for first, second in _TITLE_STYLE_CONFLICTS:
        present = {word for word, _ in _find_title_style_words(title)}

        if first in present and second in present:
            title = _strip_title_word(title, second)

    style_words = _find_title_style_words(title)

    if len(style_words) > _TITLE_STYLE_MAX:
        for word, _ in style_words[_TITLE_STYLE_MAX:]:
            title = _strip_title_word(title, word)

    title = _ensure_title_seo_suffix(title)

    return title


# ============================================================
# DEPOP TEMPLATE — deterministic title suffix + description assembly.
# The model supplies the variable parts (title's descriptive portion,
# description_bullets, condition, size, hashtags); everything fixed
# about the format is assembled here in code so it's byte-for-byte
# consistent every time, never dependent on the model reliably
# reproducing exact boilerplate wording.
# ============================================================

DEPOP_TITLE_SEO_SUFFIX = "Women's Used Clothing"

DEPOP_STYLED_DISCLAIMER = (
    "Item may be styled, tucked, pinned, or tied for photos. "
    "Accessories not included."
)

DEPOP_RESALE_NOTE = (
    "Note: this is a one-of-a-kind resale piece — once it's sold, "
    "it's gone! See our Returns & Refunds policy for details."
)

DEPOP_FIT_SIZING_FOOTER = (
    "Fit & Sizing: Vintage and Y2K sizing often runs differently than "
    "modern sizing — check out our Size Guide for tips on getting the "
    "right fit. Questions about condition or measurements? See our "
    "FAQ or reach out before you buy."
)


def _ensure_title_seo_suffix(title):
    """Guarantees every title ends with "– Women's Used Clothing" —
    the fixed SEO suffix baked into every listing so the app's two
    target search terms ("womens used clothing" / the aesthetic word
    already covers "womens y2k") are always present, regardless of
    whether the model remembered to add it itself."""
    title = str(title or "").strip()

    if not title:
        return title

    lowered = title.lower().replace("women's", "womens")

    if "womens used clothing" in lowered:
        return title

    return f"{title} – {DEPOP_TITLE_SEO_SUFFIX}"


def _condition_phrase(condition):
    condition = str(condition or "").strip()

    if not condition:
        return "Pre-owned condition."

    return f"{condition.rstrip('.')} pre-owned condition."


def build_depop_description(condition, size, bullet_points, hashtags):
    """Assembles the final description from the model's variable parts
    (condition, size, bullet_points, hashtags) plus fixed boilerplate,
    matching the exact required template:

    <Condition> pre-owned condition.

    Size: <Size>

    • bullet
    • bullet
    ...

    Item may be styled, tucked, pinned, or tied for photos. Accessories
    not included.

    Note: this is a one-of-a-kind resale piece...

    #tag #tag #tag #tag #tag

    Fit & Sizing: ...
    """
    size_text = str(size or "").strip()
    size_line = (
        f"Size: {size_text}"
        if size_text and size_text.lower() != "unknown"
        else "Size: See photos for tag details"
    )

    lines = [_condition_phrase(condition), "", size_line, ""]

    for point in (bullet_points or []):
        point = str(point).strip().lstrip("•-").strip()
        if point:
            lines.append(f"• {point}")

    lines.append("")
    lines.append(DEPOP_STYLED_DISCLAIMER)
    lines.append("")
    lines.append(DEPOP_RESALE_NOTE)

    tag_line = " ".join(
        f"#{str(tag).strip().lstrip('#')}"
        for tag in (hashtags or [])
        if str(tag).strip()
    )
    if tag_line:
        lines.append("")
        lines.append(tag_line)

    lines.append("")
    lines.append(DEPOP_FIT_SIZING_FOOTER)

    return "\n".join(lines)


def apply_vintage_flag(listing, marked_vintage):
    """Force a listing's title (and is_vintage flag) to match a manual
    "Vintage?" checkbox override — used by both the upload-deck slot
    checkbox (set before generation) and the listing-review checkbox
    (set/changed after generation).

    Checking it ON guarantees "Vintage" appears in the title even if
    the dedicated tag-evidence check (_verify_vintage_from_tags)
    didn't independently confirm it. Unchecking it OFF removes
    "Vintage" from the title even if that check had added it. This is
    a manual safety net layered on top of the automatic check, which
    still always runs on its own — not a replacement for it.

    Returns a new listing dict (does not mutate the one passed in).
    """
    listing = dict(listing or {})
    marked_vintage = bool(marked_vintage)

    current_title = str(listing.get("title", "") or "").strip()

    if marked_vintage:
        if current_title and "vintage" not in current_title.lower():
            current_title = f"Vintage {current_title}"
    elif current_title:
        current_title = _strip_title_word(current_title, "vintage")

    listing["title"] = _enforce_title_style_rules(current_title)
    listing["is_vintage"] = marked_vintage

    return listing


# ============================================================
# PERSISTENT VINTAGE-TAG CORRECTION LEARNING
# Learns from the user manually adding/removing the "Vintage" tag
# (via apply_vintage_flag()) over time, so a brand/garment/title
# pattern the user keeps correcting gradually nudges future
# _verify_vintage_from_tags() calls. This is a soft calibration hint
# only, never a hard override — real tag evidence in the photos still
# has to decide each individual item; a pattern of past corrections
# is a tiebreaker for genuinely borderline cases, not a way to force a
# read the photos don't support (matches the "don't be so strict...
# but don't be crazy" balance from the vintage-bar loosening above).
# ============================================================

VINTAGE_CORRECTIONS_FILE = Path(__file__).with_name("vintage_corrections.json")
_VINTAGE_CORRECTIONS_MAX = 3000

# A bucket needs at least this many corrections before it's trusted at
# all, and the lean toward one direction must be at least this
# consistent — otherwise a single stray click could bias future items.
_VINTAGE_BIAS_MIN_SAMPLES = 2
_VINTAGE_BIAS_MIN_LEAN = 0.65

_VINTAGE_TITLE_STOPWORDS = {
    "vintage", "y2k", "fairycore", "coquette", "cute", "womens",
    "women", "woman", "girls", "girl", "size", "new", "rare",
    "authentic", "excellent", "condition", "with", "the", "and",
    "for", "top", "shirt", "bottom", "clothing",
}


def _load_vintage_corrections():
    if not VINTAGE_CORRECTIONS_FILE.exists():
        return []
    try:
        data = json.loads(VINTAGE_CORRECTIONS_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save_vintage_corrections(entries):
    try:
        VINTAGE_CORRECTIONS_FILE.write_text(
            json.dumps(entries[-_VINTAGE_CORRECTIONS_MAX:], indent=2),
            encoding="utf-8",
        )
    except Exception:
        # Correction learning should never break listing generation.
        pass


def _vintage_title_signature(title):
    """A few useful, non-generic keywords from a title — not the whole
    title, since every title is unique and wouldn't bucket anything."""
    tokens = re.findall(r"[a-z0-9]+", str(title or "").lower())
    useful = [
        token for token in tokens
        if len(token) >= 4 and token not in _VINTAGE_TITLE_STOPWORDS
    ]
    return "|".join(useful[:3])


def _vintage_bucket_keys(listing):
    """Most-specific -> broadest fallback bucket keys for this listing,
    mirroring the pricing profile's bucketing approach in app.py."""
    brand = str(listing.get("brand", "") or "").strip().lower() or "unknown"
    garment = str(
        listing.get("garment_type", listing.get("type", "")) or ""
    ).strip().lower() or "unknown"
    title_sig = _vintage_title_signature(listing.get("title", ""))

    keys = [
        f"brand={brand}|garment={garment}",
        f"garment={garment}",
        f"brand={brand}",
    ]

    if title_sig:
        keys.insert(0, f"brand={brand}|garment={garment}|title={title_sig}")

    return keys


def record_vintage_correction(listing, from_state, to_state, ai_auto_verdict):
    """Log one manual vintage-tag change for future calibration.

    Called from the upload-deck "Vintage?" checkbox (only when it
    actually overrides what the AI's own tag check concluded) and the
    listing-review "Vintage item" checkbox (every toggle, either
    direction). Never raises — a logging failure must not break the
    listing-review flow.
    """
    try:
        entries = _load_vintage_corrections()
        entries.append({
            "timestamp": time.time(),
            "brand": str(listing.get("brand", "") or ""),
            "garment_type": str(
                listing.get("garment_type", listing.get("type", "")) or ""
            ),
            "bucket_keys": _vintage_bucket_keys(listing),
            "ai_auto_verdict": bool(ai_auto_verdict),
            "from_state": bool(from_state),
            "to_state": bool(to_state),
        })
        _save_vintage_corrections(entries)
    except Exception:
        pass


def _vintage_correction_bias(listing):
    """Return (lean, sample_count, bucket_key) from correction history.

    Checks bucket keys most-specific first; the first bucket that
    clears _VINTAGE_BIAS_MIN_SAMPLES and leans at least
    _VINTAGE_BIAS_MIN_LEAN toward one direction wins. lean is
    "vintage", "non_vintage", or None if nothing qualifies.
    """
    entries = _load_vintage_corrections()
    if not entries:
        return None, 0, None

    for key in _vintage_bucket_keys(listing):
        to_vintage = 0
        to_non_vintage = 0

        for entry in entries:
            if key not in entry.get("bucket_keys", []):
                continue
            if entry.get("to_state"):
                to_vintage += 1
            else:
                to_non_vintage += 1

        total = to_vintage + to_non_vintage
        if total < _VINTAGE_BIAS_MIN_SAMPLES:
            continue

        if to_vintage / total >= _VINTAGE_BIAS_MIN_LEAN:
            return "vintage", total, key
        if to_non_vintage / total >= _VINTAGE_BIAS_MIN_LEAN:
            return "non_vintage", total, key

    return None, 0, None


def _vintage_correction_bias_hint(listing):
    """Build the prompt text block for _verify_vintage_from_tags(), or
    "" when there's no sufficiently consistent correction history to
    say anything useful yet."""
    try:
        lean, count, _key = _vintage_correction_bias(listing)
    except Exception:
        return ""

    if lean is None:
        return ""

    label = "VINTAGE" if lean == "vintage" else "NOT VINTAGE"

    return f"""

============================================================
CALIBRATION NOTE (context only)
============================================================
Across {count} previous manual correction(s) on similar items from
this reseller (same brand/garment type), they have consistently
leaned toward marking items like this {label}.

The tag photos still decide — do NOT fabricate or stretch evidence to
match this note. If the tag evidence here is genuinely borderline or
ambiguous, let this history break the tie toward {label}. If the tag
evidence clearly points the other way, trust the tag evidence instead
— this is a tiebreaker for close calls, not an override.
"""


def _verify_vintage_from_tags(image_paths, located_regions=None, bias_hint=""):
    """Dedicated vintage/discontinued-era tag reader.

    Reuses the same tag locate+crop pipeline as _verify_size_from_tags()
    (_locate_size_tag_regions() / _tag_crop_variants()) instead of doing
    its own localization pass, but sends its own dedicated reading
    prompt — this classification task (RN numbers, era-specific brand
    logo knowledge, discontinued sub-line detection) is big and precise
    enough on its own that folding it into the already-large size+brand
    prompt would dilute both instructions.

    located_regions: pass in an already-computed _locate_size_tag_regions()
    result (analyze_item() shares ONE locate call between this and
    _verify_size_from_tags() — since this now runs on every item, not
    sharing it would double the locate calls on any item that also
    needs size/brand verification). Only computed here when not
    supplied.

    Returns a dict with keys: brand, tag_era_signal,
    discontinued_element, estimated_era, confidence, rn_number,
    listing_classification, is_vintage, suggested_listing_tags,
    evidence_notes. Classification defaults to "none" (not vintage) on
    any parse failure — never force a vintage read.
    """
    if located_regions is None:
        located_regions = []

        try:
            located_regions = _locate_size_tag_regions(
                image_paths
            )
        except Exception as locator_error:
            print(
                "Vintage-tag locator failed: "
                + str(locator_error)
            )

    content = [{
        "type": "text",
        "text": r"""
You are inspecting clothing tag/label photos to identify vintage or
discontinued-era indicators. Analyze the tag image(s) and report ONLY
what is visually confirmable — do not guess an era from wear alone.

Check for these signals:

1. BRAND LOGO/TAG VERSION CHANGES
   Identify the specific tag design and match it to the brand's known
   era if recognizable, e.g.:
   - Hollister: seagull logo style, "Hollister Co." wordmark placement,
     tag shape/color (early 2000s tags differ from post-2015 redesigns)
   - Abercrombie & Fitch: moose logo variants, "Abercrombie & Fitch"
     vs "A&F" branding, discontinued sub-lines (Abercrombie Kids vs
     A&F Fierce era tags)
   - American Eagle, Aeropostale, PacSun, Urban Outfitters, Brandy
     Melville: any retired logo, retired sub-brand, or tag redesign
   - Any brand tag showing a DISCONTINUED sub-label, collab, or line
     name that no longer exists (flag this explicitly)

2. TAG CONSTRUCTION DETAILS
   - RN number (Registered Number) — report the exact number if legible
   - Care label format: symbol-only vs symbol+text, language count
   - Union label presence ("Made in USA" union tag)
   - Tag material and stitching (woven vs printed satin, fraying
     pattern consistent with age vs. artificial distressing)
   - Country of manufacture printed on tag

3. SIZE TAG FORMAT
   - Old vs current sizing label layout/font for that brand
   - Any size tag format the brand no longer uses

4. CLASSIFICATION (apply after steps 1-3)
   Use this rule to decide the listing tag:
   - "vintage": tag evidence places manufacture at roughly 15+ years
     before today — this is the RESALE-MARKET vintage bar (mid-2000s
     through early-2010s or earlier for most mall brands), NOT a
     strict 20+ year antique bar. A single, clearly-read era signal is
     enough on its own — you do NOT need every signal in steps 1-3 to
     agree. In particular, a recognizably retired tag/logo DESIGN is
     sufficient evidence by itself, even with no legible RN number or
     union label:
       - Hollister: the old seagull-logo tag / pre-2011ish wordmark
         layout (before the brand's later minimalist redesign) counts
         as vintage on its own
       - Abercrombie & Fitch: an old moose-tag design or a retired
         sub-line (e.g. Fierce-era, Abercrombie Kids-era) tag counts
       - American Eagle, Aeropostale, PacSun, Urban Outfitters, Brandy
         Melville: any tag design the brand visibly no longer uses
         counts the same way
   - "discontinued": tag shows a retired sub-brand, collab, or logo
     version that no longer exists, but doesn't meet the vintage bar
     above
   - "vintage_and_discontinued": both conditions independently met
   - "none": no tag evidence supports either — do not force a read.
     A current/modern tag design still counts as "none" even on a
     well-worn, faded, or distressed-looking garment.

OUTPUT FORMAT (JSON):
{
  "brand": "",
  "tag_era_signal": "",
  "discontinued_element": "",
  "estimated_era": "",
  "confidence": "",
  "rn_number": "",
  "listing_classification": "",
  "is_vintage": false,
  "suggested_listing_tags": [],
  "evidence_notes": ""
}

Do NOT infer vintage status purely from fabric wear, fading, or
distressing — that can be artificial or unrelated to age. Only mark
is_vintage true when tag-based evidence (an actual retired tag/logo
design, RN number range, union label, or era-specific construction)
supports the vintage bar above. Stay evidence-based in both
directions: don't require 20+ years like before, but also don't call
a current/modern tag design vintage just because the garment looks or
feels old — if the tag itself reads as a current design, set
listing_classification to "none".
""" + bias_hint
    }]

    for region in located_regions:
        image_index = region["image"]

        for box in region["boxes"]:
            try:
                variants = _tag_crop_variants(
                    image_paths[image_index],
                    box,
                )

                for variant_index, data_url in enumerate(variants):
                    content.append({
                        "type": "text",
                        "text": (
                            f"TAG_CROP image={image_index} "
                            f"rotation={variant_index * 180}°"
                        ),
                    })
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": data_url,
                            "detail": "high",
                        },
                    })

            except Exception as crop_error:
                print(
                    "Vintage tag crop failed: "
                    + str(crop_error)
                )

    # BACKUP: up to 2 full photos — catches tags the locator missed,
    # same fallback _verify_size_from_tags() uses.
    for image_path in list(image_paths)[:2]:
        for variant_index, data_url in enumerate(
            _image_variants_for_size(image_path)
        ):
            content.append({
                "type": "text",
                "text": (
                    f"FULL_PHOTO rotation="
                    f"{variant_index * 180}°"
                ),
            })
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": data_url,
                    "detail": "high",
                },
            })

    response = vision_chat_completion(
        content,
        temperature=0,
    )

    result = clean_json_response(
        response.choices[0].message.content
    )

    classification = str(
        result.get("listing_classification", "none") or "none"
    ).strip().lower()

    if classification not in _VINTAGE_CLASSIFICATIONS:
        classification = "none"

    # Defensive consistency check — is_vintage must agree with the
    # classification, in case the model returns a contradictory pair
    # (e.g. is_vintage=true with listing_classification="discontinued").
    is_vintage = bool(
        result.get("is_vintage")
    ) and classification in {
        "vintage",
        "vintage_and_discontinued",
    }

    suggested_tags = result.get("suggested_listing_tags", [])
    if not isinstance(suggested_tags, list):
        suggested_tags = []
    suggested_tags = [
        str(tag).strip() for tag in suggested_tags if str(tag).strip()
    ]

    return {
        "brand": str(result.get("brand", "") or "").strip(),
        "tag_era_signal": str(
            result.get("tag_era_signal", "") or ""
        ).strip(),
        "discontinued_element": str(
            result.get("discontinued_element", "") or ""
        ).strip(),
        "estimated_era": str(
            result.get("estimated_era", "") or ""
        ).strip(),
        "confidence": str(result.get("confidence", "") or "").strip(),
        "rn_number": str(result.get("rn_number", "") or "").strip(),
        "listing_classification": classification,
        "is_vintage": is_vintage,
        "suggested_listing_tags": suggested_tags,
        "evidence_notes": str(
            result.get("evidence_notes", "") or ""
        ).strip(),
    }


# ============================================================
# AUTO-STRAIGHTEN — BATCHED ROTATION DETECTION
# ============================================================
# Runs as its own batched pass rather than folding into grouping.py's
# classify_photos(): that call uses detail:"low" thumbnails tuned for
# a much easier complete-vs-detail judgment, and mixing schemas would
# let its fail-open default mask a missed rotation too. Uses
# vision_chat_completion()'s multi-model fallback chain since this is
# a brand-new call site with no reason to build on a weaker wrapper.
#
# REWRITE: the original approach asked the model to look at ONE
# unrotated photo and reason abstractly about which way is "up" —
# essentially a blind single-shot guess. Live testing confirmed this
# was unreliable (under-flagged photos that needed correction, and
# got the direction backwards on some it did flag). This mirrors what
# _verify_size_from_tags() already does successfully for tag
# orientation: generate the actual rotated CANDIDATES and have the
# model directly COMPARE them side by side to pick the upright one,
# rather than reasoning blind about a single image. Reuses
# _image_variants_for_size() to build the 4 candidates per photo —
# same rotation/enhance/encode pipeline already proven there.
#
# Cost tradeoff: 4 images per photo instead of 1. Batch of 3 still
# left some photos sideways after two rounds of fixes (missing-entry
# retry, direction-math fix) — the remaining failures are plausibly
# the model cross-confusing which candidate set belongs to which
# photo when several are compared in one call. Dropped to 1 photo per
# call: zero cross-photo confusion possible, at the cost of one call
# per photo instead of one call per 3. Deliberate accuracy-over-cost
# call given this is the second round of "still sideways" reports.

_ROTATION_BATCH = 1

# PIL's Image.rotate(angle) is COUNTERCLOCKWISE-positive — the exact
# opposite of "clockwise", which is what the prompt below describes to
# the model and what _rotate_photo_in_place()'s degrees_clockwise
# parameter expects (it internally negates: .rotate(-degrees_clockwise)).
# Passing (0, 90, 180, 270) straight through built B/D backwards from
# their labels (confirmed live: photos came out upside-down — a 180°
# error, which is exactly what a 90/270 swap produces). Negate the 90/
# 270 entries here so each variant is actually built as TRUE clockwise
# rotation matching its letter, matching the prompt, matching what gets
# applied later. 180 is direction-agnostic so it's unaffected either way.
_ROTATION_VARIANT_ANGLES = (0, -90, 180, -270)
_ROTATION_VARIANT_LETTERS = ("A", "B", "C", "D")
_ROTATION_LETTER_TO_DEGREES = {"A": 0, "B": 90, "C": 180, "D": 270}


# Rotation-only calls are lighter than a full analyze_item() pass (no
# JSON schema to fill, far less output), so more can run in flight at
# once without the TPM pressure that capped Generate's own worker
# count at 3 — bumped for the "orientation checking should be almost
# instant" ask.
_ROTATION_WORKERS = 6

# Judging overall garment orientation (which way is up, do hems run
# horizontal) doesn't need the same resolution size-tag text-reading
# does — smaller renders mean fewer high-detail tiles per candidate,
# which is most of the per-photo latency given 4 candidates are sent
# every time.
_ROTATION_MAX_DIM = 900


def _rotation_batch_prompt(batch_start, batch_paths):
    content = [{
        "type": "text",
        "text": r"""
You are checking clothing photos for orientation.

For EACH photo below, you are shown 4 candidate rotations of the SAME
photo, labeled A, B, C, D:
A = the photo as originally taken (0 degrees)
B = the photo rotated 90 degrees clockwise from original
C = the photo rotated 180 degrees from original
D = the photo rotated 270 degrees clockwise from original

Compare the 4 candidates for each photo SIDE BY SIDE and identify
which ONE is actually upright: garment right-side up, any visible
text reading normally left-to-right, hems/waistbands/shoulders
roughly horizontal.

Do not reason abstractly about directions or guess blind — look at
the 4 actual images provided and pick the one that visibly looks
correct. If A (the original) already looks correct, say so — that is
the most common case.

Judge each photo independently. Return ONLY JSON:
{
  "photos": [
    {"photo_number": 1, "correct_variant": "A"}
  ]
}
""",
    }]

    for offset, image_path in enumerate(batch_paths):
        photo_number = batch_start + offset + 1
        try:
            variants = _image_variants_for_size(
                image_path,
                rotations=_ROTATION_VARIANT_ANGLES,
                max_dim=_ROTATION_MAX_DIM,
            )
        except Exception as exc:
            print(
                f"Rotation check skipped photo {photo_number}: {exc}"
            )
            continue

        content.append({
            "type": "text",
            "text": f"PHOTO {photo_number}: {Path(image_path).name}",
        })

        for letter, data_url in zip(_ROTATION_VARIANT_LETTERS, variants):
            content.append({
                "type": "text",
                "text": f"PHOTO {photo_number} VARIANT {letter}",
            })
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": data_url,
                    "detail": "high",
                },
            })

    return content


def _process_rotation_batch(batch_start, batch_paths):
    """Run one batch's vision call and return (rotations, seen_numbers)
    for that batch alone — the piece of work each worker thread runs.
    """
    content = _rotation_batch_prompt(batch_start, batch_paths)

    # One retry on top of vision_chat_completion()'s own internal
    # model-chain retries — a whole batch failing after exhausting
    # every model is rare but not impossible, and a batch that's
    # silently skipped here means every photo in it just never gets
    # straightened (no error the user sees; live testing found the
    # FIRST batch of a run skipped this way while later ones
    # succeeded — a first-call fluke, plausibly rate-limit related).
    result = None
    last_batch_error = None

    for attempt in range(2):
        try:
            response = vision_chat_completion(
                content,
                temperature=0,
                response_format={"type": "json_object"},
            )
            result = clean_json_response(
                response.choices[0].message.content
            )
            break
        except Exception as exc:
            last_batch_error = exc

    batch_rotations = {}
    batch_seen = set()

    if result is None:
        print(
            f"Rotation batch {batch_start}-"
            f"{batch_start + len(batch_paths)} skipped "
            f"after retry: {last_batch_error}"
        )
        return batch_rotations, batch_seen

    photos = result.get("photos", [])
    if not isinstance(photos, list):
        return batch_rotations, batch_seen

    for entry in photos:
        if not isinstance(entry, dict):
            continue

        try:
            photo_number = int(entry.get("photo_number", -1))
        except (TypeError, ValueError):
            continue

        if not (batch_start + 1 <= photo_number <= batch_start + len(batch_paths)):
            continue

        batch_seen.add(photo_number)

        letter = str(
            entry.get("correct_variant", "A") or "A"
        ).strip().upper()
        degrees = _ROTATION_LETTER_TO_DEGREES.get(letter, 0)

        if degrees:
            batch_rotations[photo_number - 1] = degrees

    return batch_rotations, batch_seen


def detect_photo_rotations(image_paths, _is_retry_pass=False, on_progress=None):
    """Detect the clockwise rotation (0/90/180/270) needed to straighten
    each photo, by generating all 4 rotation candidates and asking the
    model to identify which one is actually upright by comparison.
    Returns {photo_index: degrees_clockwise} for photos that need
    correction — 0-rotation and unreadable photos are simply absent.

    photo_index matches the position of the photo in image_paths (0-based).

    A whole BATCH failing is retried (see below), but a batch can also
    succeed while the model just never returns an entry for one photo
    in it — that photo silently defaults to "no rotation needed" and
    stays sideways with no error anywhere. seen_numbers tracks which
    photos actually got judged (any letter, including "A"); anything
    missing after the full sweep gets one targeted retry pass.

    PERFORMANCE: with _ROTATION_BATCH=1 (one photo per call, kept for
    accuracy — see the comment above _ROTATION_BATCH), a big upload
    used to mean dozens of sequential round trips before "Generate All
    Listings" showed any progress at all. Batches now run concurrently
    across _ROTATION_WORKERS threads instead of one at a time.

    on_progress(completed, total), if given, is called after each
    batch finishes — lets a caller (e.g. app.py) drive a progress bar
    during this phase without this module needing to know about
    Streamlit at all.
    """
    image_paths = list(image_paths)
    rotations = {}
    seen_numbers = set()

    batch_starts = list(
        range(0, len(image_paths), _ROTATION_BATCH)
    )
    total_batches = len(batch_starts)
    completed_batches = 0

    if total_batches:
        worker_count = min(_ROTATION_WORKERS, total_batches)

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {
                executor.submit(
                    _process_rotation_batch,
                    batch_start,
                    image_paths[batch_start:batch_start + _ROTATION_BATCH],
                ): batch_start
                for batch_start in batch_starts
            }

            for future in as_completed(future_map):
                try:
                    batch_rotations, batch_seen = future.result()
                except Exception as exc:
                    print(f"Rotation batch failed: {exc}")
                    batch_rotations, batch_seen = {}, set()

                rotations.update(batch_rotations)
                seen_numbers.update(batch_seen)

                completed_batches += 1
                if on_progress is not None:
                    try:
                        on_progress(completed_batches, total_batches)
                    except Exception:
                        pass

    if not _is_retry_pass:
        missing_indices = [
            i for i in range(len(image_paths))
            if (i + 1) not in seen_numbers
        ]

        if missing_indices:
            print(
                f"[ROTATION] {len(missing_indices)} photo(s) got no "
                f"response in the first pass — retrying individually."
            )
            missing_paths = [image_paths[i] for i in missing_indices]
            retry_rotations = detect_photo_rotations(
                missing_paths,
                _is_retry_pass=True,
            )
            for local_index, degrees in retry_rotations.items():
                rotations[missing_indices[local_index]] = degrees

    return rotations


def rotate_photo_in_place(path, degrees_clockwise=90):
    """Bake a permanent clockwise rotation into a photo on disk.

    Shared by the manual per-photo rotate button, the "Generate All
    Listings" batched auto-straighten pass, and the Final AI QA
    rotation safety net — same EXIF-bake, mode-fix, per-suffix save.
    Lives here (not app.py) since it has no Streamlit dependency and
    qa_review.py needs it too, without creating an app.py<->qa_review.py
    import cycle.
    """
    image_path = Path(path)

    with Image.open(image_path) as image:
        corrected = ImageOps.exif_transpose(image)

        rotated = corrected.rotate(
            -degrees_clockwise,
            expand=True,
        )

        if rotated.mode not in {"RGB", "L"}:
            rotated = rotated.convert("RGB")

        suffix = image_path.suffix.lower()

        if suffix in {".jpg", ".jpeg"}:
            rotated.save(
                image_path,
                format="JPEG",
                quality=95,
                optimize=True,
                exif=b"",
            )
        elif suffix == ".png":
            rotated.save(
                image_path,
                format="PNG",
            )
        elif suffix == ".webp":
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


# ============================================================
# CLEAN AI JSON
# ============================================================

def clean_json_response(text):

    if not text:
        raise ValueError(
            "OpenAI returned an empty response."
        )

    text = text.strip()

    if text.startswith("```"):

        text = re.sub(
            r"^```(?:json)?\s*",
            "",
            text,
            flags=re.IGNORECASE
        )

        text = re.sub(
            r"\s*```$",
            "",
            text
        )

        text = text.strip()

    try:

        return json.loads(text)

    except json.JSONDecodeError:

        start = text.find("{")
        end = text.rfind("}")

        if start != -1 and end != -1:

            return json.loads(
                text[start:end + 1]
            )

        raise ValueError(
            "OpenAI returned invalid JSON:\n"
            + text
        )


# ============================================================
# CLEAN HASHTAGS
# ============================================================

def clean_hashtags(
    hashtags
):

    if isinstance(
        hashtags,
        str
    ):

        hashtags = hashtags.split()


    if not isinstance(
        hashtags,
        list
    ):

        return []


    cleaned = []


    for hashtag in hashtags:

        hashtag = str(
            hashtag
        ).strip()


        if not hashtag:
            continue


        if not hashtag.startswith("#"):

            hashtag = (
                "#" + hashtag
            )


        hashtag = hashtag.replace(
            " ",
            ""
        )


        if hashtag.lower() in {
            "#womenswear",
            "#women'swear",
            "#clothing",
            "#fashion",
            "#style",
            "#outfit",
            "#ootd"
        }:

            continue


        duplicate = any(
            hashtag.lower()
            == existing.lower()
            for existing in cleaned
        )


        if not duplicate:

            cleaned.append(
                hashtag
            )


    return cleaned[:5]


# ============================================================
# CLEAN PRICE
# ============================================================

def clean_price(
    price
):

    try:

        price = float(
            price
        )

    except (
        ValueError,
        TypeError
    ):

        price = 10


    price = max(
        7,
        min(
            price,
            20
        )
    )

    # Whole dollars only — no cents.
    return round(price)


# ============================================================
# CONFIDENCE SCORE NORMALIZATION
# ============================================================
# The model is asked for a 0-100 confidence score, but sometimes answers
# on a 0-1 scale instead (0/1 as "no/yes fully confident") despite the
# prompt. Caught via [SIZE TRACE]/[BRAND TRACE] debugging: real, specific,
# correctly-read values (e.g. brand="Silver Jeans Co.") were arriving
# with confidence=1, which every >=50/>=70 gate then treated as "1%
# confident" and threw away. A confidence of 0 or 1 is never a
# meaningful 0-100 percentage for a field the model actually populated
# with a real value, so treat those two values as the 0-1 scale and
# rescale — this is a safety net independent of prompt wording.
def _normalize_confidence(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0

    if 0 <= value <= 1:
        value *= 100

    return int(max(0, min(100, round(value))))


# ============================================================
# SIZE VALIDATION
# ============================================================

def validate_size(
    listing
):

    size = str(
        listing.get(
            "size",
            "Unknown"
        )
    ).strip()


    confidence = _normalize_confidence(
        listing.get(
            "size_confidence",
            0
        )
    )


    evidence = str(
        listing.get(
            "size_evidence",
            ""
        )
    ).strip()


    if not size:

        return (
            "-",
            0,
            "No readable size information is visible "
            "in the provided photos."
        )


    if size.lower() in {
        "-",
        "unknown",
        "none",
        "n/a",
        "not visible",
        "not readable"
    }:

        return (
            "-",
            0,
            "No readable size information is visible "
            "in the provided photos."
        )


    evidence_lower = (
        evidence.lower()
    )


    guess_phrases = [
        "looks like",
        "looks to be",
        "appears to be",
        "appears like",
        "seems like",
        "seems to be",
        "probably",
        "likely",
        "estimated",
        "estimate",
        "approximately",
        "based on fit",
        "based on appearance",
        "visual estimate"
    ]


    if any(
        phrase in evidence_lower
        for phrase in guess_phrases
    ):

        return (
            "-",
            0,
            "No readable size marking is visible; "
            "size cannot be inferred from appearance."
        )


    if not evidence:

        return (
            "-",
            0,
            "No readable size information is visible "
            "in the provided photos."
        )


    if confidence < 50:

        return (
            "-",
            0,
            "The size marking is not clear enough "
            "to confidently identify the exact size."
        )


    return (
        size,
        confidence,
        evidence
    )


def validate_brand(
    listing
):

    brand = str(
        listing.get(
            "brand",
            ""
        )
    ).strip()


    confidence = _normalize_confidence(
        listing.get(
            "brand_confidence",
            0
        )
    )


    evidence = str(
        listing.get(
            "brand_evidence",
            ""
        )
    ).strip()


    if not brand or brand.lower() in {
        "-",
        "unknown",
        "none",
        "n/a",
        "not visible",
        "not readable"
    }:

        return (
            "",
            0,
            "No brand marking is visible in the provided photos."
        )


    evidence_lower = (
        evidence.lower()
    )


    guess_phrases = [
        "looks like",
        "looks to be",
        "appears to be",
        "appears like",
        "seems like",
        "seems to be",
        "probably",
        "likely",
        "estimated",
        "estimate",
        "approximately",
        "based on style",
        "based on aesthetic",
        "based on appearance",
        "visual estimate"
    ]


    if any(
        phrase in evidence_lower
        for phrase in guess_phrases
    ):

        return (
            "",
            0,
            "No brand marking is visible; brand cannot be "
            "inferred from appearance."
        )


    if confidence < 50:

        return (
            "",
            0,
            "The brand marking is not clear enough "
            "to confidently identify the brand."
        )


    return (
        brand,
        confidence,
        evidence
    )


# ============================================================
# INITIAL LISTING GENERATION
# ============================================================

def analyze_item(
    image_paths
):

    prompt = r"""
You are an expert women's clothing reseller specializing
in Y2K, vintage, coquette, fairycore, 2000s mall brands,
and women's resale clothing.

Analyze ALL photos as ONE physical clothing item.

Return ONLY valid JSON.

============================================================
SIZE
============================================================

Size accuracy is extremely important.

Inspect EVERY photo carefully for:

- sewn neck tags
- printed neck tags
- printed size labels
- heat-transfer size markings
- screen-printed size markings
- side seam tags
- care labels
- stickers
- paper tags
- numerical sizes
- letter sizes
- kids/youth sizes

A PRINTED size counts.

If the fabric itself clearly says M,
return M.

If it clearly says L,
return L.

If it clearly says L/G,
return L/G.

If it clearly says L/6,
return L/6.

Do NOT confuse:

L/G

with:

L/6

Read the actual characters.

NEVER infer size from:

- how large the garment looks
- model fit
- garment dimensions
- sleeve length
- chest width
- brand's normal sizing
- visual proportions

If no readable size marking exists:

size = "-"

size_confidence = 0

size_evidence =
"No readable size information is visible in the provided photos."

If a printed size is visible, explicitly say what it reads.

Do not say "looks like M."

Do not say "probably M."

============================================================
BRAND
============================================================

Read the actual brand label whenever possible.

Never guess a brand based only on appearance or aesthetic.

NEVER infer brand from:

- garment style or aesthetic
- fabric quality alone
- how the garment "looks like" a certain brand
- typical brands for this type of item

If no brand marking is visible or legible:

brand = ""

brand_confidence = 0

brand_evidence =
"No brand marking is visible in the provided photos."

If a brand IS legible, quote it exactly as printed and explain where
it was read (e.g. "printed on neck label").

============================================================
GARMENT
============================================================

Identify the garment as specifically as possible.

For tops distinguish between:

tank top
cami
baby tee
short sleeve
long sleeve
tube top
halter
polo
blouse
cardigan
sweater
corset-style top
peplum top

Identify visible details such as:

lace
embroidery
graphics
logos
buttons
zippers
ribbing
mesh
satin
knit
trim
hardware
straps
neckline
sleeves
ruffles
peplum
floral print
bows

============================================================
GARMENT TYPE
============================================================

Set "garment_type" to EXACTLY ONE of these strings — copy it exactly,
do not paraphrase or invent a new one:

Tank Top, Cami, Crop Top, Baby Tee, T-Shirt, Long Sleeve Top, Blouse,
Button Up, Polo, Halter Top, Tube Top, Corset Top, Bodysuit, Cardigan,
Sweater, Hoodie, Sweatshirt, Tunic, Jacket, Coat, Vest, Dress, Skirt,
Pants, Jeans, Shorts, Leggings, Jumpsuit, Romper, Other

Pick the single closest match to the actual garment shown (a pair of
shorts is "Shorts", not "Other" — only use "Other" when the garment
genuinely doesn't resemble anything in the list).

============================================================
CATEGORY
============================================================

A list of the EXACT valid Shopify category strings follows this prompt
as a separate message. You MUST copy one of those strings character
for character into "category" — do not paraphrase it, shorten it,
change capitalization, or invent a category that isn't in the list.

Pick the single most specific matching category (e.g. prefer
"Clothing Tops > T-Shirts" over the broader "Clothing Tops" when the
garment is clearly a t-shirt).

If genuinely nothing in the list fits, return "" — do not guess a
close-but-wrong category just to fill the field.

============================================================
AESTHETIC
============================================================

Only identify aesthetics genuinely supported by the item.

Possible:

Y2K
vintage
fairycore
coquette
grunge
boho
preppy
gothic
emo
punk
western
cottagecore
athletic
streetwear

Do not automatically call every item Y2K.

Do not automatically call every item vintage.

============================================================
TITLE
============================================================

The title is the single most important searchable field. It is
SEPARATE from the description now (no longer the description's first
line) — return it as its own "title" field.

Format EXACTLY like this pattern (a real example):

Y2K Merona dress – Women's Used Clothing

That is: [aesthetic/era word if justified] [brand] [garment
type/standout detail] – Women's Used Clothing

Include, in a natural order, before the fixed "– Women's Used
Clothing" suffix:

- Any genuinely justified aesthetic/era word(s) from the AESTHETIC
  list above
- Brand (if known)
- Primary color (when it adds real search value)
- Garment type
- One standout visible detail if it helps buyers search (e.g. "Lace
  Trim", "Cargo", "Corset")

Do NOT write the "– Women's Used Clothing" suffix yourself — it is
appended automatically. Just write the descriptive part before it.

Aesthetic/era words in the title — these three rules are absolute:

1. NEVER include both "Vintage" and "Y2K" in the same title — they
   describe conflicting eras. If both are genuinely supported, use
   "Vintage" only (it's the stronger, more specific claim).
2. NEVER include both "Fairycore" and "Coquette" in the same title —
   pick whichever one fits the item better.
3. NEVER include more than 2 aesthetic/era words in the title total,
   even if more than 2 are genuinely justified by the item.

Keep the title concise and natural — do not force in every eligible
word just because it's technically justified.

============================================================
CONDITION
============================================================

Inspect every photo for actual visible flaws.

Look for:

stains
holes
tears
pilling
fading
discoloration
stretching
loose stitching
damaged graphics
missing buttons
broken zippers

Do not assume secondhand clothing has flaws.

If there are no visible flaws:

condition = "Excellent"

condition_evidence =
"No visible stains, holes, damage, or other flaws in the provided photos."

============================================================
DESCRIPTION_BULLETS
============================================================

The full description is assembled automatically from a fixed
template (condition line, size line, your bullets, a fixed styling
disclaimer, a fixed resale note, hashtags, and a fixed fit/sizing
footer) — you do NOT write the full description text. Your only job
here is "description_bullets": a JSON array of short bullet-point
strings describing the item, in this order:

1. Brand (if known) — just the brand name, e.g. "Merona"
2. Color/pattern — e.g. "Black, white, gray, and pink striped pattern"
3. One bullet per other genuinely visible, useful detail: silhouette/
   design (e.g. "Sleeveless design"), waist/fit construction (e.g.
   "Elastic cinched waist"), fabric/feel (e.g. "Soft stretch knit
   fabric"), overall fit description (e.g. "Flattering fit and flare
   silhouette"), length (e.g. "Knee-length"), and a closing wearability
   line (e.g. "Lightweight and comfortable for everyday wear")

Rules:

- 6-9 bullets total, each SHORT (a few words to one short clause) —
  not full sentences, not paragraphs.
- Only include details you can actually see or reasonably infer from
  the photos. Do not invent fabric content, fit, or features.
- Do not repeat the brand or color in multiple bullets.
- Do not use the word "fitted".
- Do not include emojis in the bullets.
- Do not include the condition or size here — those come from their
  own fields.
- Sound like a real Depop seller's quick specs list, not generic AI
  marketing copy (no "perfect addition to your wardrobe", no
  "elevate your wardrobe").

============================================================
HASHTAGS
============================================================

Return EXACTLY 5 hashtags.

Every hashtag must be relevant to the actual item.

Use:

- actual aesthetic
- actual garment type
- actual visible detail
- brand when useful
- strong search term

Never use generic filler.

NEVER use:

#womenswear

#clothing

#fashion

#style

#outfit

#ootd

just to fill space.

============================================================
PRICE
============================================================

Do not claim live marketplace data.

Common/basic:

$7-$12

Better Y2K/mall-brand:

$9-$14

Highly desirable:

$15-$20 maximum

Never exceed $20.

Whole dollar amounts only — no cents (e.g. 12, not 12.50).

Vintage weighting:

If "vintage" is genuinely justified in the AESTHETIC style list above
(real evidence — actual age, era-accurate construction/tag, genuine
wear patterns — not just an old-looking cut), push suggested_price
toward the TOP of whichever tier the item already belongs in based on
brand/desirability. Vintage does NOT move an item into a higher tier
by itself. A basic, unbranded vintage tee is still common/basic
($7-$12, toward the $12 end) — it does not jump to $15-$20 just for
being vintage. A well-preserved, well-branded vintage piece with real
evidence prices at the top of whichever tier its brand/desirability
already earns.

When uncertain, choose the lower reasonable price.

============================================================
CONFIDENCE SCALE
============================================================

size_confidence and brand_confidence are INTEGERS from 0 to 100 (a
percentage), NOT 0 to 1. If you clearly read the size or brand text,
that is high confidence — 90 to 100, NOT "1". Use 0 only when nothing
readable was found.

============================================================
OUTPUT
============================================================

{
    "title": "",
    "brand": "",
    "brand_confidence": 0,
    "brand_evidence": "",
    "garment_type": "",
    "category": "",
    "size": "",
    "size_confidence": 0,
    "size_evidence": "",
    "color": "",
    "pattern": "",
    "style": [],
    "condition": "",
    "condition_evidence": "",
    "description_bullets": [],
    "suggested_price": 0,
    "hashtags": []
}
"""


    content = [
        {
            "type": "text",
            "text": prompt
        },
        {
            "type": "text",
            "text": (
                "VALID SHOPIFY CATEGORIES (copy one of these strings "
                "exactly into \"category\"):\n"
                + "\n".join(SHOPIFY_CATEGORY_PATHS)
            ),
        },
    ]


    for image_path in image_paths:

        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": image_to_data_url(
                        image_path
                    )
                }
            }
        )


    response = vision_chat_completion(
        content,
        temperature=0.1,
    )


    listing = clean_json_response(
        response.choices[0]
        .message
        .content
    )


    # --------------------------------------------------------
    # VERIFY SIZE
    # --------------------------------------------------------

    (
        verified_size,
        verified_confidence,
        verified_evidence
    ) = validate_size(
        listing
    )


    listing["size"] = (
        verified_size
    )

    listing["size_confidence"] = (
        verified_confidence
    )

    listing["size_evidence"] = (
        verified_evidence
    )


    # --------------------------------------------------------
    # VERIFY BRAND (mirrors size — see validate_brand())
    # --------------------------------------------------------

    (
        verified_brand,
        verified_brand_confidence,
        verified_brand_evidence
    ) = validate_brand(
        listing
    )

    listing["brand"] = (
        verified_brand
    )

    listing["brand_confidence"] = (
        verified_brand_confidence
    )

    listing["brand_evidence"] = (
        verified_brand_evidence
    )


    # --------------------------------------------------------
    # DEDICATED TAG SIZE+BRAND VERIFICATION — ONLY WHEN NEEDED
    # This is ONE call regardless of why it fires — _verify_size_from_tags()
    # always reads both size and brand from the same crops. The trigger
    # below used to check size confidence only, on the theory that brand
    # came "for free" whenever it happened to run for size. In practice
    # a garment's size is often readable from a normal wide shot (large
    # printed size marking) while its brand tag is small and only
    # readable from a zoomed crop — so items with an easy size and a
    # hard brand never got the tag pass at all, and brand silently
    # stayed empty. Triggering on EITHER field needing it still costs
    # at most one extra call per item, never two.
    # --------------------------------------------------------
    size_needs_verification = (
        verified_size in {"Unknown", "-", ""} or verified_confidence < 70
    )
    brand_needs_verification = (
        not verified_brand or verified_brand_confidence < 70
    )

    # Located UNCONDITIONALLY, once, and shared with
    # _verify_vintage_from_tags() below (which now also runs on every
    # item) — this used to only run when size/brand needed verification,
    # leaving _verify_vintage_from_tags() to silently redo the exact
    # same locate call on its own whenever size/brand were already
    # confident. Making it unconditional here doesn't add a new API
    # call in that case — it just captures the call that was already
    # happening downstream, once, in one place. The captured regions
    # also power the "hover to see the size tag" review-screen feature
    # (size_tag_photo_indexes below) even for items whose size/brand
    # were confident enough to skip the dedicated tag re-read.
    try:
        located_tag_regions = _locate_size_tag_regions(
            image_paths
        )
    except Exception as locator_error:
        print(
            "Size-tag locator failed: "
            + str(locator_error)
        )
        located_tag_regions = []

    # Which photo indexes contain a visible tag/label region, for the
    # review screen's size-hover preview — not a size/brand READ, just
    # "which of this item's already-saved photos should I show when the
    # user hovers Size to sanity-check it."
    try:
        listing["size_tag_photo_indexes"] = sorted({
            int(region.get("image"))
            for region in (located_tag_regions or [])
            if region.get("boxes")
        })
    except Exception:
        listing["size_tag_photo_indexes"] = []

    if size_needs_verification or brand_needs_verification:
        try:
            (
                tag_size,
                tag_confidence,
                tag_evidence,
                tag_brand,
                tag_brand_confidence,
                tag_brand_evidence,
                tag_crop_data_url,
            ) = _verify_size_from_tags(
                image_paths,
                located_regions=located_tag_regions,
            )

            # The actual cropped tag image the model read size/brand
            # from — powers the "hover to see the tag" review-screen
            # preview for both fields, instead of a whole uncropped
            # photo the user has to squint at to find the tag in.
            if tag_crop_data_url:
                listing["evidence_crop_data_url"] = tag_crop_data_url

            # Only touch size if size itself is what needed verifying —
            # a brand-only trigger must never overwrite an already-good
            # size just because this particular crop pass didn't
            # re-confirm it as cleanly the second time.
            if size_needs_verification:
                if tag_size != "-" and tag_confidence >= 70:
                    listing["size"] = tag_size
                    listing["size_confidence"] = tag_confidence
                    listing["size_evidence"] = tag_evidence
                else:
                    listing["size"] = "-"
                    listing["size_confidence"] = 0
                    listing["size_evidence"] = (
                        tag_evidence
                        or verified_evidence
                        or "The size marking could not be read confidently from the tag photos."
                    )

            # Unlike size, do NOT clear a decent primary-pass brand to
            # "-" just because this tag crop didn't happen to contain a
            # brand mark — a crop legitimately missing a size marking
            # doesn't mean it's legitimately missing a brand mark. Only
            # overwrite when the tag pass actually found one.
            if tag_brand != "-" and tag_brand_confidence >= 70:
                listing["brand"] = tag_brand
                listing["brand_confidence"] = tag_brand_confidence
                listing["brand_evidence"] = tag_brand_evidence

        except Exception as size_error:
            print(
                "Dedicated size-tag verification skipped: "
                + str(size_error)
            )
            if size_needs_verification:
                if verified_size not in {"Unknown", "-", ""} and verified_confidence >= 50:
                    listing["size"] = verified_size
                    listing["size_confidence"] = verified_confidence
                    listing["size_evidence"] = verified_evidence
                else:
                    listing["size"] = "-"
                    listing["size_confidence"] = 0
                    listing["size_evidence"] = (
                        "No size could be confidently read from the provided tag photos."
                    )
            # brand: leave whatever validate_brand() already set (almost
            # always empty at this point) — nothing better to fall back to.


    # --------------------------------------------------------
    # VINTAGE / DISCONTINUED-ERA TAG ANALYSIS
    # Runs on EVERY item, unconditionally — widened from an earlier
    # version that only ran this when size/brand needed verification or
    # the main pass's own AESTHETIC guess already suspected "vintage".
    # That gated version missed genuinely vintage items that don't
    # "look" vintage stylistically (so the aesthetic guess never fires)
    # but have confidently-read size/brand (so the tag pass never runs
    # either) — tag evidence (RN number, retired logo era, union label)
    # is supposed to be authoritative regardless of visible wear or
    # style, so it needs to actually run every time to make good on
    # that. Costs one extra locate-if-needed + reading call per item.
    # --------------------------------------------------------
    try:
        bias_hint = _vintage_correction_bias_hint(listing)
    except Exception:
        bias_hint = ""

    try:
        vintage_result = _verify_vintage_from_tags(
            image_paths,
            located_regions=located_tag_regions,
            bias_hint=bias_hint,
        )

        listing["is_vintage"] = vintage_result["is_vintage"]
        listing["vintage_classification"] = vintage_result[
            "listing_classification"
        ]
        listing["vintage_evidence"] = {
            "tag_era_signal": vintage_result["tag_era_signal"],
            "discontinued_element": vintage_result[
                "discontinued_element"
            ],
            "estimated_era": vintage_result["estimated_era"],
            "confidence": vintage_result["confidence"],
            "rn_number": vintage_result["rn_number"],
            "suggested_listing_tags": vintage_result[
                "suggested_listing_tags"
            ],
            "evidence_notes": vintage_result["evidence_notes"],
        }

    except Exception as vintage_error:
        print(
            "Dedicated vintage-tag verification skipped: "
            + str(vintage_error)
        )
        listing.setdefault("is_vintage", False)
        listing.setdefault("vintage_classification", "none")

    # The title above was written in the FIRST vision call, before this
    # dedicated tag check ran — the model had only its own conservative
    # aesthetic guess to go on, not real tag evidence. If the tag check
    # now confirms genuine vintage status, make sure the title actually
    # says so instead of silently leaving it out just because it was
    # generated too early to know.
    if listing.get("is_vintage"):
        current_title = str(listing.get("title", "") or "").strip()

        if current_title and "vintage" not in current_title.lower():
            listing["title"] = f"Vintage {current_title}"

    # Deterministic safety net — guarantees the never-Vintage+Y2K,
    # never-Fairycore+Coquette, max-2-style-words rules hold even if
    # the model's own title (or the "Vintage " prefix just added
    # above) violated them. See _enforce_title_style_rules().
    listing["title"] = _enforce_title_style_rules(
        listing.get("title", "")
    )

    # --------------------------------------------------------
    # CLEAN OTHER FIELDS
    # --------------------------------------------------------

    listing["hashtags"] = (
        clean_hashtags(
            listing.get(
                "hashtags",
                []
            )
        )
    )

    if len(
        listing.get(
            "hashtags",
            []
        )
    ) != 5:

        try:

            repaired_hashtags = regenerate_field(
                "hashtags",
                listing,
                image_paths
            )

            repaired_hashtags = clean_hashtags(
                repaired_hashtags
            )

            if len(
                repaired_hashtags
            ) == 5:

                listing["hashtags"] = (
                    repaired_hashtags
                )

        except Exception as hashtag_error:

            print(
                "Hashtag auto-repair failed: "
                + str(hashtag_error)
            )


    listing["suggested_price"] = (
        clean_price(
            listing.get(
                "suggested_price",
                10
            )
        )
    )


    if not listing.get(
        "condition"
    ):

        listing["condition"] = (
            "Excellent"
        )


    if not listing.get(
        "condition_evidence"
    ):

        listing[
            "condition_evidence"
        ] = (
            "No visible stains, holes, damage, "
            "or other flaws in the provided photos."
        )

    # Deterministic template assembly — see build_depop_description().
    # Uses the FINAL condition/size/hashtags (after the fixes above),
    # so the description always reflects what's actually shown
    # elsewhere on the listing, never a stale pre-repair value.
    listing["description_bullets"] = [
        str(point).strip()
        for point in listing.get("description_bullets", []) or []
        if str(point).strip()
    ]
    listing["description"] = build_depop_description(
        listing.get("condition"),
        listing.get("size"),
        listing.get("description_bullets"),
        listing.get("hashtags"),
    )

    return listing


# ============================================================
# REGENERATE ONE FIELD
# ============================================================

def regenerate_field(
    field,
    listing,
    image_paths
):

    if field not in {
        "title",
        "description",
        "hashtags",
        "price",
        "size"
    }:

        raise ValueError(
            f"Invalid regeneration field: {field}"
        )


    if field == "size":
        # Size gets the dedicated tag-locate-and-read pipeline, not the
        # generic single-field prompt below — it needs the same
        # strict "never guess" reading _verify_size_from_tags() already
        # does, not another guess from the full garment photos.
        (
            tag_size,
            tag_confidence,
            tag_evidence,
            tag_brand,
            tag_brand_confidence,
            tag_brand_evidence,
            tag_crop_data_url,
        ) = _verify_size_from_tags(
            image_paths
        )

        if tag_crop_data_url:
            listing["evidence_crop_data_url"] = tag_crop_data_url

        if tag_size != "-" and tag_confidence >= 70:
            listing["size_confidence"] = tag_confidence
            listing["size_evidence"] = tag_evidence
        else:
            tag_size = "-"
            listing["size_confidence"] = 0
            listing["size_evidence"] = (
                tag_evidence
                or "The size marking could not be read confidently from the tag photos."
            )

        if tag_brand != "-" and tag_brand_confidence >= 70:
            listing["brand"] = tag_brand
            listing["brand_confidence"] = tag_brand_confidence
            listing["brand_evidence"] = tag_brand_evidence

        return tag_size


    current_listing = {
        "title": listing.get(
            "title",
            ""
        ),

        "brand": listing.get(
            "brand",
            ""
        ),

        "category": listing.get(
            "category",
            ""
        ),

        "size": listing.get(
            "size",
            "Unknown"
        ),

        "color": listing.get(
            "color",
            ""
        ),

        "pattern": listing.get(
            "pattern",
            ""
        ),

        "style": listing.get(
            "style",
            []
        ),

        "condition": listing.get(
            "condition",
            "Excellent"
        )
    }


    if field == "title":

        prompt = f"""
Create a better Depop title for this exact clothing item.

Use the photos as the source of truth.

Current information:

{json.dumps(current_listing, indent=2)}

Rules:

- Be specific.
- Identify the actual garment.
- Include brand when useful.
- Include an aesthetic/era word (Y2K, vintage, fairycore, coquette,
  etc.) only if genuinely justified.
- NEVER include both "Vintage" and "Y2K" in the same title — if both
  are arguably justified, use "Vintage" only.
- NEVER include both "Fairycore" and "Coquette" in the same title —
  pick whichever fits better.
- NEVER include more than 2 aesthetic/era words in the title total.
- Mention important visible details.
- Do not invent anything.
- Keep it concise.
- No emojis.
- Return ONLY valid JSON.

Format:

{{
    "value": ""
}}
"""


    elif field == "description":

        prompt = f"""
Rewrite the item-detail bullets for this exact clothing item's Depop
description. The condition/size/disclaimer/hashtags/footer are fixed
boilerplate assembled automatically — you're only rewriting the
bullets.

Use the photos as the source of truth.

Current listing:

{json.dumps(listing, indent=2)}

Rules:

- 6-9 short bullet-point strings, each a few words to one short
  clause — not full sentences.
- Order: brand, then color/pattern, then other genuinely visible
  details (silhouette/design, waist/fit construction, fabric/feel,
  overall fit, length), ending with a short wearability line.
- Describe actual visible details only. Do not invent anything.
- Do not use the word "fitted".
- Do not repeat the brand or color in multiple bullets.
- Do not include the condition or size — those are separate fields.
- Do not include emojis.

Return ONLY valid JSON:

{{
    "value": [
        "bullet 1",
        "bullet 2"
    ]
}}
"""


    elif field == "hashtags":

        prompt = f"""
Create EXACTLY 5 highly relevant Depop hashtags.

Use the photos and listing information.

Current listing:

{json.dumps(listing, indent=2)}

Rules:

- Exactly five hashtags.
- Every hashtag must be relevant.
- Use the actual garment type.
- Use the actual aesthetic when justified.
- Use actual visible details.
- Use brand when useful.
- Do NOT use #womenswear.
- Do NOT use #clothing.
- Do NOT use #fashion.
- Do NOT use #style.
- Do NOT use #outfit.
- Do NOT use #ootd.
- Do not use generic filler.
- Do not repeat hashtags.

Return ONLY valid JSON:

{{
    "value": [
        "#tag1",
        "#tag2",
        "#tag3",
        "#tag4",
        "#tag5"
    ]
}}
"""


    else:

        prompt = f"""
Suggest a realistic resale price for this exact clothing item.

Current listing:

{json.dumps(listing, indent=2)}

Rules:

Common/basic:
$7-$12

Better Y2K/mall-brand:
$9-$14

Highly desirable:
$15-$20

Never exceed $20.

Whole dollar amounts only — no cents (e.g. 12, not 12.50).

Vintage weighting: if "vintage" is genuinely justified in this
listing's "style" list (real evidence, not just an old-looking cut),
push the price toward the TOP of whichever tier above it already
belongs in based on brand/desirability. Vintage does NOT move an item
into a higher tier by itself — a basic, unbranded vintage tee still
prices as common/basic ($7-$12, toward the $12 end).

When uncertain, choose the lower reasonable price.

Return ONLY valid JSON:

{{
    "value": 12
}}
"""


    content = [
        {
            "type": "text",
            "text": prompt
        }
    ]


    for image_path in image_paths:

        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": image_to_data_url(
                        image_path
                    )
                }
            }
        )


    response = vision_chat_completion(
        content,
        temperature=0.2,
    )


    result = clean_json_response(
        response.choices[0]
        .message
        .content
    )


    value = result.get(
        "value"
    )


    if field == "hashtags":

        return clean_hashtags(
            value
        )


    if field == "price":

        return clean_price(
            value
        )

    if field == "description":
        bullets = [
            str(point).strip().lstrip("•-").strip()
            for point in (value or [])
            if str(point).strip()
        ]
        return build_depop_description(
            listing.get("condition"),
            listing.get("size"),
            bullets,
            listing.get("hashtags"),
        )

    result_value = str(
        value
        if value is not None
        else ""
    ).strip()

    if field == "title":
        result_value = _enforce_title_style_rules(result_value)

    return result_value


