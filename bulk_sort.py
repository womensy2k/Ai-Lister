from __future__ import annotations

import copy
import json
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image, ExifTags
import imagehash

from ai_listing import vision_chat_completion, client, image_to_data_url, clean_json_response
from grouping import get_capture_timestamp

# ============================================================
# BULK PHOTO AUTO-SORT
# A completely separate, additive flow from the manual upload deck.
# Nothing in app.py's existing manual_groups/bulk_paths/"Generate All
# Listings" pipeline is touched by this module — its only integration
# point is producing groups shaped so app.py can append them straight
# into manual_groups once the user confirms a batch.
#
# Core principle (per the user's own framing): no single signal is
# trustworthy. Every grouping decision below combines multiple
# independent signals and scores confidence rather than trusting any
# one heuristic.
# ============================================================

BULK_SORT_STATE_FILE = Path(__file__).with_name("bulk_sort_state.json")

# ------------------------------------------------------------
# TUNABLE THRESHOLDS
# These are first-guess starting points, not tuned values — the
# reasoning log exists specifically so they can be adjusted from real
# batches instead of re-guessed blind. See the plan doc for the
# reasoning behind each starting value.
# ------------------------------------------------------------
TIME_GAP_SECONDS = 150  # 2.5 min candidate-cluster split
# Raised from 0.80 after a real 20-photo batch showed 0.80 wasn't
# strict enough: 3 visibly different pairs of jeans (shared generic
# "distressed blue denim" vocabulary) scored above it and merged into
# one group. Used as the EMBEDDING fallback threshold when no usable
# feature text exists on one/both sides to compare directly - see
# TEXT_SPLIT_THRESHOLD below for when text IS available, and
# validate_and_split_clusters()'s docstring for the full reasoning.
SPLIT_SIMILARITY_THRESHOLD = 0.88  # below this, a time cluster gets split
# A SEPARATE, more lenient threshold for when direct feature-text
# comparison is available, added after a real 53-photo batch showed
# the single 0.88 bar applied to BOTH embedding and text was
# over-splitting: two photos with essentially IDENTICAL
# distinguishing-features text (1.0 overlap) still got split apart
# because their embedding similarity (0.85-0.87) fell just short of
# 0.88. Text has now proven reliable in BOTH directions on real data
# (caught the original over-merge with low text similarity; caught
# these over-splits with high text similarity), so when it's
# available it decides on its own, more lenient bar rather than being
# yoked to the stricter embedding-only threshold.
TEXT_SPLIT_THRESHOLD = 0.70  # below this, feature-text similarity is treated as a real disagreement
# Lowered from 0.65 after a real batch showed a genuinely correct
# match (a tag photo scoring 0.6402 against its true full_garment
# match, the highest of its 9 candidates) narrowly missing the bar and
# falling to Pass B, where it paired with an unrelated tag instead.
# Confirmed the nearest wrong-match ceiling in the same batch was
# 0.5765 (comfortable headroom below 0.63), so this isn't just
# loosening the bar generally - it's closing a specific gap between a
# real match and a real non-match.
RESCUE_ACCEPT_THRESHOLD = 0.63  # minimum combined rescue score to attach
GROUP_CONFIDENCE_GREEN_THRESHOLD = 0.85

# Rebalanced from embedding-first (0.40/0.30) after a real case showed
# embedding_sim can be the LESS reliable signal for a tag_closeup with
# sparse text: batch_9 ("white tag, solid dark blue denim", no
# distinguishing_features) scored 0.78 embedding similarity against
# the WRONG group (light-blue shorts) vs only ~0.59 against its actual
# dark-blue-denim jeans match — the embedding of a short, generic
# phrase doesn't reliably separate similarly-worded candidates. The
# direct pattern/features text comparison (fabric_continuity) called
# it correctly both times (0.18 wrong vs 0.25-0.29 right), so it now
# carries the larger share.
RESCUE_WEIGHTS = {
    "embedding_sim": 0.25,
    "fabric_continuity": 0.45,
    "tag_compatibility": 0.20,
    "background_match": 0.10,
}

_CLASSIFY_ATTR_BATCH = 6  # smaller than grouping.py's 16 - much richer per-image schema
_CLASSIFY_WORKERS = 3  # mirrors detect_photo_rotations()'s deliberately-tuned worker count
_UNDO_STACK_MAX = 20

EMBEDDING_MODEL = "text-embedding-3-small"

VALID_GARMENT_TYPES = {
    "shirt", "jeans", "jacket", "dress", "shorts",
    "skirt", "sweater", "other", "not_a_garment",
}
VALID_IMAGE_ROLES = {
    "full_garment", "detail_shot", "tag_closeup",
    "flaw_closeup", "folded", "worn", "other",
}
# image_role values that don't carry their own garment_type identity -
# a tag closeup or flaw closeup can legitimately belong to any garment,
# so they're excluded from the "cluster disagrees on garment_type"
# split check in Step 2.
_ROLES_WITHOUT_OWN_GARMENT_TYPE = {"tag_closeup", "detail_shot", "flaw_closeup"}


# ============================================================
# PERSISTENCE
# Mirrors app.py's PRICING_PROFILE_FILE pattern exactly: defensive
# load (missing/corrupt file -> empty state, never raises) and
# defensive save (persistence must never break the review UI).
# ============================================================

def _new_state():
    return {
        "images": {},           # resolved_path -> signal dict (the cache)
        "groups": [],            # list of group dicts
        "needs_attention": [],   # list of resolved_path strings
        "reasoning_log": [],     # list of decision dicts
        "manual_corrections": [],
        "undo_stack": [],
        "next_group_id": 1,
        "stray_matches": {},    # resolved_path -> {"group_id", "score"}
    }


def load_bulk_sort_state():
    if not BULK_SORT_STATE_FILE.exists():
        return _new_state()

    try:
        loaded = json.loads(BULK_SORT_STATE_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"bulk_sort_state.json failed to load, starting fresh: {exc}")
        return _new_state()

    if not isinstance(loaded, dict):
        return _new_state()

    state = _new_state()
    state.update({key: loaded.get(key, state[key]) for key in state})
    return state


def save_bulk_sort_state(state):
    try:
        BULK_SORT_STATE_FILE.write_text(
            json.dumps(state, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    except Exception as exc:
        # Persistence must never break the review UI - the user can
        # keep working, they just risk losing the session on refresh.
        print(f"bulk_sort_state.json failed to save: {exc}")


def _resolved_key(image_path):
    return str(Path(image_path).resolve())


def _mtime_ns(image_path):
    try:
        return Path(image_path).stat().st_mtime_ns
    except Exception:
        return None


# ============================================================
# EXIF EXTRACTION
# Only orientation is used elsewhere in this codebase
# (ai_listing.py's ImageOps.exif_transpose calls) - GPS, camera
# model, lens, focal length, and exposure are new here.
# ============================================================

def _clean_exif_str(value):
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    text = str(value).strip().strip("\x00").strip()
    return text or None


def _ratio_to_float(value):
    try:
        return round(float(value), 4)
    except Exception:
        return None


def _gps_to_decimal(dms, ref):
    try:
        degrees, minutes, seconds = [float(v) for v in dms]
    except Exception:
        return None

    decimal = degrees + minutes / 60.0 + seconds / 3600.0

    if ref in ("S", "W"):
        decimal = -decimal

    return round(decimal, 6)


def _parse_gps(gps_ifd):
    gps_tag_names = {
        tag_id: name for tag_id, name in ExifTags.GPSTAGS.items()
    }
    named = {
        gps_tag_names.get(tag_id, tag_id): value
        for tag_id, value in gps_ifd.items()
    }

    lat = named.get("GPSLatitude")
    lon = named.get("GPSLongitude")

    if lat is None or lon is None:
        return None

    lat_decimal = _gps_to_decimal(lat, named.get("GPSLatitudeRef"))
    lon_decimal = _gps_to_decimal(lon, named.get("GPSLongitudeRef"))

    if lat_decimal is None or lon_decimal is None:
        return None

    return {"lat": lat_decimal, "lon": lon_decimal}


def extract_exif_signals(image_path):
    """Best-effort EXIF extraction beyond just orientation: capture
    time (reuses grouping.get_capture_timestamp - same
    DateTimeOriginal -> DateTimeDigitized -> DateTime fallback chain),
    GPS, camera model, lens model, orientation, focal length,
    exposure. Every field is independently best-effort - a missing
    field is just None, never fatal, mirroring
    get_capture_timestamp()'s own try/except-return-None shape.
    """
    image_path = Path(image_path)

    signals = {
        "capture_time": None,
        "gps": None,
        "camera_model": None,
        "lens_model": None,
        "orientation": None,
        "focal_length": None,
        "exposure_time": None,
        "f_number": None,
        "iso": None,
    }

    signals["capture_time_iso"] = None
    capture_time = get_capture_timestamp(image_path)
    if capture_time is not None:
        signals["capture_time_iso"] = capture_time.isoformat()

    try:
        with Image.open(image_path) as image:
            base_exif = image.getexif()

            if not base_exif:
                return signals

            base_tag_names = {
                tag_id: name for tag_id, name in ExifTags.TAGS.items()
            }

            for tag_id, value in base_exif.items():
                name = base_tag_names.get(tag_id)
                if name == "Model":
                    signals["camera_model"] = _clean_exif_str(value)
                elif name == "Orientation":
                    signals["orientation"] = value

            try:
                exif_ifd = base_exif.get_ifd(ExifTags.IFD.Exif)
            except Exception:
                exif_ifd = {}

            for tag_id, value in exif_ifd.items():
                name = base_tag_names.get(tag_id)
                if name == "LensModel":
                    signals["lens_model"] = _clean_exif_str(value)
                elif name == "FocalLength":
                    signals["focal_length"] = _ratio_to_float(value)
                elif name == "ExposureTime":
                    signals["exposure_time"] = _ratio_to_float(value)
                elif name == "FNumber":
                    signals["f_number"] = _ratio_to_float(value)
                elif name == "ISOSpeedRatings":
                    try:
                        signals["iso"] = int(value)
                    except (TypeError, ValueError):
                        pass

            try:
                gps_ifd = base_exif.get_ifd(ExifTags.IFD.GPSInfo)
            except Exception:
                gps_ifd = {}

            if gps_ifd:
                signals["gps"] = _parse_gps(gps_ifd)

    except Exception as exc:
        print(f"EXIF extraction skipped for {image_path.name}: {exc}")

    return signals


# ============================================================
# STRUCTURED GARMENT CLASSIFICATION
# ============================================================

def _classification_prompt(batch_paths, batch_start_number):
    content = [{
        "type": "text",
        "text": r"""
You are analyzing clothing resale photos to extract structured
attributes used for automated photo grouping - matching which photos
belong to the same physical garment, including telling apart
near-identical items (e.g. two similar pairs of blue jeans).

For EACH photo below, extract:

- image_role: exactly one of full_garment, detail_shot, tag_closeup,
  flaw_closeup, folded, worn, other
- garment_type: exactly one of shirt, jeans, jacket, dress, shorts,
  skirt, sweater, other, not_a_garment
- dominant_color: the single most prominent color, one or two words
- secondary_colors: list of other visible colors, up to 3
- pattern_description: describe any print/pattern/graphic specifically
  (e.g. "floral print", "solid navy", "graphic tee - skull design",
  "houndstooth") - be SPECIFIC, this is the strongest signal for
  telling apart near-identical items
- distinguishing_features: specific wear marks, stains, unique
  hardware, distinctive stitching, damage - anything that would let
  you tell THIS physical item apart from another very similar one.
  Empty string if nothing distinguishing is visible - do not invent
  detail.
- visible_text: any readable brand/size/tag text, exactly as printed.
  Empty string if none readable.
- background_description: brief description of the surface/background
  this photo was taken on or against

Only report what is ACTUALLY visible. Do not guess or infer anything
not directly supported by the photo. If a photo doesn't show a garment
at all, set garment_type to "not_a_garment".

Judge each photo independently. Return ONLY JSON:
{
  "photos": [
    {
      "photo_number": 1,
      "image_role": "",
      "garment_type": "",
      "dominant_color": "",
      "secondary_colors": [],
      "pattern_description": "",
      "distinguishing_features": "",
      "visible_text": "",
      "background_description": ""
    }
  ]
}
""",
    }]

    for offset, image_path in enumerate(batch_paths):
        photo_number = batch_start_number + offset
        content.append({
            "type": "text",
            "text": f"PHOTO {photo_number}: {Path(image_path).name}",
        })
        content.append({
            "type": "image_url",
            "image_url": {
                "url": image_to_data_url(image_path),
                "detail": "high",
            },
        })

    return content


def _default_classification():
    return {
        "image_role": "other",
        "garment_type": "other",
        "dominant_color": "",
        "secondary_colors": [],
        "pattern_description": "",
        "distinguishing_features": "",
        "visible_text": "",
        "background_description": "",
    }


def _clean_classification_entry(entry):
    result = _default_classification()

    if not isinstance(entry, dict):
        return result

    role = str(entry.get("image_role", "") or "").strip().lower()
    if role in VALID_IMAGE_ROLES:
        result["image_role"] = role

    garment_type = str(entry.get("garment_type", "") or "").strip().lower()
    if garment_type in VALID_GARMENT_TYPES:
        result["garment_type"] = garment_type

    result["dominant_color"] = str(entry.get("dominant_color", "") or "").strip()

    secondary = entry.get("secondary_colors", [])
    if isinstance(secondary, list):
        result["secondary_colors"] = [
            str(c).strip() for c in secondary[:3] if str(c).strip()
        ]

    result["pattern_description"] = str(
        entry.get("pattern_description", "") or ""
    ).strip()
    result["distinguishing_features"] = str(
        entry.get("distinguishing_features", "") or ""
    ).strip()
    result["visible_text"] = str(entry.get("visible_text", "") or "").strip()
    result["background_description"] = str(
        entry.get("background_description", "") or ""
    ).strip()

    return result


def _process_classification_batch(batch_paths, batch_start_number):
    """One batch's vision call. Never raises - a failed batch returns
    default ("other"/"other") classifications for every photo in it,
    which (by construction, via the confidence scoring downstream)
    can't clear a confident group placement and lands those photos in
    Needs Attention rather than failing the whole run.
    """
    content = _classification_prompt(batch_paths, batch_start_number)

    result = None
    last_error = None

    for attempt in range(2):
        try:
            response = vision_chat_completion(
                content,
                temperature=0,
                response_format={"type": "json_object"},
            )
            result = clean_json_response(response.choices[0].message.content)
            break
        except Exception as exc:
            last_error = exc

    batch_results = {
        batch_start_number + offset: _default_classification()
        for offset in range(len(batch_paths))
    }

    if result is None:
        print(
            f"Classification batch {batch_start_number}-"
            f"{batch_start_number + len(batch_paths) - 1} failed: {last_error}"
        )
        return batch_results

    photos = result.get("photos", [])
    if not isinstance(photos, list):
        return batch_results

    for entry in photos:
        if not isinstance(entry, dict):
            continue
        try:
            photo_number = int(entry.get("photo_number", -1))
        except (TypeError, ValueError):
            continue

        if photo_number not in batch_results:
            continue

        batch_results[photo_number] = _clean_classification_entry(entry)

    return batch_results


def classify_all_images(image_paths, on_progress=None):
    """Structured garment classification over the whole batch, batched
    (_CLASSIFY_ATTR_BATCH per call) and parallelized
    (_CLASSIFY_WORKERS threads) - mirrors
    ai_listing.detect_photo_rotations()'s established shape exactly,
    including the on_progress(completed, total) callback contract.

    Returns {index: classification_dict} for every index 0..len-1 -
    every image gets an entry, defaults included, so callers never
    need to handle a missing key.
    """
    image_paths = list(image_paths)
    results = {}

    batch_starts = list(range(0, len(image_paths), _CLASSIFY_ATTR_BATCH))
    total_batches = len(batch_starts)
    completed_batches = 0

    if not batch_starts:
        return results

    worker_count = min(_CLASSIFY_WORKERS, total_batches)

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_map = {
            executor.submit(
                _process_classification_batch,
                image_paths[start:start + _CLASSIFY_ATTR_BATCH],
                start,
            ): start
            for start in batch_starts
        }

        for future in as_completed(future_map):
            try:
                batch_results = future.result()
            except Exception as exc:
                start = future_map[future]
                print(f"Classification batch at {start} failed: {exc}")
                batch_results = {
                    start + offset: _default_classification()
                    for offset in range(
                        min(_CLASSIFY_ATTR_BATCH, len(image_paths) - start)
                    )
                }

            results.update(batch_results)

            completed_batches += 1
            if on_progress is not None:
                try:
                    on_progress(completed_batches, total_batches)
                except Exception:
                    pass

    return {i: results.get(i, _default_classification()) for i in range(len(image_paths))}


# ============================================================
# EMBEDDINGS
# No CLIP/local embedding model exists anywhere in this project (no
# torch, no open_clip - confirmed absent). Rather than add that heavy
# a dependency, this embeds a canonical TEXT summary of each image's
# structured classification attributes via OpenAI's cheap
# text-embedding-3-small - a real vector, cheap, reuses the existing
# OpenAI client, on a separate rate-limit bucket from chat
# completions. See the plan doc for the full reasoning; the important
# part operationally is that this is isolated behind ONE function
# (embed_image_attributes) so swapping in a real CLIP model later only
# touches this function - the rest of the pipeline just consumes a
# float vector per image, unaware of how it was produced.
# ============================================================

def _attribute_text(classification):
    parts = [
        classification.get("garment_type", ""),
        classification.get("dominant_color", ""),
        " ".join(classification.get("secondary_colors", [])),
        classification.get("pattern_description", ""),
        classification.get("distinguishing_features", ""),
        classification.get("visible_text", ""),
    ]
    return " ".join(p for p in parts if p).strip()


def embed_image_attributes(classifications_by_index):
    """classifications_by_index: {index: classification_dict}.
    Returns {index: [float, ...]} - images whose attribute text is
    empty (e.g. a totally unreadable photo) get an all-zero vector
    rather than being skipped, so every index always has an entry and
    downstream cosine-similarity code never needs a presence check.
    """
    indices = sorted(classifications_by_index.keys())
    texts = [
        _attribute_text(classifications_by_index[i]) or "unknown item"
        for i in indices
    ]

    if not texts:
        return {}

    embeddings_by_index = {}

    # Chunk defensively in case of a very large batch - the embeddings
    # endpoint accepts array input but has practical request-size
    # limits worth not pushing on a 100+ image batch.
    chunk_size = 500

    for chunk_start in range(0, len(texts), chunk_size):
        chunk_indices = indices[chunk_start:chunk_start + chunk_size]
        chunk_texts = texts[chunk_start:chunk_start + chunk_size]

        try:
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=chunk_texts,
            )
            for local_i, item in enumerate(response.data):
                embeddings_by_index[chunk_indices[local_i]] = list(item.embedding)
        except Exception as exc:
            print(f"Embedding batch at {chunk_start} failed: {exc}")
            for i in chunk_indices:
                embeddings_by_index[i] = []

    return embeddings_by_index


def _cosine_similarity(vec_a, vec_b):
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0

    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return max(0.0, min(1.0, dot / (norm_a * norm_b)))


# ============================================================
# PERCEPTUAL HASH
# Free (no API call), supplementary near-duplicate signal - not a
# CLIP replacement, just an extra cheap signal for the literal
# same-photo-different-angle case.
# ============================================================

def compute_phash(image_path):
    try:
        with Image.open(image_path) as image:
            return str(imagehash.phash(image))
    except Exception as exc:
        print(f"Perceptual hash skipped for {Path(image_path).name}: {exc}")
        return None


def _phash_similarity(hash_a, hash_b):
    if not hash_a or not hash_b:
        return 0.0
    try:
        distance = imagehash.hex_to_hash(hash_a) - imagehash.hex_to_hash(hash_b)
    except Exception:
        return 0.0
    # phash is a 64-bit hash (8x8 DCT default) - max Hamming distance 64.
    return max(0.0, 1.0 - (distance / 64.0))


# ============================================================
# TEXT SIMILARITY (for fabric-continuity / background-match scoring)
# Plain token-overlap, no extra API call - deliberately cheap since
# it's one of four sub-signals, not the primary one.
# ============================================================

def _tokenize(text):
    return set(re.findall(r"[a-z0-9]+", str(text or "").lower()))


def _jaccard_similarity(text_a, text_b):
    tokens_a = _tokenize(text_a)
    tokens_b = _tokenize(text_b)

    if not tokens_a or not tokens_b:
        return 0.0

    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b

    return len(intersection) / len(union) if union else 0.0


def _overlap_coefficient(text_a, text_b):
    """|A∩B| / min(|A|,|B|) instead of symmetric Jaccard's |A∩B|/|A∪B|.

    Confirmed live (see _score_rescue_candidate's fabric_continuity
    use) that symmetric Jaccard structurally penalizes exactly the
    case this signal exists to catch: a tag_closeup crop only sees a
    strip of fabric and produces a terse pattern/features description,
    while the matching full_garment shot of the SAME item produces a
    much richer one - real example, "faded denim" vs "faded denim with
    decorative stitching and rhinestone embellishments on back
    pockets", scored only 0.20 under Jaccard (dragged down by the
    richer side's extra vocabulary) despite the sparser side's tokens
    being FULLY contained in the richer one. Overlap coefficient scores
    that same real pair at 1.0 - it measures containment (is the
    sparser description's vocabulary a subset of the richer one's?)
    rather than penalizing one side for simply having more to say.
    """
    tokens_a = _tokenize(text_a)
    tokens_b = _tokenize(text_b)

    if not tokens_a or not tokens_b:
        return 0.0

    intersection = tokens_a & tokens_b
    smaller = min(len(tokens_a), len(tokens_b))

    return len(intersection) / smaller if smaller else 0.0


# ============================================================
# SIGNAL EXTRACTION ORCHESTRATION (with disk-persisted caching)
# ============================================================

def extract_all_signals(image_paths, state, on_progress=None):
    """Extracts EXIF + classification + embedding + phash for every
    image, skipping anything already cached in state["images"] under
    an unchanged (resolved_path, mtime_ns) key - this is what makes
    re-running grouping after a threshold tweak or a manual edit free
    of new vision-API calls.

    Mutates and returns state. on_progress(done, total) is called once
    per image that actually needed (re-)analysis - a fully-cached
    re-run calls it 0 times, matching "near-zero re-run cost".
    """
    image_paths = [Path(p) for p in image_paths]
    images_cache = state["images"]

    to_analyze = []  # (index, path) needing classification/embedding
    for index, path in enumerate(image_paths):
        key = _resolved_key(path)
        mtime = _mtime_ns(path)
        cached = images_cache.get(key)

        if cached is not None and cached.get("mtime_ns") == mtime:
            continue

        to_analyze.append((index, path))

    total = len(to_analyze)
    if total == 0:
        if on_progress is not None:
            try:
                on_progress(0, 0)
            except Exception:
                pass
        return state

    # EXIF + phash are local/instant - do them inline while building
    # the classification batch input.
    exif_by_index = {}
    phash_by_index = {}
    for index, path in to_analyze:
        exif_by_index[index] = extract_exif_signals(path)
        phash_by_index[index] = compute_phash(path)

    classify_paths = [path for _, path in to_analyze]
    classification_results = classify_all_images(
        classify_paths,
        on_progress=on_progress,
    )
    # classify_all_images returns 0-indexed against classify_paths, not
    # against image_paths - remap.
    classification_by_index = {
        to_analyze[local_i][0]: classification_results[local_i]
        for local_i in range(len(to_analyze))
    }

    embeddings_by_local = embed_image_attributes(
        {local_i: classification_by_index[to_analyze[local_i][0]] for local_i in range(len(to_analyze))}
    )
    embedding_by_index = {
        to_analyze[local_i][0]: embeddings_by_local.get(local_i, [])
        for local_i in range(len(to_analyze))
    }

    for index, path in to_analyze:
        key = _resolved_key(path)
        images_cache[key] = {
            "mtime_ns": _mtime_ns(path),
            "exif": exif_by_index[index],
            "classification": classification_by_index[index],
            "embedding": embedding_by_index[index],
            "phash": phash_by_index[index],
        }

    save_bulk_sort_state(state)
    return state


# ============================================================
# GROUPING ALGORITHM
# ============================================================

def _signals_for(state, path):
    return state["images"].get(_resolved_key(path), {})


def cluster_by_time(image_paths, state):
    """Step 1 - time-based CANDIDATE clusters only. Untimestamped
    photos get no cluster at all here (handled entirely by Step 3
    rescue - they have no time signal to cluster on).
    Returns list of clusters, each a list of resolved-path strings, in
    chronological order; untimestamped paths are returned separately.
    """
    timestamped = []
    untimestamped = []

    for path in image_paths:
        key = _resolved_key(path)
        signals = state["images"].get(key, {})
        capture_time_iso = signals.get("exif", {}).get("capture_time_iso")

        if capture_time_iso:
            timestamped.append((capture_time_iso, key))
        else:
            untimestamped.append(key)

    timestamped.sort(key=lambda pair: pair[0])

    clusters = []
    current = []
    previous_time = None

    for time_iso, key in timestamped:
        from datetime import datetime
        current_time = datetime.fromisoformat(time_iso)

        if previous_time is not None:
            gap = (current_time - previous_time).total_seconds()
            if gap > TIME_GAP_SECONDS:
                clusters.append(current)
                current = []

        current.append(key)
        previous_time = current_time

    if current:
        clusters.append(current)

    return clusters, untimestamped


def _cluster_garment_types(cluster, state):
    types = set()
    for key in cluster:
        signals = state["images"].get(key, {})
        classification = signals.get("classification", {})
        role = classification.get("image_role", "other")
        if role in _ROLES_WITHOUT_OWN_GARMENT_TYPE:
            continue
        garment_type = classification.get("garment_type", "other")
        if garment_type != "not_a_garment":
            types.add(garment_type)
    return types


def _pattern_feature_text(classification):
    return " ".join(
        p for p in (
            classification.get("pattern_description", ""),
            classification.get("distinguishing_features", ""),
        ) if p
    )


def _best_text_conflict(feature_text, target_sub, state, threshold):
    """Compare feature_text against target_sub's AGGREGATE text (every
    member's pattern/features joined), not against whichever single
    member happens to match best.

    Confirmed live this matters: comparing against the best individual
    match is vulnerable to "chaining" (a classic single-link
    clustering failure mode) - once a sub-cluster has grown to include
    several members, a new candidate only needs to match ONE lenient
    existing member to join, even if it's clearly different from
    others already in the group. Real case: khaki plaid shorts and
    olive-green plaid shorts (share almost all their pattern
    vocabulary - "plaid pattern with [color], white, and burgundy
    stripes" - differing mainly in the color word) ended up in one
    group because each new item found SOME already-included item
    close enough, even though the group as a whole mixes two colors.
    Comparing against the pooled aggregate instead means a color that
    genuinely isn't part of the group's collective identity shows up
    as a real drop in overlap, not just a lucky single match.

    Only compares when both sides have enough text to meaningfully
    compare (a sparse/empty description isn't evidence of
    disagreement, just a plain photo). Returns (conflict: bool,
    similarity: float|None).
    """
    aggregate_text = _group_aggregate_text(target_sub, state)

    if len(_tokenize(feature_text)) < 3 or len(_tokenize(aggregate_text)) < 3:
        return False, None

    similarity = _overlap_coefficient(feature_text, aggregate_text)
    conflict = similarity < threshold
    return conflict, similarity


def validate_and_split_clusters(clusters, state):
    """Step 2 - walk each time cluster in chronological order,
    deciding for every photo whether it continues the currently-open
    sub-cluster or starts a new one.

    Two kinds of photos, two levels of scrutiny:

    - IDENTITY-BEARING (full_garment/worn/folded) photos get the full
      three-signal check against the current sub-cluster: garment_type
      conflict, embedding similarity, AND a direct pattern/
      distinguishing-features text check (added after a real 20-photo
      batch showed embedding similarity ALONE at the old 0.80
      threshold wasn't strict enough - three visibly different pairs
      of jeans, all "distressed/faded blue denim", scored above 0.80
      against each other on their shared generic vocabulary and merged
      into one 8-photo group despite clearly different
      distinguishing_features text. Threshold raised to 0.88 and the
      text check added as an independent second signal - the exact
      "strongest disambiguator" signal the user's own spec called out.
      Only applied when both sides have non-trivial text, so a terse
      description isn't treated as "clearly different" just for being
      terse).

    - tag_closeup/detail_shot/flaw_closeup photos carry no independent
      garment identity, so they don't get the garment_type/embedding
      checks - but they DO get the same feature-text conflict check
      against whatever sub-cluster is currently open, and defer to
      Step 3's full multi-signal rescue scoring if either that check
      finds a conflict, or nothing is open yet to attach to. An
      earlier version deferred EVERY tag/detail/flaw photo
      unconditionally, on the theory that "blind auto-attach" was the
      whole problem - confirmed live that this overcorrected: two tag
      photos that WERE correctly time-adjacent to their real groups
      ended up pairing with each other in Step 3 instead, since sparse
      tag-vs-tag embeddings can coincidentally score higher than
      tag-vs-true-group. The light text check here catches the actual
      failure mode (an unrelated tag glued onto whatever's open) while
      still trusting time-adjacency when nothing contradicts it.

    Returns (validated_clusters, deferred_keys) - deferred_keys are
    the tag/detail/flaw closeups that failed their light check (or had
    nothing open to attach to) and need Step 3's full scoring.
    """
    validated = []
    deferred = []

    for cluster in clusters:
        if len(cluster) <= 1:
            if cluster:
                key = cluster[0]
                role = state["images"].get(key, {}).get("classification", {}).get(
                    "image_role", "other"
                )
                # A lone tag/detail/flaw photo with no time-adjacent
                # garment photo at all is exactly the "weak fit" case
                # that needs Step 3's full scoring, not a free pass as
                # a standalone one-photo group.
                if role in _ROLES_WITHOUT_OWN_GARMENT_TYPE:
                    deferred.append(key)
                else:
                    validated.append(cluster)
            continue

        sub_clusters = []

        for key in cluster:
            signals = state["images"].get(key, {})
            classification = signals.get("classification", {})
            role = classification.get("image_role", "other")
            feature_text = _pattern_feature_text(classification)

            target_sub = sub_clusters[-1] if sub_clusters else None

            if role in _ROLES_WITHOUT_OWN_GARMENT_TYPE:
                if target_sub is None:
                    deferred.append(key)
                    continue

                text_conflict, best_text_similarity = _best_text_conflict(
                    feature_text, target_sub, state, TEXT_SPLIT_THRESHOLD
                )
                color_conflict = _color_conflict(
                    classification.get("dominant_color", ""),
                    _group_dominant_colors(target_sub, state),
                )
                conflict = text_conflict or color_conflict

                _log_decision(
                    state, "split_check", key,
                    {
                        "role": role,
                        "text_similarity": (
                            round(best_text_similarity, 4)
                            if best_text_similarity is not None else None
                        ),
                        "color_conflict": color_conflict,
                        "deferred": conflict,
                    },
                    None, best_text_similarity, TEXT_SPLIT_THRESHOLD,
                )

                if conflict:
                    deferred.append(key)
                else:
                    target_sub.append(key)
                continue

            garment_type = classification.get("garment_type", "other")
            embedding = signals.get("embedding", [])

            if target_sub is None:
                sub_clusters.append([key])
                continue

            sub_types = _cluster_garment_types(target_sub, state)
            type_conflict = (
                garment_type != "other"
                and sub_types
                and garment_type not in sub_types
            )

            best_similarity = max(
                (
                    _cosine_similarity(
                        embedding,
                        state["images"].get(other_key, {}).get("embedding", []),
                    )
                    for other_key in target_sub
                ),
                default=1.0,
            )

            _, best_text_similarity = _best_text_conflict(
                feature_text, target_sub, state, TEXT_SPLIT_THRESHOLD
            )
            color_conflict = _color_conflict(
                classification.get("dominant_color", ""),
                _group_dominant_colors(target_sub, state),
            )

            # Text, when available, DECIDES on its own more lenient
            # threshold rather than being ORed alongside embedding —
            # confirmed live that treating a borderline-low embedding
            # score as an independent veto split apart photos with
            # near-identical (0.80-1.0) feature-text overlap. Embedding
            # only decides when there's no usable text on one/both
            # sides to compare directly. color_conflict is checked
            # separately either way — general pattern-text overlap
            # alone let two different colors through when they shared
            # almost all their OTHER pattern vocabulary (see
            # _color_conflict's docstring).
            if best_text_similarity is not None:
                should_split = (
                    type_conflict
                    or color_conflict
                    or best_text_similarity < TEXT_SPLIT_THRESHOLD
                )
            else:
                should_split = (
                    type_conflict
                    or color_conflict
                    or best_similarity < SPLIT_SIMILARITY_THRESHOLD
                )

            _log_decision(
                state, "split_check", key,
                {
                    "type_conflict": type_conflict,
                    "color_conflict": color_conflict,
                    "embedding_similarity": round(best_similarity, 4),
                    "text_similarity": (
                        round(best_text_similarity, 4)
                        if best_text_similarity is not None else None
                    ),
                    "split": should_split,
                },
                None, best_similarity, SPLIT_SIMILARITY_THRESHOLD,
            )

            if should_split:
                sub_clusters.append([key])
            else:
                target_sub.append(key)

        validated.extend(sub_clusters)

    return validated, deferred


def _group_aggregate_text(group_keys, state):
    """Aggregate color/pattern/features text across a group's photos,
    used as the "fabric continuity" comparison target for rescue
    scoring.
    """
    parts = []
    for key in group_keys:
        classification = state["images"].get(key, {}).get("classification", {})
        parts.append(classification.get("dominant_color", ""))
        parts.append(classification.get("pattern_description", ""))
        parts.append(classification.get("distinguishing_features", ""))
    return " ".join(p for p in parts if p)


def _group_dominant_garment_type(group_keys, state):
    counts = {}
    for key in group_keys:
        classification = state["images"].get(key, {}).get("classification", {})
        role = classification.get("image_role", "other")
        if role in _ROLES_WITHOUT_OWN_GARMENT_TYPE:
            continue
        garment_type = classification.get("garment_type", "other")
        if garment_type != "not_a_garment":
            counts[garment_type] = counts.get(garment_type, 0) + 1

    if not counts:
        return None

    return max(counts, key=counts.get)


def _group_dominant_colors(group_keys, state):
    """All dominant_color tokens seen across the group, pooled (not
    just the single most common) - a group photographed under
    slightly different lighting might read as "dark blue" in one shot
    and "navy" in another, so the comparison in _color_conflict below
    checks for ANY shared color word, not majority agreement.
    """
    tokens = set()
    for key in group_keys:
        classification = state["images"].get(key, {}).get("classification", {})
        tokens |= _tokenize(classification.get("dominant_color", ""))
    return tokens


def _color_conflict(candidate_color, group_color_tokens):
    """True only on a CLEAN, total color-word mismatch - e.g. "khaki"
    vs "olive green" share nothing, but "dark blue" vs "medium blue"
    share "blue" and "gray" vs "heather gray" share "gray", neither of
    which should be treated as a conflict (same color, different
    wording/lighting). Added after a real batch showed general
    pattern-text overlap alone wasn't enough to separate two
    genuinely-different-colored items that happened to share almost
    all their OTHER pattern vocabulary ("plaid pattern with $COLOR,
    white, and burgundy stripes") - color is only 1-2 tokens out of
    10+ shared generic ones, so it needs its own dedicated check
    rather than being diluted into the general text-overlap score.
    """
    candidate_tokens = _tokenize(candidate_color)
    if not candidate_tokens or not group_color_tokens:
        return False
    return not (candidate_tokens & group_color_tokens)


def _group_background_text(group_keys, state):
    counts = {}
    for key in group_keys:
        classification = state["images"].get(key, {}).get("classification", {})
        background = classification.get("background_description", "")
        if background:
            counts[background] = counts.get(background, 0) + 1

    if not counts:
        return ""

    return max(counts, key=counts.get)


_TAG_TEXT_GARMENT_HINTS = {
    "jeans": {"bootcut", "jean", "jeans", "denim", "waist"},
    "pants": {"waist", "trouser", "pants", "inseam"},
    "shorts": {"waist", "short", "shorts"},
    "dress": {"dress"},
    "skirt": {"skirt"},
    "jacket": {"jacket", "coat"},
    "sweater": {"sweater", "knit"},
    "shirt": {"shirt", "tee", "top", "blouse"},
}


def _tag_compatibility_score(visible_text, candidate_garment_type):
    """1.0 if compatible or neutral (no strong signal either way), 0.0
    if the tag text directly contradicts the candidate group's
    garment_type (e.g. tag mentions "dress" but the candidate group is
    jeans). Deliberately conservative: only penalizes on a genuine
    contradiction, never rewards beyond neutral for an absent/vague
    tag (most photos have no readable tag text at all, so this can't
    dominate the combined rescue score either way).
    """
    if not visible_text or not candidate_garment_type:
        return 1.0

    text_tokens = _tokenize(visible_text)

    mentioned_types = {
        garment_type
        for garment_type, hints in _TAG_TEXT_GARMENT_HINTS.items()
        if text_tokens & hints
    }

    if not mentioned_types:
        return 1.0

    if candidate_garment_type in mentioned_types:
        return 1.0

    return 0.0


def _log_decision(state, decision_type, photo_key, signals_dict, chosen_group, score, threshold):
    from datetime import datetime
    state["reasoning_log"].append({
        "decision_type": decision_type,
        "photo": photo_key,
        "signals": signals_dict,
        "chosen_group": chosen_group,
        "score": round(score, 4) if score is not None else None,
        "threshold": threshold,
        "timestamp": datetime.now().isoformat(),
    })


def _garment_type_conflict(photo_classification, candidate_garment_type):
    """Hard disqualifier, mirroring how Step 2 already treats a
    garment_type disagreement as a hard split trigger rather than a
    weighted vote.

    The vision model classifies garment_type for EVERY photo,
    including detail_shot/flaw_closeup ones - a detail shot of a
    waistband still gets labeled "shorts" or "jeans" even though that
    field isn't used for Step 2's cluster-identity voting (a close-up
    of a zipper alone might not always reflect the item type
    reliably). But confirmed live that IGNORING it in Step 3 rescue
    scoring let a real mismatch through: a shorts detail_shot ("light
    blue denim, embroidered brand name on waistband, metal button")
    scored 0.84 combined against a JEANS group purely on embedding +
    fabric-text similarity (both photos happened to mention waistband
    branding), clearing RESCUE_ACCEPT_THRESHOLD despite being
    explicitly labeled "shorts". A soft per-signal penalty isn't
    enough to stop this (even zeroing that one signal out would have
    still cleared 0.65) - disqualifying outright is the correct
    strength of signal here, same as Step 2's treatment.
    """
    photo_type = photo_classification.get("garment_type", "other")
    if photo_type in ("other", "not_a_garment") or not candidate_garment_type:
        return False
    return photo_type != candidate_garment_type


def _score_rescue_candidate(photo_key, candidate_cluster, state):
    """The rescue score for one photo against one candidate cluster
    (which may be a single other orphan photo, not just a pre-formed
    time cluster - see rescue_orphans()'s second pass). Returns
    (combined_score, component_scores_dict).
    """
    photo_signals = state["images"].get(photo_key, {})
    photo_classification = photo_signals.get("classification", {})

    candidate_garment_type = _group_dominant_garment_type(candidate_cluster, state)

    if _garment_type_conflict(photo_classification, candidate_garment_type):
        return 0.0, {
            "embedding_sim": 0.0,
            "fabric_continuity": 0.0,
            "tag_compatibility": 0.0,
            "background_match": 0.0,
            "garment_type_conflict": True,
            "combined": 0.0,
        }

    photo_embedding = photo_signals.get("embedding", [])
    photo_text = _attribute_text(photo_classification)
    photo_background = photo_classification.get("background_description", "")
    photo_visible_text = photo_classification.get("visible_text", "")

    group_embeddings = [
        state["images"].get(k, {}).get("embedding", []) for k in candidate_cluster
    ]
    embedding_sim = max(
        (_cosine_similarity(photo_embedding, other) for other in group_embeddings),
        default=0.0,
    )

    fabric_continuity = _overlap_coefficient(
        photo_text, _group_aggregate_text(candidate_cluster, state)
    )

    tag_compatibility = _tag_compatibility_score(
        photo_visible_text, candidate_garment_type
    )

    background_match = _jaccard_similarity(
        photo_background, _group_background_text(candidate_cluster, state)
    )

    combined = (
        RESCUE_WEIGHTS["embedding_sim"] * embedding_sim
        + RESCUE_WEIGHTS["fabric_continuity"] * fabric_continuity
        + RESCUE_WEIGHTS["tag_compatibility"] * tag_compatibility
        + RESCUE_WEIGHTS["background_match"] * background_match
    )

    component_scores = {
        "embedding_sim": round(embedding_sim, 4),
        "fabric_continuity": round(fabric_continuity, 4),
        "tag_compatibility": round(tag_compatibility, 4),
        "background_match": round(background_match, 4),
        "garment_type_conflict": False,
        "combined": round(combined, 4),
    }

    return combined, component_scores


def rescue_orphans(clusters, orphan_keys, state):
    """Step 3 - the critical pass for the user's tag problem. Runs on
    every "orphan" key handed in: untimestamped photos, plus EVERY
    tag_closeup/detail_shot/flaw_closeup photo (validate_and_split_
    clusters() now defers all of them here rather than auto-attaching
    them to whatever time cluster happened to be open — see that
    function's docstring for why).

    Two passes:

    Pass A - score each rescue candidate against every PRE-EXISTING
    time cluster on four independent sub-signals, attach to the best
    match if it clears RESCUE_ACCEPT_THRESHOLD.

    Pass B - candidates Pass A couldn't place (including the common
    case of a batch with NO timestamped clusters at all - e.g. every
    photo lacks EXIF, confirmed as a real scenario during testing, not
    hypothetical) get compared against EACH OTHER, not just against
    pre-formed clusters. Without this, two untimestamped photos of the
    same physical item (a tag closeup and its garment shot, say) would
    always land in Needs Attention even when their attributes clearly
    match, simply because neither had a time-based cluster to anchor
    to. Greedy single-link grouping: each remaining candidate joins
    the first new cluster it scores above threshold against, or starts
    a new one of its own.

    Every scored candidate (not just the winner) gets logged so the
    weighting can be tuned from real data later.

    Returns (clusters, needs_attention_keys) - clusters is mutated in
    place (rescued photos appended to their winning cluster, and any
    Pass-B clusters appended as brand new clusters) and also returned
    for clarity.
    """
    rescue_candidates = list(orphan_keys)
    still_unplaced = []

    # --- Pass A: against pre-existing time clusters ---
    for photo_key in rescue_candidates:
        best_score = -1.0
        best_cluster_index = None
        all_candidate_scores = []

        for cluster_index, cluster in enumerate(clusters):
            if photo_key in cluster:
                continue

            combined, component_scores = _score_rescue_candidate(
                photo_key, cluster, state
            )
            all_candidate_scores.append((cluster_index, component_scores))

            if combined > best_score:
                best_score = combined
                best_cluster_index = cluster_index

        if best_cluster_index is not None and best_score >= RESCUE_ACCEPT_THRESHOLD:
            clusters[best_cluster_index].append(photo_key)
            _log_decision(
                state, "rescue_attach", photo_key,
                {"candidates": [s for _, s in all_candidate_scores], "pass": "A"},
                best_cluster_index, best_score, RESCUE_ACCEPT_THRESHOLD,
            )
        else:
            still_unplaced.append(
                (photo_key, best_score if best_score >= 0 else None, all_candidate_scores)
            )

    # --- Pass B: remaining candidates against EACH OTHER ---
    needs_attention = []
    new_clusters = []

    for photo_key, pass_a_score, pass_a_candidates in still_unplaced:
        best_score = -1.0
        best_new_cluster_index = None
        peer_candidate_scores = []

        for cluster_index, cluster in enumerate(new_clusters):
            combined, component_scores = _score_rescue_candidate(
                photo_key, cluster, state
            )
            peer_candidate_scores.append((cluster_index, component_scores))

            if combined > best_score:
                best_score = combined
                best_new_cluster_index = cluster_index

        if best_new_cluster_index is not None and best_score >= RESCUE_ACCEPT_THRESHOLD:
            new_clusters[best_new_cluster_index].append(photo_key)
            _log_decision(
                state, "rescue_attach", photo_key,
                {
                    "candidates": [s for _, s in pass_a_candidates] + [s for _, s in peer_candidate_scores],
                    "pass": "B",
                },
                f"new_cluster_{best_new_cluster_index}", best_score, RESCUE_ACCEPT_THRESHOLD,
            )
        else:
            new_clusters.append([photo_key])
            _log_decision(
                state, "rescue_unattached", photo_key,
                {
                    "candidates": [s for _, s in pass_a_candidates] + [s for _, s in peer_candidate_scores],
                    "pass": "B",
                },
                None,
                best_score if best_score >= 0 else pass_a_score,
                RESCUE_ACCEPT_THRESHOLD,
            )

    # Pass-B clusters with 2+ photos are real rescued groups; ones that
    # never found a peer are still singletons with no confident
    # placement anywhere - those go to Needs Attention, not into a
    # group of one built purely out of desperation.
    for cluster in new_clusters:
        if len(cluster) > 1:
            clusters.append(cluster)
        else:
            needs_attention.extend(cluster)

    return clusters, needs_attention


def score_groups(clusters, state, rescued_keys):
    """Step 4 - confidence scoring. image_fit_confidence is 1.0 for
    photos placed directly by Step 1/2 (time + visual agreement both
    held), or the raw rescue combined score for photos placed by Step
    3 (already in the 0.65-1.0 range by construction, since anything
    below RESCUE_ACCEPT_THRESHOLD never got attached).
    group_confidence is the MINIMUM image_fit_confidence in the group,
    not an average - one weak image should visibly drag the whole
    group's color down rather than being diluted out, per the "flag
    the image, not just hide it in an average" requirement.
    """
    groups = []

    for cluster in clusters:
        if not cluster:
            continue

        image_fit = {}
        for key in cluster:
            if key in rescued_keys:
                # Pull the actual rescue score back out of the log
                # rather than recomputing it.
                matching = [
                    entry for entry in state["reasoning_log"]
                    if entry["decision_type"] == "rescue_attach" and entry["photo"] == key
                ]
                image_fit[key] = matching[-1]["score"] if matching else RESCUE_ACCEPT_THRESHOLD
            else:
                image_fit[key] = 1.0

        group_confidence = min(image_fit.values())

        has_tag_photo = any(
            state["images"].get(k, {}).get("classification", {}).get("image_role") == "tag_closeup"
            for k in cluster
        )
        has_rescued_photo = any(k in rescued_keys for k in cluster)
        single_photo = len(cluster) == 1

        garment_types_present = _cluster_garment_types(cluster, state)
        contradictory = len(garment_types_present) > 1

        if contradictory:
            status = "red"
        elif group_confidence < GROUP_CONFIDENCE_GREEN_THRESHOLD or not has_tag_photo or single_photo or has_rescued_photo:
            status = "yellow"
        else:
            status = "green"

        groups.append({
            "photo_paths": list(cluster),
            "image_fit_confidence": {k: round(v, 4) for k, v in image_fit.items()},
            "group_confidence": round(group_confidence, 4),
            "status": status,
            "has_tag_photo": has_tag_photo,
            "has_rescued_photo": has_rescued_photo,
            "dominant_garment_type": _group_dominant_garment_type(cluster, state),
        })

        _log_decision(
            state, "group_scored", None,
            {
                "photo_count": len(cluster),
                "has_tag_photo": has_tag_photo,
                "has_rescued_photo": has_rescued_photo,
                "contradictory_garment_types": contradictory,
            },
            None, group_confidence, GROUP_CONFIDENCE_GREEN_THRESHOLD,
        )

    return groups


def _group_representative_photo(photo_paths, state):
    """Pick one photo to represent a group when scoring it against
    OTHER groups - prefers a full_garment shot (carries the most
    identity signal), falling back to whatever's first.
    """
    for key in photo_paths:
        role = state["images"].get(key, {}).get("classification", {}).get("image_role")
        if role == "full_garment":
            return key
    return photo_paths[0] if photo_paths else None


def find_best_match_for_group(group, all_groups, state):
    """For a non-green group, find the single most likely OTHER group
    it might actually belong with - reuses the exact same 4-signal
    scoring rescue_orphans() uses for orphan photos, just applied
    group-to-group via each group's representative photo. Returns
    (group_id, score) or (None, None) if nothing scores above zero.
    """
    candidate_key = _group_representative_photo(group["photo_paths"], state)
    if candidate_key is None:
        return None, None

    best_group_id = None
    best_score = -1.0

    for other in all_groups:
        if other["group_id"] == group["group_id"]:
            continue
        score, _ = _score_rescue_candidate(candidate_key, other["photo_paths"], state)
        if score > best_score:
            best_score = score
            best_group_id = other["group_id"]

    if best_group_id is None or best_score <= 0:
        return None, None

    return best_group_id, best_score


def find_best_match_for_stray(photo_key, all_groups, state):
    """Same idea as find_best_match_for_group but for a single
    unassigned (Needs Attention) photo.
    """
    best_group_id = None
    best_score = -1.0

    for group in all_groups:
        score, _ = _score_rescue_candidate(photo_key, group["photo_paths"], state)
        if score > best_score:
            best_score = score
            best_group_id = group["group_id"]

    if best_group_id is None or best_score <= 0:
        return None, None

    return best_group_id, best_score


def compute_possible_matches(state):
    """Populates, on every non-green group, which OTHER group it most
    likely actually belongs with (possible_match_group_id/score) - and
    on state["stray_matches"], the same for every Needs Attention
    photo. This is what lets the review UI position an uncertain
    group/photo next to its likely match instead of leaving the user
    to guess which of N groups a stray probably belongs to.

    Cheap - pure local scoring over already-cached signals, no API
    calls. Called after run_grouping_pipeline() and after every manual
    mutation, so matches never go stale.
    """
    for group in state["groups"]:
        if group["status"] == "green":
            group["possible_match_group_id"] = None
            group["possible_match_score"] = None
            continue

        match_id, match_score = find_best_match_for_group(
            group, state["groups"], state
        )
        group["possible_match_group_id"] = match_id
        group["possible_match_score"] = (
            round(match_score, 4) if match_score is not None else None
        )

    stray_matches = {}
    for photo_key in state["needs_attention"]:
        match_id, match_score = find_best_match_for_stray(
            photo_key, state["groups"], state
        )
        stray_matches[photo_key] = {
            "group_id": match_id,
            "score": round(match_score, 4) if match_score is not None else None,
        }
    state["stray_matches"] = stray_matches

    return state


def run_grouping_pipeline(image_paths, state, on_progress=None):
    """Orchestrates the full pipeline: extract signals (cached) ->
    cluster by time -> validate/split -> rescue -> score. Mutates and
    returns state with state["groups"]/state["needs_attention"]
    replaced by this run's result. Does NOT touch
    state["manual_corrections"]/state["undo_stack"] - those persist
    across grouping re-runs.

    state["reasoning_log"] IS reset at the start of each run — it
    represents the reasoning behind the CURRENT grouping, not an
    ever-growing history. Without this, re-sorting the same batch
    (after adding photos, say) would leave stale entries from the
    previous run mixed in with new ones, polluting the "Why this
    rating?" view for photos that stayed in the same group across
    runs.

    state["next_group_id"] is also reset to 1 each run, for the same
    reason group numbering should read as "1, 2, 3..." rather than
    climbing higher every time the user re-sorts — safe because
    state["groups"] is fully replaced by this run's result anyway
    (any manually-created groups from a prior review session don't
    survive a fresh sort regardless of their id).
    """
    state["reasoning_log"] = []
    state["next_group_id"] = 1

    state = extract_all_signals(image_paths, state, on_progress=on_progress)

    clusters, untimestamped = cluster_by_time(image_paths, state)
    clusters, deferred_tag_detail_photos = validate_and_split_clusters(clusters, state)

    # Contradictory (irreconcilable) groups after splitting are marked
    # red in score_groups, not force-merged or force-split further -
    # per the plan's "surfaced instead of silently picking one".

    orphan_keys = list(untimestamped) + deferred_tag_detail_photos

    clusters, needs_attention_from_rescue = rescue_orphans(
        clusters, orphan_keys, state
    )

    rescued_keys = {
        entry["photo"] for entry in state["reasoning_log"]
        if entry["decision_type"] == "rescue_attach"
    }

    groups = score_groups(clusters, state, rescued_keys)

    # Anything red gets pulled OUT of groups and into needs_attention
    # at the image level too, per "never force-assigned into a group" -
    # a contradictory group's photos are surfaced individually rather
    # than left inside a group card labeled red with no clear next
    # step.
    final_groups = []
    needs_attention = list(needs_attention_from_rescue)

    for group in groups:
        if group["status"] == "red":
            needs_attention.extend(group["photo_paths"])
        else:
            final_groups.append(group)

    for index, group in enumerate(final_groups):
        group["group_id"] = state["next_group_id"] + index

    state["next_group_id"] += len(final_groups)
    state["groups"] = final_groups
    state["needs_attention"] = needs_attention

    state = compute_possible_matches(state)

    save_bulk_sort_state(state)
    return state


# ============================================================
# MANUAL REASSIGNMENT
# ============================================================

def _push_undo_snapshot(state):
    snapshot = {
        "groups": copy.deepcopy(state["groups"]),
        "needs_attention": copy.deepcopy(state["needs_attention"]),
    }
    state["undo_stack"].append(snapshot)
    if len(state["undo_stack"]) > _UNDO_STACK_MAX:
        state["undo_stack"].pop(0)


def undo(state):
    if not state["undo_stack"]:
        return state

    snapshot = state["undo_stack"].pop()
    state["groups"] = snapshot["groups"]
    state["needs_attention"] = snapshot["needs_attention"]
    state = compute_possible_matches(state)
    save_bulk_sort_state(state)
    return state


def _find_group(state, group_id):
    for group in state["groups"]:
        if group["group_id"] == group_id:
            return group
    return None


def _remove_photo_everywhere(state, photo_key):
    for group in state["groups"]:
        if photo_key in group["photo_paths"]:
            group["photo_paths"].remove(photo_key)
            group["image_fit_confidence"].pop(photo_key, None)
    state["groups"] = [g for g in state["groups"] if g["photo_paths"]]
    if photo_key in state["needs_attention"]:
        state["needs_attention"].remove(photo_key)


def _rescore_group(state, group):
    """Re-run Step 4 scoring on a single group after a manual
    mutation - cheap, pure local computation over already-cached
    signals, no API calls.
    """
    cluster = group["photo_paths"]

    image_fit = {
        k: group["image_fit_confidence"].get(k, 1.0) for k in cluster
    }
    group_confidence = min(image_fit.values()) if image_fit else 0.0

    has_tag_photo = any(
        state["images"].get(k, {}).get("classification", {}).get("image_role") == "tag_closeup"
        for k in cluster
    )
    has_rescued_photo = group.get("has_rescued_photo", False)
    single_photo = len(cluster) == 1
    garment_types_present = _cluster_garment_types(cluster, state)
    contradictory = len(garment_types_present) > 1

    if contradictory:
        status = "red"
    elif group_confidence < GROUP_CONFIDENCE_GREEN_THRESHOLD or not has_tag_photo or single_photo or has_rescued_photo:
        status = "yellow"
    else:
        status = "green"

    group["image_fit_confidence"] = {k: round(v, 4) for k, v in image_fit.items()}
    group["group_confidence"] = round(group_confidence, 4)
    group["status"] = status
    group["has_tag_photo"] = has_tag_photo
    group["dominant_garment_type"] = _group_dominant_garment_type(cluster, state)


def move_photos(state, photo_keys, target_group_id, manual=True):
    """Move one or more photos into an existing group. A manually
    moved photo is treated as a fully-trusted (1.0 fit) placement -
    the user just confirmed it by hand.
    """
    _push_undo_snapshot(state)

    target = _find_group(state, target_group_id)
    if target is None:
        return state

    from datetime import datetime
    for photo_key in photo_keys:
        origin_group = next(
            (g for g in state["groups"] if photo_key in g["photo_paths"]), None
        )
        origin_id = origin_group["group_id"] if origin_group else None
        origin_signals = state["images"].get(photo_key, {}).get("classification", {})

        _remove_photo_everywhere(state, photo_key)
        target["photo_paths"].append(photo_key)
        target["image_fit_confidence"][photo_key] = 1.0

        if manual:
            state["manual_corrections"].append({
                "photo": photo_key,
                "from_group": origin_id,
                "to_group": target_group_id,
                "original_signals": origin_signals,
                "timestamp": datetime.now().isoformat(),
            })

    _rescore_group(state, target)
    state["groups"] = [g for g in state["groups"] if g["photo_paths"]]
    state = compute_possible_matches(state)
    save_bulk_sort_state(state)
    return state


def create_group_from_photos(state, photo_keys):
    _push_undo_snapshot(state)

    for photo_key in photo_keys:
        _remove_photo_everywhere(state, photo_key)

    new_group = {
        "group_id": state["next_group_id"],
        "photo_paths": list(photo_keys),
        "image_fit_confidence": {k: 1.0 for k in photo_keys},
        "group_confidence": 1.0,
        "status": "yellow",
        "has_tag_photo": False,
        "has_rescued_photo": False,
        "dominant_garment_type": None,
    }
    state["next_group_id"] += 1
    state["groups"].append(new_group)
    _rescore_group(state, new_group)
    state = compute_possible_matches(state)
    save_bulk_sort_state(state)
    return state


def merge_groups(state, group_id_a, group_id_b):
    _push_undo_snapshot(state)

    group_a = _find_group(state, group_id_a)
    group_b = _find_group(state, group_id_b)

    if group_a is None or group_b is None or group_a is group_b:
        return state

    group_a["photo_paths"].extend(group_b["photo_paths"])
    group_a["image_fit_confidence"].update(group_b["image_fit_confidence"])
    state["groups"] = [g for g in state["groups"] if g["group_id"] != group_id_b]

    _rescore_group(state, group_a)
    state = compute_possible_matches(state)
    save_bulk_sort_state(state)
    return state


def split_group(state, group_id, photo_keys_for_new_group):
    _push_undo_snapshot(state)

    group = _find_group(state, group_id)
    if group is None:
        return state

    remaining = [p for p in group["photo_paths"] if p not in photo_keys_for_new_group]
    moved = [p for p in group["photo_paths"] if p in photo_keys_for_new_group]

    if not moved or not remaining:
        # Nothing to split - either selection is empty or covers the
        # whole group.
        state["undo_stack"].pop()
        return state

    group["photo_paths"] = remaining
    for key in moved:
        group["image_fit_confidence"].pop(key, None)

    new_group = {
        "group_id": state["next_group_id"],
        "photo_paths": moved,
        "image_fit_confidence": {
            k: group["image_fit_confidence"].get(k, 1.0) for k in moved
        },
        "group_confidence": 1.0,
        "status": "yellow",
        "has_tag_photo": False,
        "has_rescued_photo": group.get("has_rescued_photo", False),
        "dominant_garment_type": None,
    }
    state["next_group_id"] += 1
    state["groups"].append(new_group)

    _rescore_group(state, group)
    _rescore_group(state, new_group)
    state = compute_possible_matches(state)
    save_bulk_sort_state(state)
    return state


def send_to_needs_attention(state, photo_keys):
    _push_undo_snapshot(state)

    for photo_key in photo_keys:
        _remove_photo_everywhere(state, photo_key)
        if photo_key not in state["needs_attention"]:
            state["needs_attention"].append(photo_key)

    state = compute_possible_matches(state)
    save_bulk_sort_state(state)
    return state


# ============================================================
# CLI TEST ENTRY POINT
# Mirrors grouping.py's own __main__ block pattern - a fast way to
# test the pipeline against a real messy batch before any UI exists.
# ============================================================

if __name__ == "__main__":
    import sys

    folder = sys.argv[1] if len(sys.argv) > 1 else "uploads"
    paths = [
        str(p) for p in Path(folder).iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    ]

    print(f"Running bulk-sort pipeline on {len(paths)} photo(s) from {folder}...")

    def _progress(done, total):
        print(f"  ...{done}/{total}")

    try:
        state = load_bulk_sort_state()
        state = run_grouping_pipeline(paths, state, on_progress=_progress)

        print(f"\n{len(state['groups'])} group(s), {len(state['needs_attention'])} needing attention:\n")
        for group in state["groups"]:
            print(
                f"  Group {group['group_id']} [{group['status'].upper()}] "
                f"confidence={group['group_confidence']} "
                f"type={group['dominant_garment_type']} "
                f"photos={len(group['photo_paths'])}"
            )
            for photo_path in group["photo_paths"]:
                fit = group["image_fit_confidence"].get(photo_path)
                print(f"      {Path(photo_path).name}  fit={fit}")

        if state["needs_attention"]:
            print("\n  Needs Attention:")
            for photo_path in state["needs_attention"]:
                print(f"      {Path(photo_path).name}")

    except Exception as error:
        print(f"BULK SORT ERROR: {error}")
