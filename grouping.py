import base64
import json
import io
from pathlib import Path

from dotenv import load_dotenv

from ai_listing import vision_chat_completion


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}


# ============================================================
# CAMERA TIMESTAMP HELPERS
# ============================================================

def get_capture_timestamp(image_path):
    """
    Return the camera's EXIF capture timestamp as a datetime.

    Preference:
      DateTimeOriginal -> DateTimeDigitized -> DateTime

    If EXIF is unavailable, return None.
    """
    from PIL import Image, ExifTags
    from datetime import datetime

    image_path = Path(image_path)

    try:
        with Image.open(image_path) as image:
            exif = image.getexif()

            if not exif:
                return None

            tag_names = {
                tag_id: name
                for tag_id, name in ExifTags.TAGS.items()
            }

            preferred = (
                "DateTimeOriginal",
                "DateTimeDigitized",
                "DateTime",
            )

            values = {}

            for tag_id, value in exif.items():
                name = tag_names.get(tag_id)
                if name in preferred:
                    values[name] = value

            for name in preferred:
                value = values.get(name)

                if not value:
                    continue

                if isinstance(value, bytes):
                    value = value.decode(
                        "utf-8",
                        errors="ignore",
                    )

                value = str(value).strip()

                try:
                    return datetime.strptime(
                        value,
                        "%Y:%m:%d %H:%M:%S",
                    )
                except ValueError:
                    continue

    except Exception:
        return None

    return None


def sort_paths_by_capture_time(image_paths):
    """
    Sort by actual camera capture time.

    Timestamped photos are ordered chronologically.
    Untimestamped photos retain their original upload order and are placed
    after timestamped photos so we never invent a timestamp.
    """
    records = []

    for original_index, path in enumerate(
        image_paths
    ):
        path = Path(path)

        records.append({
            "path": path,
            "original_index": original_index,
            "capture_time": get_capture_timestamp(path),
        })

    records.sort(
        key=lambda record: (
            record["capture_time"] is None,
            record["capture_time"]
            if record["capture_time"] is not None
            else record["original_index"],
            record["original_index"],
        )
    )

    return records


# ============================================================
# FILES
# ============================================================

def get_image_files(folder="uploads"):

    folder = Path(folder)

    if not folder.exists():
        return []

    files = []

    for file in folder.iterdir():

        if not file.is_file():
            continue

        if file.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        files.append(file)

    return [
        record["path"]
        for record in sort_paths_by_capture_time(files)
    ]


# ============================================================
# IMAGE ENCODING
# ============================================================
# Already resizes to 1400px / quality 65 before sending — this part of
# the pipeline was fine. Left as-is other than a small in-process cache
# so a photo used across multiple stages/batches isn't re-encoded from
# scratch every time.

_DATA_URL_CACHE = {}
_DATA_URL_CACHE_MAX = 300


def image_to_data_url(image_path):
    """Compress vision images so each request stays comfortably small."""
    from PIL import Image, ImageOps

    image_path = Path(image_path)

    try:
        stat = image_path.stat()
        cache_key = (str(image_path.resolve()), stat.st_mtime_ns)
    except OSError:
        cache_key = None

    if cache_key is not None and cache_key in _DATA_URL_CACHE:
        return _DATA_URL_CACHE[cache_key]

    with Image.open(image_path) as image:
        try:
            image = ImageOps.exif_transpose(image)
        except Exception:
            pass

        if image.mode != "RGB":
            if "A" in image.getbands():
                background = Image.new("RGB", image.size, "white")
                background.paste(image, mask=image.getchannel("A"))
                image = background
            else:
                image = image.convert("RGB")

        max_dimension = 1400
        if max(image.size) > max_dimension:
            scale = max_dimension / max(image.size)
            image = image.resize(
                (
                    max(1, int(image.width * scale)),
                    max(1, int(image.height * scale)),
                ),
                Image.Resampling.LANCZOS,
            )

        buffer = io.BytesIO()
        image.save(
            buffer,
            format="JPEG",
            quality=65,
            optimize=True,
            progressive=True,
        )

    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    result = "data:image/jpeg;base64," + encoded

    if cache_key is not None:
        if len(_DATA_URL_CACHE) >= _DATA_URL_CACHE_MAX:
            _DATA_URL_CACHE.pop(next(iter(_DATA_URL_CACHE)))
        _DATA_URL_CACHE[cache_key] = result

    return result


# ============================================================
# OPENAI JSON
# ============================================================
# Vision calls in this file route through ai_listing.vision_chat_completion(),
# the shared multi-model fallback chain (gpt-4.1 -> gpt-4o -> mini tiers) —
# not a local single-model wrapper. One model getting rate-limited no
# longer stalls photo classification/grouping/ordering.

def parse_json_response(response):

    content = response.choices[0].message.content

    if not content:
        raise RuntimeError(
            "OpenAI returned an empty response."
        )

    content = content.strip()

    try:

        return json.loads(
            content
        )

    except json.JSONDecodeError:

        if content.startswith("```"):

            cleaned = (
                content
                .replace("```json", "")
                .replace("```", "")
                .strip()
            )

            try:

                return json.loads(
                    cleaned
                )

            except json.JSONDecodeError:
                pass

        raise RuntimeError(
            "OpenAI returned invalid JSON:\n"
            + content
        )


# ============================================================
# BUILD PHOTO CONTENT
# ============================================================

def build_photo_content(
    image_paths
):

    content = []

    for number, image_path in enumerate(
        image_paths,
        start=1
    ):

        content.append(
            {
                "type": "text",
                "text": (
                    "PHOTO "
                    + str(number)
                    + ": "
                    + Path(image_path).name
                ),
            }
        )

        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": image_to_data_url(
                        image_path
                    ),
                },
            }
        )

    return content


# ============================================================
# STAGE 1
# CLASSIFY PHOTOS
# ============================================================

def classify_photos(image_paths):
    """
    Classify in batches.

    PERFORMANCE FIX: batch size raised from 10 to 16. Fewer, slightly
    larger batches means fewer separate model round-trips (and fewer
    hits of the inter-request gap), which matters more than keeping any
    single request tiny now that the throttle gap itself is much
    smaller.
    """
    prompt = """
You are classifying clothing photography.

For EVERY photo, determine whether it is:

1. COMPLETE
   The photo shows enough of the actual clothing garment to identify it
   as a physical garment.

2. DETAIL
   The photo primarily shows supporting information such as a size tag,
   brand tag, care label, printed size, printed brand, neck label,
   graphic close-up, embroidery close-up, lace close-up, button/zipper
   close-up, or fabric close-up.

A photo does NOT need to show the entire garment to be COMPLETE.
If enough actual garment construction is visible, classify COMPLETE.

DO NOT GROUP PHOTOS IN THIS STEP.

Return ONLY JSON:
{
  "photos": [
    {"photo_number": 1, "type": "complete"},
    {"photo_number": 2, "type": "detail"}
  ]
}
"""

    complete = []
    detail = []

    CLASSIFY_BATCH = 16

    for start_index in range(0, len(image_paths), CLASSIFY_BATCH):
        batch_paths = image_paths[
            start_index:start_index + CLASSIFY_BATCH
        ]
        batch_numbers = list(
            range(
                start_index + 1,
                start_index + 1 + len(batch_paths)
            )
        )

        content = [{"type": "text", "text": prompt}]

        for number, image_path in zip(batch_numbers, batch_paths):
            content.append({
                "type": "text",
                "text": f"ORIGINAL PHOTO {number}: {image_path.name}",
            })
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": image_to_data_url(image_path),
                    "detail": "low"
                }
            })

        print(
            f"Stage 1 classification batch "
            f"{batch_numbers[0]}-{batch_numbers[-1]} "
            f"of {len(image_paths)}..."
        )

        response = vision_chat_completion(
            content,
            temperature=0,
            response_format={"type": "json_object"},
        )

        result = parse_json_response(response)
        raw_photos = result.get("photos", [])

        seen = set()

        if isinstance(raw_photos, list):
            for entry in raw_photos:
                if not isinstance(entry, dict):
                    continue

                try:
                    number = int(entry.get("photo_number"))
                except (ValueError, TypeError):
                    continue

                if number not in batch_numbers or number in seen:
                    continue

                seen.add(number)

                photo_type = str(
                    entry.get("type", "")
                ).lower().strip()

                if photo_type == "detail":
                    detail.append(number)
                else:
                    complete.append(number)

        for number in batch_numbers:
            if number not in seen:
                complete.append(number)

    return {
        "complete": sorted(set(complete)),
        "detail": sorted(set(detail)),
    }


# ============================================================
# STAGE 2
# GROUP COMPLETE GARMENTS ONLY
# ============================================================

def _group_complete_batch(
    image_paths,
    batch_numbers
):
    prompt = """
You are an expert clothing resale photography organizer.

You are looking ONLY at COMPLETE GARMENT photos.
Determine which photos show the SAME physical garment.

Only combine photos with strong visual evidence:
- exact same graphic or embroidery
- exact same unique pattern
- exact same buttons/zipper/pockets/lace
- exact same distinctive construction or wear
- clearly recognizable front/back relationship

Do NOT merge merely because of brand, color, size, material, category,
aesthetic, or similar shape/style.

Front and back of the same physical garment = ONE ITEM.

If uncertain, KEEP THEM SEPARATE.

Every supplied photo must appear exactly once.

Return ONLY JSON:
{
  "groups": [
    {
      "photo_numbers": [1, 2],
      "reason": "Same physical garment."
    }
  ]
}
"""

    content = [{"type": "text", "text": prompt}]

    for number in batch_numbers:
        image_path = image_paths[number - 1]
        content.append({
            "type": "text",
            "text": f"ORIGINAL PHOTO {number}: {image_path.name}"
        })
        content.append({
            "type": "image_url",
            "image_url": {
                "url": image_to_data_url(image_path)
            }
        })

    response = vision_chat_completion(
        content,
        temperature=0,
        response_format={"type": "json_object"},
    )

    result = parse_json_response(response)
    raw_groups = result.get("groups", [])
    if not isinstance(raw_groups, list):
        raw_groups = []

    groups = []
    used = set()

    for raw in raw_groups:
        if not isinstance(raw, dict):
            continue

        numbers = []
        for value in raw.get("photo_numbers", []):
            try:
                number = int(value)
            except (ValueError, TypeError):
                continue

            if number not in batch_numbers or number in used:
                continue

            numbers.append(number)
            used.add(number)

        if numbers:
            groups.append({
                "photo_numbers": numbers,
                "reason": str(
                    raw.get(
                        "reason",
                        "Same physical garment."
                    )
                )
            })

    for number in batch_numbers:
        if number not in used:
            groups.append({
                "photo_numbers": [number],
                "reason": "Kept separate because no confident match was found."
            })

    return groups


def group_complete_garments(
    image_paths,
    complete_numbers
):
    """
    PERFORMANCE FIX: batch size raised from 10 to 14 (fewer round trips),
    and the artificial time.sleep(3) between batches — which existed
    purely to "give the rolling token window breathing room" on top of
    an already-throttled request loop — has been removed. The 429
    backoff inside vision_chat_completion already handles genuine rate limits.
    """
    if not complete_numbers:
        return []

    BATCH_SIZE = 14
    OVERLAP = 2

    all_groups = []
    start = 0

    while start < len(complete_numbers):

        end = min(
            start + BATCH_SIZE,
            len(complete_numbers)
        )

        batch = complete_numbers[start:end]

        print(
            f"Grouping timestamp batch "
            f"{batch[0]}-{batch[-1]} "
            f"of {len(complete_numbers)}..."
        )

        all_groups.extend(
            _group_complete_batch(
                image_paths,
                batch
            )
        )

        if end >= len(complete_numbers):
            break

        start = max(
            end - OVERLAP,
            start + 1,
        )

    merged = []
    photo_to_group = {}

    for group in all_groups:
        numbers = list(dict.fromkeys(group["photo_numbers"]))
        matching_indexes = sorted({
            photo_to_group[n]
            for n in numbers
            if n in photo_to_group
        })

        if not matching_indexes:
            merged.append({
                "photo_numbers": numbers,
                "reason": group.get("reason", "")
            })
            index = len(merged) - 1
        else:
            index = matching_indexes[0]
            merged[index]["photo_numbers"].extend(numbers)
            merged[index]["photo_numbers"] = list(
                dict.fromkeys(
                    merged[index]["photo_numbers"]
                )
            )

            for other in reversed(matching_indexes[1:]):
                merged[index]["photo_numbers"].extend(
                    merged[other]["photo_numbers"]
                )
                merged[index]["photo_numbers"] = list(
                    dict.fromkeys(
                        merged[index]["photo_numbers"]
                    )
                )
                del merged[other]

                photo_to_group.clear()
                for i, existing in enumerate(merged):
                    for n in existing["photo_numbers"]:
                        photo_to_group[n] = i

        for n in merged[index]["photo_numbers"]:
            photo_to_group[n] = index

    seen = set()
    final = []

    for group in merged:
        numbers = []
        for number in sorted(group["photo_numbers"]):
            if number in complete_numbers and number not in seen:
                numbers.append(number)
                seen.add(number)

        if numbers:
            final.append({
                "photo_numbers": numbers,
                "reason": group.get("reason", "")
            })

    for number in complete_numbers:
        if number not in seen:
            final.append({
                "photo_numbers": [number],
                "reason": "Unmatched complete photo kept separate."
            })

    return final


def reconcile_garment_groups(image_paths, garment_groups):
    """
    One small global pass to merge only groups that are clearly the same
    physical garment. It sees one representative image per group.
    """
    if len(garment_groups) < 2:
        return garment_groups

    prompt = """
You are reconciling clothing groups created by separate batches.

Each CANDIDATE GROUP represents one proposed physical garment.
Determine whether any two groups are actually the SAME physical garment.

MERGE ONLY with strong visual evidence:
- exact graphic/embroidery
- exact distinctive pattern
- exact pocket/button/zipper construction
- unmistakable front/back relationship
- clearly identical garment details

DO NOT merge based only on brand, color, size, category, or general style.

If uncertain, KEEP GROUPS SEPARATE.

Return ONLY JSON:
{
  "merges": [
    {"group_a": 1, "group_b": 4, "reason": "Same physical garment."}
  ]
}
"""

    content = [{"type": "text", "text": prompt}]

    for group_number, group in enumerate(garment_groups, start=1):
        numbers = group.get("photo_numbers", [])
        if not numbers:
            continue

        number = numbers[0]
        image_path = image_paths[number - 1]

        content.append({
            "type": "text",
            "text": f"CANDIDATE GROUP {group_number} — PHOTO {number}"
        })
        content.append({
            "type": "image_url",
            "image_url": {
                "url": image_to_data_url(image_path),
                "detail": "low"
            }
        })

    print(
        f"Reconciling {len(garment_groups)} groups in one lightweight pass..."
    )

    response = vision_chat_completion(
        content,
        temperature=0,
        response_format={"type": "json_object"},
    )

    result = parse_json_response(response)
    merges = result.get("merges", [])

    if not isinstance(merges, list):
        return garment_groups

    parent = list(range(len(garment_groups)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for merge in merges:
        if not isinstance(merge, dict):
            continue
        try:
            a = int(merge.get("group_a")) - 1
            b = int(merge.get("group_b")) - 1
        except (TypeError, ValueError):
            continue

        if 0 <= a < len(garment_groups) and 0 <= b < len(garment_groups):
            union(a, b)

    merged = []
    root_to_index = {}

    for index, group in enumerate(garment_groups):
        root = find(index)

        if root not in root_to_index:
            root_to_index[root] = len(merged)
            merged.append({
                "photo_numbers": [],
                "reason": group.get("reason", "")
            })

        target = merged[root_to_index[root]]
        target["photo_numbers"].extend(group.get("photo_numbers", []))

    for group in merged:
        group["photo_numbers"] = sorted(set(group["photo_numbers"]))

    return [g for g in merged if g["photo_numbers"]]



def reconcile_timestamp_boundaries(
    image_paths,
    garment_groups,
):
    """
    Verify ONLY neighboring timestamp groups.

    Group 1 can only be compared to group 2, group 2 to group 3, etc.
    The model may merge a neighboring pair only with very strong exact
    physical-garment evidence.
    """

    if len(garment_groups) < 2:
        return garment_groups

    prompt = """
You are verifying boundaries between clothing-photo groups that are already
ordered by CAMERA CAPTURE TIMESTAMP.

For each neighboring pair of groups, decide whether they are actually the
SAME PHYSICAL GARMENT accidentally split at a batch boundary.

MERGE ONLY when there is unmistakable physical identity:
- exact same garment construction
- exact same graphic/embroidery
- exact same unique distressing
- exact same pockets/buttons/zippers
- obvious front/back relationship
- unmistakable matching physical details

DO NOT merge because of:
- same brand
- same color
- same size
- same category
- same style
- similar appearance

If uncertain, KEEP THEM SEPARATE.

You are NOT allowed to compare non-neighboring groups.

Return ONLY JSON:
{
  "merges": [
    {
      "left_group": 1,
      "right_group": 2,
      "merge": true,
      "confidence": 0.98
    }
  ]
}
"""

    merged = []
    index = 0

    while index < len(garment_groups):

        if index >= len(garment_groups) - 1:
            merged.append(
                garment_groups[index]
            )
            break

        left = garment_groups[index]
        right = garment_groups[index + 1]

        left_numbers = list(
            dict.fromkeys(
                left.get(
                    "photo_numbers",
                    [],
                )
            )
        )

        right_numbers = list(
            dict.fromkeys(
                right.get(
                    "photo_numbers",
                    [],
                )
            )
        )

        content = [
            {
                "type": "text",
                "text": prompt,
            },
            {
                "type": "text",
                "text": (
                    "LEFT GROUP PHOTOS: "
                    + str(left_numbers)
                    + "\nRIGHT GROUP PHOTOS: "
                    + str(right_numbers)
                ),
            },
        ]

        for label, numbers in (
            ("LEFT", left_numbers),
            ("RIGHT", right_numbers),
        ):
            content.append({
                "type": "text",
                "text": (
                    label
                    + " GROUP REPRESENTATIVE PHOTOS"
                ),
            })

            for number in numbers[:4]:
                content.append({
                    "type": "text",
                    "text": (
                        label
                        + " PHOTO "
                        + str(number)
                    ),
                })

                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": image_to_data_url(
                            image_paths[number - 1]
                        ),
                        "detail": "high",
                    },
                })

        try:
            response = vision_chat_completion(
                content,
                temperature=0,
                response_format={
                    "type": "json_object",
                },
            )

            result = parse_json_response(
                response
            )

            merges = result.get(
                "merges",
                [],
            )

            should_merge = False
            confidence = 0.0

            if isinstance(merges, list):
                for entry in merges:
                    if not isinstance(
                        entry,
                        dict,
                    ):
                        continue

                    try:
                        left_group = int(
                            entry.get(
                                "left_group"
                            )
                        )
                        right_group = int(
                            entry.get(
                                "right_group"
                            )
                        )
                        confidence = float(
                            entry.get(
                                "confidence",
                                0.0,
                            )
                            or 0.0
                        )
                    except (
                        TypeError,
                        ValueError,
                    ):
                        continue

                    if (
                        left_group == 1
                        and right_group == 2
                        and bool(
                            entry.get(
                                "merge",
                                False,
                            )
                        )
                        and confidence >= 0.97
                    ):
                        should_merge = True
                        break

            if should_merge:
                merged_group = dict(left)

                merged_group[
                    "photo_numbers"
                ] = list(
                    dict.fromkeys(
                        left_numbers
                        + right_numbers
                    )
                )

                merged_group[
                    "reason"
                ] = (
                    "Neighboring timestamp groups "
                    "verified as the same physical garment "
                    f"({confidence:.2f})."
                )

                print(
                    "Timestamp boundary verified same garment: "
                    + str(left_numbers)
                    + " + "
                    + str(right_numbers)
                    + f" ({confidence:.2f})"
                )

                merged.append(
                    merged_group
                )

                index += 2
                continue

        except Exception as error:
            print(
                "Timestamp boundary verification failed: "
                + str(error)
            )

        merged.append(
            left
        )

        index += 1

    return merged


# ============================================================
# STAGE 3
# ATTACH DETAIL / TAG PHOTOS
# ============================================================

def validate_groups(groups, total_photos):
    """
    Final safety validation.

    Every original photo must appear exactly once UNLESS it was explicitly
    removed as a true duplicate front/back photo.
    """
    if not isinstance(groups, list):
        raise ValueError("Grouping result is not a list.")

    try:
        total = int(total_photos)
    except (TypeError, ValueError):
        raise ValueError("Invalid total photo count.")

    expected = set(range(1, total + 1))

    deleted_duplicates = set()
    for group in groups:
        if not isinstance(group, dict):
            continue
        for raw in group.get("deleted_duplicate_photos", []):
            try:
                number = int(raw)
            except (TypeError, ValueError):
                continue
            if number in expected:
                deleted_duplicates.add(number)

    expected -= deleted_duplicates
    seen = set()

    for group in groups:
        if not isinstance(group, dict):
            raise ValueError("Invalid group object.")

        numbers = group.get("photo_numbers", [])
        if not isinstance(numbers, list):
            raise ValueError("Group photo_numbers must be a list.")

        for raw_number in numbers:
            try:
                number = int(raw_number)
            except (TypeError, ValueError):
                raise ValueError("Invalid photo number in grouping result.")

            if number not in set(range(1, total + 1)):
                raise ValueError(
                    f"Invalid photo number {number}; expected 1-{total}."
                )

            if number in seen:
                raise ValueError(
                    f"Photo {number} appears in more than one final group."
                )

            if number in deleted_duplicates:
                raise ValueError(
                    f"Photo {number} was marked deleted but still appears "
                    "in a final group."
                )

            seen.add(number)

    missing = expected - seen
    if missing:
        raise ValueError(
            f"Grouping lost photos: {sorted(missing)}"
        )

    return True



def final_normalize(groups, all_photo_numbers):
    """
    Deterministically normalize final groups.
    """
    normalized = []
    used = set()
    deleted_duplicates = set()
    for group in groups or []:
        for raw in group.get("deleted_duplicate_photos", []):
            try:
                deleted_duplicates.add(int(raw))
            except (TypeError, ValueError):
                pass

    if isinstance(all_photo_numbers, int):
        valid_photo_numbers = set(
            range(1, all_photo_numbers + 1)
        )
    else:
        try:
            valid_photo_numbers = {
                int(value)
                for value in all_photo_numbers
            }
        except (TypeError, ValueError):
            valid_photo_numbers = set()

    for group in groups or []:
        if not isinstance(group, dict):
            continue

        numbers = []
        for raw in group.get("photo_numbers", []):
            try:
                number = int(raw)
            except (TypeError, ValueError):
                continue

            if number in valid_photo_numbers and number not in used:
                numbers.append(number)
                used.add(number)

        if numbers:
            clean = dict(group)
            clean["photo_numbers"] = numbers
            normalized.append(clean)

    for raw_number in sorted(valid_photo_numbers):
        try:
            number = int(raw_number)
        except (TypeError, ValueError):
            continue

        if number not in used and number not in deleted_duplicates:
            normalized.append({
                "photo_numbers": [number],
                "reason": "Unassigned photo kept separate for safety."
            })

    for item_number, group in enumerate(normalized, start=1):
        group["item_number"] = item_number

    return normalized


def attach_detail_photos(image_paths, garment_groups, detail_numbers):
    """Attach details/tags to existing garments without creating new items."""
    if not detail_numbers or not garment_groups:
        return garment_groups

    prompt = """
Assign each DETAIL/TAG photo to exactly one EXISTING clothing item.
Do NOT create, split, merge, or delete items.

Use exact visual evidence first:
- exact color
- exact pattern/graphic
- garment construction
- pockets/buttons/zippers
- brand/logo
- size/tag information

A clearly different garment color is strong evidence that the photo does NOT
belong to that candidate. Never use photo number or item number as evidence.

Return ONLY JSON:
{"assignments":[{"detail_photo":12,"item_number":3}]}
"""

    def request_for(photo_numbers, reps_per_item=3):
        content=[{"type":"text","text":prompt}]
        for item, group in enumerate(garment_groups,1):
            reps=list(dict.fromkeys(group.get("photo_numbers",[])))[:reps_per_item]
            content.append({"type":"text","text":f"CANDIDATE ITEM {item}"})
            for n in reps:
                path=image_paths[n-1]
                content.append({"type":"text","text":f"ITEM PHOTO {n}: {path.name}"})
                content.append({"type":"image_url","image_url":{
                    "url":image_to_data_url(path),"detail":"low"
                }})
        for n in photo_numbers:
            path=image_paths[n-1]
            content.append({"type":"text","text":f"DETAIL/TAG PHOTO {n}: {path.name}"})
            content.append({"type":"image_url","image_url":{
                "url":image_to_data_url(path),"detail":"high"
            }})
        response=vision_chat_completion(
            content,
            temperature=0,
            response_format={"type":"json_object"},
        )
        result=parse_json_response(response)
        return result.get("assignments",[])

    print(f"Stage 3 detail/tag matching: {len(detail_numbers)} photos...")
    assignments=request_for(detail_numbers,3)

    valid=set(range(1,len(garment_groups)+1))
    assigned=set()

    def apply(assignments):
        for a in assignments if isinstance(assignments,list) else []:
            if not isinstance(a,dict): continue
            try:
                photo=int(a.get("detail_photo")); item=int(a.get("item_number"))
            except (TypeError,ValueError): continue
            if photo in detail_numbers and item in valid and photo not in assigned:
                garment_groups[item-1]["photo_numbers"].append(photo)
                assigned.add(photo)

    apply(assignments)

    remaining=[n for n in detail_numbers if n not in assigned]
    if remaining:
        print(f"Stage 3 focused retry: {len(remaining)} photos...")
        apply(request_for(remaining,4))

    remaining=[n for n in detail_numbers if n not in assigned]
    if remaining:
        print(f"Stage 3 safety attachment: {len(remaining)} unresolved photos.")
        for n in remaining:
            garment_groups[0]["photo_numbers"].append(n)

    for group in garment_groups:
        group["photo_numbers"]=list(dict.fromkeys(group["photo_numbers"]))
    return garment_groups

def order_group_photos(image_paths, garment_groups, complete_numbers):
    """
    Final organization only. NEVER changes garment membership.
    """
    if not garment_groups:
        return garment_groups

    prompt="""
Organize photos that ALREADY belong to ONE physical clothing garment.
NEVER move a photo to another garment and NEVER create a garment.

Classify every photo as exactly one:
front, back, detail, tag, duplicate_front, duplicate_back.

FRONT/BACK:
- Keep the clearest primary front and primary back.
- Delete ONLY a genuinely redundant duplicate: essentially the same
  side/view with no meaningful new information.
- KEEP different angles, left/right details, different stains, pockets,
  sleeves/legs, construction closeups, and other useful views.

DETAIL = garment closeup, damage/stain, fabric, pocket, stitching, etc.
TAG = brand/size/care/material label. TAG ALWAYS COMES LAST.

Return ONLY JSON:
{"photos":[
 {"photo_number":1,"role":"front"},
 {"photo_number":2,"role":"back"},
 {"photo_number":3,"role":"detail"},
 {"photo_number":4,"role":"tag"},
 {"photo_number":5,"role":"duplicate_front"}
]}
"""
    output=[]
    for idx,group in enumerate(garment_groups,1):
        nums=list(dict.fromkeys(group.get("photo_numbers",[])))
        if len(nums)<=1:
            clean=dict(group)
            clean["photo_numbers"]=nums
            clean["deleted_duplicate_photos"]=[]
            output.append(clean)
            continue

        content=[{"type":"text","text":prompt}]
        for n in nums:
            path=image_paths[n-1]
            content.append({"type":"text","text":f"PHOTO {n}: {path.name}"})
            content.append({"type":"image_url","image_url":{
                "url":image_to_data_url(path),"detail":"high"
            }})

        print(f"Ordering item {idx}: {len(nums)} photos...")
        result=parse_json_response(vision_chat_completion(
            content,
            temperature=0,
            response_format={"type":"json_object"},
        ))
        entries=result.get("photos",[])
        roles={}

        for entry in entries if isinstance(entries,list) else []:
            if not isinstance(entry,dict): continue
            try: n=int(entry.get("photo_number"))
            except (TypeError,ValueError): continue
            role=str(entry.get("role","")).lower().strip()
            if n in nums and role in {
                "front","back","detail","tag","duplicate_front","duplicate_back"
            }:
                roles[n]=role

        for n in nums:
            roles.setdefault(n,"detail")

        fronts=[n for n in nums if roles[n]=="front"]
        backs=[n for n in nums if roles[n]=="back"]

        front_keep=fronts[:1]
        back_keep=backs[:1]

        deleted={
            n for n in nums
            if roles[n] in {"duplicate_front","duplicate_back"}
        }

        extras=set(fronts[1:]+backs[1:])

        details=[
            n for n in nums
            if n not in deleted and (roles[n]=="detail" or n in extras)
        ]
        tags=[
            n for n in nums
            if n not in deleted and roles[n]=="tag"
        ]

        ordered=front_keep+back_keep+details+tags

        clean=dict(group)
        clean["photo_numbers"]=ordered
        clean["deleted_duplicate_photos"]=sorted(deleted)
        clean["photo_roles"]={
            str(n):(
                "front" if n in front_keep else
                "back" if n in back_keep else
                "tag" if n in tags else
                "detail"
            )
            for n in ordered
        }
        output.append(clean)

    return output



def reconcile_orphan_photos(
    image_paths,
    garment_groups,
    complete_numbers,
    detail_numbers,
):
    """
    Timestamp-first safety pass. Only removes duplicate membership while
    preserving the established timestamp groups.
    """

    seen = set()
    cleaned = []

    for group in garment_groups or []:
        if not isinstance(group, dict):
            continue

        numbers = []

        for raw in group.get(
            "photo_numbers",
            [],
        ):
            try:
                number = int(raw)
            except (
                TypeError,
                ValueError,
            ):
                continue

            if number in seen:
                continue

            numbers.append(number)
            seen.add(number)

        if numbers:
            clean = dict(group)
            clean["photo_numbers"] = numbers
            cleaned.append(clean)

    return cleaned


# ============================================================
# MAIN PUBLIC FUNCTION
# ============================================================

def group_photos_with_ai(
    image_paths
):
    """
    Production grouping pipeline.

    Stage 1: classify complete vs detail
    Stage 2: group ONLY complete garments
    Stage 3: attach detail/tag photos
    Stage 4: deterministic validation

    PERFORMANCE FIX: removed the hardcoded time.sleep(5)/time.sleep(10)
    "cooldowns" between stages. These were stacked on top of the
    per-request throttle and the 429 backoff, effectively double- and
    triple-throttling the same limit. The 429/model-fallback backoff in
    vision_chat_completion is the correct place for that reaction now.
    """

    image_paths = [
        Path(path)
        for path in image_paths
    ]

    timestamp_records = sort_paths_by_capture_time(
        image_paths
    )

    image_paths = [
        record["path"]
        for record in timestamp_records
    ]

    timestamped_count = sum(
        record["capture_time"] is not None
        for record in timestamp_records
    )

    print(
        "Camera timestamp order: "
        + str(timestamped_count)
        + "/"
        + str(len(image_paths))
        + " photos timestamped."
    )

    if not image_paths:

        raise ValueError(
            "No photos were provided."
        )


    total_photos = len(
        image_paths
    )


    print(
        "\n===================================="
    )

    print(
        "DEPOP AI PHOTO GROUPER"
    )

    print(
        "===================================="
    )

    print(
        "\nFound "
        + str(total_photos)
        + " photos."
    )


    # ========================================================
    # STAGE 1
    # ========================================================

    print(
        "\nStage 1/3: "
        "Classifying photos..."
    )


    classification = classify_photos(
        image_paths
    )


    complete_numbers = classification[
        "complete"
    ]

    detail_numbers = classification[
        "detail"
    ]


    print(
        "Complete garment photos: "
        + str(complete_numbers)
    )

    print(
        "Detail/tag photos: "
        + str(detail_numbers)
    )


    if not complete_numbers:

        groups = []

        for number in range(
            1,
            total_photos + 1
        ):

            groups.append(
                {
                    "photo_numbers": [number],
                    "reason": (
                        "No complete garment photo was "
                        "confidently identified for this image."
                    ),
                }
            )


        groups = final_normalize(
            groups,
            total_photos
        )


        validate_groups(
            groups,
            total_photos
        )


        return {
            "groups": groups
        }


    # ========================================================
    # STAGE 2
    # ========================================================

    print(
        "\nStage 2/3: "
        "Grouping complete garments..."
    )


    garment_groups = group_complete_garments(
        image_paths,
        complete_numbers
    )


    print(
        "Complete-garment grouping created "
        + str(len(garment_groups))
        + " groups."
    )

    if len(garment_groups) > 1:
        print("\nStage 2 timestamp-boundary verification...")
        garment_groups = reconcile_timestamp_boundaries(
            image_paths,
            garment_groups
        )


    # ========================================================
    # STAGE 3
    # ========================================================

    if detail_numbers:

        print(
            "\nStage 3/3: "
            "Attaching tag/detail photos..."
        )


        garment_groups = attach_detail_photos(
            image_paths,
            garment_groups,
            detail_numbers
        )

    else:

        print(
            "\nStage 3/3: "
            "No detail photos to attach."
        )


    # ========================================================
    # FINAL OWNERSHIP RECONCILIATION
    # ========================================================

    print("\nFinal ownership reconciliation...")
    garment_groups = reconcile_orphan_photos(
        image_paths,
        garment_groups,
        complete_numbers,
        detail_numbers
    )

    # ========================================================
    # FINAL PHOTO ORDERING
    # ========================================================

    print("\nFinal photo ordering: front -> back -> details -> tag...")
    garment_groups = order_group_photos(
        image_paths,
        garment_groups,
        complete_numbers
    )

    # ========================================================
    # FINAL NORMALIZATION
    # ========================================================

    final_groups = final_normalize(
        garment_groups,
        total_photos
    )


    validate_groups(
        final_groups,
        total_photos
    )


    # ========================================================
    # OUTPUT
    # ========================================================

    print(
        "\n===================================="
    )

    print(
        "FINAL GROUPING"
    )

    print(
        "===================================="
    )


    print(
        "\nCreated "
        + str(len(final_groups))
        + " groups.\n"
    )


    for group in final_groups:

        print(
            "ITEM "
            + str(
                group["item_number"]
            )
        )

        print(
            "Photos: "
            + str(
                group["photo_numbers"]
            )
        )

        print(
            "Reason: "
            + group["reason"]
        )

        print()


    print(
        "Grouping validation: PASSED"
    )


    return {
        "groups": final_groups
    }


# ============================================================
# TERMINAL TEST
# ============================================================

def print_groups(
    image_paths,
    grouping_result
):

    groups = grouping_result.get(
        "groups",
        []
    )


    print(
        "\nCreated "
        + str(len(groups))
        + " groups.\n"
    )


    for group in groups:

        print(
            "ITEM "
            + str(
                group["item_number"]
            )
        )


        print(
            "Reason: "
            + group.get(
                "reason",
                ""
            )
        )


        for photo_number in group[
            "photo_numbers"
        ]:

            index = (
                photo_number - 1
            )


            if (
                0 <= index
                < len(image_paths)
            ):

                print(
                    "  PHOTO "
                    + str(photo_number)
                    + ": "
                    + image_paths[index].name
                )


        print()


if __name__ == "__main__":

    photos = get_image_files(
        "uploads"
    )


    if not photos:

        print(
            "No photos found in uploads."
        )

    else:

        try:

            result = group_photos_with_ai(
                photos
            )


            print_groups(
                photos,
                result
            )


        except Exception as error:

            print(
                "\nGROUPING ERROR:"
            )

            print(
                str(error)
            )
