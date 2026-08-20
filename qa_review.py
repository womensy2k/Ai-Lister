import base64
import html
import io
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIError, APITimeoutError
from PIL import Image

from ai_listing import (
    analyze_item,
    regenerate_field,
    apply_vintage_flag,
    detect_photo_rotations,
    rotate_photo_in_place,
)


# ============================================================
# OPENAI SETUP
# ============================================================

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError(
        "OPENAI_API_KEY was not found."
    )

client = OpenAI(
    api_key=api_key
)

QA_MODEL = "gpt-4o-mini"

QA_MAX_RETRIES = 6
# Was 0.75s — a GLOBAL minimum gap enforced via a shared timestamp,
# which meant every QA call effectively serialized to ~1 per 0.75s
# even after the scan loop below was parallelized, silently
# defeating the whole point of running items concurrently. Generate's
# own analyze_item() has no equivalent proactive gate and relies
# purely on retry-with-backoff for actual 429s (see qa_request's
# except block below) — same approach here now that QA runs 3-wide.
QA_MIN_REQUEST_INTERVAL = 0.0


_last_qa_request = 0.0


# ============================================================
# RATE LIMIT HANDLING
# ============================================================

def _get_retry_after_seconds(error):

    message = str(error)

    patterns = [
        r"try again in\s+([0-9.]+)\s*ms",
        r"try again in\s+([0-9.]+)\s*s",
        r"try again in\s+([0-9.]+)\s*seconds?"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            message,
            re.IGNORECASE
        )

        if match:

            value = float(
                match.group(1)
            )

            if "ms" in match.group(0).lower():

                return max(
                    0.5,
                    value / 1000
                )

            return max(
                0.5,
                value
            )

    return None


def qa_request(
    messages,
    temperature=0.0
):

    global _last_qa_request

    last_error = None

    for attempt in range(
        QA_MAX_RETRIES + 1
    ):

        gap = (
            QA_MIN_REQUEST_INTERVAL
            -
            (
                time.monotonic()
                -
                _last_qa_request
            )
        )

        if gap > 0:

            time.sleep(
                gap
            )

        try:

            response = (
                client.chat.completions.create(
                    model=QA_MODEL,
                    messages=messages,
                    temperature=temperature,
                    response_format={
                        "type": "json_object"
                    }
                )
            )

            _last_qa_request = (
                time.monotonic()
            )

            return response

        except (
            RateLimitError,
            APIError,
            APITimeoutError
        ) as error:

            last_error = error

            if attempt >= QA_MAX_RETRIES:

                raise

            retry_after = (
                _get_retry_after_seconds(
                    error
                )
            )

            if retry_after is not None:

                delay = retry_after

            else:

                delay = min(
                    30.0,
                    1.5 * (
                        2 ** attempt
                    )
                )

            time.sleep(
                delay
            )

    raise last_error


# ============================================================
# IMAGE PREPARATION
# ============================================================

def image_to_data_url(
    image_path
):

    image_path = Path(
        image_path
    )

    with Image.open(
        image_path
    ) as source:

        image = source.convert(
            "RGB"
        )

        max_dimension = 1400

        width, height = image.size

        if max(
            width,
            height
        ) > max_dimension:

            scale = (
                max_dimension
                /
                max(
                    width,
                    height
                )
            )

            image = image.resize(
                (
                    max(
                        1,
                        int(
                            width * scale
                        )
                    ),
                    max(
                        1,
                        int(
                            height * scale
                        )
                    )
                ),
                Image.Resampling.LANCZOS
            )

        output = io.BytesIO()

        image.save(
            output,
            format="JPEG",
            quality=75,
            optimize=True
        )

        encoded = base64.b64encode(
            output.getvalue()
        ).decode(
            "ascii"
        )

    return (
        "data:image/jpeg;base64,"
        + encoded
    )


# ============================================================
# JSON CLEANING
# ============================================================

def clean_json(
    text
):

    if not text:

        raise ValueError(
            "QA AI returned an empty response."
        )

    text = text.strip()

    if text.startswith(
        "```"
    ):

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

        return json.loads(
            text
        )

    except json.JSONDecodeError:

        start = text.find(
            "{"
        )

        end = text.rfind(
            "}"
        )

        if (
            start != -1
            and end != -1
        ):

            return json.loads(
                text[
                    start:end + 1
                ]
            )

        raise ValueError(
            "QA AI returned invalid JSON."
        )


# ============================================================
# QA PROMPT
# ============================================================

def build_qa_prompt(
    listing,
    photo_count
):

    title = str(
        listing.get(
            "title",
            ""
        )
    ).strip()

    brand = str(
        listing.get(
            "brand",
            ""
        )
    ).strip()

    size = str(
        listing.get(
            "size",
            ""
        )
    ).strip()

    color = str(
        listing.get(
            "color",
            ""
        )
    ).strip()

    category = str(
        listing.get(
            "category",
            ""
        )
    ).strip()

    description = str(
        listing.get(
            "description",
            ""
        )
    ).strip()

    condition = str(
        listing.get(
            "condition",
            ""
        )
    ).strip()

    structured_fields = {
        "type": listing.get("type", ""),
        "pattern": listing.get("pattern", ""),
        "clothing_features": listing.get("clothing_features", []),
        "neckline": listing.get("neckline", ""),
        "sleeve_length_type": listing.get("sleeve_length_type", ""),
        "sleeve_style": listing.get("sleeve_style", ""),
        "top_length_type": listing.get("top_length_type", ""),
        "fabric": listing.get("fabric", ""),
        "stretch_level": listing.get("stretch_level", ""),
        "top_fit": listing.get("top_fit", ""),
        "target_gender": listing.get("target_gender", ""),
        "age_group": listing.get("age_group", ""),
        "care_instructions": listing.get("care_instructions", ""),
    }

    return f"""
You are the FINAL QUALITY CONTROL AI for a women's
vintage/Y2K clothing resale inventory system.

You are checking a generated listing BEFORE it can be
sent to Shopify.

There are {photo_count} photos of this physical garment.

You MUST inspect the supplied photos.

Do NOT trust the generated listing blindly.

============================================================
GENERATED LISTING
============================================================

TITLE:
{title}

BRAND:
{brand}

SIZE:
{size}

COLOR:
{color}

CATEGORY:
{category}

STRUCTURED SHOPIFY ATTRIBUTES:
{json.dumps(structured_fields, indent=2)}

These attributes must agree with the photos. If an attribute is
wrong, incomplete, or unsupported by the photos, flag it as a
QA issue.

CONDITION:
{condition}

DESCRIPTION:
{description}

============================================================
YOUR JOB
============================================================

Double-check the listing against the actual photos.

The most important fields are:

1. BRAND
2. COLOR
3. GARMENT TYPE
4. SIZE
5. CONDITION
6. STYLE / AESTHETIC
7. TITLE COMPLETENESS
8. TITLE ACCURACY

============================================================
BRAND CHECK
============================================================

Determine the brand from visible tags, labels, logos, or
other strong evidence.

If the generated brand is wrong, this is a RED issue.

If the brand cannot be confidently determined, do not invent one.

============================================================
SIZE CHECK
============================================================

Look specifically for visible size tags, and zoom your attention into
that exact region of the photo — the size is usually small printed or
sewn text, easy to misread at a glance.

Read the tag LITERALLY, character for character. Report the size
exactly as printed, not a "roughly equivalent" size from a different
sizing system.

Do NOT convert or translate the size to a different format:

- If the tag says "10 REG", the observed size is "10 REG" — NOT
  "Medium", NOT "M", NOT any letter-size equivalent, even if 10 is
  commonly close to a Medium in some size charts. A numeric size and a
  letter size are only a match if the tag itself shows both (e.g. a
  tag printed as "M / 10").
- If the tag says "L/6", that is NOT "L/G" — read the actual
  characters, don't assume a common variant.
- If the tag says a plain number ("10", "27", "32"), do not guess
  whether that means junior sizing, waist inches, or numeric misses
  sizing — report the number as printed.

Compare the actual, literally-read tag text against the generated
size:

- If they match exactly (or the generated size is the same size
  written in an equally literal way), PASS.
- If the tag is legible and clearly reads something DIFFERENT from
  the generated size, RED.
- If the tag is illegible, blurry, or not visible in any photo, do
  NOT invent a size and do NOT fail the check just because you're
  unsure — say so in "reason" and only flag it if the generated
  listing claims a specific size with no visible evidence at all.

Never fail a size check by comparing the generated size against a
size YOU inferred by association or estimation rather than something
actually printed on the tag.

============================================================
COLOR CHECK
============================================================

Determine the primary visible color.

The title must include the primary useful color when it is
important for identifying the garment.

If generated color is wrong:

RED.

============================================================
GARMENT TYPE
============================================================

Make sure the garment type is accurate.

Examples:

tank ≠ cami
tank ≠ tee
shirt ≠ jacket
jacket ≠ cardigan
skirt ≠ dress
dress ≠ top

If the generated type is wrong:

RED.

============================================================
CONDITION CHECK
============================================================

Inspect every photo for actual visible flaws: stains, holes, tears,
pilling, fading, discoloration, stretching, loose stitching, damaged
graphics, missing buttons, broken zippers.

Compare what you actually see against the generated CONDITION text.

If the generated condition claims "Excellent" / no flaws, but a flaw
is clearly visible in the photos:

RED.

If the generated condition describes a specific flaw (a stain, a
hole, etc.) that is NOT actually visible anywhere in the photos:

RED.

Do not invent a flaw that isn't visible just to be cautious, and do
not clear a listing that describes flaws you can't actually see
evidence for either way — only flag a clear, visible mismatch.

============================================================
STYLE / AESTHETIC
============================================================

Determine which aesthetics are genuinely supported.

Possible aesthetics include:

Y2K
Vintage
Fairycore
Coquette

Other useful aesthetics may be used when appropriate.

IMPORTANT:

Do NOT automatically label everything Y2K.

Do NOT automatically label everything Vintage.

Do NOT automatically label everything Fairycore.

Do NOT automatically label everything Coquette.

Only use an aesthetic when the garment visually supports it.

If an aesthetic is clearly supported and relevant, it should
appear in the title.

Aesthetic/era word rules — these are absolute, apply them even if
more aesthetics are technically supported by the photos:

- Never require both "Vintage" and "Y2K" in the title at the same
  time — they describe conflicting eras. If both are arguably
  supported, only "Vintage" should be required/expected.
- Never require both "Fairycore" and "Coquette" in the title at the
  same time — only require whichever one fits better.
- Never require more than 2 aesthetic/era words in the title total,
  even when 3 or more are arguably supported by the photos.

Do NOT flag a title as incomplete just because it left out a
second/third aesthetic word, or the "other half" of a conflicting
pair — that omission is correct, not a QA issue.

============================================================
TITLE REQUIREMENTS
============================================================

The title should contain the important Shopify sorting
attributes.

When supported, verify that the title contains:

- brand
- primary color
- garment type
- Y2K
- Vintage
- Fairycore
- Coquette
- important garment details

Do not require an aesthetic that is not supported.

Example of a strong title:

Vintage American Eagle Brown Sherpa Jacket

Example of another strong title:

Y2K Coquette Ambiance Pink Lace Tank

Example:

Vintage Hollister Black Ribbed Button-Up

Note the first example does NOT say "Vintage/Y2K" — even when both
eras feel arguably supported, only one belongs in the title.

============================================================
TITLE ACCURACY
============================================================

The title must not claim something the photos contradict.

Examples of BAD titles:

Charlotte Russe when the tag says Ambiance

Cami when the garment is clearly a tank

Blue when the garment is clearly pink

Y2K Fairycore when there is no reasonable support

Vintage when there is no evidence supporting the claim

============================================================
FINAL DECISION
============================================================

PASS ONLY if:

- brand is correct or appropriately unknown
- size is correct or appropriately unknown
- color is correct
- garment type is correct
- condition matches what's actually visible in the photos
- title accurately represents the garment
- required supported aesthetics are represented
- important Shopify sorting information is present
- there are no significant contradictions

FAIL if ANY important field needs correction.

Every check's "status" MUST directly agree with its own "reason" text
— they are graded together, so a contradiction between them (e.g.
status "fail" with a reason describing a match, or status "pass" with
a reason describing a mismatch) is treated as an error. Before writing
each check, decide the reason first, then set status to match what
the reason actually says.

For checks that have both "generated" and "observed" fields (brand,
size, color, garment_type, condition): if "generated" and "observed"
describe the same thing, status MUST be "pass" — never write the same
value into both fields and then mark it "fail". Only use "fail" when
"observed" genuinely describes something different from "generated".

============================================================
OUTPUT
============================================================

Return ONLY valid JSON.

Use exactly:

{{
    "status": "pass",
    "score": 98,
    "issues": [],
    "checks": {{
        "brand": {{
            "status": "pass",
            "generated": "American Eagle",
            "observed": "American Eagle",
            "reason": "Brand tag matches."
        }},
        "size": {{
            "status": "pass",
            "generated": "M",
            "observed": "M",
            "reason": "Visible size tag matches."
        }},
        "color": {{
            "status": "pass",
            "generated": "Brown",
            "observed": "Brown",
            "reason": "Primary garment color matches."
        }},
        "garment_type": {{
            "status": "pass",
            "generated": "Jacket",
            "observed": "Jacket",
            "reason": "Garment construction matches."
        }},
        "condition": {{
            "status": "pass",
            "generated": "Excellent",
            "observed": "Excellent",
            "reason": "No visible stains, holes, or damage in the photos."
        }},
        "shopify_attributes": {{
            "status": "pass",
            "issues": [],
            "reason": "Shopify attributes agree with the photos."
        }},
        "style": {{
            "status": "pass",
            "required": [
                "Vintage"
            ],
            "missing": [],
            "reason": "Aesthetic is supported and present. Y2K was also arguably supported but correctly omitted since it conflicts with Vintage."
        }},
        "title": {{
            "status": "pass",
            "generated": "{title}",
            "recommended": "{title}",
            "missing_attributes": [],
            "reason": "Title contains the important supported attributes."
        }}
    }},
    "recommended_title": "{title}",
    "recommended_size": "{size}",
    "recommended_brand": "{brand}",
    "recommended_color": "{color}",
    "recommended_category": "{category}"
}}

STATUS MUST be exactly:

"pass"

or

"fail"

If anything important is wrong, use "fail".
"""


# ============================================================
# STATUS / REASON RECONCILIATION
# The QA model fills in a check's "status" and "reason" as two
# separate outputs, and they can disagree — seen live: status="fail"
# on a size check whose own reason read "Size tag is visible and
# matches the generated size." Trusting status blindly in that case
# means a genuinely correct listing gets flagged. This is a lightweight
# safety net, not a replacement for good prompting — it only overrides
# status when the reason text is unambiguous.
# ============================================================

_QA_MISMATCH_PHRASES = (
    "does not match", "doesn't match", "do not match", "don't match",
    "no match", "not a match", "mismatch", "incorrect", "is wrong",
    "different from", "differs from", "not the same",
    "does not correspond", "doesn't correspond",
)

_QA_MATCH_PHRASES = (
    "match", "matches", "matched", "correct", "consistent",
    "same as", "agrees with", "confirmed",
)


def _reason_says_match(reason):
    lowered = reason.lower()
    if any(phrase in lowered for phrase in _QA_MISMATCH_PHRASES):
        return False
    return any(phrase in lowered for phrase in _QA_MATCH_PHRASES)


def _reason_says_mismatch(reason):
    lowered = reason.lower()
    return any(phrase in lowered for phrase in _QA_MISMATCH_PHRASES)


def _normalize_qa_value(value):
    """Loose-equality text normalization for comparing a check's own
    "generated" vs "observed" fields — collapses whitespace/case/the
    spacing around slashes and dashes ("5 / 6" vs "5/6") without
    trying to be a general fuzzy matcher."""
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*([/\-])\s*", r"\1", text)
    return text


def _values_equivalent(generated, observed):
    if not generated or not observed:
        return False
    return _normalize_qa_value(generated) == _normalize_qa_value(observed)


# ============================================================
# RUN QA ON ONE LISTING
# ============================================================

def run_listing_qa(
    listing,
    photo_paths
):

    photo_paths = list(
        photo_paths or []
    )

    content = [
        {
            "type": "text",
            "text": build_qa_prompt(
                listing,
                len(photo_paths)
            )
        }
    ]

    for index, photo_path in enumerate(
        photo_paths,
        start=1
    ):

        content.append(
            {
                "type": "text",
                "text":
                    f"PHOTO {index}"
            }
        )

        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url":
                        image_to_data_url(
                            photo_path
                        ),
                    # "low" detail downsamples to a small fixed
                    # resolution — nowhere near enough to reliably read
                    # small printed tag text (e.g. misreading "10 REG"
                    # as "Medium"). Size/brand/condition checks need to
                    # actually read fine print, not guess from a blurry
                    # thumbnail, so QA uses the same "high" detail the
                    # main generation pipeline already uses for tags.
                    "detail":
                        "high"
                }
            }
        )

    response = qa_request(
        messages=[
            {
                "role": "system",
                "content":
                    (
                        "You are a strict clothing resale "
                        "quality-control system. "
                        "Photos are the source of truth. "
                        "Return only valid JSON."
                    )
            },
            {
                "role": "user",
                "content": content
            }
        ],
        temperature=0.0
    )

    result = clean_json(
        response
        .choices[0]
        .message
        .content
    )

    # --------------------------------------------------------
    # HARD NORMALIZATION
    # --------------------------------------------------------

    checks = result.get("checks", {})
    if not isinstance(checks, dict):
        checks = {}

    normalized_checks = {}
    for name, check in checks.items():
        if not isinstance(check, dict):
            continue

        check_status = str(
            check.get("status", "fail") or "fail"
        ).lower().strip()
        if check_status not in {"pass", "fail"}:
            check_status = "fail"

        reason_text = str(check.get("reason", "") or "").strip()

        # Most objective signal first: brand/size/color/garment_type/
        # condition checks all carry their own "generated" and
        # "observed" values. If those two are literally the same
        # thing, it cannot be a genuine mismatch no matter what
        # "status" says — this is what directly fixes the reported
        # case (generated="5/6", observed="5/6", status="fail").
        # Only falls through to the fuzzier reason-text heuristic
        # below for checks without both fields (style, title,
        # shopify_attributes) or where they're worded differently.
        if _values_equivalent(
            check.get("generated"), check.get("observed")
        ):
            check_status = "pass"
        elif check_status == "fail" and _reason_says_match(reason_text):
            check_status = "pass"
        elif check_status == "pass" and _reason_says_mismatch(reason_text):
            check_status = "fail"

        check = dict(check)
        check["status"] = check_status
        normalized_checks[name] = check

    result["checks"] = normalized_checks

    # "issues" is DERIVED from "checks" rather than trusted from the
    # model's own separate "issues" array — those two could disagree
    # (seen live: an "issues" entry claiming a field failed while its
    # own "reason" text said it actually matched, because the model
    # was inconsistent between the two arrays it was asked to fill in
    # independently). Deriving one from the other guarantees they can
    # never contradict each other, and also fixes structured
    # {field, status, reason, ...} objects rendering as a raw Python
    # dict repr instead of a readable sentence.
    derived_issues = []

    for name, check in normalized_checks.items():
        if check["status"] != "fail":
            continue

        label = name.replace("_", " ").title()
        nested_issues = check.get("issues")

        if isinstance(nested_issues, list) and nested_issues:
            for nested in nested_issues:
                nested_text = str(nested).strip()
                if nested_text:
                    derived_issues.append(f"{label}: {nested_text}")
            continue

        reason = str(check.get("reason", "") or "").strip()

        if reason:
            derived_issues.append(f"{label}: {reason}")
        else:
            derived_issues.append(f"{label} needs review.")

    result["issues"] = derived_issues

    # Status follows directly from whether any individual check
    # failed — guarantees the pass/fail banner can never disagree with
    # the checks grid shown right below it.
    result["status"] = "fail" if derived_issues else "pass"

    try:
        result["score"] = int(result.get("score", 0) or 0)
    except (TypeError, ValueError):
        # A non-numeric score from the model must not fail the whole
        # scan — the derived status/checks above are already solid,
        # a garbled score field alone shouldn't discard them.
        result["score"] = 0

    result["score"] = max(
        0,
        min(
            100,
            result["score"]
        )
    )

    return result


# ============================================================
# APPROVAL STATE
# ============================================================

def initialize_qa_state(
    listings
):

    if "qa_results" not in st.session_state:

        st.session_state[
            "qa_results"
        ] = {}

    if "qa_approved" not in st.session_state:

        st.session_state[
            "qa_approved"
        ] = set()

    if "qa_red_selected" not in st.session_state:

        st.session_state[
            "qa_red_selected"
        ] = set()

    if "qa_green_selected" not in st.session_state:

        st.session_state[
            "qa_green_selected"
        ] = set()

    for index in range(
        len(listings)
    ):

        if index not in st.session_state[
            "qa_results"
        ]:

            st.session_state[
                "qa_results"
            ][index] = None


# ============================================================
# QA STATUS HELPERS
# ============================================================

def listing_is_good(
    index
):

    result = st.session_state[
        "qa_results"
    ].get(
        index
    )

    if not result:
        return False

    return (
        result.get(
            "status"
        ) == "pass"
    )


def listing_needs_review(
    index
):

    result = st.session_state[
        "qa_results"
    ].get(
        index
    )

    if not result:
        return False

    return (
        result.get(
            "status"
        ) == "fail"
    )


# ============================================================
# AI CORRECTION
# Applies a fix for exactly the fields the QA pass flagged, using the
# cheap single-field regenerate_field() (ai_listing.py) when it
# supports that field, and falling back to one fresh full analyze_item()
# pass — merging in only the still-failing fields — for anything
# regenerate_field doesn't handle (brand, color, garment_type, ...).
# Never touches a field QA didn't flag, so it can't undo a manual edit
# to something that was already fine.
# ============================================================

_QA_REGENERATE_FIELDS = {"title", "description", "hashtags", "price", "size"}

# QA "checks" names that aren't a single correctable listing field —
# nothing to directly regenerate for these.
_QA_NON_FIELD_CHECKS = {"shopify_attributes"}


def _qa_failing_fields(result):
    checks = (result or {}).get("checks", {}) or {}
    failing = set()

    for name, check in checks.items():
        if (
            isinstance(check, dict)
            and str(check.get("status", "")).lower() == "fail"
            and name not in _QA_NON_FIELD_CHECKS
        ):
            failing.add(name)

    return failing


def ai_correct_listing(listing, photo_paths, result):
    """Correct exactly the fields QA flagged as wrong.

    Returns (updated_listing, fixed_fields, failed_fields). Never
    raises — a correction failure for one field must not lose fixes
    already made to others in the same pass.
    """
    failing = _qa_failing_fields(result)

    if not failing:
        return listing, [], []

    fixed = []
    failed = []

    regenerate_targets = failing & _QA_REGENERATE_FIELDS
    fallback_targets = failing - _QA_REGENERATE_FIELDS

    # "description" is now assembled FROM condition/size/hashtags (see
    # build_depop_description), so if it's being fixed alongside any
    # of those in the same pass, it must regenerate LAST — otherwise
    # it could embed stale values that get corrected only after it
    # already ran. A plain set has no defined order, so this sorts
    # description to the end explicitly rather than relying on luck.
    regenerate_targets = sorted(
        regenerate_targets, key=lambda field: field == "description"
    )

    for field in regenerate_targets:
        try:
            value = regenerate_field(field, listing, photo_paths)

            if field == "hashtags":
                value = [
                    tag if str(tag).startswith("#") else "#" + str(tag).strip()
                    for tag in (value or [])
                    if str(tag).strip()
                ]
            elif field == "size" and isinstance(value, dict):
                value = value.get("value", "Unknown")
            elif field == "title":
                # Preserve the manual Vintage checkbox state — a QA
                # title repair must not silently undo it.
                value = apply_vintage_flag(
                    {"title": value},
                    listing.get("is_vintage", False),
                )["title"]

            listing[field] = value
            fixed.append(field)
        except Exception as error:
            failed.append(f"{field} ({error})")

    if fallback_targets:
        try:
            fresh = analyze_item(photo_paths)
        except Exception as error:
            fresh = None
            failed.extend(f"{field} ({error})" for field in fallback_targets)

        if fresh:
            for field in fallback_targets:
                if field == "garment_type":
                    listing["garment_type"] = fresh.get(
                        "garment_type", listing.get("garment_type")
                    )
                    listing["type"] = listing["garment_type"]
                else:
                    listing[field] = fresh.get(field, listing.get(field))
                fixed.append(field)
        elif fresh is not None:
            failed.extend(fallback_targets)

    return listing, fixed, failed


def _bulk_ai_correct(indexes, listings):
    """Run ai_correct_listing() + an immediate re-scan for every listing
    index in indexes. Used by both "AI Correct All Flagged Items" and
    "AI Correct Selected". Returns (now_passing_count, still_failing_count).
    """
    indexes = list(indexes)
    now_passing = 0
    still_failing = 0

    if not indexes:
        return now_passing, still_failing

    progress = st.progress(0)

    for position, index in enumerate(indexes, start=1):
        listing_data = listings[index]
        listing = listing_data.get("listing", {}) or {}
        photos = listing_data.get("photos", [])
        result = st.session_state["qa_results"].get(index)

        if result:
            updated_listing, _fixed, _failed = ai_correct_listing(
                listing, photos, result
            )
            listing_data["listing"] = updated_listing

            try:
                fresh_result = run_listing_qa(updated_listing, photos)
                st.session_state["qa_results"][index] = fresh_result

                if fresh_result.get("status") == "pass":
                    st.session_state["qa_approved"].add(index)
                    now_passing += 1
                else:
                    st.session_state["qa_approved"].discard(index)
                    still_failing += 1
            except Exception:
                still_failing += 1

        progress.progress(position / len(indexes))

    return now_passing, still_failing


def _run_full_qa_scan(listings):
    """The actual QA pass: rotation safety net, then one run_listing_qa()
    call per item, run CONCURRENTLY across items (mirrors Generate's own
    ThreadPoolExecutor pattern) instead of one at a time — sequential
    scanning was the dominant cost for a 10-50 item batch. Runs
    automatically the first time the QA screen is shown (right after
    approval) and again on demand via the "Run AI Final QA" button for
    a manual re-scan later.
    """
    # ----------------------------------------------------
    # ROTATION SAFETY NET
    # Now trusts rotation_checked_paths (the set "Generate All
    # Listings" already maintains) instead of independently
    # re-verifying every photo — orientation checking is a quick skim
    # for photos that haven't been looked at yet, not a full second
    # opinion on ones already checked. This was the single biggest
    # lever for "double-checking orientation is way too slow": it
    # was paying the full (expensive, 4-image-per-photo) rotation
    # check TWICE per batch — once at Generate time, once here.
    # ----------------------------------------------------
    rotation_checked_paths = st.session_state.setdefault(
        "rotation_checked_paths", set()
    )

    all_qa_photos = []
    for listing_data in listings:
        all_qa_photos.extend(listing_data.get("photos", []) or [])

    photos_to_check = [
        photo_path for photo_path in all_qa_photos
        if str(Path(photo_path).resolve()) not in rotation_checked_paths
    ]

    if photos_to_check:
        rotation_status = st.empty()

        rotation_status.write(
            f"Quick orientation skim on {len(photos_to_check)} "
            "new photo(s)..."
        )

        try:
            rotations = detect_photo_rotations(photos_to_check)

            for photo_index, degrees in rotations.items():
                if 0 <= photo_index < len(photos_to_check):
                    try:
                        rotate_photo_in_place(
                            photos_to_check[photo_index],
                            degrees,
                        )
                    except Exception as rotate_error:
                        print(
                            "Final QA auto-straighten skipped a "
                            f"photo: {rotate_error}"
                        )

            for photo_path in photos_to_check:
                rotation_checked_paths.add(str(Path(photo_path).resolve()))

            if rotations:
                rotation_status.success(
                    f"Straightened {len(rotations)} photo(s) that "
                    "were still sideways."
                )
            else:
                rotation_status.empty()

        except Exception as rotation_error:
            print(
                "Final QA rotation safety net skipped: "
                f"{rotation_error}"
            )

    progress = st.progress(0)
    status_text = st.empty()
    total = len(listings)
    completed = 0

    # Same worker count Generate already uses for the same kind of
    # per-item vision-call work — proven not to trip the org's TPM
    # ceiling at that concurrency.
    worker_count = min(3, max(1, total))

    def scan_one(index, listing_data):
        listing = listing_data.get("listing", {})
        photo_paths = listing_data.get("photos", [])
        try:
            return index, run_listing_qa(listing, photo_paths), None
        except Exception as error:
            return index, None, error

    status_text.write(f"Scanning {total} listing(s) in batches of {worker_count}...")

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(scan_one, index, listing_data): index
            for index, listing_data in enumerate(listings)
        }

        for future in as_completed(futures):
            index, result, error = future.result()

            if error is not None:
                st.session_state["qa_results"][index] = {
                    "status": "fail",
                    "score": 0,
                    "issues": ["QA failed to run: " + str(error)],
                    "checks": {},
                }
                st.session_state["qa_approved"].discard(index)
            else:
                st.session_state["qa_results"][index] = result
                if result.get("status") == "pass":
                    st.session_state["qa_approved"].add(index)
                else:
                    st.session_state["qa_approved"].discard(index)

            completed += 1
            status_text.write(f"Scanned {completed} of {total} listing(s)...")
            progress.progress(completed / total)

    status_text.success("AI final QA complete.")


# ============================================================
# FIELD-LEVEL GREEN/YELLOW/RED DISPLAY
# Mirrors the "Generated Listings Review" screen's own field layout
# (Title, Brand+Category, Garment Type+Price, Size+Color+Pattern,
# Condition, Description, Hashtags) instead of a separate score +
# issues list + a generic checks grid — the point is to make a glance
# at this card feel like the same listing you already reviewed, just
# with color telling you exactly which field to double check.
# ============================================================

def _combine_checks(*checks):
    """Combine multiple QA checks into one effective status+reason —
    used for Title, which should reflect BOTH the dedicated "title"
    check and the "style" check (an aesthetic-tag mismatch belongs to
    the title, since that's where aesthetic words actually show up)."""
    valid = [c for c in checks if isinstance(c, dict)]

    if not valid:
        return None

    failing = [c for c in valid if c.get("status") == "fail"]

    if failing:
        reason = " ".join(
            str(c.get("reason", "") or "").strip()
            for c in failing
            if c.get("reason")
        )
        return {"status": "fail", "reason": reason}

    return {"status": "pass", "reason": ""}


def _qa_field_chip_html(label, value, check=None, full_width=False):
    if isinstance(check, dict):
        status = str(check.get("status", "") or "").lower()
        if status not in {"pass", "fail"}:
            status = "unchecked"
        reason = str(check.get("reason", "") or "").strip()
    else:
        status = "unchecked"
        reason = ""

    value_text = str(value or "").strip() or "—"
    if len(value_text) > 220:
        value_text = value_text[:220].rstrip() + "…"

    classes = "qa-field-chip qa-field-full" if full_width else "qa-field-chip"

    reason_html = ""
    if status == "fail" and reason:
        reason_html = (
            f'<div class="qa-field-reason">⚠️ {html.escape(reason)}</div>'
        )

    return (
        f'<div class="{classes}" data-status="{status}">'
        f'<div class="qa-field-label">{html.escape(label)}</div>'
        f'<div class="qa-field-value">{html.escape(value_text)}</div>'
        f"{reason_html}"
        "</div>"
    )


def _qa_field_grid_html(listing, checks):
    garment_type = str(
        listing.get("garment_type", listing.get("type", "")) or ""
    )
    category = str(
        listing.get("category_path", listing.get("category", "")) or ""
    )

    price = listing.get("suggested_price")
    price_text = f"${price}" if price not in (None, "", "-") else ""

    hashtags = listing.get("hashtags") or []
    if isinstance(hashtags, list):
        hashtags_text = " ".join(str(tag) for tag in hashtags)
    else:
        hashtags_text = str(hashtags)

    title_check = _combine_checks(
        checks.get("title"), checks.get("style")
    )

    rows = [
        _qa_field_chip_html(
            "Title", listing.get("title", ""), title_check, full_width=True
        ),
        (
            '<div class="qa-field-row">'
            + _qa_field_chip_html(
                "Brand", listing.get("brand", ""), checks.get("brand")
            )
            + _qa_field_chip_html("Category", category)
            + "</div>"
        ),
        (
            '<div class="qa-field-row">'
            + _qa_field_chip_html(
                "Garment Type", garment_type, checks.get("garment_type")
            )
            + _qa_field_chip_html("Price", price_text)
            + "</div>"
        ),
        (
            '<div class="qa-field-row">'
            + _qa_field_chip_html(
                "Size", listing.get("size", ""), checks.get("size")
            )
            + _qa_field_chip_html(
                "Color", listing.get("color", ""), checks.get("color")
            )
            + _qa_field_chip_html("Pattern", listing.get("pattern", ""))
            + "</div>"
        ),
        _qa_field_chip_html(
            "Condition",
            listing.get("condition", ""),
            checks.get("condition"),
            full_width=True,
        ),
        _qa_field_chip_html(
            "Description",
            listing.get("description", ""),
            full_width=True,
        ),
        _qa_field_chip_html(
            "Hashtags", hashtags_text, full_width=True
        ),
    ]

    return '<div class="qa-field-grid">' + "".join(rows) + "</div>"


# ============================================================
# RENDER QA REVIEW
# ============================================================

def render_qa_review(
    listings
):

    initialize_qa_state(
        listings
    )

    # Streamlit's own native "view fullscreen" hover control on
    # st.image is a separate mechanism from anything this app builds —
    # clicking a QA photo preview to look closer opened Streamlit's own
    # fullscreen overlay unexpectedly (no custom zoom control exists
    # here to explain it), which is exactly what "an image randomly
    # pops up, and I have to X it out" describes. These previews are
    # just a visual reference for the QA scan, not meant to be a zoom
    # tool, so disable it outright rather than leave a surprise trap.
    st.markdown(
        """
        <style>
        div[class*="st-key-qa_photo_"] div[data-testid="stElementToolbar"],
        div[class*="st-key-qa_photo_"] [data-testid="StyledFullScreenButton"],
        div[class*="st-key-qa_photo_"] button[title*="fullscreen" i],
        div[class*="st-key-qa_photo_"] button[aria-label*="fullscreen" i] {
            display: none !important;
            visibility: hidden !important;
            pointer-events: none !important;
        }

        .qa-field-grid {
            display: flex;
            flex-direction: column;
            gap: 8px;
            margin: 10px 0 14px 0;
        }
        .qa-field-row {
            display: flex;
            gap: 8px;
        }
        .qa-field-row .qa-field-chip {
            flex: 1 1 0;
            min-width: 0;
        }
        .qa-field-chip {
            border-radius: 10px;
            padding: 8px 11px;
            border: 1px solid;
        }
        .qa-field-chip[data-status="pass"] {
            background: rgba(57, 217, 138, .10);
            border-color: rgba(57, 217, 138, .45);
        }
        .qa-field-chip[data-status="fail"] {
            background: rgba(239, 68, 68, .10);
            border-color: rgba(239, 68, 68, .5);
        }
        .qa-field-chip[data-status="unchecked"] {
            background: rgba(245, 196, 81, .10);
            border-color: rgba(245, 196, 81, .45);
        }
        .qa-field-label {
            font-size: 9px;
            font-weight: 900;
            letter-spacing: .06em;
            text-transform: uppercase;
            opacity: .6;
            margin-bottom: 3px;
        }
        .qa-field-value {
            font-size: 13px;
            font-weight: 600;
            word-break: break-word;
            white-space: pre-wrap;
        }
        .qa-field-reason {
            font-size: 11px;
            opacity: .8;
            margin-top: 4px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="brand-step">
            <span class="brand-step-num">5</span>
            <span class="brand-step-text">AI Final Quality Check</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ========================================================
    # RUN QA — auto-runs once right after approval. "Approve Listings
    # & Run Final AI QA" (on the Review screen) is the ONLY button
    # needed to get a scan — no second click required here. A manual
    # re-scan trigger still exists for the rare case something needs
    # re-checking outside the AI-correct flows (which already
    # re-scan automatically on their own), but it's tucked into a
    # small expander rather than sitting on screen as a second
    # "required-looking" primary button.
    # ========================================================

    never_scanned = all(
        result is None
        for result in st.session_state["qa_results"].values()
    )

    if never_scanned:
        _run_full_qa_scan(listings)
        st.rerun()

    with st.expander("⚙️ Re-scan options", expanded=False):
        if st.button("🔄 Re-scan All Listings", key="run_final_qa"):
            _run_full_qa_scan(listings)
            st.rerun()


    # ========================================================
    # COUNTS
    # ========================================================

    good_indexes = [
        index
        for index in range(
            len(listings)
        )
        if listing_is_good(index)
    ]

    red_indexes = [
        index
        for index in range(
            len(listings)
        )
        if listing_needs_review(index)
    ]

    approved_indexes = (
        st.session_state[
            "qa_approved"
        ]
    )

    unresolved_red = [
        index
        for index in red_indexes
        if index not in approved_indexes
    ]

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total",
            len(listings)
        )

    with col2:

        st.metric(
            "Good",
            len(good_indexes)
        )

    with col3:

        st.metric(
            "Needs Review",
            len(unresolved_red)
        )

    with col4:

        st.metric(
            "Approved",
            len(approved_indexes)
        )

    if unresolved_red:

        if st.button(
            f"🤖 AI Correct All Flagged Items ({len(unresolved_red)})",
            key="ai_correct_all_flagged",
            type="primary",
        ):

            with st.spinner(
                f"AI correcting {len(unresolved_red)} item(s)..."
            ):
                now_passing, still_failing = _bulk_ai_correct(
                    unresolved_red, listings
                )

            if now_passing:
                st.success(
                    f"{now_passing} item(s) now pass QA."
                )

            if still_failing:
                st.warning(
                    f"{still_failing} item(s) still need attention — "
                    "see the NEEDS REVIEW tab below."
                )

            st.rerun()


    # ========================================================
    # TABS
    # ========================================================

    good_tab, red_tab, all_tab = st.tabs(
        [
            "🟩 GOOD",
            "🟥 NEEDS REVIEW",
            "ALL"
        ]
    )


    # ========================================================
    # SELECTION HELPERS
    # ========================================================

    def select_good(
        index,
        checked
    ):

        # Green and red selections are mutually exclusive.
        if checked:

            st.session_state[
                "qa_red_selected"
            ].clear()

            st.session_state[
                "qa_green_selected"
            ].add(
                index
            )

        else:

            st.session_state[
                "qa_green_selected"
            ].discard(
                index
            )


    def select_red(
        index,
        checked
    ):

        # Red and green selections are mutually exclusive.
        if checked:

            st.session_state[
                "qa_green_selected"
            ].clear()

            st.session_state[
                "qa_red_selected"
            ].add(
                index
            )

        else:

            st.session_state[
                "qa_red_selected"
            ].discard(
                index
            )


    # ========================================================
    # RENDER ONE QA CARD
    # ========================================================

    def render_card(
        index,
        listing_data,
        color_type
    ):

        listing = listing_data.get(
            "listing",
            {}
        )

        result = st.session_state[
            "qa_results"
        ].get(
            index
        )

        title = listing.get(
            "title",
            "Untitled Item"
        )

        if result:

            is_good = (
                result.get(
                    "status"
                ) == "pass"
            )

            if is_good:

                st.success(
                    f"🟩 #{index + 1}  {title}"
                )

            else:

                st.error(
                    f"🟥 #{index + 1}  {title}"
                )

        else:

            st.warning(
                f"⚪ #{index + 1}  {title} — Not scanned"
            )

        # ----------------------------------------------------
        # PHOTO PREVIEW
        # ----------------------------------------------------

        photos = listing_data.get(
            "photos",
            []
        )

        if photos:

            photo_columns = st.columns(
                min(
                    len(photos),
                    5
                )
            )

            for photo_index, photo_path in enumerate(
                photos[:5]
            ):

                with photo_columns[
                    photo_index
                    %
                    len(photo_columns)
                ]:

                    # Own keyed container so the CSS below (disabling
                    # Streamlit's native fullscreen-expand control) is
                    # scoped to just these QA preview thumbnails.
                    with st.container(
                        key=f"qa_photo_{color_type}_{index}_{photo_index}"
                    ):
                        st.image(
                            photo_path,
                            use_container_width=True
                        )


        # ----------------------------------------------------
        # SELECT CHECKBOX
        # ----------------------------------------------------

        if color_type == "green":

            selected = (
                index
                in
                st.session_state[
                    "qa_green_selected"
                ]
            )

            checked = st.checkbox(
                "Select",
                value=selected,
                key=f"qa_green_{index}"
            )

            select_good(
                index,
                checked
            )

        elif color_type == "red":

            selected = (
                index
                in
                st.session_state[
                    "qa_red_selected"
                ]
            )

            checked = st.checkbox(
                "Select",
                value=selected,
                key=f"qa_red_{index}"
            )

            select_red(
                index,
                checked
            )


        # ----------------------------------------------------
        # QA DETAILS — same field layout as the Generated Listings
        # Review screen, colored green/red/yellow per QA check
        # instead of a separate score/issues/checks list to read
        # through. A glance tells you exactly which field to
        # double-check; the reason for any red field is shown right
        # there instead of in a separate list further down.
        # ----------------------------------------------------

        if result:

            score = result.get("score", 0)
            st.caption(f"QA Score: {score}/100")

            checks = result.get("checks", {})
            if not isinstance(checks, dict):
                checks = {}

            st.markdown(
                _qa_field_grid_html(listing, checks),
                unsafe_allow_html=True,
            )


        # ----------------------------------------------------
        # AI CORRECTION — exactly the fields QA flagged, nothing else.
        # Deep manual editing of every field already happened on the
        # "Generated Listings Review" screen above; this step is about
        # verifying and quickly fixing what's wrong, not re-editing an
        # entire listing from scratch again.
        # ----------------------------------------------------

        failing_fields = _qa_failing_fields(result) if result else set()

        if failing_fields:

            if st.button(
                "🤖 AI Correct This Item",
                key=f"qa_ai_correct_{color_type}_{index}",
            ):

                with st.spinner("AI correcting flagged fields..."):

                    updated_listing, fixed, failed = ai_correct_listing(
                        listing,
                        photos,
                        result,
                    )

                    listing_data["listing"] = updated_listing

                    try:
                        fresh_result = run_listing_qa(
                            updated_listing,
                            photos,
                        )

                        st.session_state["qa_results"][index] = fresh_result

                        if fresh_result.get("status") == "pass":
                            st.session_state["qa_approved"].add(index)
                        else:
                            st.session_state["qa_approved"].discard(index)

                    except Exception as rescan_error:
                        failed.append(f"re-scan ({rescan_error})")

                if fixed:
                    st.success(f"Fixed: {', '.join(fixed)}")

                if failed:
                    st.error(f"Could not fix: {', '.join(failed)}")

                st.rerun()

        elif result:

            st.caption("✓ Nothing flagged — looks good.")

        st.divider()


    # ========================================================
    # GOOD TAB
    # ========================================================

    with good_tab:

        st.subheader(
            f"🟩 Good — {len(good_indexes)} listings"
        )

        if not good_indexes:

            st.info(
                "No listings have passed QA yet."
            )

        else:

            for index in good_indexes:

                render_card(
                    index,
                    listings[index],
                    "green"
                )

        selected_green = (
            st.session_state[
                "qa_green_selected"
            ]
        )

        if selected_green:

            st.success(
                f"{len(selected_green)} green "
                "listing(s) selected."
            )

            # Nothing to AI-correct here — these already passed QA, so
            # there's no flagged field to fix. Just approve.
            if st.button(
                "Approve Selected",
                key="approve_green"
            ):

                st.session_state[
                    "qa_approved"
                ].update(
                    selected_green
                )

                st.session_state[
                    "qa_green_selected"
                ].clear()

                st.rerun()


    # ========================================================
    # RED TAB
    # ========================================================

    with red_tab:

        st.subheader(
            f"🟥 Needs Review — {len(unresolved_red)} listings"
        )

        if not unresolved_red:

            st.success(
                "No unresolved QA issues."
            )

        else:

            for index in unresolved_red:

                render_card(
                    index,
                    listings[index],
                    "red"
                )

        selected_red = (
            st.session_state[
                "qa_red_selected"
            ]
        )

        if selected_red:

            st.error(
                f"{len(selected_red)} red "
                "listing(s) selected."
            )

            col_a, col_b = st.columns(2)

            with col_a:

                if st.button(
                    "🤖 AI Correct Selected",
                    key="ai_edit_red"
                ):

                    with st.spinner(
                        f"AI correcting {len(selected_red)} item(s)..."
                    ):
                        now_passing, still_failing = _bulk_ai_correct(
                            selected_red, listings
                        )

                    if now_passing:
                        st.success(
                            f"{now_passing} item(s) now pass QA."
                        )

                    if still_failing:
                        st.warning(
                            f"{still_failing} item(s) still need attention."
                        )

                    st.rerun()

            with col_b:

                if st.button(
                    "Approve Anyway",
                    key="approve_red"
                ):

                    st.session_state[
                        "qa_approved"
                    ].update(
                        selected_red
                    )

                    st.session_state[
                        "qa_red_selected"
                    ].clear()

                    st.rerun()


    # ========================================================
    # ALL TAB
    # ========================================================

    with all_tab:

        st.subheader(
            f"All — {len(listings)} listings"
        )

        for index, listing_data in enumerate(
            listings
        ):

            render_card(
                index,
                listing_data,
                "none"
            )


    # ========================================================
    # ========================================================
    # FINAL SHOPIFY GATE
    # ========================================================
    #
    # A listing is allowed through when it has either:
    #   - passed Final AI QA and is approved, OR
    #   - been manually approved with "Approve Anyway".
    #
    # This means a red QA result does NOT permanently block Shopify
    # once the user explicitly approves that listing.
    # ========================================================

    st.divider()

    total_count = len(listings)

    scanned_count = sum(
        1
        for index in range(total_count)
        if st.session_state["qa_results"].get(index) is not None
    )

    approved_indexes = set(
        st.session_state["qa_approved"]
    )

    all_scanned = (
        total_count > 0
        and scanned_count == total_count
    )

    all_approved = (
        total_count > 0
        and set(range(total_count)).issubset(
            approved_indexes
        )
    )

    if all_scanned and all_approved:

        st.session_state["qa_complete"] = True
        st.session_state["final_qa_passed"] = True

        st.success(
            "🟩 Final AI QA complete — "
            f"all {total_count} listings are approved. "
            "Shopify is unlocked."
        )

        # ========================================================
        # SHOPIFY UPLOAD
        # ========================================================
        # Keep this action here so the user can immediately send the
        # approved listings to Shopify after the final QA gate passes.
        if st.button(
            "🛍️ Create Shopify Drafts",
            type="primary",
            key="create_shopify_drafts_final_qa",
        ):
            try:
                from shopify_integration import create_shopify_draft, _token

                upload_total = len(approved_indexes)
                upload_progress = st.progress(0)
                upload_status = st.empty()

                # Fetched ONCE for the whole batch instead of letting
                # create_shopify_draft() re-authenticate (a fresh OAuth
                # POST to Shopify) on every single item — also means a
                # missing/bad credential fails fast with one clear
                # error instead of failing the same way N times in the
                # loop below.
                shopify_token = _token()

                shopify_results = []
                shopify_errors = []

                for upload_number, index in enumerate(
                    sorted(approved_indexes),
                    start=1,
                ):
                    listing_data = listings[index] or {}
                    listing = listing_data.get("listing") or {}
                    photos = listing_data.get("photos") or []

                    upload_status.write(
                        f"Creating Shopify draft "
                        f"{upload_number} of {upload_total}..."
                    )

                    try:
                        result = create_shopify_draft(
                            listing,
                            photos,
                            token=shopify_token,
                        )

                        shopify_results.append({
                            "index": index,
                            "result": result,
                        })

                    except Exception as error:
                        shopify_errors.append({
                            "index": index,
                            "title": listing.get(
                                "title",
                                f"Listing #{index + 1}",
                            ),
                            "error": str(error),
                        })

                    upload_progress.progress(
                        upload_number / upload_total
                    )

                st.session_state[
                    "shopify_upload_results"
                ] = shopify_results

                st.session_state[
                    "shopify_upload_errors"
                ] = shopify_errors

                if shopify_errors:
                    upload_status.warning(
                        f"Shopify upload finished with "
                        f"{len(shopify_results)} successful and "
                        f"{len(shopify_errors)} failed."
                    )
                else:
                    upload_status.success(
                        f"🛍️ Successfully created "
                        f"{len(shopify_results)} Shopify draft(s)."
                    )

                if shopify_results:
                    st.markdown("### Shopify Drafts Created")

                    for item in shopify_results:
                        result = item["result"]

                        product_id = result.get(
                            "product_id",
                            "Unknown",
                        )

                        st.success(
                            f"✓ {result.get('title', 'Untitled')} "
                            f"— {product_id}"
                        )

                if shopify_errors:
                    st.markdown("### Shopify Upload Errors")

                    for item in shopify_errors:
                        st.error(
                            f"#{item['index'] + 1} "
                            f"{item['title']}: "
                            f"{item['error']}"
                        )

            except Exception as error:
                st.error(
                    "Could not start the Shopify uploader: "
                    f"{error}"
                )

        return True

    st.session_state["final_qa_passed"] = False

    if not all_scanned:

        st.warning(
            "🔒 Shopify locked — Final AI QA has not "
            "finished scanning every listing."
        )

    else:

        missing = len(
            set(range(total_count))
            - approved_indexes
        )

        st.warning(
            "🔒 Shopify locked — "
            f"{missing} listing(s) still need approval."
        )

    return False

