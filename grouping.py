import base64
import json
import io
from pathlib import Path

from dotenv import load_dotenv

from ai_listing import vision_chat_completion


load_dotenv()


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
# PHOTO ORDERING
# Used by app.py's "Generate All Listings" step to order each already-
# grouped item's photos front -> back -> details -> tag. Never changes
# garment membership — grouping itself is done manually in the app via
# the numbered upload slots.
# ============================================================

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
