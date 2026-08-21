"""
Shared Supabase data-access layer for everything the new Dashboard /
My Listings / Templates / History / Analytics / Favorites / Settings
pages need, plus the persistence hooks wired into the existing
generate/QA/Shopify flow (app.py, qa_review.py).

Every function takes an explicit user_id and never trusts anything
else to scope a query — but the real enforcement is Row Level Security
on every table (see supabase/migrations/). This module doesn't ADD
security, it relies on the database enforcing it, same principle the
whole auth/database foundation was built on: don't rely on the UI,
don't trust a client-side id, enforce ownership at the database layer.

Persistence failures here are logged and swallowed, never raised —
this module sits alongside the existing generation/QA/Shopify flow,
not inside its critical path, so a database hiccup must never break
listing generation itself.
"""

import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from supabase_client import get_supabase_client

PHOTO_BUCKET = "listing-photos"


# ============================================================
# LISTINGS — persistence hooks called from app.py / qa_review.py
# ============================================================

def persist_generated_listing(user_id, item_number, listing, photos, marked_vintage=False, batch_id=None):
    """Called right after a listing is generated. Creates one `listings`
    row plus one `photos` row per photo, uploading each photo to
    Supabase Storage rather than only keeping the local temp path —
    local files don't survive a Streamlit Cloud restart, so without
    this "My Listings" thumbnails would silently break after the next
    redeploy. Returns the new listing's id, or None on failure."""
    client = get_supabase_client()

    row = {
        "user_id": user_id,
        "batch_id": batch_id,
        "item_number": item_number,
        "sku": f"WY2K-{uuid.uuid4().hex[:8].upper()}",
        "title": listing.get("title", "") or "",
        "brand": listing.get("brand", "") or "",
        "brand_confidence": _safe_number(listing.get("brand_confidence")),
        "brand_evidence": listing.get("brand_evidence", "") or "",
        "garment_type": listing.get("garment_type", "") or "",
        "category": listing.get("category", "") or "",
        "category_gid": listing.get("category_gid", "") or "",
        "size": listing.get("size", "") or "",
        "size_confidence": _safe_number(listing.get("size_confidence")),
        "size_evidence": listing.get("size_evidence", "") or "",
        "color": listing.get("color", "") or "",
        "pattern": listing.get("pattern", "") or "",
        "style": listing.get("style") or [],
        "condition": listing.get("condition", "") or "",
        "condition_evidence": listing.get("condition_evidence", "") or "",
        "description": listing.get("description", "") or "",
        "description_bullets": listing.get("description_bullets") or [],
        "hashtags": listing.get("hashtags") or [],
        "suggested_price": _safe_number(listing.get("suggested_price")),
        "is_vintage": bool(listing.get("is_vintage", False)),
        "vintage_classification": listing.get("vintage_classification", "") or "",
        "vintage_evidence": listing.get("vintage_evidence"),
        "status": "draft",
        "source": "listing_generator",
    }

    try:
        result = client.table("listings").insert(row).execute()
        listing_id = result.data[0]["id"]
    except Exception as error:
        print(f"persist_generated_listing: insert failed: {error}")
        return None

    for index, path in enumerate(photos or []):
        try:
            _upload_and_link_photo(client, user_id, listing_id, path, index)
        except Exception as error:
            print(f"persist_generated_listing: photo {path} failed: {error}")

    log_activity(user_id, "listing_created", "listing", listing_id, row["title"] or f"Item {item_number}")

    return listing_id


def _safe_number(value):
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _guess_content_type(path):
    return {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".webp": "image/webp",
    }.get(Path(path).suffix.lower(), "image/jpeg")


def _upload_and_link_photo(client, user_id, listing_id, local_path, sort_order):
    local_path = Path(local_path)
    storage_path = f"{user_id}/{listing_id}/{sort_order}_{local_path.name}"

    with open(local_path, "rb") as handle:
        client.storage.from_(PHOTO_BUCKET).upload(
            path=storage_path,
            file=handle.read(),
            file_options={"content-type": _guess_content_type(local_path), "upsert": "true"},
        )

    client.table("photos").insert({
        "listing_id": listing_id,
        "user_id": user_id,
        "storage_path": storage_path,
        "sort_order": sort_order,
        "is_main": sort_order == 0,
    }).execute()


def get_listing_photo_urls(user_id, listing_id, expires_in=3600):
    """Signed URLs (the bucket is private/RLS-protected, not public —
    matches every other user-owned resource in this app), main photo
    first."""
    client = get_supabase_client()
    try:
        photos = (
            client.table("photos")
            .select("storage_path, sort_order, is_main")
            .eq("listing_id", listing_id)
            .eq("user_id", user_id)
            .order("sort_order")
            .execute()
        ).data
    except Exception as error:
        print(f"get_listing_photo_urls: query failed: {error}")
        return []

    urls = []
    for photo in photos:
        try:
            signed = client.storage.from_(PHOTO_BUCKET).create_signed_url(
                photo["storage_path"], expires_in
            )
            url = signed.get("signedURL") or signed.get("signedUrl") or signed.get("signed_url")
            if url:
                urls.append(url)
        except Exception:
            continue
    return urls


def update_listing_status(user_id, listing_id, status, **extra_fields):
    """Bump a persisted listing's lifecycle status (draft -> ready ->
    listed -> sold -> archived). Called from the QA-approval and
    Shopify-draft-success hooks — never from anywhere in the UI
    directly, so the status always reflects a real pipeline event."""
    if not listing_id:
        return False
    client = get_supabase_client()
    row = {"status": status, "updated_at": datetime.now(timezone.utc).isoformat()}
    row.update(extra_fields)
    try:
        client.table("listings").update(row).eq("id", listing_id).eq("user_id", user_id).execute()
        return True
    except Exception as error:
        print(f"update_listing_status: failed: {error}")
        return False


# ============================================================
# MY LISTINGS
# ============================================================

def list_user_listings(user_id, status=None, search=None, sort="created_desc", limit=200):
    client = get_supabase_client()
    query = client.table("listings").select("*").eq("user_id", user_id)

    if status and status != "All":
        query = query.eq("status", status)
    if search:
        like = f"%{search}%"
        query = query.or_(f"title.ilike.{like},brand.ilike.{like}")

    order_column, desc = {
        "created_desc": ("created_at", True),
        "created_asc": ("created_at", False),
        "price_desc": ("suggested_price", True),
        "price_asc": ("suggested_price", False),
        "title_asc": ("title", False),
    }.get(sort, ("created_at", True))

    query = query.order(order_column, desc=desc).limit(limit)

    try:
        return query.execute().data
    except Exception as error:
        print(f"list_user_listings: query failed: {error}")
        return []


def get_listing(user_id, listing_id):
    client = get_supabase_client()
    try:
        result = (
            client.table("listings")
            .select("*")
            .eq("id", listing_id)
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
        return result.data[0] if result.data else None
    except Exception as error:
        print(f"get_listing: query failed: {error}")
        return None


def update_listing_fields(user_id, listing_id, **fields):
    client = get_supabase_client()
    fields["updated_at"] = datetime.now(timezone.utc).isoformat()
    try:
        client.table("listings").update(fields).eq("id", listing_id).eq("user_id", user_id).execute()
        label = fields.get("title") or listing_id
        log_activity(user_id, "listing_updated", "listing", listing_id, label)
        return True
    except Exception as error:
        print(f"update_listing_fields: failed: {error}")
        return False


def delete_listing(user_id, listing_id, label=None):
    client = get_supabase_client()
    try:
        photos = (
            client.table("photos")
            .select("storage_path")
            .eq("listing_id", listing_id)
            .eq("user_id", user_id)
            .execute()
        ).data
        paths = [p["storage_path"] for p in photos]
        if paths:
            try:
                client.storage.from_(PHOTO_BUCKET).remove(paths)
            except Exception as error:
                print(f"delete_listing: photo cleanup failed (continuing): {error}")

        client.table("listings").delete().eq("id", listing_id).eq("user_id", user_id).execute()
        log_activity(user_id, "listing_deleted", "listing", listing_id, label or listing_id)
        return True
    except Exception as error:
        print(f"delete_listing: failed: {error}")
        return False


def duplicate_listing(user_id, listing_id):
    client = get_supabase_client()
    original = get_listing(user_id, listing_id)
    if not original:
        return None

    copy_fields = {
        key: value for key, value in original.items()
        if key not in ("id", "created_at", "updated_at", "shopify_product_id")
    }
    copy_fields["title"] = f"{copy_fields.get('title', '')} (Copy)".strip()
    copy_fields["status"] = "draft"
    copy_fields["sku"] = f"WY2K-{uuid.uuid4().hex[:8].upper()}"

    try:
        result = client.table("listings").insert(copy_fields).execute()
        new_id = result.data[0]["id"]
    except Exception as error:
        print(f"duplicate_listing: insert failed: {error}")
        return None

    try:
        photos = (
            client.table("photos")
            .select("storage_path, sort_order, is_main")
            .eq("listing_id", listing_id)
            .eq("user_id", user_id)
            .execute()
        ).data
        for photo in photos:
            client.table("photos").insert({
                "listing_id": new_id,
                "user_id": user_id,
                "storage_path": photo["storage_path"],
                "sort_order": photo["sort_order"],
                "is_main": photo["is_main"],
            }).execute()
    except Exception as error:
        print(f"duplicate_listing: photo copy failed (continuing): {error}")

    log_activity(user_id, "listing_created", "listing", new_id, copy_fields["title"])
    return new_id


# ============================================================
# TEMPLATES
# ============================================================

def list_templates(user_id):
    client = get_supabase_client()
    try:
        return (
            client.table("templates")
            .select("*")
            .eq("user_id", user_id)
            .order("name")
            .execute()
        ).data
    except Exception as error:
        print(f"list_templates: query failed: {error}")
        return []


def create_template(user_id, name, description="", default_condition="", default_category="",
                     default_category_gid="", default_brand="", default_hashtags=None):
    client = get_supabase_client()
    row = {
        "user_id": user_id,
        "name": name,
        "description": description,
        "default_condition": default_condition,
        "default_category": default_category,
        "default_category_gid": default_category_gid,
        "default_brand": default_brand,
        "default_hashtags": default_hashtags or [],
    }
    try:
        result = client.table("templates").insert(row).execute()
        template_id = result.data[0]["id"]
        log_activity(user_id, "template_created", "template", template_id, name)
        return template_id
    except Exception as error:
        print(f"create_template: insert failed: {error}")
        return None


def update_template(user_id, template_id, **fields):
    client = get_supabase_client()
    fields["updated_at"] = datetime.now(timezone.utc).isoformat()
    try:
        client.table("templates").update(fields).eq("id", template_id).eq("user_id", user_id).execute()
        log_activity(user_id, "template_updated", "template", template_id, fields.get("name", template_id))
        return True
    except Exception as error:
        print(f"update_template: failed: {error}")
        return False


def delete_template(user_id, template_id, label=None):
    client = get_supabase_client()
    try:
        client.table("templates").delete().eq("id", template_id).eq("user_id", user_id).execute()
        log_activity(user_id, "template_deleted", "template", template_id, label or template_id)
        return True
    except Exception as error:
        print(f"delete_template: failed: {error}")
        return False


def duplicate_template(user_id, template_id):
    client = get_supabase_client()
    try:
        original = (
            client.table("templates")
            .select("*")
            .eq("id", template_id)
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        ).data
    except Exception as error:
        print(f"duplicate_template: query failed: {error}")
        return None

    if not original:
        return None

    copy_fields = {
        key: value for key, value in original[0].items()
        if key not in ("id", "created_at", "updated_at")
    }
    copy_fields["name"] = f"{copy_fields.get('name', '')} (Copy)".strip()

    try:
        result = client.table("templates").insert(copy_fields).execute()
        new_id = result.data[0]["id"]
        log_activity(user_id, "template_created", "template", new_id, copy_fields["name"])
        return new_id
    except Exception as error:
        print(f"duplicate_template: insert failed: {error}")
        return None


# ============================================================
# ACTIVITY LOG
# ============================================================

def log_activity(user_id, action, item_type, item_id, item_label):
    client = get_supabase_client()
    try:
        client.table("activity_log").insert({
            "user_id": user_id,
            "action": action,
            "item_type": item_type,
            "item_id": item_id,
            "item_label": item_label,
        }).execute()
    except Exception as error:
        print(f"log_activity: insert failed (non-fatal): {error}")


def list_activity(user_id, limit=100):
    client = get_supabase_client()
    try:
        return (
            client.table("activity_log")
            .select("*")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        ).data
    except Exception as error:
        print(f"list_activity: query failed: {error}")
        return []


# ============================================================
# FAVORITES
# ============================================================

def is_favorited(user_id, listing_id):
    client = get_supabase_client()
    try:
        result = (
            client.table("favorites")
            .select("id")
            .eq("user_id", user_id)
            .eq("listing_id", listing_id)
            .limit(1)
            .execute()
        )
        return bool(result.data)
    except Exception:
        return False


def toggle_favorite(user_id, listing_id):
    """Returns the new state (True = now favorited)."""
    client = get_supabase_client()
    if is_favorited(user_id, listing_id):
        try:
            client.table("favorites").delete().eq("user_id", user_id).eq("listing_id", listing_id).execute()
            return False
        except Exception as error:
            print(f"toggle_favorite: delete failed: {error}")
            return True
    else:
        try:
            client.table("favorites").insert({"user_id": user_id, "listing_id": listing_id}).execute()
            return True
        except Exception as error:
            print(f"toggle_favorite: insert failed: {error}")
            return False


def list_favorite_listings(user_id):
    client = get_supabase_client()
    try:
        favorite_rows = (
            client.table("favorites")
            .select("listing_id")
            .eq("user_id", user_id)
            .execute()
        ).data
    except Exception as error:
        print(f"list_favorite_listings: query failed: {error}")
        return []

    listing_ids = [row["listing_id"] for row in favorite_rows]
    if not listing_ids:
        return []

    try:
        return (
            client.table("listings")
            .select("*")
            .in_("id", listing_ids)
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .execute()
        ).data
    except Exception as error:
        print(f"list_favorite_listings: listings query failed: {error}")
        return []


# ============================================================
# DASHBOARD / ANALYTICS
# ============================================================

def get_listing_stats(user_id):
    """Real counts only — every number here is a live query, nothing
    hardcoded or estimated."""
    client = get_supabase_client()
    try:
        rows = (
            client.table("listings")
            .select("status, suggested_price, created_at")
            .eq("user_id", user_id)
            .execute()
        ).data
    except Exception as error:
        print(f"get_listing_stats: query failed: {error}")
        rows = []

    now = datetime.now(timezone.utc)
    today = now.date()
    week_start = today - timedelta(days=today.weekday())
    month_start = today.replace(day=1)

    total = len(rows)
    by_status = {"draft": 0, "ready": 0, "listed": 0, "sold": 0, "archived": 0}
    created_today = 0
    created_this_week = 0
    created_this_month = 0
    priced = []

    for row in rows:
        status = row.get("status") or "draft"
        if status in by_status:
            by_status[status] += 1

        price = row.get("suggested_price")
        if price is not None:
            priced.append(float(price))

        created_at = row.get("created_at")
        if created_at:
            try:
                created_date = datetime.fromisoformat(created_at.replace("Z", "+00:00")).date()
                if created_date == today:
                    created_today += 1
                if created_date >= week_start:
                    created_this_week += 1
                if created_date >= month_start:
                    created_this_month += 1
            except (ValueError, TypeError):
                pass

    return {
        "total": total,
        "by_status": by_status,
        "created_today": created_today,
        "created_this_week": created_this_week,
        "created_this_month": created_this_month,
        "avg_price": (sum(priced) / len(priced)) if priced else None,
        "total_inventory_value": sum(priced) if priced else None,
        "priced_count": len(priced),
    }


def get_listings_created_per_day(user_id, days=14):
    client = get_supabase_client()
    try:
        rows = (
            client.table("listings")
            .select("created_at")
            .eq("user_id", user_id)
            .execute()
        ).data
    except Exception as error:
        print(f"get_listings_created_per_day: query failed: {error}")
        rows = []

    today = datetime.now(timezone.utc).date()
    counts = {(today - timedelta(days=offset)): 0 for offset in range(days - 1, -1, -1)}

    for row in rows:
        created_at = row.get("created_at")
        if not created_at:
            continue
        try:
            created_date = datetime.fromisoformat(created_at.replace("Z", "+00:00")).date()
        except (ValueError, TypeError):
            continue
        if created_date in counts:
            counts[created_date] += 1

    return counts


# ============================================================
# SETTINGS — listing preference defaults
# ============================================================

def get_listing_preferences(user_id):
    client = get_supabase_client()
    try:
        result = (
            client.table("profiles")
            .select("default_condition, default_category, default_category_gid, default_hashtags, plan")
            .eq("id", user_id)
            .limit(1)
            .execute()
        )
        return result.data[0] if result.data else {}
    except Exception as error:
        print(f"get_listing_preferences: query failed: {error}")
        return {}


def save_listing_preferences(user_id, default_condition, default_category, default_category_gid, default_hashtags):
    client = get_supabase_client()
    try:
        client.table("profiles").update({
            "default_condition": default_condition,
            "default_category": default_category,
            "default_category_gid": default_category_gid,
            "default_hashtags": default_hashtags or [],
        }).eq("id", user_id).execute()
        return True
    except Exception as error:
        print(f"save_listing_preferences: failed: {error}")
        return False
