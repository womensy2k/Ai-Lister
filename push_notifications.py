"""
Web Push notifications — sent when a user's own listing generation
finishes or a Shopify publish result comes back (success or failure),
so they can background the installed home-screen app and still find
out what happened.

This app has no separate background worker or job queue. "Send a push
when X happens" means "send synchronously right after X happens,
inside the same request that just did X" — see the send_push() calls
in app.py (generation) and qa_review.py (Shopify publish).

Getting a subscription into push_subscriptions in the first place does
NOT happen here. It goes through docs/notifications.html — a
DIFFERENT origin (GitHub Pages, not this app) that iOS requires for
the actual permission prompt/service-worker registration, since that's
where the installed PWA's manifest lives — via the
consume_push_pairing_token() RPC added in
supabase/migrations/0004_push_notifications.sql. create_pairing_token()
below is this app's half of that handshake: mint the one-time token
the Settings page hands the user a link to.
"""

import json
import os
import secrets
from datetime import datetime, timedelta, timezone

from pywebpush import webpush, WebPushException

from supabase_client import get_supabase_client


def create_pairing_token(user_id, ttl_seconds=600):
    """Mint a short-lived, single-use token proving which account a new
    push subscription belongs to. Returns the token string, or None on
    failure. RLS-scoped insert — only ever mints a token for the
    caller's own logged-in user_id."""
    token = secrets.token_urlsafe(32)
    expires_at = (
        datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)
    ).isoformat()

    try:
        get_supabase_client().table("push_pairing_tokens").insert({
            "token": token,
            "user_id": user_id,
            "expires_at": expires_at,
        }).execute()
        return token
    except Exception as error:
        print(f"create_pairing_token failed: {error}")
        return None


def list_subscriptions(user_id):
    """Devices the user currently has notifications enabled on, for the
    Settings page's "enabled on N devices" display + per-device remove."""
    try:
        response = (
            get_supabase_client()
            .table("push_subscriptions")
            .select("id, user_agent, created_at")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .execute()
        )
        return response.data or []
    except Exception:
        return []


def delete_subscription(user_id, subscription_id):
    try:
        get_supabase_client().table("push_subscriptions").delete().eq(
            "id", subscription_id
        ).eq("user_id", user_id).execute()
        return True
    except Exception:
        return False


def send_push(user_id, title, body, url=None):
    """Best-effort — never raises, never blocks the caller's own
    generation/publish flow on a notification failure. Sends to every
    device the user has enabled notifications on; drops any
    subscription the push service itself reports as gone (404/410 —
    the OS or browser cleared it on its end)."""
    private_key = os.getenv("VAPID_PRIVATE_KEY")
    contact_email = os.getenv("VAPID_CONTACT_EMAIL")
    if not private_key or not contact_email:
        return

    try:
        response = (
            get_supabase_client()
            .table("push_subscriptions")
            .select("id, endpoint, p256dh_key, auth_key")
            .eq("user_id", user_id)
            .execute()
        )
        subscriptions = response.data or []
    except Exception as error:
        print(f"send_push: couldn't load subscriptions: {error}")
        return

    if not subscriptions:
        return

    payload = json.dumps({"title": title, "body": body, "url": url or "/"})

    for sub in subscriptions:
        try:
            webpush(
                subscription_info={
                    "endpoint": sub["endpoint"],
                    "keys": {
                        "p256dh": sub["p256dh_key"],
                        "auth": sub["auth_key"],
                    },
                },
                data=payload,
                vapid_private_key=private_key,
                vapid_claims={"sub": f"mailto:{contact_email}"},
            )
        except WebPushException as error:
            status_code = getattr(error.response, "status_code", None)
            if status_code in (404, 410):
                delete_subscription(user_id, sub["id"])
            else:
                print(f"Push send failed for subscription {sub.get('id')}: {error}")
        except Exception as error:
            print(f"Push send failed for subscription {sub.get('id')}: {error}")
