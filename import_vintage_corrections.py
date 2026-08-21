"""
One-time import: seed your existing vintage_corrections.json (~15KB of
real accumulated vintage-tagging history) into the new per-user
vintage_corrections table, under your own account, so that learning
isn't lost when the app moves off the old global JSON file.

Run this ONCE, after your Supabase project exists and you've signed up
for your own account in the app:

    python import_vintage_corrections.py your@email.com

Uses the service-role key (bypasses Row Level Security) since this is
an administrative, one-time operation, not something a normal logged-in
user does through the app — that's why it's a standalone script and not
a button in the UI.
"""

import json
import sys
from pathlib import Path

from supabase_client import get_supabase_admin_client

VINTAGE_CORRECTIONS_FILE = Path(__file__).with_name("vintage_corrections.json")


def _find_user_id(client, email):
    email = email.strip().lower()
    page = 1
    while True:
        response = client.auth.admin.list_users(page=page, per_page=200)
        users = response if isinstance(response, list) else getattr(response, "users", [])
        if not users:
            return None
        for user in users:
            if str(getattr(user, "email", "")).strip().lower() == email:
                return user.id
        page += 1


def main():
    if len(sys.argv) != 2:
        print("Usage: python import_vintage_corrections.py your@email.com")
        sys.exit(1)

    email = sys.argv[1]

    if not VINTAGE_CORRECTIONS_FILE.exists():
        print(f"No {VINTAGE_CORRECTIONS_FILE.name} found — nothing to import.")
        sys.exit(0)

    entries = json.loads(VINTAGE_CORRECTIONS_FILE.read_text(encoding="utf-8"))
    if not isinstance(entries, list) or not entries:
        print(f"{VINTAGE_CORRECTIONS_FILE.name} has no entries — nothing to import.")
        sys.exit(0)

    client = get_supabase_admin_client()

    user_id = _find_user_id(client, email)
    if not user_id:
        print(
            f"No account found for {email}. Sign up in the app first, "
            "then re-run this."
        )
        sys.exit(1)

    rows = [
        {
            "user_id": user_id,
            "corrected_at": _timestamp_to_iso(entry.get("timestamp")),
            "brand": entry.get("brand", ""),
            "garment_type": entry.get("garment_type", ""),
            "bucket_keys": entry.get("bucket_keys", []),
            "ai_auto_verdict": bool(entry.get("ai_auto_verdict", False)),
            "from_state": bool(entry.get("from_state", False)),
            "to_state": bool(entry.get("to_state", False)),
        }
        for entry in entries
    ]

    # Insert in chunks — a single 3000-row insert is fine for Postgres,
    # but chunking keeps any single request small and makes a partial
    # failure easy to see and re-run instead of an all-or-nothing call.
    chunk_size = 500
    inserted = 0
    for start in range(0, len(rows), chunk_size):
        chunk = rows[start:start + chunk_size]
        client.table("vintage_corrections").insert(chunk).execute()
        inserted += len(chunk)
        print(f"Inserted {inserted} / {len(rows)}...")

    print(f"Done — imported {inserted} vintage-correction entries for {email}.")


def _timestamp_to_iso(value):
    import datetime
    try:
        return datetime.datetime.fromtimestamp(
            float(value), tz=datetime.timezone.utc
        ).isoformat()
    except (TypeError, ValueError):
        return datetime.datetime.now(datetime.timezone.utc).isoformat()


if __name__ == "__main__":
    main()
