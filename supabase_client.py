"""
Supabase client access — deliberately NOT a shared global/cache_resource
singleton for the per-user client.

Streamlit Cloud runs one Python process serving every concurrently
connected browser session (st.session_state is what's actually
per-session, not the module namespace or st.cache_resource). The
supabase-py client's auth session (who it's currently authenticated as)
is mutable internal state — if two different users' sessions shared one
cached client object, one user's request could run with another user's
auth token attached. So the per-user client lives in st.session_state,
constructed once per browser session; client construction itself is
cheap (no network call), so there's no performance reason to share it
the way ai_listing.py's _get_openai() shares the OpenAI client.

The service-role client is different: it carries no per-user session at
all (it always acts as the service role), so it's safe to lazy-cache as
a plain module-level singleton, same pattern as _get_openai().
"""

import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

_supabase_module = None
_admin_client = None


def _get_supabase_module():
    global _supabase_module
    if _supabase_module is None:
        import supabase as supabase_module
        _supabase_module = supabase_module
    return _supabase_module


def get_supabase_client():
    """The per-browser-session client. Auth state (get_supabase_client().auth.
    sign_in_with_password(...) / set_session(...)) is attached to THIS
    instance and must never be shared across sessions."""
    if "_supabase_client" not in st.session_state:
        supabase_module = _get_supabase_module()

        url = os.getenv("SUPABASE_URL")
        anon_key = os.getenv("SUPABASE_ANON_KEY")
        if not url or not anon_key:
            raise ValueError(
                "SUPABASE_URL / SUPABASE_ANON_KEY were not found. "
                "Make sure your .env file (or Streamlit Cloud secrets) "
                "contains them."
            )

        st.session_state["_supabase_client"] = supabase_module.create_client(
            url, anon_key
        )

    return st.session_state["_supabase_client"]


def get_supabase_admin_client():
    """Service-role client — bypasses Row Level Security entirely.

    Only for genuinely administrative, server-side-only operations (the
    one-time vintage_corrections.json import is the only caller today).
    NEVER use this for anything that reads/writes on behalf of a
    specific end user during normal app use — that must go through
    get_supabase_client() with that user's own session attached, so RLS
    actually applies. Safe to cache as a singleton since it carries no
    per-user auth state at all.
    """
    global _admin_client

    if _admin_client is None:
        supabase_module = _get_supabase_module()

        url = os.getenv("SUPABASE_URL")
        service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        if not url or not service_role_key:
            raise ValueError(
                "SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY were not found. "
                "This key is only needed for admin/import scripts — "
                "regular app usage doesn't require it."
            )

        _admin_client = supabase_module.create_client(url, service_role_key)

    return _admin_client
