"""
Authentication — signup, login, logout, password reset, and session
persistence across a browser refresh.

This is the auth GATE for the whole app: app.py and
description_generator.py both call require_auth() as the very first
thing they do, before rendering anything else (same "before everything
else runs" placement principle already used for app.py's page router).
If there's no valid session, require_auth() renders the login/signup
screen and st.stop()s — none of the actual app code below it executes.

Two real integration wrinkles worth knowing about up front, since both
need live verification against a real Supabase project (can't be tested
without one):

1. Streamlit has no built-in session cookie. CookieManager
   (extra_streamlit_components) proxies to the browser's real cookies
   via its own component round-trip — the FIRST script run of a session
   can render before the cookie value has actually come back from the
   browser, so an already-logged-in user can flash the login screen for
   an instant before the component's own rerun corrects it. Cosmetic,
   not a correctness bug — worth polishing once this is live.

2. Supabase's password-reset email link lands the user back on this
   app's URL carrying the recovery token either as a URL FRAGMENT
   (`#access_token=...&type=recovery`, older "implicit" flow) or as a
   `?code=...` query param (newer PKCE flow) — which one your specific
   project uses depends on its Auth settings. Fragments are never sent
   to the server at all, so Python can't see them directly; this file
   handles BOTH: a `?code=` query param via exchange_code_for_session(),
   and a small JS snippet (same st.html(unsafe_allow_javascript=True)
   technique already used elsewhere in this app for real clipboard
   copy) that rewrites a fragment into a query param Python CAN read.
   Verify which path your project actually takes once Supabase exists.
"""

import base64
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import extra_streamlit_components as stx
import streamlit as st

from supabase_client import get_supabase_client

_COOKIE_NAME = "depop_ai_session"


# ============================================================
# COOKIE MANAGER
# ============================================================
# NOT wrapped in st.cache_resource — CookieManager() itself creates a
# widget/component instance under the hood, and Streamlit explicitly
# warns/errors on calling a widget command inside a cached function
# (confirmed live: CachedWidgetWarning).
#
# Memoized in st.session_state instead — confirmed live that calling
# stx.CookieManager(key="...") a SECOND time within the same script run
# (e.g. once to read the cookie while restoring a session, again to
# write it right after a successful login) raises "multiple elements
# with the same key," since Streamlit doesn't allow two component
# instances sharing one key in the same run. Reusing one instance for
# the whole browser session avoids that regardless of how many times
# read/write happens in a single run.
def _cookie_manager():
    if "_cookie_manager_instance" not in st.session_state:
        st.session_state["_cookie_manager_instance"] = stx.CookieManager(
            key="depop_ai_cookie_manager"
        )
    return st.session_state["_cookie_manager_instance"]


def _store_session_cookie(access_token, refresh_token):
    expires_at = datetime.now(timezone.utc) + timedelta(days=30)
    _cookie_manager().set(
        _COOKIE_NAME,
        f"{access_token}::{refresh_token}",
        expires_at=expires_at,
        key="set_session_cookie",
    )


def _read_session_cookie():
    # .get_all() (not .get()), and NOT skipped just because the manager
    # instance is memoized across reruns — CookieManager.__init__ only
    # snapshots real browser cookies into self.cookies ONCE, at first
    # construction, which on a fresh page load can still be the empty
    # `default={}` handed back before the browser round-trip finishes.
    # Reusing that stale snapshot forever is what silently broke session
    # restore on refresh (confirmed live). Calling .get_all() explicitly
    # re-issues the read on every run instead of trusting a one-time
    # snapshot, so it picks up the real value once the round trip lands.
    cookies = _cookie_manager().get_all(key="read_session_cookie")
    raw = cookies.get(_COOKIE_NAME)
    if not raw or "::" not in raw:
        return None, None
    access_token, refresh_token = raw.split("::", 1)
    return access_token, refresh_token


def _clear_session_cookie():
    try:
        _cookie_manager().delete(_COOKIE_NAME, key="delete_session_cookie")
    except (KeyError, Exception):
        pass


# ============================================================
# SESSION RESTORE
# ============================================================

def _restore_session_from_cookie():
    """Try to re-establish a Supabase session from the stored cookie.
    Returns the user dict on success, None if there's no cookie or it's
    no longer valid (expired/revoked) — never raises."""
    access_token, refresh_token = _read_session_cookie()
    if not access_token or not refresh_token:
        return None

    client = get_supabase_client()
    try:
        client.auth.set_session(access_token, refresh_token)
        session = client.auth.get_session()
        user = session.user if session else None
        if user is None:
            return None

        # set_session() transparently refreshes an expired access token
        # using the refresh token — but Supabase ROTATES the refresh
        # token on every refresh, invalidating the one that was just
        # used. If the new pair isn't written back to the cookie here,
        # the next restore (a new tab, the next day) replays the
        # now-dead old refresh token and fails, forcing a real
        # re-login even though the user never logged out — this is
        # what silently broke "remember me" after about an hour.
        # Re-persisting also slides the 30-day expiry forward on every
        # active visit rather than a hard 30-day cutoff from login.
        if session.access_token != access_token or session.refresh_token != refresh_token:
            _store_session_cookie(session.access_token, session.refresh_token)
            time.sleep(0.4)

        return {"id": user.id, "email": user.email}
    except Exception:
        _clear_session_cookie()
        return None


def _set_logged_in(session, user, remember=True):
    st.session_state["auth_user"] = {"id": user.id, "email": user.email}
    if not remember:
        # Explicitly unchecked "Remember me" — session_state alone
        # keeps them logged in for this browser tab, but nothing is
        # written to a cookie, so closing the browser (or a fresh tab)
        # lands back on the login screen instead of restoring.
        return
    _store_session_cookie(session.access_token, session.refresh_token)
    # CookieManager.set() is a real Streamlit component call — it posts
    # a message to its iframe and returns immediately, it does NOT wait
    # for the browser to actually execute document.cookie = ... first.
    # Confirmed live: calling st.rerun() right after .set() (every
    # caller of _set_logged_in does this next) tore the component down
    # before that message was processed, so the cookie was never
    # actually written and login never survived a refresh. This delay
    # gives the round trip time to complete before the rerun.
    time.sleep(0.5)


def logout():
    """Call from anywhere (e.g. a sidebar 'Log Out' button) to sign out
    and force back to the login screen on the next rerun."""
    try:
        get_supabase_client().auth.sign_out()
    except Exception:
        pass
    _clear_session_cookie()
    time.sleep(0.5)  # same component round-trip reason as _set_logged_in
    st.session_state.pop("auth_user", None)
    st.session_state.pop("_supabase_client", None)
    st.rerun()


# ============================================================
# STYLING — self-contained since this renders before app.py's own
# _inject_global_brand_css() ever gets a chance to run for a logged-out
# visitor. Same palette (grep-verified against app.py), not reinvented.
# ============================================================

@st.cache_data(show_spinner=False)
def _auth_logo_data_url():
    """Base64-embeds the real brand logo (same file app.py's sidebar
    uses) so the login/signup card can show it with no separate file
    request — cached so it isn't re-read/re-encoded on every rerun.
    Kept as its own copy rather than importing app.py's version, since
    this module has to stay self-contained (it renders before app.py
    even gets a chance to run for a logged-out visitor)."""
    path = Path(__file__).with_name("logo_transparent.png")
    if not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _inject_auth_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Baloo+2:wght@700;800&display=swap');
        div[class*="st-key-auth_shell"] {
            max-width: 420px;
            margin: 48px auto 0 auto;
        }
        div[class*="st-key-auth_card"] {
            background: #FFFFFF;
            border: 1px solid rgba(240, 42, 160, .16);
            border-radius: 22px;
            padding: 34px 32px;
            box-shadow: 0 12px 32px rgba(240, 42, 160, .10);
        }
        .auth-wordmark {
            font-family: 'Baloo 2', sans-serif;
            font-weight: 800;
            font-size: 26px;
            color: #F02AA0;
            text-align: center;
            margin-bottom: 4px;
        }
        .auth-logo-wrap {
            text-align: center;
            margin-bottom: 4px;
        }
        .auth-logo-wrap img {
            width: 62%;
            max-width: 200px;
            height: auto;
        }
        .auth-subtitle {
            text-align: center;
            color: #6B5B66;
            font-size: 13.5px;
            margin-bottom: 22px;
        }
        div[data-testid="stForm"] button[kind="primaryFormSubmit"] {
            background: linear-gradient(135deg, #F02AA0, #FF6EC7) !important;
            border: none !important;
            box-shadow: 0 4px 12px rgba(240, 42, 160, .30) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# FORMS
# ============================================================

def _render_login_form():
    with st.form("auth_login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        remember = st.checkbox("Remember me", value=True, key="auth_remember_me")
        submitted = st.form_submit_button(
            "Log In", type="primary", width="stretch"
        )

    if submitted:
        if not email or not password:
            st.error("Enter both your email and password.")
            return
        try:
            client = get_supabase_client()
            response = client.auth.sign_in_with_password(
                {"email": email, "password": password}
            )
            if response.session is None:
                st.error("Login failed. Check your email and password.")
                return
            _set_logged_in(response.session, response.user, remember=remember)
            st.rerun()
        except Exception as error:
            st.error(f"Login failed: {error}")

    left, right = st.columns(2)
    with left:
        if st.button("Create an account", key="auth_goto_signup", width="stretch"):
            st.session_state["auth_mode"] = "signup"
            st.rerun()
    with right:
        if st.button("Forgot password?", key="auth_goto_forgot", width="stretch"):
            st.session_state["auth_mode"] = "forgot"
            st.rerun()


def _render_signup_form():
    with st.form("auth_signup_form"):
        email = st.text_input("Email")
        password = st.text_input(
            "Password", type="password",
            help="At least 8 characters.",
        )
        confirm = st.text_input("Confirm Password", type="password")
        submitted = st.form_submit_button(
            "Sign Up", type="primary", width="stretch"
        )

    if submitted:
        if not email or not password:
            st.error("Enter an email and password.")
            return
        if len(password) < 8:
            st.error("Password must be at least 8 characters.")
            return
        if password != confirm:
            st.error("Passwords don't match.")
            return
        try:
            client = get_supabase_client()
            response = client.auth.sign_up(
                {"email": email, "password": password}
            )
            if response.session is not None:
                # Email confirmation disabled in Supabase project
                # settings — logged in immediately.
                _set_logged_in(response.session, response.user)
                st.rerun()
            else:
                st.success(
                    "Account created — check your email to confirm it, "
                    "then log in below."
                )
                st.session_state["auth_mode"] = "login"
        except Exception as error:
            st.error(f"Sign up failed: {error}")

    if st.button("Already have an account? Log in", key="auth_goto_login", width="stretch"):
        st.session_state["auth_mode"] = "login"
        st.rerun()


def _render_forgot_password_form():
    st.caption(
        "Enter your email and we'll send you a link to reset your password."
    )
    with st.form("auth_forgot_form"):
        email = st.text_input("Email")
        submitted = st.form_submit_button(
            "Send Reset Link", type="primary", width="stretch"
        )

    if submitted:
        if not email:
            st.error("Enter your email.")
            return
        try:
            client = get_supabase_client()
            client.auth.reset_password_for_email(email)
            st.success(
                "If that email has an account, a reset link is on its way."
            )
        except Exception as error:
            st.error(f"Couldn't send the reset link: {error}")

    if st.button("Back to log in", key="auth_goto_login_2", width="stretch"):
        st.session_state["auth_mode"] = "login"
        st.rerun()


def _render_set_new_password_form():
    """Landing form after clicking the reset-password email link — the
    recovery session is already active (established by
    _try_consume_recovery_link() before this ever renders)."""
    st.caption("Choose a new password for your account.")
    with st.form("auth_set_new_password_form"):
        password = st.text_input(
            "New Password", type="password",
            help="At least 8 characters.",
        )
        confirm = st.text_input("Confirm New Password", type="password")
        submitted = st.form_submit_button(
            "Set New Password", type="primary", width="stretch"
        )

    if submitted:
        if len(password) < 8:
            st.error("Password must be at least 8 characters.")
            return
        if password != confirm:
            st.error("Passwords don't match.")
            return
        try:
            client = get_supabase_client()
            client.auth.update_user({"password": password})
            st.success("Password updated — you're logged in.")
            user_response = client.auth.get_user()
            session = client.auth.get_session()
            if user_response and user_response.user and session:
                _set_logged_in(session, user_response.user)
            st.session_state.pop("auth_recovery_pending", None)
            st.rerun()
        except Exception as error:
            st.error(f"Couldn't update your password: {error}")


# ============================================================
# PASSWORD-RESET LINK HANDLING — see module docstring point 2.
# ============================================================

def _try_consume_recovery_link():
    """Returns True if this page load is a password-reset landing
    (recovery session now active, or a fragment-rewrite JS was just
    injected and we're waiting for the resulting rerun)."""
    query_params = st.query_params

    code = query_params.get("code")
    link_type = query_params.get("type")

    if code and link_type == "recovery":
        try:
            client = get_supabase_client()
            client.auth.exchange_code_for_session({"auth_code": code})
            st.session_state["auth_recovery_pending"] = True
            st.query_params.clear()
            return True
        except Exception as error:
            st.error(f"That reset link is invalid or expired: {error}")
            return False

    access_token = query_params.get("access_token")
    refresh_token = query_params.get("refresh_token")
    if access_token and refresh_token and link_type == "recovery":
        try:
            client = get_supabase_client()
            client.auth.set_session(access_token, refresh_token)
            st.session_state["auth_recovery_pending"] = True
            st.query_params.clear()
            return True
        except Exception as error:
            st.error(f"That reset link is invalid or expired: {error}")
            return False

    if st.session_state.get("auth_recovery_pending"):
        return True

    # Neither a query-param form of the link, nor already resolved this
    # run — if the URL fragment carries the token instead (the
    # "implicit flow" shape), rewrite it into a query param so the NEXT
    # script run (Python) can actually read it. Fragments are
    # client-side only; this is the one thing JS has to do here.
    st.html(
        """
        <script>
        (function() {
            if (!window.location.hash || window.location.hash.indexOf("access_token") === -1) return;
            var params = new URLSearchParams(window.location.hash.slice(1));
            if (params.get("type") !== "recovery") return;
            var url = new URL(window.location.href);
            url.hash = "";
            url.searchParams.set("access_token", params.get("access_token") || "");
            url.searchParams.set("refresh_token", params.get("refresh_token") || "");
            url.searchParams.set("type", "recovery");
            window.location.replace(url.toString());
        })();
        </script>
        """,
        unsafe_allow_javascript=True,
    )
    return False


# ============================================================
# ENTRY POINT
# ============================================================

def _render_auth_screen():
    _inject_auth_css()

    # st.container(key=...) rather than manually opening/closing a
    # <div> across separate st.markdown() calls — Streamlit wraps each
    # st.markdown() call in its OWN container, so a hand-split div
    # doesn't actually nest the content visually (confirmed live: it
    # rendered as an empty styled box floating above the real,
    # unstyled content). st-key-<key> is the real, proven way to scope
    # CSS to a block of native Streamlit content, used everywhere else
    # in this app already.
    with st.container(key="auth_shell"):
        with st.container(key="auth_card"):
            logo_url = _auth_logo_data_url()
            if logo_url:
                st.markdown(
                    f'<div class="auth-logo-wrap"><img src="{logo_url}" alt="Womens Y2K"></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="auth-wordmark">Womens Y2K</div>',
                    unsafe_allow_html=True,
                )

            if st.session_state.get("auth_recovery_pending"):
                st.markdown(
                    '<div class="auth-subtitle">Reset your password</div>',
                    unsafe_allow_html=True,
                )
                _render_set_new_password_form()
            else:
                mode = st.session_state.setdefault("auth_mode", "login")
                subtitle = {
                    "login": "Log in to your account",
                    "signup": "Create your account",
                    "forgot": "Reset your password",
                }[mode]
                st.markdown(
                    f'<div class="auth-subtitle">{subtitle}</div>',
                    unsafe_allow_html=True,
                )

                if mode == "login":
                    _render_login_form()
                elif mode == "signup":
                    _render_signup_form()
                else:
                    _render_forgot_password_form()


def _inject_pwa_head_tags():
    """Makes THIS app's own URL directly installable as a standalone,
    chrome-less home-screen app on iOS — instead of relying on a
    separate GitHub Pages wrapper page.

    That wrapper existed because Streamlit Cloud's own page template
    writes its own generic <head> tags (manifest, apple-touch-icon) on
    initial page load, and there's no way to out-race that from normal
    Streamlit content — anything rendered via st.markdown()/st.html()
    lands in the BODY, never the actual <head>, regardless of where in
    the script it's called from. But a wrapper page turned out to trade
    one problem for another: it had its own top-level page redirect
    into this app to fix third-party-cookie login persistence, and
    iOS reveals the address bar/Safari chrome on ANY navigation that
    crosses outside the installed app's own origin — which that
    redirect always did, defeating the "no browser chrome" point of
    installing it as an app in the first place.
    Both problems go away if the icon installed points at THIS origin
    directly (no wrapper, no cross-origin redirect ever). Real
    top-level JS (via st.html's unsafe_allow_javascript, same mechanism
    already used elsewhere in this app for real clipboard access) DOES
    have full document.head access after the page has already loaded —
    it just edits the DOM directly instead of trying to win an
    initial-parse race. Removes whatever Streamlit Cloud's own template
    already put there first, so ours are the ones iOS actually reads
    when "Add to Home Screen" is tapped. Idempotent — always removes
    before adding, so re-running this every rerun never duplicates tags.
    """
    st.html(
        """
        <script>
        (function() {
            // This app's own content actually renders inside a SAME-ORIGIN
            // nested iframe (Streamlit Community Cloud's gateway/viewer
            // shell loads it at a "/~/+/..." sub-path, confirmed live via
            // window.top.location vs window.location differing) — not as
            // the true top-level page itself. Safari's Add to Home Screen
            // reads the TOP-LEVEL document's <head>, so editing plain
            // `document.head` here was silently editing the WRONG
            // document the whole time (confirmed live: a real iPhone kept
            // picking up Streamlit's own name/icon no matter what this
            // script did locally). window.top.document IS reachable
            // without a cross-origin error specifically because it's the
            // same origin (y2klister.streamlit.app either way, just a
            // different path) — same-origin policy only cares about
            // scheme+host+port, not path. Falls back to the local
            // document if window.top is ever cross-origin or unavailable
            // (e.g. running the plain open-source template locally,
            // outside Streamlit Cloud, where there's no nesting at all).
            var targetDoc = document;
            var targetWin = window;
            try {
                if (window.top && window.top.document) {
                    targetDoc = window.top.document;
                    targetWin = window.top;
                }
            } catch (error) {
                targetDoc = document;
                targetWin = window;
            }

            var STALE_SELECTORS = [
                'link[rel="manifest"]',
                'link[rel="apple-touch-icon"]',
                'link[rel="apple-touch-icon-precomposed"]',
                'meta[name="apple-mobile-web-app-capable"]',
                'meta[name="apple-mobile-web-app-title"]',
                'meta[name="apple-mobile-web-app-status-bar-style"]',
                'meta[name="theme-color"]'
            ];

            function applyTags() {
                var head = targetDoc.head;
                STALE_SELECTORS.forEach(function (selector) {
                    head.querySelectorAll(selector).forEach(function (el) { el.remove(); });
                });

                function addMeta(name, content) {
                    var el = targetDoc.createElement('meta');
                    el.setAttribute('name', name);
                    el.setAttribute('content', content);
                    head.appendChild(el);
                }
                function addLink(rel, href) {
                    var el = targetDoc.createElement('link');
                    el.setAttribute('rel', rel);
                    el.setAttribute('href', href);
                    head.appendChild(el);
                }

                addMeta('apple-mobile-web-app-capable', 'yes');
                addMeta('apple-mobile-web-app-status-bar-style', 'default');
                addMeta('apple-mobile-web-app-title', 'Y2K Lister');
                addMeta('theme-color', '#F02AA0');
                addLink('apple-touch-icon', 'https://womensy2k.github.io/Ai-Lister/apple-touch-icon.png');
                addLink('manifest', 'https://womensy2k.github.io/Ai-Lister/manifest.json');

                if (targetDoc.title !== 'Y2K Lister') {
                    targetDoc.title = 'Y2K Lister';
                }
            }

            applyTags();

            // Streamlit Cloud's own shell manages its head tags via
            // react-helmet and can re-render/re-assert them after this
            // already ran once — a one-time fix can't reliably beat a
            // moving target, so this keeps re-winning instead: any time
            // something in the target document's <head> changes,
            // immediately re-apply.
            if (!targetWin.__y2kPwaHeadObserverInstalled) {
                targetWin.__y2kPwaHeadObserverInstalled = true;
                // applyTags() itself mutates <head> (remove-then-add),
                // which would otherwise make the observer re-trigger
                // itself forever — disconnect before touching the DOM,
                // reconnect once done, so it only reacts to changes
                // from OUTSIDE this function (Streamlit Cloud's own
                // shell), never its own.
                var observer = new MutationObserver(function () {
                    observer.disconnect();
                    applyTags();
                    observer.observe(targetDoc.head, { childList: true, subtree: true });
                });
                observer.observe(targetDoc.head, { childList: true, subtree: true });
            }
        })();
        </script>
        """,
        unsafe_allow_javascript=True,
    )


def require_auth():
    """Call at the very top of every page, before rendering anything
    else. Returns {"id", "email"} for the logged-in user. Renders the
    login/signup screen and st.stop()s if there's no valid session."""
    _inject_pwa_head_tags()

    if _try_consume_recovery_link():
        _render_auth_screen()
        st.stop()

    if "auth_user" in st.session_state:
        return st.session_state["auth_user"]

    restored = _restore_session_from_cookie()
    if restored is not None:
        st.session_state["auth_user"] = restored
        return restored

    _render_auth_screen()
    st.stop()
