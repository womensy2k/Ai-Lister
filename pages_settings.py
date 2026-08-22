"""
Settings — account info, listing-preference defaults (applied by the
existing Create Listing flow, see app.py), and account actions. Reuses
the existing Supabase auth session (auth.py, supabase_client.py) for
password changes and sign-out — no separate auth system.

No "Appearance" section: the app doesn't currently have a theme-
switching system (just a fixed light theme in .streamlit/config.toml),
so there's nothing real to expose here rather than inventing one.
"""

import streamlit as st

from ai_listing import SHOPIFY_CATEGORY_PATHS
from auth import logout
from supabase_client import get_supabase_client, get_supabase_admin_client
from app_data import get_listing_preferences, save_listing_preferences
from push_notifications import create_pairing_token, list_subscriptions, delete_subscription

# GitHub Pages origin the installed home-screen app is served from —
# NOT this Streamlit app's own domain. iOS only allows the push
# permission prompt / service-worker registration to happen from the
# same origin+scope as the installed PWA's manifest, which lives there
# (see docs/index.html and docs/notifications.html). Update this if the
# Pages site is ever moved to a custom domain.
_NOTIFICATIONS_SETUP_URL = "https://womensy2k.github.io/Ai-Lister/notifications.html"


def _render_account_section(user_id, user_email):
    st.markdown("#### Account")
    st.text_input("Email", value=user_email, disabled=True)

    with st.expander("Change password"):
        with st.form("settings_change_password"):
            new_password = st.text_input("New Password", type="password", help="At least 8 characters.")
            confirm_password = st.text_input("Confirm New Password", type="password")
            submitted = st.form_submit_button("Update Password", type="primary")

        if submitted:
            if len(new_password) < 8:
                st.error("Password must be at least 8 characters.")
            elif new_password != confirm_password:
                st.error("Passwords don't match.")
            else:
                try:
                    get_supabase_client().auth.update_user({"password": new_password})
                    st.success("Password updated.")
                except Exception as error:
                    st.error(f"Couldn't update your password: {error}")


def _render_listing_preferences_section(user_id):
    st.markdown("#### Listing Preferences")
    st.caption("Defaults applied automatically when the AI leaves a field unclear — never overrides a confident AI read.")

    prefs = get_listing_preferences(user_id)

    with st.form("settings_listing_prefs"):
        condition_options = ["", "New with tags", "Excellent", "Good", "Fair"]
        existing_condition = prefs.get("default_condition") or ""
        condition = st.selectbox(
            "Default Condition", condition_options,
            index=condition_options.index(existing_condition) if existing_condition in condition_options else 0,
        )
        category_options = [""] + SHOPIFY_CATEGORY_PATHS
        existing_category = prefs.get("default_category") or ""
        category = st.selectbox(
            "Default Category", category_options,
            index=category_options.index(existing_category) if existing_category in category_options else 0,
        )
        hashtags_text = st.text_input(
            "Default Hashtags (comma-separated)",
            value=", ".join(prefs.get("default_hashtags") or []),
        )
        saved = st.form_submit_button("Save Preferences", type="primary")

    if saved:
        hashtags = [tag.strip() for tag in hashtags_text.split(",") if tag.strip()]
        if save_listing_preferences(user_id, condition, category, "", hashtags):
            st.success("Saved.")
        else:
            st.error("Couldn't save preferences — try again.")


def _render_notifications_section(user_id):
    st.markdown("#### Notifications")
    st.caption(
        "Get a push notification on your phone when a batch finishes generating "
        "or a Shopify publish completes — only works from the version of this app "
        "added to your iPhone's Home Screen (iOS 16.4+), not a regular Safari tab."
    )

    subscriptions = list_subscriptions(user_id)
    if subscriptions:
        st.success(f"Enabled on {len(subscriptions)} device(s).")
        for sub in subscriptions:
            label = sub.get("user_agent") or "Unknown device"
            if len(label) > 60:
                label = label[:57] + "..."
            cols = st.columns([4, 1])
            with cols[0]:
                st.caption(f"📱 {label} — since {str(sub.get('created_at', ''))[:10]}")
            with cols[1]:
                if st.button("Remove", key=f"push_remove_{sub['id']}", width="stretch"):
                    delete_subscription(user_id, sub["id"])
                    st.rerun()
    else:
        st.caption("Not enabled on any device yet.")

    if st.button("🔔 Enable Notifications on This Device", key="push_enable_btn"):
        token = create_pairing_token(user_id)
        if token:
            setup_link = f"{_NOTIFICATIONS_SETUP_URL}?token={token}"
            st.session_state["push_setup_link"] = setup_link
        else:
            st.error("Couldn't start setup — try again.")

    setup_link = st.session_state.get("push_setup_link")
    if setup_link:
        st.markdown(
            f'<a href="{setup_link}" target="_blank" rel="noopener" '
            f'style="display:inline-block; margin-top:8px; padding:10px 18px; '
            f'border-radius:999px; background:linear-gradient(135deg,#F02AA0,#FF6EC7); '
            f'color:#fff; font-weight:800; font-size:13px; text-decoration:none;">'
            f'Tap here to finish enabling &rarr;</a>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Open this on the phone you added Y2K Lister to — for the notification "
            "permission prompt to work, it has to be tapped from inside that installed "
            "app, not from a link shared elsewhere. This link expires in 10 minutes."
        )


def _render_account_actions_section(user_id):
    st.markdown("#### Account Actions")

    if st.button("Sign Out", key="settings_sign_out", width="stretch"):
        logout()

    with st.expander("Delete account"):
        st.warning(
            "This permanently deletes your account and every listing, photo, template, "
            "and history entry associated with it. This cannot be undone."
        )
        confirm_text = st.text_input("Type DELETE to confirm", key="settings_delete_confirm_text")
        if st.button("Permanently Delete My Account", key="settings_delete_account", type="primary"):
            if confirm_text.strip() != "DELETE":
                st.error("Type DELETE (all caps) to confirm.")
            else:
                try:
                    get_supabase_admin_client().auth.admin.delete_user(user_id)
                    st.success("Account deleted.")
                    logout()
                except Exception as error:
                    st.error(f"Couldn't delete your account: {error}")


def render_settings(user_id):
    st.markdown(
        """
        <div class="brand-header">
            <div class="brand-header-shimmer"></div>
            <div class="brand-header-greet">Settings</div>
            <div class="brand-header-title">Account &amp; preferences &#10023;</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    user_email = st.session_state.get("auth_user", {}).get("email", "")

    _render_account_section(user_id, user_email)
    st.divider()
    _render_listing_preferences_section(user_id)
    st.divider()
    _render_notifications_section(user_id)
    st.divider()
    _render_account_actions_section(user_id)
