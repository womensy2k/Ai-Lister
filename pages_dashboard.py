"""
Dashboard — real per-user stats and quick actions. Reuses the existing
brand CSS classes (.depop-card, .depop-stat-tile, .brand-step) injected
by app.py's _inject_global_brand_css(), which already runs before this
page is ever reached — no separate styling system here.
"""

import streamlit as st

from app_data import get_listing_stats, list_activity


def _stat_tile(number, label):
    st.markdown(
        f"""
        <div class="depop-stat-tile">
            <div class="depop-stat-num">{number}</div>
            <div class="depop-stat-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dashboard(user_id):
    st.markdown(
        """
        <div class="brand-header">
            <div class="brand-header-shimmer"></div>
            <div class="brand-header-greet">Dashboard</div>
            <div class="brand-header-title">Welcome back &#10023;</div>
            <div class="brand-header-sub">Here's what's happening with your listings.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    stats = get_listing_stats(user_id)
    by_status = stats["by_status"]

    st.markdown("#### Listings")
    row1 = st.columns(4)
    with row1[0]:
        _stat_tile(stats["total"], "Total Listings")
    with row1[1]:
        _stat_tile(by_status["draft"], "Draft")
    with row1[2]:
        _stat_tile(by_status["listed"], "Published")
    with row1[3]:
        _stat_tile(by_status["sold"], "Sold")

    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

    row2 = st.columns(3)
    with row2[0]:
        _stat_tile(stats["created_today"], "Created Today")
    with row2[1]:
        _stat_tile(stats["created_this_week"], "Created This Week")
    with row2[2]:
        _stat_tile(by_status["ready"], "Ready for Shopify")

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    st.divider()

    st.markdown("#### Quick Actions")
    action_cols = st.columns(3)
    with action_cols[0]:
        if st.button("🧵 New Batch", key="dash_new_batch", width="stretch", type="primary"):
            st.session_state["main_nav_control"] = "🧵 AI Listing Generator"
            st.rerun()
    with action_cols[1]:
        if st.button("🛍️ View Listings", key="dash_view_listings", width="stretch"):
            st.session_state["main_nav_control"] = "🛍️ My Listings"
            st.rerun()
    with action_cols[2]:
        if st.button("🧩 View Templates", key="dash_view_templates", width="stretch"):
            st.session_state["main_nav_control"] = "🧩 Templates"
            st.rerun()

    st.divider()

    st.markdown("#### Recent Activity")
    activity = list_activity(user_id, limit=8)
    if not activity:
        st.info("Nothing yet — activity shows up here once you create or edit a listing.")
    else:
        for entry in activity:
            action_label = {
                "listing_created": "🧵 Created listing",
                "listing_updated": "✏️ Updated listing",
                "listing_deleted": "🗑️ Deleted listing",
                "template_created": "🧩 Created template",
                "template_updated": "✏️ Updated template",
                "template_deleted": "🗑️ Deleted template",
            }.get(entry["action"], entry["action"])
            when = str(entry.get("created_at", ""))[:16].replace("T", " ")
            st.markdown(
                f"<div style='padding:8px 0; border-bottom:1px solid rgba(43,33,48,.08); "
                f"font-size:13.5px;'>"
                f"<b>{action_label}</b> — {entry.get('item_label') or 'Untitled'} "
                f"<span style='color:#9A8B94;'>· {when}</span></div>",
                unsafe_allow_html=True,
            )
