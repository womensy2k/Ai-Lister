"""
History — real, append-only activity log (see
supabase/migrations/0003_pages.sql's activity_log table). Every row is
written by app_data.log_activity(), called from the actual generate/
edit/delete hooks in app.py, qa_review.py, and pages_templates.py —
nothing here is fabricated, and activity only exists from the point
this feature shipped forward, same as the user asked.
"""

import streamlit as st

from app_data import list_activity

ACTION_LABELS = {
    "listing_created": ("🧵", "Listing created"),
    "listing_updated": ("✏️", "Listing updated"),
    "listing_deleted": ("🗑️", "Listing deleted"),
    "template_created": ("🧩", "Template created"),
    "template_updated": ("✏️", "Template updated"),
    "template_deleted": ("🗑️", "Template deleted"),
}


def render_history(user_id):
    st.markdown(
        """
        <div class="brand-header">
            <div class="brand-header-shimmer"></div>
            <div class="brand-header-greet">History</div>
            <div class="brand-header-title">Activity log &#10023;</div>
            <div class="brand-header-sub">Every listing and template change, in order.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    action_types = ["All"] + sorted({label for _, label in ACTION_LABELS.values()})
    filter_choice = st.selectbox("Filter", action_types, label_visibility="collapsed")

    activity = list_activity(user_id, limit=200)
    if not activity:
        st.info(
            "No activity recorded yet. This page only started tracking activity once it "
            "shipped — nothing from before is fabricated here."
        )
        return

    if filter_choice != "All":
        activity = [
            entry for entry in activity
            if ACTION_LABELS.get(entry["action"], ("", entry["action"]))[1] == filter_choice
        ]

    if not activity:
        st.info("Nothing matches that filter.")
        return

    st.caption(f"{len(activity)} event(s)")

    for entry in activity:
        icon, label = ACTION_LABELS.get(entry["action"], ("•", entry["action"]))
        when = str(entry.get("created_at", ""))[:19].replace("T", " ")
        st.markdown(
            f"""
            <div style="display:flex; align-items:center; gap:12px; padding:10px 4px;
                        border-bottom:1px solid rgba(43,33,48,.08);">
                <span style="font-size:18px;">{icon}</span>
                <div style="flex:1;">
                    <div style="font-weight:700; font-size:13.5px;">{label}</div>
                    <div style="font-size:12.5px; color:#9A8B94;">{entry.get('item_label') or 'Untitled'}</div>
                </div>
                <div style="font-size:11.5px; color:#B7A9B1; white-space:nowrap;">{when} UTC</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
