"""
Analytics — real numbers only, sourced from the listings table. No
sales/revenue data exists anywhere in this app (no marketplace
integration reports back sold status yet), so that's shown as a clear
"coming soon" note rather than invented.
"""

import streamlit as st

from app_data import get_listing_stats, get_listings_created_per_day


def render_analytics(user_id):
    st.markdown(
        """
        <div class="brand-header">
            <div class="brand-header-shimmer"></div>
            <div class="brand-header-greet">Analytics</div>
            <div class="brand-header-title">Your listing activity &#10023;</div>
            <div class="brand-header-sub">Real numbers from your account — nothing estimated.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    stats = get_listing_stats(user_id)

    if stats["total"] == 0:
        st.info("No listings yet — analytics will fill in once you generate your first batch.")
        return

    row1 = st.columns(4)
    with row1[0]:
        st.metric("Total Listings", stats["total"])
    with row1[1]:
        st.metric("Created This Week", stats["created_this_week"])
    with row1[2]:
        st.metric("Created This Month", stats["created_this_month"])
    with row1[3]:
        st.metric("Drafts", stats["by_status"]["draft"])

    row2 = st.columns(4)
    with row2[0]:
        st.metric("Published", stats["by_status"]["listed"])
    with row2[1]:
        st.metric("Ready for Shopify", stats["by_status"]["ready"])
    with row2[2]:
        avg_price = stats["avg_price"]
        st.metric("Avg. Listing Price", f"${avg_price:.0f}" if avg_price is not None else "—")
    with row2[3]:
        total_value = stats["total_inventory_value"]
        st.metric("Total Inventory Value", f"${total_value:.0f}" if total_value is not None else "—")

    if stats["priced_count"] < stats["total"]:
        st.caption(
            f"Price stats based on {stats['priced_count']} of {stats['total']} listing(s) with a set price."
        )

    st.divider()
    st.markdown("#### Listings Created (Last 14 Days)")
    daily_counts = get_listings_created_per_day(user_id, days=14)
    chart_data = {date.strftime("%b %d"): count for date, count in daily_counts.items()}
    st.bar_chart(chart_data)

    st.divider()
    st.markdown("#### Sales Analytics")
    st.info(
        "Sales/revenue analytics will become available once a marketplace integration "
        "(like Shopify) reports back sold status — that data doesn't exist yet, so it's "
        "not shown here rather than guessed at."
    )
