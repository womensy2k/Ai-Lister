"""
Favorites — real per-user favorites against the `favorites` table
(supabase/migrations/0003_pages.sql). Supabase (via RLS), not
localStorage, is the source of truth; the favorite/unfavorite toggle
lives here and on the My Listings page (pages_my_listings.py), both
reading/writing the same table.
"""

import streamlit as st

from app_data import list_favorite_listings, get_listing_photo_urls, toggle_favorite


def render_favorites(user_id):
    st.markdown(
        """
        <div class="brand-header">
            <div class="brand-header-shimmer"></div>
            <div class="brand-header-greet">Favorites</div>
            <div class="brand-header-title">Your favorited listings &#10023;</div>
            <div class="brand-header-sub">Saved from My Listings — click the heart there to add more.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    listings = list_favorite_listings(user_id)
    if not listings:
        st.info("No favorites yet. Head to **My Listings** and tap the heart on anything you want to save here.")
        return

    st.caption(f"{len(listings)} favorite(s)")

    cols_per_row = 4
    for row_start in range(0, len(listings), cols_per_row):
        row = listings[row_start:row_start + cols_per_row]
        row_cols = st.columns(cols_per_row)
        for col, listing in zip(row_cols, row):
            with col:
                with st.container(key=f"fav_card_{listing['id']}"):
                    photo_urls = get_listing_photo_urls(user_id, listing["id"])
                    if photo_urls:
                        st.image(photo_urls[0], width="stretch")
                    else:
                        st.markdown(
                            '<div style="aspect-ratio:1; background:#f3ede6; border-radius:8px;"></div>',
                            unsafe_allow_html=True,
                        )
                    st.markdown(f"**{listing.get('title') or 'Untitled'}**")
                    price = listing.get("suggested_price")
                    st.caption(f"${price:.0f}" if price is not None else "No price set")
                    if st.button("💗 Remove", key=f"fav_remove_{listing['id']}", width="stretch"):
                        toggle_favorite(user_id, listing["id"])
                        st.rerun()
