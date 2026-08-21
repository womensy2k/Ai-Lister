"""
My Listings — real listing management against the `listings` table
persisted by the existing generate/QA flow (see app_data.py's
persist_generated_listing, wired into app.py/qa_review.py). Only ever
queries with the current user's id, and every query is additionally
RLS-protected at the database layer.
"""

import streamlit as st

from app_data import (
    list_user_listings,
    get_listing_photo_urls,
    update_listing_fields,
    delete_listing,
    duplicate_listing,
    toggle_favorite,
    is_favorited,
)

STATUS_OPTIONS = ["All", "draft", "ready", "listed", "sold", "archived"]
STATUS_LABELS = {
    "draft": "Draft", "ready": "Ready", "listed": "Listed",
    "sold": "Sold", "archived": "Archived",
}
STATUS_COLORS = {
    "draft": "#9A8B94", "ready": "#F75AA9", "listed": "#22c55e",
    "sold": "#6B5B66", "archived": "#B7A9B1",
}


def _status_pill(status):
    color = STATUS_COLORS.get(status, "#9A8B94")
    label = STATUS_LABELS.get(status, status)
    return (
        f'<span style="background:{color}22; color:{color}; font-size:11px; '
        f'font-weight:800; padding:2px 9px; border-radius:999px;">{label}</span>'
    )


def render_my_listings(user_id):
    st.markdown(
        """
        <div class="brand-header">
            <div class="brand-header-shimmer"></div>
            <div class="brand-header-greet">My Listings</div>
            <div class="brand-header-title">Manage your listings &#10023;</div>
            <div class="brand-header-sub">Everything you've generated, in one place.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    filter_cols = st.columns([2, 1, 1])
    with filter_cols[0]:
        search = st.text_input("Search", placeholder="Search by title or brand...", label_visibility="collapsed")
    with filter_cols[1]:
        status_filter = st.selectbox("Status", STATUS_OPTIONS, label_visibility="collapsed")
    with filter_cols[2]:
        sort = st.selectbox(
            "Sort",
            ["created_desc", "created_asc", "price_desc", "price_asc", "title_asc"],
            format_func=lambda v: {
                "created_desc": "Newest first", "created_asc": "Oldest first",
                "price_desc": "Price: high to low", "price_asc": "Price: low to high",
                "title_asc": "Title A-Z",
            }[v],
            label_visibility="collapsed",
        )

    listings = list_user_listings(user_id, status=status_filter, search=search or None, sort=sort)

    if not listings:
        st.info(
            "No listings yet. Head to **AI Listing Generator** to create your first batch — "
            "generated listings show up here automatically."
        )
        return

    st.caption(f"{len(listings)} listing(s)")

    selected_ids = st.session_state.setdefault("my_listings_selected", set())

    bulk_cols = st.columns([1, 1, 3])
    with bulk_cols[0]:
        if st.button("Select All", key="ml_select_all", width="stretch"):
            st.session_state["my_listings_selected"] = {listing["id"] for listing in listings}
            st.rerun()
    with bulk_cols[1]:
        if st.button("Clear", key="ml_clear_selection", width="stretch"):
            st.session_state["my_listings_selected"] = set()
            st.rerun()
    if selected_ids:
        with bulk_cols[2]:
            if st.button(f"🗄️ Archive {len(selected_ids)} Selected", key="ml_bulk_archive", width="stretch"):
                for listing_id in selected_ids:
                    update_listing_fields(user_id, listing_id, status="archived")
                st.session_state["my_listings_selected"] = set()
                st.success("Archived.")
                st.rerun()

    st.divider()

    for listing in listings:
        listing_id = listing["id"]
        with st.container(key=f"ml_card_{listing_id}"):
            cols = st.columns([0.4, 1.2, 3, 1, 1, 1.4])

            with cols[0]:
                checked = st.checkbox(
                    "Select", value=listing_id in selected_ids,
                    key=f"ml_check_{listing_id}", label_visibility="collapsed",
                )
                if checked:
                    selected_ids.add(listing_id)
                else:
                    selected_ids.discard(listing_id)

            with cols[1]:
                photo_urls = get_listing_photo_urls(user_id, listing_id)
                if photo_urls:
                    st.image(photo_urls[0], width="stretch")
                else:
                    st.markdown(
                        '<div style="aspect-ratio:1; background:#f3ede6; border-radius:8px;"></div>',
                        unsafe_allow_html=True,
                    )

            with cols[2]:
                st.markdown(f"**{listing.get('title') or 'Untitled'}**")
                meta_bits = [
                    listing.get("brand") or "",
                    listing.get("size") or "",
                    str(listing.get("created_at", ""))[:10],
                ]
                st.caption(" · ".join(bit for bit in meta_bits if bit))
                st.markdown(_status_pill(listing.get("status", "draft")), unsafe_allow_html=True)

            with cols[3]:
                price = listing.get("suggested_price")
                st.markdown(f"**${price:.0f}**" if price is not None else "—")

            with cols[4]:
                favorited = is_favorited(user_id, listing_id)
                if st.button(
                    "💗" if favorited else "🤍",
                    key=f"ml_fav_{listing_id}",
                    help="Favorite" if not favorited else "Unfavorite",
                ):
                    toggle_favorite(user_id, listing_id)
                    st.rerun()

            with cols[5]:
                action = st.selectbox(
                    "Actions",
                    ["…", "Edit", "Duplicate", "Archive", "Delete"],
                    key=f"ml_action_{listing_id}",
                    label_visibility="collapsed",
                )
                if action != "…":
                    # Reset the selectbox's own widget state back to
                    # the placeholder BEFORE rerunning — Streamlit
                    # preserves a widget's session_state value across
                    # reruns by key, so without this the same action
                    # would keep re-triggering on every subsequent
                    # rerun instead of firing once.
                    st.session_state[f"ml_action_{listing_id}"] = "…"

                if action == "Edit":
                    st.session_state["ml_editing_id"] = listing_id
                    st.rerun()
                elif action == "Duplicate":
                    new_id = duplicate_listing(user_id, listing_id)
                    if new_id:
                        st.success("Duplicated.")
                        st.rerun()
                elif action == "Archive":
                    update_listing_fields(user_id, listing_id, status="archived")
                    st.rerun()
                elif action == "Delete":
                    st.session_state["ml_confirm_delete_id"] = listing_id
                    st.rerun()

            if st.session_state.get("ml_confirm_delete_id") == listing_id:
                st.warning(f"Delete **{listing.get('title') or 'this listing'}**? This can't be undone.")
                confirm_cols = st.columns(2)
                with confirm_cols[0]:
                    if st.button("Yes, delete it", key=f"ml_confirm_yes_{listing_id}", type="primary"):
                        delete_listing(user_id, listing_id, label=listing.get("title"))
                        st.session_state.pop("ml_confirm_delete_id", None)
                        st.rerun()
                with confirm_cols[1]:
                    if st.button("Cancel", key=f"ml_confirm_no_{listing_id}"):
                        st.session_state.pop("ml_confirm_delete_id", None)
                        st.rerun()

            if st.session_state.get("ml_editing_id") == listing_id:
                with st.form(f"ml_edit_form_{listing_id}"):
                    st.markdown("**Edit listing**")
                    new_title = st.text_input("Title", value=listing.get("title", ""))
                    edit_cols = st.columns(3)
                    with edit_cols[0]:
                        new_brand = st.text_input("Brand", value=listing.get("brand", ""))
                    with edit_cols[1]:
                        new_size = st.text_input("Size", value=listing.get("size", ""))
                    with edit_cols[2]:
                        new_price = st.number_input(
                            "Price", value=float(listing.get("suggested_price") or 0), min_value=0.0, step=1.0
                        )
                    new_status = st.selectbox(
                        "Status", STATUS_OPTIONS[1:],
                        index=STATUS_OPTIONS[1:].index(listing.get("status", "draft"))
                        if listing.get("status", "draft") in STATUS_OPTIONS[1:] else 0,
                    )
                    new_description = st.text_area("Description", value=listing.get("description", ""), height=140)

                    save_cols = st.columns(2)
                    with save_cols[0]:
                        saved = st.form_submit_button("Save Changes", type="primary", width="stretch")
                    with save_cols[1]:
                        cancelled = st.form_submit_button("Cancel", width="stretch")

                if saved:
                    update_listing_fields(
                        user_id, listing_id,
                        title=new_title, brand=new_brand, size=new_size,
                        suggested_price=new_price, status=new_status,
                        description=new_description,
                    )
                    st.session_state.pop("ml_editing_id", None)
                    st.success("Saved.")
                    st.rerun()
                if cancelled:
                    st.session_state.pop("ml_editing_id", None)
                    st.rerun()

        st.divider()
