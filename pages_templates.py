"""
Templates — real CRUD against the `templates` table (see
supabase/migrations/0003_pages.sql, app_data.py). "Use Template" is
wired into the AI Listing Generator's upload step (see app.py) by
stashing the chosen template's defaults in session_state; generation
applies them onto each result without changing the existing upload/
review/QA flow itself.
"""

import streamlit as st

from ai_listing import SHOPIFY_CATEGORY_PATHS
from app_data import (
    list_templates,
    create_template,
    update_template,
    delete_template,
    duplicate_template,
)


def _template_card(user_id, template):
    template_id = template["id"]
    with st.container(key=f"tpl_card_{template_id}"):
        st.markdown(f"**{template.get('name', 'Untitled')}**")
        if template.get("description"):
            st.caption(template["description"])

        chips = []
        if template.get("default_brand"):
            chips.append(f"Brand: {template['default_brand']}")
        if template.get("default_condition"):
            chips.append(f"Condition: {template['default_condition']}")
        if template.get("default_category"):
            chips.append(f"Category: {template['default_category']}")
        if template.get("default_hashtags"):
            chips.append(f"{len(template['default_hashtags'])} hashtag(s)")
        if chips:
            st.caption(" · ".join(chips))

        button_cols = st.columns(4)
        with button_cols[0]:
            if st.button("Use", key=f"tpl_use_{template_id}", width="stretch", type="primary"):
                st.session_state["active_template"] = template
                st.session_state["main_nav_control"] = "🧵 AI Listing Generator"
                st.success(f"“{template['name']}” will be applied to your next generated batch.")
                st.rerun()
        with button_cols[1]:
            if st.button("Edit", key=f"tpl_edit_{template_id}", width="stretch"):
                st.session_state["tpl_editing_id"] = template_id
                st.rerun()
        with button_cols[2]:
            if st.button("Duplicate", key=f"tpl_dup_{template_id}", width="stretch"):
                duplicate_template(user_id, template_id)
                st.rerun()
        with button_cols[3]:
            if st.button("Delete", key=f"tpl_del_{template_id}", width="stretch"):
                st.session_state["tpl_confirm_delete_id"] = template_id
                st.rerun()

        if st.session_state.get("tpl_confirm_delete_id") == template_id:
            st.warning(f"Delete **{template.get('name')}**?")
            confirm_cols = st.columns(2)
            with confirm_cols[0]:
                if st.button("Yes, delete", key=f"tpl_confirm_yes_{template_id}", type="primary"):
                    delete_template(user_id, template_id, label=template.get("name"))
                    st.session_state.pop("tpl_confirm_delete_id", None)
                    st.rerun()
            with confirm_cols[1]:
                if st.button("Cancel", key=f"tpl_confirm_no_{template_id}"):
                    st.session_state.pop("tpl_confirm_delete_id", None)
                    st.rerun()

        if st.session_state.get("tpl_editing_id") == template_id:
            _template_form(user_id, existing=template)


def _template_form(user_id, existing=None):
    is_edit = existing is not None
    form_key = f"tpl_form_{existing['id']}" if is_edit else "tpl_form_new"

    with st.form(form_key):
        name = st.text_input("Template Name", value=(existing or {}).get("name", ""), placeholder="e.g. Y2K Top")
        description = st.text_area(
            "Description", value=(existing or {}).get("description", ""),
            placeholder="What this template is for...", height=80,
        )
        cols = st.columns(2)
        with cols[0]:
            brand = st.text_input("Default Brand", value=(existing or {}).get("default_brand", ""))
            condition_options = ["", "New with tags", "Excellent", "Good", "Fair"]
            existing_condition = (existing or {}).get("default_condition", "") or ""
            condition = st.selectbox(
                "Default Condition", condition_options,
                index=condition_options.index(existing_condition) if existing_condition in condition_options else 0,
            )
        with cols[1]:
            category_options = [""] + SHOPIFY_CATEGORY_PATHS
            existing_category = (existing or {}).get("default_category", "")
            category = st.selectbox(
                "Default Category", category_options,
                index=category_options.index(existing_category) if existing_category in category_options else 0,
            )
            hashtags_text = st.text_input(
                "Default Hashtags (comma-separated)",
                value=", ".join((existing or {}).get("default_hashtags", []) or []),
            )

        submit_cols = st.columns(2)
        with submit_cols[0]:
            saved = st.form_submit_button("Save Template" if not is_edit else "Save Changes", type="primary", width="stretch")
        with submit_cols[1]:
            cancelled = st.form_submit_button("Cancel", width="stretch")

    if saved:
        if not name.strip():
            st.error("Give the template a name.")
            return
        hashtags = [tag.strip() for tag in hashtags_text.split(",") if tag.strip()]
        if is_edit:
            update_template(
                user_id, existing["id"],
                name=name, description=description, default_brand=brand,
                default_condition=condition, default_category=category,
                default_hashtags=hashtags,
            )
            st.session_state.pop("tpl_editing_id", None)
        else:
            create_template(
                user_id, name, description=description, default_brand=brand,
                default_condition=condition, default_category=category,
                default_hashtags=hashtags,
            )
            st.session_state["tpl_show_new_form"] = False
        st.success("Saved.")
        st.rerun()

    if cancelled:
        st.session_state.pop("tpl_editing_id", None)
        st.session_state["tpl_show_new_form"] = False
        st.rerun()


def render_templates(user_id):
    st.markdown(
        """
        <div class="brand-header">
            <div class="brand-header-shimmer"></div>
            <div class="brand-header-greet">Templates</div>
            <div class="brand-header-title">Reusable listing defaults &#10023;</div>
            <div class="brand-header-sub">Save common brand/condition/hashtag combos and apply them in one click.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.get("active_template"):
        st.info(
            f"“{st.session_state['active_template']['name']}” is set to apply to your next "
            "generated batch — head to AI Listing Generator when ready, or clear it below."
        )
        if st.button("Clear active template", key="tpl_clear_active"):
            st.session_state.pop("active_template", None)
            st.rerun()
        st.divider()

    if st.button("➕ New Template", key="tpl_new_toggle", type="primary"):
        st.session_state["tpl_show_new_form"] = not st.session_state.get("tpl_show_new_form", False)
        st.rerun()

    if st.session_state.get("tpl_show_new_form"):
        _template_form(user_id)
        st.divider()

    templates = list_templates(user_id)
    if not templates:
        st.info("No templates yet. Create one above — e.g. “Y2K Top,” “Hollister,” “Coquette.”")
        return

    st.caption(f"{len(templates)} template(s)")
    for template in templates:
        _template_card(user_id, template)
        st.divider()
