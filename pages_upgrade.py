"""
Upgrade — visual plan comparison only. No payment processing (no
Stripe or similar exists anywhere in this project) — every action here
is inert by design, honestly labeled "Coming Soon" rather than
pretending to charge anyone.
"""

import streamlit as st

PLANS = [
    {
        "name": "Free",
        "price": "$0",
        "period": "forever",
        "features": [
            "Core AI listing generation",
            "Manual listing workflow",
            "Basic templates",
            "Standard support",
        ],
        "current": True,
    },
    {
        "name": "Pro",
        "price": "$—",
        "period": "/month",
        "features": [
            "Everything in Free",
            "Higher AI generation limits",
            "Priority generation speed",
            "Advanced analytics",
        ],
        "current": False,
    },
    {
        "name": "Business",
        "price": "$—",
        "period": "/month",
        "features": [
            "Everything in Pro",
            "Multiple marketplace integrations",
            "Team accounts",
            "Dedicated support",
        ],
        "current": False,
    },
]


def render_upgrade(user_id):
    st.markdown(
        """
        <div class="brand-header">
            <div class="brand-header-shimmer"></div>
            <div class="brand-header-greet">Upgrade</div>
            <div class="brand-header-title">Plans &amp; pricing &#10023;</div>
            <div class="brand-header-sub">Billing isn't live yet — this is a preview of what's coming.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info("💡 Payments aren't set up yet. Every plan below is a preview — nothing here charges you anything.")

    cols = st.columns(3)
    for col, plan in zip(cols, PLANS):
        with col:
            with st.container(key=f"upgrade_plan_{plan['name'].lower()}"):
                st.markdown(f"### {plan['name']}")
                st.markdown(
                    f"<div style='font-size:28px; font-weight:900; color:#F75AA9;'>"
                    f"{plan['price']}<span style='font-size:14px; color:#9A8B94; font-weight:600;'> {plan['period']}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
                for feature in plan["features"]:
                    st.markdown(f"✓ {feature}")
                st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
                if plan["current"]:
                    st.button("Current Plan", key=f"upgrade_btn_{plan['name']}", disabled=True, width="stretch")
                else:
                    st.button("🔒 Coming Soon", key=f"upgrade_btn_{plan['name']}", disabled=True, width="stretch")
