"""About: Ilian Zalomai."""

from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st

from src.ui.shared import _about_page_styles, _page_footer

_ASSETS = Path(__file__).resolve().parent.parent.parent / "assets"


def _photo_html(filename: str, alt: str) -> str:
    p = _ASSETS / filename
    if not p.exists():
        return ""
    b64 = base64.b64encode(p.read_bytes()).decode()
    ext = p.suffix.lstrip(".")
    return (
        f"<div class='hero-photo'>"
        f"<img src='data:image/{ext};base64,{b64}' alt='{alt}' />"
        f"</div>"
    )


def page_about_ilian() -> None:
    _about_page_styles()
    _f = "font-family:'DM Sans',sans-serif;"

    photo = _photo_html("photo_ilian.jpeg", "Ilian Zalomai")

    # ── hero banner ──
    st.markdown(
        "<div class='about-hero'><div class='about-hero-inner'>"
        f"{photo}"
        "<div class='hero-body'>"
        "<p class='overline'>About the Author</p>"
        "<h1>Ilian Zalomai</h1>"
        "<p class='subtitle'>MSF Candidate &middot; Purdue Daniels School of Business"
        " &middot; Payment Systems &amp; Fraud | Former Deloitte</p>"
        "<p class='tagline'>Fintech and banking professional with 4+ years leading payment "
        "systems and fraud operations at scale, combined with consulting experience at Deloitte "
        "across Frankfurt and Leipzig. Bridging operational finance and quantitative analytics.</p>"
        "<div class='links'>"
        "<a href='https://www.linkedin.com/in/ilian-zalomai-55iz/' target='_blank'>LinkedIn</a>"
        "</div></div>"
        "</div></div>",
        unsafe_allow_html=True,
    )

    # ── two-column body ──
    col_main, col_side = st.columns([1.55, 1], gap="large")

    with col_main:
        # bio
        st.markdown(
            "<div class='about-card'>"
            "<p class='about-card-title'>Profile</p>"
            f"<p style='{_f}color:#1a1a1a;font-size:0.74rem;line-height:1.75;margin:0 0 10px 0;'>"
            "Experienced in payment systems, fraud prevention, and banking technology consulting. "
            "I have spent four years building and managing risk and security operations across "
            "high-volume travel fintech platforms, and have contributed to banking transformation "
            "initiatives at Deloitte across Germany.</p>"
            f"<p style='{_f}color:#1a1a1a;font-size:0.74rem;line-height:1.75;margin:0;'>"
            "The MSF at Purdue's Financial Analytics track is deepening my foundation in "
            "quantitative methods and derivatives - bringing rigor to the intersection of "
            "risk management, data analytics, and financial markets.</p>"
            "</div>",
            unsafe_allow_html=True,
        )

        # project
        st.markdown(
            "<div class='about-card'>"
            "<p class='about-card-title'>This Project</p>"
            f"<p style='{_f}color:#1a1a1a;font-size:0.74rem;line-height:1.75;margin:0 0 12px 0;'>"
            "Contributed to the Equity &amp; Commodities Spillover Monitor as Course Project 3 for "
            "Prof. Cinder Zhang's MGMT 69000-120. Focused on geopolitical context analysis, "
            "war impact mapping, and the narrative framework connecting macro events to market regimes.</p>"
            "<div class='stat-row'>"
            "<div class='stat-item'><p class='stat-num'>4</p><p class='stat-label'>Regime States</p></div>"
            "<div class='stat-item'><p class='stat-num'>10</p><p class='stat-label'>Dashboard Pages</p></div>"
            "<div class='stat-item'><p class='stat-num'>6</p><p class='stat-label'>Trade Ideas</p></div>"
            "<div class='stat-item'><p class='stat-num'>5</p><p class='stat-label'>Analytical Layers</p></div>"
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )

        # experience
        st.markdown(
            "<div class='about-card'>"
            "<p class='about-card-title'>Experience</p>"

            "<div class='exp-item'>"
            "<p class='exp-role'>Payment Systems, Fraud &amp; Security Supervisor</p>"
            "<p class='exp-org'>Firebird Tours (Rail.Ninja &middot; Firebirdtours &middot; Triptile)</p>"
            "<p class='exp-meta'>Jan 2024 &ndash; Present &middot; Full-time</p>"
            "<p class='exp-desc'>Managing payment systems, risk, and security across high-volume "
            "travel platforms. Overseeing fraud monitoring, optimizing anti-fraud rules and "
            "customer verification, and building analytical reports and models. Implemented "
            "outbound-chargeback processes and increased chargeback win rate while reducing "
            "average document preparation time by 70%.</p></div>"

            "<div class='exp-item'>"
            "<p class='exp-role'>Banking Technology Strategy &amp; Transformation Analyst</p>"
            "<p class='exp-org'>Deloitte</p>"
            "<p class='exp-meta'>Apr 2024 &ndash; Aug 2024 &middot; Frankfurt, Germany</p>"
            "<p class='exp-desc'>Supported Privileged Access Management assessment implementation. "
            "Conducted comprehensive research on compliance trends with internal and external "
            "regulators. Prepared management sales deck materials and participated in team "
            "coordination across initiative workstreams.</p></div>"

            "<div class='exp-item'>"
            "<p class='exp-role'>Banking Operations Analyst</p>"
            "<p class='exp-org'>Deloitte</p>"
            "<p class='exp-meta'>Apr 2023 &ndash; Jul 2023 &middot; Leipzig, Germany</p>"
            "<p class='exp-desc'>Supported banking initiatives in Credit, Cloud, and Web 3.0/Metaverse. "
            "Researched current market trends including platformification, cloud hyperscaler "
            "sustainability, and metaverse applications in banking. Prepared and presented "
            "management decks across initiative workstreams.</p></div>"

            "<div class='exp-item'>"
            "<p class='exp-role'>Payment Systems, Fraud &amp; Security Team Lead</p>"
            "<p class='exp-org'>Firebird Tours</p>"
            "<p class='exp-meta'>Mar 2022 &ndash; Dec 2023 &middot; 1 yr 10 mos</p></div>"

            "<div class='exp-item'>"
            "<p class='exp-role'>Credit Department Analyst</p>"
            "<p class='exp-org'>Volksbanken Raiffeisenbanken &middot; Part-time</p>"
            "<p class='exp-meta'>May 2022 &ndash; Jul 2022 &middot; Mittweida, Germany</p>"
            "<p class='exp-desc'>Collected, verified, and organized credit documentation. "
            "Supported research on cost-of-living trends and banking approach "
            "plausibility assessments.</p></div>"

            "<div class='exp-item'>"
            "<p class='exp-role'>MUN Security Council Chairman &amp; Organising Committee</p>"
            "<p class='exp-org'>United Nations in Belarus (FIRMUN / OctoMUN)</p>"
            "<p class='exp-meta'>Oct 2018 &ndash; Mar 2022 &middot; Minsk, Belarus</p>"
            "<p class='exp-desc'>Served as Security Council Chairman and President of General "
            "Assembly across multiple Model UN conferences. Co-chaired EcoFin committee and "
            "led the organizing committee for FIRMUN and OctoMUN editions.</p></div>"

            "</div>",
            unsafe_allow_html=True,
        )

    with col_side:
        # education
        st.markdown(
            "<div class='about-card'>"
            "<p class='about-card-title'>Education</p>"
            "<div class='edu-item'>"
            "<p class='edu-school'>Purdue University</p>"
            "<p class='edu-dept'>Mitchell E. Daniels, Jr. School of Business</p>"
            "<p class='edu-degree'>M.S. Finance &middot; Financial Analytics Track</p>"
            "<p class='edu-year'>Jul 2025 &ndash; May 2026</p></div>"
            "<div class='edu-item'>"
            "<p class='edu-school'>Hochschule Mittweida</p>"
            "<p class='edu-dept'>Germany</p>"
            "<p class='edu-degree'>B.A. Business Administration</p>"
            "<p class='edu-year'>Mar 2022 &ndash; Apr 2024</p></div>"
            "</div>",
            unsafe_allow_html=True,
        )

        # interests
        st.markdown(
            "<div class='about-card'>"
            "<p class='about-card-title'>Interests</p>"
            "<div>"
            "<span class='interest-tag interest-gold'>Fintech &amp; Risk</span>"
            "<span class='interest-tag interest-neutral'>Payment Systems</span>"
            "<span class='interest-tag interest-gold'>Financial Analytics</span>"
            "<span class='interest-tag interest-neutral'>Fraud Prevention</span>"
            "<span class='interest-tag interest-gold'>Banking Strategy</span>"
            "</div></div>",
            unsafe_allow_html=True,
        )

        # acknowledgments
        st.markdown(
            "<div class='about-card'>"
            "<p class='about-card-title'>Acknowledgments</p>"
            "<p class='ack-text'><strong>Prof. Cinder Zhang</strong>, MGMT 69000: "
            "Connecting geopolitical events to systematic regime-based market analysis</p>"
            "</div>",
            unsafe_allow_html=True,
        )

    _page_footer()
