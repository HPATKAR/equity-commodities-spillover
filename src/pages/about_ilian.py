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
    _f = "font-family:var(--font-sans);"

    photo = _photo_html("photo_ilian.jpeg", "Ilian Zalomai")

    # ── hero banner ──
    st.markdown(
        "<div class='about-hero'><div class='about-hero-inner'>"
        f"{photo}"
        "<div class='hero-body'>"
        "<p class='overline'>About the Author</p>"
        "<h1>Ilian Zalomai</h1>"
        "<p class='subtitle'>MSF Candidate &middot; Purdue Daniels School of Business</p>"
        "<p class='tagline'>Finance professional with interests in global markets, "
        "institutional research, and quantitative strategies. Committed to rigorous "
        "analytical work that bridges theory and real-world market dynamics.</p>"
        "<div class='links'>"
        "<a href='https://www.linkedin.com/in/ilian-zalomai-55iz/' target='_blank'>LinkedIn</a>"
        "</div></div>"
        "</div></div>",
        unsafe_allow_html=True,
    )

    # ── two-column body ──
    col_main, col_side = st.columns([1.55, 1], gap="large")

    with col_main:
        st.markdown(
            "<div class='about-card'>"
            "<p class='about-card-title'>Profile</p>"
            f"<p style='{_f}color:#1a1a1a;font-size:0.74rem;line-height:1.75;margin:0 0 10px 0;'>"
            "Deeply interested in how macro forces and geopolitical dynamics shape financial markets. "
            "Brings a research-oriented mindset to quantitative projects, with a focus on "
            "translating complex datasets into actionable investment insights.</p>"
            f"<p style='{_f}color:#1a1a1a;font-size:0.74rem;line-height:1.75;margin:0;'>"
            "Collaborative by nature and motivated by intellectually challenging problems, "
            "always looking to build connections at the intersection of finance, data, "
            "and global economic thinking.</p>"
            "</div>",
            unsafe_allow_html=True,
        )

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

        st.markdown(
            "<div class='about-card'>"
            "<p class='about-card-title'>Experience</p>"

            "<div class='exp-item'>"
            "<p class='exp-role'>MSF Candidate</p>"
            "<p class='exp-org'>Purdue University — Daniels School of Business</p>"
            "<p class='exp-meta'>2025 &ndash; 2026 &middot; West Lafayette, IN</p>"
            "<p class='exp-desc'>Graduate coursework spanning portfolio management, "
            "international finance, derivatives, and macroeconomic analysis. "
            "Actively engaged in applied research connecting geopolitical risk to asset pricing.</p></div>"

            "</div>",
            unsafe_allow_html=True,
        )

    with col_side:
        st.markdown(
            "<div class='about-card'>"
            "<p class='about-card-title'>Education</p>"
            "<div class='edu-item'>"
            "<p class='edu-school'>Purdue University</p>"
            "<p class='edu-dept'>Mitchell E. Daniels, Jr. School of Business</p>"
            "<p class='edu-degree'>Master of Science in Finance</p>"
            "<p class='edu-year'>2025 &ndash; 2026</p></div>"
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            "<div class='about-card'>"
            "<p class='about-card-title'>Interests</p>"
            "<div>"
            "<span class='interest-tag interest-gold'>Global Macro</span>"
            "<span class='interest-tag interest-neutral'>Geopolitical Risk</span>"
            "<span class='interest-tag interest-gold'>Institutional Research</span>"
            "<span class='interest-tag interest-neutral'>Portfolio Strategy</span>"
            "<span class='interest-tag interest-gold'>Emerging Markets</span>"
            "</div></div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            "<div class='about-card'>"
            "<p class='about-card-title'>Acknowledgments</p>"
            "<p class='ack-text'><strong>Prof. Cinder Zhang</strong>, MGMT 69000: "
            "Connecting geopolitical events to systematic regime-based analysis</p>"
            "</div>",
            unsafe_allow_html=True,
        )

    _page_footer()
