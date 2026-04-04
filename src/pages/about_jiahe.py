"""About: Jiahe Miao."""

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


def page_about_jiahe() -> None:
    _about_page_styles()
    _f = "font-family:'DM Sans',sans-serif;color:#8890a1;font-size:0.62rem;line-height:1.70;"

    photo = _photo_html("photo_jiahe.jpeg", "Jiahe Miao")

    # ── hero banner ──
    st.markdown(
        "<div class='about-hero'><div class='about-hero-inner'>"
        f"{photo}"
        "<div class='hero-body'>"
        "<p class='overline'>About the Author</p>"
        "<div class='about-hero-name'>Jiahe Miao</div>"
        "<p class='subtitle'>MSF Candidate &middot; Purdue Daniels School of Business"
        " &middot; B.S. Information Systems, Kelley School of Business</p>"
        "<p class='tagline'>Capital markets and corporate finance professional with "
        "internship experience across Chinese securities, banking, and U.S. fintech. "
        "Brings a quantitative and data-driven lens to financial analysis and investor research.</p>"
        "<div class='links'>"
        "<a href='https://www.linkedin.com/in/jiahe-miao071/' target='_blank'>LinkedIn</a>"
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
            f"<p style='{_f}margin:0 0 10px 0;'>"
            "Passionate about the intersection of quantitative finance and global capital markets. "
            "With hands-on experience in securities research, banking operations, and venture-backed "
            "pitch development, I enjoy building financial models that translate data into decisions.</p>"
            f"<p style='{_f}margin:0;'>"
            "A background in information systems gives me an edge in structuring and automating "
            "analytical workflows. Always eager to connect and exchange ideas on markets, "
            "modeling, and finance.</p>"
            "</div>",
            unsafe_allow_html=True,
        )

        # project
        st.markdown(
            "<div class='about-card'>"
            "<p class='about-card-title'>This Project</p>"
            f"<p style='{_f}margin:0 0 10px 0;'>"
            "Quantitative Research contributor on the Equity-Commodities Spillover Monitor - "
            "Course Project 3 for Prof. Cinder Zhang's MGMT 69000-120.</p>"
            f"<ul style='{_f}margin:0 0 14px 0;padding-left:1.1rem'>"
            "<li style='margin-bottom:5px'>Designed the 4-state correlation regime taxonomy (Low / Moderate / Elevated / Crisis) "
            "and calibrated the rolling window thresholds for regime classification</li>"
            "<li style='margin-bottom:5px'>Specified the 6 regime-conditioned trade idea structures: entry/exit ranges, "
            "correlation breakpoint triggers, and risk management rationale</li>"
            "<li style='margin-bottom:5px'>Researched the Diebold-Yilmaz FEVD methodology and contributed to network "
            "edge threshold calibration and directional spillover interpretation</li>"
            "<li style='margin-bottom:5px'>Defined the Fixed Income cross-asset stress signal framework "
            "(TLT / HYG / LQD / EMB metrics) and validated the private credit bubble proxy selection</li>"
            "<li style='margin-bottom:0'>Reviewed equity and commodity return series for data quality; "
            "resolved TOPIX ETF proxy and Nickel ticker issues</li>"
            "</ul>"
            "<div class='stat-row'>"
            "<div class='stat-item'><p class='stat-num'>4</p><p class='stat-label'>Regime States</p></div>"
            "<div class='stat-item'><p class='stat-num'>6</p><p class='stat-label'>Trade Structures</p></div>"
            "<div class='stat-item'><p class='stat-num'>32</p><p class='stat-label'>Assets Covered</p></div>"
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
            "<p class='exp-role'>Investment Consultant (Practicum)</p>"
            "<p class='exp-org'>Fino Advisors LLC</p>"
            "<p class='exp-meta'>Jan 2026 &ndash; Present &middot; Remote</p>"
            "<p class='exp-desc'>Conducted investor research to support clients' Angel and Pre-A "
            "fundraising, evaluating 40+ venture firms' investment history, fund size, and sector focus. "
            "Developed 20-page pitch deck materials including financial models forecasting future "
            "cash flows and market research on clean energy demand and market size.</p></div>"

            "<div class='exp-item'>"
            "<p class='exp-role'>Corporate Finance Intern</p>"
            "<p class='exp-org'>China Everbright Bank</p>"
            "<p class='exp-meta'>Dec 2023 &ndash; Feb 2024 &middot; Zhengzhou, Henan, China</p>"
            "<p class='exp-desc'>Analyzed financial products from 12 commercial banks, assessing "
            "risks and producing a research-based strategic report. Strengthened client relationships "
            "by recommending financial products tailored to individual risk profiles. Optimized the "
            "bank's cash management systems and products, improving efficiency and regulatory "
            "compliance.</p></div>"

            "<div class='exp-item'>"
            "<p class='exp-role'>Integrated Finance Intern</p>"
            "<p class='exp-org'>Shenwan Hongyuan Securities Co., Ltd.</p>"
            "<p class='exp-meta'>Jun 2023 &ndash; Aug 2023 &middot; Zhengzhou, Henan, China</p>"
            "<p class='exp-desc'>Performed in-depth analysis of bond issuance projects, identifying "
            "risk factors and ensuring regulatory compliance. Supported IPO due diligence by "
            "analyzing financial statements, forecasting cash flows and EBITDA, and identifying "
            "potential risks. Developed financial models to calculate total unbundling amounts "
            "for corporate equity incentive programs.</p></div>"

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
            "<p class='edu-degree'>Master of Science in Finance</p>"
            "<p class='edu-year'>Jul 2025 &ndash; May 2026</p></div>"
            "<div class='edu-item'>"
            "<p class='edu-school'>Indiana University</p>"
            "<p class='edu-dept'>Kelley School of Business</p>"
            "<p class='edu-degree'>B.S. Information Systems</p>"
            "<p class='edu-year'>Aug 2021 &ndash; May 2025</p></div>"
            "</div>",
            unsafe_allow_html=True,
        )

        # interests
        st.markdown(
            "<div class='about-card'>"
            "<p class='about-card-title'>Interests</p>"
            "<div>"
            "<span class='interest-tag interest-gold'>Capital Markets</span>"
            "<span class='interest-tag interest-neutral'>Corporate Finance</span>"
            "<span class='interest-tag interest-gold'>Fixed Income</span>"
            "<span class='interest-tag interest-neutral'>Financial Modeling</span>"
            "<span class='interest-tag interest-gold'>Venture Research</span>"
            "</div></div>",
            unsafe_allow_html=True,
        )

        # acknowledgments
        st.markdown(
            "<div class='about-card'>"
            "<p class='about-card-title'>Acknowledgments</p>"
            "<p class='ack-text'><strong>Prof. Cinder Zhang</strong>, MGMT 69000: "
            "Regime-based thinking and equity-commodities spillover framework design</p>"
            "</div>",
            unsafe_allow_html=True,
        )

    _page_footer()
