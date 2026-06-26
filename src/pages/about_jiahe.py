"""About: Jiahe Miao."""

from __future__ import annotations

import streamlit as st

from src.ui.shared import _page_footer
from src.ui.about_shared import (
    _ABOUT_STYLE, _photo_html, _hero, _exp, _stat_row, _edu, _ack,
    _SEC,
)


def page_about_jiahe() -> None:
    st.markdown(_ABOUT_STYLE, unsafe_allow_html=True)

    photo = _photo_html("photo_jiahe.jpeg", "Jiahe Miao")

    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown(
        _hero(
            photo_html=photo,
            role_lbl="CONTRIBUTOR · QUANTITATIVE RESEARCH",
            name="Jiahe Miao",
            sub=(
                "MSF CANDIDATE &nbsp;·&nbsp; PURDUE DANIELS &nbsp;·&nbsp; "
                "B.S. Information Systems, Kelley School of Business"
            ),
            tagline=(
                "Capital markets and corporate finance professional with internship experience "
                "across Chinese securities, banking, and U.S. fintech. Brings a quantitative and "
                "data-driven lens to financial analysis and investor research, with a background in "
                "information systems that enables structured and automated analytical workflows."
            ),
            links_html=(
                "<a href='https://www.linkedin.com/in/jiahe-miao071/' target='_blank' "
                "class='abt-link'>LinkedIn</a>"
            ),
        ),
        unsafe_allow_html=True,
    )

    col_main, col_side = st.columns([1.55, 1], gap="large")

    with col_main:

        # ── Profile ───────────────────────────────────────────────────────────
        st.markdown(
            f"<div style='{_SEC}'>"
            f"<span class='abt-label'>Profile</span>"
            f"<span class='abt-body' style='display:block;margin-bottom:6px;'>"
            f"Passionate about the intersection of quantitative finance and global capital markets. "
            f"With hands-on experience in securities research, banking operations, and "
            f"venture-backed pitch development, I enjoy building financial models that translate "
            f"data into clear, actionable investment decisions.</span>"
            f"<span class='abt-body' style='display:block;'>"
            f"A background in information systems gives me an edge in structuring and automating "
            f"analytical workflows &mdash; from bond issuance analysis to equity incentive modeling. "
            f"Always eager to connect and exchange ideas on markets, modeling, and finance.</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── This Project ──────────────────────────────────────────────────────
        st.markdown(
            f"<div style='{_SEC}'>"
            f"<span class='abt-label'>This Project</span>"
            f"<span class='abt-body' style='display:block;margin-bottom:6px;'>"
            f"Quantitative Research contributor on the Equity-Commodities Spillover Monitor "
            f"&mdash; Course Project 3 for Prof. Cinder Zhang's MGMT 69000-120.</span>"
            f"<ul style='margin:0 0 10px;padding-left:14px;list-style:disc;'>"
            f"<li style='margin-bottom:4px;'><span class='abt-body'>Designed the 4-state correlation "
            f"regime taxonomy (Low / Moderate / Elevated / Crisis) and calibrated rolling window "
            f"thresholds for regime classification, defining the threshold rationale and lookback "
            f"methodology</span></li>"
            f"<li style='margin-bottom:4px;'><span class='abt-body'>Specified 6 regime-conditioned "
            f"trade idea structures: entry/exit ranges, correlation breakpoint triggers, and risk "
            f"management rationale for each state</span></li>"
            f"<li style='margin-bottom:4px;'><span class='abt-body'>Researched the Diebold-Yilmaz "
            f"FEVD methodology and contributed to network edge threshold calibration and directional "
            f"spillover interpretation framework</span></li>"
            f"<li style='margin-bottom:4px;'><span class='abt-body'>Defined the Fixed Income "
            f"cross-asset stress signal framework (TLT/HYG/LQD/EMB metrics) and validated the "
            f"private credit bubble proxy selection (BKLN/ARCC/OBDC/FSK/JBBB) and composite score "
            f"weighting logic</span></li>"
            f"<li style='margin-bottom:0;'><span class='abt-body'>Reviewed equity and commodity "
            f"return series for data quality; flagged and resolved TOPIX ETF proxy selection (1306.T) "
            f"and Nickel ticker alignment (NILSY)</span></li>"
            f"</ul>"
            + _stat_row([
                ("4",  "Regime States"),
                ("6",  "Trade Structures"),
                ("32", "Assets Covered"),
                ("5",  "Analytical Layers"),
            ])
            + f"</div>",
            unsafe_allow_html=True,
        )

        # ── Experience ────────────────────────────────────────────────────────
        st.markdown(
            f"<div style='{_SEC}'>"
            f"<span class='abt-label'>Professional Experience</span>"
            + _exp(
                "Investment Consultant (Practicum)", "Fino Advisors LLC",
                "January 2026 &ndash; Present &middot; Remote",
                [
                    "Conducted investor research to support clients' Angel and Pre-A fundraising "
                    "rounds, evaluating 40+ venture firms' investment history, fund size, and sector "
                    "focus to build targeted outreach lists.",
                    "Developed 20-page pitch deck materials including financial models forecasting "
                    "future cash flows and market research on clean energy demand and market size "
                    "for a sustainability-focused startup.",
                ],
            )
            + _exp(
                "Corporate Finance Intern", "China Everbright Bank",
                "December 2023 &ndash; February 2024 &middot; Zhengzhou, Henan, China",
                [
                    "Analyzed financial products from 12 commercial banks, assessing risks and "
                    "producing a research-based strategic report on product competitiveness and "
                    "risk-adjusted positioning.",
                    "Strengthened client relationships by recommending financial products tailored "
                    "to individual risk profiles; optimized cash management systems improving "
                    "efficiency and regulatory compliance.",
                ],
            )
            + _exp(
                "Integrated Finance Intern", "Shenwan Hongyuan Securities Co., Ltd.",
                "June 2023 &ndash; August 2023 &middot; Zhengzhou, Henan, China",
                [
                    "Performed in-depth analysis of bond issuance projects, identifying risk factors "
                    "and ensuring regulatory compliance across the underwriting pipeline.",
                    "Supported IPO due diligence by analyzing financial statements, forecasting cash "
                    "flows and EBITDA, and identifying potential risks for prospective listings.",
                    "Developed financial models to calculate total unbundling amounts for corporate "
                    "equity incentive programs, supporting legal and compensation structuring.",
                ],
            )
            + "</div>",
            unsafe_allow_html=True,
        )

    with col_side:

        # ── Education ─────────────────────────────────────────────────────────
        st.markdown(
            f"<div style='{_SEC}'>"
            f"<span class='abt-label'>Education</span>"
            + _edu(
                "Purdue University",
                "Mitchell E. Daniels, Jr. School of Business",
                "Master of Science in Finance",
                "July 2025 &ndash; May 2026 &middot; West Lafayette, IN",
            )
            + _edu(
                "Indiana University", "Kelley School of Business",
                "B.S. Information Systems",
                "August 2021 &ndash; May 2025 &middot; Bloomington, IN",
                last=True,
            )
            + "</div>",
            unsafe_allow_html=True,
        )

        # ── Interests ─────────────────────────────────────────────────────────
        st.markdown(
            f"<div style='{_SEC}'>"
            f"<span class='abt-label'>Interests</span>"
            f"<div style='margin-top:2px;'>"
            f"<span class='abt-tag-g'>Capital Markets</span>"
            f"<span class='abt-tag-n'>Corporate Finance</span>"
            f"<span class='abt-tag-g'>Fixed Income</span>"
            f"<span class='abt-tag-n'>Financial Modeling</span>"
            f"<span class='abt-tag-g'>Venture Research</span>"
            f"<span class='abt-tag-n'>Quantitative Finance</span>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

        # ── Acknowledgments ───────────────────────────────────────────────────
        st.markdown(
            f"<div style='{_SEC}'>"
            f"<span class='abt-label'>Acknowledgments</span>"
            + _ack(
                "Prof. Cinder Zhang",
                "MGMT 69000 &middot; Regime-based thinking and equity-commodities spillover "
                "framework design",
            )
            + "</div>",
            unsafe_allow_html=True,
        )

    _page_footer()
