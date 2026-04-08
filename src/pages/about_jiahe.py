"""About: Jiahe Miao."""

from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st

from src.ui.shared import _page_footer

_ASSETS = Path(__file__).resolve().parent.parent.parent / "assets"
_GOLD = "#CFB991"

_STYLE = """<style>
.abt-label{font-family:'JetBrains Mono',monospace!important;font-size:9px!important;font-weight:700!important;text-transform:uppercase;letter-spacing:.18em;color:#8E9AAA!important;display:block;}
.abt-name{font-family:'DM Sans',sans-serif!important;font-size:14px!important;font-weight:700!important;color:#e8e9ed!important;letter-spacing:-.01em;line-height:1.15;display:block;}
.abt-sub{font-family:'DM Sans',sans-serif!important;font-size:10px!important;color:#CFB991!important;font-weight:500;line-height:1.5;display:block;}
.abt-tgln{font-family:'DM Sans',sans-serif!important;font-size:11px!important;color:#a8b0c0!important;line-height:1.68;display:block;}
.abt-body{font-family:'DM Sans',sans-serif!important;font-size:11px!important;color:#a8b0c0!important;line-height:1.72;}
.abt-role{font-family:'DM Sans',sans-serif!important;font-size:11px!important;font-weight:700!important;color:#e8e9ed!important;line-height:1.3;display:block;}
.abt-org{font-family:'DM Sans',sans-serif!important;font-size:10px!important;font-weight:600!important;color:#CFB991!important;display:block;}
.abt-meta{font-family:'JetBrains Mono',monospace!important;font-size:9px!important;color:#8E9AAA!important;display:block;}
.abt-sch{font-family:'DM Sans',sans-serif!important;font-size:11px!important;font-weight:700!important;color:#e8e9ed!important;display:block;}
.abt-dept{font-family:'DM Sans',sans-serif!important;font-size:10px!important;color:#a8b0c0!important;display:block;}
.abt-deg{font-family:'DM Sans',sans-serif!important;font-size:10px!important;color:#CFB991!important;font-weight:500;display:block;}
.abt-year{font-family:'JetBrains Mono',monospace!important;font-size:9px!important;color:#8E9AAA!important;display:block;}
.abt-link{font-family:'JetBrains Mono',monospace!important;font-size:9px!important;font-weight:700!important;color:#CFB991!important;text-decoration:none!important;letter-spacing:.10em;text-transform:uppercase;opacity:.9;margin-right:1.2rem;display:inline-block;}
.abt-link:hover{opacity:1!important;}
.abt-num{font-family:'JetBrains Mono',monospace!important;font-size:16px!important;font-weight:700!important;color:#CFB991!important;line-height:1;display:block;}
.abt-slbl{font-family:'DM Sans',sans-serif!important;font-size:8px!important;text-transform:uppercase;letter-spacing:.14em;color:#8E9AAA!important;display:block;}
.abt-tag-g{display:inline-block;padding:2px 8px;font-family:'DM Sans',sans-serif!important;font-size:9px!important;font-weight:600!important;margin:2px;background:rgba(207,185,145,.08);color:#CFB991!important;border:1px solid rgba(207,185,145,.22);}
.abt-tag-n{display:inline-block;padding:2px 8px;font-family:'DM Sans',sans-serif!important;font-size:9px!important;font-weight:600!important;margin:2px;background:transparent;color:#a8b0c0!important;border:1px solid #2a2a2a;}
.abt-ack{font-family:'DM Sans',sans-serif!important;font-size:11px!important;color:#a8b0c0!important;line-height:1.68;}
</style>"""

_SEC  = "border-top:1px solid #1e1e1e;padding:0.75rem 0 0.25rem 0;margin-bottom:0.1rem;"
_EXPI = "padding:0.36rem 0 0.36rem 0.65rem;border-left:1px solid #1e1e1e;margin-bottom:0.36rem;"
_EDUI = "padding:0.36rem 0;border-bottom:1px solid #1a1a1a;"
_SROW = "display:flex;gap:0;margin-top:0.55rem;border-top:1px solid #1e1e1e;border-bottom:1px solid #1e1e1e;"
_SITM = "flex:1;text-align:center;padding:0.36rem 0.15rem;border-right:1px solid #1e1e1e;"


def _photo_html(filename: str, alt: str) -> str:
    p = _ASSETS / filename
    if not p.exists():
        return ""
    b64 = base64.b64encode(p.read_bytes()).decode()
    ext = p.suffix.lstrip(".")
    return (
        f"<div style='flex-shrink:0;width:130px;overflow:hidden;'>"
        f"<img src='data:image/{ext};base64,{b64}' alt='{alt}' "
        f"style='width:100%;height:100%;object-fit:cover;object-position:top center;"
        f"display:block;filter:grayscale(10%);' />"
        f"</div>"
    )


def _exp(role, org, meta, bullets=None):
    bullets_html = ""
    if bullets:
        items = "".join(
            f"<li style='margin-bottom:3px;'><span class='abt-body'>{b}</span></li>"
            for b in bullets
        )
        bullets_html = (
            f"<ul style='margin:4px 0 0 0;padding-left:14px;list-style:disc;'>{items}</ul>"
        )
    return (
        f"<div style='{_EXPI}'>"
        f"<span class='abt-role' style='margin:0 0 2px 0;'>{role}</span>"
        f"<span class='abt-org' style='margin:0 0 2px 0;'>{org}</span>"
        f"<span class='abt-meta' style='margin:0 0 4px 0;'>{meta}</span>"
        f"{bullets_html}"
        f"</div>"
    )


def page_about_jiahe() -> None:
    st.markdown(_STYLE, unsafe_allow_html=True)

    photo = _photo_html("photo_jiahe.jpeg", "Jiahe Miao")

    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown(
        f"<div style='background:#141414;border-top:2px solid {_GOLD};"
        f"border-bottom:1px solid #1e1e1e;overflow:hidden;margin-bottom:1.2rem;'>"
        f"<div style='display:flex;align-items:stretch;'>"
        f"{photo}"
        f"<div style='flex:1;padding:0.9rem 1.2rem;display:flex;flex-direction:column;"
        f"justify-content:center;border-left:1px solid #1e1e1e;'>"
        f"<span class='abt-label' style='margin:0 0 6px 0;'>About the Author</span>"
        f"<span class='abt-name' style='margin:0 0 4px 0;'>Jiahe Miao</span>"
        f"<span class='abt-sub' style='margin:0 0 6px 0;'>MSF Candidate &middot; Purdue Daniels School of Business "
        f"&middot; B.S. Information Systems, Kelley School of Business</span>"
        f"<span class='abt-tgln' style='margin:0 0 8px 0;'>Capital markets and corporate finance professional with internship experience "
        f"across Chinese securities, banking, and U.S. fintech. Brings a quantitative and data-driven lens "
        f"to financial analysis and investor research, with a background in information systems that enables "
        f"structured and automated analytical workflows.</span>"
        f"<div style='display:flex;flex-wrap:wrap;gap:0;'>"
        f"<a href='https://www.linkedin.com/in/jiahe-miao071/' target='_blank' class='abt-link'>LinkedIn</a>"
        f"</div>"
        f"</div></div></div>",
        unsafe_allow_html=True,
    )

    col_main, col_side = st.columns([1.55, 1], gap="large")

    with col_main:

        # ── Profile ───────────────────────────────────────────────────────────
        st.markdown(
            f"<div style='{_SEC}'>"
            f"<span class='abt-label' style='margin-bottom:8px;'>Profile</span>"
            f"<span class='abt-body' style='display:block;margin-bottom:6px;'>Passionate about the intersection of quantitative finance "
            f"and global capital markets. With hands-on experience in securities research, banking operations, and "
            f"venture-backed pitch development, I enjoy building financial models that translate data into clear, "
            f"actionable investment decisions.</span>"
            f"<span class='abt-body' style='display:block;'>A background in information systems gives me an edge in structuring and automating "
            f"analytical workflows &mdash; from bond issuance analysis to equity incentive modeling. Always eager to connect "
            f"and exchange ideas on markets, modeling, and finance.</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── This Project ──────────────────────────────────────────────────────
        st.markdown(
            f"<div style='{_SEC}'>"
            f"<span class='abt-label' style='margin-bottom:8px;'>This Project</span>"
            f"<span class='abt-body' style='display:block;margin-bottom:6px;'>Quantitative Research contributor on the "
            f"Equity-Commodities Spillover Monitor &mdash; Course Project 3 for Prof. Cinder Zhang's MGMT 69000-120.</span>"
            f"<ul style='margin:0 0 0 0;padding-left:14px;list-style:disc;'>"
            f"<li style='margin-bottom:3px;'><span class='abt-body'>Designed the 4-state correlation regime taxonomy "
            f"(Low / Moderate / Elevated / Crisis) and calibrated rolling window thresholds for regime classification, "
            f"defining the threshold rationale and lookback methodology</span></li>"
            f"<li style='margin-bottom:3px;'><span class='abt-body'>Specified 6 regime-conditioned trade idea structures: "
            f"entry/exit ranges, correlation breakpoint triggers, and risk management rationale for each state</span></li>"
            f"<li style='margin-bottom:3px;'><span class='abt-body'>Researched the Diebold-Yilmaz FEVD methodology and contributed "
            f"to network edge threshold calibration and directional spillover interpretation framework</span></li>"
            f"<li style='margin-bottom:3px;'><span class='abt-body'>Defined the Fixed Income cross-asset stress signal framework "
            f"(TLT/HYG/LQD/EMB metrics) and validated the private credit bubble proxy selection "
            f"(BKLN/ARCC/OBDC/FSK/JBBB) and composite score weighting logic</span></li>"
            f"<li style='margin-bottom:0;'><span class='abt-body'>Reviewed equity and commodity return series for data quality; "
            f"flagged and resolved TOPIX ETF proxy selection (1306.T) and Nickel ticker alignment (NILSY)</span></li>"
            f"</ul>"
            f"<div style='{_SROW}'>"
            f"<div style='{_SITM}'><span class='abt-num'>4</span><span class='abt-slbl'>Regime States</span></div>"
            f"<div style='{_SITM}'><span class='abt-num'>6</span><span class='abt-slbl'>Trade Structures</span></div>"
            f"<div style='{_SITM}'><span class='abt-num'>32</span><span class='abt-slbl'>Assets Covered</span></div>"
            f"<div style='{_SITM}border-right:none;'><span class='abt-num'>5</span><span class='abt-slbl'>Analytical Layers</span></div>"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── Experience ────────────────────────────────────────────────────────
        st.markdown(
            f"<div style='{_SEC}'>"
            f"<span class='abt-label' style='margin-bottom:8px;'>Professional Experience</span>"
            + _exp(
                "Investment Consultant (Practicum)",
                "Fino Advisors LLC",
                "January 2026 &ndash; Present &middot; Remote",
                [
                    "Conducted investor research to support clients' Angel and Pre-A fundraising rounds, evaluating "
                    "40+ venture firms' investment history, fund size, and sector focus to build targeted outreach lists.",
                    "Developed 20-page pitch deck materials including financial models forecasting future cash flows "
                    "and market research on clean energy demand and market size for a sustainability-focused startup.",
                ],
            )
            + _exp(
                "Corporate Finance Intern",
                "China Everbright Bank",
                "December 2023 &ndash; February 2024 &middot; Zhengzhou, Henan, China",
                [
                    "Analyzed financial products from 12 commercial banks, assessing risks and producing a research-based "
                    "strategic report on product competitiveness and risk-adjusted positioning.",
                    "Strengthened client relationships by recommending financial products tailored to individual risk profiles; "
                    "optimized cash management systems improving efficiency and regulatory compliance.",
                ],
            )
            + _exp(
                "Integrated Finance Intern",
                "Shenwan Hongyuan Securities Co., Ltd.",
                "June 2023 &ndash; August 2023 &middot; Zhengzhou, Henan, China",
                [
                    "Performed in-depth analysis of bond issuance projects, identifying risk factors and ensuring "
                    "regulatory compliance across the underwriting pipeline.",
                    "Supported IPO due diligence by analyzing financial statements, forecasting cash flows and EBITDA, "
                    "and identifying potential risks for prospective listings.",
                    "Developed financial models to calculate total unbundling amounts for corporate equity incentive "
                    "programs, supporting legal and compensation structuring.",
                ],
            )
            + "</div>",
            unsafe_allow_html=True,
        )

    with col_side:

        # ── Education ─────────────────────────────────────────────────────────
        st.markdown(
            f"<div style='{_SEC}'>"
            f"<span class='abt-label' style='margin-bottom:8px;'>Education</span>"
            f"<div style='{_EDUI}'>"
            f"<span class='abt-sch' style='margin:0 0 2px 0;'>Purdue University</span>"
            f"<span class='abt-dept' style='margin:0 0 2px 0;'>Mitchell E. Daniels, Jr. School of Business</span>"
            f"<span class='abt-deg' style='margin:0 0 2px 0;'>Master of Science in Finance</span>"
            f"<span class='abt-year'>July 2025 &ndash; May 2026 &middot; West Lafayette, IN</span></div>"
            f"<div style='{_EDUI}border-bottom:none;'>"
            f"<span class='abt-sch' style='margin:0 0 2px 0;'>Indiana University</span>"
            f"<span class='abt-dept' style='margin:0 0 2px 0;'>Kelley School of Business</span>"
            f"<span class='abt-deg' style='margin:0 0 2px 0;'>B.S. Information Systems</span>"
            f"<span class='abt-year'>August 2021 &ndash; May 2025 &middot; Bloomington, IN</span></div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── Interests ─────────────────────────────────────────────────────────
        st.markdown(
            f"<div style='{_SEC}'>"
            f"<span class='abt-label' style='margin-bottom:6px;'>Interests</span>"
            f"<div>"
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
            f"<span class='abt-label' style='margin-bottom:8px;'>Acknowledgments</span>"
            f"<div class='abt-ack' style='margin-bottom:0;'>"
            f"<strong style='color:#c8c8c8;font-weight:600;'>Prof. Cinder Zhang</strong> "
            f"&middot; MGMT 69000: Regime-based thinking and equity-commodities spillover framework design</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    _page_footer()
