"""About: Jiahe Miao."""

from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st

from src.ui.shared import _page_footer

_ASSETS = Path(__file__).resolve().parent.parent.parent / "assets"

_SANS = "font-family:'DM Sans',sans-serif;"
_MONO = "font-family:'JetBrains Mono',monospace;"
_GOLD = "#CFB991"

_S_LABEL  = f"{_MONO}font-size:0.50rem;font-weight:700;text-transform:uppercase;letter-spacing:0.18em;color:#555960;"
_S_NAME   = f"{_SANS}font-size:0.82rem;font-weight:700;color:#e8e9ed;letter-spacing:-0.01em;line-height:1.15;margin:0 0 0.16rem 0;"
_S_SUB    = f"{_SANS}font-size:0.60rem;color:{_GOLD};font-weight:500;margin:0 0 0.28rem 0;line-height:1.5;"
_S_TAG    = f"{_SANS}font-size:0.62rem;color:#8890a1;line-height:1.70;margin:0 0 0.55rem 0;"
_S_BODY   = f"{_SANS}font-size:0.62rem;color:#8890a1;line-height:1.70;"
_S_ROLE   = f"{_SANS}font-size:0.62rem;font-weight:700;color:#e8e9ed;margin:0 0 0.04rem 0;line-height:1.3;"
_S_ORG    = f"{_SANS}font-size:0.59rem;font-weight:600;color:{_GOLD};margin:0 0 0.05rem 0;"
_S_META   = f"{_MONO}font-size:0.50rem;color:#555960;margin:0 0 0.16rem 0;"
_S_DESC   = f"{_SANS}font-size:0.62rem;color:#8890a1;line-height:1.68;margin:0;"
_S_SCH    = f"{_SANS}font-size:0.62rem;font-weight:700;color:#e8e9ed;margin:0 0 0.04rem 0;"
_S_DEPT   = f"{_SANS}font-size:0.57rem;color:#8890a1;margin:0 0 0.04rem 0;"
_S_DEG    = f"{_SANS}font-size:0.59rem;color:{_GOLD};font-weight:500;margin:0 0 0.04rem 0;"
_S_YEAR   = f"{_MONO}font-size:0.50rem;color:#555960;margin:0;"
_S_ACK    = f"{_SANS}font-size:0.62rem;color:#8890a1;line-height:1.68;margin:0 0 0.28rem 0;"
_TAG_G    = f"display:inline-block;padding:0.15rem 0.55rem;{_SANS}font-size:0.52rem;font-weight:600;margin:0.08rem;background:rgba(207,185,145,0.08);color:{_GOLD};border:1px solid rgba(207,185,145,0.22);"
_TAG_N    = f"display:inline-block;padding:0.15rem 0.55rem;{_SANS}font-size:0.52rem;font-weight:600;margin:0.08rem;background:transparent;color:#8890a1;border:1px solid #1e1e1e;"
_S_NUM    = f"{_MONO}font-size:0.88rem;font-weight:700;color:{_GOLD};margin:0;line-height:1;"
_S_SLBL   = f"{_SANS}font-size:0.46rem;text-transform:uppercase;letter-spacing:0.14em;color:#555960;margin:2px 0 0 0;"

_SEC  = f"border-top:1px solid #1e1e1e;padding:0.80rem 0 0.3rem 0;margin-bottom:0.1rem;"
_EXPI = f"padding:0.45rem 0 0.45rem 0.70rem;border-left:1px solid #1e1e1e;margin-bottom:0.45rem;"
_EDUI = f"padding:0.40rem 0;border-bottom:1px solid #1a1a1a;"
_LINK = f"{_MONO}font-size:0.50rem;font-weight:700;color:{_GOLD};text-decoration:none;letter-spacing:0.10em;text-transform:uppercase;opacity:0.85;margin-right:1.2rem;"
_STAT_ROW = "display:flex;gap:0;margin-top:0.65rem;border-top:1px solid #1e1e1e;border-bottom:1px solid #1e1e1e;"
_STAT_ITM = "flex:1;text-align:center;padding:0.45rem 0.2rem;border-right:1px solid #1e1e1e;"


def _photo_html(filename: str, alt: str) -> str:
    p = _ASSETS / filename
    if not p.exists():
        return ""
    b64 = base64.b64encode(p.read_bytes()).decode()
    ext = p.suffix.lstrip(".")
    return (
        f"<div style='flex-shrink:0;width:148px;overflow:hidden;'>"
        f"<img src='data:image/{ext};base64,{b64}' alt='{alt}' "
        f"style='width:100%;height:100%;object-fit:cover;object-position:top center;"
        f"display:block;filter:grayscale(12%);' />"
        f"</div>"
    )


def page_about_jiahe() -> None:
    photo = _photo_html("photo_jiahe.jpeg", "Jiahe Miao")

    # ── hero banner ───────────────────────────────────────────────────────────
    st.markdown(
        f"<div style='background:#141414;border-top:2px solid {_GOLD};"
        f"border-bottom:1px solid #1e1e1e;overflow:hidden;margin-bottom:1.4rem;'>"
        f"<div style='display:flex;align-items:stretch;'>"
        f"{photo}"
        f"<div style='flex:1;padding:1.1rem 1.4rem;display:flex;flex-direction:column;"
        f"justify-content:center;border-left:1px solid #1e1e1e;'>"
        f"<div style='{_S_LABEL}margin:0 0 0.32rem 0;'>About the Author</div>"
        f"<div style='{_S_NAME}'>Jiahe Miao</div>"
        f"<div style='{_S_SUB}'>MSF Candidate &middot; Purdue Daniels School of Business "
        f"&middot; B.S. Information Systems, Kelley School of Business</div>"
        f"<div style='{_S_TAG}'>Capital markets and corporate finance professional with "
        f"internship experience across Chinese securities, banking, and U.S. fintech. "
        f"Brings a quantitative and data-driven lens to financial analysis and investor research.</div>"
        f"<div style='display:flex;flex-wrap:wrap;gap:0;'>"
        f"<a href='https://www.linkedin.com/in/jiahe-miao071/' target='_blank' style='{_LINK}'>LinkedIn</a>"
        f"</div>"
        f"</div></div></div>",
        unsafe_allow_html=True,
    )

    col_main, col_side = st.columns([1.55, 1], gap="large")

    with col_main:

        # Profile
        st.markdown(
            f"<div style='{_SEC}'>"
            f"<div style='{_S_LABEL}margin-bottom:0.55rem;'>Profile</div>"
            f"<div style='{_S_BODY}margin-bottom:0.45rem;'>Passionate about the intersection of quantitative finance and global capital markets. "
            f"With hands-on experience in securities research, banking operations, and venture-backed "
            f"pitch development, I enjoy building financial models that translate data into decisions.</div>"
            f"<div style='{_S_BODY}'>A background in information systems gives me an edge in structuring and automating "
            f"analytical workflows. Always eager to connect and exchange ideas on markets, modeling, and finance.</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # This Project
        st.markdown(
            f"<div style='{_SEC}'>"
            f"<div style='{_S_LABEL}margin-bottom:0.55rem;'>This Project</div>"
            f"<div style='{_S_BODY}margin-bottom:0.40rem;'>Quantitative Research contributor on the "
            f"Equity-Commodities Spillover Monitor - Course Project 3 for Prof. Cinder Zhang's MGMT 69000-120.</div>"
            f"<ul style='{_S_BODY}margin:0 0 0.5rem 0;padding-left:1.0rem;'>"
            f"<li style='margin-bottom:0.30rem;'>Designed the 4-state correlation regime taxonomy (Low / Moderate / Elevated / Crisis) "
            f"and calibrated the rolling window thresholds for regime classification</li>"
            f"<li style='margin-bottom:0.30rem;'>Specified the 6 regime-conditioned trade idea structures: entry/exit ranges, "
            f"correlation breakpoint triggers, and risk management rationale</li>"
            f"<li style='margin-bottom:0.30rem;'>Researched the Diebold-Yilmaz FEVD methodology and contributed to network "
            f"edge threshold calibration and directional spillover interpretation</li>"
            f"<li style='margin-bottom:0.30rem;'>Defined the Fixed Income cross-asset stress signal framework "
            f"(TLT / HYG / LQD / EMB metrics) and validated the private credit bubble proxy selection</li>"
            f"<li style='margin-bottom:0;'>Reviewed equity and commodity return series for data quality; "
            f"resolved TOPIX ETF proxy and Nickel ticker issues</li>"
            f"</ul>"
            f"<div style='{_STAT_ROW}'>"
            f"<div style='{_STAT_ITM}'><div style='{_S_NUM}'>4</div><div style='{_S_SLBL}'>Regime States</div></div>"
            f"<div style='{_STAT_ITM}'><div style='{_S_NUM}'>6</div><div style='{_S_SLBL}'>Trade Structures</div></div>"
            f"<div style='{_STAT_ITM}'><div style='{_S_NUM}'>32</div><div style='{_S_SLBL}'>Assets Covered</div></div>"
            f"<div style='{_STAT_ITM}border-right:none;'><div style='{_S_NUM}'>5</div><div style='{_S_SLBL}'>Analytical Layers</div></div>"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Experience
        def _exp(role, org, meta, desc=""):
            desc_html = f"<div style='{_S_DESC}margin-top:0.18rem;'>{desc}</div>" if desc else ""
            return (
                f"<div style='{_EXPI}'>"
                f"<div style='{_S_ROLE}'>{role}</div>"
                f"<div style='{_S_ORG}'>{org}</div>"
                f"<div style='{_S_META}'>{meta}</div>"
                f"{desc_html}"
                f"</div>"
            )

        st.markdown(
            f"<div style='{_SEC}'>"
            f"<div style='{_S_LABEL}margin-bottom:0.55rem;'>Experience</div>"
            + _exp("Investment Consultant (Practicum)", "Fino Advisors LLC",
                   "Jan 2026 &ndash; Present &middot; Remote",
                   "Conducted investor research to support clients' Angel and Pre-A fundraising, evaluating 40+ venture firms' "
                   "investment history, fund size, and sector focus. Developed 20-page pitch deck materials including financial "
                   "models forecasting future cash flows and market research on clean energy demand and market size.")
            + _exp("Corporate Finance Intern", "China Everbright Bank",
                   "Dec 2023 &ndash; Feb 2024 &middot; Zhengzhou, Henan, China",
                   "Analyzed financial products from 12 commercial banks, assessing risks and producing a research-based strategic "
                   "report. Strengthened client relationships by recommending financial products tailored to individual risk profiles. "
                   "Optimized the bank's cash management systems and products, improving efficiency and regulatory compliance.")
            + _exp("Integrated Finance Intern", "Shenwan Hongyuan Securities Co., Ltd.",
                   "Jun 2023 &ndash; Aug 2023 &middot; Zhengzhou, Henan, China",
                   "Performed in-depth analysis of bond issuance projects, identifying risk factors and ensuring regulatory compliance. "
                   "Supported IPO due diligence by analyzing financial statements, forecasting cash flows and EBITDA, and identifying "
                   "potential risks. Developed financial models to calculate total unbundling amounts for corporate equity incentive programs.")
            + "</div>",
            unsafe_allow_html=True,
        )

    with col_side:

        # Education
        st.markdown(
            f"<div style='{_SEC}'>"
            f"<div style='{_S_LABEL}margin-bottom:0.55rem;'>Education</div>"
            f"<div style='{_EDUI}'>"
            f"<div style='{_S_SCH}'>Purdue University</div>"
            f"<div style='{_S_DEPT}'>Mitchell E. Daniels, Jr. School of Business</div>"
            f"<div style='{_S_DEG}'>Master of Science in Finance</div>"
            f"<div style='{_S_YEAR}'>Jul 2025 &ndash; May 2026</div></div>"
            f"<div style='{_EDUI}border-bottom:none;'>"
            f"<div style='{_S_SCH}'>Indiana University</div>"
            f"<div style='{_S_DEPT}'>Kelley School of Business</div>"
            f"<div style='{_S_DEG}'>B.S. Information Systems</div>"
            f"<div style='{_S_YEAR}'>Aug 2021 &ndash; May 2025</div></div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Interests
        st.markdown(
            f"<div style='{_SEC}'>"
            f"<div style='{_S_LABEL}margin-bottom:0.45rem;'>Interests</div>"
            f"<div>"
            f"<span style='{_TAG_G}'>Capital Markets</span>"
            f"<span style='{_TAG_N}'>Corporate Finance</span>"
            f"<span style='{_TAG_G}'>Fixed Income</span>"
            f"<span style='{_TAG_N}'>Financial Modeling</span>"
            f"<span style='{_TAG_G}'>Venture Research</span>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

        # Acknowledgments
        st.markdown(
            f"<div style='{_SEC}'>"
            f"<div style='{_S_LABEL}margin-bottom:0.50rem;'>Acknowledgments</div>"
            f"<div style='{_S_ACK}margin-bottom:0;'><strong style='color:#c8c8c8;font-weight:600;'>Prof. Cinder Zhang</strong>, "
            f"MGMT 69000: Regime-based thinking and equity-commodities spillover framework design</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    _page_footer()
