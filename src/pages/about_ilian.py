"""About: Ilian Zalomai."""

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


def page_about_ilian() -> None:
    photo = _photo_html("photo_ilian.jpeg", "Ilian Zalomai")

    # ── hero banner ───────────────────────────────────────────────────────────
    st.markdown(
        f"<div style='background:#141414;border-top:2px solid {_GOLD};"
        f"border-bottom:1px solid #1e1e1e;overflow:hidden;margin-bottom:1.4rem;'>"
        f"<div style='display:flex;align-items:stretch;'>"
        f"{photo}"
        f"<div style='flex:1;padding:1.1rem 1.4rem;display:flex;flex-direction:column;"
        f"justify-content:center;border-left:1px solid #1e1e1e;'>"
        f"<div style='{_S_LABEL}margin:0 0 0.32rem 0;'>About the Author</div>"
        f"<div style='{_S_NAME}'>Ilian Zalomai</div>"
        f"<div style='{_S_SUB}'>MSF Candidate &middot; Purdue Daniels School of Business "
        f"&middot; Payment Systems &amp; Fraud | Former Deloitte</div>"
        f"<div style='{_S_TAG}'>Fintech and banking professional with 4+ years leading payment "
        f"systems and fraud operations at scale, combined with consulting experience at Deloitte "
        f"across Frankfurt and Leipzig. Bridging operational finance and quantitative analytics.</div>"
        f"<div style='display:flex;flex-wrap:wrap;gap:0;'>"
        f"<a href='https://www.linkedin.com/in/ilian-zalomai-55iz/' target='_blank' style='{_LINK}'>LinkedIn</a>"
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
            f"<div style='{_S_BODY}margin-bottom:0.45rem;'>Experienced in payment systems, fraud prevention, and banking technology consulting. "
            f"I have spent four years building and managing risk and security operations across "
            f"high-volume travel fintech platforms, and have contributed to banking transformation "
            f"initiatives at Deloitte across Germany.</div>"
            f"<div style='{_S_BODY}'>The MSF at Purdue's Financial Analytics track is deepening my foundation in "
            f"quantitative methods and derivatives - bringing rigor to the intersection of "
            f"risk management, data analytics, and financial markets.</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # This Project
        st.markdown(
            f"<div style='{_SEC}'>"
            f"<div style='{_S_LABEL}margin-bottom:0.55rem;'>This Project</div>"
            f"<div style='{_S_BODY}margin-bottom:0.40rem;'>Geopolitical Research and Scenario Design contributor on the "
            f"Equity-Commodities Spillover Monitor - Course Project 3 for Prof. Cinder Zhang's MGMT 69000-120.</div>"
            f"<ul style='{_S_BODY}margin:0 0 0.5rem 0;padding-left:1.0rem;'>"
            f"<li style='margin-bottom:0.30rem;'>Designed the War Impact Map scoring framework: per-war baseline scores, "
            f"country-level commodity exposure weights, and concurrent-war amplifier for Ukraine, Gaza/Red Sea, and Iran/Hormuz</li>"
            f"<li style='margin-bottom:0.30rem;'>Authored all geopolitical event descriptions in the dashboard "
            f"(13 events, 2008-2025) connecting conflict timelines to commodity and equity market impacts</li>"
            f"<li style='margin-bottom:0.30rem;'>Built the Strait Watch framework: five critical maritime chokepoints, "
            f"EIA/IEA throughput data, and disruption scoring methodology</li>"
            f"<li style='margin-bottom:0.30rem;'>Researched the Iran/Hormuz escalation: oil and LNG regional exposure, "
            f"energy-importing Asia risk, and Brent/WTI spread dynamics</li>"
            f"<li style='margin-bottom:0;'>Authored narrative content across the dashboard (page intros, takeaway blocks, "
            f"section conclusions) connecting quantitative outputs to real-world market context</li>"
            f"</ul>"
            f"<div style='{_STAT_ROW}'>"
            f"<div style='{_STAT_ITM}'><div style='{_S_NUM}'>13</div><div style='{_S_SLBL}'>Events Catalogued</div></div>"
            f"<div style='{_STAT_ITM}'><div style='{_S_NUM}'>5</div><div style='{_S_SLBL}'>Chokepoints Mapped</div></div>"
            f"<div style='{_STAT_ITM}'><div style='{_S_NUM}'>3</div><div style='{_S_SLBL}'>Active Conflicts</div></div>"
            f"<div style='{_STAT_ITM}border-right:none;'><div style='{_S_NUM}'>40+</div><div style='{_S_SLBL}'>Countries Scored</div></div>"
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
            + _exp("Payment Systems, Fraud &amp; Security Supervisor",
                   "Firebird Tours (Rail.Ninja &middot; Firebirdtours &middot; Triptile)",
                   "Jan 2024 &ndash; Present &middot; Full-time",
                   "Managing payment systems, risk, and security across high-volume travel platforms. "
                   "Overseeing fraud monitoring, optimizing anti-fraud rules and customer verification, and building "
                   "analytical reports and models. Implemented outbound-chargeback processes and increased chargeback "
                   "win rate while reducing average document preparation time by 70%.")
            + _exp("Banking Technology Strategy &amp; Transformation Analyst", "Deloitte",
                   "Apr 2024 &ndash; Aug 2024 &middot; Frankfurt, Germany",
                   "Supported Privileged Access Management assessment implementation. Conducted comprehensive research "
                   "on compliance trends with internal and external regulators. Prepared management sales deck materials "
                   "and participated in team coordination across initiative workstreams.")
            + _exp("Banking Operations Analyst", "Deloitte",
                   "Apr 2023 &ndash; Jul 2023 &middot; Leipzig, Germany",
                   "Supported banking initiatives in Credit, Cloud, and Web 3.0/Metaverse. Researched current market trends "
                   "including platformification, cloud hyperscaler sustainability, and metaverse applications in banking.")
            + _exp("Payment Systems, Fraud &amp; Security Team Lead", "Firebird Tours",
                   "Mar 2022 &ndash; Dec 2023 &middot; 1 yr 10 mos")
            + _exp("Credit Department Analyst",
                   "Volksbanken Raiffeisenbanken &middot; Part-time",
                   "May 2022 &ndash; Jul 2022 &middot; Mittweida, Germany",
                   "Collected, verified, and organized credit documentation. Supported research on cost-of-living trends "
                   "and banking approach plausibility assessments.")
            + _exp("MUN Security Council Chairman &amp; Organising Committee",
                   "United Nations in Belarus (FIRMUN / OctoMUN)",
                   "Oct 2018 &ndash; Mar 2022 &middot; Minsk, Belarus",
                   "Served as Security Council Chairman and President of General Assembly across multiple Model UN conferences. "
                   "Co-chaired EcoFin committee and led the organizing committee for FIRMUN and OctoMUN editions.")
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
            f"<div style='{_S_DEG}'>M.S. Finance &middot; Financial Analytics Track</div>"
            f"<div style='{_S_YEAR}'>Jul 2025 &ndash; May 2026</div></div>"
            f"<div style='{_EDUI}border-bottom:none;'>"
            f"<div style='{_S_SCH}'>Hochschule Mittweida</div>"
            f"<div style='{_S_DEPT}'>Germany</div>"
            f"<div style='{_S_DEG}'>B.A. Business Administration</div>"
            f"<div style='{_S_YEAR}'>Mar 2022 &ndash; Apr 2024</div></div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Interests
        st.markdown(
            f"<div style='{_SEC}'>"
            f"<div style='{_S_LABEL}margin-bottom:0.45rem;'>Interests</div>"
            f"<div>"
            f"<span style='{_TAG_G}'>Fintech &amp; Risk</span>"
            f"<span style='{_TAG_N}'>Payment Systems</span>"
            f"<span style='{_TAG_G}'>Financial Analytics</span>"
            f"<span style='{_TAG_N}'>Fraud Prevention</span>"
            f"<span style='{_TAG_G}'>Banking Strategy</span>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

        # Acknowledgments
        st.markdown(
            f"<div style='{_SEC}'>"
            f"<div style='{_S_LABEL}margin-bottom:0.50rem;'>Acknowledgments</div>"
            f"<div style='{_S_ACK}margin-bottom:0;'><strong style='color:#c8c8c8;font-weight:600;'>Prof. Cinder Zhang</strong>, "
            f"MGMT 69000: Connecting geopolitical events to systematic regime-based market analysis</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    _page_footer()
