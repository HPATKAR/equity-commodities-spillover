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

_S_LABEL = f"{_MONO}font-size:0.44rem;font-weight:700;text-transform:uppercase;letter-spacing:0.18em;color:#555960;"
_S_NAME  = f"{_SANS}font-size:0.72rem;font-weight:700;color:#e8e9ed;letter-spacing:-0.01em;line-height:1.15;margin:0 0 0.14rem 0;"
_S_SUB   = f"{_SANS}font-size:0.56rem;color:{_GOLD};font-weight:500;margin:0 0 0.26rem 0;line-height:1.5;"
_S_TGLN  = f"{_SANS}font-size:0.56rem;color:#8890a1;line-height:1.70;margin:0 0 0.50rem 0;"
_S_BODY  = f"{_SANS}font-size:0.56rem;color:#8890a1;line-height:1.72;"
_S_ROLE  = f"{_SANS}font-size:0.56rem;font-weight:700;color:#e8e9ed;margin:0 0 0.04rem 0;line-height:1.3;"
_S_ORG   = f"{_SANS}font-size:0.54rem;font-weight:600;color:{_GOLD};margin:0 0 0.04rem 0;"
_S_META  = f"{_MONO}font-size:0.44rem;color:#555960;margin:0 0 0.14rem 0;"
_S_DESC  = f"{_SANS}font-size:0.56rem;color:#8890a1;line-height:1.68;margin:0;"
_S_SCH   = f"{_SANS}font-size:0.58rem;font-weight:700;color:#e8e9ed;margin:0 0 0.04rem 0;"
_S_DEPT  = f"{_SANS}font-size:0.52rem;color:#8890a1;margin:0 0 0.03rem 0;"
_S_DEG   = f"{_SANS}font-size:0.54rem;color:{_GOLD};font-weight:500;margin:0 0 0.03rem 0;"
_S_YEAR  = f"{_MONO}font-size:0.44rem;color:#555960;margin:0;"
_S_ACK   = f"{_SANS}font-size:0.56rem;color:#8890a1;line-height:1.68;margin:0 0 0.26rem 0;"
_TAG_G   = f"display:inline-block;padding:0.12rem 0.50rem;{_SANS}font-size:0.48rem;font-weight:600;margin:0.07rem;background:rgba(207,185,145,0.08);color:{_GOLD};border:1px solid rgba(207,185,145,0.22);"
_TAG_N   = f"display:inline-block;padding:0.12rem 0.50rem;{_SANS}font-size:0.48rem;font-weight:600;margin:0.07rem;background:transparent;color:#8890a1;border:1px solid #1e1e1e;"
_S_NUM   = f"{_MONO}font-size:0.80rem;font-weight:700;color:{_GOLD};margin:0;line-height:1;"
_S_SLBL  = f"{_SANS}font-size:0.42rem;text-transform:uppercase;letter-spacing:0.14em;color:#555960;margin:2px 0 0 0;"

_SEC      = "border-top:1px solid #1e1e1e;padding:0.75rem 0 0.25rem 0;margin-bottom:0.1rem;"
_EXPI     = "padding:0.40rem 0 0.40rem 0.65rem;border-left:1px solid #1e1e1e;margin-bottom:0.40rem;"
_EDUI     = "padding:0.38rem 0;border-bottom:1px solid #1a1a1a;"
_LINK     = f"{_MONO}font-size:0.44rem;font-weight:700;color:{_GOLD};text-decoration:none;letter-spacing:0.10em;text-transform:uppercase;opacity:0.85;margin-right:1.2rem;"
_STAT_ROW = "display:flex;gap:0;margin-top:0.60rem;border-top:1px solid #1e1e1e;border-bottom:1px solid #1e1e1e;"
_STAT_ITM = "flex:1;text-align:center;padding:0.40rem 0.15rem;border-right:1px solid #1e1e1e;"


def _photo_html(filename: str, alt: str) -> str:
    p = _ASSETS / filename
    if not p.exists():
        return ""
    b64 = base64.b64encode(p.read_bytes()).decode()
    ext = p.suffix.lstrip(".")
    return (
        f"<div style='flex-shrink:0;width:140px;overflow:hidden;'>"
        f"<img src='data:image/{ext};base64,{b64}' alt='{alt}' "
        f"style='width:100%;height:100%;object-fit:cover;object-position:top center;"
        f"display:block;filter:grayscale(10%);' />"
        f"</div>"
    )


def _exp(role, org, meta, bullets=None):
    bullets_html = ""
    if bullets:
        items = "".join(f"<li style='margin-bottom:0.22rem;'>{b}</li>" for b in bullets)
        bullets_html = f"<ul style='{_S_DESC}margin:0.14rem 0 0 0;padding-left:0.9rem;'>{items}</ul>"
    return (
        f"<div style='{_EXPI}'>"
        f"<div style='{_S_ROLE}'>{role}</div>"
        f"<div style='{_S_ORG}'>{org}</div>"
        f"<div style='{_S_META}'>{meta}</div>"
        f"{bullets_html}"
        f"</div>"
    )


def page_about_ilian() -> None:
    photo = _photo_html("photo_ilian.jpeg", "Ilian Zalomai")

    st.markdown(
        f"<div style='background:#141414;border-top:2px solid {_GOLD};"
        f"border-bottom:1px solid #1e1e1e;overflow:hidden;margin-bottom:1.2rem;'>"
        f"<div style='display:flex;align-items:stretch;'>"
        f"{photo}"
        f"<div style='flex:1;padding:1.0rem 1.3rem;display:flex;flex-direction:column;"
        f"justify-content:center;border-left:1px solid #1e1e1e;'>"
        f"<div style='{_S_LABEL}margin:0 0 0.28rem 0;'>About the Author</div>"
        f"<div style='{_S_NAME}'>Ilian Zalomai</div>"
        f"<div style='{_S_SUB}'>MSF Candidate &middot; Purdue Daniels School of Business "
        f"&middot; Payment Systems &amp; Fraud | Former Deloitte</div>"
        f"<div style='{_S_TGLN}'>Fintech and banking professional with 4+ years leading payment systems "
        f"and fraud operations at scale across high-volume travel platforms, combined with strategy and "
        f"transformation consulting at Deloitte across Frankfurt and Leipzig. "
        f"Bridging operational finance, risk management, and quantitative analytics.</div>"
        f"<div style='display:flex;flex-wrap:wrap;gap:0;'>"
        f"<a href='https://www.linkedin.com/in/ilian-zalomai-55iz/' target='_blank' style='{_LINK}'>LinkedIn</a>"
        f"</div>"
        f"</div></div></div>",
        unsafe_allow_html=True,
    )

    col_main, col_side = st.columns([1.55, 1], gap="large")

    with col_main:

        st.markdown(
            f"<div style='{_SEC}'>"
            f"<div style='{_S_LABEL}margin-bottom:0.50rem;'>Profile</div>"
            f"<div style='{_S_BODY}margin-bottom:0.38rem;'>Experienced in payment systems, fraud prevention, and banking "
            f"technology consulting. Four years building and managing risk and security operations across high-volume travel "
            f"fintech platforms - implementing chargeback frameworks, fraud monitoring, and customer verification systems "
            f"at scale.</div>"
            f"<div style='{_S_BODY}'>The MSF at Purdue's Financial Analytics track is deepening my quantitative foundation "
            f"in derivatives and risk modeling, bringing structured rigor to the intersection of risk management, "
            f"data analytics, and financial markets.</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            f"<div style='{_SEC}'>"
            f"<div style='{_S_LABEL}margin-bottom:0.50rem;'>This Project</div>"
            f"<div style='{_S_BODY}margin-bottom:0.35rem;'>Geopolitical Research and Scenario Design contributor on the "
            f"Equity-Commodities Spillover Monitor - Course Project 3 for Prof. Cinder Zhang's MGMT 69000-120.</div>"
            f"<ul style='{_S_BODY}margin:0 0 0.45rem 0;padding-left:0.9rem;'>"
            f"<li style='margin-bottom:0.22rem;'>Designed the War Impact Map scoring framework: per-war baseline scores, "
            f"country-level commodity exposure weights, and concurrent-war amplifier methodology for Ukraine, Gaza/Red Sea, "
            f"and Iran/Hormuz conflicts</li>"
            f"<li style='margin-bottom:0.22rem;'>Authored all 13 geopolitical event descriptions in the dashboard "
            f"(2008-2025: GFC, Arab Spring, US-China Trade War, Aramco Attack, COVID, WTI Negative, Ukraine War, "
            f"LME Nickel Squeeze, Fed Hiking Cycle, SVB Crisis, Israel-Hamas, India-Pakistan, Iran/Hormuz)</li>"
            f"<li style='margin-bottom:0.22rem;'>Built the Strait Watch framework: five critical maritime chokepoints "
            f"(Hormuz, Malacca, Suez, Bab-el-Mandeb, Turkish Straits), EIA/IEA throughput data, and disruption scoring</li>"
            f"<li style='margin-bottom:0.22rem;'>Researched the Iran/Hormuz escalation scenario: oil and LNG regional "
            f"transit exposure, energy-importing Asia risk analysis, and Brent/WTI spread interpretation</li>"
            f"<li style='margin-bottom:0;'>Authored narrative content across the dashboard - page intros, takeaway blocks, "
            f"and section conclusions connecting quantitative outputs to real-world market implications</li>"
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

        st.markdown(
            f"<div style='{_SEC}'>"
            f"<div style='{_S_LABEL}margin-bottom:0.50rem;'>Professional Experience</div>"
            + _exp(
                "Payment Systems, Fraud &amp; Security Supervisor",
                "Firebird Tours (Rail.Ninja &middot; Firebirdtours &middot; Triptile)",
                "January 2024 &ndash; Present &middot; Full-time",
                [
                    "Managing payment systems, risk, and security operations across high-volume travel platforms "
                    "(Rail.Ninja, Firebirdtours, Triptile), overseeing fraud monitoring and customer verification.",
                    "Implemented outbound-chargeback processes and increased chargeback win rate while reducing average "
                    "document preparation time by 70% through systematic workflow redesign.",
                    "Building analytical reports and models to identify fraud patterns, optimize anti-fraud rules, "
                    "and support data-driven risk decisions at scale.",
                ],
            )
            + _exp(
                "Banking Technology Strategy &amp; Transformation Analyst",
                "Deloitte",
                "April 2024 &ndash; August 2024 &middot; Frankfurt, Germany",
                [
                    "Supported Privileged Access Management (PAM) assessment implementation for a major banking client.",
                    "Conducted comprehensive research on compliance trends across internal and external regulatory frameworks; "
                    "prepared management sales deck materials and coordinated across initiative workstreams.",
                ],
            )
            + _exp(
                "Banking Operations Analyst",
                "Deloitte",
                "April 2023 &ndash; July 2023 &middot; Leipzig, Germany",
                [
                    "Supported banking transformation initiatives across Credit, Cloud, and Web 3.0/Metaverse workstreams.",
                    "Researched market trends including platformification, cloud hyperscaler sustainability, and metaverse "
                    "applications in banking; prepared and presented management decks across workstreams.",
                ],
            )
            + _exp(
                "Payment Systems, Fraud &amp; Security Team Lead",
                "Firebird Tours",
                "March 2022 &ndash; December 2023 &middot; 1 yr 10 mos",
                [
                    "Led the payment systems and fraud team, building fraud detection infrastructure and managing "
                    "chargeback resolution across multiple high-volume travel booking platforms.",
                ],
            )
            + _exp(
                "Credit Department Analyst",
                "Volksbanken Raiffeisenbanken &middot; Part-time",
                "May 2022 &ndash; July 2022 &middot; Mittweida, Germany",
                [
                    "Collected, verified, and organized credit documentation; supported research on cost-of-living "
                    "trends and banking approach plausibility assessments.",
                ],
            )
            + _exp(
                "MUN Security Council Chairman &amp; Organising Committee",
                "United Nations in Belarus (FIRMUN / OctoMUN)",
                "October 2018 &ndash; March 2022 &middot; Minsk, Belarus",
                [
                    "Served as Security Council Chairman and President of General Assembly across multiple Model UN "
                    "conferences; co-chaired EcoFin committee.",
                    "Led the organizing committee for FIRMUN and OctoMUN editions, coordinating delegate logistics, "
                    "agenda setting, and committee operations.",
                ],
            )
            + "</div>",
            unsafe_allow_html=True,
        )

    with col_side:

        st.markdown(
            f"<div style='{_SEC}'>"
            f"<div style='{_S_LABEL}margin-bottom:0.50rem;'>Education</div>"
            f"<div style='{_EDUI}'>"
            f"<div style='{_S_SCH}'>Purdue University</div>"
            f"<div style='{_S_DEPT}'>Mitchell E. Daniels, Jr. School of Business</div>"
            f"<div style='{_S_DEG}'>M.S. Finance &middot; Financial Analytics Track</div>"
            f"<div style='{_S_YEAR}'>July 2025 &ndash; May 2026 &middot; West Lafayette, IN</div></div>"
            f"<div style='{_EDUI}border-bottom:none;'>"
            f"<div style='{_S_SCH}'>Hochschule Mittweida</div>"
            f"<div style='{_S_DEPT}'>Germany</div>"
            f"<div style='{_S_DEG}'>B.A. Business Administration</div>"
            f"<div style='{_S_YEAR}'>March 2022 &ndash; April 2024</div></div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            f"<div style='{_SEC}'>"
            f"<div style='{_S_LABEL}margin-bottom:0.42rem;'>Interests</div>"
            f"<div>"
            f"<span style='{_TAG_G}'>Fintech &amp; Risk</span>"
            f"<span style='{_TAG_N}'>Payment Systems</span>"
            f"<span style='{_TAG_G}'>Financial Analytics</span>"
            f"<span style='{_TAG_N}'>Fraud Prevention</span>"
            f"<span style='{_TAG_G}'>Banking Strategy</span>"
            f"<span style='{_TAG_N}'>Derivatives &amp; Risk</span>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            f"<div style='{_SEC}'>"
            f"<div style='{_S_LABEL}margin-bottom:0.46rem;'>Acknowledgments</div>"
            f"<div style='{_S_ACK}margin-bottom:0;'>"
            f"<strong style='color:#c8c8c8;font-weight:600;'>Prof. Cinder Zhang</strong> "
            f"&middot; MGMT 69000: Connecting geopolitical events to systematic regime-based market analysis</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    _page_footer()
