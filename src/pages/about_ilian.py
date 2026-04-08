"""About: Ilian Zalomai."""

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


def page_about_ilian() -> None:
    st.markdown(_STYLE, unsafe_allow_html=True)

    photo = _photo_html("photo_ilian.jpeg", "Ilian Zalomai")

    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown(
        f"<div style='background:#141414;border-top:2px solid {_GOLD};"
        f"border-bottom:1px solid #1e1e1e;overflow:hidden;margin-bottom:1.2rem;'>"
        f"<div style='display:flex;align-items:stretch;'>"
        f"{photo}"
        f"<div style='flex:1;padding:0.9rem 1.2rem;display:flex;flex-direction:column;"
        f"justify-content:center;border-left:1px solid #1e1e1e;'>"
        f"<span class='abt-label' style='margin:0 0 6px 0;'>About the Author</span>"
        f"<span class='abt-name' style='margin:0 0 4px 0;'>Ilian Zalomai</span>"
        f"<span class='abt-sub' style='margin:0 0 6px 0;'>MSF Candidate &middot; Purdue Daniels School of Business "
        f"&middot; Payment Systems &amp; Fraud | Former Deloitte</span>"
        f"<span class='abt-tgln' style='margin:0 0 8px 0;'>Fintech and banking professional with 4+ years leading payment systems "
        f"and fraud operations at scale across high-volume travel platforms, combined with strategy and "
        f"transformation consulting at Deloitte across Frankfurt and Leipzig. "
        f"Bridging operational finance, risk management, and quantitative analytics.</span>"
        f"<div style='display:flex;flex-wrap:wrap;gap:0;'>"
        f"<a href='https://www.linkedin.com/in/ilian-zalomai-55iz/' target='_blank' class='abt-link'>LinkedIn</a>"
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
            f"<span class='abt-body' style='display:block;margin-bottom:6px;'>Experienced in payment systems, fraud prevention, and banking "
            f"technology consulting. Four years building and managing risk and security operations across high-volume travel "
            f"fintech platforms &mdash; implementing chargeback frameworks, fraud monitoring, and customer verification systems "
            f"at scale.</span>"
            f"<span class='abt-body' style='display:block;'>The MSF at Purdue's Financial Analytics track is deepening my quantitative foundation "
            f"in derivatives and risk modeling, bringing structured rigor to the intersection of risk management, "
            f"data analytics, and financial markets.</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── This Project ──────────────────────────────────────────────────────
        st.markdown(
            f"<div style='{_SEC}'>"
            f"<span class='abt-label' style='margin-bottom:8px;'>This Project</span>"
            f"<span class='abt-body' style='display:block;margin-bottom:6px;'>Geopolitical Research and Scenario Design contributor on the "
            f"Equity-Commodities Spillover Monitor &mdash; Course Project 3 for Prof. Cinder Zhang's MGMT 69000-120.</span>"
            f"<ul style='margin:0 0 0 0;padding-left:14px;list-style:disc;'>"
            f"<li style='margin-bottom:3px;'><span class='abt-body'>Designed the War Impact Map scoring framework: per-war baseline scores, "
            f"country-level commodity exposure weights, and concurrent-war amplifier methodology for Ukraine, Gaza/Red Sea, "
            f"and Iran/Hormuz conflicts</span></li>"
            f"<li style='margin-bottom:3px;'><span class='abt-body'>Authored all 13 geopolitical event descriptions in the dashboard "
            f"(2008&ndash;2025: GFC, Arab Spring, US-China Trade War, Aramco Attack, COVID, WTI Negative, Ukraine War, "
            f"LME Nickel Squeeze, Fed Hiking Cycle, SVB Crisis, Israel-Hamas, India-Pakistan, Iran/Hormuz)</span></li>"
            f"<li style='margin-bottom:3px;'><span class='abt-body'>Built the Strait Watch framework: five critical maritime chokepoints "
            f"(Hormuz, Malacca, Suez, Bab-el-Mandeb, Turkish Straits), EIA/IEA throughput data, and disruption scoring</span></li>"
            f"<li style='margin-bottom:3px;'><span class='abt-body'>Researched the Iran/Hormuz escalation scenario: oil and LNG regional "
            f"transit exposure, energy-importing Asia risk analysis, and Brent/WTI spread interpretation</span></li>"
            f"<li style='margin-bottom:0;'><span class='abt-body'>Authored narrative content across the dashboard &mdash; page intros, takeaway blocks, "
            f"and section conclusions connecting quantitative outputs to real-world market implications</span></li>"
            f"</ul>"
            f"<div style='{_SROW}'>"
            f"<div style='{_SITM}'><span class='abt-num'>13</span><span class='abt-slbl'>Events Catalogued</span></div>"
            f"<div style='{_SITM}'><span class='abt-num'>5</span><span class='abt-slbl'>Chokepoints Mapped</span></div>"
            f"<div style='{_SITM}'><span class='abt-num'>3</span><span class='abt-slbl'>Active Conflicts</span></div>"
            f"<div style='{_SITM}border-right:none;'><span class='abt-num'>40+</span><span class='abt-slbl'>Countries Scored</span></div>"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── Experience ────────────────────────────────────────────────────────
        st.markdown(
            f"<div style='{_SEC}'>"
            f"<span class='abt-label' style='margin-bottom:8px;'>Professional Experience</span>"
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

        # ── Education ─────────────────────────────────────────────────────────
        st.markdown(
            f"<div style='{_SEC}'>"
            f"<span class='abt-label' style='margin-bottom:8px;'>Education</span>"
            f"<div style='{_EDUI}'>"
            f"<span class='abt-sch' style='margin:0 0 2px 0;'>Purdue University</span>"
            f"<span class='abt-dept' style='margin:0 0 2px 0;'>Mitchell E. Daniels, Jr. School of Business</span>"
            f"<span class='abt-deg' style='margin:0 0 2px 0;'>M.S. Finance &middot; Financial Analytics Track</span>"
            f"<span class='abt-year'>July 2025 &ndash; May 2026 &middot; West Lafayette, IN</span></div>"
            f"<div style='{_EDUI}border-bottom:none;'>"
            f"<span class='abt-sch' style='margin:0 0 2px 0;'>Hochschule Mittweida</span>"
            f"<span class='abt-dept' style='margin:0 0 2px 0;'>Germany</span>"
            f"<span class='abt-deg' style='margin:0 0 2px 0;'>B.A. Business Administration</span>"
            f"<span class='abt-year'>March 2022 &ndash; April 2024</span></div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── Interests ─────────────────────────────────────────────────────────
        st.markdown(
            f"<div style='{_SEC}'>"
            f"<span class='abt-label' style='margin-bottom:6px;'>Interests</span>"
            f"<div>"
            f"<span class='abt-tag-g'>Fintech &amp; Risk</span>"
            f"<span class='abt-tag-n'>Payment Systems</span>"
            f"<span class='abt-tag-g'>Financial Analytics</span>"
            f"<span class='abt-tag-n'>Fraud Prevention</span>"
            f"<span class='abt-tag-g'>Banking Strategy</span>"
            f"<span class='abt-tag-n'>Derivatives &amp; Risk</span>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

        # ── Acknowledgments ───────────────────────────────────────────────────
        st.markdown(
            f"<div style='{_SEC}'>"
            f"<span class='abt-label' style='margin-bottom:8px;'>Acknowledgments</span>"
            f"<div class='abt-ack' style='margin-bottom:0;'>"
            f"<strong style='color:#c8c8c8;font-weight:600;'>Prof. Cinder Zhang</strong> "
            f"&middot; MGMT 69000: Connecting geopolitical events to systematic regime-based market analysis</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    _page_footer()
