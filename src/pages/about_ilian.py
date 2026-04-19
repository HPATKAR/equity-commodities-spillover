"""About: Ilian Zalomai."""

from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st

from src.ui.shared import _page_footer

_ASSETS = Path(__file__).resolve().parent.parent.parent / "assets"
_G   = "#CFB991"
_DIM = "#8E9AAA"
_M   = "font-family:'JetBrains Mono',monospace;"
_S   = "font-family:'DM Sans',sans-serif;"

_STYLE = """<style>
.ail-label{
    font-family:'JetBrains Mono',monospace!important;
    font-size:8px!important;font-weight:700!important;
    text-transform:uppercase;letter-spacing:.20em;
    color:#CFB991!important;display:block;
    border-bottom:1px solid #1e1e1e;
    padding-bottom:5px;margin-bottom:10px;
}
.ail-name{
    font-family:'JetBrains Mono',monospace!important;
    font-size:20px!important;font-weight:700!important;
    color:#e8e9ed!important;letter-spacing:-.02em;
    line-height:1.1;display:block;
}
.ail-sub{
    font-family:'JetBrains Mono',monospace!important;
    font-size:8.5px!important;color:#CFB991!important;
    font-weight:600;line-height:1.7;
    letter-spacing:.08em;display:block;
}
.ail-tgln{
    font-family:'DM Sans',sans-serif!important;
    font-size:11px!important;color:#9aa3b0!important;
    line-height:1.72;display:block;
}
.ail-body{
    font-family:'DM Sans',sans-serif!important;
    font-size:11px!important;color:#9aa3b0!important;
    line-height:1.78;
}
.ail-role{
    font-family:'DM Sans',sans-serif!important;
    font-size:11.5px!important;font-weight:700!important;
    color:#e8e9ed!important;line-height:1.3;display:block;
}
.ail-org{
    font-family:'JetBrains Mono',monospace!important;
    font-size:9px!important;font-weight:600!important;
    color:#CFB991!important;display:block;letter-spacing:.04em;
}
.ail-meta{
    font-family:'JetBrains Mono',monospace!important;
    font-size:8.5px!important;color:#8E9AAA!important;display:block;
}
.ail-sch{
    font-family:'DM Sans',sans-serif!important;
    font-size:12px!important;font-weight:700!important;
    color:#e8e9ed!important;display:block;
}
.ail-dept{
    font-family:'DM Sans',sans-serif!important;
    font-size:10px!important;color:#9aa3b0!important;display:block;
}
.ail-deg{
    font-family:'JetBrains Mono',monospace!important;
    font-size:8.5px!important;color:#CFB991!important;
    font-weight:600;letter-spacing:.04em;display:block;
}
.ail-year{
    font-family:'JetBrains Mono',monospace!important;
    font-size:8px!important;color:#8E9AAA!important;display:block;
}
.ail-link{
    font-family:'JetBrains Mono',monospace!important;
    font-size:8px!important;font-weight:700!important;
    color:#080808!important;text-decoration:none!important;
    letter-spacing:.14em;text-transform:uppercase;
    background:#CFB991;padding:3px 10px;
    margin-right:6px;display:inline-block;
    transition:opacity .15s;
}
.ail-link:hover{opacity:.85!important;}
.ail-link-ghost{
    font-family:'JetBrains Mono',monospace!important;
    font-size:8px!important;font-weight:700!important;
    color:#CFB991!important;text-decoration:none!important;
    letter-spacing:.14em;text-transform:uppercase;
    border:1px solid rgba(207,185,145,.35);
    padding:2px 9px;margin-right:6px;display:inline-block;
}
.ail-link-ghost:hover{border-color:#CFB991!important;}
.ail-num{
    font-family:'JetBrains Mono',monospace!important;
    font-size:22px!important;font-weight:700!important;
    color:#CFB991!important;line-height:1;display:block;
}
.ail-slbl{
    font-family:'JetBrains Mono',monospace!important;
    font-size:7px!important;text-transform:uppercase;
    letter-spacing:.14em;color:#8E9AAA!important;display:block;
    margin-top:4px;
}
.ail-tag-g{
    display:inline-block;padding:2px 9px;
    font-family:'JetBrains Mono',monospace!important;
    font-size:8px!important;font-weight:700!important;
    margin:2px;background:rgba(207,185,145,.08);
    color:#CFB991!important;
    border:1px solid rgba(207,185,145,.25);
    letter-spacing:.06em;text-transform:uppercase;
}
.ail-tag-n{
    display:inline-block;padding:2px 9px;
    font-family:'JetBrains Mono',monospace!important;
    font-size:8px!important;font-weight:600!important;
    margin:2px;background:transparent;
    color:#8E9AAA!important;border:1px solid #222;
    letter-spacing:.06em;text-transform:uppercase;
}
</style>"""

_SEC  = "border-top:1px solid #1e1e1e;padding:0.85rem 0 0.35rem;margin-bottom:0.15rem;"
_EXPI = ("padding:0.55rem 0 0.55rem 0.9rem;"
         "border-left:2px solid rgba(207,185,145,.30);"
         "margin-bottom:0.5rem;")
_EDUI = "padding:0.45rem 0;border-bottom:1px solid #161616;"
_SROW = ("display:flex;gap:0;margin-top:0.75rem;"
         "border:1px solid #1e1e1e;")
_SITM = "flex:1;text-align:center;padding:0.65rem 0.25rem;border-right:1px solid #1e1e1e;"


def _photo_html(filename: str, alt: str) -> str:
    p = _ASSETS / filename
    if not p.exists():
        return ""
    b64 = base64.b64encode(p.read_bytes()).decode()
    ext = p.suffix.lstrip(".")
    return (
        f"<div style='flex-shrink:0;width:150px;overflow:hidden;"
        f"border-right:1px solid #1e1e1e;'>"
        f"<img src='data:image/{ext};base64,{b64}' alt='{alt}' "
        f"style='width:100%;height:100%;object-fit:cover;"
        f"object-position:top center;display:block;filter:grayscale(15%);' />"
        f"</div>"
    )


def _exp(role, org, meta, bullets=None):
    bullets_html = ""
    if bullets:
        items = "".join(
            f"<li style='margin-bottom:4px;'>"
            f"<span class='ail-body'>{b}</span></li>"
            for b in bullets
        )
        bullets_html = (
            f"<ul style='margin:6px 0 0;padding-left:14px;list-style:disc;'>"
            f"{items}</ul>"
        )
    return (
        f"<div style='{_EXPI}'>"
        f"<span class='ail-role' style='margin:0 0 3px;'>{role}</span>"
        f"<span class='ail-org' style='margin:0 0 2px;'>{org}</span>"
        f"<span class='ail-meta' style='margin:0 0 {6 if bullets else 0}px;'>{meta}</span>"
        f"{bullets_html}"
        f"</div>"
    )


def page_about_ilian() -> None:
    st.markdown(_STYLE, unsafe_allow_html=True)

    photo = _photo_html("photo_ilian.jpeg", "Ilian Zalomai")

    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown(
        f"<div style='background:#080808;border-top:3px solid {_G};"
        f"border-bottom:1px solid #1e1e1e;overflow:hidden;margin-bottom:1.2rem;'>"
        f"<div style='display:flex;align-items:stretch;'>"
        f"{photo}"
        f"<div style='flex:1;padding:1rem 1.4rem;display:flex;flex-direction:column;"
        f"justify-content:center;border-left:1px solid #1e1e1e;'>"
        f"<span class='ail-label' style='margin:0 0 8px;'>About the Contributor</span>"
        f"<span class='ail-name' style='margin:0 0 6px;'>Ilian Zalomai</span>"
        f"<span class='ail-sub' style='margin:0 0 8px;'>MSF Candidate &middot; Purdue Daniels "
        f"&middot; Payment Systems &amp; Fraud &middot; Former Deloitte</span>"
        f"<span class='ail-tgln' style='margin:0 0 10px;'>Fintech and banking professional with "
        f"4+ years leading payment systems and fraud operations at scale across high-volume travel "
        f"platforms. Strategy and transformation consulting at Deloitte across Frankfurt and Leipzig. "
        f"Bridging operational finance, risk management, and quantitative analytics.</span>"
        f"<div>"
        f"<a href='https://www.linkedin.com/in/ilian-zalomai-55iz/' target='_blank' "
        f"class='ail-link'>LinkedIn</a>"
        f"</div>"
        f"</div></div></div>",
        unsafe_allow_html=True,
    )

    col_main, col_side = st.columns([1.55, 1], gap="large")

    with col_main:

        # ── Profile ───────────────────────────────────────────────────────────
        st.markdown(
            f"<div style='{_SEC}'>"
            f"<span class='ail-label'>Profile</span>"
            f"<span class='ail-body' style='display:block;margin-bottom:8px;'>Experienced in payment "
            f"systems, fraud prevention, and banking technology consulting. Four years building and "
            f"managing risk and security operations across high-volume travel fintech platforms &mdash; "
            f"implementing chargeback frameworks, fraud monitoring, and customer verification systems "
            f"at scale.</span>"
            f"<span class='ail-body' style='display:block;'>The MSF at Purdue&rsquo;s Financial "
            f"Analytics track is deepening the quantitative foundation in derivatives and risk modeling, "
            f"bringing structured rigor to the intersection of risk management, data analytics, and "
            f"financial markets.</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── This Project ──────────────────────────────────────────────────────
        st.markdown(
            f"<div style='{_SEC}'>"
            f"<span class='ail-label'>This Project</span>"
            f"<span class='ail-body' style='display:block;margin-bottom:8px;'>Geopolitical Research "
            f"and Scenario Design contributor on the Equity-Commodities Spillover Monitor &mdash; "
            f"Course Project 3 for Prof. Cinder Zhang&rsquo;s MGMT 69000-120.</span>"
            f"<ul style='margin:0 0 6px;padding-left:14px;list-style:disc;'>"
            f"<li style='margin-bottom:4px;'><span class='ail-body'>Designed the War Impact Map "
            f"scoring framework: per-war baseline scores, country-level commodity exposure weights, "
            f"and concurrent-war amplifier methodology for Ukraine, Gaza/Red Sea, and Iran/Hormuz "
            f"conflicts.</span></li>"
            f"<li style='margin-bottom:4px;'><span class='ail-body'>Authored all 13 geopolitical "
            f"event descriptions (2008&ndash;2025: GFC, Arab Spring, US-China Trade War, Aramco "
            f"Attack, COVID, WTI Negative, Ukraine War, LME Nickel Squeeze, Fed Hiking Cycle, SVB "
            f"Crisis, Israel-Hamas, India-Pakistan, Iran/Hormuz).</span></li>"
            f"<li style='margin-bottom:4px;'><span class='ail-body'>Built the Strait Watch framework: "
            f"five critical maritime chokepoints (Hormuz, Malacca, Suez, Bab-el-Mandeb, Turkish "
            f"Straits), EIA/IEA throughput data, and live disruption scoring.</span></li>"
            f"<li style='margin-bottom:4px;'><span class='ail-body'>Researched the Iran/Hormuz "
            f"escalation scenario: oil and LNG regional transit exposure, energy-importing Asia risk "
            f"analysis, and Brent/WTI spread interpretation.</span></li>"
            f"<li style='margin-bottom:0;'><span class='ail-body'>Authored narrative content across "
            f"the dashboard &mdash; page intros, takeaway blocks, and section conclusions connecting "
            f"quantitative outputs to real-world market implications.</span></li>"
            f"</ul>"
            f"<div style='{_SROW}'>"
            f"<div style='{_SITM}background:#080808;'>"
            f"<span class='ail-num'>13</span>"
            f"<span class='ail-slbl'>Events Catalogued</span></div>"
            f"<div style='{_SITM}background:#080808;'>"
            f"<span class='ail-num'>5</span>"
            f"<span class='ail-slbl'>Chokepoints Mapped</span></div>"
            f"<div style='{_SITM}background:#080808;'>"
            f"<span class='ail-num'>3</span>"
            f"<span class='ail-slbl'>Active Conflicts</span></div>"
            f"<div style='flex:1;text-align:center;padding:0.65rem 0.25rem;background:#080808;'>"
            f"<span class='ail-num'>40+</span>"
            f"<span class='ail-slbl'>Countries Scored</span></div>"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── Experience ────────────────────────────────────────────────────────
        st.markdown(
            f"<div style='{_SEC}'>"
            f"<span class='ail-label'>Professional Experience</span>"
            + _exp(
                "Payment Systems, Fraud &amp; Security Supervisor",
                "Firebird Tours &middot; Rail.Ninja &middot; Firebirdtours &middot; Triptile",
                "January 2024 &ndash; Present &middot; Full-time",
                [
                    "Managing payment systems, risk, and security operations across high-volume "
                    "travel platforms — overseeing fraud monitoring and customer verification at scale.",
                    "Implemented outbound-chargeback processes and increased chargeback win rate "
                    "while reducing average document preparation time by 70% through systematic "
                    "workflow redesign.",
                    "Building analytical reports and models to identify fraud patterns, optimize "
                    "anti-fraud rules, and support data-driven risk decisions.",
                ],
            )
            + _exp(
                "Banking Technology Strategy &amp; Transformation Analyst",
                "Deloitte",
                "April 2024 &ndash; August 2024 &middot; Frankfurt, Germany",
                [
                    "Supported Privileged Access Management (PAM) assessment implementation for "
                    "a major banking client.",
                    "Conducted comprehensive research on compliance trends across internal and "
                    "external regulatory frameworks; prepared management sales deck materials and "
                    "coordinated across initiative workstreams.",
                ],
            )
            + _exp(
                "Banking Operations Analyst",
                "Deloitte",
                "April 2023 &ndash; July 2023 &middot; Leipzig, Germany",
                [
                    "Supported banking transformation initiatives across Credit, Cloud, and "
                    "Web 3.0/Metaverse workstreams.",
                    "Researched market trends including platformification, cloud hyperscaler "
                    "sustainability, and metaverse applications in banking; prepared and presented "
                    "management decks across workstreams.",
                ],
            )
            + _exp(
                "Payment Systems, Fraud &amp; Security Team Lead",
                "Firebird Tours",
                "March 2022 &ndash; December 2023 &middot; 1 yr 10 mos",
                [
                    "Led the payment systems and fraud team, building fraud detection infrastructure "
                    "and managing chargeback resolution across multiple high-volume travel booking "
                    "platforms.",
                ],
            )
            + _exp(
                "Credit Department Analyst",
                "Volksbanken Raiffeisenbanken &middot; Part-time",
                "May 2022 &ndash; July 2022 &middot; Mittweida, Germany",
                [
                    "Collected, verified, and organized credit documentation; supported research on "
                    "cost-of-living trends and banking approach plausibility assessments.",
                ],
            )
            + _exp(
                "MUN Security Council Chairman &amp; Organising Committee",
                "United Nations in Belarus &middot; FIRMUN / OctoMUN",
                "October 2018 &ndash; March 2022 &middot; Minsk, Belarus",
                [
                    "Served as Security Council Chairman and President of General Assembly across "
                    "multiple Model UN conferences; co-chaired EcoFin committee.",
                    "Led the organizing committee for FIRMUN and OctoMUN editions, coordinating "
                    "delegate logistics, agenda setting, and committee operations.",
                ],
            )
            + "</div>",
            unsafe_allow_html=True,
        )

    with col_side:

        # ── Education ─────────────────────────────────────────────────────────
        st.markdown(
            f"<div style='{_SEC}'>"
            f"<span class='ail-label'>Education</span>"
            f"<div style='{_EDUI}'>"
            f"<span class='ail-sch' style='margin:0 0 2px;'>Purdue University</span>"
            f"<span class='ail-dept' style='margin:0 0 2px;'>Mitchell E. Daniels, Jr. School of Business</span>"
            f"<span class='ail-deg' style='margin:0 0 2px;'>M.S. Finance &middot; Financial Analytics Track</span>"
            f"<span class='ail-year'>July 2025 &ndash; May 2026 &middot; West Lafayette, IN</span>"
            f"</div>"
            f"<div style='padding:0.45rem 0;'>"
            f"<span class='ail-sch' style='margin:0 0 2px;'>Hochschule Mittweida</span>"
            f"<span class='ail-dept' style='margin:0 0 2px;'>Germany</span>"
            f"<span class='ail-deg' style='margin:0 0 2px;'>B.A. Business Administration</span>"
            f"<span class='ail-year'>March 2022 &ndash; April 2024</span>"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── Interests ─────────────────────────────────────────────────────────
        st.markdown(
            f"<div style='{_SEC}'>"
            f"<span class='ail-label'>Interests</span>"
            f"<div>"
            f"<span class='ail-tag-g'>Fintech &amp; Risk</span>"
            f"<span class='ail-tag-n'>Payment Systems</span>"
            f"<span class='ail-tag-g'>Financial Analytics</span>"
            f"<span class='ail-tag-n'>Fraud Prevention</span>"
            f"<span class='ail-tag-g'>Banking Strategy</span>"
            f"<span class='ail-tag-n'>Derivatives &amp; Risk</span>"
            f"<span class='ail-tag-g'>Geopolitical Risk</span>"
            f"<span class='ail-tag-n'>Maritime Trade</span>"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── Acknowledgments ───────────────────────────────────────────────────
        st.markdown(
            f"<div style='{_SEC}'>"
            f"<span class='ail-label'>Acknowledgments</span>"
            f"<div style='background:#080808;border:1px solid #1e1e1e;"
            f"border-left:2px solid rgba(207,185,145,.40);"
            f"padding:0.55rem 0.75rem;'>"
            f"<span class='ail-body'>"
            f"<strong style='color:#c8c8c8;font-weight:600;'>Prof. Cinder Zhang</strong> "
            f"&middot; MGMT 69000: Connecting geopolitical events to systematic regime-based "
            f"market analysis and teaching us to ground quantitative outputs in real-world "
            f"market implications."
            f"</span>"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    _page_footer()
