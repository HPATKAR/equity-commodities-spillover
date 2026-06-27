"""About: Ilian Zalomai."""

from __future__ import annotations

import streamlit as st

from src.ui.shared import _page_footer
from src.ui.about_shared import (
    _ABOUT_STYLE, _photo_html, _hero, _exp, _stat_row, _edu, _ack,
    _SEC,
)


def page_about_ilian() -> None:
    st.markdown(_ABOUT_STYLE, unsafe_allow_html=True)

    photo = _photo_html("photo_ilian.jpeg", "Ilian Zalomai")

    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown(
        _hero(
            photo_html=photo,
            role_lbl="CONTRIBUTOR · GEOPOLITICAL RESEARCH &amp; SCENARIO DESIGN",
            name="Ilian Zalomai",
            sub=(
                "MSF CANDIDATE &nbsp;·&nbsp; PURDUE DANIELS &nbsp;·&nbsp; "
                "Payment Systems &amp; Fraud &nbsp;·&nbsp; Former Deloitte"
            ),
            tagline=(
                "Fintech and banking professional with 4+ years leading payment systems and fraud "
                "operations at scale across high-volume travel platforms. Strategy and transformation "
                "consulting at Deloitte across Frankfurt and Leipzig. Bridging operational finance, "
                "risk management, and quantitative analytics."
            ),
            links_html=(
                "<a href='https://www.linkedin.com/in/ilian-zalomai-55iz/' target='_blank' "
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
            f"<span class='abt-body' style='display:block;margin-bottom:8px;'>"
            f"Experienced in payment systems, fraud prevention, and banking technology consulting. "
            f"Four years building and managing risk and security operations across high-volume travel "
            f"fintech platforms &mdash; implementing chargeback frameworks, fraud monitoring, and "
            f"customer verification systems at scale.</span>"
            f"<span class='abt-body' style='display:block;'>"
            f"The MSF at Purdue&rsquo;s Financial Analytics track is deepening the quantitative "
            f"foundation in derivatives and risk modeling, bringing structured rigor to the "
            f"intersection of risk management, data analytics, and financial markets.</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── This Project ──────────────────────────────────────────────────────
        st.markdown(
            f"<div style='{_SEC}'>"
            f"<span class='abt-label'>This Project</span>"
            f"<span class='abt-body' style='display:block;margin-bottom:8px;'>"
            f"Geopolitical Research and Scenario Design contributor on the "
            f"Equity-Commodities Spillover Monitor &mdash; "
            f"a research terminal built during the MSF program at Purdue Daniels School of Business.</span>"
            f"<ul style='margin:0 0 10px;padding-left:14px;list-style:disc;'>"
            f"<li style='margin-bottom:4px;'><span class='abt-body'>Designed the War Impact Map "
            f"scoring framework: per-war baseline scores, country-level commodity exposure weights, "
            f"and concurrent-war amplifier methodology for Ukraine, Gaza/Red Sea, and Iran/Hormuz "
            f"conflicts.</span></li>"
            f"<li style='margin-bottom:4px;'><span class='abt-body'>Authored all 13 geopolitical "
            f"event descriptions (2008&ndash;2025: GFC, Arab Spring, US-China Trade War, Aramco "
            f"Attack, COVID, WTI Negative, Ukraine War, LME Nickel Squeeze, Fed Hiking Cycle, SVB "
            f"Crisis, Israel-Hamas, India-Pakistan, Iran/Hormuz).</span></li>"
            f"<li style='margin-bottom:4px;'><span class='abt-body'>Built the Strait Watch framework: "
            f"five critical maritime chokepoints (Hormuz, Malacca, Suez, Bab-el-Mandeb, Turkish "
            f"Straits), EIA/IEA throughput data, and live disruption scoring.</span></li>"
            f"<li style='margin-bottom:4px;'><span class='abt-body'>Researched the Iran/Hormuz "
            f"escalation scenario: oil and LNG regional transit exposure, energy-importing Asia risk "
            f"analysis, and Brent/WTI spread interpretation.</span></li>"
            f"<li style='margin-bottom:0;'><span class='abt-body'>Authored narrative content across "
            f"the dashboard &mdash; page intros, takeaway blocks, and section conclusions connecting "
            f"quantitative outputs to real-world market implications.</span></li>"
            f"</ul>"
            + _stat_row([
                ("13", "Events Catalogued"),
                ("5",  "Chokepoints Mapped"),
                ("3",  "Active Conflicts"),
                ("40+", "Countries Scored"),
            ])
            + f"</div>",
            unsafe_allow_html=True,
        )

        # ── Experience ────────────────────────────────────────────────────────
        st.markdown(
            f"<div style='{_SEC}'>"
            f"<span class='abt-label'>Professional Experience</span>"
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
            f"<span class='abt-label'>Education</span>"
            + _edu(
                "Purdue University",
                "Mitchell E. Daniels, Jr. School of Business",
                "M.S. Finance &middot; Financial Analytics Track",
                "July 2025 &ndash; May 2026 &middot; West Lafayette, IN",
            )
            + _edu(
                "Hochschule Mittweida", "Germany",
                "B.A. Business Administration",
                "March 2022 &ndash; April 2024",
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
            f"<span class='abt-tag-g'>Fintech &amp; Risk</span>"
            f"<span class='abt-tag-n'>Payment Systems</span>"
            f"<span class='abt-tag-g'>Financial Analytics</span>"
            f"<span class='abt-tag-n'>Fraud Prevention</span>"
            f"<span class='abt-tag-g'>Banking Strategy</span>"
            f"<span class='abt-tag-n'>Derivatives &amp; Risk</span>"
            f"<span class='abt-tag-g'>Geopolitical Risk</span>"
            f"<span class='abt-tag-n'>Maritime Trade</span>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

        # ── Acknowledgments ───────────────────────────────────────────────────
        st.markdown(
            f"<div style='{_SEC}'>"
            f"<span class='abt-label'>Acknowledgments</span>"
            + _ack(
                "Prof. Cinder Zhang",
                "AI for Finance &middot; Connecting geopolitical events to systematic regime-based "
                "market analysis and grounding quantitative outputs in real-world "
                "market implications.",
            )
            + "</div>",
            unsafe_allow_html=True,
        )

    _page_footer()
