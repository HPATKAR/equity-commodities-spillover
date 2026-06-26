"""About: Heramb S. Patkar."""

from __future__ import annotations

import streamlit as st

from src.ui.shared import _page_footer
from src.ui.about_shared import (
    _ABOUT_STYLE, _photo_html, _hero, _exp, _stat_row, _edu, _ack,
    _SEC, _PUB,
)


def page_about_heramb() -> None:
    st.markdown(_ABOUT_STYLE, unsafe_allow_html=True)

    photo = _photo_html("photo_heramb.jpeg", "Heramb S. Patkar")

    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown(
        _hero(
            photo_html=photo,
            role_lbl="CONTRIBUTOR · LEAD DEVELOPER",
            name="Heramb S. Patkar",
            sub=(
                "MSF CANDIDATE &nbsp;·&nbsp; PURDUE DANIELS &nbsp;·&nbsp; "
                "hpatkar@purdue.edu &nbsp;·&nbsp; WEST LAFAYETTE, IN"
            ),
            tagline=(
                "BITS Pilani mechanical engineering graduate and NISM XV certified research analyst "
                "with experience across equity research, venture capital, and U.S. capital markets. "
                "Practicum consultant at a Houston advisory firm, VC fund analyst at Purdue's "
                "student-managed fund, and published researcher (Indian Academy of Sciences, 2025)."
            ),
            links_html=(
                "<a href='https://www.linkedin.com/in/heramb-patkar/' target='_blank' "
                "class='abt-link'>LinkedIn</a>"
                "<a href='https://github.com/HPATKAR' target='_blank' "
                "class='abt-link-ghost'>GitHub</a>"
                "<a href='https://www.ias.ac.in/article/fulltext/boms/048/0028' "
                "target='_blank' class='abt-link-ghost'>Publication</a>"
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
            f"Driven by curiosity about how businesses create value and grow stronger. "
            f"With a foundation in engineering and hands-on experience in global equity research, "
            f"I enjoy building financial models, analysing industries, and synthesising data into "
            f"decisions. Excited by intersections of analytical thinking and markets &mdash; "
            f"particularly where quantitative methods meet real-world capital allocation."
            f"</span>"
            f"<span class='abt-body' style='display:block;'>"
            f"Outside of work and markets, I enjoy exploring new places, listening to Carnatic "
            f"music, and learning from different cultures. Always open to connecting."
            f"</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── This Project ──────────────────────────────────────────────────────
        st.markdown(
            f"<div style='{_SEC}'>"
            f"<span class='abt-label'>This Project</span>"
            f"<span class='abt-body' style='display:block;margin-bottom:8px;'>"
            f"Lead developer and quantitative architect on the "
            f"Equity-Commodities Spillover Monitor &mdash; Course Project 3 for "
            f"Prof. Cinder Zhang's MGMT 69000-120 (AI for Finance). Responsible for the "
            f"entire codebase, architecture, and all 20 analytical dashboard pages."
            f"</span>"
            f"<ul style='margin:0 0 10px;padding-left:14px;list-style:disc;'>"
            f"<li style='margin-bottom:4px;'><span class='abt-body'>"
            f"Built a research-grade cross-asset analytics dashboard tracking "
            f"spillover dynamics across 15 equity indices, 17 commodity futures, "
            f"fixed income, and FX - 24 pages total including documentation and team profiles</span></li>"
            f"<li style='margin-bottom:4px;'><span class='abt-body'>"
            f"Implemented the full quantitative engine: DCC-style dynamic correlation, "
            f"Diebold-Yilmaz FEVD connectedness, Granger causality, Hidden Markov Model "
            f"regime detection, and transfer entropy</span></li>"
            f"<li style='margin-bottom:4px;'><span class='abt-body'>"
            f"Architected an 8-agent AI orchestration pipeline (Risk Officer, Macro Strategist, "
            f"Geo Analyst, Commodities Specialist, Stress Engineer, Signal Auditor, Trade Structurer, CQO) with "
            f"a 4-round dependency-ordered execution pipeline that auto-generates morning "
            f"briefings, stress outputs, and regime-filtered illustrative trade structures</span></li>"
            f"<li style='margin-bottom:4px;'><span class='abt-body'>"
            f"Built a 6-source live intelligence layer: yfinance (50+ tickers), FRED (24 macro "
            f"series), GDELT 2.0 (conflict media escalation), EIA Open Data (petroleum inventories), "
            f"IMF PortWatch (strait vessel traffic), and ACLED (armed conflict events) - "
            f"all feeding a proactive alert engine and AI analyst context</span></li>"
            f"<li style='margin-bottom:0;'><span class='abt-body'>"
            f"Designed 20 analytical pages end-to-end: Command Center, Overview, Macro Lens, "
            f"Correlation, Spillover, Geopolitical Triggers, War Impact Map, Strait Watch, "
            f"Conflict Intelligence, Threat Monitor, Transmission Matrix, Exposure Engine, "
            f"Trade Ideas, Stress Lab, Scenario Simulator, Watchlist, "
            f"Model Audit, AI Research Desk, Methodology, and Intelligence Briefing</span></li>"
            f"</ul>"
            + _stat_row([
                ("20", "Analytical Pages"),
                ("8",  "AI Agents"),
                ("6",  "Live Data Sources"),
                ("30K+", "Lines of Code"),
            ])
            + f"</div>",
            unsafe_allow_html=True,
        )

        # ── Experience ────────────────────────────────────────────────────────
        st.markdown(
            f"<div style='{_SEC}'>"
            f"<span class='abt-label'>Professional Experience</span>"
            + _exp(
                "Practicum Consultant", "Fino Advisors LLC",
                "January 2026 &ndash; Present &middot; Houston, TX (Remote)",
                [
                    "Building and refining a confidential Series A investor model (revenue drivers, "
                    "scenarios/sensitivities) and synthesising technical and financial inputs into "
                    "decision-ready insights for fundraising conversations.",
                    "Conducting valuation work and comps benchmarking; supporting investor materials "
                    "for a sustainability/infrastructure portfolio company under NDA.",
                ],
            )
            + _exp(
                "Analyst", "Student Managed Venture Fund, Purdue University",
                "January 2026 &ndash; Present &middot; West Lafayette, IN",
                [
                    "Build and refine financial models to evaluate early-stage deal flows each cycle, "
                    "screening for business model viability, unit economics, and operational red flags.",
                    "Present investment recommendations to professional VC panels, structuring due "
                    "diligence findings into decision-ready memos.",
                ],
            )
            + _exp(
                "Student Extern", "Equity Methods",
                "March 2026 &middot; Champaign, IL",
                [
                    "Used SAS to forecast stock compensation expenses for a Fortune 100 company; "
                    "refined forecasting techniques and stress-tested assumptions across grant types.",
                    "Developed client-ready decks with graphical disclosures and CAP calculations "
                    "for proxy filings; applied Pay vs. Performance (PVP) analysis.",
                ],
            )
            + _exp(
                "Equity Research Associate", "Axis Securities Ltd",
                "September 2024 &ndash; April 2025 &middot; Mumbai, India",
                [
                    "Led Auto &amp; Auto Ancillary equity research alongside the lead analyst, "
                    "building DCF models for 14 stocks, delivering IPO notes, earnings updates, "
                    "and industry outlooks to family offices and fund managers.",
                    "Pitched stock ideas and thematic plays to the PMS team (AUM ~$1 Bn) and Head "
                    "of Research, engaging senior management on pricing, margins, capex, and tariffs.",
                ],
            )
            + _exp(
                "Equity Research Intern", "Axis Securities Ltd",
                "July 2024 &ndash; August 2024 &middot; Mumbai, India",
                [
                    "Supported coverage across Pharma and Hospitality (13 stocks), contributing "
                    "to thematic research, initiating coverage, and management interactions.",
                    "Converted to full-time Associate; secured a recommendation from the Head of Research.",
                ],
            )
            + _exp(
                "Undergraduate Research Assistant", "BITS Pilani",
                "April 2022 &ndash; May 2024 &middot; Hyderabad, India",
                [
                    "Co-designed and validated a low-cost stereotaxic device for rodent brain research; "
                    "contributed to fabrication methodology, mechanical validation, and experimental protocols.",
                    "Work published in the Bulletin of Materials Science (Indian Academy of Sciences, February 2025).",
                ],
            )
            + _exp(
                "Manufacturing Engineer Intern", "Divgi TorqTransfer Systems Ltd",
                "July 2023 &ndash; December 2023 &middot; Sirsi, Karnataka, India",
            )
            + _exp(
                "HVAC Engineer Intern", "Grasim Industries Limited, Pulp &amp; Fibre",
                "May 2022 &ndash; July 2022 &middot; Nagda, Madhya Pradesh, India",
            )
            + "</div>",
            unsafe_allow_html=True,
        )

        # ── Selected Projects ─────────────────────────────────────────────────
        st.markdown(
            f"<div style='{_SEC}'>"
            f"<span class='abt-label'>Selected Projects</span>"
            + _exp(
                "Equity-Commodities Spillover Terminal",
                "Purdue Daniels &middot; MGMT 69000-120 (Prof. C. Zhang)", "",
                [
                    "Built a research-grade cross-asset analytics dashboard tracking spillover "
                    "dynamics across 15 equity indices, 17 commodity futures, fixed income, and FX.",
                    "Integrates DCC-style dynamic correlation, Diebold-Yilmaz FEVD connectedness, "
                    "Granger causality, and transfer entropy with a 7-agent AI orchestration pipeline.",
                ],
            )
            + _exp(
                "JGB Repricing Framework",
                "Purdue Daniels &middot; MGMT 69000-119 (Prof. C. Zhang)", "",
                [
                    "Built a quantitative dashboard for JGB repricing as the BOJ exits YCC, using "
                    "regime filters (MS/HMM/GARCH), yield PCA, DCC spillovers, and cross-asset "
                    "transfer entropy to generate illustrative research signals.",
                ],
            )
            + _exp(
                "PepsiCo / Vita Coco Acquisition",
                "Purdue Daniels &middot; MGMT 64500 (Prof. S. Chernenko)", "",
                [
                    "Built a DCF and EBIT exit-multiple valuation, quantified revenue and cost "
                    "synergies, and structured a $3.9&ndash;4.2B all-cash bid delivering 1.5% "
                    "EPS accretion.",
                ],
            )
            + _exp(
                "Eli Lilly / Madrigal Pharmaceuticals M&amp;A",
                "IU Bloomington MSF Case Competition", "",
                [
                    "Built a patient-based DCF for Lilly's acquisition of Madrigal (~$10.4&ndash;22B), "
                    "quantified synergies against pharma M&amp;A precedents, and recommended a "
                    "$640&ndash;809/share offer.",
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
                "2025 &ndash; 2026 &middot; West Lafayette, IN",
                "Investment Banking &middot; AI in Finance &middot; Financial Modelling &middot; Venture Capital",
            )
            + _edu(
                "BITS Pilani", "Hyderabad Campus",
                "B.E. (Hons.) Mechanical Engineering",
                "2020 &ndash; 2024 &middot; Hyderabad, India",
                "Optimization &middot; Manufacturing &middot; Nano Materials &middot; Thermal Energy Storage",
                last=True,
            )
            + "</div>",
            unsafe_allow_html=True,
        )

        # ── Publication ───────────────────────────────────────────────────────
        st.markdown(
            f"<div style='{_SEC}'>"
            f"<span class='abt-label'>Publication</span>"
            f"<div style='{_PUB}'>"
            f"<span class='abt-pub-title' style='margin:0 0 5px;'>"
            f"Design, Fabrication and Validation of a Low-Cost Stereotaxic Device "
            f"for Brain Research in Rodents</span>"
            f"<span class='abt-pub-auth' style='margin:0 0 4px;'>"
            f"A. Wadkar, <strong style='color:#c8c8c8;'>H. Patkar</strong>, "
            f"S.P. Kommajosyula</span>"
            f"<span class='abt-pub-journal' style='margin:0 0 4px;'>"
            f"Bulletin of Materials Science, Vol. 48, Article 0028</span>"
            f"<span class='abt-meta' style='margin:0 0 8px;'>"
            f"Indian Academy of Sciences &middot; February 2025</span>"
            f"<a href='https://www.ias.ac.in/article/fulltext/boms/048/0028' "
            f"target='_blank' class='abt-link'>View &rarr;</a>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

        # ── Certifications ────────────────────────────────────────────────────
        st.markdown(
            f"<div style='{_SEC}'>"
            f"<span class='abt-label'>Certifications</span>"
            f"<div style='padding:0.5rem 0;border-bottom:1px solid #161616;'>"
            f"<span class='abt-cert-name' style='margin:0 0 2px;'>"
            f"NISM Series XV: Research Analyst</span>"
            f"<span class='abt-cert-issuer'>NISM &middot; Oct 2024 &ndash; Oct 2027</span>"
            f"</div>"
            f"<div style='padding:0.5rem 0;'>"
            f"<span class='abt-cert-name' style='margin:0 0 2px;'>"
            f"Bloomberg Market Concepts</span>"
            f"<span class='abt-cert-issuer'>Bloomberg LP &middot; February 2024</span>"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── Skills ────────────────────────────────────────────────────────────
        st.markdown(
            f"<div style='{_SEC}'>"
            f"<span class='abt-label'>Skills</span>"
            f"<div style='margin-top:2px;'>"
            f"<span class='abt-tag-g'>Python</span>"
            f"<span class='abt-tag-n'>Excel</span>"
            f"<span class='abt-tag-g'>Bloomberg</span>"
            f"<span class='abt-tag-n'>SAS</span>"
            f"<span class='abt-tag-g'>LSEG</span>"
            f"<span class='abt-tag-n'>ACE Equity</span>"
            f"<span class='abt-tag-g'>Claude CLI</span>"
            f"<span class='abt-tag-n'>Autodesk Suite</span>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

        # ── Interests ─────────────────────────────────────────────────────────
        st.markdown(
            f"<div style='{_SEC}'>"
            f"<span class='abt-label'>Interests</span>"
            f"<div style='margin-top:2px;'>"
            f"<span class='abt-tag-g'>Investment Banking</span>"
            f"<span class='abt-tag-n'>Corporate Finance</span>"
            f"<span class='abt-tag-g'>Valuations</span>"
            f"<span class='abt-tag-n'>Private Equity</span>"
            f"<span class='abt-tag-g'>Equity Research</span>"
            f"<span class='abt-tag-n'>Venture Capital</span>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

        # ── Acknowledgments ───────────────────────────────────────────────────
        st.markdown(
            f"<div style='{_SEC}'>"
            f"<span class='abt-label'>Acknowledgments</span>"
            + _ack(
                "Prof. Cinder Zhang",
                "MGMT 69000 &middot; Framework-first thinking behind regime and spillover design",
            )
            + _ack(
                "Prof. Adem Atmaz",
                "MGMT 511 &middot; Fixed income intuition behind DCC-GARCH and transfer entropy",
            )
            + "</div>",
            unsafe_allow_html=True,
        )

    _page_footer()
