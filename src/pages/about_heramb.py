"""About: Heramb S. Patkar."""

from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st

from src.ui.shared import _page_footer

_ASSETS = Path(__file__).resolve().parent.parent.parent / "assets"
_GOLD = "#CFB991"

# ── CSS injected once per page load ───────────────────────────────────────────
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
.abt-cour{font-family:'JetBrains Mono',monospace!important;font-size:8px!important;color:#8E9AAA!important;letter-spacing:.02em;display:block;}
.abt-link{font-family:'JetBrains Mono',monospace!important;font-size:9px!important;font-weight:700!important;color:#CFB991!important;text-decoration:none!important;letter-spacing:.10em;text-transform:uppercase;opacity:.9;margin-right:1.2rem;display:inline-block;}
.abt-link:hover{opacity:1!important;}
.abt-num{font-family:'JetBrains Mono',monospace!important;font-size:16px!important;font-weight:700!important;color:#CFB991!important;line-height:1;display:block;}
.abt-slbl{font-family:'DM Sans',sans-serif!important;font-size:8px!important;text-transform:uppercase;letter-spacing:.14em;color:#8E9AAA!important;display:block;}
.abt-tag-g{display:inline-block;padding:2px 8px;font-family:'DM Sans',sans-serif!important;font-size:9px!important;font-weight:600!important;margin:2px;background:rgba(207,185,145,.08);color:#CFB991!important;border:1px solid rgba(207,185,145,.22);}
.abt-tag-n{display:inline-block;padding:2px 8px;font-family:'DM Sans',sans-serif!important;font-size:9px!important;font-weight:600!important;margin:2px;background:transparent;color:#a8b0c0!important;border:1px solid #2a2a2a;}
.abt-ack{font-family:'DM Sans',sans-serif!important;font-size:11px!important;color:#a8b0c0!important;line-height:1.68;}
.abt-pub-title{font-family:'DM Sans',sans-serif!important;font-size:11px!important;font-weight:600!important;color:#e8e9ed!important;line-height:1.55;display:block;}
.abt-pub-auth{font-family:'DM Sans',sans-serif!important;font-size:10px!important;color:#a8b0c0!important;display:block;}
.abt-pub-journal{font-family:'DM Sans',sans-serif!important;font-size:9px!important;color:#CFB991!important;font-style:italic;display:block;}
.abt-cert-name{font-family:'DM Sans',sans-serif!important;font-size:11px!important;font-weight:600!important;color:#e8e9ed!important;display:block;}
.abt-cert-issuer{font-family:'JetBrains Mono',monospace!important;font-size:9px!important;color:#8E9AAA!important;display:block;}
</style>"""

_SEC   = "border-top:1px solid #1e1e1e;padding:0.75rem 0 0.25rem 0;margin-bottom:0.1rem;"
_EXPI  = "padding:0.36rem 0 0.36rem 0.65rem;border-left:1px solid #1e1e1e;margin-bottom:0.36rem;"
_EDUI  = "padding:0.36rem 0;border-bottom:1px solid #1a1a1a;"
_SROW  = "display:flex;gap:0;margin-top:0.55rem;border-top:1px solid #1e1e1e;border-bottom:1px solid #1e1e1e;"
_SITM  = "flex:1;text-align:center;padding:0.36rem 0.15rem;border-right:1px solid #1e1e1e;"


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


def page_about_heramb() -> None:
    st.markdown(_STYLE, unsafe_allow_html=True)

    photo = _photo_html("photo_heramb.jpeg", "Heramb S. Patkar")

    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown(
        f"<div style='background:#141414;border-top:2px solid {_GOLD};"
        f"border-bottom:1px solid #1e1e1e;overflow:hidden;margin-bottom:1.2rem;'>"
        f"<div style='display:flex;align-items:stretch;'>"
        f"{photo}"
        f"<div style='flex:1;padding:0.9rem 1.2rem;display:flex;flex-direction:column;"
        f"justify-content:center;border-left:1px solid #1e1e1e;'>"
        f"<span class='abt-label' style='margin:0 0 6px 0;'>About the Author</span>"
        f"<span class='abt-name' style='margin:0 0 4px 0;'>Heramb S. Patkar</span>"
        f"<span class='abt-sub' style='margin:0 0 6px 0;'>MSF Candidate &middot; Purdue Daniels School of Business "
        f"&middot; hpatkar@purdue.edu &middot; West Lafayette, IN</span>"
        f"<span class='abt-tgln' style='margin:0 0 8px 0;'>BITS Pilani mechanical engineering graduate and NISM XV certified research analyst "
        f"with experience across equity research, venture capital, and U.S. capital markets. "
        f"Practicum consultant at a Houston advisory firm, VC fund analyst at Purdue's student-managed fund, "
        f"and published researcher (Indian Academy of Sciences, 2025).</span>"
        f"<div style='display:flex;flex-wrap:wrap;gap:0;'>"
        f"<a href='https://www.linkedin.com/in/heramb-patkar/' target='_blank' class='abt-link'>LinkedIn</a>"
        f"<a href='https://github.com/HPATKAR' target='_blank' class='abt-link'>GitHub</a>"
        f"<a href='https://www.ias.ac.in/article/fulltext/boms/048/0028' target='_blank' class='abt-link'>Publication</a>"
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
            f"<span class='abt-body' style='display:block;margin-bottom:6px;'>Driven by curiosity about how businesses create value and grow stronger. "
            f"With a foundation in engineering and hands-on experience in global equity research, I enjoy building financial models, "
            f"analysing industries, and synthesising data into decisions. Excited by intersections of analytical thinking and "
            f"markets &mdash; particularly where quantitative methods meet real-world capital allocation.</span>"
            f"<span class='abt-body' style='display:block;'>Outside of work and markets, I enjoy exploring new places, listening to Carnatic music, "
            f"and learning from different cultures. Always open to connecting &mdash; feel free to reach out.</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── This Project ──────────────────────────────────────────────────────
        st.markdown(
            f"<div style='{_SEC}'>"
            f"<span class='abt-label' style='margin-bottom:8px;'>This Project</span>"
            f"<span class='abt-body' style='display:block;margin-bottom:6px;'>Lead developer and sole quantitative architect on the "
            f"Equity-Commodities Spillover Monitor &mdash; Course Project 3 for Prof. Cinder Zhang's MGMT 69000-120 "
            f"(AI for Finance). Responsible for the entire codebase, architecture, and all 14 dashboard pages.</span>"
            f"<ul style='margin:0 0 0 0;padding-left:14px;list-style:disc;'>"
            f"<li style='margin-bottom:3px;'><span class='abt-body'>Built an institutional-grade cross-asset analytics dashboard tracking "
            f"spillover dynamics across 15 equity indices, 17 commodity futures, fixed income, and FX</span></li>"
            f"<li style='margin-bottom:3px;'><span class='abt-body'>Implemented the full quantitative engine: DCC-GARCH regime detection, "
            f"Diebold-Yilmaz FEVD connectedness, Granger causality, Hidden Markov Model regime detection, and transfer entropy</span></li>"
            f"<li style='margin-bottom:3px;'><span class='abt-body'>Architected a 7-agent AI orchestration pipeline (Risk Officer, Macro Strategist, "
            f"Geo Analyst, Commodities Specialist, Stress Engineer, Trade Structurer, CQO) with a 4-round dependency-ordered "
            f"execution pipeline that auto-generates morning briefings, stress outputs, and regime-filtered trade signals</span></li>"
            f"<li style='margin-bottom:3px;'><span class='abt-body'>Built the full data layer: yfinance multi-asset downloader (50+ tickers), "
            f"FRED API integration (14 macro series), real-time implied vol scraping (VIX/OVX/GVZ/VVIX), and proactive alert engine</span></li>"
            f"<li style='margin-bottom:0;'><span class='abt-body'>Designed all 14 pages end-to-end: Overview, Correlation, Spillover, Macro Intelligence, "
            f"Geopolitical Triggers, War Impact Map, Strait Watch, Trade Ideas, Portfolio Stress Test, Scenario Engine, "
            f"Commodities Watchlist, Model Accuracy, Actionable Insights, and AI Chat</span></li>"
            f"</ul>"
            f"<div style='{_SROW}'>"
            f"<div style='{_SITM}'><span class='abt-num'>14</span><span class='abt-slbl'>Pages Built</span></div>"
            f"<div style='{_SITM}'><span class='abt-num'>7</span><span class='abt-slbl'>AI Agents</span></div>"
            f"<div style='{_SITM}'><span class='abt-num'>5</span><span class='abt-slbl'>Quant Models</span></div>"
            f"<div style='{_SITM}border-right:none;'><span class='abt-num'>23K+</span><span class='abt-slbl'>Lines of Code</span></div>"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── Experience ────────────────────────────────────────────────────────
        st.markdown(
            f"<div style='{_SEC}'>"
            f"<span class='abt-label' style='margin-bottom:8px;'>Professional Experience</span>"
            + _exp(
                "Practicum Consultant",
                "Fino Advisors LLC",
                "January 2026 &ndash; Present &middot; Houston, TX (Remote)",
                [
                    "Building and refining a confidential Series A investor model (revenue drivers, scenarios/sensitivities) "
                    "and synthesising technical and financial inputs into decision-ready insights for fundraising conversations.",
                    "Conducting valuation work and comps benchmarking; supporting investor materials for a sustainability/"
                    "infrastructure portfolio company under NDA, translating assumptions into KPIs and fundraising-ready outputs.",
                ],
            )
            + _exp(
                "Analyst",
                "Student Managed Venture Fund, Purdue University",
                "January 2026 &ndash; Present &middot; West Lafayette, IN",
                [
                    "Build and refine financial models to evaluate early-stage deal flows each cycle, screening for business "
                    "model viability, unit economics, and operational red flags across Purdue-sourced startups.",
                    "Present investment recommendations to professional VC panels, structuring due diligence findings into "
                    "decision-ready memos and reducing diligence cycle time through systematic early-stage risk filtering.",
                ],
            )
            + _exp(
                "Student Extern",
                "Equity Methods",
                "March 2026 &middot; Champaign, IL",
                [
                    "Used SAS to forecast stock compensation expenses for a Fortune 100 company during a two-day externship, "
                    "refining forecasting techniques, improving projection accuracy, and stress-testing assumptions across grant types.",
                    "Developed client-ready decks with graphical disclosures and CAP calculations for proxy filings; applied "
                    "Pay vs. Performance (PVP) analysis to align exec compensation with shareholder returns and business outcomes.",
                ],
            )
            + _exp(
                "Equity Research Associate",
                "Axis Securities Ltd",
                "September 2024 &ndash; April 2025 &middot; Mumbai, India",
                [
                    "Led Auto &amp; Auto Ancillary equity research alongside the lead analyst, building DCF models for 14 stocks, "
                    "delivering IPO notes, earnings updates, and industry outlooks to family offices and fund managers; owned a "
                    "dynamic industry tracker (production, sales, inventory) across segments and engine classes.",
                    "Pitched stock ideas and thematic plays to the PMS team (AUM ~$1 Bn) and Head of Research, engaging senior "
                    "management on pricing, margins, capex, and tariffs to sharpen internal positioning and client-facing strategy.",
                ],
            )
            + _exp(
                "Equity Research Intern",
                "Axis Securities Ltd",
                "July 2024 &ndash; August 2024 &middot; Mumbai, India",
                [
                    "Supported coverage across Pharma and Hospitality (13 stocks), contributing to thematic research, "
                    "initiating coverage, and management interactions.",
                    "Converted to full-time Associate; secured a recommendation from the Head of Research.",
                ],
            )
            + _exp(
                "Undergraduate Research Assistant",
                "BITS Pilani",
                "April 2022 &ndash; May 2024 &middot; Hyderabad, India",
                [
                    "Co-designed and validated a low-cost stereotaxic device for rodent brain research; contributed to "
                    "fabrication methodology, mechanical validation, and experimental protocols.",
                    "Work published in the Bulletin of Materials Science (Indian Academy of Sciences, February 2025).",
                ],
            )
            + _exp(
                "Manufacturing Engineer Intern",
                "Divgi TorqTransfer Systems Ltd",
                "July 2023 &ndash; December 2023 &middot; Sirsi, Karnataka, India",
            )
            + _exp(
                "HVAC Engineer Intern",
                "Grasim Industries Limited, Pulp &amp; Fibre",
                "May 2022 &ndash; July 2022 &middot; Nagda, Madhya Pradesh, India",
            )
            + "</div>",
            unsafe_allow_html=True,
        )

        # ── Selected Projects ─────────────────────────────────────────────────
        st.markdown(
            f"<div style='{_SEC}'>"
            f"<span class='abt-label' style='margin-bottom:8px;'>Selected Projects</span>"
            + _exp(
                "Equity-Commodities Spillover Terminal",
                "Purdue Daniels &middot; MGMT 69000-120 (Prof. C. Zhang)",
                "",
                [
                    "Built an institutional-grade cross-asset analytics dashboard tracking spillover dynamics across 15 equity "
                    "indices, 17 commodity futures, fixed income, and FX.",
                    "Integrates DCC-GARCH regime detection, Diebold-Yilmaz FEVD connectedness, Granger causality, and transfer "
                    "entropy with a 7-agent AI orchestration pipeline to auto-generate morning briefings, scenario stress outputs, "
                    "and regime-filtered trade signals.",
                ],
            )
            + _exp(
                "JGB Repricing Framework",
                "Purdue Daniels &middot; MGMT 69000-119 (Prof. C. Zhang)",
                "",
                [
                    "Built a quantitative dashboard for JGB repricing as the BOJ exits YCC, using regime filters "
                    "(MS/HMM/GARCH), yield PCA, DCC spillovers, and cross-asset transfer entropy to auto-generate trade signals.",
                ],
            )
            + _exp(
                "PepsiCo / Vita Coco Acquisition",
                "Purdue Daniels &middot; MGMT 64500 (Prof. S. Chernenko)",
                "",
                [
                    "Built a DCF and EBIT exit-multiple valuation, quantified revenue and cost synergies, and structured "
                    "a $3.9&ndash;4.2B all-cash bid delivering 1.5% EPS accretion for a proposed acquisition of Vita Coco.",
                ],
            )
            + _exp(
                "Eli Lilly / Madrigal Pharmaceuticals M&amp;A",
                "IU Bloomington MSF Case Competition",
                "",
                [
                    "Built a patient-based DCF for Lilly's acquisition of Madrigal (~$10.4&ndash;22B), quantified synergies "
                    "against pharma M&amp;A precedents, and recommended a $640&ndash;809/share offer.",
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
            f"<span class='abt-year' style='margin:0 0 2px 0;'>2025 &ndash; 2026 &middot; West Lafayette, IN</span>"
            f"<span class='abt-cour'>Courses: Investment Banking &middot; AI in Finance &middot; "
            f"Financial Modelling &middot; Venture Capital</span>"
            f"</div>"

            f"<div style='{_EDUI}border-bottom:none;'>"
            f"<span class='abt-sch' style='margin:0 0 2px 0;'>BITS Pilani</span>"
            f"<span class='abt-dept' style='margin:0 0 2px 0;'>Hyderabad Campus</span>"
            f"<span class='abt-deg' style='margin:0 0 2px 0;'>B.E. (Hons.) Mechanical Engineering</span>"
            f"<span class='abt-year' style='margin:0 0 2px 0;'>2020 &ndash; 2024 &middot; Hyderabad, India</span>"
            f"<span class='abt-cour'>Projects: Optimization &middot; Manufacturing &middot; "
            f"Nano Materials &middot; Thermal Energy Storage</span>"
            f"</div>"

            f"</div>",
            unsafe_allow_html=True,
        )

        # ── Publication ───────────────────────────────────────────────────────
        st.markdown(
            f"<div style='{_SEC}'>"
            f"<span class='abt-label' style='margin-bottom:8px;'>Publication</span>"
            f"<div style='padding:4px 0;'>"
            f"<span class='abt-pub-title' style='margin:0 0 4px 0;'>"
            f"Design, Fabrication and Validation of a Low-Cost Stereotaxic Device "
            f"for Brain Research in Rodents</span>"
            f"<span class='abt-pub-auth' style='margin:0 0 3px 0;'>"
            f"A. Wadkar, <strong style='color:#c8c8c8;'>H. Patkar</strong>, S.P. Kommajosyula</span>"
            f"<span class='abt-pub-journal' style='margin:0 0 3px 0;'>"
            f"Bulletin of Materials Science, Vol. 48, Article 0028</span>"
            f"<span class='abt-meta' style='margin:0 0 6px 0;'>"
            f"Indian Academy of Sciences &middot; February 2025</span>"
            f"<a href='https://www.ias.ac.in/article/fulltext/boms/048/0028' target='_blank' class='abt-link'>"
            f"View Full Text</a>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

        # ── Certifications ────────────────────────────────────────────────────
        st.markdown(
            f"<div style='{_SEC}'>"
            f"<span class='abt-label' style='margin-bottom:8px;'>Certifications</span>"
            f"<div style='padding:6px 0;border-bottom:1px solid #1a1a1a;'>"
            f"<span class='abt-cert-name' style='margin:0 0 2px 0;'>NISM Series XV: Research Analyst</span>"
            f"<span class='abt-cert-issuer'>NISM &middot; Oct 2024 &ndash; Oct 2027</span></div>"
            f"<div style='padding:6px 0;'>"
            f"<span class='abt-cert-name' style='margin:0 0 2px 0;'>Bloomberg Market Concepts</span>"
            f"<span class='abt-cert-issuer'>Bloomberg LP &middot; February 2024</span></div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── Skills ────────────────────────────────────────────────────────────
        st.markdown(
            f"<div style='{_SEC}'>"
            f"<span class='abt-label' style='margin-bottom:6px;'>Skills</span>"
            f"<div>"
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
            f"<span class='abt-label' style='margin-bottom:6px;'>Interests</span>"
            f"<div>"
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
            f"<span class='abt-label' style='margin-bottom:8px;'>Acknowledgments</span>"
            f"<div class='abt-ack' style='margin-bottom:6px;'><strong style='color:#c8c8c8;font-weight:600;'>Prof. Cinder Zhang</strong> "
            f"&middot; MGMT 69000: Framework-first thinking behind regime and spillover design</div>"
            f"<div class='abt-ack' style='margin-bottom:0;'><strong style='color:#c8c8c8;font-weight:600;'>Prof. Adem Atmaz</strong> "
            f"&middot; MGMT 511: Fixed income intuition behind DCC-GARCH and transfer entropy approaches</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    _page_footer()
