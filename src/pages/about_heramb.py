"""About: Heramb S. Patkar."""

from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st

from src.ui.shared import _page_footer

_ASSETS = Path(__file__).resolve().parent.parent.parent / "assets"
_G   = "#CFB991"   # gold
_DIM = "#8E9AAA"   # dim
_M   = "font-family:'JetBrains Mono',monospace;"
_S   = "font-family:'DM Sans',sans-serif;"

_STYLE = """<style>
.ah-label{
    font-family:'JetBrains Mono',monospace!important;
    font-size:8px!important;font-weight:700!important;
    text-transform:uppercase;letter-spacing:.20em;
    color:#CFB991!important;display:block;
    border-bottom:1px solid #1e1e1e;
    padding-bottom:5px;margin-bottom:10px;
}
.ah-name{
    font-family:'JetBrains Mono',monospace!important;
    font-size:20px!important;font-weight:700!important;
    color:#e8e9ed!important;letter-spacing:-.02em;
    line-height:1.1;display:block;
}
.ah-sub{
    font-family:'JetBrains Mono',monospace!important;
    font-size:8.5px!important;color:#CFB991!important;
    font-weight:600;line-height:1.7;
    letter-spacing:.08em;display:block;
}
.ah-tgln{
    font-family:'DM Sans',sans-serif!important;
    font-size:11px!important;color:#9aa3b0!important;
    line-height:1.72;display:block;
}
.ah-body{
    font-family:'DM Sans',sans-serif!important;
    font-size:11px!important;color:#9aa3b0!important;
    line-height:1.78;
}
.ah-role{
    font-family:'DM Sans',sans-serif!important;
    font-size:11.5px!important;font-weight:700!important;
    color:#e8e9ed!important;line-height:1.3;display:block;
}
.ah-org{
    font-family:'JetBrains Mono',monospace!important;
    font-size:9px!important;font-weight:600!important;
    color:#CFB991!important;display:block;letter-spacing:.04em;
}
.ah-meta{
    font-family:'JetBrains Mono',monospace!important;
    font-size:8.5px!important;color:#8E9AAA!important;display:block;
}
.ah-sch{
    font-family:'DM Sans',sans-serif!important;
    font-size:12px!important;font-weight:700!important;
    color:#e8e9ed!important;display:block;
}
.ah-dept{
    font-family:'DM Sans',sans-serif!important;
    font-size:10px!important;color:#9aa3b0!important;display:block;
}
.ah-deg{
    font-family:'JetBrains Mono',monospace!important;
    font-size:8.5px!important;color:#CFB991!important;
    font-weight:600;letter-spacing:.04em;display:block;
}
.ah-year{
    font-family:'JetBrains Mono',monospace!important;
    font-size:8px!important;color:#8E9AAA!important;display:block;
}
.ah-link{
    font-family:'JetBrains Mono',monospace!important;
    font-size:8px!important;font-weight:700!important;
    color:#080808!important;text-decoration:none!important;
    letter-spacing:.14em;text-transform:uppercase;
    background:#CFB991;padding:3px 10px;
    margin-right:6px;display:inline-block;
    transition:opacity .15s;
}
.ah-link:hover{opacity:.85!important;}
.ah-link-ghost{
    font-family:'JetBrains Mono',monospace!important;
    font-size:8px!important;font-weight:700!important;
    color:#CFB991!important;text-decoration:none!important;
    letter-spacing:.14em;text-transform:uppercase;
    border:1px solid rgba(207,185,145,.35);
    padding:2px 9px;margin-right:6px;display:inline-block;
}
.ah-link-ghost:hover{border-color:#CFB991!important;}
.ah-num{
    font-family:'JetBrains Mono',monospace!important;
    font-size:22px!important;font-weight:700!important;
    color:#CFB991!important;line-height:1;display:block;
}
.ah-slbl{
    font-family:'JetBrains Mono',monospace!important;
    font-size:7px!important;text-transform:uppercase;
    letter-spacing:.14em;color:#8E9AAA!important;display:block;
    margin-top:4px;
}
.ah-tag-g{
    display:inline-block;padding:2px 9px;
    font-family:'JetBrains Mono',monospace!important;
    font-size:8px!important;font-weight:700!important;
    margin:2px;background:rgba(207,185,145,.08);
    color:#CFB991!important;
    border:1px solid rgba(207,185,145,.25);
    letter-spacing:.06em;text-transform:uppercase;
}
.ah-tag-n{
    display:inline-block;padding:2px 9px;
    font-family:'JetBrains Mono',monospace!important;
    font-size:8px!important;font-weight:600!important;
    margin:2px;background:transparent;
    color:#8E9AAA!important;border:1px solid #222;
    letter-spacing:.06em;text-transform:uppercase;
}
.ah-pub-title{
    font-family:'DM Sans',sans-serif!important;
    font-size:11px!important;font-weight:600!important;
    color:#e8e9ed!important;line-height:1.55;display:block;
}
.ah-pub-auth{
    font-family:'DM Sans',sans-serif!important;
    font-size:10px!important;color:#9aa3b0!important;display:block;
}
.ah-pub-journal{
    font-family:'JetBrains Mono',monospace!important;
    font-size:8.5px!important;color:#CFB991!important;display:block;
}
.ah-cert-name{
    font-family:'DM Sans',sans-serif!important;
    font-size:11px!important;font-weight:600!important;
    color:#e8e9ed!important;display:block;
}
.ah-cert-issuer{
    font-family:'JetBrains Mono',monospace!important;
    font-size:8.5px!important;color:#8E9AAA!important;display:block;
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
            f"<span class='ah-body'>{b}</span></li>"
            for b in bullets
        )
        bullets_html = (
            f"<ul style='margin:6px 0 0;padding-left:14px;list-style:disc;'>"
            f"{items}</ul>"
        )
    return (
        f"<div style='{_EXPI}'>"
        f"<span class='ah-role' style='margin:0 0 3px;'>{role}</span>"
        f"<span class='ah-org' style='margin:0 0 2px;'>{org}</span>"
        f"<span class='ah-meta' style='margin:0 0 {6 if bullets else 0}px;'>{meta}</span>"
        f"{bullets_html}"
        f"</div>"
    )


def page_about_heramb() -> None:
    st.markdown(_STYLE, unsafe_allow_html=True)

    photo = _photo_html("photo_heramb.jpeg", "Heramb S. Patkar")

    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown(
        f"<div style='background:#080808;border-top:3px solid {_G};"
        f"border:1px solid #1e1e1e;border-top:3px solid {_G};"
        f"overflow:hidden;margin-bottom:1.4rem;'>"
        f"<div style='display:flex;align-items:stretch;min-height:180px;'>"
        f"{photo}"
        f"<div style='flex:1;padding:1.1rem 1.4rem;display:flex;"
        f"flex-direction:column;justify-content:center;'>"
        f"<span class='ah-meta' style='margin:0 0 8px;letter-spacing:.20em;'>"
        f"CONTRIBUTOR · LEAD DEVELOPER</span>"
        f"<span class='ah-name' style='margin:0 0 6px;'>Heramb S. Patkar</span>"
        f"<span class='ah-sub' style='margin:0 0 10px;'>"
        f"MSF CANDIDATE &nbsp;·&nbsp; PURDUE DANIELS &nbsp;·&nbsp; "
        f"hpatkar@purdue.edu &nbsp;·&nbsp; WEST LAFAYETTE, IN</span>"
        f"<span class='ah-tgln' style='margin:0 0 14px;'>"
        f"BITS Pilani mechanical engineering graduate and NISM XV certified research analyst "
        f"with experience across equity research, venture capital, and U.S. capital markets. "
        f"Practicum consultant at a Houston advisory firm, VC fund analyst at Purdue's "
        f"student-managed fund, and published researcher (Indian Academy of Sciences, 2025)."
        f"</span>"
        f"<div style='display:flex;flex-wrap:wrap;gap:0;align-items:center;'>"
        f"<a href='https://www.linkedin.com/in/heramb-patkar/' target='_blank' class='ah-link'>"
        f"LinkedIn</a>"
        f"<a href='https://github.com/HPATKAR' target='_blank' class='ah-link-ghost'>"
        f"GitHub</a>"
        f"<a href='https://www.ias.ac.in/article/fulltext/boms/048/0028' "
        f"target='_blank' class='ah-link-ghost'>Publication</a>"
        f"</div>"
        f"</div></div></div>",
        unsafe_allow_html=True,
    )

    col_main, col_side = st.columns([1.55, 1], gap="large")

    with col_main:

        # ── Profile ───────────────────────────────────────────────────────────
        st.markdown(
            f"<div style='{_SEC}'>"
            f"<span class='ah-label'>Profile</span>"
            f"<span class='ah-body' style='display:block;margin-bottom:8px;'>"
            f"Driven by curiosity about how businesses create value and grow stronger. "
            f"With a foundation in engineering and hands-on experience in global equity research, "
            f"I enjoy building financial models, analysing industries, and synthesising data into "
            f"decisions. Excited by intersections of analytical thinking and markets &mdash; "
            f"particularly where quantitative methods meet real-world capital allocation."
            f"</span>"
            f"<span class='ah-body' style='display:block;'>"
            f"Outside of work and markets, I enjoy exploring new places, listening to Carnatic "
            f"music, and learning from different cultures. Always open to connecting."
            f"</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── This Project ──────────────────────────────────────────────────────
        st.markdown(
            f"<div style='{_SEC}'>"
            f"<span class='ah-label'>This Project</span>"
            f"<span class='ah-body' style='display:block;margin-bottom:8px;'>"
            f"Lead developer and sole quantitative architect on the "
            f"Equity-Commodities Spillover Monitor &mdash; Course Project 3 for "
            f"Prof. Cinder Zhang's MGMT 69000-120 (AI for Finance). Responsible for the "
            f"entire codebase, architecture, and all 20 analytical dashboard pages."
            f"</span>"
            f"<ul style='margin:0 0 10px;padding-left:14px;list-style:disc;'>"
            f"<li style='margin-bottom:4px;'><span class='ah-body'>"
            f"Built an institutional-grade cross-asset analytics dashboard tracking "
            f"spillover dynamics across 15 equity indices, 17 commodity futures, "
            f"fixed income, and FX - 24 pages total including documentation and team profiles</span></li>"
            f"<li style='margin-bottom:4px;'><span class='ah-body'>"
            f"Implemented the full quantitative engine: DCC-GARCH regime detection, "
            f"Diebold-Yilmaz FEVD connectedness, Granger causality, Hidden Markov Model "
            f"regime detection, and transfer entropy</span></li>"
            f"<li style='margin-bottom:4px;'><span class='ah-body'>"
            f"Architected an 8-agent AI orchestration pipeline (Risk Officer, Macro Strategist, "
            f"Geo Analyst, Commodities Specialist, Stress Engineer, Signal Auditor, Trade Structurer, CQO) with "
            f"a 4-round dependency-ordered execution pipeline that auto-generates morning "
            f"briefings, stress outputs, and regime-filtered trade signals</span></li>"
            f"<li style='margin-bottom:4px;'><span class='ah-body'>"
            f"Built a 6-source live intelligence layer: yfinance (50+ tickers), FRED (24 macro "
            f"series), GDELT 2.0 (conflict media escalation), EIA Open Data (petroleum inventories), "
            f"IMF PortWatch (strait vessel traffic), and ACLED (armed conflict events) - "
            f"all feeding a proactive alert engine and AI analyst context</span></li>"
            f"<li style='margin-bottom:0;'><span class='ah-body'>"
            f"Designed 20 analytical pages end-to-end: Command Center, Overview, Macro Lens, "
            f"Correlation, Spillover, Geopolitical Triggers, War Impact Map, Strait Watch, "
            f"Conflict Intelligence, Threat Monitor, Transmission Matrix, Exposure Engine, "
            f"Trade Ideas, Stress Lab, Scenario Simulator, Watchlist, "
            f"Model Audit, AI Research Desk, Methodology, and Intelligence Briefing</span></li>"
            f"</ul>"
            f"<div style='{_SROW}'>"
            f"<div style='{_SITM}background:#080808;'>"
            f"<span class='ah-num'>20</span>"
            f"<span class='ah-slbl'>Analytical Pages</span></div>"
            f"<div style='{_SITM}background:#080808;'>"
            f"<span class='ah-num'>8</span>"
            f"<span class='ah-slbl'>AI Agents</span></div>"
            f"<div style='{_SITM}background:#080808;'>"
            f"<span class='ah-num'>6</span>"
            f"<span class='ah-slbl'>Live Data Sources</span></div>"
            f"<div style='flex:1;text-align:center;padding:0.65rem 0.25rem;"
            f"background:#080808;'>"
            f"<span class='ah-num'>30K+</span>"
            f"<span class='ah-slbl'>Lines of Code</span></div>"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── Experience ────────────────────────────────────────────────────────
        st.markdown(
            f"<div style='{_SEC}'>"
            f"<span class='ah-label'>Professional Experience</span>"
            + _exp(
                "Practicum Consultant",
                "Fino Advisors LLC",
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
                "Analyst",
                "Student Managed Venture Fund, Purdue University",
                "January 2026 &ndash; Present &middot; West Lafayette, IN",
                [
                    "Build and refine financial models to evaluate early-stage deal flows each cycle, "
                    "screening for business model viability, unit economics, and operational red flags.",
                    "Present investment recommendations to professional VC panels, structuring due "
                    "diligence findings into decision-ready memos.",
                ],
            )
            + _exp(
                "Student Extern",
                "Equity Methods",
                "March 2026 &middot; Champaign, IL",
                [
                    "Used SAS to forecast stock compensation expenses for a Fortune 100 company; "
                    "refined forecasting techniques and stress-tested assumptions across grant types.",
                    "Developed client-ready decks with graphical disclosures and CAP calculations "
                    "for proxy filings; applied Pay vs. Performance (PVP) analysis.",
                ],
            )
            + _exp(
                "Equity Research Associate",
                "Axis Securities Ltd",
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
                "Equity Research Intern",
                "Axis Securities Ltd",
                "July 2024 &ndash; August 2024 &middot; Mumbai, India",
                [
                    "Supported coverage across Pharma and Hospitality (13 stocks), contributing "
                    "to thematic research, initiating coverage, and management interactions.",
                    "Converted to full-time Associate; secured a recommendation from the Head of Research.",
                ],
            )
            + _exp(
                "Undergraduate Research Assistant",
                "BITS Pilani",
                "April 2022 &ndash; May 2024 &middot; Hyderabad, India",
                [
                    "Co-designed and validated a low-cost stereotaxic device for rodent brain research; "
                    "contributed to fabrication methodology, mechanical validation, and experimental protocols.",
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
            f"<span class='ah-label'>Selected Projects</span>"
            + _exp(
                "Equity-Commodities Spillover Terminal",
                "Purdue Daniels &middot; MGMT 69000-120 (Prof. C. Zhang)",
                "",
                [
                    "Built an institutional-grade cross-asset analytics dashboard tracking spillover "
                    "dynamics across 15 equity indices, 17 commodity futures, fixed income, and FX.",
                    "Integrates DCC-GARCH regime detection, Diebold-Yilmaz FEVD connectedness, "
                    "Granger causality, and transfer entropy with a 7-agent AI orchestration pipeline.",
                ],
            )
            + _exp(
                "JGB Repricing Framework",
                "Purdue Daniels &middot; MGMT 69000-119 (Prof. C. Zhang)",
                "",
                [
                    "Built a quantitative dashboard for JGB repricing as the BOJ exits YCC, using "
                    "regime filters (MS/HMM/GARCH), yield PCA, DCC spillovers, and cross-asset "
                    "transfer entropy to auto-generate trade signals.",
                ],
            )
            + _exp(
                "PepsiCo / Vita Coco Acquisition",
                "Purdue Daniels &middot; MGMT 64500 (Prof. S. Chernenko)",
                "",
                [
                    "Built a DCF and EBIT exit-multiple valuation, quantified revenue and cost "
                    "synergies, and structured a $3.9&ndash;4.2B all-cash bid delivering 1.5% "
                    "EPS accretion.",
                ],
            )
            + _exp(
                "Eli Lilly / Madrigal Pharmaceuticals M&amp;A",
                "IU Bloomington MSF Case Competition",
                "",
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
            f"<span class='ah-label'>Education</span>"

            f"<div style='{_EDUI}'>"
            f"<span class='ah-sch' style='margin:0 0 2px;'>Purdue University</span>"
            f"<span class='ah-dept' style='margin:0 0 3px;'>"
            f"Mitchell E. Daniels, Jr. School of Business</span>"
            f"<span class='ah-deg' style='margin:0 0 2px;'>"
            f"Master of Science in Finance</span>"
            f"<span class='ah-year' style='margin:0 0 3px;'>"
            f"2025 &ndash; 2026 &middot; West Lafayette, IN</span>"
            f"<span class='ah-meta'>Investment Banking &middot; AI in Finance &middot; "
            f"Financial Modelling &middot; Venture Capital</span>"
            f"</div>"

            f"<div style='{_EDUI}border-bottom:none;'>"
            f"<span class='ah-sch' style='margin:0 0 2px;'>BITS Pilani</span>"
            f"<span class='ah-dept' style='margin:0 0 3px;'>Hyderabad Campus</span>"
            f"<span class='ah-deg' style='margin:0 0 2px;'>"
            f"B.E. (Hons.) Mechanical Engineering</span>"
            f"<span class='ah-year' style='margin:0 0 3px;'>"
            f"2020 &ndash; 2024 &middot; Hyderabad, India</span>"
            f"<span class='ah-meta'>Optimization &middot; Manufacturing &middot; "
            f"Nano Materials &middot; Thermal Energy Storage</span>"
            f"</div>"

            f"</div>",
            unsafe_allow_html=True,
        )

        # ── Publication ───────────────────────────────────────────────────────
        st.markdown(
            f"<div style='{_SEC}'>"
            f"<span class='ah-label'>Publication</span>"
            f"<div style='background:#080808;border:1px solid #1e1e1e;"
            f"border-left:2px solid rgba(207,185,145,.40);padding:0.6rem 0.8rem;'>"
            f"<span class='ah-pub-title' style='margin:0 0 5px;'>"
            f"Design, Fabrication and Validation of a Low-Cost Stereotaxic Device "
            f"for Brain Research in Rodents</span>"
            f"<span class='ah-pub-auth' style='margin:0 0 4px;'>"
            f"A. Wadkar, <strong style='color:#c8c8c8;'>H. Patkar</strong>, "
            f"S.P. Kommajosyula</span>"
            f"<span class='ah-pub-journal' style='margin:0 0 4px;'>"
            f"Bulletin of Materials Science, Vol. 48, Article 0028</span>"
            f"<span class='ah-meta' style='margin:0 0 8px;'>"
            f"Indian Academy of Sciences &middot; February 2025</span>"
            f"<a href='https://www.ias.ac.in/article/fulltext/boms/048/0028' "
            f"target='_blank' class='ah-link'>View &rarr;</a>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

        # ── Certifications ────────────────────────────────────────────────────
        st.markdown(
            f"<div style='{_SEC}'>"
            f"<span class='ah-label'>Certifications</span>"
            f"<div style='padding:0.5rem 0;border-bottom:1px solid #161616;'>"
            f"<span class='ah-cert-name' style='margin:0 0 2px;'>"
            f"NISM Series XV: Research Analyst</span>"
            f"<span class='ah-cert-issuer'>"
            f"NISM &middot; Oct 2024 &ndash; Oct 2027</span></div>"
            f"<div style='padding:0.5rem 0;'>"
            f"<span class='ah-cert-name' style='margin:0 0 2px;'>"
            f"Bloomberg Market Concepts</span>"
            f"<span class='ah-cert-issuer'>"
            f"Bloomberg LP &middot; February 2024</span></div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── Skills ────────────────────────────────────────────────────────────
        st.markdown(
            f"<div style='{_SEC}'>"
            f"<span class='ah-label'>Skills</span>"
            f"<div style='margin-top:2px;'>"
            f"<span class='ah-tag-g'>Python</span>"
            f"<span class='ah-tag-n'>Excel</span>"
            f"<span class='ah-tag-g'>Bloomberg</span>"
            f"<span class='ah-tag-n'>SAS</span>"
            f"<span class='ah-tag-g'>LSEG</span>"
            f"<span class='ah-tag-n'>ACE Equity</span>"
            f"<span class='ah-tag-g'>Claude CLI</span>"
            f"<span class='ah-tag-n'>Autodesk Suite</span>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

        # ── Interests ─────────────────────────────────────────────────────────
        st.markdown(
            f"<div style='{_SEC}'>"
            f"<span class='ah-label'>Interests</span>"
            f"<div style='margin-top:2px;'>"
            f"<span class='ah-tag-g'>Investment Banking</span>"
            f"<span class='ah-tag-n'>Corporate Finance</span>"
            f"<span class='ah-tag-g'>Valuations</span>"
            f"<span class='ah-tag-n'>Private Equity</span>"
            f"<span class='ah-tag-g'>Equity Research</span>"
            f"<span class='ah-tag-n'>Venture Capital</span>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

        # ── Acknowledgments ───────────────────────────────────────────────────
        st.markdown(
            f"<div style='{_SEC}'>"
            f"<span class='ah-label'>Acknowledgments</span>"
            f"<div style='background:#080808;border:1px solid #1e1e1e;"
            f"padding:0.55rem 0.8rem;margin-bottom:6px;'>"
            f"<span style='font-family:\"DM Sans\",sans-serif;font-size:11px;"
            f"font-weight:700;color:#e8e9ed;'>Prof. Cinder Zhang</span>"
            f"<span style='font-family:\"JetBrains Mono\",monospace;font-size:8px;"
            f"color:#8E9AAA;display:block;margin-top:2px;'>"
            f"MGMT 69000 &middot; Framework-first thinking behind regime and spillover design"
            f"</span></div>"
            f"<div style='background:#080808;border:1px solid #1e1e1e;"
            f"padding:0.55rem 0.8rem;'>"
            f"<span style='font-family:\"DM Sans\",sans-serif;font-size:11px;"
            f"font-weight:700;color:#e8e9ed;'>Prof. Adem Atmaz</span>"
            f"<span style='font-family:\"JetBrains Mono\",monospace;font-size:8px;"
            f"color:#8E9AAA;display:block;margin-top:2px;'>"
            f"MGMT 511 &middot; Fixed income intuition behind DCC-GARCH and transfer entropy"
            f"</span></div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    _page_footer()
