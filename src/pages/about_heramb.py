"""About: Heramb S. Patkar."""

from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st

from src.ui.shared import _page_footer

_ASSETS = Path(__file__).resolve().parent.parent.parent / "assets"

# ── Inline style constants ─────────────────────────────────────────────────
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
_S_COUR  = f"{_MONO}font-size:0.43rem;color:#555960;margin:0 0 0.03rem 0;letter-spacing:0.02em;"
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
    """Render one experience entry with optional bullet list."""
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


def page_about_heramb() -> None:
    photo = _photo_html("photo_heramb.jpeg", "Heramb S. Patkar")

    # ── Hero ─────────────────────────────────────────────────────────────────
    st.markdown(
        f"<div style='background:#141414;border-top:2px solid {_GOLD};"
        f"border-bottom:1px solid #1e1e1e;overflow:hidden;margin-bottom:1.2rem;'>"
        f"<div style='display:flex;align-items:stretch;'>"
        f"{photo}"
        f"<div style='flex:1;padding:1.0rem 1.3rem;display:flex;flex-direction:column;"
        f"justify-content:center;border-left:1px solid #1e1e1e;'>"
        f"<div style='{_S_LABEL}margin:0 0 0.28rem 0;'>About the Author</div>"
        f"<div style='{_S_NAME}'>Heramb S. Patkar</div>"
        f"<div style='{_S_SUB}'>MSF Candidate &middot; Purdue Daniels School of Business &middot; "
        f"hpatkar@purdue.edu &middot; West Lafayette, IN</div>"
        f"<div style='{_S_TGLN}'>BITS Pilani mechanical engineering graduate and NISM XV certified research analyst "
        f"with experience across equity research, venture capital, and U.S. capital markets. "
        f"Practicum consultant at a Houston advisory firm, VC fund analyst at Purdue's student-managed fund, "
        f"and published researcher (Indian Academy of Sciences, 2025).</div>"
        f"<div style='display:flex;flex-wrap:wrap;gap:0;'>"
        f"<a href='https://www.linkedin.com/in/heramb-patkar/' target='_blank' style='{_LINK}'>LinkedIn</a>"
        f"<a href='https://github.com/HPATKAR' target='_blank' style='{_LINK}'>GitHub</a>"
        f"<a href='https://www.ias.ac.in/article/fulltext/boms/048/0028' target='_blank' style='{_LINK}'>Publication</a>"
        f"</div>"
        f"</div></div></div>",
        unsafe_allow_html=True,
    )

    col_main, col_side = st.columns([1.55, 1], gap="large")

    with col_main:

        # ── Profile ───────────────────────────────────────────────────────────
        st.markdown(
            f"<div style='{_SEC}'>"
            f"<div style='{_S_LABEL}margin-bottom:0.50rem;'>Profile</div>"
            f"<div style='{_S_BODY}margin-bottom:0.38rem;'>Driven by curiosity about how businesses create value and grow stronger. "
            f"With a foundation in engineering and hands-on experience in global equity research, I enjoy building financial models, "
            f"analysing industries, and synthesising data into decisions. Excited by intersections of analytical thinking and "
            f"markets - particularly where quantitative methods meet real-world capital allocation.</div>"
            f"<div style='{_S_BODY}'>Outside of work and markets, I enjoy exploring new places, listening to Carnatic music, "
            f"and learning from different cultures. Always open to connecting - feel free to reach out.</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── This Project ──────────────────────────────────────────────────────
        st.markdown(
            f"<div style='{_SEC}'>"
            f"<div style='{_S_LABEL}margin-bottom:0.50rem;'>This Project</div>"
            f"<div style='{_S_BODY}margin-bottom:0.35rem;'>Lead developer and sole quantitative architect on the "
            f"Equity-Commodities Spillover Monitor - Course Project 3 for Prof. Cinder Zhang's MGMT 69000-120 "
            f"(AI for Finance). Responsible for the entire codebase, architecture, and all 14 dashboard pages.</div>"
            f"<ul style='{_S_BODY}margin:0 0 0.45rem 0;padding-left:0.9rem;'>"
            f"<li style='margin-bottom:0.22rem;'>Built an institutional-grade cross-asset analytics dashboard tracking "
            f"spillover dynamics across 15 equity indices, 17 commodity futures, fixed income, and FX</li>"
            f"<li style='margin-bottom:0.22rem;'>Implemented the full quantitative engine: DCC-GARCH regime detection, "
            f"Diebold-Yilmaz FEVD connectedness, Granger causality, Hidden Markov Model regime detection, and transfer entropy</li>"
            f"<li style='margin-bottom:0.22rem;'>Architected a 7-agent AI orchestration pipeline (Risk Officer, Macro Strategist, "
            f"Geo Analyst, Commodities Specialist, Stress Engineer, Trade Structurer, CQO) with a 4-round dependency-ordered "
            f"execution pipeline that auto-generates morning briefings, stress outputs, and regime-filtered trade signals</li>"
            f"<li style='margin-bottom:0.22rem;'>Built the full data layer: yfinance multi-asset downloader (50+ tickers), "
            f"FRED API integration (14 macro series), real-time implied vol scraping (VIX/OVX/GVZ/VVIX), and proactive alert engine</li>"
            f"<li style='margin-bottom:0;'>Designed all 14 pages end-to-end: Overview, Correlation, Spillover, Macro Intelligence, "
            f"Geopolitical Triggers, War Impact Map, Strait Watch, Trade Ideas, Portfolio Stress Test, Scenario Engine, "
            f"Commodities Watchlist, Model Accuracy, Actionable Insights, and AI Chat</li>"
            f"</ul>"
            f"<div style='{_STAT_ROW}'>"
            f"<div style='{_STAT_ITM}'><div style='{_S_NUM}'>14</div><div style='{_S_SLBL}'>Pages Built</div></div>"
            f"<div style='{_STAT_ITM}'><div style='{_S_NUM}'>7</div><div style='{_S_SLBL}'>AI Agents</div></div>"
            f"<div style='{_STAT_ITM}'><div style='{_S_NUM}'>5</div><div style='{_S_SLBL}'>Quant Models</div></div>"
            f"<div style='{_STAT_ITM}border-right:none;'><div style='{_S_NUM}'>23K+</div><div style='{_S_SLBL}'>Lines of Code</div></div>"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── Experience ────────────────────────────────────────────────────────
        st.markdown(
            f"<div style='{_SEC}'>"
            f"<div style='{_S_LABEL}margin-bottom:0.50rem;'>Professional Experience</div>"
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
            f"<div style='{_S_LABEL}margin-bottom:0.50rem;'>Selected Projects</div>"
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
            f"<div style='{_S_LABEL}margin-bottom:0.50rem;'>Education</div>"

            f"<div style='{_EDUI}'>"
            f"<div style='{_S_SCH}'>Purdue University</div>"
            f"<div style='{_S_DEPT}'>Mitchell E. Daniels, Jr. School of Business</div>"
            f"<div style='{_S_DEG}'>Master of Science in Finance</div>"
            f"<div style='{_S_YEAR}margin-bottom:0.04rem;'>2025 &ndash; 2026 &middot; West Lafayette, IN</div>"
            f"<div style='{_S_COUR}'>Courses: Investment Banking &middot; AI in Finance &middot; "
            f"Financial Modelling &middot; Venture Capital</div>"
            f"</div>"

            f"<div style='{_EDUI}border-bottom:none;'>"
            f"<div style='{_S_SCH}'>BITS Pilani</div>"
            f"<div style='{_S_DEPT}'>Hyderabad Campus</div>"
            f"<div style='{_S_DEG}'>B.E. (Hons.) Mechanical Engineering</div>"
            f"<div style='{_S_YEAR}margin-bottom:0.04rem;'>2020 &ndash; 2024 &middot; Hyderabad, India</div>"
            f"<div style='{_S_COUR}'>Projects: Optimization &middot; Manufacturing &middot; "
            f"Nano Materials &middot; Thermal Energy Storage</div>"
            f"</div>"

            f"</div>",
            unsafe_allow_html=True,
        )

        # ── Publication ───────────────────────────────────────────────────────
        st.markdown(
            f"<div style='{_SEC}'>"
            f"<div style='{_S_LABEL}margin-bottom:0.50rem;'>Publication</div>"
            f"<div style='padding:0.20rem 0;'>"
            f"<div style='{_SANS}font-size:0.56rem;font-weight:600;color:#e8e9ed;"
            f"margin:0 0 0.12rem 0;line-height:1.55;'>"
            f"Design, Fabrication and Validation of a Low-Cost Stereotaxic Device "
            f"for Brain Research in Rodents</div>"
            f"<div style='{_SANS}font-size:0.52rem;color:#8890a1;margin:0 0 0.06rem 0;'>"
            f"A. Wadkar, <strong style='color:#c8c8c8;'>H. Patkar</strong>, S.P. Kommajosyula</div>"
            f"<div style='{_SANS}font-size:0.51rem;color:{_GOLD};font-style:italic;margin:0 0 0.05rem 0;'>"
            f"Bulletin of Materials Science, Vol. 48, Article 0028</div>"
            f"<div style='{_MONO}font-size:0.44rem;color:#555960;margin:0 0 0.20rem 0;'>"
            f"Indian Academy of Sciences &middot; February 2025</div>"
            f"<a href='https://www.ias.ac.in/article/fulltext/boms/048/0028' target='_blank' "
            f"style='{_MONO}font-size:0.44rem;font-weight:700;color:{_GOLD};text-decoration:none;"
            f"text-transform:uppercase;letter-spacing:0.08em;border-bottom:1px solid rgba(207,185,145,0.3);'>"
            f"View Full Text</a>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

        # ── Certifications ────────────────────────────────────────────────────
        st.markdown(
            f"<div style='{_SEC}'>"
            f"<div style='{_S_LABEL}margin-bottom:0.50rem;'>Certifications</div>"
            f"<div style='padding:0.24rem 0;border-bottom:1px solid #1a1a1a;'>"
            f"<div style='{_SANS}font-size:0.56rem;font-weight:600;color:#e8e9ed;margin:0 0 0.03rem 0;'>"
            f"NISM Series XV: Research Analyst</div>"
            f"<div style='{_MONO}font-size:0.44rem;color:#555960;margin:0;'>"
            f"NISM &middot; Oct 2024 &ndash; Oct 2027</div></div>"
            f"<div style='padding:0.24rem 0;'>"
            f"<div style='{_SANS}font-size:0.56rem;font-weight:600;color:#e8e9ed;margin:0 0 0.03rem 0;'>"
            f"Bloomberg Market Concepts</div>"
            f"<div style='{_MONO}font-size:0.44rem;color:#555960;margin:0;'>"
            f"Bloomberg LP &middot; February 2024</div></div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── Skills ────────────────────────────────────────────────────────────
        st.markdown(
            f"<div style='{_SEC}'>"
            f"<div style='{_S_LABEL}margin-bottom:0.42rem;'>Skills</div>"
            f"<div>"
            f"<span style='{_TAG_G}'>Python</span>"
            f"<span style='{_TAG_N}'>Excel</span>"
            f"<span style='{_TAG_G}'>Bloomberg</span>"
            f"<span style='{_TAG_N}'>SAS</span>"
            f"<span style='{_TAG_G}'>LSEG</span>"
            f"<span style='{_TAG_N}'>ACE Equity</span>"
            f"<span style='{_TAG_G}'>Claude CLI</span>"
            f"<span style='{_TAG_N}'>Autodesk Suite</span>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

        # ── Interests ─────────────────────────────────────────────────────────
        st.markdown(
            f"<div style='{_SEC}'>"
            f"<div style='{_S_LABEL}margin-bottom:0.42rem;'>Interests</div>"
            f"<div>"
            f"<span style='{_TAG_G}'>Investment Banking</span>"
            f"<span style='{_TAG_N}'>Corporate Finance</span>"
            f"<span style='{_TAG_G}'>Valuations</span>"
            f"<span style='{_TAG_N}'>Private Equity</span>"
            f"<span style='{_TAG_G}'>Equity Research</span>"
            f"<span style='{_TAG_N}'>Venture Capital</span>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

        # ── Acknowledgments ───────────────────────────────────────────────────
        st.markdown(
            f"<div style='{_SEC}'>"
            f"<div style='{_S_LABEL}margin-bottom:0.46rem;'>Acknowledgments</div>"
            f"<div style='{_S_ACK}'><strong style='color:#c8c8c8;font-weight:600;'>Prof. Cinder Zhang</strong> "
            f"&middot; MGMT 69000: Framework-first thinking behind regime and spillover design</div>"
            f"<div style='{_S_ACK}margin-bottom:0;'><strong style='color:#c8c8c8;font-weight:600;'>Prof. Adem Atmaz</strong> "
            f"&middot; MGMT 511: Fixed income intuition behind DCC-GARCH and transfer entropy approaches</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    _page_footer()
