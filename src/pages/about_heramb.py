"""About: Heramb S. Patkar."""

from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st

from src.ui.shared import _page_footer

_ASSETS = Path(__file__).resolve().parent.parent.parent / "assets"

# ── Inline style constants (mirror sectional page values exactly) ──────────
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


def page_about_heramb() -> None:
    photo = _photo_html("photo_heramb.jpeg", "Heramb S. Patkar")

    # ── hero banner ───────────────────────────────────────────────────────────
    st.markdown(
        f"<div style='background:#141414;border-top:2px solid {_GOLD};"
        f"border-bottom:1px solid #1e1e1e;overflow:hidden;margin-bottom:1.4rem;'>"
        f"<div style='display:flex;align-items:stretch;'>"
        f"{photo}"
        f"<div style='flex:1;padding:1.1rem 1.4rem;display:flex;flex-direction:column;"
        f"justify-content:center;border-left:1px solid #1e1e1e;'>"
        f"<div style='{_S_LABEL}margin:0 0 0.32rem 0;'>About the Author</div>"
        f"<div style='{_S_NAME}'>Heramb S. Patkar</div>"
        f"<div style='{_S_SUB}'>MSF Candidate &middot; Purdue Daniels School of Business</div>"
        f"<div style='{_S_TAG}'>BITS Pilani engineering graduate and NISM XV certified research analyst "
        f"with experience across equity research, venture capital, and U.S. capital markets. "
        f"Practicum consultant, VC fund analyst, and published researcher.</div>"
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

        # Profile
        st.markdown(
            f"<div style='{_SEC}'>"
            f"<div style='{_S_LABEL}margin-bottom:0.55rem;'>Profile</div>"
            f"<div style='{_S_BODY}margin-bottom:0.45rem;'>Driven by curiosity about how businesses create impact and grow stronger. "
            f"With a background in engineering and experience in global equity research, "
            f"I enjoy analysing industries, building financial models, and uncovering insights "
            f"that drive smarter decisions. Excited by opportunities where analytical thinking "
            f"and creativity intersect to solve complex problems and deliver meaningful value.</div>"
            f"<div style='{_S_BODY}'>Beyond work, I enjoy exploring new places, listening to Carnatic music, "
            f"and learning from different cultures and perspectives. Always open to connecting, feel free to reach out.</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # This Project
        st.markdown(
            f"<div style='{_SEC}'>"
            f"<div style='{_S_LABEL}margin-bottom:0.55rem;'>This Project</div>"
            f"<div style='{_S_BODY}margin-bottom:0.40rem;'>Lead developer and quantitative architect on the "
            f"Equity-Commodities Spillover Monitor - Course Project 3 for Prof. Cinder Zhang's MGMT 69000-120.</div>"
            f"<ul style='{_S_BODY}margin:0 0 0.5rem 0;padding-left:1.0rem;'>"
            f"<li style='margin-bottom:0.30rem;'>Designed and built the full Streamlit application: multi-page architecture, "
            f"custom navbar, dark theme CSS design system, and all 14 dashboard pages</li>"
            f"<li style='margin-bottom:0.30rem;'>Implemented the quantitative engine: Diebold-Yilmaz FEVD spillover, "
            f"VAR/Granger causality, DCC-GARCH correlation, Markov regime detection, and transfer entropy</li>"
            f"<li style='margin-bottom:0.30rem;'>Built the AI agent workforce: 8 specialized agents with a 4-round "
            f"dependency-ordered orchestration pipeline, CQO quality auditing, and active remediation loop</li>"
            f"<li style='margin-bottom:0.30rem;'>Developed the data layer: yfinance multi-asset downloader, FRED API "
            f"integration (14 macro series), and real-time proactive alert engine</li>"
            f"<li style='margin-bottom:0;'>Built all 14 dashboard pages end-to-end as sole developer - "
            f"Stress Test, Scenario Engine, Strait Watch, War Impact Map, Macro Dashboard, Insights, AI Chat, and more</li>"
            f"</ul>"
            f"<div style='{_STAT_ROW}'>"
            f"<div style='{_STAT_ITM}'><div style='{_S_NUM}'>14</div><div style='{_S_SLBL}'>Pages Built</div></div>"
            f"<div style='{_STAT_ITM}'><div style='{_S_NUM}'>8</div><div style='{_S_SLBL}'>AI Agents</div></div>"
            f"<div style='{_STAT_ITM}'><div style='{_S_NUM}'>5</div><div style='{_S_SLBL}'>Quant Models</div></div>"
            f"<div style='{_STAT_ITM}border-right:none;'><div style='{_S_NUM}'>23K+</div><div style='{_S_SLBL}'>Lines of Code</div></div>"
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
            + _exp("Practicum Consultant", "Fino Advisors LLC",
                   "Jan 2026 &ndash; Present &middot; Houston, TX (Remote)",
                   "Building and refining a confidential Series A investor model (revenue drivers, scenarios/sensitivities) "
                   "and synthesising technical and financial inputs into decision-ready insights for fundraising conversations. "
                   "Conducting valuation work and comps benchmarking; supporting investor materials for a sustainability/"
                   "infrastructure portfolio company under NDA, translating assumptions into KPIs and fundraising-ready outputs.")
            + _exp("Analyst", "Student Managed Venture Fund, Purdue University",
                   "Jan 2026 &ndash; Present &middot; West Lafayette, IN",
                   "Build and refine financial models to evaluate early-stage deal flows each cycle, screening for business "
                   "model viability, unit economics, and operational red flags across Purdue-sourced startups. Present investment "
                   "recommendations to professional VC panels, structuring due diligence findings into decision-ready memos.")
            + _exp("Student Extern", "Equity Methods",
                   "Mar 2026 &middot; Champaign, IL",
                   "Used SAS to forecast stock compensation expenses for a Fortune 100 company during a two-day externship. "
                   "Developed client-ready decks with graphical disclosures and CAP calculations for proxy filings and applied "
                   "Pay vs. Performance (PVP) analysis to align exec compensation with shareholder returns.")
            + _exp("Equity Research Associate", "Axis Securities Ltd",
                   "Sep 2024 &ndash; Apr 2025 &middot; Mumbai, India",
                   "Led Auto &amp; Auto Ancillary equity research under the lead analyst, building financial models for 14 stocks, "
                   "delivering IPO notes, earnings updates, and industry outlooks to family offices and fund managers. "
                   "Pitched stock ideas and thematic plays to the PMS team (AUM ~$1 Bn) and Head of Research.")
            + _exp("Equity Research Intern", "Axis Securities Ltd",
                   "Jul 2024 &ndash; Aug 2024 &middot; Mumbai, India",
                   "Supported coverage across Pharma and Hospitality (13 stocks), contributing to thematic research, "
                   "initiating coverage, and management interactions. Converted to full-time; secured a recommendation "
                   "from the Head of Research.")
            + _exp("Undergraduate Research Assistant", "BITS Pilani",
                   "Apr 2022 &ndash; May 2024 &middot; Hyderabad, India",
                   "Co-designed and validated a low-cost stereotaxic device for rodent brain research. "
                   "Work published in the Bulletin of Materials Science (Indian Academy of Sciences, 2025).")
            + _exp("Manufacturing Engineer Intern", "Divgi TorqTransfer Systems Ltd",
                   "Jul 2023 &ndash; Dec 2023 &middot; Sirsi, Karnataka, India")
            + _exp("HVAC Engineer Intern", "Grasim Industries Limited, Pulp &amp; Fibre",
                   "May 2022 &ndash; Jul 2022 &middot; Nagda, Madhya Pradesh, India")
            + "</div>",
            unsafe_allow_html=True,
        )

        # Projects
        st.markdown(
            f"<div style='{_SEC}'>"
            f"<div style='{_S_LABEL}margin-bottom:0.55rem;'>Selected Projects</div>"
            + _exp("JGB Repricing Framework",
                   "Purdue Daniels &middot; MGMT 69000-119 (Prof. X. Zhang)", "",
                   "Built a quantitative dashboard for JGB repricing as the BOJ exits YCC, using regime filters "
                   "(MS/HMM/GARCH), yield PCA, DCC spillovers, and equity-commodities transfer entropy to auto-generate trade signals.")
            + _exp("Eli Lilly / Madrigal Pharmaceuticals M&amp;A",
                   "IU Bloomington MSF Case Competition", "",
                   "Built a patient-based DCF for Lilly's acquisition of Madrigal (~$10.4&ndash;22B), quantified synergies "
                   "against pharma M&amp;A precedents, and recommended a $640&ndash;809/share offer.")
            + _exp("PepsiCo / Vita Coco Acquisition",
                   "Purdue Daniels &middot; MGMT 64500 (Prof. S. Chernenko)", "",
                   "Built a DCF and EBIT exit-multiple valuation, quantified revenue and cost synergies, and structured "
                   "a $3.9&ndash;4.2B all-cash bid delivering 1.5% EPS accretion.")
            + _exp("Equity-Commodities Spillover Monitor",
                   "Purdue Daniels &middot; MGMT 69000-120 (Prof. X. Zhang)", "",
                   "This dashboard. Quantitative equity-commodities correlation and spillover monitor tracking geopolitical "
                   "shocks, macro regimes, and commodity supply disruptions across 15 equity indices and 17 commodity futures.")
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
            f"<div style='{_S_YEAR}'>2025 &ndash; 2026</div></div>"
            f"<div style='{_EDUI}border-bottom:none;'>"
            f"<div style='{_S_SCH}'>BITS Pilani</div>"
            f"<div style='{_S_DEPT}'>Hyderabad Campus</div>"
            f"<div style='{_S_DEG}'>B.E. (Hons.) Mechanical Engineering</div>"
            f"<div style='{_S_YEAR}'>2020 &ndash; 2024</div></div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Publication
        st.markdown(
            f"<div style='{_SEC}'>"
            f"<div style='{_S_LABEL}margin-bottom:0.55rem;'>Publication</div>"
            f"<div style='padding:0.25rem 0;'>"
            f"<div style='{_SANS}font-size:0.61rem;font-weight:600;color:#e8e9ed;margin:0 0 0.14rem 0;line-height:1.55;'>"
            f"Design, Fabrication and Validation of a Low-Cost Stereotaxic Device for Brain Research in Rodents</div>"
            f"<div style='{_SANS}font-size:0.57rem;color:#8890a1;margin:0 0 0.08rem 0;'>"
            f"A. Wadkar, <strong>H. Patkar</strong>, S.P. Kommajosyula</div>"
            f"<div style='{_SANS}font-size:0.55rem;color:{_GOLD};font-style:italic;margin:0 0 0.06rem 0;'>"
            f"Bulletin of Materials Science, Vol. 48, Article 0028</div>"
            f"<div style='{_MONO}font-size:0.50rem;color:#555960;margin:0 0 0.22rem 0;'>"
            f"Indian Academy of Sciences &middot; February 2025</div>"
            f"<a href='https://www.ias.ac.in/article/fulltext/boms/048/0028' target='_blank' "
            f"style='{_MONO}font-size:0.50rem;font-weight:700;color:{_GOLD};text-decoration:none;"
            f"text-transform:uppercase;letter-spacing:0.08em;border-bottom:1px solid rgba(207,185,145,0.3);'>View Full Text</a>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

        # Certs + Skills
        st.markdown(
            f"<div style='{_SEC}'>"
            f"<div style='{_S_LABEL}margin-bottom:0.55rem;'>Licenses &amp; Certifications</div>"
            f"<div style='padding:0.28rem 0;border-bottom:1px solid #1a1a1a;'>"
            f"<div style='{_SANS}font-size:0.60rem;font-weight:600;color:#e8e9ed;margin:0 0 0.04rem 0;'>NISM Series XV: Research Analyst</div>"
            f"<div style='{_MONO}font-size:0.50rem;color:#555960;margin:0;'>NISM &middot; Oct 2024 &ndash; Oct 2027</div></div>"
            f"<div style='padding:0.28rem 0;'>"
            f"<div style='{_SANS}font-size:0.60rem;font-weight:600;color:#e8e9ed;margin:0 0 0.04rem 0;'>Bloomberg Market Concepts</div>"
            f"<div style='{_MONO}font-size:0.50rem;color:#555960;margin:0;'>Bloomberg LP &middot; Feb 2024</div></div>"
            f"</div>"
            f"<div style='{_SEC}'>"
            f"<div style='{_S_LABEL}margin-bottom:0.45rem;'>Skills</div>"
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

        # Interests
        st.markdown(
            f"<div style='{_SEC}'>"
            f"<div style='{_S_LABEL}margin-bottom:0.45rem;'>Interests</div>"
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

        # Acknowledgments
        st.markdown(
            f"<div style='{_SEC}'>"
            f"<div style='{_S_LABEL}margin-bottom:0.50rem;'>Acknowledgments</div>"
            f"<div style='{_S_ACK}'><strong style='color:#c8c8c8;font-weight:600;'>Prof. Cinder Zhang</strong>, "
            f"MGMT 69000: Framework-first thinking behind regime and spillover design</div>"
            f"<div style='{_S_ACK}margin-bottom:0;'><strong style='color:#c8c8c8;font-weight:600;'>Prof. Adem Atmaz</strong>, "
            f"MGMT 511: Fixed income intuition behind DCC-GARCH and transfer entropy approaches</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    _page_footer()
