"""About: Heramb S. Patkar."""

from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st

from src.ui.shared import _about_page_styles, _page_footer

_ASSETS = Path(__file__).resolve().parent.parent.parent / "assets"


def _photo_html(filename: str, alt: str) -> str:
    p = _ASSETS / filename
    if not p.exists():
        return ""
    b64 = base64.b64encode(p.read_bytes()).decode()
    ext = p.suffix.lstrip(".")
    return (
        f"<div class='hero-photo'>"
        f"<img src='data:image/{ext};base64,{b64}' alt='{alt}' />"
        f"</div>"
    )


def page_about_heramb() -> None:
    _about_page_styles()
    _f = "font-family:'DM Sans',sans-serif;color:#8890a1;font-size:0.70rem;line-height:1.7;"

    photo = _photo_html("photo_heramb.jpeg", "Heramb S. Patkar")

    # ── hero banner ──
    st.markdown(
        "<div class='about-hero'><div class='about-hero-inner'>"
        f"{photo}"
        "<div class='hero-body'>"
        "<p class='overline'>About the Author</p>"
        "<h1>Heramb S. Patkar</h1>"
        "<p class='subtitle'>MSF Candidate &middot; Purdue Daniels School of Business</p>"
        "<p class='tagline'>BITS Pilani engineering graduate and NISM XV certified research analyst "
        "with experience across equity research, venture capital, and U.S. capital markets. "
        "Practicum consultant, VC fund analyst, and published researcher.</p>"
        "<div class='links'>"
        "<a href='https://www.linkedin.com/in/heramb-patkar/' target='_blank'>LinkedIn</a>"
        "<a href='https://github.com/HPATKAR' target='_blank'>GitHub</a>"
        "<a href='https://www.ias.ac.in/article/fulltext/boms/048/0028' target='_blank'>Publication</a>"
        "</div></div>"
        "</div></div>",
        unsafe_allow_html=True,
    )

    # ── two-column body ──
    col_main, col_side = st.columns([1.55, 1], gap="large")

    with col_main:
        st.markdown(
            "<div class='about-card'>"
            "<p class='about-card-title'>Profile</p>"
            f"<p style='{_f}margin:0 0 10px 0;'>"
            "Driven by curiosity about how businesses create impact and grow stronger. "
            "With a background in engineering and experience in global equity research, "
            "I enjoy analysing industries, building financial models, and uncovering insights "
            "that drive smarter decisions. Excited by opportunities where analytical thinking "
            "and creativity intersect to solve complex problems and deliver meaningful value.</p>"
            f"<p style='{_f}margin:0;'>"
            "Beyond work, I enjoy exploring new places, listening to Carnatic music, "
            "and learning from different cultures and perspectives. Always open to connecting"
            ", feel free to reach out.</p>"
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            "<div class='about-card'>"
            "<p class='about-card-title'>This Project</p>"
            f"<p style='{_f}margin:0 0 12px 0;'>"
            "Built the Equity-Commodities Spillover Monitor as Course Project 3 for "
            "Prof. Cinder Zhang's MGMT 69000-120: a quantitative dashboard that tracks equity-commodities "
            "correlation regimes, geopolitical spillover, and generates institutional-grade trade ideas "
            "across global equity and commodity markets.</p>"
            "<div class='stat-row'>"
            "<div class='stat-item'><p class='stat-num'>4</p><p class='stat-label'>Regime States</p></div>"
            "<div class='stat-item'><p class='stat-num'>10</p><p class='stat-label'>Dashboard Pages</p></div>"
            "<div class='stat-item'><p class='stat-num'>6</p><p class='stat-label'>Trade Ideas</p></div>"
            "<div class='stat-item'><p class='stat-num'>5</p><p class='stat-label'>Analytical Layers</p></div>"
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            "<div class='about-card'>"
            "<p class='about-card-title'>Experience</p>"

            "<div class='exp-item'>"
            "<p class='exp-role'>Practicum Consultant</p>"
            "<p class='exp-org'>Fino Advisors LLC</p>"
            "<p class='exp-meta'>Jan 2026 &ndash; Present &middot; Houston, TX (Remote)</p>"
            "<p class='exp-desc'>Building and refining a confidential Series A investor model "
            "(revenue drivers, scenarios/sensitivities) and synthesising technical and financial "
            "inputs into decision-ready insights for fundraising conversations. Conducting valuation "
            "work and comps benchmarking; supporting investor materials for a sustainability/"
            "infrastructure portfolio company under NDA, translating assumptions into KPIs and "
            "fundraising-ready outputs.</p></div>"

            "<div class='exp-item'>"
            "<p class='exp-role'>Analyst</p>"
            "<p class='exp-org'>Student Managed Venture Fund, Purdue University</p>"
            "<p class='exp-meta'>Jan 2026 &ndash; Present &middot; West Lafayette, IN</p>"
            "<p class='exp-desc'>Build and refine financial models to evaluate early-stage deal "
            "flows each cycle, screening for business model viability, unit economics, and "
            "operational red flags across Purdue-sourced startups. Present investment "
            "recommendations to professional VC panels, structuring due diligence findings into "
            "decision-ready memos and reducing diligence cycle time through systematic early-stage "
            "risk filtering.</p></div>"

            "<div class='exp-item'>"
            "<p class='exp-role'>Student Extern</p>"
            "<p class='exp-org'>Equity Methods</p>"
            "<p class='exp-meta'>Mar 2026 &middot; Champaign, IL</p>"
            "<p class='exp-desc'>Used SAS to forecast stock compensation expenses for a Fortune 100 "
            "company during a two-day externship, refining forecasting techniques, improving "
            "projection accuracy, and stress-testing assumptions across grant types. Developed "
            "client-ready decks with graphical disclosures and CAP calculations for proxy filings "
            "and applied Pay vs. Performance (PVP) analysis to align exec compensation with "
            "shareholder returns and business outcomes.</p></div>"

            "<div class='exp-item'>"
            "<p class='exp-role'>Equity Research Associate</p>"
            "<p class='exp-org'>Axis Securities Ltd</p>"
            "<p class='exp-meta'>Sep 2024 &ndash; Apr 2025 &middot; Mumbai, India</p>"
            "<p class='exp-desc'>Led Auto &amp; Auto Ancillary equity research under the lead "
            "analyst, building financial models for 14 stocks, delivering IPO notes, earnings "
            "updates, and industry outlooks to family offices and fund managers. Owned a dynamic "
            "industry tracker (production, sales, inventory) across segments and engine classes. "
            "Pitched stock ideas and thematic plays to the PMS team (AUM ~$1 Bn) and Head of "
            "Research, engaging senior management on pricing, margins, capex, and tariffs.</p></div>"

            "<div class='exp-item'>"
            "<p class='exp-role'>Equity Research Intern</p>"
            "<p class='exp-org'>Axis Securities Ltd</p>"
            "<p class='exp-meta'>Jul 2024 &ndash; Aug 2024 &middot; Mumbai, India</p>"
            "<p class='exp-desc'>Supported coverage across Pharma and Hospitality (13 stocks), "
            "contributing to thematic research, initiating coverage, and management interactions. "
            "Converted to full-time; secured a recommendation from the Head of Research.</p></div>"

            "<div class='exp-item'>"
            "<p class='exp-role'>Undergraduate Research Assistant</p>"
            "<p class='exp-org'>BITS Pilani</p>"
            "<p class='exp-meta'>Apr 2022 &ndash; May 2024 &middot; Hyderabad, India</p>"
            "<p class='exp-desc'>Co-designed and validated a low-cost stereotaxic device for "
            "rodent brain research. Work published in the Bulletin of Materials Science "
            "(Indian Academy of Sciences, 2025).</p></div>"

            "<div class='exp-item'>"
            "<p class='exp-role'>Manufacturing Engineer Intern</p>"
            "<p class='exp-org'>Divgi TorqTransfer Systems Ltd</p>"
            "<p class='exp-meta'>Jul 2023 &ndash; Dec 2023 &middot; Sirsi, Karnataka, India</p></div>"

            "<div class='exp-item'>"
            "<p class='exp-role'>HVAC Engineer Intern</p>"
            "<p class='exp-org'>Grasim Industries Limited, Pulp &amp; Fibre</p>"
            "<p class='exp-meta'>May 2022 &ndash; Jul 2022 &middot; Nagda, Madhya Pradesh, India</p></div>"

            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            "<div class='about-card'>"
            "<p class='about-card-title'>Selected Projects</p>"

            "<div class='exp-item'>"
            "<p class='exp-role'>JGB Repricing Framework</p>"
            "<p class='exp-org'>Purdue Daniels &middot; MGMT 69000-119 (Prof. X. Zhang)</p>"
            "<p class='exp-desc'>Built a quantitative dashboard for JGB repricing as the BOJ "
            "exits YCC, using regime filters (MS/HMM/GARCH), yield PCA, DCC spillovers, and "
            "equity-commodities transfer entropy to auto-generate trade signals.</p></div>"

            "<div class='exp-item'>"
            "<p class='exp-role'>Eli Lilly / Madrigal Pharmaceuticals M&amp;A</p>"
            "<p class='exp-org'>IU Bloomington MSF Case Competition</p>"
            "<p class='exp-desc'>Built a patient-based DCF for Lilly's acquisition of Madrigal "
            "(~$10.4&ndash;22B), quantified synergies against pharma M&amp;A precedents, and "
            "recommended a $640&ndash;809/share offer.</p></div>"

            "<div class='exp-item'>"
            "<p class='exp-role'>PepsiCo / Vita Coco Acquisition</p>"
            "<p class='exp-org'>Purdue Daniels &middot; MGMT 64500 (Prof. S. Chernenko)</p>"
            "<p class='exp-desc'>Built a DCF and EBIT exit-multiple valuation, quantified revenue "
            "and cost synergies, and structured a $3.9&ndash;4.2B all-cash bid delivering 1.5% "
            "EPS accretion for a proposed acquisition of Vita Coco by PepsiCo.</p></div>"

            "<div class='exp-item'>"
            "<p class='exp-role'>Equity-Commodities Spillover Monitor</p>"
            "<p class='exp-org'>Purdue Daniels &middot; MGMT 69000-120 (Prof. X. Zhang)</p>"
            "<p class='exp-desc'>This dashboard. Quantitative equity-commodities correlation and spillover "
            "monitor tracking geopolitical shocks, macro regimes, and commodity supply disruptions "
            "across 15 equity indices and 17 commodity futures.</p></div>"

            "</div>",
            unsafe_allow_html=True,
        )

    with col_side:
        st.markdown(
            "<div class='about-card'>"
            "<p class='about-card-title'>Education</p>"
            "<div class='edu-item'>"
            "<p class='edu-school'>Purdue University</p>"
            "<p class='edu-dept'>Mitchell E. Daniels, Jr. School of Business</p>"
            "<p class='edu-degree'>Master of Science in Finance</p>"
            "<p class='edu-year'>2025 &ndash; 2026</p></div>"
            "<div class='edu-item'>"
            "<p class='edu-school'>BITS Pilani</p>"
            "<p class='edu-dept'>Hyderabad Campus</p>"
            "<p class='edu-degree'>B.E. (Hons.) Mechanical Engineering</p>"
            "<p class='edu-year'>2020 &ndash; 2024</p></div>"
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            "<div class='about-card'>"
            "<p class='about-card-title'>Publication</p>"
            "<div class='pub-item'>"
            "<p class='pub-title'>Design, Fabrication and Validation of a Low-Cost "
            "Stereotaxic Device for Brain Research in Rodents</p>"
            "<p class='pub-authors'>A. Wadkar, <strong>H. Patkar</strong>, "
            "S.P. Kommajosyula</p>"
            "<p class='pub-journal'>Bulletin of Materials Science, Vol. 48, "
            "Article 0028</p>"
            "<p class='pub-detail'>Indian Academy of Sciences &middot; February 2025</p>"
            "<a class='pub-link' href='https://www.ias.ac.in/article/fulltext/boms/048/0028' "
            "target='_blank'>View Full Text</a>"
            "</div></div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            "<div class='about-card'>"
            "<p class='about-card-title'>Licenses &amp; Certifications</p>"
            "<div class='cert-item'>"
            "<p class='cert-name'>NISM Series XV: Research Analyst</p>"
            "<p class='cert-issuer'>NISM &middot; Oct 2024 &ndash; Oct 2027</p></div>"
            "<div class='cert-item'>"
            "<p class='cert-name'>Bloomberg Market Concepts</p>"
            "<p class='cert-issuer'>Bloomberg LP &middot; Feb 2024</p></div>"
            "</div>"
            "<div class='about-card'>"
            "<p class='about-card-title'>Skills</p>"
            "<div>"
            "<span class='interest-tag interest-gold'>Python</span>"
            "<span class='interest-tag interest-neutral'>Excel</span>"
            "<span class='interest-tag interest-gold'>Bloomberg</span>"
            "<span class='interest-tag interest-neutral'>SAS</span>"
            "<span class='interest-tag interest-gold'>LSEG</span>"
            "<span class='interest-tag interest-neutral'>ACE Equity</span>"
            "<span class='interest-tag interest-gold'>Claude CLI</span>"
            "<span class='interest-tag interest-neutral'>Autodesk Suite</span>"
            "</div></div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            "<div class='about-card'>"
            "<p class='about-card-title'>Interests</p>"
            "<div>"
            "<span class='interest-tag interest-gold'>Investment Banking</span>"
            "<span class='interest-tag interest-neutral'>Corporate Finance</span>"
            "<span class='interest-tag interest-gold'>Valuations</span>"
            "<span class='interest-tag interest-neutral'>Private Equity</span>"
            "<span class='interest-tag interest-gold'>Equity Research</span>"
            "<span class='interest-tag interest-neutral'>Venture Capital</span>"
            "</div></div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            "<div class='about-card'>"
            "<p class='about-card-title'>Acknowledgments</p>"
            "<p class='ack-text'><strong>Prof. Cinder Zhang</strong>, MGMT 69000: "
            "Framework-first thinking behind regime and spillover design</p>"
            "<p class='ack-text' style='margin-top:8px;'><strong>Prof. Adem Atmaz</strong>, "
            "MGMT 511: Fixed income intuition behind DCC-GARCH and transfer entropy approaches</p>"
            "</div>",
            unsafe_allow_html=True,
        )

    _page_footer()
