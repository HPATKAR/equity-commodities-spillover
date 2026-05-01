"""
Generate Individual Contribution PDF for MGMT 69000-120 Week 7 submission.

Usage:
    python generate_contribution_pdf.py
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, HRFlowable,
    PageBreak, Table, TableStyle, KeepTogether,
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

W, H = A4
OUTPUT = "Spillover_Monitor_Individual_Contribution.pdf"

# ── Palette ───────────────────────────────────────────────────────────────────
NAVY  = colors.HexColor("#1E3A5F")
GOLD  = colors.HexColor("#CFB991")   # Purdue gold
BLACK = colors.HexColor("#0D0D1A")
BODY  = colors.HexColor("#2D2D2D")
MUTED = colors.HexColor("#666666")
LIGHT = colors.HexColor("#F5F4F0")
WHITE = colors.white
RULE  = colors.HexColor("#CFB991")
GREEN = colors.HexColor("#2E7D52")
RED   = colors.HexColor("#C0392B")

LM = 25 * mm
RM = 25 * mm
TM = 22 * mm
BM = 22 * mm

REPO = "https://github.com/HPATKAR/equity-commodities-spillover"
COMMITS_URL = f"{REPO}/commits/main"


# ── Styles ────────────────────────────────────────────────────────────────────
def S():
    h1 = ParagraphStyle("H1", fontName="Helvetica-Bold",
                        fontSize=15, textColor=NAVY,
                        spaceBefore=20, spaceAfter=6, leading=20)
    h2 = ParagraphStyle("H2", fontName="Helvetica-Bold",
                        fontSize=11, textColor=NAVY,
                        spaceBefore=14, spaceAfter=4, leading=15)
    h3 = ParagraphStyle("H3", fontName="Helvetica-Bold",
                        fontSize=9.5, textColor=BODY,
                        spaceBefore=10, spaceAfter=3, leading=14)
    body = ParagraphStyle("Body", fontName="Helvetica",
                          fontSize=9, textColor=BODY,
                          leading=15, spaceAfter=6,
                          alignment=TA_JUSTIFY)
    bullet = ParagraphStyle("Bullet", fontName="Helvetica",
                            fontSize=9, textColor=BODY,
                            leading=14, spaceAfter=3,
                            leftIndent=14, firstLineIndent=0)
    mono = ParagraphStyle("Mono", fontName="Courier",
                          fontSize=8.5, textColor=BODY,
                          leading=13, spaceAfter=3,
                          leftIndent=14)
    center = ParagraphStyle("Center", fontName="Helvetica",
                            fontSize=9, textColor=MUTED,
                            leading=14, alignment=TA_CENTER)
    label = ParagraphStyle("Label", fontName="Helvetica",
                           fontSize=7.5, textColor=GOLD,
                           spaceBefore=0, spaceAfter=0)
    url = ParagraphStyle("URL", fontName="Helvetica",
                         fontSize=8.5, textColor=colors.HexColor("#2563EB"),
                         leading=13, spaceAfter=4)
    return h1, h2, h3, body, bullet, mono, center, label, url


# ── Helpers ───────────────────────────────────────────────────────────────────
def rule(color=RULE, thickness=0.8, spaceB=4, spaceA=8):
    return HRFlowable(width="100%", thickness=thickness,
                      color=color, spaceAfter=spaceA, spaceBefore=spaceB)


def tbl(headers, rows, col_widths, h1, h2, h3, body, bullet, mono, center, label, url):
    data = [[Paragraph(f'<b>{h}</b>', label) for h in headers]]
    for row in rows:
        data.append([Paragraph(str(c), mono if i == 0 and len(str(c)) < 12 else body)
                     for i, c in enumerate(row)])
    t = Table(data, colWidths=[w * mm for w in col_widths], repeatRows=1)
    row_count = len(data)
    ts = TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0),  NAVY),
        ("TEXTCOLOR",   (0, 0), (-1, 0),  GOLD),
        ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, 0),  8),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 5),
        ("TOPPADDING",  (0, 0), (-1, 0),  5),
        ("FONTNAME",    (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",    (0, 1), (-1, -1), 8.5),
        ("TOPPADDING",  (0, 1), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",(0, 0), (-1, -1), 6),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT, WHITE]),
        ("GRID",        (0, 0), (-1, -1), 0.4, colors.HexColor("#DDDDDD")),
        ("VALIGN",      (0, 0), (-1, -1), "TOP"),
    ])
    t.setStyle(ts)
    return t


# ── Cover ──────────────────────────────────────────────────────────────────────
def cover(story, h1, h2, h3, body, bullet, mono, center, label, url):
    story.append(Spacer(1, 30 * mm))

    story.append(Paragraph("Individual Contribution Summary", ParagraphStyle(
        "Cover1", fontName="Helvetica-Bold", fontSize=22,
        textColor=NAVY, alignment=TA_CENTER, spaceAfter=6)))

    story.append(Paragraph("Equity-Commodities Spillover Monitor", ParagraphStyle(
        "Cover2", fontName="Helvetica", fontSize=14,
        textColor=colors.HexColor("#8E6F3E"), alignment=TA_CENTER, spaceAfter=4)))

    story.append(Spacer(1, 6 * mm))
    story.append(rule(GOLD, 1.5))
    story.append(Spacer(1, 4 * mm))

    story.append(Paragraph("Heramb S. Patkar", ParagraphStyle(
        "Cover3", fontName="Helvetica-Bold", fontSize=13,
        textColor=BODY, alignment=TA_CENTER, spaceAfter=3)))

    story.append(Paragraph(
        "MSF, Financial Analytics Track  ·  Purdue University Daniels School of Business",
        ParagraphStyle("Cover4", fontName="Helvetica", fontSize=10,
                       textColor=MUTED, alignment=TA_CENTER, spaceAfter=3)))

    story.append(Paragraph(
        "MGMT 69000-120: AI for Finance  ·  Prof. Cinder Zhang  ·  April 30, 2026",
        center))

    story.append(Spacer(1, 8 * mm))
    story.append(Paragraph(
        f'<link href="{REPO}" color="#2563EB">{REPO}</link>',
        ParagraphStyle("CoverURL", fontName="Helvetica", fontSize=9,
                       textColor=colors.HexColor("#2563EB"), alignment=TA_CENTER)))

    story.append(PageBreak())


# ── Main content ───────────────────────────────────────────────────────────────
def content(story, h1, h2, h3, body, bullet, mono, center, label, url):
    sty = (h1, h2, h3, body, bullet, mono, center, label, url)

    # ── Overview ──────────────────────────────────────────────────────────────
    story.append(Paragraph("Overview", h1))
    story.append(rule())
    story.append(Paragraph(
        "I was the sole developer of the Equity-Commodities Spillover Monitor — a multi-page "
        "institutional dashboard with live data integration, five quantitative analytics models, "
        "a seven-agent AI pipeline, and a 56-case out-of-sample benchmark harness. All "
        "architecture decisions, all code, all deployment infrastructure, and all submission "
        "documents were built and committed by me across 177 commits from March 12 to "
        "April 30, 2026.", body))

    story.append(Spacer(1, 2 * mm))
    story.append(tbl(
        ["Metric", "Value"],
        [
            ("Total commits", "177"),
            ("Date range", "2026-03-12 → 2026-04-30"),
            ("Substantive commits (features, fixes, methodology)", "111+"),
            ("Dashboard pages built", "22"),
            ("Python files in src/", "60+"),
            ("Lines of Python", "~23,000"),
        ],
        [105, 50],
        *sty,
    ))

    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph("Full commit history:", body))
    story.append(Paragraph(
        f'<link href="{COMMITS_URL}" color="#2563EB">{COMMITS_URL}</link>', url))

    # ── Components owned ──────────────────────────────────────────────────────
    story.append(Paragraph("Components I Owned End-to-End", h1))
    story.append(rule())

    story.append(Paragraph("Full-Stack Application", h2))
    story.append(Paragraph(
        "Designed and built the complete multi-page Streamlit application from scratch: "
        "custom fixed navbar (iframe-based), institutional dark-theme CSS design system, "
        "all routing logic, and all 22 dashboard pages.", body))

    story.append(Paragraph("Quantitative Analytics Engine — 5 Independent Models", h2))
    items = [
        ("<b>Diebold-Yilmaz FEVD</b> — rolling VAR with BIC-optimal lag selection, 10-step "
         "generalized FEVD, pairwise spillover matrix, net directional spillover, rolling "
         "Total Spillover Index. Design choice: generalized (not Cholesky) decomposition "
         "eliminates ordering dependence."),
        ("<b>Granger Causality</b> — BIC-optimal lag via VAR(ic='bic') before testing; "
         "Holm-Bonferroni step-down correction across the full N×M×2 test grid."),
        ("<b>Transfer Entropy</b> (Schreiber 2000) — non-parametric directed information "
         "measure; lag optimized over lags 1–7; 200-permutation shuffle significance test."),
        ("<b>DCC-GARCH</b> (Engle 2002) — two-step: EWMA pre-whitening (λ=0.94) before "
         "DCC(1,1) recursion. Raw-return DCC produces biased correlations at high vol — "
         "pre-whitening is the critical design choice."),
        ("<b>Composite Risk Score (GRS)</b> — three-layer architecture: "
         "GRS = 40% CIS + 35% TPS + 25% MCS. HHI breadth multiplier (n_eff^0.25) prevents "
         "single-conflict concentration from inflating the composite. 3-state correlation "
         "regime with hysteresis and persistence gate."),
    ]
    for item in items:
        story.append(Paragraph(f"• {item}", bullet))

    story.append(Paragraph("AI Agent Workforce — 7 Agents, 4-Round Pipeline", h2))
    story.append(tbl(
        ["Round", "Agent", "Role"],
        [
            ("1 (parallel)", "macro_strategist", "Macro regime, yield curve, Fed outlook"),
            ("1 (parallel)", "geopolitical_analyst", "Active conflict summary, CIS/TPS attribution"),
            ("1 (parallel)", "commodities_specialist", "Supply/demand dynamics per commodity"),
            ("2 (sequential)", "risk_officer", "Synthesizes Round 1 → composite risk narrative + 3 risk flags"),
            ("3 (parallel)", "stress_engineer", "Worst-case scenario, tail risk quantification"),
            ("3 (parallel)", "signal_auditor", "Data quality flags, stale data warnings, model limitations"),
            ("4 (sequential)", "trade_structurer", "3 regime-conditioned trade ideas with entry/exit/rationale"),
            ("4 (sequential)", "quality_officer (CQO)", "Schema validation; auto-routes failures to remediation loop"),
        ],
        [25, 38, 97],
        *sty,
    ))
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph(
        "The CQO remediation loop routes quality flag failures back to the relevant specialist "
        "agent for active correction (up to 2 rounds). Pydantic-validated structured outputs "
        "enforce schema at every agent boundary. Dual-backend: Claude Sonnet and GPT-4o.", body))

    story.append(Paragraph("Live Data Infrastructure — 6 Sources", h2))
    story.append(tbl(
        ["Source", "Data", "Cache TTL"],
        [
            ("Yahoo Finance", "50+ tickers: equities, commodities, FX, vol indices", "900s"),
            ("FRED API", "24 macro series: yield curve, CPI, Fed Funds, GDP, PMI", "86400s"),
            ("GDELT 2.0", "News volume and tone for 6 active conflicts", "7200s"),
            ("ACLED", "Georeferenced conflict events and fatalities", "21600s"),
            ("IMF PortWatch", "AIS vessel traffic at Hormuz + 5 chokepoints", "86400s"),
            ("EIA", "Weekly US petroleum inventory reports", "86400s"),
        ],
        [32, 100, 28],
        *sty,
    ))

    story.append(Paragraph("Out-of-Sample Benchmark Harness", h2))
    story.append(Paragraph(
        "56 test cases across 15 historical market events (2008–2024). Pass threshold of 60% "
        "set at project outset, not retroactively. Results documented with explicit failure "
        "mode analysis in evals/results-2026-04.md.", body))
    story.append(tbl(
        ["Agent", "Cases", "Passed", "Hit Rate", "Gate"],
        [
            ("risk_officer", "30", "13", "43.3%", "❌"),
            ("macro_strategist", "11", "7", "63.6%", "✅"),
            ("geopolitical_analyst", "11", "0", "0.0%*", "❌"),
            ("commodities_specialist", "3", "1", "33.3%", "❌"),
            ("signal_auditor", "4", "0", "0.0%", "❌"),
            ("stress_engineer", "1", "1", "100.0%", "✅"),
        ],
        [50, 18, 18, 22, 52],
        *sty,
    ))
    story.append(Paragraph(
        "* geopolitical_analyst failures are all None actual values — GDELT requires live "
        "access and cannot be replayed for historical dates. Documented as infrastructure "
        "limitation, not a model failure.", ParagraphStyle(
            "Note", fontName="Helvetica-Oblique", fontSize=8,
            textColor=MUTED, leading=12, spaceAfter=4)))

    story.append(Paragraph("Production Deployment", h2))
    story.append(Paragraph(
        "render.yaml configures Render.com auto-deploy on git push to main. "
        "requirements.txt pins all Python dependencies; packages.txt handles system-level apt "
        "dependencies. Live at: https://equity-commodities-spillover.onrender.com", body))

    # ── Selected commits ───────────────────────────────────────────────────────
    story.append(Paragraph("Selected Commit Links", h1))
    story.append(rule())
    story.append(tbl(
        ["Commit", "Description"],
        [
            ("5826681", "methodology: academically rigorous statistical upgrades"),
            ("eca9fc8", "harness: verification loop trace logging + measured eval hit rates"),
            ("c25e8d6", "Full audit pass: close all 25 gaps against core problem statement"),
            ("04f741f", "perf: parallel data loading + extended cache TTLs on Command Center"),
            ("7d5d98d", "Add GDELT live conflict escalation feed"),
            ("7f1a834", "Major update: CQO remediation loop, UI overhaul, AI Workforce page"),
            ("260fed0", "Major platform upgrade: geo risk gauge, intelligence layer, strategy pages"),
            ("23818ad", "Add Week 7 capstone final review documentation"),
            ("306fb35", "Sync working tree changes across pages and shared UI"),
        ],
        [25, 135],
        *sty,
    ))
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph(
        "Each commit hash links directly to "
        f'<link href="{REPO}/commit/[hash]" color="#2563EB">'
        f"{REPO}/commit/[hash]</link>", body))

    # ── Commit quality examples ────────────────────────────────────────────────
    story.append(Paragraph("Commit Quality — Representative Examples", h2))
    story.append(Paragraph(
        "Commits use conventional-commit prefixes (feat:, fix:, perf:, docs:, harness:, "
        "methodology:) and describe the why, not just the what:", body))

    examples = [
        ("methodology: academically rigorous statistical upgrades",
         "Added BIC lag selection, Holm-Bonferroni multi-test correction, transfer entropy "
         "lag optimization over lags 1–7, and EWMA pre-whitening before DCC-GARCH."),
        ("harness: confidence gate — threshold-based output trust enforcement",
         "Introduced per-agent confidence thresholds; outputs below gate are flagged rather "
         "than silently served as authoritative."),
        ("fix: P0 cache crash + PortWatch API shape guard",
         "Production bug: st.cache_data was caching the exception object on GDELT 429 "
         "rate-limit errors, causing all subsequent loads to re-raise the same error."),
        ("perf: parallel data loading + extended cache TTLs on Command Center",
         "ThreadPoolExecutor(max_workers=2) overlaps GDELT HTTP (~10s) and yfinance (~8s) "
         "I/O; main-thread risk scorer kept sequential due to st.session_state write safety."),
    ]
    for title, desc in examples:
        story.append(KeepTogether([
            Paragraph(f"<b><font color='#1E3A5F'>{title}</font></b>", ParagraphStyle(
                "CommitTitle", fontName="Helvetica-Bold", fontSize=8.5,
                textColor=NAVY, spaceBefore=6, spaceAfter=1, leading=12)),
            Paragraph(desc, ParagraphStyle(
                "CommitDesc", fontName="Helvetica", fontSize=8.5,
                textColor=BODY, leading=13, spaceAfter=4, leftIndent=10)),
        ]))

    # ── Teammate collaboration ─────────────────────────────────────────────────
    story.append(Paragraph("Teammate Collaboration", h1))
    story.append(rule())
    story.append(Paragraph(
        "The division of labor was: Ilian and Jiahe designed domain-specific frameworks and "
        "provided research and narrative input; I built every system component that actually runs.", body))

    story.append(tbl(
        ["Teammate", "Their Contribution", "My Implementation"],
        [
            ("Ilian Zalomai",
             "War Impact Map scoring framework, 13 geopolitical event definitions, "
             "Strait Watch chokepoint methodology, concurrent-war amplifier, "
             "all dashboard narrative text",
             "src/analysis/conflict_model.py, config.py, "
             "src/pages/war_impact_map.py, src/pages/strait_watch.py"),
            ("Jiahe Miao",
             "Correlation regime taxonomy (4-state), rolling window methodology, "
             "6 regime-conditioned trade structures, D-Y methodology research, "
             "Fixed Income signal framework",
             "src/analysis/correlations.py, src/pages/correlation.py, "
             "src/pages/trade_ideas.py, src/pages/spillover.py"),
        ],
        [30, 72, 58],
        *sty,
    ))

    # ── Audit trail ────────────────────────────────────────────────────────────
    story.append(Paragraph("Audit Trail", h1))
    story.append(rule())
    story.append(Paragraph(
        "AUDIT.md in the repository documents 25 identified gaps against the problem statement, "
        "all marked ✅ Fixed with the date each was resolved. This serves as a week-by-week "
        "record of what I diagnosed, decided to fix, and shipped. Selected resolutions:", body))

    audit_items = [
        ("Gap 15 — circular ground truth", "Historical benchmark rewritten from non-VIX sources (realized returns + yield curve spreads)"),
        ("Gap 18 — no out-of-sample test", "56-case harness with documented pass/fail criteria; pass threshold set at project outset"),
        ("Gap 20 — TE lag hardcoded at 1", "Lag search over 1–7 days; commodity→equity lag empirically 2–5 days"),
        ("Gap 19 — linear betas only", "Regime-conditioned OLS betas replace linear scenario propagation"),
        ("Gap 16 — silent failures", "Stale data banners, insufficient data warnings, and fallback labels throughout all pages"),
        ("Gap 24 — Granger not visualized", "Transmission Matrix page with directed causality arrows and network graph"),
    ]
    story.append(tbl(
        ["Gap", "Resolution"],
        audit_items,
        [55, 105],
        *sty,
    ))

    # ── Footer ─────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 8 * mm))
    story.append(rule(GOLD, 1.0))
    story.append(Paragraph(
        f"GitHub: {REPO}  ·  "
        "Live: https://equity-commodities-spillover.onrender.com  ·  "
        "MGMT 69000-120 · Purdue University · 2026-04-30",
        center))


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    doc = SimpleDocTemplate(
        OUTPUT,
        pagesize=A4,
        leftMargin=LM, rightMargin=RM,
        topMargin=TM, bottomMargin=BM,
        title="Individual Contribution — Heramb S. Patkar",
        author="Heramb S. Patkar",
    )

    styles = S()
    story = []
    cover(story, *styles)
    content(story, *styles)
    doc.build(story)
    print(f"Saved → {OUTPUT}")


if __name__ == "__main__":
    main()
