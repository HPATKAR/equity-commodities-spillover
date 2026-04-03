"""
Generate formal submission PDF:
  Part I  - Research Brief (landscape, gap analysis, project definition, technology)
  Part II - User Manual (page-by-page operating guide)
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
from reportlab.platypus import ListFlowable, ListItem

W, H = A4
OUTPUT = "Spillover_Monitor_Submission.pdf"

# ── palette ──────────────────────────────────────────────────────────────────
NAVY   = colors.HexColor("#1E3A5F")
GOLD   = colors.HexColor("#C9A84C")
BLACK  = colors.HexColor("#1A1A2E")
BODY   = colors.HexColor("#2D2D2D")
MUTED  = colors.HexColor("#666666")
LIGHT  = colors.HexColor("#F5F6FA")
WHITE  = colors.white
RULE   = colors.HexColor("#D0D4E0")
GREEN  = colors.HexColor("#2E7D52")
RED    = colors.HexColor("#C0392B")
AMBER  = colors.HexColor("#C9A84C")

LM = 25 * mm
RM = 25 * mm
TM = 22 * mm
BM = 22 * mm


# ── styles ───────────────────────────────────────────────────────────────────
def S():
    h1 = ParagraphStyle("H1", fontName="Helvetica-Bold",
                        fontSize=15, textColor=NAVY,
                        spaceBefore=18, spaceAfter=6, leading=20)
    h2 = ParagraphStyle("H2", fontName="Helvetica-Bold",
                        fontSize=11, textColor=NAVY,
                        spaceBefore=14, spaceAfter=4, leading=15)
    h3 = ParagraphStyle("H3", fontName="Helvetica-Bold",
                        fontSize=9.5, textColor=BODY,
                        spaceBefore=10, spaceAfter=3, leading=14)
    body = ParagraphStyle("Body", fontName="Helvetica",
                          fontSize=9, textColor=BODY,
                          leading=15, spaceAfter=7,
                          alignment=TA_JUSTIFY)
    bullet = ParagraphStyle("Bullet", fontName="Helvetica",
                            fontSize=9, textColor=BODY,
                            leading=14, spaceAfter=3,
                            leftIndent=12, firstLineIndent=0)
    label = ParagraphStyle("Label", fontName="Helvetica",
                           fontSize=7, textColor=GOLD,
                           spaceBefore=0, spaceAfter=2,
                           letterSpacing=1.2)
    caption = ParagraphStyle("Caption", fontName="Helvetica-Oblique",
                             fontSize=8, textColor=MUTED,
                             spaceAfter=6, alignment=TA_CENTER)
    cover_t = ParagraphStyle("CoverT", fontName="Helvetica-Bold",
                             fontSize=24, textColor=WHITE,
                             spaceAfter=6, alignment=TA_CENTER, leading=30)
    cover_s = ParagraphStyle("CoverS", fontName="Helvetica",
                             fontSize=10.5, textColor=colors.HexColor("#C8D0E0"),
                             spaceAfter=4, alignment=TA_CENTER)
    cover_m = ParagraphStyle("CoverM", fontName="Helvetica",
                             fontSize=9, textColor=colors.HexColor("#8899BB"),
                             spaceAfter=3, alignment=TA_CENTER)
    toc = ParagraphStyle("TOC", fontName="Helvetica",
                         fontSize=9, textColor=BODY,
                         leading=16, spaceAfter=2, leftIndent=0)
    toc2 = ParagraphStyle("TOC2", fontName="Helvetica",
                          fontSize=9, textColor=MUTED,
                          leading=15, spaceAfter=1, leftIndent=14)
    note = ParagraphStyle("Note", fontName="Helvetica-Oblique",
                          fontSize=8.5, textColor=MUTED,
                          leading=13, spaceAfter=5,
                          leftIndent=10, rightIndent=10)
    return dict(h1=h1, h2=h2, h3=h3, body=body, bullet=bullet,
                label=label, caption=caption,
                cover_t=cover_t, cover_s=cover_s, cover_m=cover_m,
                toc=toc, toc2=toc2, note=note)


# ── canvas callbacks ─────────────────────────────────────────────────────────
def cover_canvas(c, doc):
    c.saveState()
    # navy full background
    c.setFillColor(NAVY)
    c.rect(0, 0, W, H, fill=1, stroke=0)
    # gold bar top
    c.setFillColor(GOLD)
    c.rect(0, H - 6, W, 6, fill=1, stroke=0)
    # lighter mid panel
    c.setFillColor(colors.HexColor("#162d4a"))
    c.rect(LM - 5, H * 0.30, W - LM - RM + 10, H * 0.42, fill=1, stroke=0)
    # footer strip
    c.setFillColor(colors.HexColor("#111D2E"))
    c.rect(0, 0, W, 18 * mm, fill=1, stroke=0)
    c.setFont("Helvetica", 7.5)
    c.setFillColor(colors.HexColor("#7788AA"))
    c.drawString(LM, 7 * mm,
        "Equity-Commodities Spillover Monitor  ·  Purdue Daniels  ·  MGMT 69000-120  ·  AI for Finance")
    c.restoreState()


def page_canvas(c, doc):
    c.saveState()
    # white background
    c.setFillColor(WHITE)
    c.rect(0, 0, W, H, fill=1, stroke=0)
    # navy top bar
    c.setFillColor(NAVY)
    c.rect(0, H - 10 * mm, W, 10 * mm, fill=1, stroke=0)
    # document title in header
    c.setFont("Helvetica", 7)
    c.setFillColor(colors.HexColor("#8899BB"))
    c.drawString(LM, H - 6.5 * mm,
                 "Equity-Commodities Spillover Monitor  ·  Research Brief & User Manual")
    c.setFillColor(GOLD)
    c.drawRightString(W - RM, H - 6.5 * mm, f"Page {doc.page}")
    # gold bottom rule
    c.setStrokeColor(GOLD)
    c.setLineWidth(0.8)
    c.line(LM, 14 * mm, W - RM, 14 * mm)
    # footer text
    c.setFont("Helvetica", 7)
    c.setFillColor(MUTED)
    c.drawString(LM, 9 * mm,
                 "Purdue University  ·  Mitchell E. Daniels, Jr. School of Business  ·  MGMT 69000-120")
    c.restoreState()


# ── helpers ──────────────────────────────────────────────────────────────────
def rule(color=RULE, thickness=0.5, space_before=4, space_after=8):
    return HRFlowable(width="100%", thickness=thickness,
                      color=color, spaceBefore=space_before, spaceAfter=space_after)


def gold_rule():
    return HRFlowable(width="100%", thickness=1.2,
                      color=GOLD, spaceBefore=2, spaceAfter=10)


def section_heading(s, number, title, subtitle=None):
    items = [
        Paragraph(f"SECTION {number}", s["label"]),
        Paragraph(title, s["h1"]),
        gold_rule(),
    ]
    if subtitle:
        items.append(Paragraph(subtitle, s["note"]))
        items.append(Spacer(1, 4))
    return items


def sub_heading(s, title):
    return [Paragraph(title, s["h2"]), rule()]


def subsub_heading(s, title):
    return [Paragraph(title, s["h3"])]


def para(s, text):
    return Paragraph(text, s["body"])


def bullets(s, items, indent=12):
    st = ParagraphStyle("bi", fontName="Helvetica", fontSize=9,
                        textColor=BODY, leading=14, spaceAfter=2,
                        leftIndent=indent)
    return [Paragraph(f"&#8226;&nbsp;&nbsp;{t}", st) for t in items]


def space(n=6):
    return Spacer(1, n)


def _cell(text, bold=False, color=BODY, size=8.5, center=False):
    """Wrap a string in a Paragraph so it word-wraps inside table cells."""
    if not isinstance(text, str):
        return text
    align = TA_CENTER if center else TA_LEFT
    st = ParagraphStyle(
        "tc",
        fontName="Helvetica-Bold" if bold else "Helvetica",
        fontSize=size,
        textColor=color,
        leading=size * 1.45,
        wordWrap="CJK",
        alignment=align,
    )
    return Paragraph(text, st)


def _wrap_row(row, bold=False, color=BODY, size=8.5):
    return [_cell(c, bold=bold, color=color, size=size) for c in row]


def make_table(data, col_widths, header=True, zebra=True):
    """Build a styled, word-wrapping table. data[0] is header row."""
    usable = W - LM - RM
    cw = [usable * f for f in col_widths] if col_widths else [usable / len(data[0])] * len(data[0])

    # wrap every cell so text never overflows
    wrapped = []
    for i, row in enumerate(data):
        if i == 0 and header:
            wrapped.append(_wrap_row(row, bold=True, color=WHITE, size=8.5))
        else:
            wrapped.append(_wrap_row(row, bold=False, color=BODY, size=8.5))

    t = Table(wrapped, colWidths=cw,
              repeatRows=1 if header else 0,
              splitByRow=1)          # allow splitting across pages
    style = [
        # header
        ("BACKGROUND",    (0, 0), (-1, 0), NAVY),
        ("TOPPADDING",    (0, 0), (-1, 0), 7),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 7),
        ("LEFTPADDING",   (0, 0), (-1, 0), 8),
        ("RIGHTPADDING",  (0, 0), (-1, 0), 6),
        # body
        ("TOPPADDING",    (0, 1), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 6),
        ("LEFTPADDING",   (0, 1), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 1), (-1, -1), 6),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        # grid
        ("LINEBELOW",     (0, 0), (-1, 0), 1.2, GOLD),
        ("LINEBELOW",     (0, 1), (-1, -2), 0.4, RULE),
        ("LINEBELOW",     (0, -1), (-1, -1), 0.6, NAVY),
        ("BOX",           (0, 0), (-1, -1), 0.6, NAVY),
    ]
    if zebra:
        for i in range(1, len(wrapped)):
            if i % 2 == 0:
                style.append(("BACKGROUND", (0, i), (-1, i),
                               colors.HexColor("#F0F2F8")))
    t.setStyle(TableStyle(style))
    return t


def callout_box(s, label, text, bg=colors.HexColor("#EEF2FB"), border=NAVY):
    """Highlighted callout / insight box."""
    inner = Table(
        [[Paragraph(f"<b>{label}</b>", ParagraphStyle(
              "cl", fontName="Helvetica-Bold", fontSize=8,
              textColor=border, spaceAfter=3)),
          ""],
         [Paragraph(text, ParagraphStyle(
              "cb", fontName="Helvetica", fontSize=8.5,
              textColor=BODY, leading=13)), ""]],
        colWidths=[W - LM - RM - 6, 0],
    )
    inner.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (0, 0), 8),
        ("BOTTOMPADDING", (0, -1), (-1, -1), 8),
        ("LINEAFTER", (0, 0), (0, -1), 3, border),
        ("BOX", (0, 0), (-1, -1), 0.5, border),
    ]))
    return [inner, space(8)]


# ════════════════════════════════════════════════════════════════════════════
# CONTENT
# ════════════════════════════════════════════════════════════════════════════
def build_cover(s, story):
    story.append(Spacer(1, 52 * mm))
    story.append(Paragraph("EQUITY-COMMODITIES", s["cover_t"]))
    story.append(Paragraph("SPILLOVER MONITOR", s["cover_t"]))
    story.append(Spacer(1, 5))
    story.append(HRFlowable(width="50%", thickness=1, color=GOLD,
                             spaceAfter=6, hAlign="CENTER"))
    story.append(Paragraph(
        "Research Brief &amp; User Manual", s["cover_s"]))
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        "Course Project 3  ·  MGMT 69000-120: AI for Finance", s["cover_m"]))
    story.append(Paragraph(
        "Purdue University  ·  Mitchell E. Daniels, Jr. School of Business",
        s["cover_m"]))
    story.append(Spacer(1, 24))
    story.append(Paragraph(
        "Heramb S. Patkar  ·  Jiahe Miao  ·  Ilian Zalomai", s["cover_s"]))
    story.append(Paragraph("April 2026", s["cover_m"]))
    story.append(PageBreak())


def build_toc(s, story):
    story.append(Spacer(1, 6))
    story.append(Paragraph("TABLE OF CONTENTS", s["label"]))
    story.append(Paragraph("Contents", s["h1"]))
    story.append(gold_rule())
    story.append(space(4))

    entries = [
        ("PART I - RESEARCH BRIEF", None),
        ("1.", "Landscape Research", True),
        ("", "1.1  Financial Analytics Industry Overview", False),
        ("", "1.2  Cross-Asset Spillover Methods: Academic State of the Art", False),
        ("", "1.3  AI in Financial Analytics", False),
        ("", "1.4  Competitive Landscape", False),
        ("2.", "Gap Analysis", True),
        ("", "2.1  Identified Gaps in Existing Solutions", False),
        ("", "2.2  Opportunity Summary", False),
        ("3.", "Project Definition and Scope", True),
        ("", "3.1  Problem Statement", False),
        ("", "3.2  Objectives", False),
        ("", "3.3  Coverage Universe", False),
        ("", "3.4  Scope Boundaries", False),
        ("4.", "Technology Choices and Justification", True),
        ("", "4.1  Technology Stack Overview", False),
        ("", "4.2  Justification by Layer", False),
        ("PART II - USER MANUAL", None),
        ("5.", "Getting Started", True),
        ("", "5.1  System Requirements & Installation", False),
        ("", "5.2  Configuration (API Keys)", False),
        ("", "5.3  Navigation Overview", False),
        ("6.", "Page-by-Page Operating Guide", True),
        ("", "Overview  ·  Macro Dashboard  ·  War Impact Map", False),
        ("", "Geopolitical Triggers  ·  Correlation Analysis", False),
        ("", "Spillover Analytics  ·  Trade Ideas  ·  Stress Test", False),
        ("", "Scenario Engine  ·  Commodities  ·  Strait Watch", False),
        ("", "Performance Review  ·  AI Analyst  ·  Insights", False),
        ("7.", "AI Workforce Reference", True),
        ("8.", "Interpreting Key Outputs", True),
        ("9.", "Troubleshooting & FAQ", True),
    ]

    for num, title, bold in [e if len(e) == 3 else (e[0], e[1], None) for e in entries]:
        if title is None:
            story.append(space(6))
            story.append(Paragraph(
                f"<b>{num}</b>",
                ParagraphStyle("pt", fontName="Helvetica-Bold", fontSize=9,
                               textColor=NAVY, spaceAfter=4, leading=14)))
        elif bold:
            story.append(Paragraph(
                f"<b>{num}</b>&nbsp;&nbsp;{title}",
                ParagraphStyle("t1", fontName="Helvetica-Bold", fontSize=9,
                               textColor=BODY, spaceAfter=3, leading=14)))
        else:
            story.append(Paragraph(
                f"&nbsp;&nbsp;&nbsp;&nbsp;{title}",
                ParagraphStyle("t2", fontName="Helvetica", fontSize=8.5,
                               textColor=MUTED, spaceAfter=2, leading=13,
                               leftIndent=8)))

    story.append(PageBreak())


# ── PART I ───────────────────────────────────────────────────────────────────
def build_part1_header(s, story):
    story.append(space(10))
    story.append(Paragraph("PART I", s["label"]))
    story.append(Paragraph("Research Brief", ParagraphStyle(
        "ph", fontName="Helvetica-Bold", fontSize=20, textColor=NAVY,
        spaceAfter=4, leading=26)))
    story.append(HRFlowable(width="100%", thickness=2, color=NAVY,
                             spaceAfter=4))
    story.append(HRFlowable(width="100%", thickness=0.5, color=GOLD,
                             spaceAfter=10))
    story.append(para(s,
        "This research brief documents the intellectual foundation, design rationale, and "
        "technology architecture of the Equity-Commodities Spillover Monitor. It is structured "
        "to satisfy the four required deliverables: landscape research, gap analysis, "
        "project definition and scope, and technology choices with justification."))
    story.append(PageBreak())


def build_section1(s, story):
    story += section_heading(s, "1", "Landscape Research",
        "Current state of the financial analytics industry, academic spillover methodology, "
        "and AI integration in finance.")

    # 1.1
    story += sub_heading(s, "1.1  Financial Analytics Industry Overview")
    story.append(para(s,
        "The professional financial analytics market is currently dominated by a small number of "
        "high-cost terminal-based platforms. Bloomberg Terminal, with approximately 330,000 "
        "active subscriptions at roughly USD 25,000 per seat per annum, commands the largest "
        "institutional footprint. Refinitiv Eikon (now LSEG Workspace), FactSet Research Systems, "
        "and S&P Global Market Intelligence collectively serve the remainder of the institutional "
        "buy-side and sell-side. These platforms are characterized by comprehensive data access "
        "but are cost-prohibitive for academic institutions, emerging market practitioners, "
        "and smaller investment firms."))
    story.append(para(s,
        "Within the quantitative finance sub-segment, dedicated risk and analytics platforms "
        "include MSCI RiskMetrics, BlackRock Aladdin, and Bloomberg PORT. These tools focus "
        "primarily on portfolio-level risk attribution and are not designed for cross-asset "
        "regime-based correlation monitoring. Open-source alternatives (QuantLib, zipline, "
        "bt) address backtesting and portfolio optimization but lack integrated visualization, "
        "real-time data ingestion, and AI synthesis layers."))

    story.append(space(4))
    story.append(para(s, "<b>Table 1.1 - Major Financial Analytics Platforms: Feature Comparison</b>"))
    t1 = make_table([
        ["Platform", "Cross-Asset\nCorrelation", "Regime\nDetection", "AI\nSynthesis", "Geopolitical\nMonitoring", "Annual\nCost (USD)"],
        ["Bloomberg Terminal", "✓", "Limited", "Bloomberg GPT (beta)", "Manual", "~$25,000"],
        ["Refinitiv Eikon", "✓", "None", "News NLP only", "Manual", "~$22,000"],
        ["FactSet", "Partial", "None", "FactSet Signals", "None", "~$18,000"],
        ["MSCI RiskMetrics", "✓", "Limited", "None", "None", "~$30,000+"],
        ["This Platform", "✓ (DCC-GARCH)", "✓ (Markov)", "7-Agent Pipeline", "✓ (Automated)", "Open-source"],
    ], [0.22, 0.14, 0.14, 0.18, 0.17, 0.15])
    story.append(t1)
    story.append(Paragraph("Source: Publicly available vendor pricing and feature documentation, 2025.", s["caption"]))
    story.append(space(8))

    # 1.2
    story += sub_heading(s, "1.2  Cross-Asset Spillover Methods: Academic State of the Art")
    story.append(para(s,
        "The academic literature on cross-asset spillover analysis has produced several "
        "foundational methodologies that inform this project. Key contributions are summarized below."))

    story.append(space(4))
    story.append(para(s, "<b>Table 1.2 - Foundational Spillover Methodologies</b>"))
    t2 = make_table([
        ["Method", "Authors", "Year", "Application in This Project"],
        ["Granger Causality", "C.W.J. Granger", "1969", "Directional lead-lag identification between asset pairs"],
        ["Markov Regime-Switching", "J.D. Hamilton", "1989", "Four-state correlation regime classification and forecasting"],
        ["GARCH / DCC-GARCH", "R.F. Engle", "1982 / 2002", "Dynamic conditional correlation estimation"],
        ["FEVD Connectedness", "Diebold & Yilmaz", "2009, 2012, 2014", "System-wide spillover index and directional net transmitter scores"],
        ["Transfer Entropy", "T. Schreiber", "2000", "Nonlinear information flow beyond linear Granger framework"],
        ["Early Warning Systems", "Kaminsky et al.", "1998", "Composite signal-based regime transition detection"],
    ], [0.20, 0.18, 0.10, 0.52])
    story.append(t2)
    story.append(Paragraph("Source: Original publications. Implementations in statsmodels, arch, scipy.", s["caption"]))
    story.append(space(8))

    story.append(para(s,
        "While each of these methods is well-established individually, their integration into a "
        "single, accessible, real-time platform with a unified interface remains absent from "
        "both commercial and open-source offerings. The academic-practitioner gap - where rigorous "
        "econometric methods remain confined to research code and are not operationalized for "
        "day-to-day market monitoring - represents the primary intellectual motivation for "
        "this project."))

    # 1.3
    story += sub_heading(s, "1.3  AI in Financial Analytics")
    story.append(para(s,
        "The application of large language models (LLMs) to financial analysis is nascent but "
        "rapidly maturing. Bloomberg introduced BloombergGPT in 2023, a 50-billion parameter "
        "model trained on financial text corpora, focused on information retrieval and "
        "summarization. Kensho Technologies (acquired by S&P Global) pioneered NLP-based event "
        "analysis. Refinitiv News Analytics provides automated sentiment scoring across "
        "news feeds. FactSet has introduced Signals for quantitative factor generation."))
    story.append(para(s,
        "However, current AI financial tools share a critical limitation: they operate as "
        "single-agent, single-domain tools performing text processing or data summarization. "
        "None implement multi-agent orchestrated workflows where specialized agents share "
        "structured outputs, build on each other's analysis, and flag inter-agent "
        "disagreements. This multi-agent synthesis architecture is the primary AI innovation "
        "in this project."))

    story += callout_box(s,
        "KEY OBSERVATION",
        "Existing AI financial tools summarize data. This platform synthesizes it - "
        "through seven agents operating in three dependency-ordered rounds, each informed "
        "by the structured outputs of its upstream peers before generating its own analysis.")

    # 1.4
    story += sub_heading(s, "1.4  Competitive Landscape")
    story.append(para(s,
        "Beyond the institutional platforms, the landscape includes a tier of data "
        "visualization and fintech analytics tools targeting retail investors and smaller "
        "institutions. TradingView provides technical analysis and basic correlation tools. "
        "Koyfin offers cross-asset charting and macro data visualization. Portfolio Visualizer "
        "provides historical backtesting. Macroaxis and Seeking Alpha provide AI-assisted "
        "fundamental analysis. None of these platforms implement quantitative spillover "
        "analytics (Granger, DCC-GARCH, Diebold-Yilmaz), geopolitical event-window forensics, "
        "parametric scenario propagation with live betas, or multi-agent AI synthesis."))
    story.append(PageBreak())


def build_section2(s, story):
    story += section_heading(s, "2", "Gap Analysis",
        "Identification of unaddressed market needs and structural limitations in existing platforms.")

    # 2.1
    story += sub_heading(s, "2.1  Identified Gaps in Existing Solutions")

    gaps = [
        ("Gap 1: Cost Accessibility",
         "Bloomberg Terminal and Refinitiv Eikon are priced at USD 18,000–30,000 per annum. "
         "This pricing structure is inaccessible to academic researchers, emerging market "
         "practitioners, independent analysts, and smaller investment funds. No open-source "
         "alternative provides comparable cross-asset analytical breadth with a unified "
         "interface."),
        ("Gap 2: Integration Fragmentation",
         "Spillover analytics, geopolitical risk monitoring, macro intelligence, implied "
         "volatility tracking, and trade idea generation currently exist as separate, "
         "disconnected workflows. A practitioner must query multiple platforms and manually "
         "synthesize outputs - a process that introduces latency and human error in high-volatility "
         "environments. No unified, end-to-end workflow exists."),
        ("Gap 3: Static Correlation Methods",
         "Standard rolling correlation (Pearson) uses fixed lookback windows and does not "
         "account for volatility clustering. The DCC-GARCH model, which captures dynamic "
         "conditional correlations and is methodologically superior for regime-change environments, "
         "is absent from all publicly available non-terminal dashboards."),
        ("Gap 4: Absence of Forward-Looking Regime Forecasting",
         "Existing tools classify the current regime but do not provide probabilistic forecasts "
         "of regime transitions. The Markov transition matrix, steady-state distribution, and "
         "mean first-passage time to Crisis - outputs that directly inform risk management "
         "decisions - are absent from standard analytics platforms."),
        ("Gap 5: AI Synthesis vs. AI Summarization",
         "Current AI financial tools retrieve and summarize data. They do not perform "
         "multi-domain synthesis across simultaneously active analytical frameworks. "
         "The architecture of a single generalist agent with access to raw data is "
         "qualitatively different from seven specialist agents - macro, geopolitical, risk, "
         "commodities, stress, trade, signal - each reading its peers' structured outputs "
         "before forming its own view."),
        ("Gap 6: Geopolitical-Market Linkage",
         "No publicly available tool systematically maps geopolitical shocks to cross-asset "
         "correlation regime changes with empirical event-window analysis. The forensic "
         "examination of pre, during, and post-event performance - across thirteen shocks "
         "since 2008 - with vol shift and correlation regime attribution is not available "
         "on any accessible platform."),
        ("Gap 7: Parametric Scenario Propagation",
         "Scenario analysis tools are either proprietary (institutional risk systems) or "
         "overly simplified (single-asset sensitivity analysis). No accessible tool combines "
         "parametric multi-channel shock inputs (oil, gold, yields, DXY, credit, geopolitical) "
         "with live OLS beta propagation to a full cross-asset universe and outputs VaR and ES "
         "at multiple confidence levels."),
    ]

    for title, body_text in gaps:
        story.append(KeepTogether([
            Paragraph(title, ParagraphStyle(
                "gt", fontName="Helvetica-Bold", fontSize=9.5,
                textColor=NAVY, spaceBefore=10, spaceAfter=3, leading=14)),
            rule(thickness=0.4, space_before=0, space_after=4),
            Paragraph(body_text, ParagraphStyle(
                "gb", fontName="Helvetica", fontSize=9,
                textColor=BODY, leading=14, spaceAfter=8,
                alignment=TA_JUSTIFY)),
        ]))

    # 2.2
    story += sub_heading(s, "2.2  Opportunity Summary")
    story.append(para(s,
        "The convergence of three trends creates a timely opportunity: (1) the increasing "
        "availability of institutional-grade free data via Yahoo Finance and FRED; (2) the "
        "maturation of open-source econometric libraries implementing rigorous academic methods; "
        "and (3) the emergence of accessible, high-capability LLM APIs enabling multi-agent "
        "orchestration. The opportunity is to build an institutional-grade cross-asset analytics "
        "platform that is fully open-source, deployable at zero marginal cost, and methodologically "
        "rigorous."))

    story.append(space(4))
    story.append(para(s, "<b>Table 2.1 - Opportunity Matrix</b>"))
    t3 = make_table([
        ["Gap Identified", "Opportunity", "Priority"],
        ["Cost accessibility", "Open-source, free-to-deploy platform", "Critical"],
        ["Integration fragmentation", "Unified 14-page analytical workspace", "Critical"],
        ["Static correlation only", "DCC-GARCH dynamic conditional correlation", "High"],
        ["No regime forecasting", "Markov transition matrix with first-passage times", "High"],
        ["AI summarization only", "7-agent dependency-ordered synthesis pipeline", "High"],
        ["No geo-market forensics", "13-event window analysis with correlation attribution", "Medium"],
        ["No scenario propagation", "6-channel parametric shock engine with live betas", "High"],
    ], [0.30, 0.50, 0.20])
    story.append(t3)
    story.append(PageBreak())


def build_section3(s, story):
    story += section_heading(s, "3", "Project Definition and Scope",
        "Formal statement of the problem, system objectives, coverage universe, "
        "analytical architecture, and scope boundaries.")

    # 3.1
    story += sub_heading(s, "3.1  Problem Statement")
    story += callout_box(s,
        "PROBLEM STATEMENT",
        "Portfolio managers, risk analysts, and researchers require a unified, real-time "
        "view of cross-asset stress transmission dynamics - integrating correlation regime "
        "detection, directional spillover identification, geopolitical event monitoring, "
        "macro intelligence, and AI-synthesized trade ideas - at a cost accessible beyond "
        "institutional terminal licensing. No such platform currently exists in the open-source "
        "or low-cost commercial landscape.",
        bg=colors.HexColor("#EEF2FB"), border=NAVY)

    # 3.2
    story += sub_heading(s, "3.2  Objectives")
    story.append(para(s, "The Equity-Commodities Spillover Monitor is designed to achieve "
                     "the following primary objectives:"))
    story += bullets(s, [
        "Provide real-time correlation regime classification (four states: Decorrelated, Normal, "
        "Elevated, Crisis) with Markov-based forward transition probability forecasts.",
        "Implement and expose institutional-grade spillover methods - Granger causality, "
        "transfer entropy, Diebold-Yilmaz FEVD - in an accessible, interactive interface.",
        "Enable parametric scenario analysis via multi-channel shock propagation using "
        "live OLS betas, with VaR and ES tail risk quantification.",
        "Provide forensic event-window analysis across thirteen tracked geopolitical shocks "
        "since the Global Financial Crisis.",
        "Deploy a seven-agent AI orchestration pipeline that synthesizes macro, geopolitical, "
        "commodity, risk, stress, and trade intelligence in a structured dependency graph.",
        "Generate a composite Early Warning System (EWS) score identifying conditions that "
        "historically precede correlation regime transitions.",
        "Serve as an open-source, educationally accessible alternative to institutional-grade "
        "cross-asset analytics platforms.",
    ])

    # 3.3
    story += sub_heading(s, "3.3  Coverage Universe")
    story.append(space(4))
    story.append(para(s, "<b>Table 3.1 - Coverage Universe</b>"))
    t4 = make_table([
        ["Asset Class", "Count", "Instruments"],
        ["Global Equity Indices", "15", "S&P 500, Nasdaq, DJIA, Russell 2000, Eurostoxx 50, DAX, "
         "CAC 40, FTSE 100, Nikkei 225, TOPIX, Hang Seng, Shanghai Composite, CSI 300, Sensex, Nifty 50"],
        ["Commodity Futures", "17", "WTI, Brent, Natural Gas, RBOB, Heating Oil, Gold, Silver, "
         "Platinum, Copper, Aluminum, Nickel, Wheat, Corn, Soybeans, Sugar #11, Coffee, Cotton"],
        ["Fixed Income ETFs", "6", "TLT (20Y+ Treasury), LQD (IG Corporate), HYG (HY Corporate), "
         "EMB (EM USD Bonds), SHY (Short Treasury), TIP (TIPS)"],
        ["FX Pairs", "6", "DXY, EUR/USD, GBP/USD, USD/JPY, USD/CNY, USD/INR"],
        ["Implied Volatility", "4", "VIX (Equity), OVX (Oil), GVZ (Gold), VVIX (Vol of Vol)"],
        ["Geopolitical Events", "13", "GFC, Arab Spring, Trade War, Aramco, COVID-19, WTI Negative, "
         "Ukraine War, LME Nickel, Fed Hike, SVB Crisis, Israel-Hamas, India-Pakistan, Iran-Hormuz"],
    ], [0.22, 0.08, 0.70])
    story.append(t4)
    story.append(space(8))

    # 3.4
    story += sub_heading(s, "3.4  Scope Boundaries")
    story.append(para(s, "<b>In scope:</b>"))
    story += bullets(s, [
        "Daily close price data for all equity, commodity, fixed income, and FX instruments",
        "Live macro data via FRED API",
        "RSS-based automated geopolitical headline ingestion and severity scoring",
        "Correlation regime detection, DCC-GARCH, Markov forecasting",
        "Granger causality, transfer entropy, Diebold-Yilmaz FEVD",
        "Parametric scenario propagation with OLS betas",
        "Historical portfolio stress testing against tracked events",
        "Seven-agent AI orchestration pipeline with peer context sharing",
        "Maritime chokepoint disruption monitoring (six straits)",
    ])
    story.append(space(4))
    story.append(para(s, "<b>Out of scope:</b>"))
    story += bullets(s, [
        "Order execution and trading infrastructure",
        "Real-time tick or intraday data at sub-daily resolution (beyond yfinance hourly)",
        "Options pricing, Greeks, or volatility surface analytics",
        "Fundamental analysis, earnings models, or company-level data",
        "Credit default swap pricing or single-name credit analytics",
        "Regulatory reporting or compliance workflows",
    ])
    story.append(PageBreak())


def build_section4(s, story):
    story += section_heading(s, "4", "Technology Choices and Justification",
        "Full-stack technology selection rationale across data, analytics, AI, and deployment layers.")

    story += sub_heading(s, "4.1  Technology Stack Overview")
    story.append(space(4))
    story.append(para(s, "<b>Table 4.1 - Full Technology Stack</b>"))
    t5 = make_table([
        ["Layer", "Library / Service", "Version", "Primary Role"],
        ["UI Framework", "Streamlit", "1.x", "Multi-page web application, widgets, caching, layout"],
        ["Visualization", "Plotly", "5.x", "Interactive charts, heatmaps, network graphs, choropleths"],
        ["Market Data", "yfinance", "0.2.x", "Historical and live equity, commodity, ETF, FX data"],
        ["Macro Data", "fredapi", "0.5.x", "FRED macro series: yield curve, CPI, GDP, PMI, rates"],
        ["News / RSS", "feedparser", "6.x", "Automated geopolitical RSS ingestion from 7 sources"],
        ["Time Series", "pandas / numpy", "2.x / 1.x", "Data wrangling, rolling statistics, return computation"],
        ["Econometrics", "statsmodels", "0.14.x", "Granger causality, VAR, OLS regression, regime stats"],
        ["GARCH Models", "arch", "6.x", "DCC-GARCH dynamic conditional correlation estimation"],
        ["ML / Validation", "scikit-learn", "1.x", "Regime classification, walk-forward cross-validation"],
        ["Scientific", "scipy", "1.x", "Transfer entropy, statistical testing, signal processing"],
        ["Graph Analytics", "networkx", "3.x", "Spillover network construction and layout algorithms"],
        ["AI (Primary)", "anthropic", "0.x", "Claude Sonnet: 7-agent orchestration and AI Analyst"],
        ["AI (Fallback)", "openai", "1.x", "GPT-4o: fallback provider for all AI agent functions"],
        ["PDF Generation", "reportlab", "4.x", "Automated PDF report generation from Python"],
        ["Image Processing", "Pillow", "10.x", "Image handling for report assets"],
        ["Deployment", "Render", "-", "Cloud hosting with Secret File support for Streamlit secrets"],
    ], [0.16, 0.17, 0.11, 0.56])
    story.append(t5)
    story.append(space(8))

    story += sub_heading(s, "4.2  Justification by Layer")

    justifications = [
        ("Streamlit (UI Framework)",
         "Streamlit was selected over Flask, FastAPI, or Dash for three reasons: (1) it is "
         "Python-native, eliminating the need for JavaScript development; (2) its caching "
         "decorators (@st.cache_data, @st.cache_resource) provide built-in TTL-based data "
         "freshness management; and (3) it supports free-tier cloud deployment with minimal "
         "configuration. The multi-page architecture and session state management are sufficient "
         "for the dashboard's complexity without introducing unnecessary framework overhead."),
        ("yfinance + fredapi (Data Layer)",
         "yfinance provides free, reliable access to Yahoo Finance historical and live data "
         "across equities, commodity ETFs, FX pairs, and volatility indices with no API key "
         "requirement. The fredapi provides official Federal Reserve Economic Data access for "
         "macro series. Together, these two libraries cover the full data universe without "
         "licensing costs. The primary limitation - absence of true commodity futures tick data "
         "- is acceptable given the daily-resolution analytical focus."),
        ("statsmodels + arch (Econometric Layer)",
         "statsmodels is the standard Python implementation of econometric methods with "
         "peer-reviewed documentation and academic acceptance. It provides Granger causality "
         "tests, vector autoregression (VAR), and OLS regression. The arch library is the "
         "canonical Python implementation of GARCH-family models, including DCC-GARCH as "
         "specified by Engle (2002). Both libraries are actively maintained and used in "
         "published academic research."),
        ("anthropic / Claude Sonnet (AI Primary)",
         "Claude Sonnet was selected as the primary AI provider for three reasons: (1) the "
         "200K token context window supports full dashboard state injection into every agent "
         "query; (2) Claude's instruction-following and structured output quality is "
         "consistently superior for financial reasoning tasks; and (3) Anthropic's API "
         "pricing is competitive relative to GPT-4o for the volume of agent calls made per "
         "session. OpenAI GPT-4o serves as a fallback provider with equivalent capability "
         "for most use cases."),
        ("Render (Deployment)",
         "Render was selected over Streamlit Community Cloud and Heroku because it supports "
         "Secret Files - a mechanism for placing actual files (including .streamlit/secrets.toml) "
         "on the server. Streamlit's secrets management requires the physical presence of this "
         "file, which Render Secret Files provides. Streamlit Community Cloud's secrets "
         "management works differently and introduces deployment complexity for multi-key "
         "configurations. Render's free tier is sufficient for demonstration and academic use."),
    ]

    for title, body_text in justifications:
        story.append(KeepTogether([
            Paragraph(title, ParagraphStyle(
                "jt", fontName="Helvetica-Bold", fontSize=9.5,
                textColor=NAVY, spaceBefore=10, spaceAfter=3)),
            Paragraph(body_text, ParagraphStyle(
                "jb", fontName="Helvetica", fontSize=9,
                textColor=BODY, leading=14, spaceAfter=8,
                alignment=TA_JUSTIFY)),
        ]))

    story.append(PageBreak())


# ── PART II ──────────────────────────────────────────────────────────────────
def build_part2_header(s, story):
    story.append(space(10))
    story.append(Paragraph("PART II", s["label"]))
    story.append(Paragraph("User Manual", ParagraphStyle(
        "ph", fontName="Helvetica-Bold", fontSize=20, textColor=NAVY,
        spaceAfter=4, leading=26)))
    story.append(HRFlowable(width="100%", thickness=2, color=NAVY, spaceAfter=4))
    story.append(HRFlowable(width="100%", thickness=0.5, color=GOLD, spaceAfter=10))
    story.append(para(s,
        "This user manual provides comprehensive operating guidance for the Equity-Commodities "
        "Spillover Monitor. It covers system setup, navigation, a page-by-page operating guide, "
        "AI workforce reference, output interpretation, and troubleshooting."))
    story.append(PageBreak())


def build_section5(s, story):
    story += section_heading(s, "5", "Getting Started",
        "System requirements, installation, configuration, and navigation overview.")

    story += sub_heading(s, "5.1  System Requirements and Installation")
    story.append(para(s, "<b>Requirements:</b> Python 3.10 or later. "
                     "All dependencies are listed in requirements.txt."))
    story += bullets(s, [
        "Clone the repository: git clone https://github.com/HPATKAR/equity-commodities-spillover.git",
        "Navigate to the project directory: cd equity-commodities-spillover",
        "Install dependencies: pip install -r requirements.txt",
        "Launch the application: streamlit run app.py",
        "Open a browser and navigate to http://localhost:8501",
    ])
    story.append(space(4))
    story.append(para(s,
        "The application will load with live data on first run. "
        "Data is cached with TTL values ranging from 15 minutes (implied vol) to 1 hour "
        "(correlation matrices, agent outputs). Initial load on a fresh environment may take "
        "30–60 seconds as all data is fetched and models are computed."))

    story += sub_heading(s, "5.2  Configuration (API Keys)")
    story.append(para(s,
        "API keys are managed via .streamlit/secrets.toml (local) or Render Secret Files "
        "(deployed). The application functions without API keys but with reduced capability: "
        "AI agent outputs and the Macro Dashboard FRED data will not be available."))

    story.append(space(4))
    story.append(para(s, "<b>Table 5.1 - API Key Configuration</b>"))
    t6 = make_table([
        ["Key", "Provider", "Required", "Purpose"],
        ["anthropic_api_key", "Anthropic (console.anthropic.com)", "Recommended", "All 7 AI agents + AI Analyst chatbot"],
        ["openai_api_key", "OpenAI (platform.openai.com)", "Optional", "Fallback provider for AI agents and chatbot"],
        ["fred_api_key", "FRED (fred.stlouisfed.org/docs/api)", "Optional", "Macro Dashboard yield curve, CPI, GDP, PMI data"],
    ], [0.24, 0.30, 0.15, 0.31])
    story.append(t6)
    story.append(space(4))

    story += callout_box(s, "SECRETS.TOML FORMAT",
        "[keys]\n"
        "anthropic_api_key = \"sk-ant-...\"\n"
        "openai_api_key    = \"sk-proj-...\"\n"
        "fred_api_key      = \"...\"\n\n"
        "[config]\n"
        "enable_auth      = false\n"
        "refresh_interval = 300",
        bg=colors.HexColor("#F0F4F0"), border=GREEN)

    story.append(para(s,
        "<b>Note on Render deployment:</b> API keys must be added as a Secret File "
        "at path .streamlit/secrets.toml, not as Environment Variables. Streamlit's "
        "st.secrets mechanism reads from the file system, not environment variables."))

    story += sub_heading(s, "5.3  Navigation Overview")
    story.append(para(s,
        "The sidebar navigation is organized into six thematic groups plus an About section. "
        "All pages are accessible from any location in the application."))
    story.append(space(4))
    story.append(para(s, "<b>Table 5.2 - Navigation Structure</b>"))
    t7 = make_table([
        ["Group", "Pages"],
        ["(Landing)", "Overview"],
        ["Macro", "Macro Intelligence Dashboard"],
        ["Analysis", "War Impact Map · Geopolitical Triggers · Correlation Analysis · Spillover Analytics"],
        ["Strategy", "Trade Ideas · Portfolio Stress Test · Scenario Engine"],
        ["Monitor", "Commodities to Watch · Strait Watch"],
        ["Research", "Performance Review · AI Analyst"],
        ["Insights", "Insights"],
        ["About", "Heramb S. Patkar · Jiahe Miao · Ilian Zalomai"],
    ], [0.20, 0.80])
    story.append(t7)
    story.append(PageBreak())


def build_section6(s, story):
    story += section_heading(s, "6", "Page-by-Page Operating Guide",
        "Detailed description of each page: purpose, components, controls, and interpretation guidance.")

    pages = [
        {
            "name": "Overview",
            "purpose": "The primary dashboard landing page. Provides a complete cross-asset "
                       "market state summary in a single view.",
            "components": [
                ("KPI Strip", "Five live metrics: Correlation Regime (color-coded green/gray/orange/red), "
                 "60-day average cross-asset correlation with 1-month delta, Best Equity (1M), "
                 "Worst Equity (1M), Best Commodity (1M)."),
                ("FI/FX Context Strip", "Five fixed income and FX signals: TLT 30-day return (duration), "
                 "HYG vs. LQD spread (credit stress), S&P 500 vs. TLT 60-day correlation "
                 "(hedge effectiveness), DXY 30-day change (dollar strength), duration slope "
                 "(TLT vs. SHY)."),
                ("Implied Volatility Panel", "Expandable section showing live OVX (oil vol), "
                 "GVZ (gold vol), VIX (equity vol), and VVIX (vol-of-vol) with freshness indicators "
                 "and historical charts."),
                ("Risk Gauge", "Composite 0–100 score across five weighted components. "
                 "Green zone: 0–30. Amber zone: 30–55. Red zone: 55–100. Updates every 5 minutes."),
                ("Early Warning System", "Five signal cards (rolling correlation slope, vol regime, "
                 "cross-asset stress, macro momentum, geopolitical escalation) with composite EWS score "
                 "and historical analogue table showing periods with similar EWS readings."),
                ("Equity-Commodity Heatmap", "Pairwise correlation heatmap across all equities "
                 "and commodities. Color intensity indicates correlation strength."),
                ("AI Workforce Output", "Risk Officer morning briefing displayed inline. "
                 "All seven agents accessible in expandable section. Divergence flags highlighted "
                 "when agents reach conflicting conclusions."),
            ],
            "interpretation": "Begin every session on this page. The regime label and risk score "
                              "provide immediate context for all downstream analysis. An EWS score "
                              "above 70 warrants heightened monitoring across the analysis pages.",
        },
        {
            "name": "Macro Intelligence Dashboard",
            "purpose": "FRED-powered macro context integrated with the AI Macro Strategist agent.",
            "components": [
                ("Yield Curve Chart", "10-year minus 2-year Treasury spread over time with recession "
                 "shading. Inversion tracker shows depth and duration of current inversion."),
                ("CPI YoY", "Consumer Price Index year-over-year change. Key input to the "
                 "Macro Strategist's inflation assessment."),
                ("Fed Funds Rate", "Effective federal funds rate. Regime shifts in rate cycles "
                 "are annotated."),
                ("GDP Growth", "Real GDP growth quarter-over-quarter. Recession periods shaded."),
                ("ISM PMI", "ISM Manufacturing PMI. Readings below 50 indicate contraction."),
                ("AI Macro Strategist", "Agent briefing incorporating yield curve shape, inflation "
                 "posture, Fed stance, and peer context from the Geopolitical Analyst and Risk Officer."),
            ],
            "interpretation": "Cross-reference the yield curve shape with the current correlation "
                              "regime. Historically, inverted yield curves coincide with elevated "
                              "equity-bond correlation (hedge breakdown). The Macro Strategist's "
                              "synthesis should be read alongside the Risk Officer's brief on the "
                              "Overview page.",
        },
        {
            "name": "War Impact Map",
            "purpose": "Global choropleth quantifying equity market exposure across three active "
                       "conflict theaters.",
            "components": [
                ("Choropleth Map", "195 countries colored by combined equity exposure score. "
                 "Toggle between Combined, Ukraine, Israel-Hamas, and Iran/Hormuz views."),
                ("Top 25 Exposed Markets", "Ranked table with per-conflict score columns and "
                 "actual equity index performance since each conflict's onset date."),
            ],
            "interpretation": "High exposure score + negative realized performance = market has "
                              "already priced the risk. High exposure score + stable performance "
                              "= potential unpriced risk. Use this page to identify positioning "
                              "opportunities in markets that have not yet reflected their "
                              "conflict exposure.",
        },
        {
            "name": "Geopolitical Triggers",
            "purpose": "Forensic event-window analysis for all thirteen tracked geopolitical shocks.",
            "components": [
                ("Event Selector", "Dropdown selection from thirteen tracked events. "
                 "Event info bar shows category, date range, and description."),
                ("Lookback Controls", "Pre-event window (15–90 days) and post-event window "
                 "(15–180 days) sliders. Asset multiselect for customizing the comparison set."),
                ("Indexed Performance Chart", "All selected assets indexed to 100 at event onset. "
                 "Vertical band marks the event window. Divergences indicate differential impact."),
                ("Volatility Shift Chart", "Pre- vs. post-event annualized realized volatility "
                 "grouped by asset. Persistent post-event vol elevation signals ongoing disruption."),
                ("Correlation Shift Table", "Asset pairs sorted by correlation shift magnitude. "
                 "Red shift (+0.1 or greater) indicates contagion. Green shift indicates decoupling."),
                ("Live Intelligence Feed", "Automated RSS ingestion showing current geopolitical "
                 "headlines tagged by region, commodity, and severity score."),
            ],
            "interpretation": "When analyzing a current event, select the most analogous "
                              "historical event and compare the pre-event window charts. "
                              "Correlation regime changes at event onset are the most reliable "
                              "early indicator of whether a shock is systemic (regime shift) or "
                              "contained (no regime change).",
        },
        {
            "name": "Correlation Analysis",
            "purpose": "Four-tab quantitative correlation analysis: rolling Pearson, DCC-GARCH, "
                       "regime detection, and Markov regime forecast.",
            "components": [
                ("Rolling Correlation Tab", "Select any two assets and a lookback window "
                 "(21–252 days). Chart shows correlation over time with regime-level reference "
                 "lines at 0.5 (Elevated) and -0.5 (Diverge). Multi-pair overlay expander "
                 "shows four default pairs simultaneously."),
                ("DCC-GARCH Tab", "Dynamic Conditional Correlation overlaid against rolling "
                 "Pearson. DCC captures volatility clustering; divergence between the two "
                 "lines indicates vol-regime-dependent correlation behavior."),
                ("Regime Detection Tab", "Scatter plot of all trading days colored by their "
                 "four-state regime classification. Compact table shows time spent in each "
                 "regime and current regime status."),
                ("Markov Forecast Tab", "Transition probability heatmap, steady-state "
                 "distribution, mean days to Crisis from each state, and average run length "
                 "per regime. Current regime row highlighted."),
            ],
            "interpretation": "The DCC-GARCH value diverging above the rolling Pearson value "
                              "indicates vol-regime-driven correlation elevation - a leading "
                              "indicator of potential regime transition. The mean days to Crisis "
                              "from the Markov forecast should be treated as a probabilistic "
                              "estimate, not a precise prediction.",
        },
        {
            "name": "Spillover Analytics",
            "purpose": "Three-method cross-asset spillover identification: Granger causality, "
                       "transfer entropy, and Diebold-Yilmaz FEVD.",
            "components": [
                ("Asset Selection", "Expandable panel for selecting equities, commodities, "
                 "and cross-asset universe. Granger max lag (1–10 days) and DY edge "
                 "threshold (1–15%) configurable."),
                ("Granger Causality Heatmap", "P-value matrix for commodity-to-equity "
                 "causality. Red cells (p < 0.05) indicate statistically significant "
                 "predictive relationships. Top pairs table sorted by p-value."),
                ("Transfer Entropy Matrix", "Net information flow from commodities to "
                 "equities. Captures nonlinear dependencies beyond linear Granger tests."),
                ("Diebold-Yilmaz FEVD Heatmap", "Forecast Error Variance Decomposition. "
                 "Off-diagonal cells show cross-asset shock transmission. Total Spillover "
                 "Index summarizes system-wide connectedness."),
                ("Network Graph", "Node size = net transmitter score. Edge width = "
                 "spillover strength. Green nodes transmit; red nodes absorb. "
                 "Layouts: bipartite, spring, circular."),
                ("Cross-Asset Section", "Granger causality table for rates, FX, "
                 "and fixed income channels (S&P 500, Gold, WTI, TLT, HYG, DXY)."),
            ],
            "interpretation": "Assets with large positive net transmitter scores in the network "
                              "graph are the primary shock sources. Monitor Granger-significant "
                              "commodity-to-equity pairs for trading signals - when a transmitter "
                              "commodity makes a significant move, the Granger-linked equity "
                              "index has historically followed within the identified lag period.",
        },
        {
            "name": "Trade Ideas",
            "purpose": "Regime-triggered cross-asset trade cards with AI Trade Structurer output.",
            "components": [
                ("Regime Filter", "Cards automatically filtered to ideas valid for the "
                 "current regime. 'Show all regimes' checkbox overrides the filter."),
                ("Category Filter", "Filter by trade category: Crisis Hedge, Geopolitical, "
                 "Macro, Growth, India/EM, Private Credit, Fixed Income."),
                ("Trade Cards", "Each card shows: directional thesis (Long/Short), "
                 "category and regime trigger, rationale paragraph, entry signal, "
                 "exit conditions, risk factors, and a live 60-day rolling correlation "
                 "chart for the primary asset pair."),
                ("AI Trade Structurer", "Generates new trade ideas incorporating full "
                 "peer context from Macro Strategist, Geopolitical Analyst, Stress Engineer, "
                 "and Commodities Specialist."),
                ("PDF Report", "One-click PDF generation of all active trade cards "
                 "with regime context and geopolitical summary."),
            ],
            "interpretation": "Use the live correlation chart on each card to confirm the "
                              "pair relationship underpinning the trade is currently intact. "
                              "A correlation that has moved significantly from the expected "
                              "range is a signal to reassess the trade rationale before entry.",
        },
        {
            "name": "Portfolio Stress Test",
            "purpose": "Custom portfolio construction and stress testing against all thirteen "
                       "tracked historical events.",
            "components": [
                ("Asset Selection", "Choose from preset allocations or build custom from "
                 "global indices, commodities, fixed income ETFs, and individual S&P 500 stocks. "
                 "Free-form ticker entry supported."),
                ("Weight Editor", "Per-asset weight inputs auto-normalized to 100%. "
                 "Weight rebalancing available for individual S&P 500 stocks."),
                ("Stress Test Results", "Per-event table: Pre-Event Return, During Return, "
                 "Post-Event Return, Max Drawdown, Sharpe Ratio. Color-coded by return sign."),
                ("Summary Statistics", "Average return across all events, "
                 "historical VaR at 5th percentile, and win rate."),
                ("Visualization", "Event returns heatmap, max drawdown bar chart, "
                 "normalized portfolio path chart."),
            ],
            "interpretation": "The worst 5% return is the most operationally relevant metric "
                              "for risk committee conversations. Compare your portfolio's worst "
                              "event return against its average return to assess the "
                              "asymmetry of downside risk.",
        },
        {
            "name": "Scenario Engine",
            "purpose": "Forward-looking parametric shock propagation with VaR and ES outputs.",
            "components": [
                ("Preset Scenarios", "Six presets: Oil Supply Shock (+40%), Risk-Off Flight, "
                 "Rate Shock (+150bps), Strait Closure, China Hard Landing, Stagflation. "
                 "One-click load populates all sliders."),
                ("Shock Sliders", "Six inputs: Oil (%), Gold (%), Natural Gas (%), "
                 "Copper (%), Yield Shift (bps), DXY (%), Credit Spreads (bps), "
                 "Geopolitical Factor (0–10)."),
                ("Equity Impact Chart", "Estimated return impact on 10 equity indices "
                 "via OLS beta propagation from commodity and macro shocks."),
                ("Commodity Impact Chart", "Estimated return impact on 8 commodity futures."),
                ("VaR / ES Table", "Historical and shocked VaR at 95% and 99% confidence, "
                 "Expected Shortfall at 95% and 99% confidence."),
            ],
            "interpretation": "The OLS beta propagation model uses historical covariance "
                              "to estimate shock transmission. Results should be treated as "
                              "order-of-magnitude estimates rather than precise forecasts. "
                              "ES (Expected Shortfall) is the more conservative risk measure "
                              "and captures average loss in the tail beyond VaR.",
        },
        {
            "name": "Commodities to Watch",
            "purpose": "Live commodity watchlist with intraday, daily, and COT positioning views.",
            "components": [
                ("Live Snapshot Table", "All 17 futures: price, 1D%, 5D%, YTD%, regime label."),
                ("Intraday Indexed Chart", "Selected commodities indexed to 100 at lookback "
                 "start. Rolling 24-hour annualized volatility chart. "
                 "Elevated threshold: 40% annualized."),
                ("Daily Chart", "Indexed daily prices with geopolitical event bands overlaid. "
                 "Rolling 30-day annualized volatility."),
                ("COT Positioning", "Net speculative positioning vs. commercial hedgers "
                 "from CFTC Commitments of Traders data. Extremes table flags crowded "
                 "positioning above +25% or below -25% of open interest."),
            ],
            "interpretation": "The COT extreme signal is a contrarian indicator - extreme "
                              "speculative longs historically precede corrections; extreme "
                              "speculative shorts precede rallies. This signal performs "
                              "best in combination with a regime filter (most reliable in "
                              "Elevated and Crisis regimes).",
        },
        {
            "name": "Strait Watch",
            "purpose": "Real-time disruption monitoring for six critical maritime chokepoints.",
            "components": [
                ("Global KPI Strip", "Total oil at risk (mb/d), global % at risk, "
                 "worst strait name, and Critical/Elevated count."),
                ("Strait Cards", "Per-strait: disruption score (0–100), vessel traffic "
                 "change vs. baseline, risk factors, commodity channels at risk, "
                 "estimated daily trade value at risk."),
                ("Status Classification", "NORMAL (0–25), CAUTION (25–50), "
                 "ELEVATED (50–75), CRITICAL (75–100)."),
                ("Commodity Price Context", "Brent crude price chart with disruption events "
                 "marked, and WTI-Brent spread showing supply route risk premium."),
            ],
            "interpretation": "CRITICAL status on Hormuz has historically correlated with "
                              "Brent price spikes within 5–10 trading days. The WTI-Brent "
                              "spread widening is a real-time market signal of perceived "
                              "supply route risk. Cross-reference with the OVX "
                              "(oil implied volatility) on the Overview page.",
        },
        {
            "name": "Performance Review",
            "purpose": "Model validation and signal accuracy metrics, all computed out-of-sample.",
            "components": [
                ("Regime Detection", "Confusion matrix, balanced accuracy, precision, "
                 "recall, F1 score, and AUC for both rule-based and ML walk-forward classifiers."),
                ("Granger Hit Rates", "Directional accuracy per significant asset pair. "
                 "'Edge' column shows hit rate minus 50% (positive edge = predictive value)."),
                ("Geopolitical Risk Score Lead-Lag", "Correlation between composite risk "
                 "score and VIX at lags of 0–20 days. Peak lag identifies the "
                 "average lead time of the EWS over realized equity volatility."),
                ("COT Contrarian Signals", "Win rate, mean reversion days, and average "
                 "drawdown per commodity for COT extreme positioning signals."),
            ],
            "interpretation": "Only Granger pairs with positive edge are used to generate "
                              "trade ideas. The EWS lead-lag peak (typically 5–7 days) "
                              "provides the operating horizon for acting on elevated "
                              "early warning scores.",
        },
        {
            "name": "AI Analyst",
            "purpose": "Claude Sonnet / GPT-4o chatbot with full live dashboard state injected "
                       "into every query.",
            "components": [
                ("Context Status Bar", "Shows age of injected market context. "
                 "'Refresh context' button forces a live update."),
                ("Suggested Questions", "Ten pre-built queries covering regime interpretation, "
                 "spillover identification, hedge effectiveness, and scenario analysis."),
                ("Chat Interface", "Free-form text input. Responses stream in real time. "
                 "Full conversation history maintained within session."),
                ("Context Inspector", "Expandable panel showing the exact market state text "
                 "injected into every query for full transparency."),
                ("Agent Activity Log", "Structured audit trail of every agent action, "
                 "routing decision, and escalation from the current session."),
            ],
            "interpretation": "The AI Analyst's responses are grounded in the injected "
                              "dashboard state, not the model's training data. Questions "
                              "about current regime, current correlations, or today's "
                              "conditions will be answered using live data. The context "
                              "inspector allows verification of what data the model used.",
        },
        {
            "name": "Insights",
            "purpose": "Plain-language verdicts synthesizing all analytical layers into "
                       "actionable one-sentence directives.",
            "components": [
                ("Overall Stress Card", "Risk score interpretation with action directive "
                 "(no action / monitor / reduce risk) and confidence score."),
                ("Diversification Status Card", "Regime-based hedge effectiveness assessment. "
                 "Green = hedge working. Red = hedge broken."),
                ("Leading Commodity Card", "Identifies the commodity with the highest "
                 "current |correlation| with the equity average."),
                ("Early Warning Card", "EWS composite score interpretation with "
                 "transition probability context."),
                ("Private Credit Risk Card", "HY OAS, BKLN, BDC, CDX HY composite signal."),
                ("India / Rupee Card", "Oil dependency, rupee transmission, Nifty 50 "
                 "driver assessment given current commodity and FX conditions."),
            ],
            "interpretation": "Each insight card shows confidence score alongside the "
                              "directive. Confidence below 60% indicates model uncertainty "
                              "and the directive should be treated as a directional signal "
                              "only. Use Insights as a starting point for deeper analysis "
                              "on the relevant analytical page.",
        },
    ]

    for page in pages:
        story.append(KeepTogether(
            subsub_heading(s, page["name"]) +
            [rule(thickness=0.4, space_before=0, space_after=5)] +
            [Paragraph(page["purpose"], ParagraphStyle(
                "pp", fontName="Helvetica-Oblique", fontSize=9,
                textColor=MUTED, leading=14, spaceAfter=6))]
        ))

        comp_data = [["Component", "Description"]]
        for comp_name, comp_desc in page["components"]:
            comp_data.append([comp_name, comp_desc])
        story.append(make_table(comp_data, [0.28, 0.72]))
        story.append(space(4))
        story += callout_box(s, "INTERPRETATION",
            page["interpretation"],
            bg=colors.HexColor("#FFF8EC"), border=GOLD)
        story.append(space(6))

    story.append(PageBreak())


def build_section7(s, story):
    story += section_heading(s, "7", "AI Workforce Reference",
        "Architecture, dependency structure, and output interpretation for all seven agents.")

    story.append(para(s,
        "The AI Workforce consists of seven specialized agents organized into three sequential "
        "rounds. Agents within a round run in parallel; each round waits for the prior round "
        "to complete before starting. Every agent receives the structured outputs of its "
        "upstream peers before generating its own analysis. This dependency-ordered pipeline "
        "ensures that later agents have full context from earlier ones."))
    story.append(space(6))

    story.append(para(s, "<b>Table 7.1 - Agent Pipeline Architecture</b>"))
    t8 = make_table([
        ["Round", "Agent", "Role", "Inputs Received"],
        ["1", "Signal Auditor", "Calibrates confidence scores from Granger hit rates",
         "Performance Review backtest data"],
        ["1", "Macro Strategist", "Yield curve, inflation, Fed posture, GDP context",
         "FRED macro data + Signal Auditor confidence"],
        ["1", "Geopolitical Analyst", "Active conflict risk, sanctions, strait disruption",
         "Active events + Strait Watch scores + RSS feed"],
        ["2", "Risk Officer", "Synthesizes Round 1 into morning briefing",
         "Signal Auditor + Macro Strategist + Geopolitical Analyst"],
        ["2", "Commodities Specialist", "COT positioning, supply shocks, sector rotation",
         "Commodity data + Signal Auditor + Macro Strategist"],
        ["3", "Stress Engineer", "Scenario stress, tail risk, portfolio shock analysis",
         "All Round 1 + Risk Officer + Commodities Specialist"],
        ["3", "Trade Structurer", "Regime-triggered trade ideas with full peer context",
         "All previous agents' structured outputs"],
    ], [0.08, 0.18, 0.32, 0.42])
    story.append(t8)
    story.append(space(8))

    story.append(para(s,
        "<b>Divergence Detection:</b> The orchestrator automatically compares agent conclusions "
        "across four key dimensions - macro direction, geopolitical risk level, commodity "
        "positioning, and overall risk posture. When two or more agents reach materially "
        "conflicting conclusions (for example, the Macro Strategist is bullish on risk assets "
        "while the Geopolitical Analyst flags acute supply disruption risk), a divergence flag "
        "is generated and displayed on the Overview page in the AI Workforce section."))

    story.append(space(4))
    story.append(para(s,
        "<b>Cache and Invalidation:</b> Agent outputs are cached for one hour. If a correlation "
        "regime change is detected between cache writes, all agents are automatically invalidated "
        "and re-run on the next page load. Manual re-run can be triggered by refreshing the "
        "Overview page."))

    story.append(para(s,
        "<b>Chief Quality Officer (CQO):</b> A separate CQO agent runs on each analytical page "
        "independently of the main workforce pipeline. It audits the page's methodology and "
        "outputs a structured note flagging assumption violations, data limitations, and "
        "model caveat. CQO output is always visible in the page footer."))

    story.append(PageBreak())


def build_section8(s, story):
    story += section_heading(s, "8", "Interpreting Key Outputs",
        "Reference guide for the platform's most important quantitative outputs.")

    outputs = [
        ("Correlation Regime",
         "Four-state classification based on rolling 60-day average absolute pairwise "
         "correlation across all equity-commodity pairs.\n\n"
         "Decorrelated (< 0.15): Low average correlation. Diversification benefits are "
         "strong. Commodity positions provide genuine portfolio diversification.\n\n"
         "Normal (0.15–0.45): Standard market environment. Moderate co-movement. "
         "Traditional allocation strategies function as expected.\n\n"
         "Elevated (0.45–0.60): Above-average co-movement. Diversification benefits "
         "are weakening. Recommend reviewing commodity allocations.\n\n"
         "Crisis (> 0.60): High systemic correlation. Assets moving together. "
         "Diversification benefits largely eliminated. Risk reduction is the priority."),
        ("Composite Risk Score (0–100)",
         "Weighted average of five components: (1) Correlation Level - current average "
         "correlation relative to historical range; (2) Volatility Regime - current realized "
         "vol relative to the 2-year historical distribution; (3) Early Warning Score - "
         "composite of five EWS sub-signals; (4) Macro Posture - yield curve, credit "
         "spread, and Fed stance signals; (5) Geopolitical Stress - active event severity "
         "and strait disruption composite. Scores below 30 represent low-stress environments; "
         "scores above 55 represent elevated-stress environments warranting risk reduction."),
        ("Diebold-Yilmaz Total Spillover Index",
         "Ranges from 0% to 100%. Represents the fraction of total forecast error variance "
         "across all assets that is attributable to cross-asset shocks (as opposed to "
         "own-asset shocks). Values above 50% indicate a highly interconnected system "
         "with limited diversification benefit. Values above 70% indicate systemic stress "
         "comparable to the 2008–2009 Global Financial Crisis."),
        ("Mean Days to Crisis (Markov)",
         "The expected number of trading days to first enter the Crisis regime (Regime 3) "
         "starting from the current regime, computed as the mean first-passage time in the "
         "estimated Markov chain. This is a probabilistic expectation under the assumption "
         "that the transition matrix estimated from the full historical sample is "
         "representative of the current regime environment. It should be treated as "
         "an order-of-magnitude planning horizon, not a precise prediction."),
        ("VaR and ES",
         "Value at Risk (VaR) at confidence level c is the loss threshold not exceeded with "
         "probability c. VaR 95% = the loss level exceeded on 5% of historical observation "
         "days. Expected Shortfall (ES, also known as Conditional VaR or CVaR) at "
         "confidence level c is the expected loss given that the loss exceeds the VaR "
         "threshold. ES is a coherent risk measure and is more conservative than VaR - "
         "it captures the average magnitude of tail losses, not just their threshold."),
    ]

    for title, body_text in outputs:
        story.append(KeepTogether([
            Paragraph(title, ParagraphStyle(
                "ot", fontName="Helvetica-Bold", fontSize=9.5,
                textColor=NAVY, spaceBefore=10, spaceAfter=3)),
            rule(thickness=0.4, space_before=0, space_after=4),
            Paragraph(body_text.replace("\n\n", "<br/><br/>"),
                      ParagraphStyle("ob", fontName="Helvetica", fontSize=9,
                                     textColor=BODY, leading=14, spaceAfter=8,
                                     alignment=TA_JUSTIFY)),
        ]))

    story.append(PageBreak())


def build_section9(s, story):
    story += section_heading(s, "9", "Troubleshooting and FAQ",
        "Common issues and their resolution.")

    faqs = [
        ("AI agents show 'Add API key' message",
         "No Anthropic or OpenAI API key is configured. Add anthropic_api_key (preferred) "
         "or openai_api_key to .streamlit/secrets.toml under the [keys] section. "
         "On Render, add a Secret File at path .streamlit/secrets.toml - do not use "
         "Environment Variables, which are not read by Streamlit's st.secrets mechanism."),
        ("Macro Dashboard shows no data",
         "The FRED API key is missing or invalid. Add fred_api_key to .streamlit/secrets.toml. "
         "A free API key can be obtained at fred.stlouisfed.org/docs/api/api_key.html."),
        ("yfinance returns empty data for a ticker",
         "Yahoo Finance may be temporarily rate-limiting requests. Wait 60 seconds and "
         "reload. If the issue persists for a specific ticker, verify the ticker symbol "
         "is correct using finance.yahoo.com."),
        ("DCC-GARCH model fails to converge",
         "This can occur with very short data windows or assets with extreme return "
         "distributions. The platform falls back to rolling Pearson correlation "
         "automatically. Increase the data lookback window or reduce the number of "
         "selected assets to improve convergence."),
        ("AI Workforce error on Overview page",
         "Check the error details expander below the error message for the full traceback. "
         "Common causes: API key invalid or rate-limited; data fetch failure for a "
         "specific asset; session state inconsistency. Refresh the page to reinitialize "
         "the agent pipeline."),
        ("Granger causality heatmap shows no significant pairs",
         "Increase the maximum lag parameter (default: 5 days) in the Spillover Analytics "
         "asset selection panel. Also try expanding the asset selection - small asset sets "
         "reduce statistical power. Results are sensitive to the data window; ensure "
         "sufficient historical data is available for the selected assets."),
        ("PDF report download not working",
         "Ensure reportlab and Pillow are installed in the environment. "
         "Run pip install reportlab Pillow if not present."),
        ("Streamlit shows 'ScriptRunContext' error on data load",
         "This is a Streamlit caching thread-safety issue. Refresh the page. "
         "If it persists, clear the Streamlit cache from the top-right menu "
         "(hamburger icon → Clear cache)."),
    ]

    faq_data = [["Issue", "Resolution"]]
    for q, a in faqs:
        faq_data.append([q, a])
    story.append(make_table(faq_data, [0.35, 0.65]))
    story.append(space(12))

    story.append(para(s,
        "<b>Contact and Support:</b> This platform is an academic project. For "
        "questions or issue reports, please contact the development team via the "
        "GitHub repository at github.com/HPATKAR/equity-commodities-spillover."))

    story.append(space(20))
    story.append(HRFlowable(width="100%", thickness=0.8, color=RULE, spaceAfter=10))
    story.append(Paragraph(
        "Equity-Commodities Spillover Monitor  ·  Purdue University  ·  "
        "Mitchell E. Daniels, Jr. School of Business  ·  MGMT 69000-120: AI for Finance  ·  "
        "Heramb S. Patkar  ·  Jiahe Miao  ·  Ilian Zalomai  ·  April 2026",
        ParagraphStyle("footer", fontName="Helvetica", fontSize=7.5,
                       textColor=MUTED, alignment=TA_CENTER, leading=12)))


# ════════════════════════════════════════════════════════════════════════════
# MAIN BUILD
# ════════════════════════════════════════════════════════════════════════════
def build():
    doc = SimpleDocTemplate(
        OUTPUT,
        pagesize=A4,
        leftMargin=LM,
        rightMargin=RM,
        topMargin=TM + 10 * mm,
        bottomMargin=BM + 6 * mm,
    )

    s = S()
    story = []

    build_cover(s, story)
    build_toc(s, story)

    build_part1_header(s, story)
    build_section1(s, story)
    build_section2(s, story)
    build_section3(s, story)
    build_section4(s, story)

    build_part2_header(s, story)
    build_section5(s, story)
    build_section6(s, story)
    build_section7(s, story)
    build_section8(s, story)
    build_section9(s, story)

    doc.build(
        story,
        onFirstPage=cover_canvas,
        onLaterPages=page_canvas,
    )
    print(f"PDF saved: {OUTPUT}")


if __name__ == "__main__":
    build()
