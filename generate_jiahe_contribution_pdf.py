"""
Individual Contribution PDF — Jiahe Miao
MGMT 69000-120: AI for Finance · Purdue University

Usage:
    python generate_jiahe_contribution_pdf.py
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, HRFlowable,
    Table, TableStyle, KeepTogether, PageBreak,
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

W, H   = A4
OUTPUT = "Jiahe_Individual_Contribution.pdf"

NAVY  = colors.HexColor("#1E3A5F")
GOLD  = colors.HexColor("#CFB991")
BODY  = colors.HexColor("#2D2D2D")
MUTED = colors.HexColor("#666666")
LIGHT = colors.HexColor("#F5F4F0")
WHITE = colors.white
TEAL  = colors.HexColor("#1A6B6B")

LM = 25 * mm
RM = 25 * mm
TM = 22 * mm
BM = 22 * mm

REPO = "https://github.com/HPATKAR/equity-commodities-spillover"


def rule(color=GOLD, thickness=0.8, spaceB=4, spaceA=8):
    return HRFlowable(width="100%", thickness=thickness,
                      color=color, spaceAfter=spaceA, spaceBefore=spaceB)


def S():
    h1   = ParagraphStyle("H1", fontName="Helvetica-Bold", fontSize=14,
                           textColor=NAVY, spaceBefore=20, spaceAfter=5, leading=18)
    h2   = ParagraphStyle("H2", fontName="Helvetica-Bold", fontSize=11,
                           textColor=NAVY, spaceBefore=12, spaceAfter=3, leading=14)
    h3   = ParagraphStyle("H3", fontName="Helvetica-Bold", fontSize=9.5,
                           textColor=BODY, spaceBefore=8, spaceAfter=2, leading=13)
    body = ParagraphStyle("Body", fontName="Helvetica", fontSize=9.5,
                          textColor=BODY, leading=15, spaceAfter=6,
                          alignment=TA_JUSTIFY)
    blt  = ParagraphStyle("Blt", fontName="Helvetica", fontSize=9.5,
                          textColor=BODY, leading=14, spaceAfter=3,
                          leftIndent=14)
    ctr  = ParagraphStyle("Ctr", fontName="Helvetica", fontSize=9,
                          textColor=MUTED, leading=13, alignment=TA_CENTER)
    lbl  = ParagraphStyle("Lbl", fontName="Helvetica-Bold", fontSize=8,
                          textColor=GOLD)
    return h1, h2, h3, body, blt, ctr, lbl


def tbl(doc_style, headers, rows, col_widths):
    h1, h2, h3, body, blt, ctr, lbl = doc_style
    data = [[Paragraph(f'<b>{h}</b>', lbl) for h in headers]]
    for row in rows:
        data.append([Paragraph(str(c), body) for c in row])
    t = Table(data, colWidths=[w * mm for w in col_widths], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  NAVY),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [LIGHT, WHITE]),
        ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#DDDDDD")),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("FONTSIZE",      (0, 1), (-1, -1), 9),
        ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
    ]))
    return t


def cover(story, sty):
    h1, h2, h3, body, blt, ctr, lbl = sty
    story.append(Spacer(1, 30 * mm))

    story.append(Paragraph("Individual Contribution Summary", ParagraphStyle(
        "CT", fontName="Helvetica-Bold", fontSize=22,
        textColor=NAVY, alignment=TA_CENTER, spaceAfter=5)))

    story.append(Paragraph("Equity-Commodities Spillover Monitor", ParagraphStyle(
        "CS", fontName="Helvetica", fontSize=13,
        textColor=colors.HexColor("#8E6F3E"), alignment=TA_CENTER, spaceAfter=4)))

    story.append(Spacer(1, 5 * mm))
    story.append(rule(GOLD, 1.5))
    story.append(Spacer(1, 5 * mm))

    story.append(Paragraph("Jiahe Miao", ParagraphStyle(
        "CE", fontName="Helvetica-Bold", fontSize=13,
        textColor=BODY, alignment=TA_CENTER, spaceAfter=3)))

    story.append(Paragraph(
        "MSF  ·  Purdue University Daniels School of Business",
        ParagraphStyle("CE2", fontName="Helvetica", fontSize=10,
                       textColor=MUTED, alignment=TA_CENTER, spaceAfter=3)))

    story.append(Paragraph(
        "MGMT 69000-120: AI for Finance  ·  Prof. Cinder Zhang  ·  May 1, 2026",
        ctr))

    story.append(Spacer(1, 6 * mm))
    story.append(Paragraph(
        f'Repository: <link href="{REPO}" color="#2563EB">{REPO}</link>',
        ParagraphStyle("CU", fontName="Helvetica", fontSize=9,
                       textColor=colors.HexColor("#2563EB"), alignment=TA_CENTER)))
    story.append(PageBreak())


def content(story, sty):
    h1, h2, h3, body, blt, ctr, lbl = sty

    # ── Overview ──────────────────────────────────────────────────────────────
    story.append(Paragraph("Overview", h1))
    story.append(rule())
    story.append(Paragraph(
        "My contributions to the Equity-Commodities Spillover Monitor focused on quantitative "
        "methodology research and analytical framework design. I was responsible for the "
        "correlation regime classification system, the regime-conditioned trade structure "
        "framework, the Diebold-Yilmaz spillover methodology specification, and the fixed "
        "income cross-asset signal framework. While Heramb Patkar handled the full software "
        "implementation, my work provided the methodological foundation and domain-specific "
        "analytical design that shaped the core outputs of the dashboard.", body))

    story.append(Paragraph(
        "My role was research-first: I studied the relevant academic literature, designed the "
        "frameworks and calibration logic, and handed well-specified inputs to the development "
        "process. I also performed data quality review that identified and resolved several "
        "data integrity issues before they could affect model outputs.", body))

    # ── Contribution breakdown ─────────────────────────────────────────────────
    story.append(Paragraph("Contribution Areas", h1))
    story.append(rule())

    story.append(Paragraph("1. Correlation Regime Taxonomy — 4-State Classification", h2))
    story.append(Paragraph(
        "I designed the full 4-state correlation regime framework used throughout the dashboard "
        "(Decorrelated / Normal / Elevated / Crisis). My specific contributions were:", body))
    for item in [
        "Defined the 4-state taxonomy and named the regimes with financially meaningful labels",
        "Specified the rolling window length (60-day) and the rationale for that lookback period — long enough for statistical stability, short enough to capture regime transitions",
        "Calibrated the threshold values separating each regime using historical percentile analysis across equity-commodity correlation data (2008–2024)",
        "Designed the hysteresis mechanism (exit threshold = entry threshold − 5 percentage points) to prevent rapid oscillation at regime boundaries",
        "Specified the persistence gate for the Crisis regime: requires ≥60% of a 10-day window to be above the Elevated threshold before classifying as Crisis",
        "Defined the 5-day median smoothing step applied before threshold comparison to reduce noise",
    ]:
        story.append(Paragraph(f"• {item}", blt))

    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph("2. Regime-Conditioned Trade Structures", h2))
    story.append(Paragraph(
        "I designed the 6 regime-conditioned trade structures presented on the Trade Ideas page. "
        "For each structure I specified:", body))
    for item in [
        "The target regime condition (e.g., 'valid only in Regime 3: High Coupling')",
        "Entry trigger: the correlation breakpoint or threshold that activates the trade",
        "Exit rationale: what signal invalidates the trade (regime transition, correlation reversal)",
        "The directional thesis: e.g., long energy / short consumer discretionary under oil supply shock regime",
        "Risk parameters: position sizing rationale relative to correlation confidence",
    ]:
        story.append(Paragraph(f"• {item}", blt))
    story.append(Paragraph(
        "These structures were then implemented as the Trade Ideas page output, with the "
        "current regime classification driving which structures are flagged as active.", body))

    story.append(Paragraph("3. Diebold-Yilmaz FEVD Methodology Research", h2))
    story.append(Paragraph(
        "I researched the Diebold-Yilmaz (2012) variance decomposition methodology and "
        "contributed to its application in the Spillover Analytics page:", body))
    for item in [
        "Studied the original D-Y (2012) paper and synthesized the key design choices: generalized vs. Cholesky FEVD, forecast horizon selection, rolling window",
        "Recommended using the generalized (not Cholesky) decomposition to eliminate VAR ordering dependence — this was adopted in the implementation",
        "Contributed to calibrating the network edge significance threshold: edges below 5% variance share are excluded from the network visualization",
        "Reviewed the directional spillover interpretation framework: helped define what 'net transmitter' vs. 'net receiver' means for equity vs. commodity assets in different regimes",
    ]:
        story.append(Paragraph(f"• {item}", blt))

    story.append(Paragraph("4. Fixed Income Cross-Asset Signal Framework", h2))
    story.append(Paragraph(
        "I defined the fixed income stress signal layer used on the Overview and Insights pages:", body))
    for item in [
        "Selected the 6 FI instruments (TLT, IEF, SHY, LQD, HYG, EMB) and specified what each signals in the context of equity-commodity stress",
        "Defined the interpretation framework: TLT/IEF rising = flight-to-quality (risk-off); HYG/LQD spread widening = credit stress transmission; EMB weakness = EM risk-off aligned with commodity demand destruction",
        "Designed the cross-asset stress signal logic: when TLT rises while HYG falls and commodity vol spikes simultaneously, this is classified as a multi-channel stress convergence event",
    ]:
        story.append(Paragraph(f"• {item}", blt))

    story.append(Paragraph("5. Private Credit Bubble Risk Proxy Validation", h2))
    story.append(Paragraph(
        "I validated the private credit bubble proxy selection on the Insights page. "
        "The proxy basket uses BKLN, ARCC, OBDC, FSK, and JBBB to approximate "
        "private credit market stress. I assessed whether these publicly-traded instruments "
        "provide a meaningful signal for private credit conditions, reviewed their "
        "correlation with direct lending spread indices, and confirmed the composite "
        "score weighting logic was methodologically defensible.", body))

    story.append(Paragraph("6. Data Quality Review", h2))
    story.append(Paragraph(
        "I performed a systematic data quality review across equity and commodity return "
        "series, identifying two material issues:", body))
    story.append(tbl(sty,
        ["Issue", "Finding", "Resolution"],
        [
            ("TOPIX ETF proxy",
             "The original TOPIX ticker (^TOPX) is not available via yfinance for historical data. Using it silently dropped all Japan data.",
             "Replaced with 1306.T (Nomura TOPIX ETF), which provides a reliable yfinance-accessible TOPIX proxy with full history."),
            ("Nickel ticker alignment",
             "The originally specified Nickel ticker (NILSY) was delisted in 2024 and returns empty data post-2023.",
             "Flagged as a known gap; Nickel is excluded from the live commodity universe with a documented note in the code comments."),
        ],
        col_widths=[28, 65, 67],
    ))

    # ── Collaboration ──────────────────────────────────────────────────────────
    story.append(Paragraph("Collaboration & Communication", h1))
    story.append(rule())
    story.append(Paragraph(
        "I worked primarily through async communication with Heramb, delivering research "
        "outputs and framework specifications in written form so they could be implemented "
        "without requiring synchronous sessions. When design questions arose during "
        "implementation (e.g., how to handle insufficient data conditions in the regime "
        "classifier, or how to display regime-conditioned trade ideas when the current regime "
        "has no matching structures), I responded promptly with specific guidance.", body))
    story.append(Paragraph(
        "I also reviewed Ilian Zalomai's geopolitical event descriptions for the War Impact "
        "Map to ensure the narrative framing was consistent with the quantitative outputs "
        "— specifically that the regime-conditioned interpretation of each event aligned "
        "with what the correlation data actually showed during that period.", body))

    # ── Honest assessment ─────────────────────────────────────────────────────
    story.append(Paragraph("Honest Assessment of My Contribution", h1))
    story.append(rule())
    story.append(Paragraph(
        "My contributions were meaningful but narrower in scope than those of my teammates. "
        "I did not write any code; all implementation was handled by Heramb. My role was "
        "to ensure the analytical frameworks were methodologically rigorous before "
        "implementation, and to provide domain-specific research inputs that I had the "
        "background to contribute.", body))
    story.append(Paragraph(
        "The correlation regime framework I designed is central to the dashboard — it drives "
        "the Trade Ideas page, the regime badge on the Command Center, the Markov transition "
        "matrix, and the regime-conditioned scenario propagation. The D-Y methodology research "
        "directly informed one of the project's headline analytical capabilities. The data "
        "quality fixes I identified (TOPIX, Nickel) prevented silent data errors that would "
        "have been difficult to trace in a live system.", body))

    # ── Footer ────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 8 * mm))
    story.append(rule(GOLD, 1.0))
    story.append(Paragraph(
        f"GitHub: {REPO}  ·  "
        "MGMT 69000-120: AI for Finance  ·  Purdue University  ·  May 1, 2026",
        ParagraphStyle("Ft", fontName="Helvetica", fontSize=8,
                       textColor=MUTED, alignment=TA_CENTER)))


def main():
    doc = SimpleDocTemplate(OUTPUT, pagesize=A4,
                            leftMargin=LM, rightMargin=RM,
                            topMargin=TM, bottomMargin=BM,
                            title="Individual Contribution — Jiahe Miao",
                            author="Jiahe Miao")
    sty   = S()
    story = []
    cover(story, sty)
    content(story, sty)
    doc.build(story)
    print(f"Saved → {OUTPUT}")


if __name__ == "__main__":
    main()
