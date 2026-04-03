"""Generate presentation script PDF using ReportLab."""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, HRFlowable, PageBreak
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER

OUTPUT = "Spillover_Monitor_Presentation_Script.pdf"

# ── colour palette ──────────────────────────────────────────────────────────
GOLD   = colors.HexColor("#C9A84C")
DARK   = colors.HexColor("#0E1117")
MID    = colors.HexColor("#1E2330")
MUTED  = colors.HexColor("#8890A1")
WHITE  = colors.white
LIGHT  = colors.HexColor("#D4D8E2")

# ── styles ───────────────────────────────────────────────────────────────────
def make_styles():
    base = getSampleStyleSheet()

    cover_title = ParagraphStyle(
        "CoverTitle",
        fontName="Helvetica-Bold",
        fontSize=22,
        textColor=WHITE,
        spaceAfter=6,
        alignment=TA_CENTER,
    )
    cover_sub = ParagraphStyle(
        "CoverSub",
        fontName="Helvetica",
        fontSize=10,
        textColor=MUTED,
        spaceAfter=4,
        alignment=TA_CENTER,
    )
    cover_tag = ParagraphStyle(
        "CoverTag",
        fontName="Helvetica-Oblique",
        fontSize=9,
        textColor=GOLD,
        spaceAfter=2,
        alignment=TA_CENTER,
    )
    section_label = ParagraphStyle(
        "SectionLabel",
        fontName="Helvetica",
        fontSize=7,
        textColor=GOLD,
        spaceBefore=14,
        spaceAfter=2,
        letterSpacing=1.5,
    )
    section_title = ParagraphStyle(
        "SectionTitle",
        fontName="Helvetica-Bold",
        fontSize=13,
        textColor=WHITE,
        spaceBefore=2,
        spaceAfter=6,
    )
    script = ParagraphStyle(
        "Script",
        fontName="Helvetica",
        fontSize=9,
        textColor=LIGHT,
        leading=15,
        spaceAfter=8,
        leftIndent=10,
        rightIndent=10,
    )
    note = ParagraphStyle(
        "Note",
        fontName="Helvetica-Oblique",
        fontSize=8,
        textColor=MUTED,
        leading=13,
        spaceAfter=6,
        leftIndent=10,
    )
    action = ParagraphStyle(
        "Action",
        fontName="Helvetica-Oblique",
        fontSize=8.5,
        textColor=GOLD,
        leading=13,
        spaceBefore=8,
        spaceAfter=5,
        leftIndent=6,
    )
    team_name = ParagraphStyle(
        "TeamName",
        fontName="Helvetica-Bold",
        fontSize=10,
        textColor=GOLD,
        spaceBefore=8,
        spaceAfter=2,
    )
    team_bio = ParagraphStyle(
        "TeamBio",
        fontName="Helvetica",
        fontSize=9,
        textColor=LIGHT,
        leading=14,
        spaceAfter=4,
        leftIndent=10,
    )
    return dict(
        cover_title=cover_title,
        cover_sub=cover_sub,
        cover_tag=cover_tag,
        section_label=section_label,
        section_title=section_title,
        script=script,
        note=note,
        action=action,
        team_name=team_name,
        team_bio=team_bio,
    )


# ── background canvas ────────────────────────────────────────────────────────
def dark_background(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(DARK)
    canvas.rect(0, 0, A4[0], A4[1], fill=1, stroke=0)
    # gold top bar
    canvas.setFillColor(GOLD)
    canvas.rect(0, A4[1] - 3, A4[0], 3, fill=1, stroke=0)
    # page number
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(MUTED)
    canvas.drawCentredString(A4[0] / 2, 10 * mm, str(doc.page))
    canvas.restoreState()


def cover_background(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(DARK)
    canvas.rect(0, 0, A4[0], A4[1], fill=1, stroke=0)
    # full-width gold accent bar (thicker on cover)
    canvas.setFillColor(GOLD)
    canvas.rect(0, A4[1] - 5, A4[0], 5, fill=1, stroke=0)
    # bottom strip
    canvas.setFillColor(MID)
    canvas.rect(0, 0, A4[0], 18 * mm, fill=1, stroke=0)
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(MUTED)
    canvas.drawString(20 * mm, 7 * mm,
        "Equity-Commodities Spillover Monitor  ·  Purdue Daniels  ·  MGMT 69000-120")
    canvas.restoreState()


# ── section helper ───────────────────────────────────────────────────────────
def section(s, label, title, body_paragraphs, note_text=None, action_text=None):
    """Return list of flowables for one section."""
    items = []
    if action_text:
        items.append(Paragraph(f"\u2192  {action_text}", s["action"]))
    items.append(Paragraph(label.upper(), s["section_label"]))
    items.append(Paragraph(title, s["section_title"]))
    items.append(HRFlowable(width="100%", thickness=0.4,
                             color=GOLD, spaceAfter=8))
    for para in body_paragraphs:
        items.append(Paragraph(para, s["script"]))
    if note_text:
        items.append(Paragraph(f"<i>\u25b6  {note_text}</i>", s["note"]))
    items.append(Spacer(1, 6))
    return items


# ── content ──────────────────────────────────────────────────────────────────
# Each section has an optional "action" field - gold stage direction shown above
# the title. These are navigation/on-screen cues, not spoken text.
SECTIONS = [
    {
        "label": "Opening - The Team",
        "title": "Who We Are",
        "body": [
            "I'm Heramb - equity research at Axis Securities, NISM-certified, BITS Pilani. "
            "Jiahe: capital markets and information systems from Kelley. "
            "Ilian: four years fintech risk ops and Deloitte banking consulting in Frankfurt. "
            "We built a cross-asset intelligence platform. Let me show you what it does.",
        ],
    },
    {
        "label": "Use Case 1 - Overview",
        "title": "\"What is happening in markets right now?\"",
        "action": "Navigate to: Overview  |  Start at the top of the page, scroll slowly downward",
        "body": [
            "This is your morning brief. Before you've touched anything: the current correlation "
            "regime, a live 0-to-100 risk score, best and worst performers across 15 equity "
            "indices and 17 commodity futures, and whether your equity-bond hedge is still intact.",

            "Scroll down - right here is a full briefing written by seven AI agents that ran "
            "before you opened the page. Macro context, geopolitical risk, commodities outlook, "
            "trade ideas, stress assessment. Two hours of analyst work. Every morning. "
            "On every load. Zero manual effort.",
        ],
        "note": "Point to: regime badge, risk score gauge, and the Risk Officer briefing block.",
    },
    {
        "label": "Use Case 2 - Portfolio Stress Test",
        "title": "\"My portfolio is down - have I seen this before?\"",
        "action": "Navigate to: Strategy \u2192 Portfolio Stress Test  |  Add 3-4 assets live",
        "body": [
            "Build your allocation here - any mix of equity indices, commodities, fixed income, "
            "or individual S&P 500 stocks. Hit run.",

            "The platform tests it against every major shock since 2008 - GFC, COVID, Ukraine, "
            "the Fed hiking cycle, the LME nickel squeeze. Pre-event, during, post: returns, "
            "max drawdown, Sharpe ratio per event. You immediately see whether today's move "
            "is a 2022 replay, or something you haven't encountered before.",
        ],
        "note": "Add S&P 500, Gold, TLT live. Point to the event heatmap after running.",
    },
    {
        "label": "Use Case 3 - War Impact Map",
        "title": "\"Which markets are exposed to what's happening geopolitically?\"",
        "action": "Navigate to: Analysis \u2192 War Impact Map  |  Switch to Combined view",
        "body": [
            "195 countries scored by equity market exposure to the three active conflict theaters "
            "right now - Ukraine, Gaza, and Iran-Hormuz. "
            "This column is the model's exposure score. This column is actual index performance "
            "since each conflict started. The gap between them - that's the mispricing.",

            "You're not just watching the news. You're seeing which markets the market "
            "hasn't caught up with yet.",
        ],
        "note": "Hover over a country in the top-25 table. Point to the exposure vs. repricing gap.",
    },
    {
        "label": "Use Case 4 - Scenario Engine",
        "title": "\"What does a Hormuz closure actually do to my book?\"",
        "action": "Navigate to: Strategy \u2192 Scenario Engine  |  Click preset: Geopolitical Escalation",
        "body": [
            "One click - oil shock, yield spike, credit spread widening, DXY move - "
            "propagated through live regression betas to every equity and commodity we cover. "
            "VaR and Expected Shortfall at 95 and 99 percent, with a waterfall showing "
            "which channel is driving which asset.",

            "You can customize every input or build a custom scenario from scratch. "
            "You walk into a risk committee with a number, not a gut feel.",
        ],
        "note": "Load the preset, let it run, point to the waterfall chart and VaR/ES table.",
    },
    {
        "label": "Use Case 5 - Trade Ideas",
        "title": "\"What should I actually do in this regime?\"",
        "action": "Navigate to: Strategy \u2192 Trade Ideas  |  Let the page load with live regime filter",
        "body": [
            "Trade cards are automatically filtered to today's live correlation regime - "
            "ideas that are structurally valid right now, not in a different market environment. "
            "Each card has a thesis, entry signal, risk consideration, and a live correlation "
            "chart confirming the pair relationship is intact today.",

            "The AI Trade Structurer synthesizes macro, geopolitical, stress, and commodities "
            "context simultaneously to generate new ideas on demand.",
        ],
        "note": "Point to the regime label on a trade card. Expand one card fully.",
    },
    {
        "label": "Use Case 6 - AI Analyst",
        "title": "\"I have a specific question. Get me an answer.\"",
        "action": "Navigate to: Research \u2192 AI Analyst  |  Type a live question",
        "body": [
            "Full dashboard state injected into every query - regime, risk score, all live "
            "returns, implied vol, active conflicts, and all seven agent outputs.",

            "Ask it: why is copper decoupling from equities right now? "
            "Or: does this setup look more like 2019 or 2022? "
            "Or: what does a Hormuz closure do to my Japan exposure? "
            "Answer in seconds, grounded in what's on this dashboard today.",
        ],
        "note": "Type a live question relevant to the current regime. Let it answer on screen.",
    },
    {
        "label": "There Is More",
        "title": "The Depth Is There - We Just Didn't Lead With It",
        "action": "Briefly scroll through: Correlation Analysis, Spillover Analytics, Strait Watch",
        "body": [
            "DCC-GARCH correlation regimes, Diebold-Yilmaz spillover networks, Markov "
            "regime transition forecasts, Granger causality grids, COT positioning overlays, "
            "geopolitical event forensics back to 2008, FRED macro intelligence, "
            "maritime chokepoint disruption scores - it is all here.",

            "If a customer gets interested, they can go as deep as they want. "
            "These pages don't need a tour. They reward exploration.",
        ],
    },
]

ABOUT_SECTION = None  # Team intro folded into Opening section

CLOSING = [
    "Finance has always had data. What it has never had is synthesis - "
    "fifteen equity markets, seventeen commodities, geopolitics, macro, and AI "
    "in one coherent view, updated continuously, with no manual work.",

    "That is what this platform does. "
    "If any of those use cases resonated, we are happy to go deeper. Questions?",
]


# ── build PDF ─────────────────────────────────────────────────────────────────
def build():
    doc = SimpleDocTemplate(
        OUTPUT,
        pagesize=A4,
        leftMargin=22 * mm,
        rightMargin=22 * mm,
        topMargin=20 * mm,
        bottomMargin=22 * mm,
    )

    s = make_styles()
    story = []

    # ── cover page ────────────────────────────────────────────────────────────
    story.append(Spacer(1, 50 * mm))
    story.append(Paragraph("EQUITY-COMMODITIES SPILLOVER MONITOR", s["cover_title"]))
    story.append(Spacer(1, 4))
    story.append(Paragraph("Presentation Script", s["cover_sub"]))
    story.append(Spacer(1, 3))
    story.append(Paragraph("Purdue University · Daniels School of Business · MGMT 69000-120", s["cover_tag"]))
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="60%", thickness=0.6, color=GOLD, spaceAfter=6))
    story.append(Paragraph("Heramb S. Patkar · Jiahe Miao · Ilian Zalomai", s["cover_sub"]))
    story.append(PageBreak())

    # ── main sections ─────────────────────────────────────────────────────────
    for sec in SECTIONS:
        story += section(
            s,
            sec["label"],
            sec["title"],
            sec["body"],
            note_text=sec.get("note"),
            action_text=sec.get("action"),
        )

    story.append(Spacer(1, 10))

    # ── closing ───────────────────────────────────────────────────────────────
    story.append(Paragraph("CLOSING".upper(), s["section_label"]))
    story.append(Paragraph("The Pitch", s["section_title"]))
    story.append(HRFlowable(width="100%", thickness=0.4, color=GOLD, spaceAfter=8))
    for para in CLOSING:
        story.append(Paragraph(para, s["script"]))

    doc.build(story, onFirstPage=cover_background, onLaterPages=dark_background)
    print(f"PDF saved: {OUTPUT}")


if __name__ == "__main__":
    build()
