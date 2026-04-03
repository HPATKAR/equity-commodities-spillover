"""
Actionable Insights - PDF Report Generator
Produces an institutional-style PDF of all insight cards with full reasoning.
Uses reportlab (already in requirements).
"""

from __future__ import annotations

import io
import re
from datetime import datetime, date

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, NextPageTemplate,
    Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether,
)

# ── Palette ──────────────────────────────────────────────────────────────────
GOLD  = colors.HexColor("#CFB991")
AGED  = colors.HexColor("#8E6F3E")
BLACK = colors.HexColor("#000000")
WHITE = colors.white
GRAY  = colors.HexColor("#555960")
LGRAY = colors.HexColor("#E8E5E0")
BGWM  = colors.HexColor("#fafaf8")
RED   = colors.HexColor("#c0392b")
GREEN = colors.HexColor("#2e7d32")
AMBER = colors.HexColor("#b7770d")
BLUE  = colors.HexColor("#2980b9")

W, H = A4

_COLOR_MAP = {
    "#1e8449": GREEN,
    "#b7770d": AMBER,
    "#b03a2e": RED,
    "#555960": GRAY,
}


def _html_to_text(html: str) -> str:
    """Strip HTML tags and decode common entities for plain-text PDF paragraphs."""
    text = re.sub(r"<br\s*/?>", "\n", html, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = (
        text.replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&nbsp;", " ")
            .replace("&ndash;", "–")
            .replace("&mdash;", "-")
    )
    return text.strip()


def _styles():
    S = {}
    S["cover_title"] = ParagraphStyle(
        "cover_title", fontSize=22, textColor=WHITE,
        fontName="Helvetica-Bold", alignment=TA_LEFT, leading=28)
    S["cover_sub"] = ParagraphStyle(
        "cover_sub", fontSize=10, textColor=GOLD,
        fontName="Helvetica", alignment=TA_LEFT, leading=14)
    S["cover_meta"] = ParagraphStyle(
        "cover_meta", fontSize=8, textColor=colors.HexColor("#aaaaaa"),
        fontName="Helvetica", alignment=TA_LEFT, leading=11)
    S["section"] = ParagraphStyle(
        "section", fontSize=9, textColor=AGED,
        fontName="Helvetica-Bold", leading=12,
        spaceAfter=3, spaceBefore=8)
    S["headline"] = ParagraphStyle(
        "headline", fontSize=10, textColor=BLACK,
        fontName="Helvetica-Bold", leading=14, spaceAfter=3)
    S["action_label"] = ParagraphStyle(
        "action_label", fontSize=6.5, textColor=AGED,
        fontName="Helvetica-Bold", leading=9,
        spaceAfter=1)
    S["action_text"] = ParagraphStyle(
        "action_text", fontSize=8, textColor=colors.HexColor("#222"),
        fontName="Helvetica", leading=12, spaceAfter=4)
    S["body"] = ParagraphStyle(
        "body", fontSize=7.5, textColor=colors.HexColor("#333"),
        fontName="Helvetica", leading=11, spaceAfter=3)
    S["caption"] = ParagraphStyle(
        "caption", fontSize=7, textColor=GRAY,
        fontName="Helvetica-Oblique", leading=10, spaceAfter=4)
    S["confidence"] = ParagraphStyle(
        "confidence", fontSize=7, textColor=GRAY,
        fontName="Helvetica", leading=10)
    S["disclaimer"] = ParagraphStyle(
        "disclaimer", fontSize=6.5, textColor=GRAY,
        fontName="Helvetica-Oblique", leading=9, spaceAfter=2)
    return S


def _build_doc(buf: io.BytesIO) -> BaseDocTemplate:
    doc = BaseDocTemplate(
        buf, pagesize=A4,
        leftMargin=15 * mm, rightMargin=15 * mm,
        topMargin=15 * mm, bottomMargin=15 * mm,
    )

    _AUTHORS = [
        ("Heramb S. Patkar", "hpatkar.github.io"),
        ("Jiahe Miao",       "linkedin.com/in/jiahe-miao071"),
        ("Ilian Zalomai",    "linkedin.com/in/ilian-zalomai-55iz"),
    ]

    def _cover_bg(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(BLACK)
        canvas.rect(0, 0, W, H, fill=1, stroke=0)
        canvas.setFillColor(GOLD)
        canvas.rect(0, H - 3 * mm, W, 3 * mm, fill=1, stroke=0)

        # Contributors panel
        contrib_y = H - 222 * mm
        canvas.setFillColor(colors.HexColor("#0d0d0d"))
        canvas.setStrokeColor(AGED)
        canvas.setLineWidth(0.5)
        canvas.rect(18 * mm, contrib_y, W - 36 * mm, 22 * mm, fill=1, stroke=1)

        canvas.setFillColor(GOLD)
        canvas.setFont("Helvetica-Bold", 6.5)
        canvas.drawString(24 * mm, contrib_y + 15.5 * mm, "AUTHORS")

        for (name, link), x in zip(_AUTHORS, [24 * mm, 90 * mm, 156 * mm]):
            canvas.setFillColor(WHITE)
            canvas.setFont("Helvetica-Bold", 8.5)
            canvas.drawString(x, contrib_y + 9 * mm, name)
            canvas.setFillColor(GRAY)
            canvas.setFont("Helvetica", 7)
            canvas.drawString(x, contrib_y + 3.5 * mm, link)

        canvas.setFillColor(BLACK)
        canvas.setFont("Helvetica-Bold", 6.5)
        canvas.drawCentredString(
            W / 2, 3.5 * mm,
            "FOR EDUCATIONAL PURPOSES ONLY · NOT INVESTMENT ADVICE · PURDUE UNIVERSITY",
        )
        canvas.restoreState()

    def _body_page(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(GOLD)
        canvas.rect(0, H - 2 * mm, W, 2 * mm, fill=1, stroke=0)
        canvas.setFillColor(LGRAY)
        canvas.rect(0, 0, W, 8 * mm, fill=1, stroke=0)
        canvas.setFont("Helvetica", 6.5)
        canvas.setFillColor(GRAY)
        canvas.drawString(
            15 * mm, 3 * mm,
            f"Actionable Insights  ·  Purdue Daniels  ·  "
            f"Generated {datetime.now().strftime('%d %b %Y %H:%M')}",
        )
        canvas.drawRightString(W - 15 * mm, 3 * mm, f"Page {doc.page}")
        canvas.restoreState()

    cover_frame = Frame(
        0, 0, W, H,
        leftPadding=20 * mm, rightPadding=20 * mm,
        topPadding=50 * mm, bottomPadding=20 * mm,
    )
    body_frame = Frame(15 * mm, 10 * mm, W - 30 * mm, H - 25 * mm)

    doc.addPageTemplates([
        PageTemplate(id="Cover", frames=[cover_frame], onPage=_cover_bg),
        PageTemplate(id="Body",  frames=[body_frame],  onPage=_body_page),
    ])
    return doc


def _insight_block(card: dict, idx: int, S: dict) -> list:
    """Build a KeepTogether block for a single insight card."""
    color_hex = card.get("color", "#555960")
    rl_color  = _COLOR_MAP.get(color_hex, GRAY)

    headline     = card.get("headline", "")
    emoji        = card.get("emoji", "")
    action       = card.get("action", "")
    detail_html  = card.get("detail_html", "")
    confidence   = card.get("confidence", 0)
    conf_label   = card.get("confidence_label", "")

    detail_text = _html_to_text(detail_html)
    conf_color  = GREEN if confidence >= 70 else (AMBER if confidence >= 45 else GRAY)

    elements = []

    # ── Coloured left-border bar (simulated with a thin coloured Table) ───
    bar_data = [[
        Paragraph(f"{emoji}  {headline}", S["headline"]),
    ]]
    bar_table = Table(bar_data, colWidths=[W - 34 * mm])
    bar_table.setStyle(TableStyle([
        ("LEFTPADDING",  (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING",   (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
        ("BACKGROUND",   (0, 0), (-1, -1), BGWM),
        ("BOX",          (0, 0), (-1, -1), 0.5, LGRAY),
        ("LINEBEFORE",   (0, 0), (0, -1),  3,   rl_color),
    ]))
    elements.append(bar_table)
    elements.append(Spacer(1, 2 * mm))

    # ── Action ────────────────────────────────────────────────────────────
    elements.append(Paragraph("▶  ACTION", S["action_label"]))
    elements.append(Paragraph(action, S["action_text"]))

    # ── Confidence ────────────────────────────────────────────────────────
    conf_color_hex = "#2e7d32" if confidence >= 70 else ("#b7770d" if confidence >= 45 else "#555960")
    elements.append(Paragraph(
        f'<font color="{conf_color_hex}">Confidence {confidence}%</font>'
        f'  ·  <font color="#888888">{conf_label}</font>',
        S["confidence"],
    ))
    elements.append(Spacer(1, 1.5 * mm))

    # ── Detail / reasoning ───────────────────────────────────────────────
    if detail_text:
        elements.append(HRFlowable(width="100%", thickness=0.4, color=LGRAY, spaceAfter=3))
        for line in detail_text.split("\n"):
            line = line.strip()
            if line:
                elements.append(Paragraph(line, S["body"]))

    elements.append(Spacer(1, 5 * mm))

    return [KeepTogether(elements)]


def build_insights_pdf(cards: list[dict], start: str, end: str) -> bytes:
    """
    Build and return the Actionable Insights PDF as bytes.

    Parameters
    ----------
    cards   : list of card dicts as returned by _build_insights()
    start   : ISO date string
    end     : ISO date string
    """
    buf = io.BytesIO()
    doc = _build_doc(buf)
    S   = _styles()
    today = date.today()
    story = []

    # ── Cover ─────────────────────────────────────────────────────────────
    story.append(Spacer(1, 20 * mm))
    story.append(Paragraph("Actionable Insights", S["cover_title"]))
    story.append(Paragraph("Market Intelligence Brief", S["cover_title"]))
    story.append(Spacer(1, 6 * mm))
    story.append(Paragraph(
        "Purdue University · Daniels School of Business · MGMT 69000-120",
        S["cover_sub"],
    ))
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph(
        f"Analysis period: {start} to {end}  ·  Generated {today.strftime('%d %B %Y')}",
        S["cover_meta"],
    ))
    story.append(Spacer(1, 6 * mm))
    story.append(Paragraph(
        "Plain-language verdicts derived from live equity and commodity market data. "
        "Each insight combines cross-asset correlation regimes, volatility z-scores, "
        "futures positioning (COT), and macro context into a single actionable recommendation. "
        "Full quantitative reasoning is provided for each signal.",
        S["cover_meta"],
    ))
    story.append(NextPageTemplate("Body"))
    story.append(PageBreak())

    # ── Insight cards ─────────────────────────────────────────────────────
    story.append(Paragraph("MARKET SIGNALS & RECOMMENDED ACTIONS", S["section"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=GOLD, spaceAfter=5))
    story.append(Paragraph(
        f"{len(cards)} signal{'s' if len(cards) != 1 else ''} computed from data "
        f"as of {today.strftime('%d %b %Y')}.",
        S["caption"],
    ))
    story.append(Spacer(1, 3 * mm))

    for i, card in enumerate(cards):
        story += _insight_block(card, i + 1, S)

    # ── Disclaimer ────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=LGRAY))
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph(
        "DISCLAIMER: This report is produced for academic purposes as part of "
        "Purdue University MGMT 69000-120. It does not constitute investment advice. "
        "All data is sourced from public providers (Yahoo Finance, FRED, CFTC) and may "
        "contain errors or omissions. Past performance is not indicative of future results. "
        "Confidence scores are model outputs, not guarantees.",
        S["disclaimer"],
    ))

    doc.build(story)
    buf.seek(0)
    return buf.read()
