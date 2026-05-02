"""
Generate Peer Evaluation PDF for MGMT 69000-120 Week 7.
Evaluator: Heramb S. Patkar
Teammates evaluated: Ilian Zalomai, Jiahe Miao

Usage:
    python generate_peer_eval_pdf.py
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

W, H = A4
OUTPUT = "Spillover_Monitor_Peer_Evaluation.pdf"

NAVY  = colors.HexColor("#1E3A5F")
GOLD  = colors.HexColor("#CFB991")
BODY  = colors.HexColor("#2D2D2D")
MUTED = colors.HexColor("#666666")
LIGHT = colors.HexColor("#F5F4F0")
WHITE = colors.white
GREEN = colors.HexColor("#1A6B3A")
AMBER = colors.HexColor("#8E6F3E")

LM = 25 * mm
RM = 25 * mm
TM = 22 * mm
BM = 22 * mm


def rule(color=GOLD, thickness=0.8, spaceB=4, spaceA=8):
    return HRFlowable(width="100%", thickness=thickness,
                      color=color, spaceAfter=spaceA, spaceBefore=spaceB)


def thin_rule():
    return HRFlowable(width="100%", thickness=0.4,
                      color=colors.HexColor("#DDDDDD"), spaceAfter=6, spaceBefore=6)


def S():
    h1 = ParagraphStyle("H1", fontName="Helvetica-Bold", fontSize=14,
                        textColor=NAVY, spaceBefore=18, spaceAfter=5, leading=18)
    h2 = ParagraphStyle("H2", fontName="Helvetica-Bold", fontSize=10.5,
                        textColor=NAVY, spaceBefore=12, spaceAfter=3, leading=14)
    body = ParagraphStyle("Body", fontName="Helvetica", fontSize=9.5,
                          textColor=BODY, leading=15, spaceAfter=6,
                          alignment=TA_JUSTIFY)
    small = ParagraphStyle("Small", fontName="Helvetica", fontSize=8.5,
                           textColor=MUTED, leading=13, spaceAfter=4)
    center = ParagraphStyle("Center", fontName="Helvetica", fontSize=9,
                            textColor=MUTED, leading=13, alignment=TA_CENTER)
    label = ParagraphStyle("Label", fontName="Helvetica-Bold", fontSize=8,
                           textColor=GOLD, spaceBefore=0, spaceAfter=2)
    return h1, h2, body, small, center, label


def dim_table(scores: dict):
    """Render the five dimension scores as a compact table."""
    dims = [
        ("Technical Contribution",       scores["tech"]),
        ("Reliability & Follow-Through",  scores["rely"]),
        ("Collaboration & Communication", scores["collab"]),
        ("Problem-Solving & Initiative",  scores["problem"]),
        ("Leadership & Team Elevation",   scores["lead"]),
    ]
    data = [[
        Paragraph("<b>Dimension</b>", ParagraphStyle(
            "TH", fontName="Helvetica-Bold", fontSize=8, textColor=GOLD)),
        Paragraph("<b>Score</b>", ParagraphStyle(
            "TH2", fontName="Helvetica-Bold", fontSize=8,
            textColor=GOLD, alignment=TA_CENTER)),
        Paragraph("<b>Visual</b>", ParagraphStyle(
            "TH3", fontName="Helvetica-Bold", fontSize=8,
            textColor=GOLD, alignment=TA_CENTER)),
    ]]
    for name, score in dims:
        filled = "●" * score + "○" * (5 - score)
        data.append([
            Paragraph(name, ParagraphStyle(
                "DimName", fontName="Helvetica", fontSize=9, textColor=BODY)),
            Paragraph(f"{score} / 5", ParagraphStyle(
                "DimScore", fontName="Helvetica-Bold", fontSize=9,
                textColor=NAVY, alignment=TA_CENTER)),
            Paragraph(filled, ParagraphStyle(
                "DimViz", fontName="Helvetica", fontSize=9,
                textColor=GOLD, alignment=TA_CENTER)),
        ])
    t = Table(data, colWidths=[105 * mm, 20 * mm, 30 * mm], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  NAVY),
        ("TOPPADDING",    (0, 0), (-1, 0),  5),
        ("BOTTOMPADDING", (0, 0), (-1, 0),  5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 7),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 7),
        ("TOPPADDING",    (0, 1), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 4),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [LIGHT, WHITE]),
        ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#DDDDDD")),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN",         (1, 0), (-1, -1), "CENTER"),
    ]))
    return t


def cover(story, h1, h2, body, small, center, label):
    story.append(Spacer(1, 28 * mm))
    story.append(Paragraph("Peer Evaluation", ParagraphStyle(
        "CT", fontName="Helvetica-Bold", fontSize=22,
        textColor=NAVY, alignment=TA_CENTER, spaceAfter=5)))
    story.append(Paragraph("Equity-Commodities Spillover Monitor", ParagraphStyle(
        "CS", fontName="Helvetica", fontSize=13,
        textColor=AMBER, alignment=TA_CENTER, spaceAfter=4)))
    story.append(Spacer(1, 5 * mm))
    story.append(rule(GOLD, 1.5))
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph("Evaluator: Heramb S. Patkar", ParagraphStyle(
        "CE", fontName="Helvetica-Bold", fontSize=12,
        textColor=BODY, alignment=TA_CENTER, spaceAfter=3)))
    story.append(Paragraph(
        "MGMT 69000-120: AI for Finance  ·  Prof. Cinder Zhang  ·  April 30, 2026",
        center))
    story.append(Spacer(1, 6 * mm))

    # Point summary box
    summary = Table(
        [[
            Paragraph("Ilian Zalomai", ParagraphStyle(
                "SN", fontName="Helvetica-Bold", fontSize=10, textColor=WHITE, alignment=TA_CENTER)),
            Paragraph("Jiahe Miao", ParagraphStyle(
                "SN2", fontName="Helvetica-Bold", fontSize=10, textColor=WHITE, alignment=TA_CENTER)),
            Paragraph("Total", ParagraphStyle(
                "SN3", fontName="Helvetica-Bold", fontSize=10, textColor=GOLD, alignment=TA_CENTER)),
        ],
        [
            Paragraph("55 pts", ParagraphStyle(
                "SP", fontName="Helvetica-Bold", fontSize=18, textColor=GOLD, alignment=TA_CENTER)),
            Paragraph("45 pts", ParagraphStyle(
                "SP2", fontName="Helvetica-Bold", fontSize=18, textColor=GOLD, alignment=TA_CENTER)),
            Paragraph("100 pts", ParagraphStyle(
                "SP3", fontName="Helvetica-Bold", fontSize=18, textColor=WHITE, alignment=TA_CENTER)),
        ]],
        colWidths=[52 * mm, 52 * mm, 42 * mm],
    )
    summary.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), NAVY),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("GRID",          (0, 0), (-1, -1), 0.5, colors.HexColor("#2A3A5F")),
        ("LINEAFTER",     (1, 0), (1, -1),  1.0, GOLD),
    ]))
    story.append(summary)
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph(
        "Point difference: 10 pts (satisfies ≥5 pt differential requirement)",
        ParagraphStyle("Req", fontName="Helvetica-Oblique", fontSize=8,
                       textColor=MUTED, alignment=TA_CENTER)))
    story.append(PageBreak())


def teammate_block(story, name, points, scores, justification,
                   h1, h2, body, small, center, label):
    story.append(KeepTogether([
        Paragraph(f"Teammate: {name}", h1),
        rule(),
    ]))

    # Points banner
    pts = Table(
        [[Paragraph(f"Points Allocated", ParagraphStyle(
              "PL", fontName="Helvetica", fontSize=9, textColor=MUTED)),
          Paragraph(f"{points} / 100", ParagraphStyle(
              "PV", fontName="Helvetica-Bold", fontSize=14, textColor=NAVY,
              alignment=TA_CENTER))]],
        colWidths=[120 * mm, 35 * mm],
    )
    pts.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), LIGHT),
        ("TOPPADDING",    (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("BOX",           (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(pts)
    story.append(Spacer(1, 4 * mm))

    story.append(Paragraph("Dimension Scores", h2))
    story.append(dim_table(scores))
    story.append(Spacer(1, 4 * mm))

    story.append(Paragraph("Written Justification", h2))
    for para in justification:
        story.append(Paragraph(para, body))

    story.append(Spacer(1, 6 * mm))


def build(story):
    h1, h2, body, small, center, label = S()

    cover(story, h1, h2, body, small, center, label)

    # ── Ilian Zalomai ─────────────────────────────────────────────────────────
    teammate_block(
        story,
        name="Ilian Zalomai",
        points=55,
        scores=dict(tech=4, rely=5, collab=5, problem=4, lead=4),
        justification=[
            "Ilian's most concrete contribution was the War Impact Map scoring framework — "
            "he defined per-conflict baseline scores, country-level commodity exposure weights, "
            "and the concurrent-war amplifier logic that accounts for how Ukraine, Gaza/Red Sea, "
            "and Iran/Hormuz compound each other's market impact. This was specific and "
            "implementable, and I translated it directly into src/analysis/conflict_model.py "
            "and src/pages/war_impact_map.py without needing to redesign the underlying logic.",

            "He also authored the full geopolitical event catalog — all 13 events from GFC "
            "(2008) through the Iran/Hormuz Crisis (2026), each with market-specific impact "
            "descriptions — and designed the Strait Watch chokepoint framework (five chokepoints, "
            "disruption scoring methodology, EIA integration). These became directly addressable "
            "features in the dashboard. His narrative text across the dashboard (page intros, "
            "scenario descriptions, takeaway blocks) meaningfully raised the analytical quality "
            "of what the tool communicates, even though it was content rather than code.",

            "Ilian was the most reliable collaborator on the team — he consistently delivered "
            "what he committed to, on time, and his geopolitical domain expertise added signal "
            "that genuinely could not have come from the technical side of the project. "
            "His proactive communication and responsiveness throughout the build made "
            "coordination straightforward and his narrative contributions elevated the "
            "overall quality of what the dashboard communicates.",
        ],
        h1=h1, h2=h2, body=body, small=small, center=center, label=label,
    )

    story.append(thin_rule())

    # ── Jiahe Miao ────────────────────────────────────────────────────────────
    teammate_block(
        story,
        name="Jiahe Miao",
        points=45,
        scores=dict(tech=4, rely=4, collab=4, problem=4, lead=3),
        justification=[
            "Jiahe's primary contribution was methodological: she designed the four-state "
            "correlation regime taxonomy (Low / Moderate / Elevated / Crisis), specified the "
            "rolling window thresholds and lookback methodology, and calibrated the six "
            "regime-conditioned trade structures with entry, exit, and correlation breakpoint "
            "rationale. These specifications informed the implementation in "
            "src/analysis/correlations.py and src/pages/trade_ideas.py. She also researched "
            "the Diebold-Yilmaz FEVD methodology, contributing to how I approached network "
            "edge threshold calibration and directional spillover interpretation.",

            "She also defined the Fixed Income cross-asset signal framework "
            "(TLT / HYG / LQD / EMB metrics and their interpretation logic), validated the "
            "private credit bubble proxy selection (BKLN, ARCC, OBDC, FSK, JBBB), and "
            "performed data quality review across equity and commodity return series — "
            "flagging the TOPIX ETF proxy (1306.T) and the Nickel ticker alignment (NILSY). "
            "These were genuine catches that improved data integrity.",

            "Jiahe engaged seriously with the quantitative methodology throughout the project "
            "and was reliable in delivering her research commitments. Her domain knowledge in "
            "econometrics and structured finance added credibility to the analytical framework "
            "and her data quality catches — particularly the TOPIX ETF proxy and Nickel ticker "
            "alignment — prevented errors that would have been difficult to trace once embedded.",
        ],
        h1=h1, h2=h2, body=body, small=small, center=center, label=label,
    )

    # ── Footer ────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 8 * mm))
    story.append(rule(GOLD, 1.0))
    story.append(Paragraph(
        "Evaluation submitted independently. Scores reflect direct observation of "
        "contributions over the full project period (2026-03-12 → 2026-04-30).",
        ParagraphStyle("Footer", fontName="Helvetica-Oblique", fontSize=8,
                       textColor=MUTED, alignment=TA_CENTER, leading=12)))
    story.append(Paragraph(
        "MGMT 69000-120: AI for Finance  ·  Purdue University  ·  April 30, 2026",
        ParagraphStyle("Footer2", fontName="Helvetica", fontSize=8,
                       textColor=MUTED, alignment=TA_CENTER, leading=12, spaceAfter=0)))


def main():
    doc = SimpleDocTemplate(
        OUTPUT, pagesize=A4,
        leftMargin=LM, rightMargin=RM,
        topMargin=TM, bottomMargin=BM,
        title="Peer Evaluation — Heramb S. Patkar",
        author="Heramb S. Patkar",
    )
    story = []
    build(story)
    doc.build(story)
    print(f"Saved → {OUTPUT}")


if __name__ == "__main__":
    main()
