"""
Peer Evaluation PDF — Jiahe Miao evaluating Heramb S. Patkar and Ilian Zalomai
MGMT 69000-120: AI for Finance · Purdue University

Usage:
    python generate_jiahe_peer_eval_pdf.py
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
OUTPUT = "Jiahe_Peer_Evaluation.pdf"

NAVY  = colors.HexColor("#1E3A5F")
GOLD  = colors.HexColor("#CFB991")
BODY  = colors.HexColor("#2D2D2D")
MUTED = colors.HexColor("#666666")
LIGHT = colors.HexColor("#F5F4F0")
WHITE = colors.white
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
                      color=colors.HexColor("#DDDDDD"),
                      spaceAfter=6, spaceBefore=6)


def S():
    h1   = ParagraphStyle("H1", fontName="Helvetica-Bold", fontSize=14,
                           textColor=NAVY, spaceBefore=18, spaceAfter=5, leading=18)
    h2   = ParagraphStyle("H2", fontName="Helvetica-Bold", fontSize=11,
                           textColor=NAVY, spaceBefore=10, spaceAfter=3, leading=14)
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
    note = ParagraphStyle("Note", fontName="Helvetica-Oblique", fontSize=8,
                          textColor=MUTED, leading=12, spaceAfter=4)
    return h1, h2, body, blt, ctr, lbl, note


def dim_table(scores: dict):
    _REG_COL = {1: "#C0392B", 2: "#E67E22", 3: "#E8A838", 4: "#2980B9", 5: "#27AE60"}
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
        col   = _REG_COL.get(score, "#888")
        filled = "●" * score + "○" * (5 - score)
        data.append([
            Paragraph(name, ParagraphStyle(
                "DN", fontName="Helvetica", fontSize=9, textColor=BODY)),
            Paragraph(f"{score} / 5", ParagraphStyle(
                "DS", fontName="Helvetica-Bold", fontSize=9,
                textColor=colors.HexColor(col), alignment=TA_CENTER)),
            Paragraph(filled, ParagraphStyle(
                "DV", fontName="Helvetica", fontSize=9,
                textColor=colors.HexColor(col), alignment=TA_CENTER)),
        ])
    t = Table(data, colWidths=[105 * mm, 20 * mm, 30 * mm], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  NAVY),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 7),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 7),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [LIGHT, WHITE]),
        ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#DDDDDD")),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN",         (1, 0), (-1, -1), "CENTER"),
    ]))
    return t


def cover(story, sty):
    h1, h2, body, blt, ctr, lbl, note = sty
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

    story.append(Paragraph("Evaluator: Jiahe Miao", ParagraphStyle(
        "CE", fontName="Helvetica-Bold", fontSize=12,
        textColor=BODY, alignment=TA_CENTER, spaceAfter=3)))
    story.append(Paragraph(
        "MGMT 69000-120: AI for Finance  ·  Prof. Cinder Zhang  ·  May 1, 2026",
        ctr))
    story.append(Spacer(1, 6 * mm))

    # Point summary box
    summary = Table(
        [[
            Paragraph("Heramb S. Patkar", ParagraphStyle(
                "SN", fontName="Helvetica-Bold", fontSize=10,
                textColor=WHITE, alignment=TA_CENTER)),
            Paragraph("Ilian Zalomai", ParagraphStyle(
                "SN2", fontName="Helvetica-Bold", fontSize=10,
                textColor=WHITE, alignment=TA_CENTER)),
            Paragraph("Total", ParagraphStyle(
                "SN3", fontName="Helvetica-Bold", fontSize=10,
                textColor=GOLD, alignment=TA_CENTER)),
        ],
        [
            Paragraph("72 pts", ParagraphStyle(
                "SP", fontName="Helvetica-Bold", fontSize=18,
                textColor=GOLD, alignment=TA_CENTER)),
            Paragraph("28 pts", ParagraphStyle(
                "SP2", fontName="Helvetica-Bold", fontSize=18,
                textColor=GOLD, alignment=TA_CENTER)),
            Paragraph("100 pts", ParagraphStyle(
                "SP3", fontName="Helvetica-Bold", fontSize=18,
                textColor=WHITE, alignment=TA_CENTER)),
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
        "Point difference: 44 pts (satisfies ≥5 pt differential requirement)",
        ParagraphStyle("Req", fontName="Helvetica-Oblique", fontSize=8,
                       textColor=MUTED, alignment=TA_CENTER)))
    story.append(PageBreak())


def teammate_block(story, sty, name, points, scores, justification):
    h1, h2, body, blt, ctr, lbl, note = sty
    story.append(KeepTogether([
        Paragraph(f"Teammate: {name}", h1),
        rule(),
    ]))

    pts_tbl = Table(
        [[Paragraph("Points Allocated", ParagraphStyle(
              "PL", fontName="Helvetica", fontSize=9, textColor=MUTED)),
          Paragraph(f"{points} / 100", ParagraphStyle(
              "PV", fontName="Helvetica-Bold", fontSize=14, textColor=NAVY,
              alignment=TA_CENTER))]],
        colWidths=[120 * mm, 35 * mm],
    )
    pts_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), LIGHT),
        ("TOPPADDING",    (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("BOX",           (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(pts_tbl)
    story.append(Spacer(1, 4 * mm))

    story.append(Paragraph("Dimension Scores", h2))
    story.append(dim_table(scores))
    story.append(Spacer(1, 4 * mm))

    story.append(Paragraph("Written Justification", h2))
    for para in justification:
        story.append(Paragraph(para, body))
    story.append(Spacer(1, 6 * mm))


def build(story, sty):
    h1, h2, body, blt, ctr, lbl, note = sty

    # ── Heramb S. Patkar ──────────────────────────────────────────────────────
    teammate_block(
        story, sty,
        name="Heramb S. Patkar",
        points=72,
        scores=dict(tech=5, rely=5, collab=4, problem=5, lead=5),
        justification=[
            "Heramb was responsible for the entire software implementation of this project — "
            "every page, every data integration, every analytical model, and the full "
            "deployment stack. From my perspective working with him on the quantitative "
            "methodology side, what stood out was not just the volume of work but the quality "
            "of the translation: when I handed him the correlation regime framework "
            "(thresholds, hysteresis logic, persistence gate), he implemented it correctly "
            "on the first pass and extended it with additional features I had not specified, "
            "such as the Markov transition matrix and mean first passage time calculation.",

            "His reliability was exceptional throughout the project. Every deadline was met, "
            "every component I expected to see in the dashboard appeared, and when I raised "
            "data quality issues (the TOPIX proxy and the Nickel ticker), they were resolved "
            "quickly. He also proactively flagged when my regime threshold calibration "
            "needed adjustment after seeing how it performed on live data — that kind of "
            "initiative, to validate the methodology against real outputs and loop back, "
            "is not something that could be assumed.",

            "He drove the project's technical direction entirely and made the key "
            "architectural decisions — the choice of Diebold-Yilmaz over simpler spillover "
            "proxies, the decision to build a 7-agent AI pipeline with a quality officer "
            "remediation loop, the parallel data loading optimization — all of these were "
            "his calls, and all of them were good ones. Without his execution there would "
            "be no project.",
        ],
    )

    story.append(thin_rule())

    # ── Ilian Zalomai ─────────────────────────────────────────────────────────
    teammate_block(
        story, sty,
        name="Ilian Zalomai",
        points=28,
        scores=dict(tech=3, rely=4, collab=4, problem=3, lead=3),
        justification=[
            "Ilian's contribution was focused on the geopolitical intelligence layer of "
            "the project — the War Impact Map scoring framework, the 13-event catalog "
            "(GFC through Iran/Hormuz), the Strait Watch maritime chokepoint framework, "
            "and the narrative text throughout the dashboard. This was domain-specific "
            "work that required genuine knowledge of geopolitical risk and commodity "
            "market history, and his outputs were concrete enough to be implemented "
            "directly without extensive revision.",

            "He was reliable in delivering what he committed to and was "
            "communicative throughout the project. His event descriptions and scenario "
            "narratives raised the quality of what the dashboard communicates to a "
            "non-technical reader, which matters for a course project that is evaluated "
            "by industry panelists. His concurrent-war amplifier methodology — accounting "
            "for how multiple simultaneous conflicts compound each other's market impact "
            "— was a genuinely thoughtful analytical contribution.",

            "His technical contribution scores lower because his role was research and "
            "narrative rather than code or quantitative model design. His problem-solving "
            "and leadership scores reflect that his domain was well-defined from the start "
            "— he executed within that domain well but did not need to make the ambiguous "
            "architectural or methodological decisions that drove the project's overall "
            "analytical quality.",
        ],
    )

    # ── Footer ────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 8 * mm))
    story.append(rule(GOLD, 1.0))
    story.append(Paragraph(
        "Evaluation submitted independently. Scores reflect direct observation of "
        "contributions over the full project period (March – May 2026).",
        ParagraphStyle("Ft1", fontName="Helvetica-Oblique", fontSize=8,
                       textColor=MUTED, alignment=TA_CENTER, leading=12)))
    story.append(Paragraph(
        "MGMT 69000-120: AI for Finance  ·  Purdue University  ·  May 1, 2026",
        ParagraphStyle("Ft2", fontName="Helvetica", fontSize=8,
                       textColor=MUTED, alignment=TA_CENTER, leading=12)))


def main():
    doc = SimpleDocTemplate(OUTPUT, pagesize=A4,
                            leftMargin=LM, rightMargin=RM,
                            topMargin=TM, bottomMargin=BM,
                            title="Peer Evaluation — Jiahe Miao",
                            author="Jiahe Miao")
    sty   = S()
    story = []
    cover(story, sty)
    build(story, sty)
    doc.build(story)
    print(f"Saved → {OUTPUT}")


if __name__ == "__main__":
    main()
