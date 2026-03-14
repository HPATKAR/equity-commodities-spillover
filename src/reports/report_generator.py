"""
Institutional-grade PDF report generator.
Purdue University · Daniels School of Business
MGMT 69000-120 · AI for Finance

Narrative structure:
  1. Cover page          — branding, contributors, metadata
  2. Executive Summary   — regime panel + interpretation
  3. Regime Timeline     — avg |corr| history with regime shading   [CHART]
  4. Correlation Matrix  — cross-asset heatmap                       [CHART]
  5. Market Stress       — composite stress index 0-100              [CHART]
  6. Commodity Performance — indexed price returns                   [CHART]
  7. Trade Ideas         — active cards + pair correlation chart     [CHART]
  8. Geopolitical Context — event cards
  9. Methodology & Data Sources
  10. Disclaimer
"""

from __future__ import annotations

import io
from datetime import datetime, date
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate,
    Paragraph, Spacer, Table, TableStyle,
    HRFlowable, NextPageTemplate, PageBreak, KeepTogether,
    Image as RLImage,
)

# ── Purdue palette ──────────────────────────────────────────────────────────
GOLD   = colors.HexColor("#CFB991")
GOLD_M = colors.HexColor("#DAAA00")
AGED   = colors.HexColor("#8E6F3E")
BLACK  = colors.HexColor("#000000")
WHITE  = colors.white
DARK   = colors.HexColor("#1a1a1a")
GRAY   = colors.HexColor("#555960")
LGRAY  = colors.HexColor("#E8E5E0")
BGWARM = colors.HexColor("#fafaf8")
RED    = colors.HexColor("#c0392b")
GREEN  = colors.HexColor("#2e7d32")
ORANGE = colors.HexColor("#e67e22")
BLUE   = colors.HexColor("#2980b9")

W, H = A4

REGIME_NAMES  = {0: "Decorrelated", 1: "Normal", 2: "Elevated", 3: "Crisis"}
REGIME_COLORS = {0: GREEN, 1: GRAY, 2: ORANGE, 3: RED}
CAT_COLORS    = {
    "Crisis Hedge": RED,
    "Geopolitical": ORANGE,
    "Macro":        BLUE,
    "Growth":       GREEN,
}

# hex versions for matplotlib
_R_HEX = {0: "#2e7d32", 1: "#555960", 2: "#e67e22", 3: "#c0392b"}
_CAT_HEX = {
    "Crisis Hedge": "#c0392b",
    "Geopolitical": "#e67e22",
    "Macro":        "#2980b9",
    "Growth":       "#2e7d32",
}
_PALETTE_HEX = [
    "#000000", "#CFB991", "#8E6F3E", "#c0392b",
    "#2e7d32", "#2980b9", "#DAAA00", "#8e44ad", "#16a085", "#e67e22",
]


# ── Matplotlib helpers ──────────────────────────────────────────────────────

def _mpl_theme():
    plt.rcParams.update({
        "font.family":        "DejaVu Sans",
        "axes.facecolor":     "#fafaf8",
        "figure.facecolor":   "white",
        "axes.edgecolor":     "#E8E5E0",
        "axes.linewidth":     0.5,
        "axes.grid":          True,
        "grid.color":         "#E8E5E0",
        "grid.linewidth":     0.4,
        "grid.alpha":         0.8,
        "axes.labelcolor":    "#333333",
        "axes.labelsize":     7,
        "xtick.color":        "#555960",
        "ytick.color":        "#555960",
        "xtick.labelsize":    6.5,
        "ytick.labelsize":    6.5,
        "axes.titlesize":     8.5,
        "axes.titleweight":   "bold",
        "axes.titlecolor":    "#000000",
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "legend.fontsize":    6.5,
        "legend.framealpha":  0.88,
        "legend.edgecolor":   "#E8E5E0",
    })


def _fig_to_rl(fig, w_mm: float, h_mm: float) -> RLImage:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return RLImage(buf, width=w_mm * mm, height=h_mm * mm)


def _shade_regimes(ax, regimes: pd.Series, alpha: float = 0.14):
    """Paint background bands by correlation regime."""
    if regimes is None or regimes.empty:
        return
    prev_r = int(regimes.iloc[0])
    prev_t = regimes.index[0]
    for i in range(1, len(regimes)):
        r = int(regimes.iloc[i])
        if r != prev_r:
            ax.axvspan(prev_t, regimes.index[i],
                       color=_R_HEX[prev_r], alpha=alpha, lw=0)
            prev_r, prev_t = r, regimes.index[i]
    ax.axvspan(prev_t, regimes.index[-1],
               color=_R_HEX[prev_r], alpha=alpha, lw=0)


# ── Chart 1: Regime timeline ────────────────────────────────────────────────

def _chart_regime_timeline(
    avg_corr: pd.Series,
    regimes: pd.Series,
    w_mm: float = 170,
    h_mm: float = 72,
) -> RLImage:
    _mpl_theme()
    fig, ax = plt.subplots(figsize=(w_mm / 25.4, h_mm / 25.4))

    if avg_corr.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, color="#555960")
        return _fig_to_rl(fig, w_mm, h_mm)

    _shade_regimes(ax, regimes)
    ax.plot(avg_corr.index, avg_corr.values,
            color="#000000", lw=1.3, zorder=3, label="Avg |Corr|")
    ax.scatter([avg_corr.index[-1]], [avg_corr.iloc[-1]],
               color="#CFB991", s=30, zorder=5)

    ax.set_title("Avg Absolute Cross-Asset Correlation  ·  60-Day Rolling")
    ax.set_ylabel("|Corr|")
    ax.set_ylim(0, min(1.05, avg_corr.quantile(0.99) * 1.18))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    patches = [mpatches.Patch(color=_R_HEX[r], alpha=0.55, label=REGIME_NAMES[r])
               for r in [0, 1, 2, 3]]
    ax.legend(handles=patches, loc="upper left", ncol=4,
              handlelength=1.0, handleheight=0.8)
    fig.tight_layout(pad=0.6)
    return _fig_to_rl(fig, w_mm, h_mm)


# ── Chart 2: Cross-asset correlation heatmap ───────────────────────────────

_EQ_HMAP  = ["S&P 500", "Nasdaq 100", "Eurostoxx 50", "DAX",
             "FTSE 100", "Nikkei 225", "Hang Seng", "Sensex"]
_CMD_HMAP = ["WTI Crude Oil", "Brent Crude", "Natural Gas",
             "Gold", "Silver", "Copper", "Wheat", "Soybeans"]


def _chart_corr_heatmap(
    eq_r: pd.DataFrame,
    cmd_r: pd.DataFrame,
    w_mm: float = 170,
    h_mm: float = 115,
) -> RLImage:
    _mpl_theme()
    eq_cols  = [c for c in _EQ_HMAP  if c in eq_r.columns]
    cmd_cols = [c for c in _CMD_HMAP if c in cmd_r.columns]

    fig, ax = plt.subplots(figsize=(w_mm / 25.4, h_mm / 25.4))

    if not eq_cols or not cmd_cols:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                transform=ax.transAxes, color="#555960")
        return _fig_to_rl(fig, w_mm, h_mm)

    combined = pd.concat([eq_r[eq_cols], cmd_r[cmd_cols]], axis=1).dropna()
    matrix   = combined.corr().loc[eq_cols, cmd_cols].values

    vmax = max(abs(matrix.max()), abs(matrix.min()), 0.3)
    im   = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(cmd_cols)))
    ax.set_xticklabels([c.replace(" ", "\n") for c in cmd_cols], fontsize=5.8)
    ax.set_yticks(range(len(eq_cols)))
    ax.set_yticklabels(eq_cols, fontsize=6.2)
    ax.tick_params(bottom=True, top=False, labelbottom=True, labeltop=False)

    for i in range(len(eq_cols)):
        for j in range(len(cmd_cols)):
            v   = matrix[i, j]
            col = "white" if abs(v) > 0.42 else "#1a1a1a"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=5.2, color=col, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.022, pad=0.02)
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label("Pearson r (full sample)", fontsize=6.2)

    ax.set_title("Cross-Asset Correlation Matrix  ·  Equities × Commodities")
    ax.set_facecolor("white")
    ax.grid(False)
    fig.tight_layout(pad=0.6)
    return _fig_to_rl(fig, w_mm, h_mm)


# ── Chart 3: Composite stress index ────────────────────────────────────────

def _chart_stress_index(
    stress: pd.Series,
    w_mm: float = 170,
    h_mm: float = 72,
) -> RLImage:
    _mpl_theme()
    fig, ax = plt.subplots(figsize=(w_mm / 25.4, h_mm / 25.4))

    if stress is None or stress.empty:
        ax.text(0.5, 0.5, "No stress data", ha="center", va="center",
                transform=ax.transAxes, color="#555960")
        return _fig_to_rl(fig, w_mm, h_mm)

    ax.axhspan(0,  40, color="#2e7d32", alpha=0.07, lw=0)
    ax.axhspan(40, 60, color="#f39c12", alpha=0.07, lw=0)
    ax.axhspan(60, 80, color="#e67e22", alpha=0.07, lw=0)
    ax.axhspan(80, 100, color="#c0392b", alpha=0.07, lw=0)

    for y, lbl, col in [(40, "Elevated", "#f39c12"),
                         (60, "High",     "#e67e22"),
                         (80, "Crisis",   "#c0392b")]:
        ax.axhline(y, color=col, lw=0.6, ls="--", alpha=0.55)
        ax.text(stress.index[max(0, len(stress) // 50)], y + 1.5,
                lbl, fontsize=5.5, color=col, alpha=0.75)

    ax.fill_between(stress.index, stress.values, alpha=0.10, color="#8E6F3E")
    ax.plot(stress.index, stress.values, color="#8E6F3E", lw=1.3, zorder=3)

    cur = float(stress.iloc[-1])
    ax.scatter([stress.index[-1]], [cur], color="#CFB991", s=32, zorder=5)
    ax.annotate(f"  {cur:.0f}", xy=(stress.index[-1], cur),
                fontsize=6.5, color="#8E6F3E", fontweight="bold",
                xycoords="data", ha="left", va="center")

    ax.set_ylim(0, 100)
    ax.set_title("Composite Market Stress Index  ·  0–100 Scale")
    ax.set_ylabel("Stress Index")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
    fig.tight_layout(pad=0.6)
    return _fig_to_rl(fig, w_mm, h_mm)


# ── Chart 4: Commodity indexed performance ─────────────────────────────────

_CMD_PERF = ["WTI Crude Oil", "Brent Crude", "Gold",
             "Silver", "Copper", "Natural Gas", "Wheat"]


def _chart_commodity_performance(
    cmd_r: pd.DataFrame,
    lookback: int = 252,
    w_mm: float = 170,
    h_mm: float = 80,
) -> RLImage:
    _mpl_theme()
    cols = [c for c in _CMD_PERF if c in cmd_r.columns]
    fig, ax = plt.subplots(figsize=(w_mm / 25.4, h_mm / 25.4))

    if not cols:
        ax.text(0.5, 0.5, "No commodity data", ha="center", va="center",
                transform=ax.transAxes, color="#555960")
        return _fig_to_rl(fig, w_mm, h_mm)

    data    = cmd_r[cols].iloc[-lookback:].dropna(how="all")
    indexed = (1 + data.fillna(0)).cumprod() * 100

    pal = ["#000000", "#CFB991", "#8E6F3E", "#c0392b",
           "#2980b9", "#e67e22", "#2e7d32"]

    for i, col in enumerate(cols):
        s = indexed[col].dropna()
        if not s.empty:
            final = s.iloc[-1]
            ax.plot(s.index, s.values,
                    color=pal[i % len(pal)], lw=1.2, label=f"{col} ({final:.0f})")

    ax.axhline(100, color="#555960", lw=0.5, ls="--", alpha=0.5)
    ax.set_title(f"Commodity Indexed Performance  ·  Last {lookback} Trading Days  (Base = 100)")
    ax.set_ylabel("Indexed Return")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
    ax.legend(loc="upper left", ncol=2, fontsize=6, handlelength=1.2)
    fig.tight_layout(pad=0.6)
    return _fig_to_rl(fig, w_mm, h_mm)


# ── Chart 5: Rolling correlations for active trade pairs ───────────────────

def _chart_trade_correlations(
    eq_r: pd.DataFrame,
    cmd_r: pd.DataFrame,
    active_trades: list[dict],
    window: int = 60,
    w_mm: float = 170,
    h_mm: float = 75,
    max_pairs: int = 5,
) -> Optional[RLImage]:
    _mpl_theme()
    all_r = pd.concat([eq_r, cmd_r], axis=1)

    pairs: list[tuple[str, pd.Series, str]] = []
    for trade in active_trades:
        if len(trade["assets"]) >= 2:
            a1, a2 = trade["assets"][0], trade["assets"][1]
            if a1 in all_r.columns and a2 in all_r.columns:
                rc = all_r[a1].rolling(window).corr(all_r[a2]).dropna()
                if not rc.empty:
                    col = _CAT_HEX.get(trade.get("category", "Macro"), "#CFB991")
                    pairs.append((f"{a1} / {a2}", rc, col))
                    if len(pairs) >= max_pairs:
                        break

    if not pairs:
        return None

    fig, ax = plt.subplots(figsize=(w_mm / 25.4, h_mm / 25.4))
    ax.axhline(0, color="#555960", lw=0.6, ls="--", alpha=0.5)

    for label, rc, col in pairs:
        ax.plot(rc.index, rc.values, color=col, lw=1.1, label=label, alpha=0.88)

    ax.set_ylim(-1, 1)
    ax.set_title(f"Rolling {window}-Day Correlation  ·  Active Trade Pairs")
    ax.set_ylabel("Pearson r")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.legend(loc="lower left", ncol=2, fontsize=6, handlelength=1.2)
    fig.tight_layout(pad=0.6)
    return _fig_to_rl(fig, w_mm, h_mm)


# ── Paragraph styles ────────────────────────────────────────────────────────

def _S() -> dict:
    return {
        "body": ParagraphStyle(
            "body", fontName="Helvetica", fontSize=9,
            leading=14, textColor=DARK, spaceAfter=6,
        ),
        "body_sm": ParagraphStyle(
            "body_sm", fontName="Helvetica", fontSize=8,
            leading=12, textColor=GRAY,
        ),
        "h2": ParagraphStyle(
            "h2", fontName="Helvetica-Bold", fontSize=12,
            textColor=BLACK, spaceBefore=14, spaceAfter=4, leading=16,
        ),
        "h3": ParagraphStyle(
            "h3", fontName="Helvetica-Bold", fontSize=10,
            textColor=BLACK, spaceBefore=8, spaceAfter=3, leading=13,
        ),
        "caption": ParagraphStyle(
            "caption", fontName="Helvetica", fontSize=7,
            textColor=GRAY, leading=10, spaceAfter=8, alignment=TA_CENTER,
        ),
        "disclaimer": ParagraphStyle(
            "disclaimer", fontName="Helvetica", fontSize=7.5,
            textColor=GRAY, leading=11, spaceAfter=4,
        ),
        "copy": ParagraphStyle(
            "copy", fontName="Helvetica", fontSize=7.5,
            textColor=GRAY, alignment=TA_CENTER, leading=10,
        ),
    }


def _ps(name: str, **kw) -> ParagraphStyle:
    return ParagraphStyle(name, **kw)


# ── Page callbacks ──────────────────────────────────────────────────────────

def _cover_page(canvas, doc):
    c = canvas
    c.saveState()

    c.setFillColor(BLACK)
    c.rect(0, 0, W, H, fill=1, stroke=0)

    c.setFillColor(GOLD)
    c.rect(0, H - 9*mm, W, 9*mm, fill=1, stroke=0)
    c.rect(0, 0, W, 9*mm, fill=1, stroke=0)

    c.setFillColor(AGED)
    c.rect(0, 9*mm, 3.5*mm, H - 18*mm, fill=1, stroke=0)

    c.setFillColor(BLACK)
    c.setFont("Helvetica-Bold", 7)
    c.drawCentredString(W / 2, H - 6*mm,
                        "PURDUE UNIVERSITY · DANIELS SCHOOL OF BUSINESS")

    c.setFillColor(WHITE)
    c.setFont("Helvetica-Bold", 28)
    for txt, y in [("EQUITY &", H - 58*mm), ("COMMODITIES", H - 72*mm),
                   ("SPILLOVER", H - 86*mm), ("MONITOR", H - 100*mm)]:
        c.drawString(18*mm, y, txt)

    c.setFillColor(GOLD)
    c.setFont("Helvetica", 13)
    c.drawString(18*mm, H - 112*mm, "Cross-Asset Research Report")

    c.setStrokeColor(AGED)
    c.setLineWidth(1)
    c.line(18*mm, H - 118*mm, W - 18*mm, H - 118*mm)

    # Details panel
    panel_y = H - 165*mm
    c.setFillColor(colors.HexColor("#0d0d0d"))
    c.setStrokeColor(AGED)
    c.setLineWidth(0.5)
    c.rect(18*mm, panel_y, W - 36*mm, 44*mm, fill=1, stroke=1)

    for lbl, val, x in zip(
        ["REPORT TYPE", "GENERATED", "CLASSIFICATION"],
        ["Institutional Research",
         datetime.now().strftime("%d %b %Y, %H:%M"),
         "Educational Use Only"],
        [24*mm, 90*mm, 157*mm],
    ):
        c.setFillColor(GRAY); c.setFont("Helvetica-Bold", 6.5)
        c.drawString(x, panel_y + 33*mm, lbl)
        c.setFillColor(WHITE); c.setFont("Helvetica", 9)
        c.drawString(x, panel_y + 26*mm, val)

    for lbl, val, x in zip(
        ["ANALYSIS PERIOD", "COURSE"],
        [getattr(doc, "_date_range", "-"), "MGMT 69000-120 · AI for Finance"],
        [24*mm, 90*mm],
    ):
        c.setFillColor(GRAY); c.setFont("Helvetica-Bold", 6.5)
        c.drawString(x, panel_y + 17*mm, lbl)
        c.setFillColor(WHITE); c.setFont("Helvetica", 9)
        c.drawString(x, panel_y + 10*mm, val)

    # Blurb
    c.setFillColor(colors.HexColor("#9D9795"))
    c.setFont("Helvetica", 8.5)
    for i, line in enumerate([
        "Quantitative cross-asset analysis across 15 global equity indices and 17",
        "commodity futures. Covers correlation regime detection, geopolitical risk",
        "transmission, spillover analytics, and regime-triggered trade ideas.",
    ]):
        c.drawString(18*mm, H - 175*mm - i * 5.5*mm, line)

    # Contributors panel
    contrib_y = H - 222*mm
    c.setFillColor(colors.HexColor("#0d0d0d"))
    c.setStrokeColor(AGED)
    c.setLineWidth(0.5)
    c.rect(18*mm, contrib_y, W - 36*mm, 22*mm, fill=1, stroke=1)

    c.setFillColor(GOLD)
    c.setFont("Helvetica-Bold", 6.5)
    c.drawString(24*mm, contrib_y + 15.5*mm, "AUTHORS")

    for (name, link), x in zip(
        [("Heramb S. Patkar",  "hpatkar.github.io"),
         ("Jiahe Miao",        "linkedin.com/in/jiahe-miao071"),
         ("Ilian Zalomai",     "linkedin.com/in/ilian-zalomai-55iz")],
        [24*mm, 90*mm, 156*mm],
    ):
        c.setFillColor(WHITE);  c.setFont("Helvetica-Bold", 8.5)
        c.drawString(x, contrib_y + 9*mm, name)
        c.setFillColor(GRAY);   c.setFont("Helvetica", 7)
        c.drawString(x, contrib_y + 3.5*mm, link)

    c.setFillColor(BLACK)
    c.setFont("Helvetica-Bold", 6.5)
    c.drawCentredString(W / 2, 3.5*mm,
                        "FOR EDUCATIONAL PURPOSES ONLY · NOT INVESTMENT ADVICE · PURDUE UNIVERSITY")
    c.restoreState()


def _interior_page(canvas, doc):
    c = canvas
    c.saveState()

    c.setStrokeColor(LGRAY); c.setLineWidth(0.4)
    c.line(15*mm, H - 14*mm, W - 15*mm, H - 14*mm)
    c.setFillColor(GRAY);  c.setFont("Helvetica", 7)
    c.drawString(15*mm, H - 11.5*mm, "EQUITY & COMMODITIES SPILLOVER MONITOR")
    c.setFillColor(GOLD);  c.setFont("Helvetica-Bold", 7)
    c.drawRightString(W - 15*mm, H - 11.5*mm, "Purdue · Daniels School of Business")

    c.setStrokeColor(LGRAY); c.setLineWidth(0.4)
    c.line(15*mm, 12*mm, W - 15*mm, 12*mm)
    c.setFillColor(GRAY);  c.setFont("Helvetica", 7)
    c.drawString(15*mm, 7*mm, "For educational purposes only · Not investment advice")
    c.drawRightString(W - 15*mm, 7*mm, f"Page {doc.page}")

    c.setFillColor(GOLD)
    c.rect(0, 12*mm, 2.5*mm, H - 24*mm, fill=1, stroke=0)
    c.restoreState()


# ── Flowable helpers ────────────────────────────────────────────────────────

def _rule(col=LGRAY, wt=0.4, before=4, after=6):
    return HRFlowable(width="100%", thickness=wt, color=col,
                      spaceBefore=before, spaceAfter=after)


def _section_header(title: str) -> list:
    return [
        Spacer(1, 4),
        Table(
            [[Paragraph(title.upper(), _ps(
                "sh", fontName="Helvetica-Bold", fontSize=9,
                textColor=GOLD, leading=12,
            ))]],
            colWidths=[W - 30*mm],
            style=TableStyle([
                ("BACKGROUND",    (0,0), (-1,-1), BLACK),
                ("TOPPADDING",    (0,0), (-1,-1), 7),
                ("BOTTOMPADDING", (0,0), (-1,-1), 7),
                ("LEFTPADDING",   (0,0), (-1,-1), 9),
            ]),
        ),
        Spacer(1, 10),
    ]


def _chart_caption(text: str, S: dict) -> Paragraph:
    return Paragraph(text, S["caption"])


def _trade_card(trade: dict) -> list:
    cat_col   = CAT_COLORS.get(trade["category"], GOLD)
    reg_names = " · ".join(REGIME_NAMES[r] for r in trade["regime"])
    dir_text  = "   |   ".join(
        f"{'▲' if d == 'Long' else '▼'} {d} {a}"
        for a, d in zip(trade["assets"], trade["direction"])
    )
    col_w = W - 30*mm

    header_row = Table(
        [[
            Paragraph(f"{trade['category'].upper()}  ·  {trade['trigger']}",
                      _ps("ch", fontName="Helvetica-Bold", fontSize=7.5,
                          textColor=WHITE, leading=10)),
            Paragraph(f"Regimes: {reg_names}",
                      _ps("cr", fontName="Helvetica", fontSize=7,
                          textColor=WHITE, alignment=TA_RIGHT, leading=10)),
        ]],
        colWidths=[col_w * 0.65, col_w * 0.35],
        style=TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), cat_col),
            ("TOPPADDING",    (0,0), (-1,-1), 6),
            ("BOTTOMPADDING", (0,0), (-1,-1), 6),
            ("LEFTPADDING",   (0,0), (-1,-1), 9),
            ("RIGHTPADDING",  (0,0), (-1,-1), 9),
            ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ]),
    )
    name_row = Table(
        [[
            Paragraph(trade["name"], _ps(
                "tn", fontName="Helvetica-Bold", fontSize=11,
                textColor=BLACK, leading=14)),
            Paragraph(dir_text, _ps(
                "dr", fontName="Helvetica", fontSize=7.5,
                textColor=GRAY, alignment=TA_RIGHT, leading=11)),
        ]],
        colWidths=[col_w * 0.60, col_w * 0.40],
        style=TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), BGWARM),
            ("TOPPADDING",    (0,0), (-1,-1), 9),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),
            ("LEFTPADDING",   (0,0), (-1,-1), 9),
            ("RIGHTPADDING",  (0,0), (-1,-1), 9),
            ("VALIGN",        (0,0), (-1,-1), "TOP"),
        ]),
    )
    rationale_row = Table(
        [[Paragraph(trade["rationale"], _ps(
            "rat", fontName="Helvetica", fontSize=8.5,
            textColor=DARK, leading=13))]],
        colWidths=[col_w],
        style=TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), WHITE),
            ("TOPPADDING",    (0,0), (-1,-1), 7),
            ("BOTTOMPADDING", (0,0), (-1,-1), 7),
            ("LEFTPADDING",   (0,0), (-1,-1), 9),
            ("RIGHTPADDING",  (0,0), (-1,-1), 9),
        ]),
    )

    def _sub(lbl, body, bg, lc):
        cw = col_w / 3 - 4
        return Table(
            [[Paragraph(lbl,  _ps(f"l{lbl}", fontName="Helvetica-Bold",
                                  fontSize=6.5, textColor=lc, leading=8))],
             [Paragraph(body, _ps(f"v{lbl}", fontName="Helvetica",
                                  fontSize=8, textColor=DARK, leading=11))]],
            colWidths=[cw],
            style=TableStyle([
                ("BACKGROUND",    (0,0), (-1,-1), bg),
                ("TOPPADDING",    (0,0), (-1,-1), 5),
                ("BOTTOMPADDING", (0,0), (-1,-1), 6),
                ("LEFTPADDING",   (0,0), (-1,-1), 7),
                ("RIGHTPADDING",  (0,0), (-1,-1), 7),
            ]),
        )

    eer_row = Table(
        [[_sub("ENTRY TRIGGER", trade["entry"],
               colors.HexColor("#f9f8f6"), AGED),
          _sub("EXIT SIGNAL",   trade["exit"],
               colors.HexColor("#f9f8f6"), GRAY),
          _sub("KEY RISKS",     trade["risk"],
               colors.HexColor("#fff8f8"), RED)]],
        colWidths=[col_w/3, col_w/3, col_w/3],
        style=TableStyle([
            ("TOPPADDING",    (0,0), (-1,-1), 0),
            ("BOTTOMPADDING", (0,0), (-1,-1), 0),
            ("LEFTPADDING",   (0,0), (-1,-1), 0),
            ("RIGHTPADDING",  (0,0), (-1,-1), 0),
            ("BOX",           (0,0), (-1,-1), 0.5, LGRAY),
            ("LINEBEFORE",    (1,0), (2,-1),  0.5, LGRAY),
        ]),
    )

    return [KeepTogether([header_row, name_row, rationale_row,
                          eer_row, Spacer(1, 14)])]


# ── Main generator ──────────────────────────────────────────────────────────

def generate_report(
    start: str,
    end: str,
    avg_corr_series: pd.Series,
    current_regime: int,
    regimes: pd.Series,
    active_trades: list[dict],
    all_trades: list[dict],
    eq_r: pd.DataFrame,
    cmd_r: pd.DataFrame,
    stress_series: Optional[pd.Series] = None,
    geopolitical_events: Optional[list[dict]] = None,
) -> bytes:
    """Build and return the full PDF as bytes."""
    buf = io.BytesIO()
    S   = _S()
    cw  = W - 30*mm   # content width

    doc = BaseDocTemplate(
        buf, pagesize=A4,
        leftMargin=15*mm, rightMargin=15*mm,
        topMargin=20*mm, bottomMargin=18*mm,
        title="Equity & Commodities Spillover Monitor",
        author="Purdue University · Daniels School of Business",
        subject="Cross-Asset Research Report",
    )
    doc._date_range = f"{start}  →  {end}"

    doc.addPageTemplates([
        PageTemplate(id="Cover",
                     frames=[Frame(0, 0, W, H,
                                   leftPadding=0, rightPadding=0,
                                   topPadding=0, bottomPadding=0)],
                     onPage=_cover_page),
        PageTemplate(id="Normal",
                     frames=[Frame(15*mm, 18*mm, cw, H - 40*mm,
                                   leftPadding=0, rightPadding=0,
                                   topPadding=0, bottomPadding=0)],
                     onPage=_interior_page),
    ])

    story = [NextPageTemplate("Normal"), PageBreak()]

    # ── 1. EXECUTIVE SUMMARY ────────────────────────────────────────────────
    r_name  = REGIME_NAMES[current_regime]
    r_color = REGIME_COLORS[current_regime]
    avg_val = float(avg_corr_series.iloc[-1]) if not avg_corr_series.empty else 0.0

    story += _section_header("Executive Summary")

    regime_panel = Table(
        [[
            Table(
                [[Paragraph("CURRENT REGIME",
                            _ps("rl", fontName="Helvetica-Bold", fontSize=7,
                                textColor=WHITE, leading=9))],
                 [Paragraph(r_name,
                            _ps("rv", fontName="Helvetica-Bold", fontSize=20,
                                textColor=WHITE, leading=24))]],
                colWidths=[65*mm],
                style=TableStyle([
                    ("BACKGROUND",    (0,0), (-1,-1), r_color),
                    ("TOPPADDING",    (0,0), (-1,-1), 9),
                    ("BOTTOMPADDING", (0,0), (-1,-1), 9),
                    ("LEFTPADDING",   (0,0), (-1,-1), 11),
                ]),
            ),
            Table(
                [
                    [Paragraph("Avg |Corr| (60d)",
                               _ps("m1l", fontName="Helvetica-Bold", fontSize=6.5, textColor=GRAY, leading=9)),
                     Paragraph("Active Ideas",
                               _ps("m2l", fontName="Helvetica-Bold", fontSize=6.5, textColor=GRAY, leading=9)),
                     Paragraph("Analysis Period",
                               _ps("m3l", fontName="Helvetica-Bold", fontSize=6.5, textColor=GRAY, leading=9))],
                    [Paragraph(f"{avg_val:.3f}",
                               _ps("m1v", fontName="Helvetica-Bold", fontSize=17, textColor=BLACK, leading=21)),
                     Paragraph(str(len(active_trades)),
                               _ps("m2v", fontName="Helvetica-Bold", fontSize=17, textColor=BLACK, leading=21)),
                     Paragraph(f"{start[:4]} – {end[:4]}",
                               _ps("m3v", fontName="Helvetica-Bold", fontSize=17, textColor=BLACK, leading=21))],
                ],
                colWidths=[48*mm, 40*mm, cw - 65*mm - 88*mm],
                style=TableStyle([
                    ("BACKGROUND",    (0,0), (-1,-1), BGWARM),
                    ("TOPPADDING",    (0,0), (-1,-1), 9),
                    ("BOTTOMPADDING", (0,0), (-1,-1), 9),
                    ("LEFTPADDING",   (0,0), (-1,-1), 11),
                    ("LINEAFTER",     (0,0), (1,-1),  0.4, LGRAY),
                    ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
                ]),
            ),
        ]],
        colWidths=[65*mm, cw - 65*mm],
        style=TableStyle([
            ("TOPPADDING",    (0,0), (-1,-1), 0),
            ("BOTTOMPADDING", (0,0), (-1,-1), 0),
            ("LEFTPADDING",   (0,0), (-1,-1), 0),
            ("RIGHTPADDING",  (0,0), (-1,-1), 0),
            ("BOX",           (0,0), (-1,-1), 0.5, LGRAY),
        ]),
    )
    story += [regime_panel, Spacer(1, 12)]

    narratives = {
        0: ("Decorrelated Regime",
            "Cross-asset correlations are below historical norms. Equities and commodities are pricing "
            "independent fundamentals, typically benign macro conditions. Growth-correlated long "
            "positioning is favoured. Diversification benefits are maximised."),
        1: ("Normal Correlation Regime",
            "Cross-asset correlations are within their historical distribution. No systemic stress signal "
            "is present. Trade ideas span both growth and macro themes. Monitor for regime transition "
            "signals, particularly acceleration in the fast correlation index."),
        2: ("Elevated Correlation Regime",
            "Cross-asset correlations are in the upper quartile. A macro or geopolitical stress driver "
            "is likely active. Risk-off positioning and crisis hedges become increasingly relevant. "
            "Correlation convergence trades activate."),
        3: ("Crisis Correlation Regime",
            "Cross-asset correlations are at extreme levels, consistent with systemic crisis. Historical "
            "analogs: GFC (2008–09), COVID crash (2020), Ukraine War onset (2022). Flight-to-quality flows "
            "dominate. Precious metals decouple positively; industrial metals and EM equities face maximum "
            "selling pressure."),
    }
    r_title, r_text = narratives[current_regime]
    story += [
        Paragraph("Regime Interpretation", S["h3"]),
        Table(
            [[Paragraph(r_title, _ps("rtn", fontName="Helvetica-Bold", fontSize=9,
                                     textColor=r_color, leading=12))],
             [Paragraph(r_text, S["body"])]],
            colWidths=[cw],
            style=TableStyle([
                ("LEFTPADDING",   (0,0), (-1,-1), 11),
                ("RIGHTPADDING",  (0,0), (-1,-1), 11),
                ("TOPPADDING",    (0,0), (-1,-1), 5),
                ("BOTTOMPADDING", (0,0), (-1,-1), 6),
                ("LINEBEFORE",    (0,0), (0,-1), 3.5, r_color),
                ("BACKGROUND",    (0,0), (-1,-1), BGWARM),
            ]),
        ),
        Spacer(1, 6),
    ]

    # ── 2. REGIME TIMELINE (chart) ──────────────────────────────────────────
    story += [PageBreak()]
    story += _section_header("Correlation Regime History")
    story += [
        Paragraph(
            "The chart below shows the 60-day rolling average of absolute cross-asset correlation "
            "across all equity-commodity pairs. Background shading indicates the detected regime "
            "(green = Decorrelated, grey = Normal, orange = Elevated, red = Crisis).",
            S["body"],
        ),
        Spacer(1, 8),
        _chart_regime_timeline(avg_corr_series, regimes, w_mm=cw / mm, h_mm=72),
        _chart_caption(
            "Figure 1: Rolling 60-day avg |cross-asset correlation| with adaptive percentile regime "
            "bands. Regime classification uses median smoothing, hysteresis, and a 10-day persistence gate.",
            S,
        ),
    ]

    # ── 3. CORRELATION MATRIX (chart) ───────────────────────────────────────
    story += [
        _rule(),
        Paragraph("Cross-Asset Correlation Matrix", S["h2"]),
        Paragraph(
            "Full-sample Pearson correlations between global equity indices and key commodity "
            "futures. Red cells indicate positive co-movement; blue indicates negative. "
            "The magnitude of correlation determines the strength of the spillover channel.",
            S["body"],
        ),
        Spacer(1, 6),
        _chart_corr_heatmap(eq_r, cmd_r, w_mm=cw / mm, h_mm=115),
        _chart_caption(
            "Figure 2: Cross-asset Pearson correlation matrix (full sample). "
            "Rows: equity indices. Columns: commodity futures. "
            "Darker shading = stronger co-movement.",
            S,
        ),
    ]

    # ── 4. STRESS INDEX (chart) ─────────────────────────────────────────────
    story += [PageBreak()]
    story += _section_header("Composite Market Stress Index")
    story += [
        Paragraph(
            "The Composite Stress Index (0–100) blends four signals: equity realised volatility "
            "(45%, VIX proxy), slow cross-asset correlation (35%), commodity energy+metals volatility "
            "(15%), and fast correlation acceleration (5%). Z-score mapping preserves absolute level "
            "information, critical for detecting VIX threshold breaches.",
            S["body"],
        ),
        Spacer(1, 8),
    ]

    if stress_series is not None and not stress_series.empty:
        story += [
            _chart_stress_index(stress_series, w_mm=cw / mm, h_mm=72),
            _chart_caption(
                "Figure 3: Composite stress index (0-100). Bands: green < 40 (normal), "
                "yellow 40–60 (elevated), orange 60–80 (high), red > 80 (crisis).",
                S,
            ),
            Spacer(1, 6),
        ]
    else:
        story += [
            Paragraph("Stress index data not available for this report.", S["body_sm"]),
            Spacer(1, 6),
        ]

    # ── 5. COMMODITY PERFORMANCE (chart) ───────────────────────────────────
    story += [
        _rule(),
        Paragraph("Commodity Price Performance", S["h2"]),
        Paragraph(
            "Indexed cumulative returns for key commodity futures over the most recent "
            "252 trading days (approximately one year). Base = 100 at start of window. "
            "Outperformers and underperformers relative to the base are immediately visible.",
            S["body"],
        ),
        Spacer(1, 8),
        _chart_commodity_performance(cmd_r, w_mm=cw / mm, h_mm=80),
        _chart_caption(
            "Figure 4: Indexed commodity performance (last 252 trading days). "
            "Values in parentheses show end-of-period index level.",
            S,
        ),
    ]

    # ── 6. ACTIVE TRADE IDEAS ───────────────────────────────────────────────
    story += [PageBreak()]
    story += _section_header(f"Active Trade Ideas: {r_name} Regime")
    story += [
        Paragraph(
            f"<b>{len(active_trades)}</b> trade idea{'s' if len(active_trades) != 1 else ''} "
            f"triggered for the current <b>{r_name}</b> regime. Each idea is grounded in "
            "historical spillover patterns and cross-asset regime analysis.",
            S["body"],
        ),
        Spacer(1, 8),
    ]

    if active_trades:
        for trade in active_trades:
            story += _trade_card(trade)

        # Rolling correlation supporting chart
        corr_chart = _chart_trade_correlations(
            eq_r, cmd_r, active_trades, w_mm=cw / mm, h_mm=75)
        if corr_chart is not None:
            story += [
                _rule(LGRAY, 0.3),
                Paragraph("Supporting Analysis: Pair Correlations", S["h3"]),
                Paragraph(
                    "Rolling 60-day Pearson correlations between the first asset pair "
                    "of each active trade idea. These time series underpin the entry "
                    "and exit signals described in the trade cards above.",
                    S["body"],
                ),
                Spacer(1, 6),
                corr_chart,
                _chart_caption(
                    "Figure 5: Rolling 60-day correlation for active trade pairs. "
                    "Colour matches trade category: red = Crisis Hedge, orange = Geopolitical, "
                    "blue = Macro, green = Growth.",
                    S,
                ),
            ]
    else:
        story += [Paragraph(
            "No trade ideas active for the current regime. "
            "See the full reference library below.", S["body"])]

    # Other-regime reference
    other = [t for t in all_trades if t not in active_trades]
    if other:
        story += [
            _rule(),
            Paragraph("Reference: Trade Ideas for Other Regimes", S["h3"]),
            Paragraph(
                "Inactive in the current regime but included for regime-transition readiness.",
                S["body"],
            ),
            Spacer(1, 6),
        ]
        for trade in other:
            story += _trade_card(trade)

    # ── 7. GEOPOLITICAL CONTEXT ─────────────────────────────────────────────
    if geopolitical_events:
        story += [PageBreak()]
        story += _section_header("Geopolitical Risk Context")
        story += [
            Paragraph(
                "Geopolitical and macroeconomic events identified as structurally significant "
                "for cross-asset correlation and commodity pricing. Active events continue to "
                "embed a risk premium into current market pricing.",
                S["body"],
            ),
            Spacer(1, 8),
        ]
        today = date.today()
        for ev in geopolitical_events:
            is_active  = (ev["end"] >= today) if isinstance(ev["end"], date) else False
            ev_color   = colors.HexColor(ev.get("color", "#CFB991"))
            status_col = RED if is_active else GREEN
            status_txt = "ACTIVE" if is_active else "RESOLVED"
            period     = (f"{ev['start'].strftime('%b %Y')} – "
                          f"{'Present' if is_active else ev['end'].strftime('%b %Y')}")
            story += [KeepTogether([
                Table([[
                    Paragraph(ev.get("name", ev["label"]),
                              _ps("en", fontName="Helvetica-Bold", fontSize=9,
                                  textColor=ev_color, leading=12)),
                    Paragraph(status_txt,
                              _ps("es", fontName="Helvetica-Bold", fontSize=7,
                                  textColor=status_col, alignment=TA_RIGHT, leading=9)),
                ]], colWidths=[cw * 0.75, cw * 0.25],
                style=TableStyle([
                    ("TOPPADDING",  (0,0),(-1,-1), 7), ("BOTTOMPADDING",(0,0),(-1,-1), 2),
                    ("LEFTPADDING", (0,0),(-1,-1), 9), ("RIGHTPADDING",  (0,0),(-1,-1), 9),
                    ("LINEBEFORE",  (0,0),(0,-1),  3.5, ev_color),
                    ("BACKGROUND",  (0,0),(-1,-1), BGWARM),
                ])),
                Table([[
                    Paragraph(period,
                              _ps("ep", fontName="Helvetica", fontSize=7.5,
                                  textColor=GRAY, leading=10)),
                    Paragraph(ev.get("category", ""),
                              _ps("ec", fontName="Helvetica-Bold", fontSize=7,
                                  textColor=GRAY, alignment=TA_RIGHT, leading=9)),
                ]], colWidths=[cw * 0.75, cw * 0.25],
                style=TableStyle([
                    ("TOPPADDING",  (0,0),(-1,-1), 2), ("BOTTOMPADDING",(0,0),(-1,-1), 5),
                    ("LEFTPADDING", (0,0),(-1,-1), 9), ("RIGHTPADDING",  (0,0),(-1,-1), 9),
                    ("BACKGROUND",  (0,0),(-1,-1), BGWARM),
                ])),
                Table([[Paragraph(ev.get("description", ""),
                                  _ps("ed", fontName="Helvetica", fontSize=8.5,
                                      textColor=DARK, leading=13))]],
                      colWidths=[cw],
                      style=TableStyle([
                          ("TOPPADDING",  (0,0),(-1,-1), 5), ("BOTTOMPADDING",(0,0),(-1,-1), 9),
                          ("LEFTPADDING", (0,0),(-1,-1), 9), ("RIGHTPADDING",  (0,0),(-1,-1), 9),
                      ])),
                _rule(LGRAY, 0.3),
            ])]

    # ── 8. METHODOLOGY ──────────────────────────────────────────────────────
    story += [PageBreak()]
    story += _section_header("Methodology & Framework")

    for title, body in [
        ("Data Universe",
         "15 global equity indices (US, Europe, Asia-Pacific, India) and 17 commodity futures "
         "(Energy, Precious Metals, Industrial Metals, Agriculture) from Yahoo Finance. "
         "Macro series (VIX, 10Y/2Y yields, CPI, DXY) from FRED. Daily log returns computed."),
        ("Correlation Regime Detection",
         "Rolling 60-day Pearson correlations computed pairwise. Mean |cross-asset corr| "
         "is the primary stress signal. Regimes use percentile thresholds (20th/55th/80th), "
         "5-day median smoothing, ±5pp hysteresis bands, and a 10-day persistence gate for Crisis."),
        ("Composite Stress Index",
         "0–100 blend: equity realised vol 45% (VIX proxy, z-score), slow correlation 35% "
         "(rolling percentile), commodity vol 15% (z-score), fast correlation 5%. "
         "Z-score mapping preserves absolute level information."),
        ("Trade Idea Framework",
         "Regime-triggered library covering Crisis Hedge, Geopolitical, Macro, and Growth "
         "categories. Each idea has quantitative entry/exit conditions and key risks."),
        ("DCC-GARCH",
         "Dynamic Conditional Correlation (Engle 2002), DCC(1,1) with a=0.05, b=0.90 "
         "(stationarity: a+b<1). Captures time-varying correlation structure."),
        ("Spillover Analytics",
         "Granger causality and Diebold-Yilmaz spillover index decompose directional "
         "transmission. Transfer entropy captures non-linear dependencies. "
         "Network centrality identifies systemically important nodes."),
    ]:
        story += [KeepTogether([
            Paragraph(title, S["h3"]),
            Paragraph(body, S["body"]),
            Spacer(1, 3),
        ])]

    story += [
        _rule(),
        Paragraph("Data Sources", S["h3"]),
        Table(
            [["Source", "Coverage", "Access"],
             ["Yahoo Finance", "Equity indices, commodity futures, FX (daily close)", "yfinance API"],
             ["FRED (Federal Reserve)", "VIX, 10Y/2Y yields, CPI, DXY, WTI, Gold", "fredapi"],
             ["FinancialDatasets.ai", "Supplementary financial data", "API key required"]],
            colWidths=[46*mm, 100*mm, 34*mm],
            style=TableStyle([
                ("BACKGROUND",    (0,0), (-1,0), BLACK),
                ("TEXTCOLOR",     (0,0), (-1,0), GOLD),
                ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
                ("FONTSIZE",      (0,0), (-1,-1), 8),
                ("FONTNAME",      (0,1), (-1,-1), "Helvetica"),
                ("GRID",          (0,0), (-1,-1), 0.3, LGRAY),
                ("TOPPADDING",    (0,0), (-1,-1), 5),
                ("BOTTOMPADDING", (0,0), (-1,-1), 5),
                ("LEFTPADDING",   (0,0), (-1,-1), 7),
                ("ROWBACKGROUNDS",(0,1), (-1,-1), [WHITE, BGWARM]),
            ]),
        ),
        Spacer(1, 16),
    ]

    # ── 9. DISCLAIMER ────────────────────────────────────────────────────────
    story += _section_header("Disclaimer")
    story += [
        Table(
            [[Paragraph(
                "This report is produced for educational purposes only as part of MGMT 69000-120 "
                "(AI for Finance) at Purdue University's Daniels School of Business. "
                "It does not constitute investment advice, a solicitation, or a recommendation "
                "to buy or sell any security or financial instrument. Past performance is not "
                "indicative of future results. All trade ideas are illustrative examples of "
                "cross-asset analytical frameworks and must not be implemented without independent "
                "professional due diligence, risk assessment, and regulatory review. "
                "The authors accept no liability for any financial loss arising from reliance on "
                "this material. Market data is sourced from public providers and may contain "
                "errors or omissions.",
                S["disclaimer"],
            )]],
            colWidths=[cw],
            style=TableStyle([
                ("BACKGROUND",    (0,0), (-1,-1), colors.HexColor("#fffdf5")),
                ("BOX",           (0,0), (-1,-1), 0.5, GOLD),
                ("TOPPADDING",    (0,0), (-1,-1), 11),
                ("BOTTOMPADDING", (0,0), (-1,-1), 11),
                ("LEFTPADDING",   (0,0), (-1,-1), 11),
                ("RIGHTPADDING",  (0,0), (-1,-1), 11),
            ]),
        ),
        Spacer(1, 10),
        Paragraph(
            f"© {datetime.now().year} Purdue University · Daniels School of Business · "
            f"MGMT 69000-120 AI for Finance",
            S["copy"],
        ),
        Paragraph(
            "Heramb S. Patkar · Jiahe Miao · Ilian Zalomai  "
            f"· Generated {datetime.now().strftime('%d %B %Y')}",
            S["copy"],
        ),
    ]

    doc.build(story)
    return buf.getvalue()
