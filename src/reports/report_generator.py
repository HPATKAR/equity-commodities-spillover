"""
Institutional-grade PDF report generator.
Purdue University · Daniels School of Business
Purdue Daniels School of Business · MSF Research Terminal

Narrative structure:
  1. Cover page          - branding, contributors, metadata
  2. Executive Summary   - regime panel + interpretation
  3. Regime Timeline     - avg |corr| history with regime shading   [CHART]
  4. Correlation Matrix  - equity-commodities heatmap                       [CHART]
  5. Market Stress       - composite stress index 0-100              [CHART]
  6. Commodity Performance - indexed price returns                   [CHART]
  7. Trade Ideas         - active cards + pair correlation chart     [CHART]
  8. Geopolitical Context - event cards
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
    Paragraph as _RLParagraph, Spacer, Table, TableStyle,
    HRFlowable, NextPageTemplate, PageBreak, KeepTogether,
    Image as RLImage,
)


def Paragraph(text=None, *args, **kwargs):
    """reportlab Paragraph with em-dashes stripped from ALL rendered text - " - "
    becomes a spaced hyphen so the report reads cleanly (en-dashes in ranges like
    3–8 weeks are preserved). Shadows the import so every call is sanitised."""
    if isinstance(text, str):
        text = text.replace(" - ", " - ").replace("&mdash;", " - ")
    return _RLParagraph(text, *args, **kwargs)

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
    "Crisis Hedge":    RED,
    "Geopolitical":    ORANGE,
    "Macro":           BLUE,
    "Growth":          GREEN,
    "Dollar Cycle":    colors.HexColor("#1abc9c"),
    "Asia Divergence": colors.HexColor("#9b59b6"),
    "Fixed Income":    colors.HexColor("#2471a3"),
    "India/EM":        colors.HexColor("#d35400"),
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
        ["ANALYSIS PERIOD", "PROGRAM"],
        [getattr(doc, "_date_range", "-"), "Purdue Daniels School of Business · MSF Research Terminal"],
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
        "Quantitative equity-commodities analysis across 15 global equity indices and 17",
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


def _prep_logo_png(logo_png):
    """The card header is a light band, so a predominantly-white logo (a light
    mark on transparency, e.g. BlackRock / Nike) would be invisible. Detect that
    case and composite the mark onto a dark rounded tile so it stays legible;
    leave dark or self-contained (opaque-background) logos untouched. Best-effort
 - returns the original bytes on any failure."""
    try:
        from PIL import Image, ImageDraw
        im = Image.open(io.BytesIO(logo_png)).convert("RGBA")
        # Luminance + transparency on a small downsample (fast, good enough).
        small = im.resize((40, 40))
        lum_sum = opaque = transparent = 0
        for (r, g, b, a) in small.getdata():
            if a < 24:
                transparent += 1
                continue
            opaque += 1
            lum_sum += 0.299 * r + 0.587 * g + 0.114 * b
        if not opaque:
            return logo_png
        mean_lum = lum_sum / opaque
        has_transparency = transparent > 0.06 * 1600
        if not (has_transparency and mean_lum > 182):
            return logo_png                       # visible on a light header → keep
        # Light mark → dark rounded tile with padding so it reads cleanly.
        w, h = im.size
        pad = int(round(0.20 * max(w, h)))
        side = max(w, h) + 2 * pad                # square tile
        tile = Image.new("RGBA", (side, side), (24, 27, 33, 255))   # slate #181b21
        tile.paste(im, ((side - w) // 2, (side - h) // 2), im)      # alpha as mask
        mask = Image.new("L", (side, side), 0)
        ImageDraw.Draw(mask).rounded_rectangle(
            [0, 0, side - 1, side - 1], radius=int(0.16 * side), fill=255)
        tile.putalpha(mask)                       # rounded transparent corners
        out = io.BytesIO()
        tile.save(out, "PNG")
        return out.getvalue()
    except Exception:
        return logo_png


def _logo_flowable(logo_png, height_mm: float = 6.5, max_w_mm: float = 11.0):
    """Small, aspect-correct company logo for a trade-card header, or None.

    Height is fixed; width derives from the image's aspect ratio and is capped so
    a wide wordmark can't blow out the column. White logos are put on a dark tile
    (_prep_logo_png) so they stay visible on the light header. Any missing/corrupt/
    undecodable image returns None so the card renders logo-less - never raises."""
    if not logo_png:
        return None
    try:
        from reportlab.lib.utils import ImageReader
        logo_png = _prep_logo_png(logo_png)
        iw, ih = ImageReader(io.BytesIO(logo_png)).getSize()
        if not iw or not ih:
            return None
        w_mm = min(height_mm * (iw / ih), max_w_mm)
        # Fresh BytesIO for the flowable (ImageReader above consumed its own).
        return RLImage(io.BytesIO(logo_png), width=w_mm * mm, height=height_mm * mm)
    except Exception:
        return None


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
    _name_para = Paragraph(trade["name"], _ps(
        "tn", fontName="Helvetica-Bold", fontSize=11,
        textColor=BLACK, leading=14))
    _dir_para = Paragraph(dir_text, _ps(
        "dr", fontName="Helvetica", fontSize=7.5,
        textColor=GRAY, alignment=TA_RIGHT, leading=11))
    _logo = _logo_flowable(trade.get("logo_png"))
    if _logo is not None:
        # [ logo | thesis name | direction ] - logo in a fixed left slot, vertically
        # centred with the name. Falls back to the 2-col layout below if no logo.
        _LOGO_SLOT = 13 * mm
        name_row = Table(
            [[_logo, _name_para, _dir_para]],
            colWidths=[_LOGO_SLOT, col_w * 0.60 - _LOGO_SLOT, col_w * 0.40],
            style=TableStyle([
                ("BACKGROUND",    (0,0), (-1,-1), BGWARM),
                ("TOPPADDING",    (0,0), (-1,-1), 9),
                ("BOTTOMPADDING", (0,0), (-1,-1), 5),
                ("LEFTPADDING",   (0,0), (-1,-1), 9),
                ("RIGHTPADDING",  (0,0), (-1,-1), 9),
                ("VALIGN",        (1,0), (-1,-1), "TOP"),
                ("VALIGN",        (0,0), (0,0),   "MIDDLE"),
                ("RIGHTPADDING",  (0,0), (0,0),   4),
            ]),
        )
    else:
        name_row = Table(
            [[_name_para, _dir_para]],
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

    risk_text = trade.get("risk") or trade.get("stop", " - ")
    eer_row = Table(
        [[_sub("ENTRY TRIGGER", trade.get("entry", " - "),
               colors.HexColor("#f9f8f6"), AGED),
          _sub("EXIT SIGNAL",   trade.get("exit", " - "),
               colors.HexColor("#f9f8f6"), GRAY),
          _sub("KEY RISKS / STOP", risk_text,
               colors.HexColor("#fff8f8"), RED)]],
        colWidths=[col_w/3, col_w/3, col_w/3],
        style=TableStyle([
            # Paint the colour on the OUTER cell so it fills the full cell height
            # and width even when a column's text is shorter than its neighbours
            # (the inner _sub table only paints its own content box).
            ("BACKGROUND",    (0,0), (0,0), colors.HexColor("#f9f8f6")),
            ("BACKGROUND",    (1,0), (1,0), colors.HexColor("#f9f8f6")),
            ("BACKGROUND",    (2,0), (2,0), colors.HexColor("#fff8f8")),
            ("TOPPADDING",    (0,0), (-1,-1), 0),
            ("BOTTOMPADDING", (0,0), (-1,-1), 0),
            ("LEFTPADDING",   (0,0), (-1,-1), 0),
            ("RIGHTPADDING",  (0,0), (-1,-1), 0),
            ("BOX",           (0,0), (-1,-1), 0.5, LGRAY),
            ("LINEBEFORE",    (1,0), (2,-1),  0.5, LGRAY),
        ]),
    )

    # Extra detail row for new-style trades (target / invalidation / hold period)
    extra_rows: list = []
    if trade.get("target") or trade.get("invalidation") or trade.get("holding_period"):
        extra_rows.append(
            Table(
                [[_sub("TARGET",       trade.get("target",         " - "),
                        colors.HexColor("#f3faf3"), GREEN),
                  _sub("INVALIDATION", trade.get("invalidation",   " - "),
                        colors.HexColor("#f5f5ff"), BLUE),
                  _sub("HOLD PERIOD",  trade.get("holding_period", " - "),
                        colors.HexColor("#f9f8f6"), GRAY)]],
                colWidths=[col_w/3, col_w/3, col_w/3],
                style=TableStyle([
                    # Colour on the OUTER cell → fills the whole cell regardless
                    # of which column's text is tallest (see eer_row note above).
                    ("BACKGROUND",    (0,0), (0,0), colors.HexColor("#f3faf3")),
                    ("BACKGROUND",    (1,0), (1,0), colors.HexColor("#f5f5ff")),
                    ("BACKGROUND",    (2,0), (2,0), colors.HexColor("#f9f8f6")),
                    ("TOPPADDING",    (0,0), (-1,-1), 0),
                    ("BOTTOMPADDING", (0,0), (-1,-1), 0),
                    ("LEFTPADDING",   (0,0), (-1,-1), 0),
                    ("RIGHTPADDING",  (0,0), (-1,-1), 0),
                    ("BOX",           (0,0), (-1,-1), 0.5, LGRAY),
                    ("LINEBEFORE",    (1,0), (2,-1),  0.5, LGRAY),
                ]),
            )
        )

    # Recent third-party coverage - real, dated, sourced headlines (last ~30d)
    # attached at report time (yfinance) to anchor the thesis to live market
    # context. Two-row band: label, then one line per headline.
    def _esc(s: str) -> str:
        return (str(s) or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    news = trade.get("recent_news") or []
    news_rows: list = []
    if news:
        _items = "<br/>".join(
            f'<font color="#1a56a0"><b>{_esc(n.get("date", ""))}</b></font>  '
            f'<i>{_esc((n.get("publisher") or "")[:24])}</i> &nbsp;&mdash;&nbsp; '
            f'{_esc((n.get("title") or "")[:118])}'
            for n in news[:3]
        )
        news_rows.append(Table(
            [[Paragraph("RECENT COVERAGE  ·  THIRD-PARTY NEWS &amp; ANALYST COMMENTARY (LAST ~30 DAYS)",
                        _ps("nl", fontName="Helvetica-Bold", fontSize=6.5,
                            textColor=BLUE, leading=9))],
             [Paragraph(_items, _ps("nv", fontName="Helvetica", fontSize=7.5,
                                    textColor=DARK, leading=12))]],
            colWidths=[col_w],
            style=TableStyle([
                ("BACKGROUND",    (0,0), (-1,-1), colors.HexColor("#f3f6fc")),
                ("LINEABOVE",     (0,0), (-1,0),  0.5, LGRAY),
                ("TOPPADDING",    (0,0), (0,0),   6),
                ("BOTTOMPADDING", (0,0), (0,0),   2),
                ("TOPPADDING",    (0,1), (0,1),   1),
                ("BOTTOMPADDING", (0,1), (0,1),   7),
                ("LEFTPADDING",   (0,0), (-1,-1), 9),
                ("RIGHTPADDING",  (0,0), (-1,-1), 9),
            ]),
        ))

    return [KeepTogether([header_row, name_row, rationale_row,
                          eer_row, *extra_rows, *news_rows, Spacer(1, 14)])]


# ── Main generator ──────────────────────────────────────────────────────────

def _fa_bar(frac: float, color, w: float = 104, h: float = 6.5):
    """Horizontal magnitude bar (reportlab Drawing) for a factor loading."""
    from reportlab.graphics.shapes import Drawing, Rect
    frac = max(0.0, min(1.0, float(frac)))
    d = Drawing(w, h)
    d.add(Rect(0, 0, w, h, fillColor=colors.HexColor("#efeee9"), strokeColor=None))
    if frac > 0:
        d.add(Rect(0, 0, max(1.0, w * frac), h, fillColor=color, strokeColor=None))
    return d


def _factor_attribution_section(d: dict, S: dict, cw: float) -> list:
    """One-page 'Book Risk Character' section for the desk report - the buy-side
    alpha-vs-beta verdict, rendered from the decomposition dict computed on the
    trade page (_compute_book_factor_decomp). Returns reportlab flowables."""
    from reportlab.graphics.shapes import Drawing, Rect

    def _esc(s):
        return (str(s) or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    _VMAP = {"ALPHA PRESENT": GREEN, "BETA BOOK": RED,
             "SECTOR-TILT BOOK": ORANGE, "INCONCLUSIVE": GRAY}
    vcol = _VMAP.get(d["verdict"], GRAY)
    n, obs = d["n_positions"], d["obs"]
    a_mf, a_t, sig = d["alpha_mf"], d["alpha_mf_t"], d["sig"]
    acol = GREEN if (a_mf > 0 and sig) else (DARK if a_mf >= 0 else RED)

    story = _section_header("Book Risk Character - Factor & Alpha Attribution")
    story.append(Paragraph(
        "Ex-post attribution of the deployed book over the sample window. The "
        "weight-normalised book return is regressed on the market (S&amp;P 500) plus "
        "market-orthogonalised thematic factors with HAC/Newey-West standard errors, "
        "separating <b>market beta</b>, <b>sector tilts</b> and genuine "
        "<b>idiosyncratic</b> return. This is descriptive of the current book, not a "
        "forward forecast.", S["body"]))
    story.append(Spacer(1, 7))

    # Verdict strip
    story.append(Table(
        [[Paragraph(f'VERDICT&nbsp;&nbsp;<b>{_esc(d["verdict"])}</b>',
                    _ps("fav", fontName="Helvetica", fontSize=9, textColor=colors.white,
                        leading=12)),
          Paragraph(f'HAC/Newey-West · {obs} obs · {d["n_factors"]} factors',
                    _ps("fas", fontName="Helvetica", fontSize=7.5, textColor=colors.white,
                        alignment=TA_RIGHT, leading=12))]],
        colWidths=[cw * 0.5, cw * 0.5],
        style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), vcol),
            ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ])))
    story.append(Spacer(1, 8))

    # KPI row (4 cells)
    def _kpi(lbl, val, sub, vc=DARK):
        _vc = "#" + vc.hexval()[2:]
        return Paragraph(
            f'<font size=6 color="#8a8f98">{_esc(lbl)}</font><br/>'
            f'<font size=15 color="{_vc}"><b>{val}</b></font><br/>'
            f'<font size=6 color="#8a8f98">{sub}</font>',
            _ps(f"k{lbl}", fontName="Helvetica", fontSize=8, leading=12))
    _ir = d["ir"]
    kpis = [
        _kpi("MARKET BETA", f'{d["beta_mkt"]:.2f}', f'vs S&amp;P &nbsp;R² {d["r2_mkt"]*100:.0f}%'),
        _kpi("JENSEN ALPHA", f'{a_mf:+.1f}%/yr', f't {a_t:.1f} · {"significant" if sig else "not sig"}', acol),
        _kpi("INFO RATIO", f'{_ir:.2f}', f'TE {d["te"]:.1f}%/yr vs S&amp;P'),
        _kpi("EFFECTIVE BETS", f'{d["enb"]:.1f}', f'of {n} positions'),
    ]
    story.append(Table(
        [kpis], colWidths=[cw / 4] * 4,
        style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), BGWARM),
            ("BOX", (0, 0), (-1, -1), 0.5, LGRAY),
            ("LINEBEFORE", (1, 0), (-1, -1), 0.5, LGRAY),
            ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING", (0, 0), (-1, -1), 9), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ])))
    story.append(Spacer(1, 12))

    # Factor loadings table
    story.append(Paragraph("Factor Loadings - market-orthogonalised", S["h3"]))
    _bmax = max((abs(b) for _, b, _ in d["loadings"]), default=1.0) or 1.0
    _rows = [[Paragraph("<b>FACTOR</b>", S["body_sm"]),
              Paragraph("<b>BETA</b>", _ps("lh1", fontName="Helvetica-Bold", fontSize=7.5,
                                           alignment=TA_RIGHT)),
              Paragraph("<b>t</b>", _ps("lh2", fontName="Helvetica-Bold", fontSize=7.5,
                                        alignment=TA_RIGHT)),
              Paragraph("", S["body_sm"])]]
    for name, b, tt in d["loadings"]:
        _is_mkt = name == d["mkt_name"]
        bc = BLUE if _is_mkt else (GREEN if b >= 0 else RED)
        _tcol = DARK if abs(tt) >= 2 else colors.HexColor("#9aa0a8")
        _rows.append([
            Paragraph(_esc(name), _ps(f"ln{name}", fontName="Helvetica-Bold", fontSize=8,
                                      textColor=_tcol)),
            Paragraph(f"{b:+.2f}", _ps(f"lb{name}", fontName="Helvetica", fontSize=8,
                                       textColor=_tcol, alignment=TA_RIGHT)),
            Paragraph(f"{tt:+.1f}", _ps(f"lt{name}", fontName="Helvetica", fontSize=8,
                                        textColor=_tcol, alignment=TA_RIGHT)),
            _fa_bar(abs(b) / _bmax, bc, w=min(150, cw * 0.30)),
        ])
    story.append(Table(
        _rows, colWidths=[cw * 0.30, cw * 0.14, cw * 0.14, cw * 0.42],
        style=TableStyle([
            ("LINEBELOW", (0, 0), (-1, 0), 0.5, LGRAY),
            ("LINEBELOW", (0, 1), (-1, -2), 0.25, colors.HexColor("#f0efeb")),
            ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ])))
    story.append(Paragraph("Faded rows are statistically insignificant (|t| &lt; 2). "
                           "Market bar in blue; sector tilts green (long) / red (short).",
                           S["caption"]))
    story.append(Spacer(1, 12))

    # Variance-explained stacked bar
    story.append(Paragraph("Variance Explained", S["h3"]))
    ms, ss, ids = d["mkt_share"], d["sect_share"], d["idio_share"]
    _sw = cw * 0.9
    stack = Drawing(_sw, 12)
    _x = 0.0
    for _sh, _c in [(ms, BLUE), (ss, ORANGE), (ids, colors.HexColor("#c9ccd4"))]:
        _seg = _sw * max(0.0, _sh) / 100.0
        if _seg > 0.5:
            stack.add(Rect(_x, 0, _seg, 12, fillColor=_c, strokeColor=None))
        _x += _seg
    story.append(stack)
    story.append(Spacer(1, 2))
    story.append(Paragraph(
        f'<font color="#2980b9">■</font> market {ms:.0f}% &nbsp;&nbsp;'
        f'<font color="#e67e22">■</font> sector tilts {ss:.0f}% &nbsp;&nbsp;'
        f'<font color="#9aa0a8">■</font> idiosyncratic {ids:.0f}%', S["caption"]))
    story.append(Spacer(1, 10))

    # Plain-English read
    _sig_word = "statistically significant" if sig else "not statistically significant"
    _skill = ("evidence of selection skill beyond the factor tilts."
              if (sig and a_mf > 0) else
              "so the book is a factor tilt, not demonstrated stock-selection alpha.")
    story.append(Table(
        [[Paragraph(
            f'This {n}-position book carries <b>{d["beta_mkt"]:.2f} market beta</b>; '
            f'<b>{d["r2_full"]*100:.0f}%</b> of its daily variance is market + sector '
            f'beta (dominated by {_esc(d["lead_txt"])}). Jensen alpha is '
            f'<b>{a_mf:+.1f}%/yr</b> and is <b>{_sig_word}</b> (t {a_t:.1f}) - {_skill} '
            f'The {n} positions span ~<b>{d["enb"]:.1f}</b> independent bets.',
            S["body"])]],
        colWidths=[cw],
        style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), BGWARM),
            ("LINEBEFORE", (0, 0), (0, -1), 2, vcol),
            ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ])))
    story.append(Paragraph(
        "Beta from HAC-robust OLS on daily returns; Jensen alpha = multi-factor "
        "intercept (annualised); effective bets = participation ratio of the position "
        "correlation spectrum. Ex-post attribution, not a forward forecast.",
        S["caption"]))
    return story


def _factor_neutral_section(d: dict, S: dict, cw: float) -> list:
    """Companion to the attribution section: does any edge survive stripping the
    academic risk factors (FF5 + Momentum)? Rendered from the skill dict computed
    on the trade page (_compute_factor_neutral_skill)."""
    def _esc(s):
        return (str(s) or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    _VMAP = {"SKILL SURVIVES": GREEN, "MARGINAL": ORANGE}
    vcol = _VMAP.get(d["verdict"], RED)
    dsr = d["dsr"]
    dsr_txt = f'{dsr*100:.0f}%' if dsr == dsr else "n/a"
    rs_col = GREEN if d["res_sharpe"] > 0.2 else (DARK if d["res_sharpe"] >= 0 else RED)

    story = [Spacer(1, 14), _rule(LGRAY, 0.4)]
    story.append(Paragraph("Factor-Neutral Skill Test (Fama-French 5 + Momentum)", S["h3"]))
    story.append(Paragraph(
        "The attribution above shows the book's tilts. This asks the harder question: after "
        "regressing the excess book return on the academic risk factors (market, size, value, "
        "profitability, investment, momentum) and removing them, does any Sharpe survive? A "
        "residual Sharpe near zero means the raw performance was factor exposure, not selection "
        "skill.", S["body"]))
    story.append(Spacer(1, 6))

    # Verdict strip
    story.append(Table(
        [[Paragraph(f'VERDICT&nbsp;&nbsp;<b>{_esc(d["verdict"])}</b>',
                    _ps("fnv", fontName="Helvetica", fontSize=9, textColor=colors.white, leading=12)),
          Paragraph(f'FF5 + Momentum · {d["obs"]} obs · deflated by {d["n_theses"]} theses',
                    _ps("fns", fontName="Helvetica", fontSize=7.5, textColor=colors.white,
                        alignment=TA_RIGHT, leading=12))]],
        colWidths=[cw * 0.5, cw * 0.5],
        style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), vcol),
            ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ])))
    story.append(Spacer(1, 8))

    def _kpi(lbl, val, sub, vc=DARK):
        _vc = "#" + vc.hexval()[2:]
        return Paragraph(
            f'<font size=6 color="#8a8f98">{_esc(lbl)}</font><br/>'
            f'<font size=15 color="{_vc}"><b>{val}</b></font><br/>'
            f'<font size=6 color="#8a8f98">{sub}</font>',
            _ps(f"fk{lbl}", fontName="Helvetica", fontSize=8, leading=12))
    a_col = GREEN if (d["alpha"] > 0 and abs(d["alpha_t"]) >= 2) else (DARK if d["alpha"] >= 0 else RED)
    story.append(Table(
        [[_kpi("RAW SHARPE", f'{d["raw_sharpe"]:.2f}', "book excess, annualised"),
          _kpi("FACTOR-NEUTRAL SHARPE", f'{d["res_sharpe"]:.2f}',
               f'{d["retained"]:.0f}% of raw survives', rs_col),
          _kpi("FF5+MOM ALPHA", f'{d["alpha"]:+.1f}%/yr',
               f't {d["alpha_t"]:.1f} · R² {d["r2"]*100:.0f}%', a_col),
          _kpi("DEFLATED SHARPE", dsr_txt, "P(skill real)", vcol)]],
        colWidths=[cw / 4] * 4,
        style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), BGWARM),
            ("BOX", (0, 0), (-1, -1), 0.5, LGRAY),
            ("LINEBEFORE", (1, 0), (-1, -1), 0.5, LGRAY),
            ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING", (0, 0), (-1, -1), 9), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ])))
    story.append(Spacer(1, 10))

    # FF5+MOM loadings table
    story.append(Paragraph("Risk-Factor Loadings", S["h3"]))
    _bmax = max((abs(b) for _, b, _ in d["loadings"]), default=1.0) or 1.0
    _rows = [[Paragraph("<b>FACTOR</b>", S["body_sm"]),
              Paragraph("<b>BETA</b>", _ps("fh1", fontName="Helvetica-Bold", fontSize=7.5, alignment=TA_RIGHT)),
              Paragraph("<b>t</b>", _ps("fh2", fontName="Helvetica-Bold", fontSize=7.5, alignment=TA_RIGHT)),
              Paragraph("", S["body_sm"])]]
    for name, b, tt in d["loadings"]:
        bc = GREEN if b >= 0 else RED
        _tc = DARK if abs(tt) >= 2 else colors.HexColor("#9aa0a8")
        _rows.append([
            Paragraph(_esc(name), _ps(f"fn{name}", fontName="Helvetica-Bold", fontSize=8, textColor=_tc)),
            Paragraph(f"{b:+.2f}", _ps(f"fb{name}", fontName="Helvetica", fontSize=8, textColor=_tc, alignment=TA_RIGHT)),
            Paragraph(f"{tt:+.1f}", _ps(f"ft{name}", fontName="Helvetica", fontSize=8, textColor=_tc, alignment=TA_RIGHT)),
            _fa_bar(abs(b) / _bmax, bc, w=min(150, cw * 0.30)),
        ])
    story.append(Table(
        _rows, colWidths=[cw * 0.34, cw * 0.12, cw * 0.12, cw * 0.42],
        style=TableStyle([
            ("LINEBELOW", (0, 0), (-1, 0), 0.5, LGRAY),
            ("LINEBELOW", (0, 1), (-1, -2), 0.25, colors.HexColor("#f0efeb")),
            ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ])))
    story.append(Spacer(1, 10))

    _skill = ("selection skill that survives every known factor."
              if d["verdict"] == "SKILL SURVIVES" else
              "no demonstrated selection skill once the known factors are removed.")
    story.append(Table(
        [[Paragraph(
            f'The book\'s raw excess Sharpe of <b>{d["raw_sharpe"]:.2f}</b> is mostly factor '
            f'exposure. Neutralising FF5 + Momentum leaves a <b>{d["res_sharpe"]:.2f}</b> residual '
            f'Sharpe ({d["retained"]:.0f}% of the raw). The deflated-Sharpe probability of real '
            f'skill, after correcting for {d["n_theses"]} theses searched, is <b>{dsr_txt}</b>. '
            f'The dominant hidden exposure the thematic panel missed is '
            f'<b>{_esc(d["lead_factor"])}</b>. In short: {_skill}',
            S["body"])]],
        colWidths=[cw],
        style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), BGWARM),
            ("LINEBEFORE", (0, 0), (0, -1), 2, vcol),
            ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ])))
    story.append(Paragraph(
        "Ken French FF5 + Momentum daily factors, HAC errors. Residual Sharpe is the factor-neutral "
        "information ratio; deflated-Sharpe probability follows Bailey and Lopez de Prado, deflated "
        "by the thesis count. The real test of selection skill, not a forward forecast.", S["caption"]))
    return story


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
    factor_decomp: Optional[dict] = None,
    skill_decomp: Optional[dict] = None,
) -> bytes:
    """Build and return the full PDF as bytes."""
    buf = io.BytesIO()
    S   = _S()
    cw  = W - 30*mm   # content width

    doc = BaseDocTemplate(
        buf, pagesize=A4,
        leftMargin=15*mm, rightMargin=15*mm,
        topMargin=20*mm, bottomMargin=18*mm,
        title="Equity-Commodities Spillover Monitor",
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

    story.append(Table(
        [[Paragraph(
            "<b>Mandate.</b> This report is cross-asset risk and hedging intelligence, not a claim of "
            "stock-selection alpha. The regime-driven book is audited against its own factors on the "
            "Book Risk Character pages: it is market and factor beta, with no statistically significant "
            "factor-neutral alpha and exposures that drift over time. Read the trade ideas as structured "
            "expressions of regime and contagion risk to size and hedge, not as a forecast of forward "
            "returns.",
            S["body"])]],
        colWidths=[cw],
        style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), BGWARM),
            ("LINEBEFORE", (0, 0), (0, -1), 2, GOLD),
            ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ])))
    story.append(Spacer(1, 9))

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
            "The chart below shows the 60-day rolling average of absolute equity-commodities correlation "
            "across all equity-commodity pairs. Background shading indicates the detected regime "
            "(green = Decorrelated, grey = Normal, orange = Elevated, red = Crisis).",
            S["body"],
        ),
        Spacer(1, 8),
        _chart_regime_timeline(avg_corr_series, regimes, w_mm=cw / mm, h_mm=72),
        _chart_caption(
            "Figure 1: Rolling 60-day avg |equity-commodities correlation| with adaptive percentile regime "
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
            "(45%, VIX proxy), slow equity-commodities correlation (35%), commodity energy+metals volatility "
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
            "historical spillover patterns and equity-commodities regime analysis.",
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

    # ── 6b. BOOK RISK CHARACTER - FACTOR & ALPHA ATTRIBUTION ────────────────
    # Is the book alpha or beta? Rendered from the decomposition computed on the
    # trade page. Best-effort: a bad/absent dict just omits the section.
    if factor_decomp:
        try:
            story += [PageBreak()]
            story += _factor_attribution_section(factor_decomp, S, cw)
            if skill_decomp:
                story += _factor_neutral_section(skill_decomp, S, cw)
        except Exception:
            pass

    # ── 7. GEOPOLITICAL CONTEXT ─────────────────────────────────────────────
    if geopolitical_events:
        story += [PageBreak()]
        story += _section_header("Geopolitical Risk Context")
        story += [
            Paragraph(
                "Geopolitical and macroeconomic events identified as structurally significant "
                "for equity-commodities correlation and commodity pricing. Active events continue to "
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
         "Rolling 60-day Pearson correlations computed pairwise. Mean |equity-commodities corr| "
         "is the primary stress signal. Regimes use percentile thresholds (20th/55th/80th), "
         "5-day median smoothing, ±5pp hysteresis bands, and a 10-day persistence gate for Crisis."),
        ("Composite Stress Index",
         "0–100 blend: equity realised vol 45% (VIX proxy, z-score), slow correlation 35% "
         "(rolling percentile), commodity vol 15% (z-score), fast correlation 5%. "
         "Z-score mapping preserves absolute level information."),
        ("Trade Idea Framework",
         "Regime-triggered library covering Crisis Hedge, Geopolitical, Macro, and Growth "
         "categories. Each idea has quantitative entry/exit conditions and key risks."),
        ("How Trade Ideas Are Decided",
         "Every candidate - single-name equities (US / India / China) and macro / commodity "
         "expressions - is mapped to the active regime's spillover and conflict signals, then "
         "run through a five-stage gate: (1) SIGNAL from the regime and structural-exposure "
         "model; (2) PRIOR-ALIGNED Stage-3 confirmation, where the leg must historically move "
         "in the predicted direction in the triggering regime; (3) a DIRECTION-AWARE, "
         "regime-conditional BACKTEST; (4) a DEFLATED-SHARPE screen whose trial count scales "
         "with the size of the candidate universe, so a wider search faces a strictly harder "
         "bar (no data-mining free lunch); and (5) RISK-ADJUSTED SIZING. The constructed book "
         "is a FULLY-INVESTED EQUITY SLEEVE: capital is sized by each idea's risk-adjusted "
         "expected edge (backtest mean ÷ volatility, shrunk by deflated-Sharpe confidence) "
         "and concentrated in the strongest names, with historically money-losing signals "
         "earning no weight. Cash and hedging are the parent portfolio's decision, not this "
         "sleeve's. Each card is anchored to recent third-party coverage (last ~30 days) so the "
         "quantitative signal can be cross-checked against the current market narrative."),
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
    story += [PageBreak()]
    story += _section_header("Important Disclaimer & Legal Notices")

    _disc_lead = (
        "PLEASE READ THESE NOTICES CAREFULLY. By accessing, reading, or otherwise using this "
        "report (the \"Report\") you acknowledge that you have read, understood, and agreed to "
        "the terms, limitations, and disclaimers set out below. If you do not agree, you should "
        "not rely on this Report in any way and should disregard its contents in their entirety."
    )

    _disc = [
        ("Nature and Purpose of this Report",
         "This Report is prepared by graduate students of Purdue University's Daniels School of "
         "Business solely for academic, research, and educational purposes in connection with a "
         "Master of Science in Finance course of study. It is an illustrative demonstration of "
         "quantitative, statistical, and econometric techniques applied to cross-asset equity and "
         "commodity spillover analysis. It is not a commercial research product, has not been "
         "prepared in accordance with any legal or regulatory requirements designed to promote the "
         "independence of investment research, and is not subject to any prohibition on dealing "
         "ahead of the dissemination of investment research."),
        ("No Investment, Financial, Legal, or Tax Advice",
         "Nothing contained in this Report constitutes, or should be construed as, investment, "
         "financial, trading, legal, tax, accounting, or any other form of professional advice, nor "
         "a personal recommendation. No content herein has been prepared with regard to the specific "
         "investment objectives, financial situation, risk tolerance, time horizon, or particular "
         "needs of any person who may read it. You must not treat any content in this Report as a "
         "substitute for the exercise of your own judgement or for advice from a licensed and "
         "qualified professional authorised in your jurisdiction."),
        ("No Offer, Solicitation, or Recommendation",
         "This Report does not constitute and shall not be construed as an offer, invitation, "
         "inducement, advertisement, or solicitation to buy, sell, subscribe for, hold, or otherwise "
         "transact in any security, commodity, future, option, derivative, fund interest, digital "
         "asset, or other financial instrument, nor to adopt any particular trading, hedging, or "
         "investment strategy, in any jurisdiction in which such an offer, solicitation, or "
         "recommendation would be unlawful or unauthorised. No security or instrument referenced "
         "herein is being offered or sold by the authors."),
        ("No Fiduciary or Advisory Relationship",
         "The authors, Purdue University, and the Daniels School of Business are not registered or "
         "licensed as investment advisers, broker-dealers, commodity trading advisors, futures "
         "commission merchants, financial planners, or in any similar capacity with any securities or "
         "commodities regulator, and are not acting in any fiduciary or advisory capacity toward any "
         "reader. No advisory, fiduciary, agency, or client relationship of any kind is created by "
         "the preparation, distribution of, or any reader's access to, this Report."),
        ("Illustrative and Hypothetical Trade Ideas",
         "All \"trade ideas,\" positions, directions, portfolio weights, targets, entry and exit "
         "levels, holding periods, and constructed books shown are hypothetical, mechanical outputs "
         "of research models and are presented purely to illustrate an analytical framework. They are "
         "not, and must not be read as, actionable trade recommendations or a model portfolio for any "
         "person. They have not been executed in any live account. No representation is made that any "
         "account has achieved, or is likely to achieve, profits, losses, or risk characteristics "
         "similar to those depicted."),
        ("Hypothetical and Back-Tested Performance",
         "Any performance, return, expected-edge, or upside figures are hypothetical and derived from "
         "back-tests, simulations, or model estimates. Hypothetical and back-tested performance has "
         "numerous inherent and material limitations: it is prepared with the benefit of hindsight; "
         "it does not involve or reflect actual financial risk, capital, or the effects of emotion, "
         "discipline, liquidity, or funding constraints on real-world decision-making; and, unless "
         "expressly stated otherwise, it does not reflect the impact of commissions, transaction "
         "costs, financing and borrowing costs, market impact, slippage, bid-offer spreads, taxes, or "
         "fees, each of which would reduce actual returns. Results are further subject to survivorship "
         "bias, look-ahead bias, selection bias, and data-mining and over-fitting risk. Small changes "
         "in data, assumptions, parameters, or sample periods can produce materially different "
         "outcomes. PAST, HYPOTHETICAL, AND BACK-TESTED PERFORMANCE IS NOT A RELIABLE INDICATOR OF, "
         "AND IS NOT INDICATIVE OR A GUARANTEE OF, FUTURE RESULTS."),
        ("Forward-Looking Statements, Projections, and Scenarios",
         "Statements regarding expected or targeted returns, annualised figures, bull and bear cases, "
         "scenario payoffs, breakeven probabilities, confidence measures, regime forecasts, and any "
         "other forward-looking content represent estimates, assumptions, and opinions only, are "
         "subject to significant business, economic, market, and model uncertainty, and may prove to "
         "be incorrect. They are not guarantees, promises, or assurances of any future outcome. Actual "
         "events, conditions, and results may differ materially from those expressed or implied, and "
         "the authors undertake no obligation to update any forward-looking statement."),
        ("Model, Methodological, and Regime Risk",
         "The analysis relies on statistical and econometric models, including correlation-regime "
         "detection, deflated-Sharpe screening, spillover, transmission, and connectedness measures, "
         "conflict-intensity scoring, and scenario engines, all of which are simplifications of a "
         "complex and evolving reality. Such models may contain errors, may be mis-specified or "
         "mis-calibrated, and can fail, particularly during structural breaks, regime shifts, policy "
         "shocks, liquidity crises, and extreme or tail events, precisely when reliable guidance is "
         "most needed. Statistical relationships and correlations estimated from historical data may "
         "weaken, reverse, or break down entirely without notice."),
        ("Third-Party Data, News, and Analyst Commentary",
         "Market data, prices, fundamentals, macroeconomic series, and any recent news headlines, "
         "articles, or analyst-commentary items reproduced or summarised in this Report are obtained "
         "from third-party sources, which may include Yahoo Finance, the yfinance library, FRED, and "
         "a variety of news publishers and aggregators, believed to be reliable but which have not "
         "been independently verified. Such third-party content is provided for background and context "
         "only. It is not endorsed, adopted, verified, or warranted by the authors; it may be "
         "inaccurate, incomplete, out of date, mis-attributed, or associated with a company or ticker "
         "only by automated matching; and its inclusion does not imply that the authors agree with, or "
         "vouch for the accuracy of, any statement, rating, price target, or opinion it contains. The "
         "authors make no representation or warranty, express or implied, as to the accuracy, "
         "completeness, timeliness, reliability, or fitness for any purpose of any data, headline, or "
         "content in this Report, and disclaim any duty to update it."),
        ("General Risk Warning",
         "All investing, trading, and speculation involves substantial risk, including the risk of "
         "losing some, all, or (where leverage or derivatives are used) more than the amount invested. "
         "The value of investments and any income from them can fall as well as rise, is not "
         "guaranteed, and you may not get back the amount originally invested. Commodities, futures, "
         "derivatives, leveraged and short strategies, and instruments exposed to geopolitical, "
         "energy, currency, and cross-border risk can be exceptionally volatile and illiquid and may "
         "result in rapid, substantial, or total loss of capital. Diversification, hedging, and "
         "risk-management models do not assure a profit and do not protect against loss in declining "
         "or dislocated markets."),
        ("No Reliance and Independent Due Diligence",
         "You should not use or rely upon this Report as the sole or primary basis for any investment "
         "or trading decision. Any person considering any transaction should conduct their own "
         "independent research, analysis, and due diligence, form their own independent view, and "
         "obtain specific and appropriate professional advice relevant to their own circumstances, "
         "objectives, and jurisdiction before acting or refraining from acting."),
        ("Limitation of Liability",
         "To the fullest extent permitted by applicable law, the authors, Purdue University, the "
         "Daniels School of Business, and their respective trustees, faculty, staff, students, and "
         "affiliates expressly disclaim and accept no responsibility or liability whatsoever, whether "
         "in contract, tort (including negligence), breach of statutory duty, or otherwise, for any "
         "direct, indirect, incidental, special, punitive, or consequential loss or damage of any "
         "kind, including loss of profit, loss of capital, or loss of opportunity, arising out of or "
         "in connection with the access to, use of, or reliance on this Report, its contents, or any "
         "data or third-party content referenced herein, even if advised of the possibility of such "
         "loss."),
        ("Intellectual Property, Confidentiality, and Distribution",
         "This Report is the academic work product of its named authors and is intended for internal "
         "coursework, faculty review, and educational use only. It may not be reproduced, "
         "redistributed, published, quoted, or relied upon by any third party, in whole or in part, "
         "for any commercial purpose, without the authors' prior written consent. All third-party "
         "trademarks, service marks, trade names, and content remain the property of their respective "
         "owners, and their use here is for identification and educational purposes only."),
        ("Jurisdiction and Governing Terms",
         "This Report and these notices are provided from an academic setting within the United States "
         "and are intended for a limited academic audience. The Report may not be lawful or appropriate "
         "for use in all jurisdictions, and it is the responsibility of any reader to inform themselves "
         "about, and to observe, any applicable laws and regulations. Nothing in these notices excludes "
         "or limits any liability that cannot lawfully be excluded or limited."),
    ]

    story += [Paragraph(_disc_lead, S["disclaimer"]), Spacer(1, 5)]
    for _lbl, _txt in _disc:
        story += [Paragraph(f"<b>{_lbl}.</b>&nbsp; {_txt}", S["disclaimer"]),
                  Spacer(1, 4)]
    story += [
        Spacer(1, 8),
        HRFlowable(width="100%", thickness=0.5, color=GOLD),
        Spacer(1, 6),
        Paragraph(
            f"© {datetime.now().year} Purdue University · Daniels School of Business · "
            f"MSF Research Terminal",
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
