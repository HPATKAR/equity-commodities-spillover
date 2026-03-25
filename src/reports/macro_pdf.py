"""
Macro Intelligence Dashboard - PDF Report Generator
Produces an institutional-style PDF snapshot of all macro sections.
Uses reportlab + matplotlib (already in requirements).
"""

from __future__ import annotations

import io
from datetime import datetime, date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate,
    Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether, Image as RLImage,
)

# ── Palette ─────────────────────────────────────────────────────────────────
GOLD  = colors.HexColor("#CFB991")
AGED  = colors.HexColor("#8E6F3E")
BLACK = colors.HexColor("#000000")
WHITE = colors.white
GRAY  = colors.HexColor("#555960")
LGRAY = colors.HexColor("#E8E5E0")
BGWM  = colors.HexColor("#fafaf8")
RED   = colors.HexColor("#c0392b")
GREEN = colors.HexColor("#2e7d32")
BLUE  = colors.HexColor("#2980b9")

W, H = A4


# ── Styles ───────────────────────────────────────────────────────────────────
def _styles():
    base = getSampleStyleSheet()
    S = {}
    common = dict(fontName="Helvetica", leading=13, spaceAfter=4)
    S["cover_title"] = ParagraphStyle("cover_title", fontSize=22, textColor=WHITE,
                                       fontName="Helvetica-Bold", alignment=TA_LEFT, leading=28)
    S["cover_sub"]   = ParagraphStyle("cover_sub",   fontSize=10, textColor=GOLD,
                                       fontName="Helvetica", alignment=TA_LEFT, leading=14)
    S["cover_meta"]  = ParagraphStyle("cover_meta",  fontSize=8,  textColor=colors.HexColor("#aaaaaa"),
                                       fontName="Helvetica", alignment=TA_LEFT, leading=11)
    S["section"]     = ParagraphStyle("section",     fontSize=9,  textColor=AGED,
                                       fontName="Helvetica-Bold", leading=12,
                                       textTransform="uppercase", letterSpacing=1.2,
                                       spaceAfter=3, spaceBefore=10)
    S["body"]        = ParagraphStyle("body",         fontSize=8,  textColor=colors.HexColor("#333"),
                                       fontName="Helvetica", leading=12, spaceAfter=3)
    S["caption"]     = ParagraphStyle("caption",      fontSize=7,  textColor=GRAY,
                                       fontName="Helvetica-Oblique", leading=10, spaceAfter=4)
    S["kpi_label"]   = ParagraphStyle("kpi_label",    fontSize=6.5, textColor=AGED,
                                       fontName="Helvetica-Bold", leading=9,
                                       textTransform="uppercase", letterSpacing=0.8)
    S["kpi_value"]   = ParagraphStyle("kpi_value",    fontSize=14, textColor=BLACK,
                                       fontName="Helvetica-Bold", leading=17)
    S["kpi_delta"]   = ParagraphStyle("kpi_delta",    fontSize=7.5, textColor=GRAY,
                                       fontName="Helvetica", leading=10)
    return S


# ── Matplotlib chart → RLImage ───────────────────────────────────────────────
def _fig_to_rl(fig, width_mm: float = 170, height_mm: float = 55) -> RLImage:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return RLImage(buf, width=width_mm * mm, height=height_mm * mm)


def _mpl_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#E8E5E0")
    ax.spines["bottom"].set_color("#E8E5E0")
    ax.tick_params(colors="#555", labelsize=6.5)
    ax.yaxis.label.set_size(7)
    ax.yaxis.label.set_color("#555")
    ax.set_facecolor("white")
    ax.grid(axis="y", color="#E8E5E0", linewidth=0.5, alpha=0.7)


# ── KPI strip ────────────────────────────────────────────────────────────────
def _kpi_table(items: list[dict], S) -> Table:
    """items = [{"label": ..., "value": ..., "delta": ..., "up": True/False/None}]"""
    headers = [Paragraph(it["label"], S["kpi_label"]) for it in items]
    values  = [Paragraph(it["value"], S["kpi_value"]) for it in items]
    deltas  = []
    for it in items:
        d = it.get("delta", "")
        up = it.get("up")
        col = "#2e7d32" if up is True else "#c0392b" if up is False else "#888"
        arrow = "▲" if up is True else "▼" if up is False else ""
        deltas.append(Paragraph(f'<font color="{col}">{arrow} {d}</font>', S["kpi_delta"]))
    data = [headers, values, deltas]
    t = Table(data, colWidths=[W / len(items) - 12 * mm] * len(items))
    t.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, LGRAY),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("BACKGROUND", (0, 0), (-1, -1), BGWM),
        ("BOX", (0, 0), (-1, -1), 0.5, LGRAY),
        ("INNERGRID", (0, 0), (-1, -1), 0.3, LGRAY),
    ]))
    return t


# ── Data table helper ────────────────────────────────────────────────────────
def _data_table(df: pd.DataFrame, S, col_widths=None) -> Table:
    headers = [Paragraph(str(c), S["kpi_label"]) for c in df.columns]
    rows = [headers]
    for _, row in df.iterrows():
        rows.append([Paragraph(str(v) if not pd.isna(v) else "-", S["body"]) for v in row])
    ncols = len(df.columns)
    cw = col_widths or [(W - 20 * mm) / ncols] * ncols
    t = Table(rows, colWidths=cw, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BLACK),
        ("TEXTCOLOR", (0, 0), (-1, 0), GOLD),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 6.5),
        ("FONTSIZE", (0, 1), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, BGWM]),
        ("GRID", (0, 0), (-1, -1), 0.3, LGRAY),
        ("LINEBELOW", (0, 0), (-1, 0), 1, GOLD),
    ]))
    return t


# ── Chart builders ───────────────────────────────────────────────────────────
def _chart_yields(yields_df: pd.DataFrame) -> RLImage:
    fig, ax = plt.subplots(figsize=(7, 2.3))
    colors_list = ["#CFB991", "#8E6F3E", "#c0392b", "#2e7d32", "#2980b9"]
    for i, col in enumerate(yields_df.columns):
        ax.plot(yields_df.index, yields_df[col], label=col,
                color=colors_list[i % len(colors_list)], linewidth=1.2)
    ax.set_ylabel("Yield (%)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    ax.legend(fontsize=6, ncol=5, loc="upper left", framealpha=0.5)
    ax.set_title("US Treasury Yields", fontsize=8, color="#333", pad=4)
    _mpl_style(ax)
    fig.tight_layout(pad=0.5)
    return _fig_to_rl(fig, 170, 58)


def _chart_yield_curve(yields_df: pd.DataFrame) -> RLImage:
    maturities = {"3M": 0.25, "2Y": 2, "5Y": 5, "10Y": 10, "30Y": 30}
    latest = yields_df.iloc[-1].dropna()
    x, y = [], []
    for lbl, yrs in maturities.items():
        if lbl in latest.index:
            x.append(yrs); y.append(latest[lbl])
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    ax.plot(x, y, color="#CFB991", linewidth=2, marker="o", markersize=5,
            markerfacecolor="#8E6F3E")
    ax.set_xlabel("Maturity (Years)", fontsize=6.5)
    ax.set_ylabel("Yield (%)")
    ax.set_title(f"Spot Yield Curve\n{yields_df.index[-1].strftime('%d %b %Y')}", fontsize=7.5,
                 color="#333", pad=4)
    _mpl_style(ax)
    fig.tight_layout(pad=0.5)
    return _fig_to_rl(fig, 75, 58)


def _chart_spreads(spreads_df: pd.DataFrame) -> RLImage:
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.3))
    curve_cols = [c for c in ["10Y–2Y (Yield Curve)", "10Y–3M"] if c in spreads_df]
    cred_cols  = [c for c in ["IG Credit Spread", "HY Credit Spread"] if c in spreads_df]
    clrs = {"10Y–2Y (Yield Curve)": "#CFB991", "10Y–3M": "#8E6F3E",
            "IG Credit Spread": "#2e7d32", "HY Credit Spread": "#c0392b"}
    for col in curve_cols:
        axes[0].plot(spreads_df.index, spreads_df[col].dropna(),
                     label=col, color=clrs[col], linewidth=1.2)
    axes[0].axhline(0, color="#c0392b", linestyle="--", linewidth=0.8, alpha=0.6)
    axes[0].set_title("Yield Curve Spreads", fontsize=7.5, color="#333")
    axes[0].legend(fontsize=5.5, loc="upper left")
    for col in cred_cols:
        axes[1].plot(spreads_df.index, spreads_df[col].dropna(),
                     label=col, color=clrs[col], linewidth=1.2)
    axes[1].set_title("Credit Spreads (OAS)", fontsize=7.5, color="#333")
    axes[1].legend(fontsize=5.5, loc="upper left")
    for ax in axes:
        _mpl_style(ax)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    fig.tight_layout(pad=0.5)
    return _fig_to_rl(fig, 170, 58)


def _chart_gdp(macro_data: dict) -> RLImage:
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.3))
    if "Real GDP Growth (QoQ %)" in macro_data:
        s = macro_data["Real GDP Growth (QoQ %)"].last("10Y")
        bar_colors = ["#2e7d32" if v >= 0 else "#c0392b" for v in s.values]
        axes[0].bar(s.index, s.values, color=bar_colors, width=60, alpha=0.85)
        axes[0].axhline(0, color="#aaa", linewidth=0.7, linestyle="--")
        axes[0].set_title("Real GDP Growth (QoQ %)", fontsize=7.5, color="#333")
        axes[0].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    if "ISM Manufacturing PMI" in macro_data:
        s = macro_data["ISM Manufacturing PMI"].last("5Y")
        axes[1].plot(s.index, s.values, color="#CFB991", linewidth=1.5)
        axes[1].axhline(50, color="#c0392b", linestyle="--", linewidth=0.8, alpha=0.7)
        axes[1].fill_between(s.index, s.values, 50,
                             where=s.values >= 50, alpha=0.12, color="#2e7d32")
        axes[1].fill_between(s.index, s.values, 50,
                             where=s.values < 50, alpha=0.12, color="#c0392b")
        axes[1].set_title("ISM Manufacturing PMI", fontsize=7.5, color="#333")
    for ax in axes:
        _mpl_style(ax)
    fig.tight_layout(pad=0.5)
    return _fig_to_rl(fig, 170, 58)


def _chart_hfreq(macro_data: dict) -> RLImage:
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.3))
    if "Retail Sales ex-Auto" in macro_data:
        s = macro_data["Retail Sales ex-Auto"].last("5Y")
        s_pct = s.pct_change(12) * 100
        axes[0].plot(s_pct.dropna().index, s_pct.dropna().values,
                     color="#2980b9", linewidth=1.5)
        axes[0].axhline(0, color="#aaa", linewidth=0.7, linestyle="--")
        axes[0].set_title("Retail Sales ex-Auto (YoY %)", fontsize=7.5, color="#333")
        axes[0].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    if "Nonfarm Payrolls (MoM k)" in macro_data:
        s = (macro_data["Nonfarm Payrolls (MoM k)"].last("5Y") / 1000)
        bar_colors = ["#2e7d32" if v >= 0 else "#c0392b" for v in s.values]
        axes[1].bar(s.index, s.values, color=bar_colors, width=20, alpha=0.85)
        axes[1].set_title("Nonfarm Payrolls (MoM, k)", fontsize=7.5, color="#333")
    for ax in axes:
        _mpl_style(ax)
    fig.tight_layout(pad=0.5)
    return _fig_to_rl(fig, 170, 58)


def _chart_money(money_data: dict) -> RLImage:
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.3))
    if "M2 Money Supply ($B)" in money_data:
        s = money_data["M2 Money Supply ($B)"].last("10Y")
        s_yoy = s.pct_change(12) * 100
        axes[0].plot(s_yoy.dropna().index, s_yoy.dropna().values,
                     color="#CFB991", linewidth=1.5)
        axes[0].axhline(0, color="#c0392b", linestyle="--", linewidth=0.8, alpha=0.7)
        axes[0].fill_between(s_yoy.dropna().index, s_yoy.dropna().values, 0,
                             alpha=0.1, color="#CFB991")
        axes[0].set_title("M2 Money Supply Growth (YoY %)", fontsize=7.5, color="#333")
        axes[0].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    if "Fed Total Assets ($B)" in money_data:
        s = money_data["Fed Total Assets ($B)"].last("10Y")
        axes[1].plot(s.index, s.values / 1000, color="#8E6F3E", linewidth=1.5)
        axes[1].fill_between(s.index, s.values / 1000, alpha=0.1, color="#8E6F3E")
        axes[1].set_title("Fed Balance Sheet ($T)", fontsize=7.5, color="#333")
        axes[1].yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.1fT"))
    for ax in axes:
        _mpl_style(ax)
    fig.tight_layout(pad=0.5)
    return _fig_to_rl(fig, 170, 58)


def _chart_inflation(macro_data: dict) -> RLImage:
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.3))
    if "CPI YoY (%)" in macro_data:
        s = macro_data["CPI YoY (%)"].last("10Y")
        axes[0].plot(s.index, s.values, color="#c0392b", linewidth=1.5)
        axes[0].axhline(2, color="#CFB991", linestyle="--", linewidth=0.8, alpha=0.8,
                        label="Fed Target (2%)")
        axes[0].set_title("CPI Inflation (YoY %)", fontsize=7.5, color="#333")
        axes[0].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
        axes[0].legend(fontsize=5.5, loc="upper left")
    if "5Y Breakeven Inflation (%)" in macro_data:
        s = macro_data["5Y Breakeven Inflation (%)"].last("10Y")
        axes[1].plot(s.index, s.values, color="#8E6F3E", linewidth=1.5)
        axes[1].axhline(2, color="#CFB991", linestyle="--", linewidth=0.8, alpha=0.8,
                        label="Fed Target (2%)")
        axes[1].set_title("5Y Breakeven Inflation (%)", fontsize=7.5, color="#333")
        axes[1].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
        axes[1].legend(fontsize=5.5, loc="upper left")
    for ax in axes:
        _mpl_style(ax)
    fig.tight_layout(pad=0.5)
    return _fig_to_rl(fig, 170, 58)


def _chart_unemployment(macro_data: dict) -> RLImage:
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    if "Unemployment Rate (%)" in macro_data:
        s = macro_data["Unemployment Rate (%)"].last("10Y")
        ax.plot(s.index, s.values, color="#c0392b", linewidth=1.5)
        ax.fill_between(s.index, s.values, alpha=0.08, color="#c0392b")
        ax.set_title("Unemployment Rate (%)", fontsize=7.5, color="#333")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    _mpl_style(ax)
    fig.tight_layout(pad=0.5)
    return _fig_to_rl(fig, 75, 58)


def _chart_consumer_sentiment(money_data: dict) -> RLImage:
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    if "Consumer Sentiment" in money_data:
        s = money_data["Consumer Sentiment"].last("5Y")
        avg = float(s.mean())
        ax.plot(s.index, s.values, color="#2980b9", linewidth=1.5)
        ax.axhline(avg, color="#CFB991", linestyle="--", linewidth=0.8,
                   label=f"Avg {avg:.0f}")
        ax.fill_between(s.index, s.values, avg, where=s.values >= avg,
                        alpha=0.09, color="#2e7d32")
        ax.fill_between(s.index, s.values, avg, where=s.values < avg,
                        alpha=0.09, color="#c0392b")
        ax.set_title("Consumer Sentiment (Michigan)", fontsize=7.5, color="#333")
        ax.legend(fontsize=5.5, loc="upper left")
    _mpl_style(ax)
    fig.tight_layout(pad=0.5)
    return _fig_to_rl(fig, 75, 58)


def _chart_vix(idx_prices: pd.DataFrame) -> RLImage:
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    if "VIX" in idx_prices.columns:
        s = idx_prices["VIX"].dropna().tail(126)  # last ~6 months
        ax.plot(s.index, s.values, color="#c0392b", linewidth=1.2)
        ax.fill_between(s.index, s.values, alpha=0.07, color="#c0392b")
        ax.axhline(20, color="#CFB991", linestyle="--", linewidth=0.8, alpha=0.8, label="20 (caution)")
        ax.axhline(30, color="#c0392b", linestyle="--", linewidth=0.8, alpha=0.8, label="30 (fear)")
        ax.set_title("VIX Fear Index", fontsize=7.5, color="#333")
        ax.legend(fontsize=5.5, loc="upper left")
    _mpl_style(ax)
    fig.tight_layout(pad=0.5)
    return _fig_to_rl(fig, 75, 58)


def _chart_pe(val_df: pd.DataFrame) -> RLImage:
    df = val_df.dropna(subset=["Forward P/E"]) if "Forward P/E" in val_df.columns else pd.DataFrame()
    if df.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 2.3))
    bar_colors = ["#c0392b" if v > 25 else "#CFB991" if v > 18 else "#2e7d32"
                  for v in df["Forward P/E"]]
    bars = ax.bar(df["Market"], df["Forward P/E"], color=bar_colors, alpha=0.85)
    ax.axhline(20, color="#CFB991", linestyle="--", linewidth=0.8, alpha=0.8,
               label="Expensive (20x)")
    ax.axhline(15, color="#2e7d32", linestyle="--", linewidth=0.8, alpha=0.8,
               label="Fair Value (15x)")
    for bar, val in zip(bars, df["Forward P/E"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}x", ha="center", fontsize=6.5, color="#333")
    ax.set_title("Forward P/E by Market", fontsize=7.5, color="#333")
    ax.legend(fontsize=6, loc="upper right")
    _mpl_style(ax)
    fig.tight_layout(pad=0.5)
    return _fig_to_rl(fig, 170, 58)


def _chart_index_perf(idx_prices: pd.DataFrame, today: date) -> RLImage:
    perf_windows = {"1M": 21, "3M": 63, "6M": 126}
    perf_rows = []
    for name in idx_prices.columns:
        if name == "VIX":
            continue
        s = idx_prices[name].dropna()
        if s.empty:
            continue
        row = {"Index": name}
        for wl, wd in perf_windows.items():
            row[wl] = round((s.iloc[-1] / s.iloc[-wd - 1] - 1) * 100, 2) if len(s) > wd else None
        perf_rows.append(row)
    df = pd.DataFrame(perf_rows).set_index("Index").dropna(how="all")
    if df.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 2.3))
    x = np.arange(len(df))
    w = 0.25
    cls = {"1M": "#CFB991", "3M": "#8E6F3E", "6M": "#c0392b"}
    for i, (col, cl) in enumerate(cls.items()):
        if col in df.columns:
            vals = df[col].fillna(0).values
            ax.bar(x + i * w, vals, w, label=col, color=cl, alpha=0.85)
    ax.axhline(0, color="#aaa", linewidth=0.6)
    ax.set_xticks(x + w)
    ax.set_xticklabels(df.index, fontsize=6, rotation=30, ha="right")
    ax.set_title("Index Performance (Rolling Returns)", fontsize=7.5, color="#333")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    ax.legend(fontsize=6)
    _mpl_style(ax)
    fig.tight_layout(pad=0.5)
    return _fig_to_rl(fig, 170, 60)


# ── Page templates ───────────────────────────────────────────────────────────
def _build_doc(buf: io.BytesIO) -> BaseDocTemplate:
    doc = BaseDocTemplate(buf, pagesize=A4,
                          leftMargin=15 * mm, rightMargin=15 * mm,
                          topMargin=15 * mm, bottomMargin=15 * mm)

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
        canvas.drawString(15 * mm, 3 * mm,
                          f"Macro Intelligence Dashboard  ·  Purdue Daniels  ·  Generated {datetime.now().strftime('%d %b %Y %H:%M')}")
        canvas.drawRightString(W - 15 * mm, 3 * mm, f"Page {doc.page}")
        canvas.restoreState()

    cover_frame = Frame(0, 0, W, H, leftPadding=20 * mm, rightPadding=20 * mm,
                        topPadding=50 * mm, bottomPadding=20 * mm)
    body_frame  = Frame(15 * mm, 10 * mm, W - 30 * mm, H - 25 * mm)

    doc.addPageTemplates([
        PageTemplate(id="Cover", frames=[cover_frame], onPage=_cover_bg),
        PageTemplate(id="Body",  frames=[body_frame],  onPage=_body_page),
    ])
    return doc


# ── Main builder ─────────────────────────────────────────────────────────────
def build_macro_pdf(
    yields_df: pd.DataFrame,
    spreads_df: pd.DataFrame,
    macro_data: dict,
    money_data: dict,
    val_df: pd.DataFrame,
    idx_prices: pd.DataFrame,
    start: str,
    end: str,
    narrative: str = "",
) -> bytes:
    """
    Build and return the macro PDF as bytes.
    Pass empty DataFrames / dicts for sections where data is unavailable.
    """
    buf = io.BytesIO()
    doc = _build_doc(buf)
    S = _styles()
    today = date.today()
    story = []

    # ── Cover ────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 20 * mm))
    story.append(Paragraph("Macro Intelligence", S["cover_title"]))
    story.append(Paragraph("Dashboard Report", S["cover_title"]))
    story.append(Spacer(1, 6 * mm))
    story.append(Paragraph(
        f"Purdue University · Daniels School of Business · MGMT 69000-120",
        S["cover_sub"]))
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph(
        f"Analysis period: {start} to {end} · Generated {today.strftime('%d %B %Y')}",
        S["cover_meta"]))
    story.append(Spacer(1, 6 * mm))
    story.append(Paragraph(
        "This report provides a structured macro intelligence brief covering economic growth, "
        "high-frequency indicators, money flows, bond yields, yield spreads, equity valuations, "
        "and index performance. Read it sequentially - each section builds on the previous to "
        "form a coherent view of the current macro regime.",
        S["cover_meta"]))
    # Switch to body template BEFORE the page break so the next page uses Body
    from reportlab.platypus import NextPageTemplate
    story.append(NextPageTemplate("Body"))
    story.append(PageBreak())

    def _section_header(title: str, subtitle: str = ""):
        items = [Paragraph(title.upper(), S["section"])]
        if subtitle:
            items.append(Paragraph(subtitle, S["caption"]))
        items.append(HRFlowable(width="100%", thickness=0.5, color=GOLD, spaceAfter=4))
        return items

    # ── Executive Summary (narrative) ─────────────────────────────────────────
    if narrative.strip():
        story += _section_header("Executive Summary",
                                  "Data-driven macro synthesis - implications for equities and commodities.")
        for para in narrative.strip().split("\n\n"):
            if para.strip():
                story.append(Paragraph(para.strip(), S["body"]))
                story.append(Spacer(1, 2 * mm))
        story.append(PageBreak())

    # ── 1. GDP & Growth ──────────────────────────────────────────────────────
    story += _section_header("1 · Economic Growth",
                              "Real GDP growth and industrial production set the broadest context for the cycle.")
    if macro_data:
        kpi_keys = [
            ("Real GDP Growth (QoQ %)", "Real GDP Growth", "%"),
            ("ISM Manufacturing PMI", "ISM PMI", ""),
            ("Unemployment Rate (%)", "Unemployment", "%"),
            ("CPI YoY (%)", "CPI YoY", "%"),
        ]
        kpi_items = []
        for key, label, suffix in kpi_keys:
            if key in macro_data and not macro_data[key].empty:
                s = macro_data[key]
                latest = s.iloc[-1]
                prev   = s.iloc[-2] if len(s) > 1 else latest
                delta  = latest - prev
                good_up = key not in ("Unemployment Rate (%)", "CPI YoY (%)")
                is_up = delta > 0
                kpi_items.append({
                    "label": label,
                    "value": f"{latest:.2f}{suffix}",
                    "delta": f"{abs(delta):.2f}{suffix}",
                    "up": is_up if good_up else not is_up,
                })
        if kpi_items:
            story.append(_kpi_table(kpi_items, S))
            story.append(Spacer(1, 3 * mm))
        try:
            story.append(_chart_gdp(macro_data))
        except Exception:
            pass
        story.append(Paragraph(
            "PMI above 50 signals manufacturing expansion; below 50 signals contraction. "
            "GDP growth below 0% for two consecutive quarters = technical recession. "
            "Industrial production leads earnings revisions by approximately one quarter.",
            S["caption"]))
    else:
        story.append(Paragraph("GDP data unavailable - FRED API key required.", S["caption"]))

    story.append(Spacer(1, 4 * mm))

    # ── 2. High-Freq Indicators ───────────────────────────────────────────────
    story += _section_header("2 · High-Frequency Indicators",
                              "PMI, retail sales and payrolls are the earliest leading indicators of cycle turns.")
    if macro_data:
        try:
            story.append(_chart_hfreq(macro_data))
        except Exception:
            pass
        # Inflation side-by-side with unemployment
        try:
            from reportlab.platypus import Table as RLTable
            infl_img = _chart_inflation(macro_data)
            unemp_img = _chart_unemployment(macro_data)
            row = RLTable([[infl_img, unemp_img]], colWidths=[115 * mm, 65 * mm])
            row.setStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                          ("LEFTPADDING", (0, 0), (-1, -1), 0),
                          ("RIGHTPADDING", (0, 0), (-1, -1), 3)])
            story.append(row)
        except Exception:
            pass
        story.append(Paragraph(
            "Retail sales declining YoY typically precede earnings downgrades in consumer discretionary by "
            "one to two quarters. Payrolls above +150k/month indicate a healthy labour market. "
            "CPI above the 2% Fed target limits the scope for rate cuts. "
            "Watch PMI new orders sub-index as the earliest signal of demand change.",
            S["caption"]))
    else:
        story.append(Paragraph("High-freq data unavailable - FRED API key required.", S["caption"]))

    story.append(PageBreak())

    # ── 3. Money Flows & Liquidity ────────────────────────────────────────────
    story += _section_header("3 · Money Flows & Liquidity",
                              "M2 growth and the Fed balance sheet determine the liquidity backdrop for risk assets.")
    if money_data:
        try:
            story.append(_chart_money(money_data))
        except Exception:
            pass
        # Consumer sentiment alongside M2/Fed
        try:
            from reportlab.platypus import Table as RLTable
            sent_img = _chart_consumer_sentiment(money_data)
            sent_row = RLTable([[sent_img]], colWidths=[75 * mm])
            sent_row.setStyle([("LEFTPADDING", (0, 0), (-1, -1), 0)])
            story.append(sent_img)
        except Exception:
            pass
        story.append(Paragraph(
            "M2 contraction (negative YoY) has historically preceded asset price stress. "
            "The 2022 drawdown coincided with the first M2 decline since the 1930s. "
            "Fed QE expands the balance sheet and injects liquidity; QT withdraws it. "
            "Consumer sentiment below its long-run average signals reduced retail participation.",
            S["caption"]))
    else:
        story.append(Paragraph("Money flow data unavailable - FRED API key required.", S["caption"]))

    story.append(Spacer(1, 4 * mm))

    # ── 4. Bond Yields ────────────────────────────────────────────────────────
    story += _section_header("4 · Bond Yields",
                              "Higher long-end yields raise the equity discount rate and compress valuations.")
    if not yields_df.empty:
        try:
            from reportlab.platypus import Table as RLTable
            yld_img   = _chart_yields(yields_df)
            curve_img = _chart_yield_curve(yields_df)
            row = RLTable([[yld_img, curve_img]], colWidths=[110 * mm, 70 * mm])
            row.setStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                          ("LEFTPADDING", (0, 0), (-1, -1), 0),
                          ("RIGHTPADDING", (0, 0), (-1, -1), 3)])
            story.append(row)
        except Exception:
            pass
        story.append(Paragraph(
            "When the 10Y yield exceeds the S&P 500 earnings yield, bonds become competitive with equities - "
            "watch this spread as a regime signal. The spot yield curve (right) shows the current shape: "
            "a flat or inverted curve signals recession expectations.",
            S["caption"]))
    else:
        story.append(Paragraph("Yield data unavailable - FRED API key required.", S["caption"]))

    story.append(PageBreak())

    # ── 5. Yield Spreads & Credit ─────────────────────────────────────────────
    story += _section_header("5 · Yield Spreads & Credit Risk",
                              "Credit spreads are the market's real-time verdict on default risk and financial conditions.")
    if not spreads_df.empty:
        try:
            story.append(_chart_spreads(spreads_df))
        except Exception:
            pass
        story.append(Paragraph(
            "Credit spreads widen before equity markets sell off - they are a useful leading risk indicator. "
            "HY spreads above 600bps have historically coincided with recessions. "
            "An inverted yield curve alongside widening spreads is the strongest combined recession signal.",
            S["caption"]))
    else:
        story.append(Paragraph("Spread data unavailable - FRED API key required.", S["caption"]))

    story.append(Spacer(1, 4 * mm))

    # ── 6. Valuations & Earnings ──────────────────────────────────────────────
    story += _section_header("6 · Valuations & Expected Earnings Growth",
                              "Forward P/E determines how much earnings growth must materialise to justify current prices.")
    if not val_df.empty and "Forward P/E" in val_df.columns:
        display_cols = ["Market", "Trailing P/E", "Forward P/E", "Earnings Yield %", "Fwd EPS Growth %"]
        show_cols = [c for c in display_cols if c in val_df.columns]
        disp = val_df[show_cols].copy()
        for c in ["Trailing P/E", "Forward P/E", "Earnings Yield %"]:
            if c in disp.columns:
                disp[c] = disp[c].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "-")
        if "Fwd EPS Growth %" in disp.columns:
            disp["Fwd EPS Growth %"] = disp["Fwd EPS Growth %"].apply(
                lambda x: f"{x:+.1f}%" if pd.notna(x) else "-")
        story.append(_data_table(disp, S))
        story.append(Spacer(1, 3 * mm))
        try:
            chart = _chart_pe(val_df)
            if chart:
                story.append(chart)
        except Exception:
            pass
        story.append(Paragraph(
            "Forward P/E above 20x demands strong earnings growth to justify. "
            "With elevated rates (section 4), the equity risk premium compresses - "
            "making growth disappointments more painful for high-multiple markets.",
            S["caption"]))
    else:
        story.append(Paragraph("Valuation data unavailable.", S["caption"]))

    story.append(PageBreak())

    # ── 7. Index Performance ──────────────────────────────────────────────────
    story += _section_header("7 · Index Performance & Market Stress",
                              "The cumulative market verdict on the macro backdrop above.")
    if not idx_prices.empty:
        try:
            chart = _chart_index_perf(idx_prices, today)
            if chart:
                story.append(chart)
        except Exception:
            pass
        # VIX alongside index table
        try:
            from reportlab.platypus import Table as RLTable
            vix_img = _chart_vix(idx_prices)
            story.append(vix_img)
        except Exception:
            pass
        story.append(Paragraph(
            "Cross-reference with sections 1–6: strong GDP and low spreads should explain positive returns. "
            "Drawdowns alongside widening credit spreads or inverted yield curves confirm the macro signal. "
            "VIX above 30 indicates fear-driven positioning - historically a contrarian buy signal when "
            "fundamentals have not yet deteriorated.",
            S["caption"]))
    else:
        story.append(Paragraph("Index data unavailable.", S["caption"]))

    story.append(Spacer(1, 6 * mm))

    # ── Disclaimer ────────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=LGRAY))
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph(
        "DISCLAIMER: This report is produced for academic purposes as part of Purdue University MGMT 69000-120. "
        "It does not constitute investment advice. All data is sourced from public providers (FRED, Yahoo Finance) "
        "and may contain errors or omissions. Past performance is not indicative of future results.",
        S["caption"]))

    doc.build(story)
    buf.seek(0)
    return buf.read()
