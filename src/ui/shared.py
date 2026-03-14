"""
Shared UI helpers - Purdue theme, chart styling, text components.
Mirrors the JGB dashboard's shared.py adapted for cross-asset use.
"""

from __future__ import annotations

import base64
import functools
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from src.data.config import PALETTE, GEOPOLITICAL_EVENTS, CATEGORY_COLORS

_ASSETS = Path(__file__).parents[2] / "assets"
_GOLD   = "#CFB991"
_BLACK  = "#000000"

# ── Plotly template ────────────────────────────────────────────────────────

_PURDUE_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        font=dict(family="DM Sans, Inter, sans-serif", color=_BLACK, size=12),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#fafaf8",
        colorway=PALETTE,
        xaxis=dict(
            showgrid=True, gridcolor="#EEEBE6", gridwidth=1,
            zeroline=False, showspikes=True, spikecolor="#CFB991",
            spikethickness=1, spikedash="dot",
            tickfont=dict(family="JetBrains Mono, monospace", size=10),
        ),
        yaxis=dict(
            showgrid=True, gridcolor="#EEEBE6", gridwidth=1,
            zeroline=False, showspikes=True, spikecolor="#CFB991",
            spikethickness=1, spikedash="dot",
            tickfont=dict(family="JetBrains Mono, monospace", size=10),
        ),
        legend=dict(
            orientation="h", yanchor="top", y=-0.18,
            xanchor="left", x=0,
            font=dict(size=10),
            bgcolor="rgba(255,255,255,0)",
            bordercolor="#E8E5E0", borderwidth=0,
        ),
        hoverlabel=dict(
            bgcolor="rgba(10,10,10,0.94)", font_color="#CFB991",
            font_family="JetBrains Mono, monospace", font_size=11,
            bordercolor="#CFB991",
        ),
        margin=dict(l=48, r=24, t=36, b=80),
    )
)
pio.templates["purdue"] = _PURDUE_TEMPLATE


def _style_fig(fig: go.Figure, height: int = 400) -> go.Figure:
    """Apply Purdue template + rangeselector to a Plotly figure."""
    fig.update_layout(
        template="purdue",
        height=height,
        xaxis=dict(
            rangeselector=dict(
                buttons=[
                    dict(count=1,  label="1M",  step="month", stepmode="backward"),
                    dict(count=3,  label="3M",  step="month", stepmode="backward"),
                    dict(count=6,  label="6M",  step="month", stepmode="backward"),
                    dict(count=1,  label="YTD", step="year",  stepmode="todate"),
                    dict(count=1,  label="1Y",  step="year",  stepmode="backward"),
                    dict(count=3,  label="3Y",  step="year",  stepmode="backward"),
                    dict(count=5,  label="5Y",  step="year",  stepmode="backward"),
                    dict(step="all", label="ALL"),
                ],
                font=dict(size=10),
                bgcolor="#f0ede8",
                activecolor=_GOLD,
            ),
            rangeslider=dict(visible=True, thickness=0.04),
            type="date",
        ),
    )
    return fig


def _chart(fig: go.Figure, **kwargs) -> None:
    """Render a Plotly figure with sensible Streamlit defaults."""
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": True,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            "scrollZoom": True,
        },
        **kwargs,
    )


# ── Event overlays ─────────────────────────────────────────────────────────

def _add_event_bands(
    fig: go.Figure,
    events: list[dict] | None = None,
    show_labels: bool = True,
) -> go.Figure:
    """Add shaded vertical bands + labels for geopolitical events."""
    if events is None:
        events = GEOPOLITICAL_EVENTS
    for ev in events:
        color = ev.get("color", "#CFB991")
        fig.add_vrect(
            x0=str(ev["start"]), x1=str(ev["end"]),
            fillcolor=color, opacity=0.07,
            layer="below", line_width=0,
            annotation_text=ev["label"] if show_labels else "",
            annotation_position="top left",
            annotation_font=dict(size=8, color=color),
            annotation_bgcolor="rgba(255,255,255,0.6)",
        )
    return fig


# ── Text components ────────────────────────────────────────────────────────

def _page_intro(text: str) -> None:
    st.markdown(
        f"""<p style="font-family:'DM Sans',sans-serif;font-size:0.82rem;
        color:#222222;line-height:1.75;max-width:760px;margin-bottom:1.2rem">
        {text}</p>""",
        unsafe_allow_html=True,
    )


def _section_note(text: str) -> None:
    st.markdown(
        f"""<div style="border-left:3px solid {_GOLD};padding:0.6rem 1rem;
        background:#fafaf8;margin:0.8rem 0;font-size:0.78rem;color:#111111;
        line-height:1.7;font-family:'DM Sans',sans-serif">{text}</div>""",
        unsafe_allow_html=True,
    )


def _definition_block(title: str, body: str) -> None:
    st.markdown(
        f"""<div style="border:1px solid #E8E5E0;border-radius:4px;
        overflow:hidden;margin:0.8rem 0">
        <div style="background:{_BLACK};padding:0.5rem 1rem">
          <span style="font-size:0.60rem;font-weight:700;letter-spacing:0.12em;
          text-transform:uppercase;color:{_GOLD}">{title}</span>
        </div>
        <div style="padding:0.7rem 1rem;font-size:0.78rem;color:#111111;
        line-height:1.7;font-family:'DM Sans',sans-serif">{body}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def _takeaway_block(text: str) -> None:
    st.markdown(
        f"""<div style="border-left:3px solid #DAAA00;padding:0.6rem 1rem;
        background:#fffdf5;margin:0.8rem 0;font-size:0.78rem;color:#111111;
        line-height:1.7;font-family:'DM Sans',sans-serif">
        <strong style="color:#8E6F3E;font-size:0.60rem;text-transform:uppercase;
        letter-spacing:0.12em">Key Takeaway</strong><br>{text}</div>""",
        unsafe_allow_html=True,
    )


def _page_conclusion(verdict: str, summary: str) -> None:
    st.markdown(
        f"""<div style="border:1px solid #E8E5E0;border-radius:4px;
        overflow:hidden;margin:1.2rem 0">
        <div style="background:{_BLACK};padding:0.7rem 1.2rem">
          <span style="font-size:0.60rem;font-weight:700;letter-spacing:0.12em;
          text-transform:uppercase;color:{_GOLD}">Assessment · {verdict}</span>
        </div>
        <div style="padding:0.8rem 1.2rem;background:#fafaf8;font-size:0.78rem;
        color:#111111;line-height:1.7;font-family:'DM Sans',sans-serif">
        {summary}</div></div>""",
        unsafe_allow_html=True,
    )


def _metric_card(label: str, value: str, delta: str = "", delta_color: str = "") -> None:
    delta_html = ""
    if delta:
        col = delta_color or ("#2e7d32" if delta.startswith("+") else "#c0392b")
        delta_html = f'<div style="font-size:0.65rem;color:{col};margin-top:2px">{delta}</div>'
    st.markdown(
        f"""<div style="border:1px solid #E8E5E0;border-radius:4px;
        padding:0.7rem 0.9rem;background:#fff;
        transition:box-shadow 0.2s">
        <div style="font-size:0.60rem;font-weight:600;letter-spacing:0.14em;
        text-transform:uppercase;color:#555960;margin-bottom:4px">{label}</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:1.05rem;
        font-weight:700;color:{_BLACK}">{value}</div>
        {delta_html}</div>""",
        unsafe_allow_html=True,
    )


# ── About page styles ───────────────────────────────────────────────────────

def _about_page_styles():
    """Inject CSS for About pages (hero banner, cards, timelines, etc.)."""
    st.markdown("""<style>
    /* ── Hero banner ───────────────────────────────────── */
    .about-hero {
        background: #ffffff;
        border-radius: 12px;
        padding: 0;
        margin-bottom: 1.2rem;
        overflow: hidden;
        border: 1px solid #e8e5e2;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    }
    .about-hero-inner {
        display: flex;
        align-items: stretch;
    }
    .about-hero .hero-photo {
        flex-shrink: 0;
        width: 190px;
        overflow: hidden;
    }
    .about-hero .hero-photo img {
        width: 100%;
        height: 100%;
        object-fit: cover;
        display: block;
    }
    .about-hero .hero-body {
        flex: 1;
        padding: 1.4rem 1.6rem;
        display: flex;
        flex-direction: column;
        justify-content: center;
        border-left: 3px solid #CFB991;
    }
    .about-hero h1 {
        font-family: 'DM Sans', sans-serif;
        font-size: 1.25rem;
        font-weight: 700;
        color: #1a1a1a;
        margin: 0 0 0.15rem 0;
        letter-spacing: -0.02em;
        line-height: 1.15;
    }
    .about-hero .overline {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.56rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.18em;
        color: #8E6F3E;
        margin: 0 0 0.35rem 0;
    }
    .about-hero .subtitle {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.74rem;
        color: #555;
        margin: 0 0 0.45rem 0;
        font-weight: 500;
    }
    .about-hero .tagline {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.70rem;
        color: #777;
        margin: 0 0 0.65rem 0;
        line-height: 1.6;
    }
    .about-hero .links {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
    }
    .about-hero .links a {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.60rem;
        font-weight: 600;
        color: #8E6F3E;
        text-decoration: none;
        padding: 0.22rem 0.7rem;
        border: 1px solid rgba(142,111,62,0.3);
        border-radius: 4px;
        transition: all 0.2s ease;
        letter-spacing: 0.02em;
    }
    .about-hero .links a:hover {
        background: rgba(207,185,145,0.1);
        border-color: #8E6F3E;
    }

    /* ── Cards ─────────────────────────────────────────── */
    .about-card {
        background: #fff;
        border: 1px solid #e8e5e2;
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.03);
        transition: all 0.2s ease;
    }
    .about-card:hover {
        border-color: #CFB991;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    }
    .about-card-title {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.60rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.16em;
        color: #8E6F3E;
        margin: 0 0 0.65rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid #f0eeeb;
    }

    /* ── Experience timeline ───────────────────────────── */
    .exp-item {
        border-left: 2px solid #e8e5e2;
        padding-left: 1rem;
        margin-bottom: 0.75rem;
        padding-bottom: 0.35rem;
        transition: border-color 0.2s ease;
    }
    .exp-item:hover { border-color: #CFB991; }
    .exp-role {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.78rem;
        font-weight: 700;
        color: #1a1a1a;
        margin: 0 0 0.1rem 0;
    }
    .exp-org {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.74rem;
        font-weight: 600;
        color: #8E6F3E;
        margin: 0 0 0.12rem 0;
    }
    .exp-meta {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.65rem;
        color: #888;
        margin: 0 0 0.28rem 0;
    }
    .exp-desc {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.70rem;
        color: #444;
        line-height: 1.6;
        margin: 0;
    }

    /* ── Education ─────────────────────────────────────── */
    .edu-item {
        padding: 0.55rem 0;
        border-bottom: 1px solid #f0eeeb;
    }
    .edu-item:last-child { border-bottom: none; }
    .edu-school {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.78rem;
        font-weight: 700;
        color: #1a1a1a;
        margin: 0 0 0.1rem 0;
    }
    .edu-dept {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.70rem;
        color: #8E6F3E;
        margin: 0 0 0.1rem 0;
        font-weight: 500;
    }
    .edu-degree {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.70rem;
        color: #444;
        margin: 0 0 0.1rem 0;
    }
    .edu-year {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.65rem;
        color: #999;
        margin: 0;
    }

    /* ── Publication ───────────────────────────────────── */
    .pub-item {
        padding: 0.4rem 0;
    }
    .pub-title {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.74rem;
        font-weight: 600;
        color: #1a1a1a;
        margin: 0 0 0.22rem 0;
        line-height: 1.55;
    }
    .pub-authors {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.70rem;
        color: #555;
        margin: 0 0 0.12rem 0;
    }
    .pub-journal {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.65rem;
        color: #8E6F3E;
        font-style: italic;
        margin: 0 0 0.1rem 0;
    }
    .pub-detail {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.65rem;
        color: #999;
        margin: 0 0 0.35rem 0;
    }
    .pub-link {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.65rem;
        font-weight: 600;
        color: #CFB991;
        text-decoration: none;
        border-bottom: 1px solid rgba(207,185,145,0.3);
        transition: border-color 0.2s;
    }
    .pub-link:hover { border-color: #CFB991; }

    /* ── Certifications ────────────────────────────────── */
    .cert-item {
        padding: 0.38rem 0;
        border-bottom: 1px solid #f5f4f2;
    }
    .cert-item:last-child { border-bottom: none; }
    .cert-name {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.74rem;
        font-weight: 600;
        color: #1a1a1a;
        margin: 0 0 0.08rem 0;
    }
    .cert-issuer {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.65rem;
        color: #999;
        margin: 0;
    }

    /* ── Interest tags ─────────────────────────────────── */
    .interest-tag {
        display: inline-block;
        border-radius: 20px;
        padding: 0.22rem 0.65rem;
        font-family: 'DM Sans', sans-serif;
        font-size: 0.60rem;
        font-weight: 600;
        margin: 0.12rem;
    }
    .interest-gold {
        background: rgba(207,185,145,0.12);
        color: #8E6F3E;
        border: 1px solid rgba(207,185,145,0.25);
    }
    .interest-neutral {
        background: #f5f4f2;
        color: #555;
        border: 1px solid #e8e5e2;
    }

    /* ── Acknowledgments ───────────────────────────────── */
    .ack-text {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.70rem;
        color: #555;
        line-height: 1.6;
        margin: 0;
    }
    .ack-text strong { color: #1a1a1a; }

    /* ── Stats row ─────────────────────────────────────── */
    .stat-row {
        display: flex;
        justify-content: space-around;
        padding: 0.55rem 0;
        border-top: 1px solid rgba(207,185,145,0.15);
        margin-top: 0.5rem;
    }
    .stat-item { text-align: center; }
    .stat-num {
        font-size: 1.05rem;
        font-weight: 700;
        color: #CFB991;
        font-family: 'JetBrains Mono', monospace;
        margin: 0;
    }
    .stat-label {
        font-size: 0.56rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #888;
        margin: 2px 0 0 0;
        font-family: 'DM Sans', sans-serif;
    }
    </style>""", unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────

@st.cache_data(ttl=86400)
def _footer_logo_b64() -> str:
    p = _ASSETS / "purdue_daniels_logo_reverse.png"
    if p.exists():
        return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()
    return ""


def _page_footer() -> None:
    from datetime import datetime
    yr = datetime.now().year
    ts = datetime.now().strftime("%B %d, %Y at %H:%M UTC")
    logo_src = _footer_logo_b64()
    _w = ("color:rgba(255,255,255,0.75);text-decoration:none;"
          "font-size:var(--fs-base);font-weight:500;transition:color 0.15s ease;")
    _g = ("font-size:var(--fs-xs);font-weight:700;text-transform:uppercase;"
          "letter-spacing:var(--ls-widest);color:#CFB991;margin:0 0 14px 0;"
          "padding-bottom:8px;border-bottom:1px solid rgba(207,185,145,0.15);")
    logo_html = (
        f"<img src='{logo_src}' alt='Purdue Daniels School of Business' "
        "style='height:40px;margin-bottom:16px;display:block;' />"
        if logo_src else
        "<span style='color:#CFB991;font-weight:800;font-size:1rem'>Purdue Daniels</span>"
    )
    st.markdown(
        "<style>"
        ".main .block-container { padding-bottom: 0 !important; margin-bottom: 0 !important; }"
        ".main { padding-bottom: 0 !important; margin-bottom: 0 !important; }"
        "[data-testid='stAppViewContainer'] { padding-bottom: 0 !important; }"
        "[data-testid='stBottom'] { display: none !important; }"
        "</style>"
        "<div style='margin-top:4rem;font-family:var(--font-sans);"
        "position:relative;width:100vw;left:50%;right:50%;margin-left:-50vw;margin-right:-50vw;"
        "margin-bottom:-10rem;padding-bottom:0;'>"
        "<div style='background:#000000;padding:44px 0 40px 0;'>"
        "<div style='display:grid;grid-template-columns:1.6fr 1fr 1fr 1fr 1fr;"
        "gap:28px;max-width:1280px;margin:0 auto;padding:0 48px;'>"
        # Col 1 - brand
        f"<div><a href='https://business.purdue.edu/' target='_blank'>{logo_html}</a>"
        "<p style='font-size:var(--fs-base);color:rgba(255,255,255,0.7);line-height:1.65;"
        "margin:0 0 16px 0;max-width:260px;'>"
        "MGMT 69000-120 &middot; AI for Finance<br/>West Lafayette, Indiana</p>"
        f"<p style='font-size:var(--fs-xs);color:rgba(207,185,145,0.6);margin:0;"
        f"font-weight:600;letter-spacing:var(--ls-wide);'>Last updated {ts}</p>"
        "</div>"
        # Col 2 - navigate
        f"<div><p style='{_g}'>Navigate</p>"
        "<ul style='list-style:none;padding:0;margin:0;'>"
        f"<li style='margin-bottom:8px;'><a href='#' style='{_w}'>Overview</a></li>"
        f"<li style='margin-bottom:8px;'><a href='#' style='{_w}'>Geopolitical Triggers</a></li>"
        f"<li style='margin-bottom:8px;'><a href='#' style='{_w}'>Correlation Analysis</a></li>"
        f"<li style='margin-bottom:8px;'><a href='#' style='{_w}'>Spillover Analytics</a></li>"
        f"<li style='margin-bottom:8px;'><a href='#' style='{_w}'>Commodities to Watch</a></li>"
        f"<li style='margin-bottom:8px;'><a href='#' style='{_w}'>Trade Ideas</a></li>"
        "</ul></div>"
        # Col 3 - about
        f"<div><p style='{_g}'>About</p>"
        "<ul style='list-style:none;padding:0;margin:0;'>"
        f"<li style='margin-bottom:8px;'><a href='?page=about_heramb' style='{_w}'>Heramb S. Patkar</a></li>"
        f"<li style='margin-bottom:8px;'><a href='?page=about_jiahe' style='{_w}'>Jiahe Miao</a></li>"
        f"<li style='margin-bottom:8px;'><a href='?page=about_ilian' style='{_w}'>Ilian Zalomai</a></li>"
        f"<li><a href='https://business.purdue.edu/' target='_blank' style='{_w}'>Daniels School of Business</a></li>"
        "</ul></div>"
        # Col 4 - connect
        f"<div><p style='{_g}'>Connect</p>"
        "<ul style='list-style:none;padding:0;margin:0;'>"
        f"<li style='margin-bottom:8px;'><a href='https://www.linkedin.com/in/heramb-patkar/' target='_blank' style='{_w}'>LinkedIn</a></li>"
        f"<li><a href='https://github.com/HPATKAR' target='_blank' style='{_w}'>GitHub</a></li>"
        "</ul></div>"
        # Col 5 - data sources
        f"<div><p style='{_g}'>Data Sources</p>"
        "<ul style='list-style:none;padding:0;margin:0;'>"
        f"<li style='margin-bottom:8px;'><a href='https://finance.yahoo.com' target='_blank' style='{_w}'>Yahoo Finance</a></li>"
        f"<li style='margin-bottom:8px;'><a href='https://fred.stlouisfed.org' target='_blank' style='{_w}'>FRED · Federal Reserve</a></li>"
        f"<li><a href='https://financialdatasets.ai' target='_blank' style='{_w}'>FinancialDatasets</a></li>"
        "</ul></div>"
        "</div></div>"
        "<div style='background:#CFB991;padding:10px 48px;text-align:center;'>"
        f"<p style='font-size:var(--fs-tiny);color:#000000;margin:0;font-weight:600;"
        f"letter-spacing:var(--ls-wide);font-family:var(--font-sans);'>"
        f"&copy; {yr} Purdue University &middot; For educational purposes only "
        f"&middot; Not investment advice</p>"
        "</div></div>",
        unsafe_allow_html=True,
    )
