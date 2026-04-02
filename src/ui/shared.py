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
            rangeslider=dict(visible=False),
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


_EQUITY_REGIONS  = {"S&P 500","DAX","Nikkei 225","FTSE 100","Eurostoxx 50","Shanghai Comp",
                    "Sensex","Hang Seng","KOSPI","Bovespa","ASX 200","CAC 40","Nifty 50"}
_COMMODITY_NAMES = {"WTI Crude Oil","Brent Crude","Gold","Silver","Copper","Natural Gas",
                    "Wheat","Corn","Soybeans","Platinum","Palladium","Sugar","Coffee",
                    "Cotton","Crude Oil"}

# Four clearly distinct dash styles - rotate within equities and commodities separately
_EQ_DASHES  = ["solid", "dash", "dot", "dashdot"]
_CMD_DASHES = ["longdash", "longdashdot", "solid", "dash"]


def _line_style(name: str, eq_idx: int, cmd_idx: int) -> dict:
    """
    Return a Plotly line dict with a distinguishable color + dash for multi-series charts.
    Equities get solid/dashed lines at width 1.9; commodities get different dash patterns at 1.6.
    Caller tracks eq_idx and cmd_idx separately and increments the appropriate counter.
    """
    from src.data.config import PALETTE
    if name in _EQUITY_REGIONS:
        color = PALETTE[eq_idx % len(PALETTE)]
        dash  = _EQ_DASHES[eq_idx % len(_EQ_DASHES)]
        return dict(color=color, width=1.9, dash=dash)
    else:
        # Offset commodity colors so they don't clash with the first equity colors
        color = PALETTE[(cmd_idx + 1) % len(PALETTE)]
        dash  = _CMD_DASHES[cmd_idx % len(_CMD_DASHES)]
        return dict(color=color, width=1.5, dash=dash)


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
            rangeslider=dict(visible=False),
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
            annotation_bgcolor="rgba(10,12,20,0.65)",
        )
    return fig


# ── Text components ────────────────────────────────────────────────────────

def _page_intro(text: str) -> None:
    st.markdown(
        f"""<p style="font-family:'DM Sans',sans-serif;font-size:0.76rem;
        color:#b8b8b8;line-height:1.75;max-width:860px;margin:0 0 1.0rem;
        padding-left:0.85rem;border-left:3px solid #2a2a2a">{text}</p>""",
        unsafe_allow_html=True,
    )


def _section_note(text: str) -> None:
    st.markdown(
        f"""<div style="border-left:3px solid {_GOLD};padding:0.5rem 0.9rem;
        background:#fafaf8;margin:0.6rem 0 0.9rem;font-size:0.70rem;color:#333333;
        line-height:1.7;font-family:'DM Sans',sans-serif">
        <span style="font-size:0.52rem;font-weight:700;letter-spacing:0.16em;
        text-transform:uppercase;color:#8E6F3E;display:block;margin-bottom:2px">
        Methodology</span>{text}</div>""",
        unsafe_allow_html=True,
    )


def _definition_block(title: str, body: str) -> None:
    st.markdown(
        f"""<div style="border:1px solid #E8E5E0;border-radius:3px;
        overflow:hidden;margin:0.6rem 0 0.9rem">
        <div style="background:{_BLACK};padding:0.45rem 1rem;display:flex;align-items:center;gap:0.5rem">
          <div style="width:3px;height:14px;background:{_GOLD};border-radius:2px;flex-shrink:0"></div>
          <span style="font-size:0.58rem;font-weight:700;letter-spacing:0.14em;
          text-transform:uppercase;color:{_GOLD}">{title}</span>
        </div>
        <div style="padding:0.65rem 1rem;font-size:0.70rem;color:#c8c8c8;
        background:#1c1c1c;line-height:1.75;font-family:'DM Sans',sans-serif">{body}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def _takeaway_block(text: str) -> None:
    st.markdown(
        f"""<div style="border:1px solid #E8E5E0;border-top:2px solid {_GOLD};
        border-radius:0 0 3px 3px;padding:0.55rem 1rem;
        background:#fffdf5;margin:0.6rem 0 0.9rem;font-size:0.70rem;color:#333333;
        line-height:1.75;font-family:'DM Sans',sans-serif">
        <span style="font-size:0.52rem;font-weight:700;letter-spacing:0.16em;
        text-transform:uppercase;color:#8E6F3E;display:block;margin-bottom:3px">
        Key Insight</span>{text}</div>""",
        unsafe_allow_html=True,
    )


def _page_conclusion(verdict: str, summary: str) -> None:
    st.markdown(
        f"""<div style="border:1px solid #E8E5E0;border-top:3px solid {_GOLD};
        border-radius:0 0 3px 3px;overflow:hidden;margin:1.2rem 0">
        <div style="background:{_BLACK};padding:0.6rem 1.2rem;display:flex;align-items:center;gap:0.6rem">
          <div style="width:3px;height:16px;background:{_GOLD};border-radius:2px;flex-shrink:0"></div>
          <span style="font-size:0.58rem;font-weight:700;letter-spacing:0.14em;
          text-transform:uppercase;color:{_GOLD}">Assessment &middot; {verdict}</span>
        </div>
        <div style="padding:0.75rem 1.2rem;background:#fafaf8;font-size:0.70rem;
        color:#333333;line-height:1.75;font-family:'DM Sans',sans-serif">
        {summary}</div></div>""",
        unsafe_allow_html=True,
    )


def _freshness_badge(
    label: str,
    value: str,
    age_seconds: int | None = None,
    stale_after: int = 900,
) -> str:
    """
    Return an HTML freshness badge.
    Green dot  = fresh (age < stale_after)
    Amber dot  = aging (stale_after <= age < 2x)
    Red dot    = stale (age >= 2x stale_after) or unknown
    """
    if age_seconds is None:
        dot_color, status = "#555960", "—"
    elif age_seconds < stale_after:
        dot_color, status = "#27ae60", "Live"
    elif age_seconds < stale_after * 2:
        dot_color, status = "#e67e22", "Aging"
    else:
        dot_color, status = "#c0392b", "Stale"

    return (
        f'<span style="display:inline-flex;align-items:center;gap:4px;'
        f'font-family:\'JetBrains Mono\',monospace;font-size:0.58rem;'
        f'color:#8890a1;margin-right:0.9rem">'
        f'<span style="color:{dot_color};font-size:0.55rem">&#9679;</span>'
        f'<span style="color:#c8c8c8">{label}</span>'
        f'<span style="color:{_GOLD}">{value}</span>'
        f'<span style="color:{dot_color};font-size:0.50rem">{status}</span>'
        f'</span>'
    )


def _data_status_bar(items: list[tuple[str, str, int | None]]) -> None:
    """
    Render a compact horizontal data-freshness strip.
    items: list of (label, value, age_seconds)
    e.g. [("VIX", "18.4", 120), ("OVX", "31.2", 300), ("Regime", "Normal", None)]
    """
    badges = "".join(_freshness_badge(l, v, a) for l, v, a in items)
    st.markdown(
        f'<div style="display:flex;flex-wrap:wrap;align-items:center;'
        f'background:#0d0d0d;border:1px solid #1e1e1e;border-radius:3px;'
        f'padding:0.35rem 0.8rem;margin:0.4rem 0 0.9rem;gap:0.1rem">'
        f'<span style="font-size:0.50rem;font-weight:700;letter-spacing:0.16em;'
        f'text-transform:uppercase;color:#3a3a3a;margin-right:0.7rem">DATA</span>'
        f'{badges}</div>',
        unsafe_allow_html=True,
    )


def _thread(text: str) -> None:
    """Narrative connector paragraph between page sections."""
    st.markdown(
        f'<p style="font-family:\'DM Sans\',sans-serif;font-size:0.69rem;'
        f'color:#8890a1;line-height:1.7;margin:0.1rem 0 0.55rem;font-style:italic">'
        f'{text}</p>',
        unsafe_allow_html=True,
    )


def _insight_note(text: str) -> None:
    """Compact formal plain-English explanation rendered beneath each infographic."""
    st.markdown(
        f"""<div style="border-left:2px solid {_GOLD};padding:0.38rem 0.85rem;
        background:#fafaf8;margin:0.1rem 0 0.75rem;font-size:0.66rem;color:#444444;
        line-height:1.65;font-family:'DM Sans',sans-serif">
        <span style="font-size:0.52rem;font-weight:700;letter-spacing:0.16em;
        text-transform:uppercase;color:#8E6F3E;display:block;margin-bottom:2px">
        Interpretation</span>{text}</div>""",
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


def _section_header(number: str, title: str, subtitle: str = "") -> None:
    """Numbered section header - establishes reading sequence within a page."""
    sub_html = (
        f'<span style="font-family:\'DM Sans\',sans-serif;font-size:0.64rem;'
        f'color:#888;font-weight:400;margin-left:0.6rem;font-style:italic">{subtitle}</span>'
        if subtitle else ""
    )
    st.markdown(
        f'<div style="margin:1.1rem 0 0.45rem;padding-bottom:0.35rem;'
        f'border-bottom:1.5px solid #E8E5E0;display:flex;align-items:baseline;gap:0">'
        f'<span style="font-family:\'DM Sans\',sans-serif;font-size:0.56rem;font-weight:700;'
        f'text-transform:uppercase;letter-spacing:0.18em;color:#CFB991;'
        f'background:#000;padding:2px 6px;border-radius:2px;margin-right:0.55rem">{number}</span>'
        f'<span style="font-family:\'DM Sans\',sans-serif;font-size:0.76rem;font-weight:700;'
        f'color:#000">{title}</span>'
        f'{sub_html}</div>',
        unsafe_allow_html=True,
    )


def _regime_banner(label: str, sub: str = "", color: str = "#8E6F3E") -> None:
    """Full-width coloured regime banner - place at the very top of a page to surface the verdict."""
    sub_html = (
        f'<span style="font-family:\'DM Sans\',sans-serif;font-size:0.70rem;'
        f'color:rgba(255,255,255,0.72);margin-left:0.75rem">{sub}</span>'
        if sub else ""
    )
    st.markdown(
        f'<div style="background:{color};padding:0.55rem 1.2rem;border-radius:4px;'
        f'display:flex;align-items:center;gap:0.75rem;margin-bottom:0.9rem">'
        f'<div style="width:7px;height:7px;border-radius:50%;'
        f'background:rgba(255,255,255,0.55);flex-shrink:0"></div>'
        f'<div>'
        f'<span style="font-family:\'DM Sans\',sans-serif;font-size:0.56rem;font-weight:700;'
        f'letter-spacing:0.18em;text-transform:uppercase;color:rgba(255,255,255,0.60)">'
        f'MACRO REGIME &nbsp;·&nbsp; </span>'
        f'<span style="font-family:\'DM Sans\',sans-serif;font-size:0.88rem;'
        f'font-weight:700;color:#fff">{label}</span>'
        f'{sub_html}'
        f'</div></div>',
        unsafe_allow_html=True,
    )


def _narrative_box(text: str) -> None:
    """Pre-loaded data-driven narrative with gold left border. Paragraphs separated by blank lines."""
    paragraphs = [p.strip() for p in text.strip().split("\n\n") if p.strip()]
    html_parts = []
    for p in paragraphs:
        if ":" in p[:60]:
            head, _, rest = p.partition(":")
            html_parts.append(f"<b>{head}:</b>{rest}")
        else:
            html_parts.append(p)
    html = "<br><br>".join(html_parts)
    st.markdown(
        f'<div style="border-left:4px solid #CFB991;padding:0.8rem 1.1rem;'
        f'background:#fafaf8;border-radius:0 4px 4px 0;margin:0.35rem 0 0.75rem">'
        f'<div style="font-family:\'DM Sans\',sans-serif;font-size:0.73rem;'
        f'color:#2A2A2A;line-height:1.80">{html}</div></div>',
        unsafe_allow_html=True,
    )


def _primary_chart(fig, caption: str = "") -> None:
    """Render the headline chart for a page/section with a gold left-border emphasis strip."""
    st.markdown(
        '<div style="border-left:3px solid #CFB991;padding-left:0.4rem;margin-bottom:0.1rem">',
        unsafe_allow_html=True,
    )
    _chart(fig)
    st.markdown('</div>', unsafe_allow_html=True)
    if caption:
        st.markdown(
            f'<p style="font-family:\'DM Sans\',sans-serif;font-size:0.62rem;'
            f'color:#888;font-style:italic;margin:0 0 0.55rem 0.5rem">{caption}</p>',
            unsafe_allow_html=True,
        )


# ── About page styles ───────────────────────────────────────────────────────

def _about_page_styles():
    """Inject CSS for About pages (hero banner, cards, timelines, etc.)."""
    st.markdown("""<style>
    /* ── Hero banner ───────────────────────────────────── */
    .about-hero {
        background: #1c1c1c;
        border-radius: 12px;
        padding: 0;
        margin-bottom: 1.2rem;
        overflow: hidden;
        border: 1px solid #2a2a2a;
        box-shadow: 0 1px 8px rgba(0,0,0,0.25);
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
        color: #CFB991;
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
        color: #c4ae88;
        margin: 0 0 0.35rem 0;
    }
    .about-hero .subtitle {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.74rem;
        color: #8890a1;
        margin: 0 0 0.45rem 0;
        font-weight: 500;
    }
    .about-hero .tagline {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.70rem;
        color: #8890a1;
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
        color: #CFB991;
        text-decoration: none;
        padding: 0.22rem 0.7rem;
        border: 1px solid rgba(207,185,145,0.3);
        border-radius: 4px;
        transition: all 0.2s ease;
        letter-spacing: 0.02em;
    }
    .about-hero .links a:hover {
        background: rgba(207,185,145,0.15);
        border-color: #CFB991;
    }

    /* ── Cards ─────────────────────────────────────────── */
    .about-card {
        background: #1c1c1c;
        border: 1px solid #2a2a2a;
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 1px 6px rgba(0,0,0,0.18);
        transition: all 0.2s ease;
    }
    .about-card:hover {
        border-color: #CFB991;
        box-shadow: 0 4px 14px rgba(0,0,0,0.30);
    }
    .about-card-title {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.60rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.16em;
        color: #CFB991;
        margin: 0 0 0.65rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid #2a2a2a;
    }

    /* ── Experience timeline ───────────────────────────── */
    .exp-item {
        border-left: 2px solid #2a2a2a;
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
        color: #e8e9ed;
        margin: 0 0 0.1rem 0;
    }
    .exp-org {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.74rem;
        font-weight: 600;
        color: #CFB991;
        margin: 0 0 0.12rem 0;
    }
    .exp-meta {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.65rem;
        color: #8890a1;
        margin: 0 0 0.28rem 0;
    }
    .exp-desc {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.70rem;
        color: #8890a1;
        line-height: 1.6;
        margin: 0;
    }

    /* ── Education ─────────────────────────────────────── */
    .edu-item {
        padding: 0.55rem 0;
        border-bottom: 1px solid #2a2a2a;
    }
    .edu-item:last-child { border-bottom: none; }
    .edu-school {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.78rem;
        font-weight: 700;
        color: #e8e9ed;
        margin: 0 0 0.1rem 0;
    }
    .edu-dept {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.70rem;
        color: #CFB991;
        margin: 0 0 0.1rem 0;
        font-weight: 500;
    }
    .edu-degree {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.70rem;
        color: #8890a1;
        margin: 0 0 0.1rem 0;
    }
    .edu-year {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.65rem;
        color: #8890a1;
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
        color: #e8e9ed;
        margin: 0 0 0.22rem 0;
        line-height: 1.55;
    }
    .pub-authors {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.70rem;
        color: #8890a1;
        margin: 0 0 0.12rem 0;
    }
    .pub-journal {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.65rem;
        color: #c4ae88;
        font-style: italic;
        margin: 0 0 0.1rem 0;
    }
    .pub-detail {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.65rem;
        color: #8890a1;
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
        border-bottom: 1px solid #2a2a2a;
    }
    .cert-item:last-child { border-bottom: none; }
    .cert-name {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.74rem;
        font-weight: 600;
        color: #e8e9ed;
        margin: 0 0 0.08rem 0;
    }
    .cert-issuer {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.65rem;
        color: #8890a1;
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
        background: rgba(207,185,145,0.15);
        color: #CFB991;
        border: 1px solid rgba(207,185,145,0.3);
    }
    .interest-neutral {
        background: rgba(207,185,145,0.08);
        color: #c4ae88;
        border: 1px solid rgba(207,185,145,0.18);
    }

    /* ── Acknowledgments ───────────────────────────────── */
    .ack-text {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.70rem;
        color: #8890a1;
        line-height: 1.6;
        margin: 0;
    }
    .ack-text strong { color: #e8e9ed; }

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
        color: #8890a1;
        margin: 2px 0 0 0;
        font-family: 'DM Sans', sans-serif;
    }
    </style>""", unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────

@st.cache_data(ttl=86400)
def _footer_logo_b64() -> str:
    # Prefer the project logo; fall back to Purdue reverse logo
    for name in ("logo.png", "purdue_daniels_logo_reverse.png"):
        p = _ASSETS / name
        if p.exists():
            ext = p.suffix.lstrip(".")
            return f"data:image/{ext};base64," + base64.b64encode(p.read_bytes()).decode()
    return ""


def _page_footer() -> None:
    import streamlit.components.v1 as _comp
    from datetime import datetime
    yr  = datetime.now().year
    ts  = datetime.now().strftime("%B %d, %Y at %H:%M UTC")
    _logo_src = _footer_logo_b64()
    if _logo_src:
        logo_html = (
            f"<img src='{_logo_src}' alt='Cross-Asset Spillover Monitor' "
            "style='width:180px;height:auto;object-fit:contain;"
            "margin-bottom:12px;display:block;' />"
        )
    else:
        logo_html = (
            "<div style='display:inline-flex;flex-direction:column;align-items:center;justify-content:center;"
            "width:72px;height:72px;background:#CFB991;border-radius:9px;margin-bottom:18px;gap:2px;'>"
            "<span style='font-size:1.15rem;font-weight:800;color:#000;line-height:1;letter-spacing:0.02em;'>X</span>"
            "<span style='font-size:0.58rem;font-weight:600;color:#000;line-height:1;letter-spacing:0.14em;opacity:0.75;'>ASSET</span>"
            "</div>"
        )

    _ft_click = _comp.html(f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
html,body{{
  background:#000;width:100%;
  font-family:'DM Sans',-apple-system,BlinkMacSystemFont,sans-serif;
  -webkit-font-smoothing:antialiased;
  overflow:visible;
}}
.ft-body{{background:#0a0a0a;padding:44px 0 40px;border-top:1px solid #1e1e1e;}}
.ft-grid{{
  display:grid;grid-template-columns:1.6fr 1fr 1fr 1fr 1fr;
  gap:28px;width:100%;padding:0 48px;
}}
.ft-logo{{height:38px;margin-bottom:16px;display:block;}}
.ft-wordmark{{
  color:#CFB991;font-weight:800;font-size:1rem;
  display:block;margin-bottom:16px;text-decoration:none;
}}
.ft-desc{{
  font-size:0.74rem;color:rgba(255,255,255,0.68);
  line-height:1.65;margin:0 0 14px;max-width:260px;
}}
.ft-ts{{
  font-size:0.58rem;color:rgba(207,185,145,0.55);
  margin:0;font-weight:600;letter-spacing:0.10em;
}}
.ft-hd{{
  font-size:0.58rem;font-weight:700;text-transform:uppercase;
  letter-spacing:0.16em;color:#CFB991;
  margin:0 0 13px;padding-bottom:8px;
  border-bottom:1px solid rgba(207,185,145,0.15);
}}
ul{{list-style:none;padding:0;margin:0;}}
li{{margin-bottom:8px;}}
li:last-child{{margin-bottom:0;}}
a{{
  color:rgba(255,255,255,0.72);
  text-decoration:none;
  font-size:0.73rem;font-weight:400;
  transition:color 0.14s;
  display:inline-block;
}}
a:hover{{color:#CFB991;}}
.ft-bar{{
  background:#0a0a0a;padding:10px 48px;text-align:center;
  border-top:1px solid rgba(207,185,145,0.25);
}}
.ft-bar p{{
  font-size:0.56rem;color:rgba(207,185,145,0.55);margin:0;
  font-weight:500;letter-spacing:0.12em;
}}
</style>
</head>
<body>
<script>
(function(){{
  var f = window.frameElement;
  if (!f) return;

  /* ── 1. Full-width: break out of Streamlit block-container ── */
  f.style.width          = '100vw';
  f.style.position       = 'relative';
  f.style.left           = '50%';
  f.style.transform      = 'translateX(-50%)';
  f.style.maxWidth       = '100vw';
  f.style.border         = 'none';
  f.style.display        = 'block';
  f.style.marginLeft     = '0';
  f.style.marginRight    = '0';
  f.style.overflow       = 'hidden';

  /* ── 2. Walk every ancestor: remove overflow clip + bottom spacing ── */
  var el = f.parentElement;
  while (el && el !== window.parent.document.body) {{
    el.style.overflow      = 'visible';
    el.style.maxWidth      = 'none';
    el.style.paddingBottom = '0';
    el.style.marginBottom  = '0';
    el = el.parentElement;
  }}

  /* ── 3. Kill Streamlit's injected bottom chrome via parent CSS ── */
  var pid = 'ec-ft-css';
  if (!window.parent.document.getElementById(pid)) {{
    var s = window.parent.document.createElement('style');
    s.id = pid;
    s.textContent =
      '.main .block-container{{padding-bottom:0!important;margin-bottom:0!important}}' +
      '[data-testid="stBottom"]{{display:none!important}}' +
      '.stDeployButton{{display:none!important}}';
    window.parent.document.head.appendChild(s);
  }}

  /* ── 4. Auto-size iframe height to exact content height ── */
  function measure() {{
    var h = document.documentElement.scrollHeight || document.body.scrollHeight;
    if (h > 20) f.style.height = h + 'px';
  }}
  /* Run immediately, after fonts, and with two fallback retries */
  measure();
  if (document.fonts && document.fonts.ready) {{
    document.fonts.ready.then(function() {{ measure(); setTimeout(measure, 150); }});
  }} else {{
    setTimeout(measure, 400);
  }}
}})();

function nav(page) {{
  window.parent.location.href = '?page=' + page;
}}
</script>
<div class="ft-body">
  <div class="ft-grid">

    <div>
      {logo_html}
      <p class="ft-desc">MGMT 69000-120 &middot; AI for Finance<br/>West Lafayette, Indiana</p>
      <p class="ft-ts">Last updated {ts}</p>
    </div>

    <div>
      <p class="ft-hd">Navigate</p>
      <ul>
        <li><a href="#" onclick="nav('overview');return false;">Overview</a></li>
        <li><a href="#" onclick="nav('war_impact_map');return false;">War Impact Map</a></li>
        <li><a href="#" onclick="nav('geopolitical');return false;">Geopolitical Triggers</a></li>
        <li><a href="#" onclick="nav('correlation');return false;">Correlation Analysis</a></li>
        <li><a href="#" onclick="nav('spillover');return false;">Spillover Analytics</a></li>
        <li><a href="#" onclick="nav('watchlist');return false;">Commodities to Watch</a></li>
      </ul>
    </div>

    <div>
      <p class="ft-hd">Strategy &amp; Research</p>
      <ul>
        <li><a href="#" onclick="nav('trade_ideas');return false;">Trade Ideas</a></li>
        <li><a href="#" onclick="nav('stress_test');return false;">Portfolio Stress Test</a></li>
        <li><a href="#" onclick="nav('model_accuracy');return false;">Performance Review</a></li>
        <li><a href="#" onclick="nav('ai_chat');return false;">AI Analyst</a></li>
      </ul>
    </div>

    <div>
      <p class="ft-hd">Authors</p>
      <ul>
        <li><a href="#" onclick="nav('about_heramb');return false;">Heramb S. Patkar</a></li>
        <li><a href="#" onclick="nav('about_jiahe');return false;">Jiahe Miao</a></li>
        <li><a href="#" onclick="nav('about_ilian');return false;">Ilian Zalomai</a></li>
        <li><a href="https://business.purdue.edu/" target="_blank">Daniels School of Business</a></li>
      </ul>
    </div>

    <div>
      <p class="ft-hd">Data Sources</p>
      <ul>
        <li><a href="https://finance.yahoo.com" target="_blank">Yahoo Finance</a></li>
        <li><a href="https://fred.stlouisfed.org" target="_blank">FRED &middot; Federal Reserve</a></li>
        <li><a href="https://financialdatasets.ai" target="_blank">FinancialDatasets</a></li>
        <li><a href="https://www.cftc.gov" target="_blank">CFTC COT Reports</a></li>
      </ul>
    </div>

  </div>
</div>
<div class="ft-bar">
  <p>&copy; {yr} Cross-Asset Spillover Monitor &middot; Purdue Daniels &middot; MGMT 69000-120 &middot; For educational purposes only &middot; Not investment advice</p>
</div>
</body></html>""", height=380, scrolling=False)
    _VALID = {'overview','war_impact_map','geopolitical','correlation','spillover',
              'watchlist','trade_ideas','stress_test','model_accuracy','ai_chat',
              'about_heramb','about_jiahe','about_ilian'}
    _ft_last = st.session_state.get("_ft_nav_last", "")
    if _ft_click and _ft_click in _VALID and _ft_click != _ft_last:
        st.session_state["_ft_nav_last"] = _ft_click
        st.session_state["current_page"] = _ft_click
        st.rerun()
