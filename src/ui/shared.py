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

_BG       = "#0d1219"    # page background — deep navy terminal
_BG_WARM  = "#111923"    # chart plot area — slightly lifted navy
_GRID     = "#162030"    # grid lines — navy-tinted, structural only
_TICK     = "#555960"    # axis tick labels — muted
_LEGEND   = "#8890a1"    # legend text

_PURDUE_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        font=dict(family="DM Sans, Inter, sans-serif", color="#e8e9ed", size=12),
        paper_bgcolor=_BG,
        plot_bgcolor=_BG_WARM,
        colorway=PALETTE,
        xaxis=dict(
            showgrid=True, gridcolor=_GRID, gridwidth=1,
            zeroline=False,
            showspikes=True, spikecolor="#CFB991",
            spikethickness=1, spikedash="dot",
            tickfont=dict(family="JetBrains Mono, monospace", size=10, color=_TICK),
            linecolor=_GRID, linewidth=1,
            rangeslider=dict(visible=False),
        ),
        yaxis=dict(
            showgrid=True, gridcolor=_GRID, gridwidth=1,
            zeroline=False,
            showspikes=True, spikecolor="#CFB991",
            spikethickness=1, spikedash="dot",
            tickfont=dict(family="JetBrains Mono, monospace", size=10, color=_TICK),
            linecolor=_GRID, linewidth=1,
        ),
        legend=dict(
            orientation="h", yanchor="top", y=-0.18,
            xanchor="left", x=0,
            font=dict(size=10, color=_LEGEND),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
        ),
        hoverlabel=dict(
            bgcolor="rgba(8,8,8,0.96)",
            font_color="#CFB991",
            font_family="JetBrains Mono, monospace",
            font_size=11,
            bordercolor="#2a2a2a",
        ),
        title=dict(
            font=dict(family="DM Sans, sans-serif", size=13, color="#c8c8c8"),
            x=0.0, xanchor="left", pad=dict(l=4, t=4),
        ),
        margin=dict(l=48, r=24, t=36, b=80),
        modebar=dict(
            bgcolor="rgba(0,0,0,0)",
            color="#555960",
            activecolor="#CFB991",
        ),
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
                font=dict(size=10, color="#8890a1"),
                bgcolor="#1a1a1a",
                activecolor=_GOLD,
                bordercolor="#2a2a2a",
                borderwidth=1,
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
        width="stretch",
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
        color:#b8b8b8;line-height:1.75;max-width:860px;margin:0 0 1.0rem;">{text}</p>""",
        unsafe_allow_html=True,
    )


def _section_note(text: str) -> None:
    st.markdown(
        f"""<div style="border-top:1px solid #2a2a2a;border-bottom:1px solid #2a2a2a;
        padding:0.45rem 0;margin:0.6rem 0 0.9rem;font-size:0.70rem;color:#8890a1;
        line-height:1.7;font-family:'DM Sans',sans-serif">
        <span style="font-size:0.52rem;font-weight:700;letter-spacing:0.16em;
        text-transform:uppercase;color:{_GOLD};display:block;margin-bottom:2px">
        Methodology</span>{text}</div>""",
        unsafe_allow_html=True,
    )


def _definition_block(title: str, body: str) -> None:
    st.markdown(
        f"""<div style="border:1px solid #1e1e1e;border-left:2px solid {_GOLD};
        margin:0.6rem 0 0.9rem;background:#131313">
        <div style="padding:0.38rem 0.9rem;border-bottom:1px solid #1a1a1a">
          <span style="font-size:0.52rem;font-weight:700;letter-spacing:0.18em;
          text-transform:uppercase;color:{_GOLD}">{title}</span>
        </div>
        <div style="padding:0.65rem 0.9rem;font-size:0.70rem;color:#b8b8b8;
        line-height:1.80;font-family:'DM Sans',sans-serif">{body}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def _takeaway_block(text: str) -> None:
    st.markdown(
        f"""<div style="border-top:2px solid {_GOLD};border-bottom:1px solid #2a2a2a;
        padding:0.5rem 0;margin:0.6rem 0 0.9rem;font-size:0.70rem;color:#c8c8c8;
        line-height:1.75;font-family:'DM Sans',sans-serif">
        <span style="font-size:0.52rem;font-weight:700;letter-spacing:0.16em;
        text-transform:uppercase;color:{_GOLD};display:block;margin-bottom:3px">
        Key Insight</span>{text}</div>""",
        unsafe_allow_html=True,
    )


def _page_conclusion(verdict: str, summary: str) -> None:
    st.markdown(
        f"""<div style="border-top:2px solid {_GOLD};border-bottom:1px solid #2a2a2a;
        margin:1.2rem 0;padding:0.6rem 0 0.75rem 0">
        <div style="margin-bottom:0.35rem">
          <span style="font-size:0.56rem;font-weight:700;letter-spacing:0.16em;
          text-transform:uppercase;color:{_GOLD}">Assessment</span>
          <span style="font-size:0.56rem;font-weight:700;letter-spacing:0.16em;
          text-transform:uppercase;color:#888">&nbsp;&middot;&nbsp;{verdict}</span>
        </div>
        <div style="font-size:0.70rem;color:#c8c8c8;line-height:1.75;
        font-family:'DM Sans',sans-serif">{summary}</div></div>""",
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
        dot_color, status = "#555960", "-"
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
        f'background:#0d0d0d;border-top:1px solid #1e1e1e;border-bottom:1px solid #1e1e1e;'
        f'padding:0.35rem 0.8rem;margin:0.4rem 0 0.9rem;gap:0.1rem">'
        f'<span style="font-size:0.50rem;font-weight:700;letter-spacing:0.16em;'
        f'text-transform:uppercase;color:#555960;margin-right:0.7rem">DATA</span>'
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
        f"""<div style="border-top:1px solid #2a2a2a;padding:0.38rem 0;
        margin:0.1rem 0 0.75rem;font-size:0.66rem;color:#8890a1;
        line-height:1.65;font-family:'DM Sans',sans-serif">
        <span style="font-size:0.52rem;font-weight:700;letter-spacing:0.16em;
        text-transform:uppercase;color:{_GOLD};display:block;margin-bottom:2px">
        Interpretation</span>{text}</div>""",
        unsafe_allow_html=True,
    )


def _metric_card(label: str, value: str, delta: str = "", delta_color: str = "") -> None:
    delta_html = ""
    if delta:
        col = delta_color or ("#27ae60" if delta.startswith("+") else "#c0392b")
        delta_html = (
            f'<div style="font-family:\'JetBrains Mono\',monospace;'
            f'font-size:0.60rem;color:{col};margin-top:3px;letter-spacing:0.02em">'
            f'{delta}</div>'
        )
    st.markdown(
        f"""<div style="border-left:2px solid rgba(207,185,145,0.22);
        padding:0.50rem 0 0.50rem 0.70rem;background:#141414;
        border-bottom:1px solid #1a1a1a;transition:border-left-color 0.2s">
        <div style="font-size:0.52rem;font-weight:700;letter-spacing:0.16em;
        text-transform:uppercase;color:#555960;margin-bottom:5px">{label}</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:1.02rem;
        font-weight:700;color:#CFB991;line-height:1.1">{value}</div>
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
        f'border-bottom:1px solid #2a2a2a;display:flex;align-items:baseline;gap:0">'
        f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.56rem;font-weight:700;'
        f'text-transform:uppercase;letter-spacing:0.10em;color:{_GOLD};'
        f'margin-right:0.65rem">{number}</span>'
        f'<span style="font-family:\'DM Sans\',sans-serif;font-size:0.76rem;font-weight:700;'
        f'color:#e8e9ed">{title}</span>'
        f'{sub_html}</div>',
        unsafe_allow_html=True,
    )


def _regime_banner(label: str, sub: str = "", color: str = "#8E6F3E") -> None:
    """Flat inline regime label with ambient glow at crisis intensity."""
    sub_html = (
        f'<span style="font-family:\'DM Sans\',sans-serif;font-size:0.68rem;'
        f'color:#666;margin-left:0.75rem;font-weight:400;font-style:italic">{sub}</span>'
        if sub else ""
    )
    # Ambient glow — more visible at higher severity (crisis red vs. normal gold)
    glow = f"box-shadow:0 -1px 8px {color}26"
    st.markdown(
        f'<div style="border-top:2px solid {color};border-bottom:1px solid #1e1e1e;'
        f'padding:0.42rem 0.5rem 0.42rem 0;{glow};'
        f'display:flex;align-items:baseline;gap:0;margin-bottom:0.9rem;'
        f'background:linear-gradient(180deg,{color}08 0%,transparent 100%)">'
        f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.50rem;font-weight:700;'
        f'letter-spacing:0.18em;text-transform:uppercase;color:#444a55;margin-right:0.65rem">'
        f'REGIME</span>'
        f'<span style="font-family:\'DM Sans\',sans-serif;font-size:0.84rem;'
        f'font-weight:700;color:{color};letter-spacing:-0.01em">{label}</span>'
        f'{sub_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


def _narrative_box(text: str) -> None:
    """Pre-loaded data-driven narrative. Paragraphs separated by blank lines."""
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
        f'<div style="background:#131313;border-top:1px solid #222;border-bottom:1px solid #222;'
        f'padding:0.80rem 0.90rem;margin:0.35rem 0 0.75rem">'
        f'<div style="font-family:\'DM Sans\',sans-serif;font-size:0.73rem;'
        f'color:#c0c0c0;line-height:1.85;max-width:820px">{html}</div></div>',
        unsafe_allow_html=True,
    )


def _primary_chart(fig, caption: str = "") -> None:
    """Render the headline chart for a page/section."""
    _chart(fig)
    if caption:
        st.markdown(
            f'<p style="font-family:\'DM Sans\',sans-serif;font-size:0.62rem;'
            f'color:#888;font-style:italic;margin:0 0 0.55rem 0">{caption}</p>',
            unsafe_allow_html=True,
        )


# ── About page styles ───────────────────────────────────────────────────────

def _about_page_styles():
    """Inject CSS for About pages (hero banner, cards, timelines, etc.)."""
    st.markdown("""<style>
    /* ── Base font for about pages ─────────────────────── */
    .about-hero, .about-hero *, .about-card, .about-card *,
    .exp-item, .edu-item, .pub-item, .cert-item {
        font-family: 'DM Sans', sans-serif;
    }

    /* ── Hero banner ───────────────────────────────────── */
    .about-hero {
        background: #141414;
        padding: 0;
        margin-bottom: 1.4rem;
        overflow: hidden;
        border-top: 2px solid #CFB991;
        border-bottom: 1px solid #222;
    }
    .about-hero-inner {
        display: flex;
        align-items: stretch;
    }
    .about-hero .hero-photo {
        flex-shrink: 0;
        width: 160px;
        overflow: hidden;
    }
    .about-hero .hero-photo img {
        width: 100%;
        height: 100%;
        object-fit: cover;
        object-position: top center;
        display: block;
        filter: grayscale(15%);
    }
    .about-hero .hero-body {
        flex: 1;
        padding: 1.2rem 1.5rem;
        display: flex;
        flex-direction: column;
        justify-content: center;
        border-left: 1px solid #222;
    }
    .about-hero .overline {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.50rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.18em;
        color: #555960;
        margin: 0 0 0.35rem 0;
    }
    .about-hero-name {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.82rem;
        font-weight: 700;
        color: #e8e9ed;
        margin: 0 0 0.18rem 0;
        letter-spacing: -0.01em;
        line-height: 1.15;
    }
    .about-hero .subtitle {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.60rem;
        color: #CFB991;
        margin: 0 0 0.35rem 0;
        font-weight: 500;
        line-height: 1.5;
    }
    .about-hero .tagline {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.62rem;
        color: #8890a1;
        margin: 0 0 0.60rem 0;
        line-height: 1.70;
        max-width: 560px;
    }
    .about-hero .links {
        display: flex;
        gap: 1.2rem;
        flex-wrap: wrap;
        align-items: center;
    }
    .about-hero .links a {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.50rem;
        font-weight: 700;
        color: #CFB991;
        text-decoration: none;
        letter-spacing: 0.10em;
        text-transform: uppercase;
        opacity: 0.85;
    }
    .about-hero .links a:hover { opacity: 1; }

    /* ── Section separator / card ──────────────────────── */
    .about-card {
        background: transparent;
        border-top: 1px solid #1e1e1e;
        padding: 0.80rem 0 0.2rem 0;
        margin-bottom: 0.1rem;
    }
    .about-card-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.50rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.18em;
        color: #555960;
        margin: 0 0 0.60rem 0;
    }

    /* ── Experience timeline ───────────────────────────── */
    .exp-item {
        padding: 0.5rem 0 0.5rem 0.75rem;
        border-left: 1px solid #1e1e1e;
        margin-bottom: 0.5rem;
        position: relative;
    }
    .exp-item::before {
        content: '';
        position: absolute;
        left: -3px;
        top: 0.65rem;
        width: 5px;
        height: 5px;
        background: #2a2a2a;
        border-radius: 50%;
    }
    .exp-role {
        font-size: 0.62rem;
        font-weight: 700;
        color: #e8e9ed;
        margin: 0 0 0.06rem 0;
        line-height: 1.3;
    }
    .exp-org {
        font-size: 0.59rem;
        font-weight: 600;
        color: #CFB991;
        margin: 0 0 0.08rem 0;
    }
    .exp-meta {
        font-size: 0.54rem;
        color: #555960;
        margin: 0 0 0.22rem 0;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.02em;
    }
    .exp-desc {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.62rem;
        color: #8890a1;
        line-height: 1.68;
        margin: 0;
    }

    /* ── Education ─────────────────────────────────────── */
    .edu-item {
        padding: 0.45rem 0;
        border-bottom: 1px solid #1a1a1a;
    }
    .edu-item:last-child { border-bottom: none; }
    .edu-school {
        font-size: 0.62rem;
        font-weight: 700;
        color: #e8e9ed;
        margin: 0 0 0.06rem 0;
    }
    .edu-dept {
        font-size: 0.57rem;
        color: #8890a1;
        margin: 0 0 0.06rem 0;
    }
    .edu-degree {
        font-size: 0.59rem;
        color: #CFB991;
        margin: 0 0 0.06rem 0;
        font-weight: 500;
    }
    .edu-year {
        font-size: 0.52rem;
        color: #555960;
        margin: 0;
        font-family: 'JetBrains Mono', monospace;
    }

    /* ── Publication ───────────────────────────────────── */
    .pub-item { padding: 0.3rem 0; }
    .pub-title {
        font-size: 0.61rem;
        font-weight: 600;
        color: #e8e9ed;
        margin: 0 0 0.18rem 0;
        line-height: 1.55;
    }
    .pub-authors {
        font-size: 0.57rem;
        color: #8890a1;
        margin: 0 0 0.1rem 0;
    }
    .pub-journal {
        font-size: 0.55rem;
        color: #CFB991;
        font-style: italic;
        margin: 0 0 0.08rem 0;
    }
    .pub-detail {
        font-size: 0.54rem;
        color: #555960;
        margin: 0 0 0.3rem 0;
        font-family: 'JetBrains Mono', monospace;
    }
    .pub-link {
        font-size: 0.54rem;
        font-weight: 600;
        color: #CFB991;
        text-decoration: none;
        border-bottom: 1px solid rgba(207,185,145,0.3);
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .pub-link:hover { border-color: #CFB991; }

    /* ── Certifications ────────────────────────────────── */
    .cert-item {
        padding: 0.32rem 0;
        border-bottom: 1px solid #1a1a1a;
    }
    .cert-item:last-child { border-bottom: none; }
    .cert-name {
        font-size: 0.60rem;
        font-weight: 600;
        color: #e8e9ed;
        margin: 0 0 0.05rem 0;
    }
    .cert-issuer {
        font-size: 0.54rem;
        color: #555960;
        margin: 0;
        font-family: 'JetBrains Mono', monospace;
    }

    /* ── Interest tags ─────────────────────────────────── */
    .interest-tag {
        display: inline-block;
        padding: 0.16rem 0.55rem;
        font-size: 0.52rem;
        font-weight: 600;
        margin: 0.1rem 0.08rem;
        letter-spacing: 0.03em;
    }
    .interest-gold {
        background: rgba(207,185,145,0.08);
        color: #CFB991;
        border: 1px solid rgba(207,185,145,0.22);
    }
    .interest-neutral {
        background: transparent;
        color: #8890a1;
        border: 1px solid #1e1e1e;
    }

    /* ── Acknowledgments ───────────────────────────────── */
    .ack-text {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.62rem;
        color: #8890a1;
        line-height: 1.68;
        margin: 0 0 0.35rem 0;
    }
    .ack-text strong { color: #c8c8c8; font-weight: 600; }

    /* ── Stats row ─────────────────────────────────────── */
    .stat-row {
        display: flex;
        gap: 0;
        margin-top: 0.7rem;
        border-top: 1px solid #1e1e1e;
        border-bottom: 1px solid #1e1e1e;
    }
    .stat-item {
        flex: 1;
        text-align: center;
        padding: 0.5rem 0.25rem;
        border-right: 1px solid #1e1e1e;
    }
    .stat-item:last-child { border-right: none; }
    .stat-num {
        font-size: 0.88rem;
        font-weight: 700;
        color: #CFB991;
        font-family: 'JetBrains Mono', monospace;
        margin: 0;
        line-height: 1;
    }
    .stat-label {
        font-size: 0.46rem;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        color: #555960;
        margin: 3px 0 0 0;
    }
    </style>""", unsafe_allow_html=True)

# ── Page header ────────────────────────────────────────────────────────────

def _no_api_key_banner(context: str = "AI analysis features") -> None:
    """
    Show a one-liner st.info banner when no AI provider key is configured.
    Safe to call on any page — silently does nothing if a key is present.
    """
    try:
        import streamlit as _st
        _keys = _st.secrets.get("keys", {})
        _has_key = bool(
            (_keys.get("anthropic_api_key") or "")
            or (_keys.get("openai_api_key") or "")
        )
    except Exception:
        _has_key = False
    if not _has_key:
        import streamlit as _st
        _st.info(
            f"{context} require an AI provider key. "
            "Add `anthropic_api_key` or `openai_api_key` under `[keys]` in "
            "`.streamlit/secrets.toml` to enable agent-generated insights.",
            icon="🔑",
        )


def _page_header(title: str, subtitle: str = "", eyebrow: str = "") -> None:
    """
    Branded page header used on every page.
    Gold left-border structural anchor · logo mark eyebrow · clean h1 title.
    """
    import streamlit.components.v1 as _cmp
    _cmp.html('<script>window.parent.scrollTo({top:0,behavior:"instant"});</script>', height=0)
    _Fh = "font-family:'DM Sans',sans-serif;"
    _Mh = "font-family:'JetBrains Mono',monospace;"
    _eye = eyebrow or "Cross-Asset Spillover Monitor \u00b7 Purdue Daniels \u00b7 MGMT 69000-120"
    _logo = _footer_logo_b64()
    _logo_img = (
        f'<img src="{_logo}" alt="" style="height:14px;width:auto;object-fit:contain;'
        f'opacity:0.55;flex-shrink:0;display:block;margin-right:8px;" />'
        if _logo else ""
    )
    _sub_html = (
        f'<p style="{_Fh}font-size:0.72rem;color:#8890a1;margin:0 0 0.5rem 0">{subtitle}</p>'
        if subtitle else '<div style="margin-bottom:0.5rem"></div>'
    )
    st.markdown(
        f'<div style="border-left:2px solid #CFB991;'
        f'box-shadow:-1px 0 12px rgba(207,185,145,0.10);'
        f'padding-left:12px;margin-bottom:0.75rem">'
        f'<div style="display:flex;align-items:center;margin-bottom:4px">'
        f'{_logo_img}'
        f'<span style="{_Mh}font-size:6.5px;font-weight:700;letter-spacing:.20em;'
        f'text-transform:uppercase;color:#444a55">{_eye}</span>'
        f'</div>'
        f'<h1 style="{_Fh}font-size:1.22rem;font-weight:700;'
        f'color:#e8e8e8;margin:0 0 2px;letter-spacing:-0.01em;line-height:1.2">{title}</h1>'
        f'{_sub_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


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
    from datetime import datetime, date as _date
    yr  = datetime.now().year
    ts  = datetime.now().strftime("%B %d, %Y at %H:%M UTC")

    # ── Controls tray — inline, just above the footer ─────────────────────
    # Rendered here (inside _page_footer) so every page gets it automatically
    # without any per-page wiring. Skipped on About pages and when agent state
    # is not yet initialised (e.g., very first cold boot before init_agents()).
    if (
        not st.session_state.get("_is_about", False)
        and st.session_state.get("agents")
    ):
        try:
            from src.analysis.agent_state import AGENTS, pending_count
            from src.ui.agent_panel import _render_workforce_content

            _n   = sum(1 for a in st.session_state["agents"].values()
                       if a.get("enabled", True))
            _p   = pending_count()
            _dc  = "#27ae60" if _n == 8 else ("#e67e22" if _n >= 5 else "#c0392b")
            _s0  = st.session_state.get("g_start", _date(2010, 1, 1))
            _e0  = st.session_state.get("g_end",   _date.today())
            _ss  = _s0.strftime("%Y-%m-%d")
            _es  = _e0.strftime("%Y-%m-%d")
            _lbl = f"CONTROLS  ·  {_n}/8"

            # CSS: style the trigger button and popover panel.
            # No position:fixed — button sits inline in page flow.
            st.markdown("""<style>
/* Controls tray row: right-align within its column */
div[data-testid="stColumn"]:has(div[data-testid="stPopover"]) {
    display: flex !important;
    justify-content: flex-end !important;
    align-items: center !important;
}
div[data-testid="stPopover"] > button {
    background: #0d0d0d !important;
    border: 1px solid rgba(207,185,145,0.28) !important;
    border-left: 3px solid #CFB991 !important;
    color: #CFB991 !important;
    border-radius: 3px !important;
    padding: 0 1.1rem 0 0.85rem !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.56rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    box-shadow: none !important;
    height: 30px !important;
    white-space: nowrap !important;
    transition: background 0.15s !important;
}
div[data-testid="stPopover"] > button:hover {
    background: rgba(207,185,145,0.07) !important;
}
div[data-testid="stPopoverBody"] {
    background: #070707 !important;
    border: 1px solid #282828 !important;
    border-top: 2px solid #CFB991 !important;
    border-radius: 3px !important;
    min-width: 380px !important;
    max-width: 500px !important;
    padding: 0.9rem 1rem 0.8rem !important;
    box-shadow: 0 -8px 28px rgba(0,0,0,0.6) !important;
}
div[data-testid="stPopoverBody"] [data-testid="stTextInput"] label p {
    font-size: 0.48rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
    color: #555960 !important;
}
div[data-testid="stPopoverBody"] [data-testid="stTextInput"] input {
    font-size: 0.62rem !important;
    font-family: 'JetBrains Mono', monospace !important;
    height: 28px !important;
    background: #111 !important;
    border-color: #222 !important;
    color: #c8c8c8 !important;
    letter-spacing: 0.04em !important;
}
div[data-testid="stPopoverBody"] .stButton > button {
    padding: 0.20rem 0.3rem !important;
    font-size: 0.46rem !important;
    letter-spacing: 0.04em !important;
    min-width: 0 !important;
    border-radius: 2px !important;
    text-align: left !important;
}
</style>""", unsafe_allow_html=True)

            # Separator + right-aligned trigger row
            st.markdown(
                '<div style="border-top:1px solid #1e1e1e;margin:2rem 0 0.5rem"></div>',
                unsafe_allow_html=True,
            )
            _gap, _btn_col = st.columns([0.82, 0.18])
            with _btn_col:
                with st.popover(_lbl, width="stretch"):
                    # Section header helper
                    def _sec(label: str, sub: str = "") -> None:
                        _sh = (
                            f'<span style="font-family:\'JetBrains Mono\',monospace;'
                            f'font-size:0.42rem;color:#444;letter-spacing:0.08em;margin-left:6px">'
                            f'{sub}</span>' if sub else ""
                        )
                        st.markdown(
                            f'<div style="display:flex;align-items:baseline;'
                            f'border-left:2px solid #CFB991;padding-left:6px;margin:0 0 7px 0">'
                            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.47rem;'
                            f'font-weight:700;letter-spacing:0.14em;text-transform:uppercase;'
                            f'color:#8890a1">{label}</span>'
                            f'{_sh}</div>',
                            unsafe_allow_html=True,
                        )

                    _sec("Date Range", f"{_ss} → {_es}")
                    _c1, _c2 = st.columns(2)
                    _sr = _c1.text_input("From", value=_ss, key="g_start_txt", help="YYYY-MM-DD")
                    _er = _c2.text_input("To",   value=_es, key="g_end_txt",   help="YYYY-MM-DD")
                    try:
                        _ps = _date.fromisoformat(_sr)
                        if _ps != _s0:
                            st.session_state["g_start"] = _ps
                    except ValueError:
                        pass
                    try:
                        _pe = _date.fromisoformat(_er)
                        if _pe != _e0:
                            st.session_state["g_end"] = _pe
                    except ValueError:
                        pass

                    st.markdown(
                        '<div style="border-top:1px solid #1a1a1a;margin:0.75rem 0 0.65rem"></div>',
                        unsafe_allow_html=True,
                    )

                    _al = "ALL OPERATIONAL" if _n == 8 else f"{_n}/8 ACTIVE"
                    st.markdown(
                        f'<div style="display:flex;align-items:baseline;'
                        f'border-left:2px solid #CFB991;padding-left:6px;margin:0 0 7px 0">'
                        f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.47rem;'
                        f'font-weight:700;letter-spacing:0.14em;text-transform:uppercase;'
                        f'color:#8890a1">AI Workforce</span>'
                        f'<span style="font-size:0.42rem;margin-left:6px;color:{_dc}">● {_al}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    _render_workforce_content(
                        agents_state=st.session_state["agents"],
                        agent_list=list(AGENTS.items()),
                        n_pending=_p,
                        key_prefix="ctrl_",
                    )
        except Exception:
            pass  # never crash the footer

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
            "width:72px;height:72px;background:#CFB991;border-radius:0;margin-bottom:18px;gap:2px;'>"
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

  /* ── 1. Full-width: style BOTH the iframe AND its Streamlit parent div ──
     Matching the navbar's dual-target approach.
     left:50%+translateX(-50%) is computed against the PARENT width, not the
     viewport, so it breaks when the parent is constrained by block-container
     max-width. Instead: free the parent first, then set iframe to width:100%. */
  var p = f.parentElement;
  if (p) {{
    p.style.cssText = 'width:100%!important;max-width:none!important;' +
      'overflow:visible!important;padding:0!important;margin:0!important;';
  }}
  f.style.cssText = 'width:100%!important;border:none!important;' +
    'display:block!important;margin:0!important;overflow:hidden!important;';

  /* ── 2. Walk ancestors above p: remove overflow clip + bottom spacing ── */
  var el = p ? p.parentElement : null;
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

  /* ── 4. Auto-size: update BOTH iframe height AND parent div height ──
     Streamlit allocates a fixed height slot for the parent div from the
     components.html(height=) param. Without updating p.style.height the
     parent retains that allocation even after the iframe shrinks, producing
     a blank gap below the footer content. */
  function measure() {{
    var h = document.documentElement.scrollHeight || document.body.scrollHeight;
    if (h > 20) {{
      f.style.height = h + 'px';
      if (p) p.style.height = h + 'px';
    }}
  }}
  /* Run immediately, after fonts load, and once more 200 ms later */
  measure();
  if (document.fonts && document.fonts.ready) {{
    document.fonts.ready.then(function() {{ measure(); setTimeout(measure, 200); }});
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
        <li><a href="#" onclick="nav('scenario_engine');return false;">Scenario Engine</a></li>
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
        <li><a href="https://business.purdue.edu/" target="_blank" rel="noopener noreferrer">Daniels School of Business</a></li>
      </ul>
    </div>

    <div>
      <p class="ft-hd">Data Sources</p>
      <ul>
        <li><a href="https://finance.yahoo.com" target="_blank" rel="noopener noreferrer">Yahoo Finance</a></li>
        <li><a href="https://fred.stlouisfed.org" target="_blank" rel="noopener noreferrer">FRED &middot; Federal Reserve</a></li>
        <li><a href="https://financialdatasets.ai" target="_blank" rel="noopener noreferrer">FinancialDatasets</a></li>
        <li><a href="https://www.cftc.gov" target="_blank" rel="noopener noreferrer">CFTC COT Reports</a></li>
      </ul>
    </div>

  </div>
</div>
<div class="ft-bar">
  <p>&copy; {yr} Cross-Asset Spillover Monitor &middot; Purdue Daniels &middot; MGMT 69000-120 &middot; For educational purposes only &middot; Not investment advice</p>
</div>
</body></html>""", height=300, scrolling=False)
    # components.html() always returns None — navigation is handled entirely
    # by window.parent.location.href inside nav() in the JS above.
    # The _ft_click return-value check that previously appeared here was dead
    # code and has been removed.


# ── Nexus-inspired arrangement components ─────────────────────────────────────
# These helpers render structural HTML using the CSS classes defined in app.py.
# They encode the Nexus Terminal's panel/feed/badge layout language into
# reusable Python functions so pages stay DRY.

def _nx_badge(text: str, level: str = "nominal") -> str:
    """
    Return an inline HTML severity badge string (not rendered — caller embeds in f-string).
    level: 'critical' | 'warning' | 'nominal' | 'live' | 'high-impact' | 'med-impact' |
           'low-impact' | 'active' | 'draft' | 'error'
    """
    css = f"nx-badge nx-badge-{level}"
    return f'<span class="{css}">{text}</span>'


def _nx_live_dot(color: str = "green") -> str:
    """Return the animated live-status dot HTML. color: 'green' | 'red' | 'amber'"""
    extra = "" if color == "green" else f" {color}"
    return f'<span class="nx-live-dot{extra}"></span>'


def _nx_panel_header(title: str, meta: str = "", badge: str = "") -> None:
    """Render a Nexus-style panel header row: TITLE  [meta text]  [badge]."""
    right = ""
    if badge:
        right += badge + " "
    if meta:
        right += f'<span class="nx-panel-meta">{meta}</span>'
    st.markdown(
        f'<div class="nx-panel-header">'
        f'<span class="nx-panel-title">{title}</span>'
        f'<span style="display:flex;align-items:center;gap:6px">{right}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _nx_intel_table(rows: list[dict]) -> None:
    """
    Render a Nexus-style intelligence alert table.
    Each row dict: {ts, entity, condition, level, action_label (opt), action_url (opt)}
    level: 'critical' | 'warning' | 'nominal'
    """
    html = '<div style="padding:0">'
    for r in rows:
        badge = _nx_badge(r.get("level", "nominal").upper(), r.get("level", "nominal"))
        action_html = ""
        if r.get("action_label"):
            action_html = (
                f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.50rem;'
                f'font-weight:700;letter-spacing:0.08em;text-transform:uppercase;'
                f'border:1px solid rgba(207,185,145,0.35);color:#CFB991;'
                f'padding:2px 8px;cursor:pointer;white-space:nowrap">'
                f'{r["action_label"]}</span>'
            )
        html += (
            f'<div class="nx-intel-row">'
            f'<span class="nx-intel-ts">{r.get("ts","")}</span>'
            f'<span class="nx-intel-entity">{r.get("entity","")}</span>'
            f'<span class="nx-intel-condition">{r.get("condition","")}</span>'
            f'<span class="nx-intel-severity">{badge} {action_html}</span>'
            f'</div>'
        )
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def _nx_feed_item(
    type_label: str,
    body: str,
    ts: str = "",
    impact: str = "",
    level: str = "intel",   # 'critical' | 'warning' | 'intel' | 'data'
) -> str:
    """
    Return HTML for a single Nexus-style right-panel feed item.
    Caller wraps multiple items in a .nx-feed-panel div.
    """
    impact_html = f'<div class="nx-feed-impact">{impact}</div>' if impact else ""
    return (
        f'<div class="nx-feed-item {level}">'
        f'<div class="nx-feed-item-header">'
        f'<span class="nx-feed-item-type">{type_label}</span>'
        f'<span class="nx-feed-item-ts">{ts}</span>'
        f'</div>'
        f'<div class="nx-feed-item-body">{body}</div>'
        f'{impact_html}'
        f'</div>'
    )


def _nx_kpi_tile(
    ticker: str,
    value: str,
    delta: str = "",
    delta_up: bool | None = None,
) -> str:
    """Return HTML for a single Nexus KPI ticker tile. Embed via st.markdown."""
    if delta:
        if delta_up is True:
            d_html = f'<div class="nx-kpi-tile-delta-up">▲ {delta}</div>'
        elif delta_up is False:
            d_html = f'<div class="nx-kpi-tile-delta-down">▼ {delta}</div>'
        else:
            d_html = f'<div style="color:#8890a1;font-size:0.62rem">{delta}</div>'
    else:
        d_html = ""
    return (
        f'<div class="nx-kpi-tile">'
        f'<div class="nx-kpi-tile-ticker">{ticker}</div>'
        f'<div class="nx-kpi-tile-value">{value}</div>'
        f'{d_html}'
        f'</div>'
    )


def _nx_route_row(name: str, status: str, level: str = "nominal") -> str:
    """Return HTML for a single trade-route status row."""
    badge = _nx_badge(status, level)
    icon = {"nominal": "✓", "warning": "⚠", "critical": "✕", "live": "●"}.get(level, "·")
    icon_color = {
        "nominal": "#27ae60", "warning": "#e67e22",
        "critical": "#c0392b", "live": "#27ae60",
    }.get(level, "#8890a1")
    return (
        f'<div class="nx-route-row">'
        f'<span style="color:{icon_color};margin-right:6px;font-size:0.62rem">{icon}</span>'
        f'<span class="nx-route-name">{name}</span>'
        f'{badge}'
        f'</div>'
    )
