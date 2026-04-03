"""
Proactive AI alert banner.
Renders structured alerts from the alert engine and optionally enriches
them with a single AI-generated morning briefing (Claude or GPT-4o).
Cached 30 min so API is called at most twice per hour.
"""

from __future__ import annotations

import datetime
import streamlit as st

from src.analysis.proactive_alerts import Alert

_F = "font-family:'DM Sans',sans-serif;"

_SEV_META = {
    "critical": {"border": "#c0392b", "badge_bg": "#c0392b", "badge_fg": "#fff",  "label": "CRITICAL"},
    "warning":  {"border": "#e67e22", "badge_bg": "#e67e22", "badge_fg": "#000",  "label": "WARNING"},
    "info":     {"border": "#2980b9", "badge_bg": "#2980b9", "badge_fg": "#fff",  "label": "INFO"},
}

_PAGE_LABELS = {
    "overview":     "Overview",
    "correlation":  "Correlation Analysis",
    "watchlist":    "Commodities Watchlist",
    "spillover":    "Spillover Analytics",
    "trade_ideas":  "Trade Ideas",
    "insights":     "Actionable Insights",
}


# ── AI briefing (cached) ─────────────────────────────────────────────────────

@st.cache_data(ttl=1800, show_spinner=False)
def _generate_briefing(alert_summary: str, market_context: str,
                        provider: str, api_key: str) -> str:
    """
    Call Claude or GPT-4o to write a single 3–5 sentence morning briefing
    covering all active alerts. Cached 30 minutes.
    """
    system = (
        "You are a senior cross-asset risk officer embedded in the Cross-Asset Spillover Monitor "
        "at Purdue University Daniels School of Business. You monitor live equity, commodity, "
        "fixed income, and FX markets. Write a terse, institutional morning briefing."
    )
    user = (
        f"ACTIVE ALERTS (computed from live data):\n{alert_summary}\n\n"
        f"MARKET CONTEXT:\n{market_context}\n\n"
        "Write a 3–5 sentence morning briefing covering the most important risks and their "
        "likely cross-asset transmission paths. Be precise and quantitative. "
        "No disclaimers. No greetings. Start directly with the key risk."
    )
    try:
        if provider == "anthropic":
            import anthropic as _ant
            client = _ant.Anthropic(api_key=api_key)
            resp = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=300,
                messages=[{"role": "user", "content": user}],
                system=system,
            )
            return resp.content[0].text.strip()
        else:
            from openai import OpenAI as _OAI
            client = _OAI(api_key=api_key)
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                max_tokens=300,
                temperature=0.2,
            )
            return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Briefing unavailable: {e}"


# ── Renderer ─────────────────────────────────────────────────────────────────

def render_alert_banner(
    alerts: list[Alert],
    market_context: str = "",
    max_alerts: int = 5,
) -> None:
    """
    Render the proactive alert feed at the top of a page.
    Shows at most `max_alerts` cards plus an optional AI briefing.
    Does nothing if the alert list is empty.
    """
    if not alerts:
        return

    # ── Read API keys from secrets ────────────────────────────────────────
    anthropic_key = openai_key = ""
    try:
        _keys = st.secrets.get("keys", {})
        anthropic_key = _keys.get("anthropic_api_key", "") or ""
        openai_key    = _keys.get("openai_api_key",    "") or ""
    except Exception:
        pass
    provider = "anthropic" if anthropic_key else ("openai" if openai_key else None)
    api_key  = anthropic_key or openai_key

    # ── Section header ────────────────────────────────────────────────────
    n_critical = sum(1 for a in alerts if a.severity == "critical")
    n_warning  = sum(1 for a in alerts if a.severity == "warning")
    header_color = "#c0392b" if n_critical else ("#e67e22" if n_warning else "#2980b9")

    st.markdown(
        f'<div style="{_F}display:flex;align-items:center;gap:0.7rem;margin-bottom:0.6rem">'
        f'<span style="width:3px;height:1.1rem;background:{header_color};'
        f'border-radius:2px;display:inline-block"></span>'
        f'<span style="font-size:0.58rem;font-weight:700;letter-spacing:0.14em;'
        f'text-transform:uppercase;color:{header_color}">Live Alerts</span>'
        f'<span style="font-size:0.58rem;color:#555960">'
        f'{len(alerts)} signal{"s" if len(alerts) != 1 else ""} · '
        f'{datetime.datetime.now().strftime("%H:%M")}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── AI briefing (if key available and alerts exist) ───────────────────
    if provider and market_context:
        alert_summary = "\n".join(
            f"[{a.severity.upper()}] {a.title}: {a.body}" for a in alerts[:max_alerts]
        )
        with st.spinner("AI Analyst generating briefing…"):
            briefing = _generate_briefing(alert_summary, market_context, provider, api_key)

        if briefing and not briefing.startswith("Briefing unavailable"):
            st.markdown(
                f'<div style="{_F}background:#1c1c1c;border:1px solid #2a2a2a;'
                f'border-radius:0;'
                f'padding:0.75rem 1rem;margin-bottom:0.7rem">'
                f'<div style="font-size:0.52rem;font-weight:700;letter-spacing:0.14em;'
                f'text-transform:uppercase;color:#CFB991;margin-bottom:0.4rem">'
                f'AI Risk Officer · Morning Briefing</div>'
                f'<div style="font-size:0.78rem;color:#c8c8c8;line-height:1.65">'
                f'{briefing}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Individual alert cards ────────────────────────────────────────────
    shown = alerts[:max_alerts]
    cols  = st.columns(min(len(shown), 3))

    for i, alert in enumerate(shown):
        meta = _SEV_META.get(alert.severity, _SEV_META["info"])
        page_label = _PAGE_LABELS.get(alert.page_hint, alert.page_hint.replace("_", " ").title())

        with cols[i % len(cols)]:
            st.markdown(
                f'<div style="{_F}background:#1c1c1c;border:1px solid #2a2a2a;'
                f'border-top:2px solid {meta["border"]};border-radius:0;'
                f'padding:0.6rem 0.8rem;height:100%">'

                # Severity badge + category
                f'<div style="display:flex;align-items:center;gap:0.4rem;margin-bottom:0.4rem">'
                f'<span style="font-size:0.5rem;font-weight:700;letter-spacing:0.1em;'
                f'text-transform:uppercase;background:{meta["badge_bg"]};color:{meta["badge_fg"]};'
                f'padding:1px 5px;border-radius:2px">{meta["label"]}</span>'
                f'<span style="font-size:0.58rem;color:#555960;text-transform:uppercase;'
                f'letter-spacing:0.08em">{alert.category}</span>'
                f'</div>'

                # Title
                f'<div style="font-size:0.78rem;font-weight:700;color:#e8e9ed;'
                f'line-height:1.3;margin-bottom:0.35rem">{alert.title}</div>'

                # Body (first sentence only in card)
                f'<div style="font-size:0.68rem;color:#8890a1;line-height:1.5">'
                f'{alert.body.split(".")[0]}.</div>'

                # Page hint
                f'<div style="font-size:0.55rem;color:#555960;margin-top:0.4rem;'
                f'text-transform:uppercase;letter-spacing:0.08em">→ {page_label}</div>'

                f'</div>',
                unsafe_allow_html=True,
            )

    if len(alerts) > max_alerts:
        st.markdown(
            f'<p style="{_F}font-size:0.62rem;color:#555960;margin-top:0.4rem">'
            f'+ {len(alerts) - max_alerts} more signal{"s" if len(alerts) - max_alerts != 1 else ""} '
            f'- navigate to the relevant page for full detail.</p>',
            unsafe_allow_html=True,
        )

    st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
