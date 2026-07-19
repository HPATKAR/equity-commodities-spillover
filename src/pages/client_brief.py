"""
Client Brief (Lite) - the terminal's intelligence in plain English.

Persona 3 (family offices, sophisticated RIAs) does not want Diebold-Yilmaz,
deflated Sharpe or CIS bands. They want to tell an end client, in a paragraph,
what is going on in markets and what it means for a portfolio like theirs. This
page takes the same live signals the rest of the terminal computes (stress score,
correlation regime, active-conflict intensity, the alert feed) and translates them
through a deterministic plain-language layer into a short, jargon-free brief,
tailored to a Conservative / Balanced / Growth profile, exportable as a white-label
PDF on the advisor's own firm name. No AI key required; the translation is rules,
not a model, so it always runs and always says the same thing for the same data.
"""
from __future__ import annotations

import datetime as _dt

import streamlit as st

from src.ui.shared import (_page_header, _page_footer, _page_intro,
                           _definition_block)
# Reuse the cached alert-engine inputs from the Alert Center (regime, stress,
# avg_corr, the alert feed) so this page adds no new data plumbing.
from src.pages.alert_center import _alert_inputs

_GOLD, _TXT, _MUT, _BRD = "#CFB991", "#e8e9ed", "#8890a1", "#1e1e1e"
_PROFILES = ["Conservative", "Balanced", "Growth"]
_RNAME = ["Decorrelated", "Normal", "Elevated", "Crisis"]

_ENERGY = {"WTI Crude Oil", "Brent Crude", "Natural Gas", "Gasoline (RBOB)",
           "Heating Oil", "Crude Oil", "Gasoline", "Brent", "WTI", "Nickel"}
_AG = {"Wheat", "Corn", "Soybeans", "Soybean Oil", "Sugar #11", "Sugar",
       "Coffee", "Cotton"}
_METALS = {"Gold", "Silver", "Platinum", "Palladium", "Copper", "Aluminum"}


# ── Plain-language translation layer ──────────────────────────────────────────
def _plain_channel(comms: list) -> str:
    comms = set(comms or [])
    if comms & _ENERGY:
        return "energy and oil prices"
    if comms & _AG:
        return "food and grain prices"
    if comms & _METALS:
        return "metals prices"
    return "broad market sentiment"


def _env_section(ctx: dict) -> tuple[str, str]:
    s = float(ctx.get("risk_score", 0.0))
    r = int(ctx.get("regime", 1))
    ac = float(ctx.get("avg_corr", 0.0))
    if r >= 3 or s >= 65:
        head = "Markets are under significant stress."
    elif r >= 2 or s >= 50:
        head = "Markets are showing rising stress."
    else:
        head = "Markets are relatively calm."
    band = ("high" if s >= 65 else "elevated" if s >= 50
            else "moderate" if s >= 35 else "low")
    together = ("moving closely together" if ac >= 0.35
                else "showing some tendency to move together" if ac >= 0.20
                else "moving largely independently")
    divers = ("not offering much protection" if ac >= 0.35
              else "offering less protection than usual" if ac >= 0.20
              else "working well")
    body = (f"{head} Our cross-asset stress gauge reads {s:.0f} out of 100, which is "
            f"{band}. Stocks and commodities are {together}, so the diversification in "
            f"a typical portfolio is {divers} at the moment.")
    return head, body


def _drivers_section(max_n: int = 3) -> list[str]:
    try:
        from src.analysis.conflict_model import score_all_conflicts
        scores = score_all_conflicts()
    except Exception:
        return []
    active = [s for s in scores.values() if s.get("state", "active") == "active"]
    active.sort(key=lambda s: float(s.get("cis", 0.0)), reverse=True)
    out = []
    for s in active[:max_n]:
        cis = float(s.get("cis", 0.0))
        name = s.get("name") or s.get("label") or "an ongoing conflict"
        if cis >= 70:
            band = "a major source of market risk"
        elif cis >= 55:
            band = "an elevated concern"
        else:
            band = "worth keeping an eye on"
        chan = _plain_channel(s.get("affected_commodities", []))
        out.append(f"The {name} is currently {band}. For portfolios, it matters "
                   f"mainly through {chan}.")
    return out


def _plainify_alert(a: dict) -> str:
    cat, d = a.get("category", ""), a.get("data", {})
    if cat == "stress":
        return ("Overall market stress has picked up, so conditions are choppier "
                "than normal.")
    if cat == "regime":
        return ("Stocks and commodities are increasingly moving together, which "
                "weakens the protection diversification normally provides.")
    if cat == "correlation":
        return ("The link between stocks and commodities is tightening, so the two "
                "are cushioning each other less than usual.")
    if cat == "volatility":
        nm = d.get("commodity", "a commodity")
        return f"Price swings in {nm} have picked up sharply."
    if cat == "cot":
        nm = d.get("market", "a commodity")
        return (f"Traders have become heavily one-sided in {nm}, a positioning "
                "extreme that often precedes a reversal.")
    if cat == "country_exposure":
        return ("A few overseas stock markets are carrying higher geopolitical risk "
                "than usual right now.")
    if cat == "conflict":
        return "News flow around an active conflict has surged over the past week."
    if cat == "supply":
        nm = d.get("commodity", "energy")
        return (f"{nm} inventories are unusually low, which leaves prices more "
                "exposed to any supply shock.")
    return a.get("title", "")


def _watching_section(alerts: list, max_n: int = 4) -> list[str]:
    seen, out = set(), []
    for a in alerts:
        txt = _plainify_alert(a)
        key = txt[:40]
        if txt and key not in seen:
            seen.add(key)
            out.append(txt)
        if len(out) >= max_n:
            break
    return out


def _meaning_section(profile: str, ctx: dict) -> str:
    stressed = float(ctx.get("risk_score", 0.0)) >= 50
    if profile == "Conservative":
        return ("A portfolio built mainly for capital preservation, like yours, is "
                "positioned to weather this better than most. The priorities now are "
                "keeping enough cash on hand, favouring high-quality bonds, and "
                "resisting the temptation to reach for extra yield in riskier corners "
                "of the market." if stressed else
                "Your conservative positioning remains appropriate. Calmer conditions "
                "are a good moment to confirm that income sources are dependable and "
                "that cash reserves are sized where you want them.")
    if profile == "Balanced":
        return ("In a balanced portfolio like yours, this is exactly the environment "
                "where diversification is meant to earn its keep. It is worth "
                "reviewing whether your stock exposure is partially hedged, and "
                "whether safe havens such as gold or high-quality bonds deserve a "
                "slightly larger role until the stress subsides." if stressed else
                "Your balanced mix is well suited to current conditions. No dramatic "
                "changes are called for; rebalancing back to your targets is the main "
                "discipline that matters here.")
    # Growth
    return ("A growth-oriented portfolio like yours carries more stock-market risk, "
            "so a stress episode tends to be felt more sharply. This is the moment to "
            "confirm that position sizes are deliberate, and to consider modest hedges "
            "or trimming the most stretched holdings rather than reacting after the "
            "fact." if stressed else
            "Your growth tilt is working with a constructive market. The main "
            "discipline now is to keep winners from growing into oversized positions, "
            "and to hold some flexibility for when conditions eventually turn.")


def _build_brief(res: dict, profile: str) -> dict:
    ctx = res.get("ctx", {})
    alerts = res.get("alerts", [])
    head, env = _env_section(ctx)
    return {
        "headline": head,
        "profile": profile,
        "asof": ctx.get("asof", ""),
        "stress": float(ctx.get("risk_score", 0.0)),
        "regime": int(ctx.get("regime", 1)),
        "sections": [
            ("The Market Environment", [env]),
            ("What Is Driving It", _drivers_section() or
             ["No single geopolitical flashpoint is dominating markets at the moment; "
              "the current tone is being set more by broad economic conditions."]),
            ("What It Means For You", [_meaning_section(profile, ctx)]),
            ("What We Are Watching", _watching_section(alerts) or
             ["Nothing is flashing on our watch list beyond normal day-to-day "
              "market movement."]),
        ],
    }


# ── Page ──────────────────────────────────────────────────────────────────────
def page_client_brief(start: str, end: str, fred_key: str = "") -> None:
    _page_header("Client Brief",
                 "The terminal's intelligence, in plain English, on your letterhead")
    _page_intro(
        "A jargon-free version of what the rest of the terminal computes, written for "
        "an end client rather than a quant. Pick a risk profile and the terminal turns "
        "the live stress score, correlation regime, active-conflict intensity and alert "
        "feed into a short, plain-language brief: what is happening, what is driving it, "
        "what it means for a portfolio like theirs, and what to watch. Export it as a "
        "clean, white-label PDF on your own firm name. The translation is rule-based, so "
        "it runs without any AI key and reads the same way every time for the same data."
    )
    _definition_block(
        "Who this is for",
        "Family offices and advisers who need a client-ready narrative, not a "
        "factsheet of Greek letters. The profiles map to the usual vernacular: "
        "<b>Conservative</b> (capital preservation and income), <b>Balanced</b> (a "
        "roughly even growth-and-safety mix), <b>Growth</b> (equity-tilted). The brief "
        "is educational commentary, not personalised investment advice."
    )

    # Inputs
    c1, c2, c3, c4 = st.columns([1.3, 1.3, 1.2, 1.3], gap="medium")
    with c1:
        firm = st.text_input("Firm name", value=st.session_state.get("_cb_firm", ""),
                            placeholder="Your Advisory LLC", key="_cb_firm")
    with c2:
        client = st.text_input("Prepared for (client)", value="Valued Client",
                              key="_cb_client")
    with c3:
        advisor = st.text_input("Prepared by", value="", placeholder="Adviser name",
                              key="_cb_advisor")
    with c4:
        profile = st.radio("Client risk profile", _PROFILES, index=1,
                         horizontal=True, key="_cb_profile")

    with st.spinner("Reading the market and writing the brief..."):
        res = _alert_inputs(start, end)
    if res is None:
        st.error("Could not load market data to build the brief. Check the connection "
                 "and try again.")
        _page_footer()
        return

    brief = _build_brief(res, profile)

    # ── On-screen brief (clean, readable prose, not the dense mono grid) ──────
    _prose = ("font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;"
              "color:#d7dae1;font-size:.92rem;line-height:1.62")
    _hdr = (f'<div style="border:1px solid {_BRD};border-top:3px solid {_GOLD};'
            f'background:#0a0a0a;padding:.9rem 1.1rem;margin:.3rem 0 .2rem">'
            f'<div style="font-family:-apple-system,Segoe UI,Roboto,sans-serif;'
            f'font-size:1.15rem;font-weight:800;color:#f2f3f5;letter-spacing:.01em">'
            f'{(firm or "Market &amp; Risk Brief")}</div>'
            f'<div style="font-family:-apple-system,Segoe UI,Roboto,sans-serif;'
            f'font-size:.82rem;color:{_GOLD};font-weight:600;margin-top:1px">'
            f'Market &amp; Risk Brief</div>'
            f'<div style="font-family:-apple-system,Segoe UI,Roboto,sans-serif;'
            f'font-size:.72rem;color:{_MUT};margin-top:6px">'
            f'Prepared for {client or "Valued Client"}'
            f'{(" &middot; by " + advisor) if advisor else ""} &middot; '
            f'{brief["profile"]} profile &middot; '
            f'{_dt.date.today().strftime("%d %b %Y")}</div></div>')
    st.markdown(_hdr, unsafe_allow_html=True)

    # Headline
    st.markdown(
        f'<div style="font-family:-apple-system,Segoe UI,Roboto,sans-serif;'
        f'font-size:1.28rem;font-weight:800;color:#f2f3f5;margin:.9rem 0 .2rem;'
        f'line-height:1.3">{brief["headline"]}</div>', unsafe_allow_html=True)

    for title, paras in brief["sections"]:
        st.markdown(
            f'<div style="font-family:-apple-system,Segoe UI,Roboto,sans-serif;'
            f'font-size:.82rem;font-weight:700;letter-spacing:.06em;text-transform:uppercase;'
            f'color:{_GOLD};margin:1.05rem 0 .35rem">{title}</div>',
            unsafe_allow_html=True)
        if len(paras) > 1:
            _items = "".join(f'<li style="margin:.2rem 0">{p}</li>' for p in paras)
            st.markdown(f'<ul style="{_prose};margin:.1rem 0;padding-left:1.1rem">'
                        f'{_items}</ul>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="{_prose}">{paras[0]}</div>',
                        unsafe_allow_html=True)

    st.markdown(
        f'<div style="font-family:-apple-system,Segoe UI,Roboto,sans-serif;'
        f'font-size:.66rem;color:{_MUT};border-top:1px solid {_BRD};margin-top:1.2rem;'
        f'padding-top:.5rem">This brief is general market commentary for educational '
        f'purposes, prepared from public data. It is not personalised investment advice '
        f'or a recommendation to buy or sell any security. Past conditions do not predict '
        f'future results.</div>', unsafe_allow_html=True)

    # ── White-label PDF export ───────────────────────────────────────────────
    st.markdown(
        f'<div style="border-top:1px solid {_BRD};margin:1.1rem 0 .4rem;padding-top:.6rem">'
        f'<span style="font-family:JetBrains Mono,monospace;font-size:.62rem;font-weight:700;'
        f'letter-spacing:.1em;color:{_TXT}">EXPORT</span>'
        f'<span style="font-family:JetBrains Mono,monospace;font-size:.54rem;color:{_MUT};'
        f'margin-left:8px">A one-page, white-label PDF on your firm name, ready to send.</span>'
        f'</div>', unsafe_allow_html=True)
    _e1, _e2 = st.columns([1, 3], gap="medium")
    with _e1:
        _mk = st.button("Generate Client Brief (PDF)", width="stretch")
    if _mk:
        try:
            with st.spinner("Building the PDF..."):
                from src.reports.report_generator import generate_client_brief
                _pdf = generate_client_brief(
                    firm=firm or "Market & Risk Brief", client_name=client or "Valued Client",
                    prepared_by=advisor, profile=brief["profile"],
                    headline=brief["headline"], sections=brief["sections"],
                    stress=brief["stress"], regime=_RNAME[brief["regime"]],
                    as_of=brief["asof"])
            st.session_state["_cb_pdf"] = _pdf
            st.session_state["_cb_pdf_name"] = (
                (firm or "market").strip().replace(" ", "_").lower()
                + "_client_brief.pdf")
        except Exception as _e:
            st.error(f"Could not build the PDF: {_e}")
    if st.session_state.get("_cb_pdf"):
        with _e1:
            st.download_button("Download Client Brief", data=st.session_state["_cb_pdf"],
                              file_name=st.session_state.get("_cb_pdf_name",
                                                             "client_brief.pdf"),
                              mime="application/pdf", width="stretch")

    st.caption("Plain-language brief generated from live signals by a rule-based "
               "translation layer. Educational commentary, not investment advice.")
    _page_footer()
