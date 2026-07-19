"""
Signals Export - the terminal's headline signals as a downloadable feed.

The export slice of the data-feed product (persona 4: geopolitical / API shops).
It serves exactly what src/analysis/signals_feed.build_signals_payload() returns:
CIS / TPS per conflict and aggregate, Diebold-Yilmaz total connectedness, the
stress score and correlation regime, and a per-commodity Geopolitical Stress Index,
downloadable as versioned JSON or tidy CSV, with a documented schema. A live HTTP
endpoint (GET /v1/signals) is the roadmap; the payload is identical, so the endpoint
is a thin wrapper over the same function, not a rewrite.
"""
from __future__ import annotations

import json as _json

import pandas as pd
import streamlit as st

from src.ui.shared import (_page_header, _page_footer, _page_intro,
                           _definition_block)
from src.analysis.signals_feed import (build_signals_payload, payload_to_long_rows,
                                       FIELD_DOCS, SCHEMA)

_M = "font-family:'JetBrains Mono',monospace;"
_GOLD, _TXT, _MUT, _FNT, _BRD = "#CFB991", "#e8e9ed", "#8890a1", "#555960", "#1e1e1e"


@st.cache_data(show_spinner=False, ttl=900, max_entries=4)
def _cached_payload(start: str, end: str) -> dict:
    return build_signals_payload(start, end)


def _stat(lbl, val, sub, col=_TXT, last=False):
    br = "" if last else f"border-right:1px solid {_BRD}"
    return (f'<div style="flex:1;padding:.45rem .7rem;{br}">'
            f'<div style="{_M}font-size:.5rem;letter-spacing:.1em;color:{_MUT}">{lbl}</div>'
            f'<div style="{_M}font-size:1.05rem;font-weight:700;color:{col};margin:1px 0">{val}</div>'
            f'<div style="{_M}font-size:.48rem;color:{_FNT}">{sub}</div></div>')


def page_signals_export(start: str, end: str, fred_key: str = "") -> None:
    _page_header("Signals Export",
                 "The terminal's headline signals as a versioned, documented feed")
    _page_intro(
        "A machine-readable feed of the signals the rest of the terminal computes: "
        "Conflict Intensity and Transmission Propensity per conflict and in aggregate, "
        "the Diebold-Yilmaz total connectedness index, the cross-asset stress score and "
        "correlation regime, and a per-commodity Geopolitical Stress Index. Download it "
        "as versioned JSON or tidy CSV against a documented schema. This is the export "
        "slice of the data-feed product; the payload is produced by a single pure "
        "function, so a live <code>GET /v1/signals</code> endpoint would return the "
        "identical object."
    )
    _definition_block(
        "Honest scope",
        "This ships the feed as a <b>download</b>, not a live HTTP service. Real-time "
        "serving needs an API gateway, authentication and rate limiting (the accounts / "
        "infrastructure work), and institutional consumers will also require licensed "
        "data in place of the current public sources. The schema and the assembly "
        "function are built so that endpoint is a thin wrapper, not a rewrite."
    )

    with st.spinner("Assembling the signals payload..."):
        payload = _cached_payload(start, end)

    mk = payload.get("market") or {}
    cf = payload.get("conflict") or {}
    gsi = payload.get("commodity_gsi") or []

    # ── Summary strip ────────────────────────────────────────────────────────
    _conn = mk.get("dy_total_connectedness_pct")
    _cis = cf.get("portfolio_cis")
    st.markdown(
        f'<div style="border:1px solid {_BRD};background:#0a0a0a;margin:.3rem 0 .5rem">'
        f'<div style="display:flex;justify-content:space-between;align-items:baseline;'
        f'padding:.4rem .8rem;border-bottom:1px solid {_BRD}">'
        f'<span style="{_M}font-size:.6rem;font-weight:700;letter-spacing:.14em;color:{_TXT}">'
        f'FEED SNAPSHOT</span><span style="{_M}font-size:.5rem;color:{_MUT}">'
        f'{payload.get("schema")} &middot; data to {payload.get("as_of_data") or "n/a"}'
        f'<span style="background:{_GOLD};color:#000;font-weight:700;padding:1px 7px;'
        f'margin-left:6px;letter-spacing:.08em">DOWNLOAD FEED</span></span></div>'
        f'<div style="display:flex;border-bottom:1px solid {_BRD}">'
        + _stat("STRESS", f'{mk.get("stress_score","n/a")}',
                f'{mk.get("regime_label","")} regime')
        + _stat("DY CONNECTEDNESS", f'{_conn if _conn is not None else "n/a"}%',
                f'{mk.get("dy_universe_n","?")} assets')
        + _stat("PORTFOLIO CIS", f'{_cis if _cis is not None else "n/a"}',
                f'TPS {cf.get("portfolio_tps","n/a")}')
        + _stat("CONFLICTS", f'{cf.get("n_active","n/a")}',
                f'{len(cf.get("conflicts",[]))} tracked')
        + _stat("COMMODITY GSI", f'{len(gsi)}',
                'commodities scored', last=True)
        + '</div></div>',
        unsafe_allow_html=True)

    # ── Downloads ────────────────────────────────────────────────────────────
    _json_str = _json.dumps(payload, indent=2, default=str)
    _rows = payload_to_long_rows(payload)
    _csv_str = pd.DataFrame(_rows).to_csv(index=False) if _rows else ""
    _asof = payload.get("as_of_data") or "latest"

    d1, d2, _sp = st.columns([1, 1, 2], gap="medium")
    with d1:
        st.download_button("Download JSON", data=_json_str,
                          file_name=f"spillover_signals_{_asof}.json",
                          mime="application/json", width="stretch")
    with d2:
        st.download_button("Download CSV", data=_csv_str,
                          file_name=f"spillover_signals_{_asof}.csv",
                          mime="text/csv", width="stretch",
                          disabled=not _csv_str)

    # ── Payload preview ──────────────────────────────────────────────────────
    with st.expander("Preview payload (JSON)", expanded=False):
        st.json(payload)
    with st.expander(f"Preview tidy rows (CSV, {len(_rows)} rows)", expanded=False):
        if _rows:
            st.dataframe(pd.DataFrame(_rows), width="stretch", height=280)

    # ── Schema documentation ─────────────────────────────────────────────────
    st.markdown(
        f'<div style="{_M}font-size:.62rem;font-weight:700;letter-spacing:.1em;'
        f'color:{_TXT};margin:1rem 0 .2rem">SCHEMA &middot; {SCHEMA}</div>',
        unsafe_allow_html=True)
    _doc_rows = ""
    for field, rng, desc in FIELD_DOCS:
        _doc_rows += (
            f'<tr style="border-top:1px solid {_BRD}">'
            f'<td style="{_M}font-size:.56rem;color:{_GOLD};padding:3px 8px;white-space:nowrap">{field}</td>'
            f'<td style="{_M}font-size:.54rem;color:{_MUT};padding:3px 8px;white-space:nowrap">{rng}</td>'
            f'<td style="{_M}font-size:.54rem;color:#c9ccd4;padding:3px 8px">{desc}</td></tr>')
    st.markdown(
        f'<div style="border:1px solid {_BRD};background:#0a0a0a;overflow-x:auto">'
        f'<table style="width:100%;border-collapse:collapse">'
        f'<tr style="background:#141414"><td style="{_M}font-size:.5rem;font-weight:700;'
        f'letter-spacing:.08em;color:{_MUT};padding:4px 8px">FIELD</td>'
        f'<td style="{_M}font-size:.5rem;font-weight:700;letter-spacing:.08em;color:{_MUT};padding:4px 8px">RANGE</td>'
        f'<td style="{_M}font-size:.5rem;font-weight:700;letter-spacing:.08em;color:{_MUT};padding:4px 8px">DESCRIPTION</td></tr>'
        f'{_doc_rows}</table></div>',
        unsafe_allow_html=True)

    # ── Future endpoint contract (honest illustration) ───────────────────────
    st.markdown(
        f'<div style="{_M}font-size:.62rem;font-weight:700;letter-spacing:.1em;'
        f'color:{_TXT};margin:1.1rem 0 .2rem">API CONTRACT (ROADMAP)</div>'
        f'<div style="{_M}font-size:.54rem;color:{_MUT};margin-bottom:.3rem">The same '
        f'payload, served over HTTP once the gateway and auth exist. Shown so a consumer '
        f'can build against the contract today.</div>', unsafe_allow_html=True)
    _sample = (
        "GET /v1/signals?start=2018-01-01&end=" + str(end) + "\n"
        "Authorization: Bearer <api-key>          # roadmap: not enforced today\n"
        "Accept: application/json\n\n"
        "200 OK  application/json\n"
        + _json.dumps({k: payload.get(k) for k in
                       ("schema", "generated_utc", "as_of_data", "market")},
                      indent=2, default=str)
        + "\n  ... (conflict, commodity_gsi omitted for brevity)")
    st.code(_sample, language="http")

    st.caption("Signals Export serves src/analysis/signals_feed.build_signals_payload "
               "verbatim. Public-data provenance; research and educational use, not "
               "investment advice or a licensed institutional feed.")
    _page_footer()
