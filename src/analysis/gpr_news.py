"""
News GPR Layer — Threat Score vs Act Score.

Extends geo_rss ingestion with fine-grained classification:

  Threat Score  — rhetoric, force build-up, warnings, military signaling,
                  sanctions threats, diplomatic breakdown, naval movements
  Act Score     — realized adverse events: strikes, explosions, sanctions
                  imposed, vessels seized, blockades, troop crossings

  News GPR = α·Act + (1−α)·Threat
  α is dynamic: rises when act headlines dominate; falls in de-escalation.

Per-conflict routing maps each headline to one or more conflict IDs from
the CONFLICTS registry so every conflict gets its own Threat/Act contribution.

Architecture:
    ingest_headlines()          (from geo_rss) →  raw GeoHeadline list
    classify_headline()         → adds {news_type, act_score, threat_score, conflicts}
    compute_news_gpr()          → portfolio-level News GPR + per-conflict breakdown
    get_news_gpr_layer()        → cached 15-min, returns full result dict

Usage:
    from src.analysis.gpr_news import get_news_gpr_layer
    result = get_news_gpr_layer()
    # result.keys(): news_gpr, threat_score, act_score, alpha,
    #                per_conflict, headlines, n_threat, n_act
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import streamlit as st

# ── Threat keyword taxonomy ───────────────────────────────────────────────────
# Words that signal future-oriented or rhetorical risk — not yet realized.

_THREAT_KEYWORDS: list[str] = [
    # Diplomatic / rhetorical escalation
    "warned", "warning", "threaten", "threatens", "threat", "ultimatum",
    "diplomatic crisis", "diplomatic breakdown", "expel", "expelled",
    "recalled ambassador", "suspend talks", "broke off",
    # Military signaling
    "military exercises", "drills", "naval exercises", "troop build",
    "troop buildup", "mobilize", "mobilization", "deploy", "deployed",
    "moving troops", "forward position", "forward deploy", "on alert",
    "raise alert", "combat readiness", "red line",
    # Sanctions threats
    "sanctions threat", "sanction threat", "considering sanctions",
    "threatened sanctions", "sanction warning", "potential sanctions",
    "could sanction", "may sanction",
    # Force signals
    "prepare strike", "preparing strike", "consider strike", "considering strike",
    "ready to act", "military option", "all options on the table",
    "could target", "possible attack", "preemptive", "deterrence",
    # Intelligence / cyber
    "cyber threat", "intelligence warning", "hacking attempt",
    "espionage", "spy",
]

# ── Act keyword taxonomy ──────────────────────────────────────────────────────
# Words that signal a realized adverse event — something already happened.

_ACT_KEYWORDS: list[str] = [
    # Kinetic events
    "attack", "attacked", "strike", "struck", "bombed", "bombing",
    "explosion", "exploded", "blast", "shelling", "shelled", "missile",
    "rocket fire", "drone strike", "airstrike", "ground offensive",
    "troops crossed", "invaded", "invasion", "captured", "seized",
    "troops entered", "soldiers killed", "civilians killed", "casualt",
    # Maritime / shipping acts
    "vessel seized", "ship seized", "tanker seized", "tanker attacked",
    "shipping disruption", "port closed", "port blocked", "canal closed",
    "blockade", "blockaded", "mine", "mines placed",
    "houthi attack", "houthi strike",
    # Sanctions imposed (realized)
    "sanctions imposed", "sanctions announced", "new sanctions",
    "sanctioned", "asset freeze", "assets frozen", "cut off swift",
    "swift ban", "export ban", "import ban", "embargo imposed",
    # Economic disruption
    "pipeline shutdown", "pipeline cut", "gas cut", "supply cut",
    "production halt", "refinery shutdown", "energy cut",
    "export halted", "border closed", "airspace closed",
    # Political acts
    "coup", "government collapsed", "president ousted", "martial law",
    "state of emergency", "ceasefire collapsed", "ceasefire violated",
]

# ── Conflict routing keywords ─────────────────────────────────────────────────
# Maps headline text → conflict IDs from CONFLICTS registry.
# Multiple conflicts can be tagged per headline.

_CONFLICT_ROUTING: dict[str, list[str]] = {
    "ukraine_russia": [
        "ukraine", "russia", "russian", "ukrainian", "putin", "zelensky",
        "donbas", "crimea", "nato", "gazprom", "nord stream", "kharkiv",
        "kyiv", "moscow", "kherson", "zaporizhzhia", "odessa",
    ],
    "red_sea_houthi": [
        "houthi", "red sea", "bab-el-mandeb", "bab el mandeb", "aden",
        "yemen", "suez transit", "gulf of aden",
    ],
    "israel_gaza": [
        "israel", "israeli", "gaza", "hamas", "west bank", "idf",
        "hezbollah", "netanyahu", "rafah", "tel aviv",
    ],
    "iran_conflict": [
        "iran", "iranian", "tehran", "hormuz", "strait of hormuz",
        "irgc", "nuclear deal", "jcpoa", "nuclear program", "enrichment",
    ],
    "india_pakistan": [
        "india pakistan", "india-pakistan", "kashmir", "loc", "line of control",
        "islamabad india", "new delhi pakistan", "indo-pak",
    ],
    "taiwan_strait": [
        "taiwan", "taiwanese", "strait of taiwan", "pla", "china military",
        "beijing military", "tsmc blockade", "chip war taiwan",
    ],
}

# ── Source credibility weights (mirrors geo_rss FEEDS) ───────────────────────

_SOURCE_WEIGHT: dict[str, float] = {
    "Reuters": 1.00,
    "FT":      0.95,
    "WSJ":     0.90,
    "BBC":     0.90,
    "AP":      0.90,
    "NYT":     0.85,
}
_DEFAULT_SOURCE_WEIGHT = 0.75

# ── Dynamic alpha parameters ──────────────────────────────────────────────────
# α = weight on Act Score in final blend; 1-α = weight on Threat Score.
# α_base: default when act and threat headlines are balanced.
# α adjusts upward when act_count / total_count is high (more realized events).

_ALPHA_BASE   = 0.55   # acts weighted slightly more than threats by default
_ALPHA_MIN    = 0.35
_ALPHA_MAX    = 0.80


# ── Enriched headline dataclass ───────────────────────────────────────────────

@dataclass
class GPRHeadline:
    """A geo_rss GeoHeadline enriched with Threat/Act classification."""
    id:           str
    title:        str
    summary:      str
    url:          str
    source:       str
    published:    str
    relevance:    float
    severity:     str
    # GPR additions
    news_type:    str            = "neutral"    # "threat" | "act" | "neutral"
    threat_score: float          = 0.0          # 0–100
    act_score:    float          = 0.0          # 0–100
    conflicts:    list[str]      = field(default_factory=list)


# ── Classification ────────────────────────────────────────────────────────────

def classify_headline(h) -> GPRHeadline:
    """
    Enrich a GeoHeadline with Threat/Act classification and conflict routing.
    `h` is a GeoHeadline (or any object with .title, .summary, .id, ... attrs).
    """
    text = (h.title + " " + h.summary).lower()
    w    = _SOURCE_WEIGHT.get(h.source, _DEFAULT_SOURCE_WEIGHT)

    # Count keyword hits
    threat_hits = sum(1 for kw in _THREAT_KEYWORDS if kw in text)
    act_hits    = sum(1 for kw in _ACT_KEYWORDS    if kw in text)

    # Severity multiplier
    sev_mult = {"high": 1.40, "medium": 1.00, "low": 0.60}.get(
        getattr(h, "severity", "low"), 1.0
    )

    # Raw scores — each keyword hit contributes; cap at 100
    raw_threat = min(threat_hits * 18 * w * sev_mult, 100.0)
    raw_act    = min(act_hits    * 22 * w * sev_mult, 100.0)

    # Determine dominant type
    if raw_act > raw_threat and raw_act >= 15:
        news_type = "act"
    elif raw_threat > raw_act and raw_threat >= 12:
        news_type = "threat"
    elif raw_act > 0 or raw_threat > 0:
        news_type = "threat"  # mixed → classify as threat (more cautious)
    else:
        news_type = "neutral"

    # Conflict routing
    conflicts: list[str] = []
    for conflict_id, keywords in _CONFLICT_ROUTING.items():
        if any(kw in text for kw in keywords):
            conflicts.append(conflict_id)

    return GPRHeadline(
        id=h.id,
        title=h.title,
        summary=getattr(h, "summary", ""),
        url=getattr(h, "url", ""),
        source=h.source,
        published=getattr(h, "published", ""),
        relevance=getattr(h, "relevance", 0.0),
        severity=getattr(h, "severity", "low"),
        news_type=news_type,
        threat_score=round(raw_threat, 1),
        act_score=round(raw_act, 1),
        conflicts=conflicts,
    )


# ── Portfolio aggregation ─────────────────────────────────────────────────────

def _compute_alpha(threat_headlines: list[GPRHeadline],
                   act_headlines:    list[GPRHeadline]) -> float:
    """
    Dynamic α: rises toward _ALPHA_MAX when acts dominate.
    α = base + 0.25 * (act_frac - 0.5) clamped to [min, max].
    """
    n_total = len(threat_headlines) + len(act_headlines)
    if n_total == 0:
        return _ALPHA_BASE

    act_frac = len(act_headlines) / n_total
    alpha = _ALPHA_BASE + 0.50 * (act_frac - 0.50)
    return float(np.clip(alpha, _ALPHA_MIN, _ALPHA_MAX))


def _aggregate_score(headlines: list[GPRHeadline],
                     score_attr: str) -> float:
    """
    Aggregate individual headline scores into a portfolio-level score.
    Uses a dampened sum to avoid over-inflating with many low-signal headlines:
        portfolio = mean * (1 + log1p(n) / 4)
    Clipped to [0, 100].
    """
    scores = [getattr(h, score_attr) for h in headlines if getattr(h, score_attr) > 0]
    if not scores:
        return 0.0
    n    = len(scores)
    mean = float(np.mean(scores))
    agg  = mean * (1.0 + np.log1p(n) / 4.0)
    return float(np.clip(agg, 0.0, 100.0))


def _per_conflict_scores(
    classified: list[GPRHeadline],
) -> dict[str, dict]:
    """
    Return per-conflict Threat and Act score contributions.
    {conflict_id: {threat, act, news_gpr, n_headlines}}
    """
    from src.data.config import CONFLICTS
    conflict_ids = [c["id"] for c in CONFLICTS]

    buckets: dict[str, list[GPRHeadline]] = {cid: [] for cid in conflict_ids}
    for h in classified:
        for cid in h.conflicts:
            if cid in buckets:
                buckets[cid].append(h)

    results: dict[str, dict] = {}
    for cid, hs in buckets.items():
        if not hs:
            results[cid] = {"threat": 0.0, "act": 0.0, "news_gpr": 0.0,
                            "n_headlines": 0}
            continue
        threat_hs = [h for h in hs if h.news_type in ("threat", "neutral")]
        act_hs    = [h for h in hs if h.news_type == "act"]
        t_score   = _aggregate_score(hs, "threat_score")
        a_score   = _aggregate_score(hs, "act_score")
        alpha     = _compute_alpha(threat_hs, act_hs)
        gpr       = alpha * a_score + (1.0 - alpha) * t_score
        results[cid] = {
            "threat":      round(t_score, 1),
            "act":         round(a_score, 1),
            "news_gpr":    round(float(np.clip(gpr, 0, 100)), 1),
            "n_headlines": len(hs),
            "alpha":       round(alpha, 3),
        }

    return results


# ── Main entry point ──────────────────────────────────────────────────────────

@st.cache_data(ttl=900, show_spinner=False)
def get_news_gpr_layer(max_headlines: int = 60) -> dict:
    """
    Fetch, classify, and aggregate the News GPR layer.
    Cached 15 minutes — same TTL as the RSS feed.

    Returns:
        news_gpr      float   0–100  portfolio-level blended GPR
        threat_score  float   0–100  threat-only aggregate
        act_score     float   0–100  act-only aggregate
        alpha         float   0–1    dynamic act weight used in blend
        per_conflict  dict           per-conflict {threat, act, news_gpr, n}
        headlines     list[GPRHeadline]  all classified & routed headlines
        n_threat      int
        n_act         int
        n_neutral     int
        n_raw         int     total headlines before classification filter
        fetched_at    str    ISO timestamp
        data_status   str    "live" | "no_feed" | "no_headlines" | "no_classified"
        feed_error    str    error message if data_status != "live", else ""
    """
    feed_error = ""
    raw: list = []

    try:
        import feedparser  # noqa: F401 — verify installed before calling
        from src.ingestion.geo_rss import ingest_headlines
        raw = ingest_headlines()
    except ImportError:
        feed_error = "feedparser not installed — run: pip install feedparser"
    except Exception as exc:
        feed_error = f"Feed ingestion error: {exc}"

    if feed_error or not raw:
        data_status = "no_feed" if feed_error or not raw else "no_headlines"
        if raw and not feed_error:
            data_status = "no_headlines"
        return {
            "news_gpr":     0.0,
            "threat_score": 0.0,
            "act_score":    0.0,
            "alpha":        float(_ALPHA_BASE),
            "per_conflict": {},
            "headlines":    [],
            "n_threat":     0,
            "n_act":        0,
            "n_neutral":    0,
            "n_raw":        0,
            "fetched_at":   datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "data_status":  "no_feed",
            "feed_error":   feed_error or "No headlines returned from feeds",
        }

    # Classify
    classified: list[GPRHeadline] = []
    seen_ids: set[str] = set()
    for h in raw[:max_headlines]:
        if h.id in seen_ids:
            continue
        seen_ids.add(h.id)
        classified.append(classify_headline(h))

    # Split by type
    threat_hs  = [h for h in classified if h.news_type == "threat"]
    act_hs     = [h for h in classified if h.news_type == "act"]
    neutral_hs = [h for h in classified if h.news_type == "neutral"]

    if not classified:
        data_status = "no_classified"
    elif not threat_hs and not act_hs:
        data_status = "no_classified"
    else:
        data_status = "live"

    # Portfolio scores
    portfolio_threat = _aggregate_score(classified, "threat_score")
    portfolio_act    = _aggregate_score(classified, "act_score")
    alpha            = _compute_alpha(threat_hs, act_hs)
    news_gpr         = float(np.clip(
        alpha * portfolio_act + (1.0 - alpha) * portfolio_threat,
        0.0, 100.0
    ))

    # Per-conflict breakdown
    per_conflict = _per_conflict_scores(classified)

    return {
        "news_gpr":     round(news_gpr, 1),
        "threat_score": round(portfolio_threat, 1),
        "act_score":    round(portfolio_act, 1),
        "alpha":        round(alpha, 3),
        "per_conflict": per_conflict,
        "headlines":    classified,
        "n_threat":     len(threat_hs),
        "n_act":        len(act_hs),
        "n_neutral":    len(neutral_hs),
        "n_raw":        len(raw),
        "fetched_at":   datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "data_status":  data_status,
        "feed_error":   "",
    }


def debug_news_gpr() -> dict:
    """
    Lightweight sanity-check — call this to manually verify the pipeline.

    Returns a plain dict with pipeline state. No Streamlit required.
    Example:
        from src.analysis.gpr_news import debug_news_gpr
        import json; print(json.dumps(debug_news_gpr(), indent=2))
    """
    report: dict = {}
    try:
        import feedparser
        report["feedparser"] = feedparser.__version__
    except ImportError:
        report["feedparser"] = "NOT INSTALLED"
        return report

    try:
        from src.ingestion.geo_rss import ingest_headlines
        raw = ingest_headlines()
        report["raw_headlines"] = len(raw)
        if raw:
            h0 = raw[0]
            report["sample_headline"] = h0.title[:80]
            report["sample_severity"] = h0.severity
            report["sample_relevance"] = h0.relevance
    except Exception as e:
        report["ingest_error"] = str(e)
        return report

    if not raw:
        report["pipeline_status"] = "no_headlines"
        return report

    classified = [classify_headline(h) for h in raw[:20]]
    n_act     = sum(1 for h in classified if h.news_type == "act")
    n_threat  = sum(1 for h in classified if h.news_type == "threat")
    n_neutral = sum(1 for h in classified if h.news_type == "neutral")
    report["classified_20"] = {"act": n_act, "threat": n_threat, "neutral": n_neutral}

    if n_act + n_threat == 0:
        report["pipeline_status"] = "no_classified"
        return report

    result = get_news_gpr_layer()
    report["pipeline_status"] = result["data_status"]
    report["news_gpr"]     = result["news_gpr"]
    report["threat_score"] = result["threat_score"]
    report["act_score"]    = result["act_score"]
    report["alpha"]        = result["alpha"]
    report["n_threat"]     = result["n_threat"]
    report["n_act"]        = result["n_act"]
    report["fetched_at"]   = result["fetched_at"]
    return report


# ── Convenience helpers ───────────────────────────────────────────────────────

def get_conflict_news_gpr(conflict_id: str) -> float:
    """Return the News GPR contribution for a single conflict (0–100)."""
    result = get_news_gpr_layer()
    return result["per_conflict"].get(conflict_id, {}).get("news_gpr", 0.0)


def get_top_headlines(
    news_type: Optional[str] = None,
    conflict_id: Optional[str] = None,
    n: int = 10,
) -> list[GPRHeadline]:
    """
    Filtered headline list.
    news_type: "threat" | "act" | "neutral" | None (all)
    conflict_id: filter by routed conflict; None returns all
    """
    result = get_news_gpr_layer()
    hs = result["headlines"]

    if news_type:
        hs = [h for h in hs if h.news_type == news_type]
    if conflict_id:
        hs = [h for h in hs if conflict_id in h.conflicts]

    # Sort by act_score desc for acts, threat_score desc for threats, relevance otherwise
    if news_type == "act":
        hs.sort(key=lambda h: h.act_score,    reverse=True)
    elif news_type == "threat":
        hs.sort(key=lambda h: h.threat_score, reverse=True)
    else:
        hs.sort(key=lambda h: h.relevance,    reverse=True)

    return hs[:n]


# ── Streamlit render helper ───────────────────────────────────────────────────

def render_threat_act_feed(
    news_type: Optional[str] = None,
    conflict_id: Optional[str] = None,
    max_items: int = 8,
) -> None:
    """
    Render a compact Threat / Act classified headline feed.
    news_type: "threat" | "act" | None (all)
    """
    _TYPE_COLOR  = {"act": "#c0392b", "threat": "#e67e22", "neutral": "#555960"}
    _TYPE_LABEL  = {"act": "ACT",     "threat": "THREAT",  "neutral": "INFO"}
    _SEV_COLOR   = {"high": "#c0392b", "medium": "#e67e22", "low": "#555960"}

    headlines = get_top_headlines(news_type=news_type,
                                  conflict_id=conflict_id, n=max_items)

    if not headlines:
        st.markdown(
            '<p style="font-family:\'JetBrains Mono\',monospace;font-size:9px;'
            'color:#555960">No headlines available.</p>',
            unsafe_allow_html=True,
        )
        return

    rows_html = ""
    for h in headlines:
        type_color = _TYPE_COLOR.get(h.news_type, "#555960")
        type_label = _TYPE_LABEL.get(h.news_type, "INFO")
        sev_color  = _SEV_COLOR.get(h.severity, "#555960")
        conflict_tags = "".join([
            f'<span style="font-size:8px;background:#0d1117;border:1px solid #2a2a2a;'
            f'padding:1px 4px;margin-right:3px;color:#8E9AAA">{cid.replace("_"," ").upper()}</span>'
            for cid in h.conflicts[:2]
        ])
        url_attr = f'href="{h.url}" target="_blank"' if h.url else ""
        score_val = h.act_score if h.news_type == "act" else h.threat_score
        rows_html += (
            f'<div style="border-bottom:1px solid #1a1a1a;padding:5px 0;">'
            f'<div style="display:flex;align-items:flex-start;gap:6px;">'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:7px;'
            f'font-weight:700;color:{type_color};border:1px solid {type_color};'
            f'padding:1px 4px;flex-shrink:0;margin-top:1px">{type_label}</span>'
            f'<div style="flex:1;min-width:0">'
            f'<a {url_attr} style="font-family:\'DM Sans\',sans-serif;font-size:10px;'
            f'font-weight:500;color:#e8e9ed;text-decoration:none;'
            f'display:block;line-height:1.4">{h.title}</a>'
            f'<div style="margin-top:2px;display:flex;align-items:center;gap:4px;">'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:7px;'
            f'color:#8E9AAA">{h.source}</span>'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:7px;'
            f'color:{sev_color}">{h.severity.upper()}</span>'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:7px;'
            f'color:{type_color}">{score_val:.0f}</span>'
            f'{conflict_tags}'
            f'</div>'
            f'</div></div></div>'
        )

    result = get_news_gpr_layer()
    alpha  = result["alpha"]
    n_act  = result["n_act"]
    n_thr  = result["n_threat"]

    st.markdown(
        f'<div style="background:#0d0d0d;border:1px solid #222;padding:4px 10px 6px">'
        f'{rows_html}'
        f'<div style="margin-top:6px;padding-top:4px;border-top:1px solid #1a1a1a;'
        f'display:flex;gap:12px">'
        f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:7px;'
        f'color:#c0392b">ACT {n_act}</span>'
        f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:7px;'
        f'color:#e67e22">THREAT {n_thr}</span>'
        f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:7px;'
        f'color:#8E9AAA">α={alpha:.2f}</span>'
        f'</div></div>',
        unsafe_allow_html=True,
    )
