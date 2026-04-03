"""
Lightweight geopolitical RSS ingestion.

Replaces manual event maintenance with automated headline scraping.
Architecture:
  1. Fetch RSS feeds (Reuters, AP, BBC geopolitics, FT Energy)
  2. Keyword-tag each headline by region, commodity impact, and severity
  3. Score relevance (0–100) based on keyword density + source weight
  4. Route high-confidence events to agent_state pending-review queue
  5. Low-confidence events stored in session_state for human review

Tuned for: oil/gas, gold, strait disruptions, sanctions, central-bank actions,
           geopolitical escalation, supply chain.

Usage:
    from src.ingestion.geo_rss import ingest_headlines, get_pending_headlines
    headlines = ingest_headlines()          # fetch + tag (cached 15 min)
    get_pending_headlines()                 # return unreviewed items from session
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from typing import Optional
import streamlit as st

# ── Feed registry ─────────────────────────────────────────────────────────────
# Free RSS feeds that reliably carry geopolitical/market-relevant headlines.
# Weights 1.0 = most credible; 0.7 = secondary.

FEEDS: list[dict] = [
    {"url": "https://feeds.reuters.com/reuters/businessNews",    "source": "Reuters",  "weight": 1.0},
    {"url": "https://feeds.reuters.com/reuters/worldNews",       "source": "Reuters",  "weight": 1.0},
    {"url": "https://feeds.bbci.co.uk/news/world/rss.xml",       "source": "BBC",      "weight": 0.9},
    {"url": "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",  "source": "NYT","weight": 0.85},
    {"url": "https://feeds.a.dj.com/rss/RSSWorldNews.xml",       "source": "WSJ",      "weight": 0.9},
    {"url": "https://www.ft.com/rss/home",                       "source": "FT",       "weight": 0.95},
    {"url": "https://apnews.com/rss/world-news",                 "source": "AP",       "weight": 0.9},
]

# ── Taxonomy ──────────────────────────────────────────────────────────────────

_REGION_TAGS: dict[str, list[str]] = {
    "Middle East": ["iran", "iraq", "saudi", "opec", "gulf", "israel", "hamas",
                    "hezbollah", "yemen", "houthi", "hormuz", "oman"],
    "Russia/Ukraine": ["russia", "ukraine", "putin", "zelensky", "nato", "crimea",
                       "donbas", "sanctions", "gazprom", "nord stream"],
    "China/Taiwan": ["china", "taiwan", "beijing", "xi jinping", "pla", "strait of taiwan",
                     "south china sea", "hong kong"],
    "North Africa": ["libya", "egypt", "suez", "algeria", "morocco", "africa"],
    "Asia Pacific": ["india", "pakistan", "korea", "japan", "asean", "philippines"],
    "Americas":     ["venezuela", "mexico", "colombia", "brazil", "latam", "caribbean"],
    "Europe":       ["germany", "france", "uk", "ecb", "eu", "european union", "boe"],
    "Global":       ["g7", "g20", "imf", "world bank", "wto", "un security council"],
}

_COMMODITY_TAGS: dict[str, list[str]] = {
    "oil":     ["oil", "crude", "brent", "wti", "opec", "barrel", "petroleum",
                "refinery", "pipeline", "lng", "gas field"],
    "gas":     ["natural gas", "lng", "pipeline", "gas supply", "energy crisis"],
    "gold":    ["gold", "safe haven", "precious metals", "bullion"],
    "copper":  ["copper", "metals", "mining", "chile", "congo"],
    "wheat":   ["wheat", "grain", "food supply", "black sea", "harvest"],
    "shipping":["shipping", "freight", "vessel", "tanker", "chokepoint",
                "strait", "canal", "suez", "panama", "hormuz", "bosporus"],
}

_SEVERITY_HIGH: list[str] = [
    "attack", "strike", "bomb", "war", "invasion", "conflict", "crisis",
    "collapse", "sanctions", "blockade", "seized", "explosion", "missile",
    "nuclear", "shutdown", "disruption", "coup", "default",
]
_SEVERITY_MED: list[str] = [
    "tension", "threat", "escalate", "protest", "election", "vote",
    "central bank", "rate decision", "interest rate", "inflation",
    "embargo", "tariff", "supply cut", "halt",
]
_NOISE: list[str] = [
    "sports", "celebrity", "entertainment", "weather", "health", "covid",
    "vaccine", "recipe", "travel guide", "fashion",
]

# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class GeoHeadline:
    id:             str             # SHA-1 of url+title
    title:          str
    summary:        str
    url:            str
    source:         str
    published:      str             # ISO string or feed date
    regions:        list[str]       = field(default_factory=list)
    commodities:    list[str]       = field(default_factory=list)
    severity:       str             = "low"      # "high" | "medium" | "low"
    relevance:      float           = 0.0        # 0–100
    auto_publish:   bool            = False      # True if above auto-threshold
    reviewed:       bool            = False
    decision:       Optional[str]   = None       # "approved" | "rejected"


def _score_headline(title: str, summary: str, source_weight: float) -> tuple[
    float, list[str], list[str], str
]:
    """Score relevance and tag a headline. Returns (score, regions, commodities, severity)."""
    text = (title + " " + summary).lower()

    # Noise filter
    if any(n in text for n in _NOISE):
        return 0.0, [], [], "low"

    regions: list[str] = []
    for region, kws in _REGION_TAGS.items():
        if any(k in text for k in kws):
            regions.append(region)

    commodities: list[str] = []
    for commodity, kws in _COMMODITY_TAGS.items():
        if any(k in text for k in kws):
            commodities.append(commodity)

    # Not relevant if no regional or commodity match
    if not regions and not commodities:
        return 0.0, [], [], "low"

    # Severity
    if any(s in text for s in _SEVERITY_HIGH):
        severity = "high"
        sev_score = 40
    elif any(s in text for s in _SEVERITY_MED):
        severity = "medium"
        sev_score = 20
    else:
        severity = "low"
        sev_score = 5

    # Base relevance
    base = (len(regions) * 10 + len(commodities) * 12 + sev_score) * source_weight
    score = min(base, 100.0)

    return score, regions, commodities, severity


def _make_id(url: str, title: str) -> str:
    return hashlib.sha1(f"{url}:{title}".encode()).hexdigest()[:12]


# ── Ingestion ─────────────────────────────────────────────────────────────────

@st.cache_data(ttl=900, show_spinner=False)
def ingest_headlines(
    auto_publish_threshold: float = 65.0,
    max_per_feed: int = 20,
) -> list[GeoHeadline]:
    """
    Fetch all feeds, tag, and score.
    Headlines above auto_publish_threshold are flagged for immediate review.
    Cached 15 minutes.
    Returns list of GeoHeadline sorted by relevance desc.
    """
    try:
        import feedparser
    except ImportError:
        return []

    results: list[GeoHeadline] = []
    seen: set[str] = set()

    for feed_def in FEEDS:
        try:
            parsed = feedparser.parse(feed_def["url"])
        except Exception:
            continue

        entries = parsed.get("entries", [])[:max_per_feed]
        for entry in entries:
            title   = entry.get("title",   "")
            summary = entry.get("summary", "") or entry.get("description", "")
            url     = entry.get("link",    "")
            pub     = entry.get("published", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

            hid = _make_id(url, title)
            if hid in seen:
                continue
            seen.add(hid)

            score, regions, commodities, severity = _score_headline(
                title, summary, feed_def["weight"]
            )
            if score < 10:
                continue

            # Clean summary
            summary_clean = re.sub(r"<[^>]+>", "", summary).strip()[:300]

            results.append(GeoHeadline(
                id=hid, title=title, summary=summary_clean,
                url=url, source=feed_def["source"], published=pub,
                regions=regions, commodities=commodities,
                severity=severity, relevance=score,
                auto_publish=(score >= auto_publish_threshold),
            ))

    results.sort(key=lambda h: h.relevance, reverse=True)
    return results


def route_to_review_queue(
    headlines: list[GeoHeadline],
    max_items: int = 10,
) -> int:
    """
    Push high-relevance unreviewed headlines into agent_state pending review queue.
    Returns count of items routed.
    """
    try:
        from src.analysis.agent_state import add_pending, init_agents
        init_agents()
    except Exception:
        return 0

    # Track already-routed IDs in session_state to avoid duplicates
    routed: set[str] = st.session_state.setdefault("_geo_rss_routed", set())
    count = 0

    for h in headlines[:max_items]:
        if not h.auto_publish:
            continue
        if h.id in routed:
            continue

        severity_map = {"high": "critical", "medium": "warning", "low": "info"}
        region_str = ", ".join(h.regions) if h.regions else "Global"
        commodity_str = ", ".join(h.commodities) if h.commodities else "general"

        try:
            add_pending(
                agent_id   = "geopolitical_analyst",
                title      = h.title,
                summary    = (
                    f"[{h.source}] {h.summary[:200]}"
                    if h.summary else f"[{h.source}] No summary available."
                ),
                rationale  = (
                    f"Regions: {region_str}. "
                    f"Commodities affected: {commodity_str}. "
                    f"Severity: {h.severity}. "
                    f"Relevance score: {h.relevance:.0f}/100. "
                    f"Auto-flagged by RSS ingestion engine."
                ),
                confidence = h.relevance / 100,
                severity   = severity_map.get(h.severity, "info"),
                extra      = {
                    "source":     h.source,
                    "url":        h.url,
                    "published":  h.published,
                    "regions":    h.regions,
                    "commodities": h.commodities,
                    "rss_id":     h.id,
                },
            )
            routed.add(h.id)
            count += 1
        except Exception:
            pass

    return count


def get_pending_headlines() -> list[GeoHeadline]:
    """Return unreviewed high-relevance headlines from last ingestion run."""
    try:
        cached = ingest_headlines()
        return [h for h in cached if h.auto_publish and not h.reviewed]
    except Exception:
        return []


def render_rss_panel(max_items: int = 8) -> None:
    """
    Render a compact RSS headline panel for the Geopolitical page.
    Shows headline, source badge, region/commodity tags, severity indicator.
    Calls route_to_review_queue() automatically.
    """
    headlines = ingest_headlines()
    if not headlines:
        st.markdown(
            '<p style="font-size:0.70rem;color:#8890a1">RSS feed unavailable - '
            'install feedparser: <code>pip install feedparser</code></p>',
            unsafe_allow_html=True,
        )
        return

    # Auto-route high-relevance items
    n_routed = route_to_review_queue(headlines)
    if n_routed > 0:
        st.markdown(
            f'<p style="font-size:0.62rem;color:#27ae60;margin:0 0 0.5rem">'
            f'&#9679; {n_routed} headline{"s" if n_routed > 1 else ""} routed to Pending Review</p>',
            unsafe_allow_html=True,
        )

    _SEV_COLOR = {"high": "#c0392b", "medium": "#e67e22", "low": "#555960"}
    _SEV_LABEL = {"high": "HIGH", "medium": "MED", "low": "LOW"}

    rows_html = ""
    for h in headlines[:max_items]:
        sev_color = _SEV_COLOR.get(h.severity, "#555960")
        sev_label = _SEV_LABEL.get(h.severity, "LOW")
        region_tags = "".join([
            f'<span style="font-size:0.55rem;background:#1e1e1e;border:1px solid #2a2a2a;'
            f'border-radius:2px;padding:1px 5px;margin:0 3px 0 0;color:#8890a1">{r}</span>'
            for r in h.regions[:2]
        ])
        cmd_tags = "".join([
            f'<span style="font-size:0.55rem;background:#1e1a12;border:1px solid #3a2e10;'
            f'border-radius:2px;padding:1px 5px;margin:0 3px 0 0;color:#CFB991">{c}</span>'
            for c in h.commodities[:2]
        ])
        url_attr = f'href="{h.url}" target="_blank"' if h.url else ""
        rows_html += (
            f'<div style="border-bottom:1px solid #1e1e1e;padding:0.55rem 0;">'
            f'<div style="display:flex;align-items:flex-start;gap:0.6rem">'
            f'<span style="font-size:0.55rem;font-weight:700;color:{sev_color};'
            f'background:rgba(0,0,0,0.3);border:1px solid {sev_color};'
            f'border-radius:2px;padding:1px 5px;flex-shrink:0;margin-top:1px">{sev_label}</span>'
            f'<div>'
            f'<a {url_attr} style="font-size:0.72rem;font-weight:500;color:#e8e9ed;'
            f'text-decoration:none;line-height:1.4;display:block">{h.title}</a>'
            f'<div style="margin-top:3px">'
            f'<span style="font-size:0.58rem;color:#8890a1;margin-right:6px">{h.source}</span>'
            f'{region_tags}{cmd_tags}'
            f'</div>'
            f'</div></div></div>'
        )

    st.markdown(
        f'<div style="background:#111;border:1px solid #2a2a2a;border-radius:0;'
        f'padding:0.2rem 0.8rem 0.4rem">{rows_html}</div>',
        unsafe_allow_html=True,
    )
