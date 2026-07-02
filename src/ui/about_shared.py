"""
Shared CSS, layout constants, and HTML helpers for all About / Bio pages.

All three bio pages (heramb, ilian, jiahe) import _ABOUT_STYLE and the helper
functions from here so that typography, spacing, and colour are pixel-identical
across every profile page.

Typography scale mirrors the global design system (shared.py + palette.py):
  0.44rem  JetBrains Mono  stat sub-labels
  0.50rem  JetBrains Mono  uppercase section labels, meta dates, tags, links
  0.53rem  JetBrains Mono  sub-lines (degree, journal, cert issuer)
  0.56rem  JetBrains Mono  company names in experience blocks
  0.63rem  DM Sans         dept / secondary data
  0.70rem  DM Sans         body text, publication title, cert name (matches shared._page_intro)
  0.72rem  DM Sans bold    job title
  0.75rem  DM Sans bold    school name
  1.25rem  JetBrains Mono  hero name
  1.38rem  JetBrains Mono  stat numbers

Colours: palette.py — GOLD #CFB991 · TEXT #e8e9ed · TEXT_SOFT #c8c8c8
         TEXT_MUTED #b8b8b8 · LABEL #8890a1 · TICK #555960
         BG #000000 · CARD #0d0d0d · BORDER #1e1e1e
"""

from __future__ import annotations

import base64
from pathlib import Path

_ASSETS = Path(__file__).resolve().parents[2] / "assets"

# ── Shared CSS ────────────────────────────────────────────────────────────────

_ABOUT_STYLE = """<style>
/* ── About Pages — shared design system ──────────────────────────────────────
   Single source of truth. Import via _ABOUT_STYLE in each bio page.
──────────────────────────────────────────────────────────────────────────── */

/* Section label — gold uppercase with bottom rule */
.abt-label{
  font-family:'JetBrains Mono',monospace!important;
  font-size:0.50rem!important;font-weight:700!important;
  text-transform:uppercase;letter-spacing:.20em;
  color:#CFB991!important;display:block;
  border-bottom:1px solid #1e1e1e;padding-bottom:5px;margin-bottom:10px;
}

/* Hero: full name */
.abt-name{
  font-family:'JetBrains Mono',monospace!important;
  font-size:1.25rem!important;font-weight:700!important;
  color:#e8e9ed!important;letter-spacing:-.02em;line-height:1.1;display:block;
}

/* Hero: subtitle (role · school · location) */
.abt-sub{
  font-family:'JetBrains Mono',monospace!important;
  font-size:0.53rem!important;color:#CFB991!important;
  font-weight:600;line-height:1.7;letter-spacing:.08em;display:block;
}

/* Hero + body: plain prose */
.abt-tgln{
  font-family:'DM Sans',sans-serif!important;
  font-size:0.70rem!important;color:#b8b8b8!important;line-height:1.72;display:block;
}
.abt-body{
  font-family:'DM Sans',sans-serif!important;
  font-size:0.70rem!important;color:#b8b8b8!important;line-height:1.78;
}

/* Experience: job title */
.abt-role{
  font-family:'DM Sans',sans-serif!important;
  font-size:0.72rem!important;font-weight:700!important;
  color:#e8e9ed!important;line-height:1.3;display:block;
}

/* Experience: company / org */
.abt-org{
  font-family:'JetBrains Mono',monospace!important;
  font-size:0.56rem!important;font-weight:600!important;
  color:#CFB991!important;display:block;letter-spacing:.04em;
}

/* Dates, metadata */
.abt-meta{
  font-family:'JetBrains Mono',monospace!important;
  font-size:0.50rem!important;color:#8890a1!important;display:block;
}

/* Education: school name */
.abt-sch{
  font-family:'DM Sans',sans-serif!important;
  font-size:0.75rem!important;font-weight:700!important;color:#e8e9ed!important;display:block;
}
/* Education: department */
.abt-dept{
  font-family:'DM Sans',sans-serif!important;
  font-size:0.63rem!important;color:#b8b8b8!important;display:block;
}
/* Education: degree */
.abt-deg{
  font-family:'JetBrains Mono',monospace!important;
  font-size:0.53rem!important;color:#CFB991!important;
  font-weight:600;letter-spacing:.04em;display:block;
}
/* Education: year range */
.abt-year{
  font-family:'JetBrains Mono',monospace!important;
  font-size:0.50rem!important;color:#8890a1!important;display:block;
}

/* CTA buttons */
.abt-link{
  font-family:'JetBrains Mono',monospace!important;
  font-size:0.50rem!important;font-weight:700!important;
  color:#080808!important;text-decoration:none!important;
  letter-spacing:.14em;text-transform:uppercase;
  background:#CFB991;padding:3px 10px;margin-right:6px;
  display:inline-block;transition:opacity .15s;
}
.abt-link:hover{opacity:.85!important;}
.abt-link-ghost{
  font-family:'JetBrains Mono',monospace!important;
  font-size:0.50rem!important;font-weight:700!important;
  color:#CFB991!important;text-decoration:none!important;
  letter-spacing:.14em;text-transform:uppercase;
  border:1px solid rgba(207,185,145,.35);padding:2px 9px;
  margin-right:6px;display:inline-block;
}
.abt-link-ghost:hover{border-color:#CFB991!important;}

/* Stat row */
.abt-num{
  font-family:'JetBrains Mono',monospace!important;
  font-size:1.38rem!important;font-weight:700!important;
  color:#CFB991!important;line-height:1;display:block;
}
.abt-slbl{
  font-family:'JetBrains Mono',monospace!important;
  font-size:0.50rem!important;text-transform:uppercase;
  letter-spacing:.14em;color:#8890a1!important;display:block;margin-top:4px;
}

/* Interest / skill tags */
.abt-tag-g{
  display:inline-block;padding:2px 9px;
  font-family:'JetBrains Mono',monospace!important;
  font-size:0.50rem!important;font-weight:700!important;
  margin:2px;background:rgba(207,185,145,.08);color:#CFB991!important;
  border:1px solid rgba(207,185,145,.25);letter-spacing:.06em;text-transform:uppercase;
}
.abt-tag-n{
  display:inline-block;padding:2px 9px;
  font-family:'JetBrains Mono',monospace!important;
  font-size:0.50rem!important;font-weight:600!important;
  margin:2px;background:transparent;color:#8890a1!important;
  border:1px solid #222;letter-spacing:.06em;text-transform:uppercase;
}

/* Publication */
.abt-pub-title{
  font-family:'DM Sans',sans-serif!important;
  font-size:0.70rem!important;font-weight:600!important;
  color:#e8e9ed!important;line-height:1.55;display:block;
}
.abt-pub-auth{
  font-family:'DM Sans',sans-serif!important;
  font-size:0.63rem!important;color:#b8b8b8!important;display:block;
}
.abt-pub-journal{
  font-family:'JetBrains Mono',monospace!important;
  font-size:0.53rem!important;color:#CFB991!important;display:block;
}

/* Certifications */
.abt-cert-name{
  font-family:'DM Sans',sans-serif!important;
  font-size:0.70rem!important;font-weight:600!important;color:#e8e9ed!important;display:block;
}
.abt-cert-issuer{
  font-family:'JetBrains Mono',monospace!important;
  font-size:0.50rem!important;color:#8890a1!important;display:block;
}

/* Acknowledgments */
.abt-ack-name{
  font-family:'DM Sans',sans-serif!important;
  font-size:0.70rem!important;font-weight:700!important;
  color:#e8e9ed!important;display:block;margin-bottom:2px;
}
.abt-ack-note{
  font-family:'JetBrains Mono',monospace!important;
  font-size:0.50rem!important;color:#8890a1!important;display:block;
}
</style>"""

# ── Layout constants ──────────────────────────────────────────────────────────

_SEC  = "border-top:1px solid #1e1e1e;padding:0.85rem 0 0.35rem;margin-bottom:0.15rem;"
_EXPI = ("padding:0.55rem 0 0.55rem 0.9rem;"
         "border-left:2px solid rgba(207,185,145,.25);"
         "margin-bottom:0.5rem;")
_EDUI = "padding:0.45rem 0;border-bottom:1px solid #161616;"
_PUB  = ("background:#080808;border:1px solid #1e1e1e;"
         "border-left:2px solid rgba(207,185,145,.40);padding:0.6rem 0.8rem;")
_ACK  = ("background:#080808;border:1px solid #1e1e1e;"
         "border-left:2px solid rgba(207,185,145,.35);padding:0.55rem 0.8rem;margin-bottom:6px;")


# ── HTML helpers ──────────────────────────────────────────────────────────────

def _photo_html(filename: str, alt: str, width: int = 150) -> str:
    """Embed photo as base64 with consistent crop and subtle desaturation."""
    p = _ASSETS / filename
    if not p.exists():
        return ""
    b64 = base64.b64encode(p.read_bytes()).decode()
    ext = p.suffix.lstrip(".")
    return (
        f"<div style='flex-shrink:0;width:{width}px;overflow:hidden;"
        f"border-right:1px solid #1e1e1e;'>"
        f"<img src='data:image/{ext};base64,{b64}' alt='{alt}' "
        f"style='width:100%;height:100%;object-fit:cover;"
        f"object-position:top center;display:block;filter:grayscale(12%);'/>"
        f"</div>"
    )


def _hero(photo_html: str, role_lbl: str, name: str, sub: str,
          tagline: str, links_html: str) -> str:
    """Full-width hero card with photo, name, sub-line, tagline, and CTA buttons."""
    return (
        f"<div style='background:#080808;border:1px solid #1e1e1e;"
        f"border-top:3px solid #CFB991;overflow:hidden;margin-bottom:1.4rem;'>"
        f"<div style='display:flex;align-items:stretch;min-height:180px;'>"
        f"{photo_html}"
        f"<div style='flex:1;padding:1.1rem 1.4rem;display:flex;flex-direction:column;"
        f"justify-content:center;border-left:1px solid #1e1e1e;'>"
        f"<span class='abt-meta' style='margin:0 0 8px;letter-spacing:.20em;'>{role_lbl}</span>"
        f"<span class='abt-name' style='margin:0 0 6px;'>{name}</span>"
        f"<span class='abt-sub'  style='margin:0 0 10px;'>{sub}</span>"
        f"<span class='abt-tgln' style='margin:0 0 14px;'>{tagline}</span>"
        f"<div style='display:flex;flex-wrap:wrap;gap:0;align-items:center;'>"
        f"{links_html}</div>"
        f"</div></div></div>"
    )


def _exp(role: str, org: str, meta: str, bullets: list[str] | None = None) -> str:
    """One experience / project block with optional bullet points."""
    bpad = 6 if bullets else 0
    b_html = ""
    if bullets:
        items = "".join(
            f"<li style='margin-bottom:4px;'><span class='abt-body'>{b}</span></li>"
            for b in bullets
        )
        b_html = (
            f"<ul style='margin:6px 0 0;padding-left:14px;list-style:disc;'>{items}</ul>"
        )
    return (
        f"<div style='{_EXPI}'>"
        f"<span class='abt-role' style='margin:0 0 3px;'>{role}</span>"
        f"<span class='abt-org'  style='margin:0 0 2px;'>{org}</span>"
        f"<span class='abt-meta' style='margin:0 0 {bpad}px;'>{meta}</span>"
        f"{b_html}"
        f"</div>"
    )


def _stat_row(stats: list[tuple[str, str]]) -> str:
    """Row of (number, label) stat tiles."""
    cells = []
    for i, (num, lbl) in enumerate(stats):
        border = "" if i == len(stats) - 1 else "border-right:1px solid #1e1e1e;"
        cells.append(
            f"<div style='flex:1;text-align:center;padding:0.65rem 0.25rem;"
            f"background:#080808;{border}'>"
            f"<span class='abt-num'>{num}</span>"
            f"<span class='abt-slbl'>{lbl}</span>"
            f"</div>"
        )
    return (
        "<div style='display:flex;gap:0;margin-top:0.75rem;border:1px solid #1e1e1e;'>"
        + "".join(cells)
        + "</div>"
    )


def _edu(school: str, dept: str, degree: str, years: str,
         notes: str = "", last: bool = False) -> str:
    """One education block. Set last=True on the final entry to remove the bottom border."""
    notes_html = (
        f"<span class='abt-meta' style='margin-top:2px;'>{notes}</span>"
        if notes else ""
    )
    style = _EDUI + ("border-bottom:none;" if last else "")
    return (
        f"<div style='{style}'>"
        f"<span class='abt-sch'  style='margin:0 0 2px;'>{school}</span>"
        f"<span class='abt-dept' style='margin:0 0 2px;'>{dept}</span>"
        f"<span class='abt-deg'  style='margin:0 0 2px;'>{degree}</span>"
        f"<span class='abt-year' style='margin:0 0 2px;'>{years}</span>"
        f"{notes_html}"
        f"</div>"
    )


def _ack(name: str, note: str) -> str:
    """One acknowledgment entry."""
    return (
        f"<div style='{_ACK}'>"
        f"<span class='abt-ack-name'>{name}</span>"
        f"<span class='abt-ack-note'>{note}</span>"
        f"</div>"
    )
