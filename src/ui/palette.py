"""
Canonical UI color palette for the Cross-Asset Spillover Monitor.

All inline-HTML colors across pages should reference these constants.
Import pattern:
    from src.ui.palette import GOLD, DANGER, CARD, BORDER, TEXT, LABEL
"""

# ── Brand ──────────────────────────────────────────────────────────────────
GOLD   = "#CFB991"   # Boilermaker Gold — primary accent
GOLD_D = "#8E6F3E"   # Aged Brown — dimmed gold, eyebrow labels

# ── Backgrounds ───────────────────────────────────────────────────────────
BG      = "#000000"   # page background — pitch black
BG_WARM = "#080808"   # chart plot area — near-black
CARD    = "#0d0d0d"   # card surface
CARD2   = "#141414"   # card surface — elevated
CARD3   = "#1a1a1a"   # hover state

# ── Borders ───────────────────────────────────────────────────────────────
BORDER  = "#1e1e1e"   # structural rule
BORDER2 = "#2a2a2a"   # softer rule
GRID    = "#1a1a1a"   # chart grid lines

# ── Text ──────────────────────────────────────────────────────────────────
TEXT       = "#e8e9ed"   # primary text
TEXT_SOFT  = "#c8c8c8"   # secondary text
TEXT_MUTED = "#b8b8b8"   # muted text
LABEL      = "#8890a1"   # micro-labels, axes
TICK       = "#555960"   # axis ticks, faint UI
LEGEND     = "#8890a1"   # chart legend

# ── Status ────────────────────────────────────────────────────────────────
DANGER = "#c0392b"   # crisis / critical
WARN   = "#e67e22"   # elevated / warning
SAFE   = "#27ae60"   # normal / live
INFO   = "#2980b9"   # informational / neutral signal

# ── Misc ──────────────────────────────────────────────────────────────────
NAVY = "#1E3A5F"   # accent — used in select panels
