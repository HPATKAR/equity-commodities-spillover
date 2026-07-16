"""
Forward Scenario Tree - branching conditional forecast for a chosen conflict.

From today's state, branch into escalate / hold / de-escalate with probabilities
seeded by the live ACLED+GDELT escalation signal, then branch each node once
more (two 15-trading-day steps ≈ 30-day horizon, 3² = 9 terminal paths).

Every node is priced with the EXISTING machinery - regime-conditional OLS betas
and fixed sensitivity tables from the Scenario Engine, branch shock sizes scaled
by the conflict's transmission channels (conflict_commodity_matrix) and TPS.
Nothing new is estimated here; this module only chains what already exists.

The tree collapses across all paths, weighted by path probability, into a
DISTRIBUTION of 30-day outcomes. Node-level dispersion comes from the existing
historical-simulation VaR (var95 → σ_daily), scaled by √horizon, so the mixture
is continuous rather than 9 point masses.

This is a conditional forecast under stated, on-screen, perturbable assumptions
 - NOT a point prediction. Probability bands widen with depth by construction;
that widening is the feature, not a defect.

ASSUMPTION NOTE: branch priors, the persistence tilt, per-step shock sizes, and
the de-escalation asymmetry (relief = 0.6 × shock) are stated scenario
assumptions, all displayed and perturbable in the UI.
"""

from __future__ import annotations

import numpy as np

BRANCHES = ("escalate", "hold", "deescalate")

BRANCH_LABELS = {"escalate": "Escalate", "hold": "Hold", "deescalate": "De-escalate"}

# ── Prior mapping: fused escalation signal → step-1 branch probabilities ─────
# Stated assumption, perturbable in the UI. Symmetric around "stable".
BRANCH_PRIORS: dict[str, dict[str, float]] = {
    "escalating":    {"escalate": 0.50, "hold": 0.35, "deescalate": 0.15},
    "stable":        {"escalate": 0.25, "hold": 0.50, "deescalate": 0.25},
    "de-escalating": {"escalate": 0.15, "hold": 0.35, "deescalate": 0.50},
}

# Persistence: conflicts trend - an escalation step raises the odds of another.
# Multiplicative tilt on the base priors, renormalised; strength ∈ [0, 1] in UI.
_PERSISTENCE_WEIGHTS: dict[str, dict[str, float]] = {
    "escalate":   {"escalate": 1.6, "hold": 1.0, "deescalate": 0.6},
    "hold":       {"escalate": 1.0, "hold": 1.0, "deescalate": 1.0},
    "deescalate": {"escalate": 0.6, "hold": 1.0, "deescalate": 1.6},
}

# ── Per-step branch shocks (15 trading days), before conflict scaling ────────
# Escalate magnitudes sit between the "Oil Supply Shock" and "Strait Closure"
# presets, halved for a single step. De-escalation relief is asymmetric:
# unwinds are historically smaller than the shocks they reverse (0.6 ×).
_ESCALATE_BASE = {
    "oil_pct": 18.0, "gold_pct": 6.0, "natgas_pct": 14.0,
    "yield_bps": 15.0, "dxy_pct": 0.8, "credit_bps": 45.0, "geo": 5.0,
}
_RELIEF_RATIO = 0.6

_REGIME_SHIFT = {"escalate": +1, "hold": 0, "deescalate": -1}

STEP_DAYS = 15          # trading days per tree step
HORIZON_DAYS = 30       # total horizon
PLAUSIBILITY_FLOOR = 0.05

# Assets surfaced in the collapsed distribution (crude / gold / equities)
TREE_ASSETS = ["WTI Crude Oil", "Gold", "S&P 500", "DAX"]
_SEVERITY_ASSETS = ["S&P 500", "DAX"]   # worst-path ranking = equity drawdown


# ── Live escalation signal ────────────────────────────────────────────────────

def live_escalation_signal(conflict: dict) -> tuple[str, str]:
    """
    Fused escalation signal for one conflict, mirroring compute_cis() fusion:
    ACLED + GDELT corroborated when both available, single source otherwise,
    static registry value as last resort.

    Returns (signal, source) where signal ∈ {escalating, stable, de-escalating}
    and source ∈ {acled+gdelt, gdelt, acled, static}.
    """
    acled_sig, gdelt_sig = "", ""
    try:
        from src.data.acled import fetch_acled_intensity, acled_to_cis_dimensions, acled_configured
        if acled_configured() and conflict.get("acled_id"):
            result = fetch_acled_intensity(conflict["acled_id"], days=30)
            dims = acled_to_cis_dimensions(result, conflict)
            if "escalation_trend" in dims:
                acled_sig = str(dims["escalation_trend"])
    except Exception:
        pass
    try:
        from src.data.gdelt import fetch_gdelt_escalation, gdelt_corroboration
        if conflict.get("acled_id"):
            gd = fetch_gdelt_escalation(conflict["acled_id"], timespan="7d")
            if gd.get("data_available"):
                gdelt_sig = str(gd["escalation_signal"])
    except Exception:
        pass

    if acled_sig and gdelt_sig:
        from src.data.gdelt import gdelt_corroboration
        return gdelt_corroboration(acled_sig, gdelt_sig)["final_signal"], "acled+gdelt"
    if gdelt_sig:
        return gdelt_sig, "gdelt"
    if acled_sig:
        return acled_sig, "acled"
    return conflict.get("escalation_trend", "stable"), "static"


# ── Branch shocks scaled by the conflict's transmission channels ──────────────

def branch_step_shocks(
    branch: str,
    matrix_row: dict[str, float],
    tps: float,
) -> tuple[dict[str, float], dict[str, float]]:
    """
    One step's shocks for a branch, scaled by the conflict's commodity
    relevance (conflict_commodity_matrix row) and TPS.

    Returns (proxy_shocks, fixed_shocks):
      proxy_shocks - {asset_name: decimal_return} for _propagate_shock()
      fixed_shocks - {yield_bps, dxy_pct, credit_bps, geo} for
                     _apply_fixed_sensitivity()
    """
    if branch == "hold":
        return {}, {}

    sign = 1.0 if branch == "escalate" else -_RELIEF_RATIO
    oil_rel = float(matrix_row.get("WTI Crude Oil", 0.0))
    gas_rel = float(matrix_row.get("Natural Gas", 0.0))
    # Gold is a safe-haven bid, not a channel commodity - floor at 0.5 so any
    # active conflict moves it; metals channel lifts it further.
    metals_rel = max(
        float(matrix_row.get(m, 0.0)) for m in ("Copper", "Silver", "Platinum")
    )
    gold_rel = max(0.5, metals_rel)
    tps_scale = max(0.0, min(1.5, tps / 100.0 + 0.5))   # TPS 50 → 1.0, TPS 100 → 1.5

    proxy = {
        "WTI Crude Oil": sign * _ESCALATE_BASE["oil_pct"] / 100 * oil_rel,
        "Gold":          sign * _ESCALATE_BASE["gold_pct"] / 100 * gold_rel,
        "Natural Gas":   sign * _ESCALATE_BASE["natgas_pct"] / 100 * gas_rel,
    }
    proxy = {k: v for k, v in proxy.items() if abs(v) > 1e-9}
    fixed = {
        "yield_bps":  sign * _ESCALATE_BASE["yield_bps"] * tps_scale,
        "dxy_pct":    sign * _ESCALATE_BASE["dxy_pct"] * tps_scale,
        "credit_bps": sign * _ESCALATE_BASE["credit_bps"] * tps_scale,
        "geo":        sign * _ESCALATE_BASE["geo"] * (tps / 100.0),
    }
    return proxy, fixed


# ── Probabilities ─────────────────────────────────────────────────────────────

def conditional_priors(
    base: dict[str, float],
    parent_branch: str,
    persistence: float,
) -> dict[str, float]:
    """
    Step-2 branch probabilities conditional on the step-1 branch.
    persistence ∈ [0, 1]: 0 = independent steps (reuse base priors),
    1 = full persistence tilt (_PERSISTENCE_WEIGHTS).
    """
    w = _PERSISTENCE_WEIGHTS[parent_branch]
    raw = {b: base[b] * (1.0 + persistence * (w[b] - 1.0)) for b in BRANCHES}
    total = sum(raw.values())
    return {b: v / total for b, v in raw.items()}


# ── Tree construction ─────────────────────────────────────────────────────────

def build_tree(
    priors: dict[str, float],
    persistence: float,
    price_fn,                    # callable(proxy_shocks, fixed_shocks, regime) -> {asset: return}
    current_regime: int,
    matrix_row: dict[str, float],
    tps: float,
    cond_overrides: dict[str, dict[str, float]] | None = None,
) -> dict:
    """
    Two-step tree. Each node prices its step shock with the betas of the
    regime it lands in - escalation paths shift toward Elevated/Crisis betas,
    de-escalation paths toward Normal/Decorrelated (GAP 19 machinery).

    Returns {"step1": [3 nodes], "leaves": [9 nodes]}; every node carries
    branch, path, prob (conditional), path_prob, regime, and cum (compounded
    cumulative return per asset).
    """
    step1, leaves = [], []
    for b1 in BRANCHES:
        r1 = min(3, max(0, current_regime + _REGIME_SHIFT[b1]))
        px1, fx1 = branch_step_shocks(b1, matrix_row, tps)
        imp1 = price_fn(px1, fx1, r1)
        node1 = {
            "branch": b1, "path": (b1,), "depth": 1,
            "prob": priors[b1], "path_prob": priors[b1],
            "regime": r1, "cum": dict(imp1),
        }
        step1.append(node1)

        cond = (cond_overrides or {}).get(b1) or conditional_priors(priors, b1, persistence)
        for b2 in BRANCHES:
            r2 = min(3, max(0, r1 + _REGIME_SHIFT[b2]))
            px2, fx2 = branch_step_shocks(b2, matrix_row, tps)
            imp2 = price_fn(px2, fx2, r2)
            cum = {
                a: (1.0 + imp1.get(a, 0.0)) * (1.0 + imp2.get(a, 0.0)) - 1.0
                for a in set(imp1) | set(imp2)
            }
            leaves.append({
                "branch": b2, "path": (b1, b2), "depth": 2,
                "prob": cond[b2], "path_prob": priors[b1] * cond[b2],
                "regime": r2, "cum": cum,
            })
    return {"step1": step1, "leaves": leaves}


# ── Collapse to distribution ─────────────────────────────────────────────────

def sigma_daily_from_var(var_es: dict[str, dict[str, float]]) -> dict[str, float]:
    """Daily return σ per asset implied by historical VaR95 (var95 = 1.645σ)."""
    return {
        a: (d["var95"] / 100.0) / 1.645
        for a, d in var_es.items()
        if d.get("var95", 0) > 0
    }

def collapse_distribution(
    nodes: list[dict],
    sigma_daily: dict[str, float],
    days: int,
    assets: list[str] | None = None,
    n_draws: int = 20_000,
    seed: int = 7,
) -> dict[str, dict[str, float]]:
    """
    Probability-weighted mixture over the given nodes.
    Each node contributes a normal centred on its cumulative impact with
    σ = σ_daily × √days (existing VaR-implied dispersion - no new estimation).

    Returns {asset: {p5, p25, p50, p75, p95, mean}} in decimal returns.
    """
    assets = assets or TREE_ASSETS
    rng = np.random.default_rng(seed)
    probs = np.array([n["path_prob"] for n in nodes], dtype=float)
    probs = probs / probs.sum()
    comp = rng.choice(len(nodes), size=n_draws, p=probs)

    out: dict[str, dict[str, float]] = {}
    for a in assets:
        sd = sigma_daily.get(a, 0.01) * np.sqrt(days)
        means = np.array([n["cum"].get(a, 0.0) for n in nodes])
        draws = means[comp] + rng.normal(0.0, sd, size=n_draws)
        q = np.percentile(draws, [5, 25, 50, 75, 95])
        out[a] = {
            "p5": float(q[0]), "p25": float(q[1]), "p50": float(q[2]),
            "p75": float(q[3]), "p95": float(q[4]), "mean": float(draws.mean()),
        }
    return out


def terminal_samples(
    leaves: list[dict],
    sigma_daily: dict[str, float],
    asset: str,
    days: int = HORIZON_DAYS,
    n_draws: int = 20_000,
    seed: int = 7,
) -> np.ndarray:
    """Raw mixture draws for one asset's terminal distribution (for histograms)."""
    rng = np.random.default_rng(seed)
    probs = np.array([n["path_prob"] for n in leaves], dtype=float)
    probs = probs / probs.sum()
    comp = rng.choice(len(leaves), size=n_draws, p=probs)
    sd = sigma_daily.get(asset, 0.01) * np.sqrt(days)
    means = np.array([n["cum"].get(asset, 0.0) for n in leaves])
    return means[comp] + rng.normal(0.0, sd, size=n_draws)


# ── Worst plausible path + early warning ─────────────────────────────────────

def worst_plausible_path(
    leaves: list[dict],
    floor: float = PLAUSIBILITY_FLOOR,
) -> dict:
    """
    Highest-equity-loss leaf with path probability ≥ floor, so a 1% doomsday
    branch cannot headline. If no leaf clears the floor, the floor relaxes to
    the most probable leaf's probability (guaranteed non-empty).
    """
    eligible = [n for n in leaves if n["path_prob"] >= floor]
    if not eligible:
        eligible = [max(leaves, key=lambda n: n["path_prob"])]

    def severity(n: dict) -> float:
        return float(np.mean([n["cum"].get(a, 0.0) for a in _SEVERITY_ASSETS]))

    worst = min(eligible, key=severity)
    return {**worst, "severity": severity(worst)}


def early_signals(worst_path: tuple[str, ...], current_signal: str, source: str) -> list[dict]:
    """
    Observables that distinguish the worst path's first branch from the
    alternatives - the "you are on this path if you see X" list.
    Static descriptions; the UI attaches live current readings where available.
    """
    first = worst_path[0]
    if first == "escalate":
        step1_txt = "within the next ~5 trading days"
    else:
        step1_txt = f"after an initial '{BRANCH_LABELS[first].lower()}' phase (~{STEP_DAYS} trading days)"

    return [
        {
            "signal": "GDELT media escalation",
            "watch": f"escalation_signal flips to (or stays) 'escalating' {step1_txt}; "
                     "tone_delta turning more negative confirms",
            "current": f"{current_signal} (source: {source})",
        },
        {
            "signal": "ACLED event acceleration",
            "watch": "weekly event count and fatalities rising vs 30-day baseline "
                     " - the direct measure GDELT media volume proxies with a 12–48h lag",
            "current": "live" if "acled" in source else "not configured - GDELT/static proxy in use",
        },
        {
            "signal": "Energy futures curve",
            "watch": "front-month WTI/Brent basis moving into (or deepening) "
                     "backwardation - the market pricing near-term supply tightness "
                     "corroborates the escalation branch",
            "current": "",   # UI fills from fetch_curve_snapshot()
        },
    ]
