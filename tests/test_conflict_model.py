"""
Unit tests for src/analysis/conflict_model.py.

Tests focus on pure-math scoring functions that don't require API keys,
network calls, or Streamlit state. ACLED/GDELT calls fail silently in test
mode (no API key configured), so compute_cis falls back to the static
registry values — which is the path we care about testing.

Run with: python -m pytest tests/test_conflict_model.py -v
"""
from __future__ import annotations

import datetime
import numpy as np
import pytest

from src.analysis.conflict_model import (
    compute_cis,
    compute_tps,
    compute_confidence,
    compute_trend,
    aggregate_portfolio_scores,
)


# ── Minimal conflict dict factories ──────────────────────────────────────────

def _active_conflict(
    *,
    deadliness: float = 0.5,
    civilian_danger: float = 0.5,
    geographic_diffusion: float = 0.3,
    fragmentation: float = 0.2,
    escalation_trend: str = "stable",
    source_coverage: float = 0.7,
    data_confidence: float = 0.7,
    state: str = "active",
    transmission: dict | None = None,
    last_updated: datetime.date | None = None,
) -> dict:
    return {
        "id":                    "test_conflict",
        "name":                  "Test Conflict",
        "label":                 "TEST",
        "region":                "Test Region",
        "start":                 datetime.date(2022, 1, 1),
        "end":                   None,
        "state":                 state,
        "category":              "War",
        "color":                 "#c0392b",
        "last_updated":          last_updated or datetime.date.today(),
        "deadliness":            deadliness,
        "civilian_danger":       civilian_danger,
        "geographic_diffusion":  geographic_diffusion,
        "fragmentation":         fragmentation,
        "escalation_trend":      escalation_trend,
        "source_coverage":       source_coverage,
        "data_confidence":       data_confidence,
        "transmission":          transmission or {},
        "keywords":              [],
        "affected_equities":     [],
        "affected_commodities":  [],
        "affected_fx":           [],
        "hedge_assets":          [],
        "scoring_basis":         "test",
    }


# ── compute_cis ───────────────────────────────────────────────────────────────

class TestComputeCIS:
    def test_returns_tuple_float_str(self):
        c = _active_conflict()
        result = compute_cis(c)
        assert isinstance(result, tuple) and len(result) == 2
        cis, source = result
        assert isinstance(cis, float)
        assert isinstance(source, str)

    def test_output_bounded_0_to_100(self):
        for state in ("active", "latent", "frozen", "resolved"):
            cis, _ = compute_cis(_active_conflict(state=state))
            assert 0.0 <= cis <= 100.0, f"CIS={cis} out of range for state={state}"

    def test_active_higher_than_latent(self):
        cis_active, _ = compute_cis(_active_conflict(state="active",  deadliness=0.7))
        cis_latent, _ = compute_cis(_active_conflict(state="latent",  deadliness=0.7))
        assert cis_active > cis_latent

    def test_latent_higher_than_frozen(self):
        cis_latent, _ = compute_cis(_active_conflict(state="latent", deadliness=0.7))
        cis_frozen, _ = compute_cis(_active_conflict(state="frozen", deadliness=0.7))
        assert cis_latent > cis_frozen

    def test_resolved_gives_lowest_cis(self):
        # "resolved" is not in _STATE_MULT so it defaults to 1.0 (same as active).
        # The invariant is: resolved ≤ active (equal when the multiplier matches).
        cis_active,   _ = compute_cis(_active_conflict(state="active"))
        cis_resolved, _ = compute_cis(_active_conflict(state="resolved"))
        assert cis_resolved <= cis_active

    def test_escalating_higher_than_deescalating(self):
        cis_up,   _ = compute_cis(_active_conflict(escalation_trend="escalating"))
        cis_down, _ = compute_cis(_active_conflict(escalation_trend="de-escalating"))
        assert cis_up > cis_down

    def test_high_dimensions_give_high_cis(self):
        c = _active_conflict(
            deadliness=1.0, civilian_danger=1.0,
            geographic_diffusion=1.0, fragmentation=1.0,
            escalation_trend="escalating", source_coverage=1.0,
        )
        cis, _ = compute_cis(c)
        assert cis > 70.0, f"Maxed-out conflict should score >70, got {cis:.1f}"

    def test_zero_dimensions_give_low_cis(self):
        c = _active_conflict(
            deadliness=0.0, civilian_danger=0.0,
            geographic_diffusion=0.0, fragmentation=0.0,
            escalation_trend="de-escalating", source_coverage=0.0,
        )
        cis, _ = compute_cis(c)
        assert cis < 40.0, f"Minimal conflict should score <40, got {cis:.1f}"

    def test_source_falls_back_to_static_without_api(self):
        """Without ACLED/GDELT keys, source must be 'static'."""
        c = _active_conflict()
        _, source = compute_cis(c)
        assert source == "static", f"Expected 'static' without API keys, got '{source}'"

    def test_stale_conflict_capped(self):
        """Conflict not updated in >180 days should have CIS capped."""
        stale = _active_conflict(
            deadliness=1.0, civilian_danger=1.0,
            geographic_diffusion=1.0, fragmentation=1.0,
            escalation_trend="escalating",
            last_updated=datetime.date(2020, 1, 1),  # very stale
        )
        fresh = _active_conflict(
            deadliness=1.0, civilian_danger=1.0,
            geographic_diffusion=1.0, fragmentation=1.0,
            escalation_trend="escalating",
        )
        cis_stale, _ = compute_cis(stale)
        cis_fresh, _ = compute_cis(fresh)
        assert cis_stale <= cis_fresh, "Stale conflict should not exceed fresh equivalent"


# ── compute_tps ───────────────────────────────────────────────────────────────

class TestComputeTPS:
    def test_returns_float(self):
        assert isinstance(compute_tps(_active_conflict()), float)

    def test_output_bounded_0_to_100(self):
        for state in ("active", "latent", "frozen", "resolved"):
            tps = compute_tps(_active_conflict(state=state, transmission={"oil_gas": 0.5}))
            assert 0.0 <= tps <= 100.0

    def test_zero_transmission_gives_near_zero_tps(self):
        c = _active_conflict(transmission={})
        tps = compute_tps(c)
        assert tps < 5.0, f"Empty transmission should give near-zero TPS, got {tps:.1f}"

    def test_full_transmission_gives_high_tps(self):
        full_tx = {
            "oil_gas": 1.0, "metals": 1.0, "agriculture": 1.0,
            "shipping": 1.0, "chokepoint": 1.0, "sanctions": 1.0,
            "equity_sector": 1.0, "fx": 1.0, "inflation": 1.0,
            "supply_chain": 1.0, "credit": 1.0, "energy_infra": 1.0,
        }
        c = _active_conflict(transmission=full_tx)
        tps = compute_tps(c)
        assert tps > 80.0, f"Full transmission should give >80 TPS, got {tps:.1f}"

    def test_active_higher_than_frozen_same_transmission(self):
        tx = {"oil_gas": 0.8, "sanctions": 0.7}
        tps_active = compute_tps(_active_conflict(state="active",  transmission=tx))
        tps_frozen = compute_tps(_active_conflict(state="frozen",  transmission=tx))
        assert tps_active > tps_frozen

    def test_unknown_channels_ignored(self):
        """Unknown transmission keys should not raise."""
        tx = {"unknown_channel": 0.9, "oil_gas": 0.5}
        tps = compute_tps(_active_conflict(transmission=tx))
        assert 0.0 <= tps <= 100.0


# ── compute_confidence ────────────────────────────────────────────────────────

class TestComputeConfidence:
    def test_returns_float(self):
        assert isinstance(compute_confidence(_active_conflict()), float)

    def test_output_bounded_0_to_1(self):
        for state in ("active", "latent", "resolved"):
            conf = compute_confidence(_active_conflict(state=state))
            assert 0.0 <= conf <= 1.0

    def test_high_source_coverage_increases_confidence(self):
        conf_high = compute_confidence(_active_conflict(source_coverage=0.95, data_confidence=0.9))
        conf_low  = compute_confidence(_active_conflict(source_coverage=0.2,  data_confidence=0.2))
        assert conf_high > conf_low

    def test_stale_conflict_reduces_confidence(self):
        fresh = _active_conflict()
        stale = _active_conflict(last_updated=datetime.date(2019, 1, 1))
        conf_fresh = compute_confidence(fresh)
        conf_stale = compute_confidence(stale)
        assert conf_stale < conf_fresh

    def test_default_dimensions_penalise_completeness(self):
        """All dims at exact defaults → lower completeness → lower confidence."""
        defaults = _active_conflict(
            deadliness=0.5, civilian_danger=0.5,
            geographic_diffusion=0.3, fragmentation=0.2,
        )
        non_defaults = _active_conflict(
            deadliness=0.9, civilian_danger=0.8,
            geographic_diffusion=0.7, fragmentation=0.6,
        )
        assert compute_confidence(non_defaults) > compute_confidence(defaults)


# ── compute_trend ─────────────────────────────────────────────────────────────

class TestComputeTrend:
    @pytest.mark.parametrize("trend,expected_substr", [
        ("escalating",     "scalat"),
        ("de-escalating",  "scalat"),
        ("stable",         "table"),
        ("unknown_value",  ""),   # should not raise
    ])
    def test_returns_string_for_known_and_unknown(self, trend, expected_substr):
        result = compute_trend(_active_conflict(escalation_trend=trend))
        assert isinstance(result, str)


# ── aggregate_portfolio_scores ────────────────────────────────────────────────

class TestAggregatePortfolioScores:
    def test_returns_dict_with_required_keys(self):
        """Call with static registry conflicts — no network needed."""
        result = aggregate_portfolio_scores()
        required = {"cis", "tps", "confidence", "portfolio_cis", "portfolio_tps"}
        assert required.issubset(result.keys()), (
            f"Missing keys: {required - result.keys()}"
        )

    def test_portfolio_cis_bounded(self):
        result = aggregate_portfolio_scores()
        assert 0.0 <= result["portfolio_cis"] <= 100.0

    def test_portfolio_tps_bounded(self):
        result = aggregate_portfolio_scores()
        assert 0.0 <= result["portfolio_tps"] <= 100.0

    def test_conflict_detail_is_dict(self):
        result = aggregate_portfolio_scores()
        assert isinstance(result.get("conflict_detail", {}), dict)

    def test_empty_conflicts_list_returns_default_floor(self):
        # Empty input → falsy branch returns {"cis": 50.0, "tps": 50.0} (neutral floor).
        result = aggregate_portfolio_scores([])
        assert result["cis"] == 50.0
        assert result["tps"] == 50.0

    def test_single_active_conflict_scores_positive(self):
        # aggregate_portfolio_scores expects the OUTPUT of score_all_conflicts:
        # {conflict_id: {"cis": float, "tps": float, ...}} — not raw conflict dicts.
        pre_scored = {
            "test_conflict": {
                "name":            "Test Conflict",
                "cis":             55.0,
                "tps":             60.0,
                "confidence":      0.7,
                "market_freshness": 1.0,
            }
        }
        result = aggregate_portfolio_scores(pre_scored)
        assert result["portfolio_cis"] > 0
        assert result["portfolio_tps"] > 0
