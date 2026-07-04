"""
Step 4 of 4 — ranking + desk-report feed.

Pins: ranking is attractiveness (DSR + conviction + constraint room), never
raw Sharpe/P&L; the book changes when state changes; locked trades never
enter the book; the PDF feed carries every key the EXISTING generator's
trade card consumes.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.trade_allocator import rank_trades, desk_report_feed


def _t(name, dsr, conv, weight=0.0, eligible=True, clipped=False,
       conflict=None, cluster=None, sharpe_raw=0.0):
    return {
        "name": name, "is_eligible": eligible,
        "alloc_weight": weight,
        "alloc_detail": {"dsr": dsr, "conviction": conv,
                         "sharpe_raw": sharpe_raw},
        "constraint_detail": {"conflict_group": conflict, "cluster_id": cluster,
                              "clipped": clipped},
        # generator-schema fields
        "category": "Macro", "regime": [2, 3],
        "assets": ["Gold", "S&P 500"], "direction": ["Long", "Short"],
        "trigger": "test trigger", "rationale": "test rationale",
        "entry": "entry rule", "exit": "exit rule", "risk": "risk text",
    }


# ── Attractiveness, never raw Sharpe ─────────────────────────────────────────

def test_raw_sharpe_perturbation_never_changes_ranking():
    a = [_t("A", dsr=0.8, conv=0.5, sharpe_raw=-3.0),
         _t("B", dsr=0.6, conv=0.5, sharpe_raw=+9.0)]
    b = [_t("A", dsr=0.8, conv=0.5, sharpe_raw=+9.0),
         _t("B", dsr=0.6, conv=0.5, sharpe_raw=-3.0)]
    r1 = [t["name"] for t in rank_trades(a)]
    r2 = [t["name"] for t in rank_trades(b)]
    assert r1 == r2 == ["A", "B"]        # only DSR ordered this book


def test_dsr_perturbation_reorders():
    book = [_t("A", dsr=0.9, conv=0.5), _t("B", dsr=0.4, conv=0.5)]
    assert [t["name"] for t in rank_trades(book)] == ["A", "B"]
    book2 = [_t("A", dsr=0.4, conv=0.5), _t("B", dsr=0.9, conv=0.5)]
    assert [t["name"] for t in rank_trades(book2)] == ["B", "A"]


def test_conviction_breaks_dsr_ties():
    book = [_t("LOW", dsr=0.7, conv=0.2), _t("HIGH", dsr=0.7, conv=0.9)]
    assert rank_trades(book)[0]["name"] == "HIGH"


def test_constraint_room_breaks_ties():
    # Same evidence; CAPPED sits at the single cap with weight, FREE has room
    book = [_t("CAPPED", dsr=0.7, conv=0.5, weight=0.35, clipped=True),
            _t("FREE", dsr=0.7, conv=0.5, weight=0.10)]
    ranked = rank_trades(book)
    assert ranked[0]["name"] == "FREE"
    assert ranked[0]["attr_detail"]["room"] > ranked[1]["attr_detail"]["room"]


# ── Live book semantics ──────────────────────────────────────────────────────

def test_state_change_changes_the_book():
    state1 = rank_trades([_t("X", 0.9, 0.9), _t("Y", 0.5, 0.9)])   # X leads
    state2 = rank_trades([_t("X", 0.2, 0.9), _t("Y", 0.5, 0.9)])   # Y leads
    assert [t["name"] for t in state1] != [t["name"] for t in state2]


def test_locked_trades_never_enter_the_book():
    book = rank_trades([_t("OK", 0.7, 0.5),
                        _t("LOCKED", 0.99, 0.99, eligible=False)])
    assert [t["name"] for t in book] == ["OK"]


def test_ranks_are_dense_and_ordered():
    ranked = rank_trades([_t(f"T{i}", 0.5 + i / 20, 0.5) for i in range(5)])
    assert [t["rank"] for t in ranked] == [1, 2, 3, 4, 5]
    attrs = [t["attractiveness"] for t in ranked]
    assert attrs == sorted(attrs, reverse=True)


# ── PDF feed: exact schema of the EXISTING generator's trade card ────────────

_GENERATOR_KEYS = ("category", "regime", "assets", "direction",
                   "trigger", "name", "rationale")


def test_feed_carries_every_generator_key():
    ranked = rank_trades([_t("A", 0.8, 0.6, weight=0.2)])
    feed = desk_report_feed(ranked)
    for k in _GENERATOR_KEYS:
        assert k in feed[0], f"generator key {k!r} missing from feed"


def test_feed_rides_rank_weight_and_invalidation_in_existing_fields():
    ranked = rank_trades([_t("Long Gold / Short SPX", 0.8, 0.6, weight=0.215)])
    f = desk_report_feed(ranked)[0]
    assert f["name"].startswith("#1 · 21.5% wt · Long Gold / Short SPX")
    assert "ATTR" in f["trigger"] and "DSR 0.80" in f["trigger"]
    assert f["risk"].startswith("INVALIDATED IF: exit rule")


def test_feed_does_not_mutate_originals():
    ranked = rank_trades([_t("A", 0.8, 0.6)])
    desk_report_feed(ranked)
    assert ranked[0]["name"] == "A"        # copies only


# ── PDF feed mirrors the screen's cash state ─────────────────────────────────

def test_feed_cash_book_leads_with_desk_call():
    ranked = rank_trades([_t("A", 0.3, 0.5), _t("B", 0.1, 0.5)])  # zero gross
    feed = desk_report_feed(ranked)
    assert len(feed) == 3                  # lead card + 2 watchlist entries
    assert feed[0]["name"] == "DESK CALL — 0% DEPLOYED · 100% CASH"
    assert "0.50 deploy bar" in feed[0]["rationale"]     # strict default bar
    assert "best 0.30" in feed[0]["rationale"]
    for k in _GENERATOR_KEYS:              # lead card renders through the
        assert k in feed[0]                # SAME existing trade-card schema


def test_feed_cash_book_labels_trades_as_watchlist():
    ranked = rank_trades([_t("A", 0.3, 0.5)])
    f = desk_report_feed(ranked)[1]
    assert f["name"].startswith(
        "WATCHLIST #1 — NOT YET ALLOCATABLE · 0.0% wt · A")


def test_feed_deployed_book_has_no_cash_card():
    ranked = rank_trades([_t("A", 0.8, 0.6, weight=0.2)])
    feed = desk_report_feed(ranked)
    assert len(feed) == 1                  # never a full-looking cash card
    assert feed[0]["name"].startswith("#1 · 20.0% wt · A")


def test_feed_empty_book_stays_empty():
    assert desk_report_feed([]) == []      # generator's own empty-state text
