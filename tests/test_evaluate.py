"""Tests für Geschäftsmetriken und Schwellenwert."""

import numpy as np
import pytest

from churn.config import OFFER_COST, REVENUE_MONTHS, SUCCESS_RATE
from churn.evaluate import campaign_profit, find_best_threshold, lift_at_k, risk_level


def test_campaign_profit_by_hand():
    y = np.array([1, 0, 1, 0])
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    monthly = np.array([100.0, 50.0, 80.0, 20.0])
    # Angesprochen: Kunde 1 (Kündiger) und Kunde 2 (kein Kündiger)
    expected = SUCCESS_RATE * REVENUE_MONTHS * 100.0 - 2 * OFFER_COST
    assert campaign_profit(y, scores, monthly, threshold=0.5) == pytest.approx(expected)


def test_nobody_targeted_means_zero_profit():
    assert campaign_profit([1, 0], [0.1, 0.2], [100, 100], threshold=0.99) == 0.0


def test_perfect_ranking_has_max_lift():
    y = np.array([1] * 10 + [0] * 90)
    scores = np.linspace(1, 0, 100)
    assert lift_at_k(y, scores, k=0.10) == pytest.approx(10.0)


def test_best_threshold_is_in_grid():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    scores = np.clip(y * 0.5 + rng.random(500) * 0.5, 0, 1)
    t, profit = find_best_threshold(y, scores, np.full(500, 70.0))
    assert 0.05 <= t <= 0.95
    assert profit > 0


@pytest.mark.parametrize(
    ("score", "expected"),
    [(0.80, "Hoch"), (0.39, "Hoch"), (0.30, "Beobachten"), (0.10, "Niedrig")],
)
def test_risk_level(score, expected):
    assert risk_level(score, threshold=0.39) == expected
