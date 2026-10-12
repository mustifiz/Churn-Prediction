"""Technische und geschäftliche Bewertungsmetriken."""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from churn.config import OFFER_COST, REVENUE_MONTHS, SUCCESS_RATE, WATCH_MARGIN


def lift_at_k(y_true, scores, k: float = 0.10) -> float:
    """Churn-Rate in den Top-k % nach Score geteilt durch die Gesamt-Churn-Rate."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    n_top = max(1, int(len(y_true) * k))
    top_idx = np.argsort(scores)[::-1][:n_top]
    return float(y_true[top_idx].mean() / y_true.mean())


def campaign_profit(y_true, scores, monthly_charges, threshold: float) -> float:
    """Netto-Ersparnis einer Kampagne an alle Kunden mit Score ≥ Schwellenwert.

    Ersparnis = gerettete Jahresumsätze echter Kündiger − Kosten aller Angebote.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    monthly_charges = np.asarray(monthly_charges, dtype=float)
    targeted = scores >= threshold
    saved = (SUCCESS_RATE * REVENUE_MONTHS * monthly_charges * y_true)[targeted].sum()
    return float(saved - OFFER_COST * targeted.sum())


def find_best_threshold(y_true, scores, monthly_charges, grid=None) -> tuple[float, float]:
    """Sucht den Schwellenwert mit der höchsten Netto-Ersparnis. Gibt (Schwellenwert, Ersparnis) zurück."""
    if grid is None:
        grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    profits = [campaign_profit(y_true, scores, monthly_charges, t) for t in grid]
    best = int(np.argmax(profits))
    return float(grid[best]), float(profits[best])


def classification_report(y_true, scores, threshold: float) -> dict[str, float]:
    """Alle Bewertungsmetriken für einen Schwellenwert."""
    y_pred = (np.asarray(scores) >= threshold).astype(int)
    return {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "recall": float(recall_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred)),
        "lift_top10": lift_at_k(y_true, scores),
    }


def risk_level(score: float, threshold: float) -> str:
    """Risikostufe für den Fachbereich: Hoch = Kampagne, Beobachten = knapp darunter."""
    if score >= threshold:
        return "Hoch"
    if score >= threshold - WATCH_MARGIN:
        return "Beobachten"
    return "Niedrig"
