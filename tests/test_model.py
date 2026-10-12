"""Tests für Training, Modellqualität und Scoring.

Das Modell wird einmal in ein temporäres Verzeichnis trainiert. Die Qualitätsgrenzen
sind bewusst konservativ: Sie schlagen an, wenn eine Code-Änderung das Modell verschlechtert.
"""

import json

import pytest

from churn.score import load_model, score_customers
from churn.train import train


@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    model_dir = tmp_path_factory.mktemp("models")
    metadata = train(model_dir=model_dir)
    return model_dir, metadata


@pytest.fixture(scope="session")
def model(trained):
    model_dir, _ = trained
    return load_model(model_dir / "churn_model.joblib", model_dir / "model_metadata.json")


def test_artifacts_are_written(trained):
    model_dir, _ = trained
    assert (model_dir / "churn_model.joblib").exists()
    meta = json.loads((model_dir / "model_metadata.json").read_text(encoding="utf-8"))
    assert {"threshold", "features", "test_metrics", "sklearn_version"} <= meta.keys()


def test_model_quality_gate(trained):
    """Quality Gate: Modell muss klar besser als Zufall sein (Zufall: PR-AUC ≈ 0,27, Lift = 1)."""
    _, meta = trained
    metrics = meta["test_metrics"]
    assert metrics["pr_auc"] >= 0.55
    assert metrics["roc_auc"] >= 0.80
    assert metrics["recall"] >= 0.70
    assert metrics["lift_top10"] >= 2.3
    assert metrics["net_savings"] > metrics["net_savings_all_customers"]


def test_threshold_is_plausible(trained):
    _, meta = trained
    assert 0.2 <= meta["threshold"] <= 0.7


def test_score_single_customer(model, customer):
    result = score_customers(customer, model).iloc[0]
    assert 0.0 <= result["churn_score"] <= 1.0
    assert result["risk_level"] == "Hoch"
    assert result["reason_1"] != ""
    assert result["recommended_action"]


def test_loyal_customer_scores_lower(model, customer):
    loyal = customer.copy()
    loyal.loc[0, ["tenure", "contract", "payment_method", "online_security", "tech_support"]] = [
        60, "Two year", "Bank transfer (automatic)", "Yes", "Yes"
    ]
    loyal.loc[0, "total_charges"] = 60 * 70.70
    risky_score = score_customers(customer, model).iloc[0]["churn_score"]
    loyal_score = score_customers(loyal, model).iloc[0]["churn_score"]
    assert loyal_score < risky_score
    assert score_customers(loyal, model).iloc[0]["risk_level"] == "Niedrig"
