"""Tests für das Feature Engineering (muss dem Gold-Layer in Databricks entsprechen)."""

import pytest

from churn.config import FEATURES
from churn.features import build_features


def test_feature_columns_and_order(customer):
    X = build_features(customer)
    assert list(X.columns) == FEATURES


def test_high_risk_customer_features(customer):
    row = build_features(customer).iloc[0]
    assert row["contract_months"] == 1
    assert row["is_new_customer"] == 1
    assert row["num_addon_services"] == 0
    assert row["has_protection"] == 0
    assert row["fiber_without_protection"] == 1
    assert row["is_auto_payment"] == 0
    assert row["avg_monthly_spend"] == pytest.approx(151.65 / 2, abs=0.01)
    assert row["charge_increase"] == pytest.approx(70.70 - 151.65 / 2, abs=0.01)


def test_new_customer_without_history(customer):
    """tenure = 0: Durchschnitt = aktuelle Monatsgebühr, keine Division durch null."""
    customer.loc[0, ["tenure", "total_charges"]] = [0, 0.0]
    row = build_features(customer).iloc[0]
    assert row["avg_monthly_spend"] == pytest.approx(70.70)
    assert row["charge_increase"] == pytest.approx(0.0)


def test_no_missing_values_on_full_data(clean_df):
    X = build_features(clean_df)
    assert X.isna().sum().sum() == 0
    assert len(X) == len(clean_df)
