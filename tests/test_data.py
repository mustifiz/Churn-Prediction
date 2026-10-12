"""Tests für Laden und Bereinigen der Rohdaten."""

import pandas as pd

from churn.data import clean_raw_data, to_snake_case


def test_to_snake_case():
    assert to_snake_case("customerID") == "customer_id"
    assert to_snake_case("StreamingTV") == "streaming_tv"
    assert to_snake_case("tenure") == "tenure"


def test_blank_total_charges_become_zero(raw_customer):
    raw_customer.loc[0, "TotalCharges"] = " "
    df = clean_raw_data(raw_customer)
    assert df.loc[0, "total_charges"] == 0.0


def test_target_is_binary(customer):
    assert customer.loc[0, "churn"] == 1
    assert pd.api.types.is_integer_dtype(customer["churn"])


def test_full_dataset_quality(clean_df):
    assert len(clean_df) == 7043
    assert clean_df["customer_id"].is_unique
    assert clean_df["total_charges"].isna().sum() == 0
    assert set(clean_df["churn"].unique()) == {0, 1}
    assert 0.25 < clean_df["churn"].mean() < 0.28
