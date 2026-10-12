"""Gemeinsame Test-Fixtures."""

import pandas as pd
import pytest

from churn.config import RAW_DATA_PATH
from churn.data import clean_raw_data, load_data


@pytest.fixture(scope="session")
def clean_df() -> pd.DataFrame:
    """Vollständiger, bereinigter Datensatz (einmal pro Testlauf geladen)."""
    return load_data(RAW_DATA_PATH)


@pytest.fixture
def raw_customer() -> pd.DataFrame:
    """Ein typischer Hochrisiko-Kunde im Rohformat der CSV."""
    return pd.DataFrame([{
        "customerID": "TEST-0001",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.70,
        "TotalCharges": "151.65",
        "Churn": "Yes",
    }])


@pytest.fixture
def customer(raw_customer) -> pd.DataFrame:
    """Derselbe Kunde nach der Bereinigung."""
    return clean_raw_data(raw_customer)
