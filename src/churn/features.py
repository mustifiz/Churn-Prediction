"""Feature Engineering (entspricht dem Gold-Layer `gold_churn_features` in Databricks)."""

import numpy as np
import pandas as pd

from churn.config import BINARY_FEATURES, FEATURES, NUMERIC_FEATURES

ADDON_SERVICES = [
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
]
CONTRACT_MONTHS = {"Month-to-month": 1, "One year": 12, "Two year": 24}


def _yes(series: pd.Series) -> pd.Series:
    """'Yes' → 1, alles andere → 0."""
    return (series == "Yes").astype(int)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Erzeugt alle Modellmerkmale aus bereinigten Kundendaten.

    Erwartet die Spalten des Silver-Layers (snake_case). Gibt die Modellmerkmale
    in fester Reihenfolge zurück, numerische Spalten als float64.
    """
    out = pd.DataFrame(index=df.index)

    # Demografie (gender bewusst ausgeschlossen: kein Signal, Fairness)
    out["senior_citizen"] = df["senior_citizen"].astype(int)
    out["has_partner"] = _yes(df["partner"])
    out["has_dependents"] = _yes(df["dependents"])

    # Vertrag
    out["tenure"] = df["tenure"]
    out["contract"] = df["contract"]
    out["contract_months"] = df["contract"].map(CONTRACT_MONTHS).fillna(24)
    out["is_new_customer"] = (df["tenure"] <= 12).astype(int)
    out["paperless_billing"] = _yes(df["paperless_billing"])
    out["payment_method"] = df["payment_method"]
    out["is_auto_payment"] = df["payment_method"].str.contains("automatic").astype(int)

    # Dienste
    out["internet_service"] = df["internet_service"]
    out["has_phone"] = _yes(df["phone_service"])
    out["has_multiple_lines"] = _yes(df["multiple_lines"])
    out["num_addon_services"] = sum(_yes(df[col]) for col in ADDON_SERVICES)
    out["has_protection"] = ((df["online_security"] == "Yes") | (df["tech_support"] == "Yes")).astype(int)
    out["fiber_without_protection"] = (
        (df["internet_service"] == "Fiber optic")
        & (df["online_security"] == "No")
        & (df["tech_support"] == "No")
    ).astype(int)

    # Kosten
    out["monthly_charges"] = df["monthly_charges"]
    out["total_charges"] = df["total_charges"]
    avg_spend = np.where(df["tenure"] > 0, df["total_charges"] / df["tenure"].replace(0, 1), df["monthly_charges"])
    out["avg_monthly_spend"] = np.round(avg_spend, 2)
    out["charge_increase"] = np.round(df["monthly_charges"] - avg_spend, 2)

    out = out[FEATURES]
    return out.astype({col: "float64" for col in NUMERIC_FEATURES + BINARY_FEATURES})
