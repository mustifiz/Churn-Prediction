"""Laden und Bereinigen der Rohdaten (entspricht dem Silver-Layer in Databricks)."""

import re
from pathlib import Path

import pandas as pd

from churn.config import RAW_DATA_PATH


def to_snake_case(name: str) -> str:
    """Wandelt CamelCase in snake_case um (customerID → customer_id)."""
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).lower()


def clean_raw_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Bereinigt die Rohdaten: Spaltennamen, Datentypen, fehlende Werte, Zielvariable 0/1."""
    df = raw.copy()
    df.columns = [to_snake_case(c) for c in df.columns]

    # Neukunden (tenure = 0) haben ein Leerzeichen statt eines Betrags → 0.0
    df["total_charges"] = pd.to_numeric(df["total_charges"].astype(str).str.strip(), errors="coerce")
    df["total_charges"] = df["total_charges"].fillna(0.0)

    df["monthly_charges"] = df["monthly_charges"].astype(float)
    df["tenure"] = df["tenure"].astype(int)
    df["senior_citizen"] = df["senior_citizen"].astype(int)

    # Ab pandas 3 sind Texte vom Typ "str" statt "object" → auf „nicht numerisch“ prüfen
    if "churn" in df.columns and not pd.api.types.is_numeric_dtype(df["churn"]):
        df["churn"] = (df["churn"] == "Yes").astype(int)

    return df


def load_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Lädt die Telco-CSV und gibt bereinigte Daten zurück."""
    return clean_raw_data(pd.read_csv(path))
