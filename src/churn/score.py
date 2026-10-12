"""Scoring: Risiko-Score, Risikostufe, Gründe und Maßnahme pro Kunde.

Wird vom Batch-Job (python -m churn.score) und von der Streamlit-App genutzt.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from churn.config import ID_COLUMN, METADATA_PATH, MODEL_PATH, RAW_DATA_PATH, REVENUE_MONTHS, SCORES_PATH
from churn.data import load_data
from churn.evaluate import risk_level
from churn.explain import contributions, reason_text, recommended_action, top_reasons
from churn.features import build_features


@dataclass
class ChurnModel:
    """Geladenes Modell samt Metadaten (Schwellenwert, Referenz für Erklärungen)."""

    pipeline: Pipeline
    metadata: dict

    @property
    def threshold(self) -> float:
        return float(self.metadata["threshold"])

    @property
    def reference(self) -> np.ndarray:
        return np.asarray(self.metadata["explain_reference"])


def load_model(model_path: Path = MODEL_PATH, metadata_path: Path = METADATA_PATH) -> ChurnModel:
    """Lädt Modell und Metadaten von der Festplatte."""
    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)
    return ChurnModel(pipeline=joblib.load(model_path), metadata=metadata)


def score_customers(customers: pd.DataFrame, model: ChurnModel) -> pd.DataFrame:
    """Bewertet bereinigte Kundendaten (Silver-Format) und ergänzt Score, Stufe, Gründe und Maßnahme."""
    X = build_features(customers)
    scores = model.pipeline.predict_proba(X)[:, 1]
    contrib = contributions(model.pipeline, X, model.reference)
    reasons = [top_reasons(row) for _, row in contrib.iterrows()]

    result = pd.DataFrame(index=customers.index)
    if ID_COLUMN in customers.columns:
        result[ID_COLUMN] = customers[ID_COLUMN]
    result["churn_score"] = scores.round(4)
    result["risk_level"] = [risk_level(s, model.threshold) for s in scores]
    result["annual_revenue_at_risk"] = (REVENUE_MONTHS * customers["monthly_charges"]).round(2)
    for i in range(3):
        result[f"reason_{i + 1}"] = [reason_text(r[i]) if len(r) > i else "" for r in reasons]
    result["recommended_action"] = [recommended_action(r) for r in reasons]

    for col in ["contract", "tenure", "monthly_charges", "internet_service", "payment_method"]:
        result[col] = customers[col]
    result["priority_rank"] = result["churn_score"].rank(ascending=False, method="first").astype(int)
    return result.sort_values("priority_rank")


def run_batch_scoring(data_path: Path = RAW_DATA_PATH, output_path: Path = SCORES_PATH) -> pd.DataFrame:
    """Bewertet alle aktiven Kunden (churn = 0) und speichert die Liste als CSV."""
    df = load_data(data_path)
    active = df[df["churn"] == 0].reset_index(drop=True)
    scored = score_customers(active, load_model())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_path, index=False)
    return scored


if __name__ == "__main__":
    result = run_batch_scoring()
    print(f"{len(result)} aktive Kunden bewertet → {SCORES_PATH}")
    print(result["risk_level"].value_counts().to_string())
