"""Modell-Pipeline (Champion aus Phase 4/5: Logistische Regression)."""

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn.config import BINARY_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """One-Hot-Encoding für Kategorien, Skalierung für numerische Merkmale."""
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
        ]
    )


def build_pipeline(C: float = 1.0) -> Pipeline:
    """Erstellt die vollständige Pipeline aus Vorverarbeitung und Klassifikator."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", LogisticRegression(C=C, class_weight="balanced", max_iter=2000)),
        ]
    )
