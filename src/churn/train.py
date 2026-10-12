"""Training: Daten laden → Merkmale → Schwellenwert wählen → testen → finales Modell speichern.

Aufruf:  python -m churn.train
"""

import json
import logging
import platform
from datetime import UTC, datetime

import joblib
import numpy as np
import sklearn
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from churn.config import (
    CV_FOLDS,
    FEATURES,
    METADATA_PATH,
    MODEL_DIR,
    MODEL_PATH,
    OFFER_COST,
    RANDOM_STATE,
    RAW_DATA_PATH,
    REVENUE_MONTHS,
    SUCCESS_RATE,
    TARGET,
    TEST_SIZE,
)
from churn.data import load_data
from churn.evaluate import campaign_profit, classification_report, find_best_threshold
from churn.explain import background_mean, encoded_feature_names
from churn.features import build_features
from churn.model import build_pipeline

logger = logging.getLogger(__name__)


def train(data_path=RAW_DATA_PATH, model_dir=MODEL_DIR) -> dict:
    """Trainiert das Champion-Modell und speichert Modell und Metadaten. Gibt die Metadaten zurück."""
    df = load_data(data_path)
    X, y = build_features(df), df[TARGET]
    monthly = df["monthly_charges"].to_numpy()

    X_train, X_test, y_train, y_test, m_train, m_test = train_test_split(
        X, y, monthly, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    pipeline = build_pipeline()

    # 1) Schwellenwert nur auf Train wählen (Out-of-Fold-Scores, kein Blick auf Test)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof = cross_val_predict(pipeline, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    threshold, oof_profit = find_best_threshold(y_train, oof, m_train)
    logger.info("Schwellenwert %.2f (OOF-Ersparnis %.0f $)", threshold, oof_profit)

    # 2) Einmalige Bewertung auf dem Test-Split
    fitted = clone(pipeline).fit(X_train, y_train)
    test_scores = fitted.predict_proba(X_test)[:, 1]
    test_metrics = classification_report(y_test, test_scores, threshold)
    test_metrics["net_savings"] = campaign_profit(y_test, test_scores, m_test, threshold)
    test_metrics["net_savings_all_customers"] = campaign_profit(y_test, np.ones(len(y_test)), m_test, 0.5)
    logger.info("Test: %s", {k: round(v, 3) for k, v in test_metrics.items()})

    # 3) Finales Modell auf allen Daten (gleiche Konfiguration)
    final = clone(pipeline).fit(X, y)

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, model_dir / MODEL_PATH.name)

    metadata = {
        "model_type": "logistic_regression",
        "threshold": threshold,
        "features": FEATURES,
        "test_metrics": {k: round(v, 4) for k, v in test_metrics.items()},
        "business_assumptions": {
            "offer_cost": OFFER_COST,
            "success_rate": SUCCESS_RATE,
            "revenue_months": REVENUE_MONTHS,
        },
        "explain_reference": background_mean(final, X).round(6).tolist(),
        "encoded_features": encoded_feature_names(final),
        "n_train_rows": int(len(X)),
        "churn_rate": round(float(y.mean()), 4),
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    with open(model_dir / METADATA_PATH.name, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info("Modell gespeichert: %s", model_dir / MODEL_PATH.name)
    return metadata


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    meta = train()
    print(json.dumps({"threshold": meta["threshold"], **meta["test_metrics"]}, indent=2))
