"""Zentrale Konfiguration: Pfade, Merkmale und Geschäftsannahmen.

Eine einzige Definition für Training, Scoring und App verhindert
Training-Serving-Skew (unterschiedliche Merkmale in Training und Produktion).
"""

from pathlib import Path

# Projektpfade
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "telco_churn.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SCORES_PATH = PROCESSED_DIR / "churn_scores.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "churn_model.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Merkmale (identisch zum Gold-Layer in Databricks)
TARGET = "churn"
ID_COLUMN = "customer_id"
CATEGORICAL_FEATURES = ["contract", "payment_method", "internet_service"]
NUMERIC_FEATURES = [
    "tenure",
    "monthly_charges",
    "total_charges",
    "avg_monthly_spend",
    "charge_increase",
    "contract_months",
    "num_addon_services",
]
BINARY_FEATURES = [
    "senior_citizen",
    "has_partner",
    "has_dependents",
    "is_new_customer",
    "paperless_billing",
    "is_auto_payment",
    "has_phone",
    "has_multiple_lines",
    "has_protection",
    "fiber_without_protection",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES + BINARY_FEATURES

# Zulässige Ausprägungen (für Validierung und App-Formular)
CATEGORY_VALUES = {
    "contract": ["Month-to-month", "One year", "Two year"],
    "payment_method": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
    "internet_service": ["DSL", "Fiber optic", "No"],
}

# Geschäftsannahmen der Retention-Kampagne (mit dem Fachbereich abzustimmen)
OFFER_COST = 50.0  # Kosten pro angesprochenem Kunden in $
SUCCESS_RATE = 0.30  # Anteil der Kündiger, die das Angebot hält
REVENUE_MONTHS = 12  # Ein geretteter Kunde bringt 12 Monatsbeiträge

# Risikostufen relativ zum Schwellenwert
WATCH_MARGIN = 0.15  # Score ≥ Schwellenwert − 0,15 → „Beobachten“
