"""Erklärungen pro Kunde (Reason Codes) für die Logistische Regression.

Bei einem linearen Modell ist der SHAP-Wert eines Merkmals exakt
Koeffizient × (Wert − Mittelwert der Trainingsdaten). Deshalb braucht die App
keine zusätzliche SHAP-Bibliothek.
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from churn.config import CATEGORICAL_FEATURES

REASON_TEXT = {
    "tenure": "Kurze Kundendauer",
    "contract": "Vertragsart",
    "contract_months": "Kurze Vertragslaufzeit",
    "monthly_charges": "Hohe Monatsgebühr",
    "total_charges": "Niedriger Gesamtumsatz bisher",
    "avg_monthly_spend": "Hohe durchschnittliche Ausgaben",
    "charge_increase": "Preis gestiegen",
    "num_addon_services": "Wenige Zusatzdienste",
    "has_protection": "Kein OnlineSecurity/TechSupport",
    "fiber_without_protection": "Glasfaser ohne Schutzdienste",
    "internet_service": "Internetart",
    "payment_method": "Zahlungsart",
    "is_auto_payment": "Keine automatische Zahlung",
    "is_new_customer": "Neukunde (≤ 12 Monate)",
    "paperless_billing": "Papierlose Rechnung",
    "senior_citizen": "Seniorenkunde",
    "has_partner": "Kein Partner",
    "has_dependents": "Keine Angehörigen",
    "has_phone": "Telefonvertrag",
    "has_multiple_lines": "Mehrere Leitungen",
}

# Neutrale Merkmalsnamen für Diagramme: Balken können das Risiko erhöhen ODER senken,
# daher keine Formulierung, die schon eine Richtung vorgibt („Kein …“, „Kurze …“)
FEATURE_LABEL = {
    "tenure": "Kundendauer",
    "contract": "Vertragsart",
    "contract_months": "Vertragslaufzeit",
    "monthly_charges": "Monatsgebühr",
    "total_charges": "Bisheriger Gesamtumsatz",
    "avg_monthly_spend": "Durchschnittliche Ausgaben",
    "charge_increase": "Preisentwicklung",
    "num_addon_services": "Anzahl Zusatzdienste",
    "has_protection": "OnlineSecurity/TechSupport",
    "fiber_without_protection": "Glasfaser ohne Schutzdienste",
    "internet_service": "Internetart",
    "payment_method": "Zahlungsart",
    "is_auto_payment": "Automatische Zahlung",
    "is_new_customer": "Neukunde (≤ 12 Monate)",
    "paperless_billing": "Papierlose Rechnung",
    "senior_citizen": "Seniorenkunde",
    "has_partner": "Partner",
    "has_dependents": "Angehörige",
    "has_phone": "Telefonvertrag",
    "has_multiple_lines": "Mehrere Leitungen",
}

# Empfohlene Maßnahme je Hauptgrund (Vorschlag, mit dem Retention-Team abzustimmen)
ACTIONS = {
    "contract": "Rabatt für Wechsel in einen Jahresvertrag anbieten",
    "contract_months": "Rabatt für Wechsel in einen Jahresvertrag anbieten",
    "tenure": "Onboarding-Anruf und Willkommensbonus",
    "is_new_customer": "Onboarding-Anruf und Willkommensbonus",
    "monthly_charges": "Tarif-Check: günstigeren Tarif anbieten",
    "avg_monthly_spend": "Tarif-Check: günstigeren Tarif anbieten",
    "charge_increase": "Preiserhöhung erklären oder abfedern",
    "num_addon_services": "Zusatzdienst-Bundle kostenlos testen lassen",
    "has_protection": "OnlineSecurity/TechSupport 3 Monate gratis",
    "fiber_without_protection": "OnlineSecurity/TechSupport 3 Monate gratis",
    "internet_service": "Qualitäts-Check der Glasfaserleitung",
    "payment_method": "Umstellung auf Lastschrift mit Bonus",
    "is_auto_payment": "Umstellung auf Lastschrift mit Bonus",
}
DEFAULT_ACTION = "Persönlicher Beratungsanruf"


def encoded_feature_names(pipeline: Pipeline) -> list[str]:
    """Namen der Spalten nach der Vorverarbeitung (ohne Präfix 'cat__' usw.)."""
    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    return [name.split("__", 1)[1] for name in names]


def to_source_feature(encoded_name: str) -> str:
    """Ordnet eine One-Hot-Spalte ihrem Ursprungsmerkmal zu (contract_One year → contract)."""
    for cat in CATEGORICAL_FEATURES:
        if encoded_name.startswith(f"{cat}_"):
            return cat
    return encoded_name


def background_mean(pipeline: Pipeline, X_train: pd.DataFrame) -> np.ndarray:
    """Mittelwert der vorverarbeiteten Trainingsdaten (Referenzpunkt der Erklärungen)."""
    return pipeline.named_steps["preprocessor"].transform(X_train).mean(axis=0)


def contributions(pipeline: Pipeline, X: pd.DataFrame, reference: np.ndarray) -> pd.DataFrame:
    """Beitrag jedes Ursprungsmerkmals zum Log-Odds-Score (positiv = erhöht das Risiko)."""
    model = pipeline.named_steps["model"]
    if not hasattr(model, "coef_"):
        raise TypeError("Reason Codes sind nur für lineare Modelle implementiert.")
    X_enc = pipeline.named_steps["preprocessor"].transform(X)
    contrib = (X_enc - reference) * model.coef_[0]
    frame = pd.DataFrame(contrib, columns=encoded_feature_names(pipeline), index=X.index)
    return frame.T.groupby(to_source_feature).sum().T


def top_reasons(contrib_row: pd.Series, k: int = 3) -> list[str]:
    """Die k Ursprungsmerkmale mit dem größten positiven Beitrag."""
    positive = contrib_row[contrib_row > 0].sort_values(ascending=False)
    return list(positive.index[:k])


def reason_text(feature: str) -> str:
    """Verständlicher Text für ein Merkmal."""
    return REASON_TEXT.get(feature, feature)


def feature_label(feature: str) -> str:
    """Neutraler Name eines Merkmals (für Diagramme mit positiven und negativen Beiträgen)."""
    return FEATURE_LABEL.get(feature, feature)


def recommended_action(reasons: list[str]) -> str:
    """Maßnahme passend zum wichtigsten Grund."""
    for feature in reasons:
        if feature in ACTIONS:
            return ACTIONS[feature]
    return DEFAULT_ACTION
