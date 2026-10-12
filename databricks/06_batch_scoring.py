# Databricks notebook source
# MAGIC %md
# MAGIC # Churn Prediction – Notebook 06: Batch-Scoring & Übergabe an den Fachbereich
# MAGIC
# MAGIC **Ziel:** Für jeden **aktiven** Kunden einen Risiko-Score, eine Risikostufe und die
# MAGIC **3 wichtigsten Gründe** berechnen und als Tabelle `workspace.churn.churn_scores` bereitstellen.
# MAGIC
# MAGIC > „Ein Score, den der Vertrieb nicht versteht oder der nicht ins CRM gelangt, bleibt wirkungslos.“
# MAGIC > – sinngemäß nach Datasolut
# MAGIC
# MAGIC Dieses Notebook läuft später als **wöchentlicher Job** (Workflows) und
# MAGIC exportiert zusätzlich Modell + Scores für die Streamlit-App (Phase 7).

# COMMAND ----------

# Bibliothek installieren und Python neu starten
%pip install -q shap
dbutils.library.restartPython()

# COMMAND ----------

import json
import os
from datetime import datetime, timezone

import joblib
import mlflow
import numpy as np
import pandas as pd
import shap

MODEL_NAME = "workspace.churn.churn_model"
GOLD_TABLE = "workspace.churn.gold_churn_features"
SCORES_TABLE = "workspace.churn.churn_scores"
EXPORT_DIR = "/Volumes/workspace/churn/artifacts"

mlflow.set_registry_uri("databricks-uc")
spark.sql("CREATE VOLUME IF NOT EXISTS workspace.churn.artifacts")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Champion-Modell und Schwellenwert aus der Registry laden

# COMMAND ----------

client = mlflow.MlflowClient()
champion = client.get_model_version_by_alias(MODEL_NAME, "champion")
THRESHOLD = float(champion.tags["threshold"])
MODEL_VERSION = champion.version

pipeline = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")
FEATURES = list(pipeline.feature_names_in_)
print(f"Modell v{MODEL_VERSION} ({champion.tags.get('model_type')}), Schwellenwert {THRESHOLD}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Aktive Kunden scoren
# MAGIC Gekündigte Kunden (`churn = 1`) sind bereits weg – eine Kampagne lohnt sich nur für aktive Kunden.

# COMMAND ----------

gold = spark.table(GOLD_TABLE).toPandas()
active = gold[gold["churn"] == 0].reset_index(drop=True)
X_active = active[FEATURES].astype({c: "float64" for c in FEATURES
                                    if c not in ("contract", "payment_method", "internet_service")})

active["churn_score"] = pipeline.predict_proba(X_active)[:, 1]


def risk_level(score: float) -> str:
    """Risikostufe für den Fachbereich: Hoch = Kampagne, Beobachten = knapp darunter."""
    if score >= THRESHOLD:
        return "Hoch"
    if score >= THRESHOLD - 0.15:
        return "Beobachten"
    return "Niedrig"


active["risk_level"] = active["churn_score"].apply(risk_level)
active["annual_revenue_at_risk"] = (12 * active["monthly_charges"]).round(2)
active["priority_rank"] = active["churn_score"].rank(ascending=False, method="first").astype(int)

display(active["risk_level"].value_counts().rename_axis("risk_level").reset_index(name="kunden"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Die 3 wichtigsten Gründe pro Kunde (SHAP Reason Codes)
# MAGIC One-Hot-Spalten (z. B. `contract_Month-to-month`) werden wieder zum Ursprungsmerkmal zusammengefasst
# MAGIC und in verständliche deutsche Texte übersetzt.

# COMMAND ----------

REASON_TEXT = {
    "tenure": "Kurze Kundendauer",
    "contract": "Vertragsart (Monatsvertrag)",
    "contract_months": "Kurze Vertragslaufzeit",
    "monthly_charges": "Hohe Monatsgebühr",
    "total_charges": "Niedriger Gesamtumsatz bisher",
    "avg_monthly_spend": "Durchschnittliche Ausgaben",
    "charge_increase": "Preis gestiegen",
    "num_addon_services": "Wenige Zusatzdienste",
    "has_protection": "Kein OnlineSecurity/TechSupport",
    "fiber_without_protection": "Glasfaser ohne Schutzdienste",
    "internet_service": "Internetart (Glasfaser)",
    "payment_method": "Zahlungsart (Electronic check)",
    "is_auto_payment": "Keine automatische Zahlung",
    "is_new_customer": "Neukunde (≤ 12 Monate)",
    "paperless_billing": "Papierlose Rechnung",
    "senior_citizen": "Seniorenkunde",
    "has_partner": "Kein Partner",
    "has_dependents": "Keine Angehörigen",
    "has_phone": "Telefonvertrag",
    "has_multiple_lines": "Mehrere Leitungen",
}

preprocessor = pipeline.named_steps["preprocessor"]
model = pipeline.named_steps["model"]
encoded_names = [n.split("__", 1)[1] for n in preprocessor.get_feature_names_out()]


def to_source_feature(encoded: str) -> str:
    """Ordnet eine One-Hot-Spalte ihrem Ursprungsmerkmal zu (contract_One year → contract)."""
    for cat in ("contract", "payment_method", "internet_service"):
        if encoded.startswith(f"{cat}_"):
            return cat
    return encoded


X_enc = pd.DataFrame(preprocessor.transform(X_active), columns=encoded_names)
background = X_enc.sample(n=min(200, len(X_enc)), random_state=42)
if hasattr(model, "coef_"):
    shap_values = shap.LinearExplainer(model, background)(X_enc).values
else:
    shap_values = shap.TreeExplainer(model)(X_enc).values[:, :, 1]

# Beiträge je Ursprungsmerkmal aufsummieren
contrib = (pd.DataFrame(shap_values, columns=encoded_names)
           .T.groupby(to_source_feature).sum().T)


def top_reasons(row: pd.Series, k: int = 3) -> list[str]:
    """Die k Merkmale mit dem größten positiven Beitrag zum Kündigungsrisiko."""
    positive = row[row > 0].sort_values(ascending=False).head(k)
    return [REASON_TEXT.get(name, name) for name in positive.index]


reasons = contrib.apply(top_reasons, axis=1)
for i in range(3):
    active[f"reason_{i + 1}"] = reasons.apply(lambda r, i=i: r[i] if len(r) > i else None)

display(active.sort_values("priority_rank")[[
    "customer_id", "churn_score", "risk_level", "annual_revenue_at_risk",
    "reason_1", "reason_2", "reason_3"]].head(15))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Score-Tabelle schreiben (Übergabe an CRM / Dashboard)

# COMMAND ----------

scored_at = datetime.now(timezone.utc)
SCORE_COLUMNS = [
    "customer_id", "churn_score", "risk_level", "priority_rank", "annual_revenue_at_risk",
    "reason_1", "reason_2", "reason_3",
    "contract", "tenure", "monthly_charges", "internet_service", "payment_method", "num_addon_services",
]
scores = active[SCORE_COLUMNS].copy()
scores["churn_score"] = scores["churn_score"].round(4)
scores["model_version"] = str(MODEL_VERSION)
scores["threshold"] = THRESHOLD
scores["scored_at"] = scored_at

(spark.createDataFrame(scores)
 .write.mode("overwrite").option("overwriteSchema", True)
 .saveAsTable(SCORES_TABLE))
print(f"{len(scores)} aktive Kunden gescort → {SCORES_TABLE}")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Zusammenfassung für den Fachbereich
# MAGIC SELECT risk_level,
# MAGIC        COUNT(*)                            AS kunden,
# MAGIC        ROUND(AVG(churn_score), 3)          AS mittlerer_score,
# MAGIC        ROUND(SUM(annual_revenue_at_risk))  AS jahresumsatz_gefaehrdet
# MAGIC FROM workspace.churn.churn_scores
# MAGIC GROUP BY risk_level
# MAGIC ORDER BY mittlerer_score DESC

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Häufigste Hauptgründe bei Hochrisiko-Kunden → welche Kampagne passt?
# MAGIC SELECT reason_1, COUNT(*) AS kunden
# MAGIC FROM workspace.churn.churn_scores
# MAGIC WHERE risk_level = 'Hoch'
# MAGIC GROUP BY reason_1
# MAGIC ORDER BY kunden DESC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Export für die Streamlit-App (Phase 7)
# MAGIC Die App läuft außerhalb von Databricks (öffentlich für das Portfolio) und braucht daher
# MAGIC das Modell als Datei plus eine Metadaten-Datei.

# COMMAND ----------

os.makedirs(EXPORT_DIR, exist_ok=True)
joblib.dump(pipeline, f"{EXPORT_DIR}/churn_model.joblib")

metadata = {
    "model_name": MODEL_NAME,
    "model_version": str(MODEL_VERSION),
    "model_type": champion.tags.get("model_type"),
    "threshold": THRESHOLD,
    "features": FEATURES,
    "sklearn_version": __import__("sklearn").__version__,  # App muss dieselbe Version nutzen
    "exported_at": scored_at.isoformat(),
}
with open(f"{EXPORT_DIR}/model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

scores.drop(columns=["scored_at"]).to_csv(f"{EXPORT_DIR}/churn_scores.csv", index=False)
display(dbutils.fs.ls(EXPORT_DIR))
