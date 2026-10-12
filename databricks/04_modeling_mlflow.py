# Databricks notebook source
# MAGIC %md
# MAGIC # Churn Prediction – Notebook 04: Modellierung & Experiment Tracking mit MLflow
# MAGIC
# MAGIC **Ziel:** Mehrere Modelle fair vergleichen und jeden Versuch in MLflow protokollieren.
# MAGIC
# MAGIC | Modell | Rolle |
# MAGIC |---|---|
# MAGIC | DummyClassifier | Untergrenze: „rät“ nur die Klassenverteilung |
# MAGIC | Logistische Regression | Baseline: einfach, interpretierbar |
# MAGIC | Random Forest | Bagging-Ensemble |
# MAGIC | XGBoost | Gradient Boosting (Standard laut Datasolut 2026) |
# MAGIC | LightGBM | Gradient Boosting, schnell |
# MAGIC
# MAGIC **Bewertung:** 5-fache stratifizierte Kreuzvalidierung **nur auf dem Train-Split**.
# MAGIC Der Test-Split bleibt unangetastet bis Notebook 05 (finale Bewertung).
# MAGIC
# MAGIC **Metriken:** PR-AUC (primär), ROC-AUC, Recall, Precision, F1, Lift im Top-10 %.

# COMMAND ----------

# Bibliotheken installieren (Serverless hat nicht alle vorinstalliert)
%pip install -q xgboost lightgbm
dbutils.library.restartPython()

# COMMAND ----------

import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

RANDOM_STATE = 42
GOLD_TABLE = "workspace.churn.gold_churn_features"

# Experiment im eigenen Benutzerordner anlegen
user = spark.sql("SELECT current_user()").first()[0]
EXPERIMENT_PATH = f"/Users/{user}/churn-prediction"
mlflow.set_experiment(EXPERIMENT_PATH)
print("MLflow-Experiment:", EXPERIMENT_PATH)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Daten laden – nur der Train-Split

# COMMAND ----------

gold = spark.table(GOLD_TABLE).toPandas()
train = gold[gold["split"] == "train"].reset_index(drop=True)

TARGET = "churn"
CATEGORICAL_FEATURES = ["contract", "payment_method", "internet_service"]
NUMERIC_FEATURES = [
    "tenure", "monthly_charges", "total_charges", "avg_monthly_spend", "charge_increase",
    "contract_months", "num_addon_services",
]
BINARY_FEATURES = [
    "senior_citizen", "has_partner", "has_dependents", "is_new_customer", "paperless_billing",
    "is_auto_payment", "has_phone", "has_multiple_lines", "has_protection", "fiber_without_protection",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES + BINARY_FEATURES

X_train, y_train = train[FEATURES], train[TARGET]

# Ganzzahlige Spalten als float: verhindert MLflow-Schema-Warnungen bei fehlenden Werten im Scoring
X_train = X_train.astype({c: "float64" for c in NUMERIC_FEATURES + BINARY_FEATURES})

print(f"Train: {X_train.shape}, Churn-Rate: {y_train.mean():.1%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Preprocessing-Pipeline
# MAGIC `ColumnTransformer` lernt Encoding und Skalierung **innerhalb jedes CV-Folds nur auf den Trainingsdaten**
# MAGIC → kein Data Leakage.

# COMMAND ----------

def build_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    """Erstellt den Vorverarbeitungsschritt; Skalierung nur für lineare Modelle nötig."""
    numeric_step = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
            ("num", numeric_step, NUMERIC_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
        ]
    )


# Gewicht für die Minderheitsklasse (≈ 2,8): gleicht die Unbalance aus
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

MODELS = {
    "dummy": (DummyClassifier(strategy="stratified", random_state=RANDOM_STATE), False),
    "logreg": (LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0), True),
    "random_forest": (
        RandomForestClassifier(n_estimators=400, max_depth=8, min_samples_leaf=5,
                               class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE),
        False,
    ),
    "xgboost": (
        XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                      colsample_bytree=0.8, scale_pos_weight=pos_weight, eval_metric="logloss",
                      random_state=RANDOM_STATE, n_jobs=-1),
        False,
    ),
    "lightgbm": (
        LGBMClassifier(n_estimators=400, num_leaves=15, learning_rate=0.05, subsample=0.8,
                       subsample_freq=1, colsample_bytree=0.8, scale_pos_weight=pos_weight,
                       random_state=RANDOM_STATE, verbose=-1),
        False,
    ),
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Business-Metrik: Lift im Top-10 %
# MAGIC „Wie viel mehr Kündiger finden wir unter den 10 % riskantesten Kunden als bei Zufallsauswahl?“

# COMMAND ----------

def lift_at_k(y_true, y_score, k: float = 0.10) -> float:
    """Churn-Rate in den Top-k % nach Score geteilt durch die Gesamt-Churn-Rate."""
    y_true = np.asarray(y_true)
    n_top = max(1, int(len(y_true) * k))
    top_idx = np.argsort(y_score)[::-1][:n_top]
    return y_true[top_idx].mean() / y_true.mean()


SCORING = {
    "pr_auc": "average_precision",
    "roc_auc": "roc_auc",
    "recall": "recall",
    "precision": "precision",
    "f1": "f1",
    "lift_top10": make_scorer(lift_at_k, response_method="predict_proba"),
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Training & Kreuzvalidierung – jeder Lauf wird in MLflow protokolliert

# COMMAND ----------

results = []

for name, (estimator, scale_numeric) in MODELS.items():
    pipeline = Pipeline([("preprocessor", build_preprocessor(scale_numeric)), ("model", estimator)])

    with mlflow.start_run(run_name=name):
        cv_scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=SCORING, n_jobs=1)

        # Mittelwert und Streuung je Metrik protokollieren
        metrics = {}
        for metric in SCORING:
            values = cv_scores[f"test_{metric}"]
            metrics[f"cv_{metric}_mean"] = float(np.mean(values))
            metrics[f"cv_{metric}_std"] = float(np.std(values))
        mlflow.log_metrics(metrics)

        mlflow.log_params({"model_type": name, "n_features": len(FEATURES), "cv_folds": 5})
        mlflow.log_params({f"hp_{k}": v for k, v in estimator.get_params().items()
                           if isinstance(v, (int, float, str, bool)) and v is not None})

        # Finales Modell auf allem Train-Daten fitten und mit Signatur speichern
        pipeline.fit(X_train, y_train)
        signature = infer_signature(X_train, pipeline.predict_proba(X_train)[:, 1])
        # cloudpickle: funktioniert mit allen sklearn-/XGBoost-/LightGBM-Objekten in der Pipeline
        mlflow.sklearn.log_model(pipeline, name="model", signature=signature,
                                 input_example=X_train.head(3), serialization_format="cloudpickle")

        results.append({"model": name, **{k.replace("cv_", ""): v for k, v in metrics.items()
                                          if k.endswith("_mean")}})
        print(f"{name:14s} PR-AUC={metrics['cv_pr_auc_mean']:.3f}  ROC-AUC={metrics['cv_roc_auc_mean']:.3f}  "
              f"Recall={metrics['cv_recall_mean']:.3f}  Lift@10%={metrics['cv_lift_top10_mean']:.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Modellvergleich

# COMMAND ----------

comparison = (
    pd.DataFrame(results)
    .set_index("model")
    .round(3)
    .sort_values("pr_auc_mean", ascending=False)
)
display(comparison.reset_index())

# COMMAND ----------

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 3.5))
plot_df = comparison[["pr_auc_mean", "roc_auc_mean", "recall_mean"]].iloc[::-1]
plot_df.plot.barh(ax=ax, color=["#E45756", "#4C78A8", "#72B7B2"])
ax.set_xlim(0, 1)
ax.set_title("Modellvergleich (5-fache CV auf Train)")
ax.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Ergebnisse in der MLflow-UI ansehen
# MAGIC Linke Leiste → **Experiments** → `churn-prediction` → Läufe markieren → **Compare**.
# MAGIC
# MAGIC **Fragen zur Interpretation**
# MAGIC 1. Wie weit liegt die Logistische Regression hinter den Boosting-Modellen?
# MAGIC 2. Lohnt sich die zusätzliche Komplexität?
# MAGIC 3. Welches Modell hat die kleinste Streuung (`_std`) – also die stabilsten Ergebnisse?
