# Databricks notebook source
# MAGIC %md
# MAGIC # Churn Prediction – Notebook 05: Evaluation, Schwellenwert, SHAP & Model Registry
# MAGIC
# MAGIC **Entscheidung aus Phase 4:** Champion = Logistische Regression, Challenger = Random Forest.
# MAGIC Kein weiteres Tuning von XGBoost: Der Abstand liegt innerhalb der CV-Streuung (siehe `docs/04_modellvergleich.md`).
# MAGIC
# MAGIC | Schritt | Frage |
# MAGIC |---|---|
# MAGIC | 1. Out-of-Fold-Vorhersagen | Wie verhalten sich die Modelle auf ungesehenen Train-Daten? |
# MAGIC | 2. Schwellenwert | Ab welchem Score lohnt sich eine Retention-Kampagne? (nur Train!) |
# MAGIC | 3. Test-Bewertung | Wie gut sind die Modelle wirklich? (Test-Split, einmalig) |
# MAGIC | 4. Kalibrierung | Kann man den Score als Wahrscheinlichkeit lesen? |
# MAGIC | 5. SHAP | Warum kündigt ein Kunde? |
# MAGIC | 6. Registry | Gewinner versioniert in Unity Catalog ablegen |

# COMMAND ----------

# Bibliothek installieren und Python neu starten
%pip install -q shap
dbutils.library.restartPython()

# COMMAND ----------

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
GOLD_TABLE = "workspace.churn.gold_churn_features"
MODEL_NAME = "workspace.churn.churn_model"  # Unity Catalog: katalog.schema.modell

user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user}/churn-prediction")

COLOR_CHAMP, COLOR_CHALL, COLOR_GREY = "#E45756", "#4C78A8", "#9D9D9D"
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Daten & Pipelines (identisch zu Notebook 04)
# MAGIC In Phase 6 (MLOps) wandert diese Konfiguration in ein Python-Modul (`src/churn/`),
# MAGIC damit Training, Scoring und App garantiert dieselbe Definition nutzen.

# COMMAND ----------

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

gold = spark.table(GOLD_TABLE).toPandas()
gold = gold.astype({c: "float64" for c in NUMERIC_FEATURES + BINARY_FEATURES})
train = gold[gold["split"] == "train"].reset_index(drop=True)
test = gold[gold["split"] == "test"].reset_index(drop=True)

X_train, y_train = train[FEATURES], train[TARGET]
X_test, y_test = test[FEATURES], test[TARGET]
print(f"Train: {len(train)} | Test: {len(test)}")


def build_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    """Vorverarbeitung: One-Hot für Kategorien, optional Skalierung für numerische Merkmale."""
    numeric_step = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ("num", numeric_step, NUMERIC_FEATURES),
        ("bin", "passthrough", BINARY_FEATURES),
    ])


CANDIDATES = {
    "logreg": Pipeline([
        ("preprocessor", build_preprocessor(scale_numeric=True)),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)),
    ]),
    "random_forest": Pipeline([
        ("preprocessor", build_preprocessor(scale_numeric=False)),
        ("model", RandomForestClassifier(n_estimators=400, max_depth=8, min_samples_leaf=5,
                                         class_weight="balanced", n_jobs=-1,
                                         random_state=RANDOM_STATE)),
    ]),
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Out-of-Fold-Vorhersagen auf Train
# MAGIC Jeder Train-Kunde bekommt einen Score von einem Modell, das ihn **nicht** gesehen hat.
# MAGIC Damit können wir den Schwellenwert wählen, ohne den Test-Split anzufassen.

# COMMAND ----------

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
oof_scores = {
    name: cross_val_predict(pipe, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    for name, pipe in CANDIDATES.items()
}
for name, score in oof_scores.items():
    print(f"{name:14s} OOF PR-AUC = {average_precision_score(y_train, score):.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Kostenbasierter Schwellenwert
# MAGIC **Annahmen (aus Phase 1, mit dem Fachbereich abzustimmen):**
# MAGIC - Retention-Angebot kostet **50 $** pro angesprochenem Kunden
# MAGIC - Das Angebot hält **30 %** der echten Kündiger
# MAGIC - Ein geretteter Kunde bringt **12 × monthly_charges** (Jahresumsatz)
# MAGIC
# MAGIC **Netto-Ersparnis** = Σ (gerettete Jahresumsätze) − Σ (Kampagnenkosten)

# COMMAND ----------

OFFER_COST = 50.0
SUCCESS_RATE = 0.30
REVENUE_MONTHS = 12


def campaign_profit(y_true, scores, monthly_charges, threshold: float) -> float:
    """Netto-Ersparnis einer Kampagne an alle Kunden mit Score ≥ Schwellenwert."""
    targeted = scores >= threshold
    saved_revenue = (SUCCESS_RATE * REVENUE_MONTHS * monthly_charges * y_true)[targeted].sum()
    return float(saved_revenue - OFFER_COST * targeted.sum())


def profit_curve(y_true, scores, monthly_charges, thresholds) -> np.ndarray:
    """Netto-Ersparnis für eine Liste von Schwellenwerten."""
    y_true, monthly_charges = np.asarray(y_true), np.asarray(monthly_charges)
    return np.array([campaign_profit(y_true, scores, monthly_charges, t) for t in thresholds])


thresholds = np.round(np.arange(0.05, 0.96, 0.01), 2)
mc_train = train["monthly_charges"].to_numpy()

best_threshold = {}
fig, ax = plt.subplots(figsize=(8, 3.8))
for name, color in [("logreg", COLOR_CHAMP), ("random_forest", COLOR_CHALL)]:
    profits = profit_curve(y_train, oof_scores[name], mc_train, thresholds)
    best_threshold[name] = float(thresholds[profits.argmax()])
    ax.plot(thresholds, profits / 1000, color=color, label=name)
    ax.scatter(best_threshold[name], profits.max() / 1000, color=color, zorder=3)
    print(f"{name:14s} bester Schwellenwert = {best_threshold[name]:.2f} "
          f"→ Netto-Ersparnis (Train, OOF) = {profits.max():,.0f} $")
ax.axvline(0.5, color=COLOR_GREY, linestyle="--", linewidth=1, label="Standard 0,5")
ax.set_xlabel("Schwellenwert")
ax.set_ylabel("Netto-Ersparnis (Tsd. $)")
ax.set_title("Kampagnen-Ersparnis je Schwellenwert (Train, Out-of-Fold)")
ax.legend()
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Finale Bewertung auf dem Test-Split (einmalig!)

# COMMAND ----------

def lift_at_k(y_true, scores, k: float = 0.10) -> float:
    """Churn-Rate in den Top-k % nach Score geteilt durch die Gesamt-Churn-Rate."""
    y_true = np.asarray(y_true)
    top = np.argsort(scores)[::-1][: max(1, int(len(y_true) * k))]
    return y_true[top].mean() / y_true.mean()


mc_test = test["monthly_charges"].to_numpy()
test_scores, test_rows = {}, []

for name, pipe in CANDIDATES.items():
    pipe.fit(X_train, y_train)
    scores = pipe.predict_proba(X_test)[:, 1]
    test_scores[name] = scores
    t = best_threshold[name]
    y_pred = (scores >= t).astype(int)
    test_rows.append({
        "model": name,
        "threshold": t,
        "pr_auc": average_precision_score(y_test, scores),
        "roc_auc": roc_auc_score(y_test, scores),
        "recall": recall_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "lift_top10": lift_at_k(y_test, scores),
        "kunden_angesprochen": int(y_pred.sum()),
        "netto_ersparnis": campaign_profit(y_test.to_numpy(), scores, mc_test, t),
        "ersparnis_bei_0_5": campaign_profit(y_test.to_numpy(), scores, mc_test, 0.5),
    })

test_results = pd.DataFrame(test_rows).round(3)
display(test_results)

# Vergleich: Kampagne an ALLE Kunden ohne Modell
all_profit = campaign_profit(y_test.to_numpy(), np.ones(len(y_test)), mc_test, 0.5)
print(f"Zum Vergleich – alle {len(y_test)} Testkunden ansprechen: {all_profit:,.0f} $")

# COMMAND ----------

# Precision-Recall-Kurve und kumulative Gewinnkurve (Gains Chart)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for name, color in [("logreg", COLOR_CHAMP), ("random_forest", COLOR_CHALL)]:
    prec, rec, _ = precision_recall_curve(y_test, test_scores[name])
    axes[0].plot(rec, prec, color=color,
                 label=f"{name} (PR-AUC {average_precision_score(y_test, test_scores[name]):.3f})")

    order = np.argsort(test_scores[name])[::-1]
    gains = np.cumsum(y_test.to_numpy()[order]) / y_test.sum()
    share = np.arange(1, len(order) + 1) / len(order)
    axes[1].plot(share * 100, gains * 100, color=color, label=name)

axes[0].axhline(y_test.mean(), color=COLOR_GREY, linestyle="--", linewidth=1, label="Zufall")
axes[0].set_xlabel("Recall")
axes[0].set_ylabel("Precision")
axes[0].set_title("Precision-Recall-Kurve (Test)")
axes[0].legend(fontsize=8)

axes[1].plot([0, 100], [0, 100], color=COLOR_GREY, linestyle="--", linewidth=1, label="Zufall")
axes[1].set_xlabel("Angesprochene Kunden (%)")
axes[1].set_ylabel("Gefundene Kündiger (%)")
axes[1].set_title("Gains Chart (Test)")
axes[1].legend(fontsize=8)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Gewinner wählen & Kalibrierung prüfen
# MAGIC Kriterium: höchste Netto-Ersparnis auf Test. Bei Gleichstand gewinnt das einfachere Modell.

# COMMAND ----------

ranked = test_results.sort_values("netto_ersparnis", ascending=False).reset_index(drop=True)
gap = ranked.loc[0, "netto_ersparnis"] - ranked.loc[1, "netto_ersparnis"]
# Unter 2 % Vorsprung zählt als Gleichstand → Champion (einfacher, erklärbar) behalten
WINNER = ranked.loc[0, "model"] if gap > 0.02 * abs(ranked.loc[1, "netto_ersparnis"]) else "logreg"
WINNER_THRESHOLD = best_threshold[WINNER]
winner_pipe = CANDIDATES[WINNER]
print(f"Gewinner: {WINNER} (Schwellenwert {WINNER_THRESHOLD:.2f}, Vorsprung {gap:,.0f} $)")

# Kalibrierung: Stimmt „Score 0,6“ mit „60 % kündigen tatsächlich“ überein?
frac_pos, mean_pred = calibration_curve(y_test, test_scores[WINNER], n_bins=10, strategy="quantile")
fig, ax = plt.subplots(figsize=(4.5, 4))
ax.plot([0, 1], [0, 1], color=COLOR_GREY, linestyle="--", linewidth=1, label="perfekt kalibriert")
ax.plot(mean_pred, frac_pos, marker="o", color=COLOR_CHAMP, label=WINNER)
ax.set_xlabel("Vorhergesagter Score")
ax.set_ylabel("Tatsächliche Churn-Rate")
ax.set_title("Kalibrierungskurve (Test)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. SHAP – Warum kündigt ein Kunde?
# MAGIC **Global:** Welche Merkmale treiben das Risiko insgesamt? **Lokal:** Warum hat *dieser* Kunde einen hohen Score?

# COMMAND ----------

import shap

preprocessor = winner_pipe.named_steps["preprocessor"]
model = winner_pipe.named_steps["model"]
feature_names = [n.split("__", 1)[1] for n in preprocessor.get_feature_names_out()]

X_train_t = pd.DataFrame(preprocessor.transform(X_train), columns=feature_names)
X_test_t = pd.DataFrame(preprocessor.transform(X_test), columns=feature_names)

if WINNER == "logreg":
    explainer = shap.LinearExplainer(model, X_train_t)
    shap_values = explainer(X_test_t)
else:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test_t)[:, :, 1]  # Klasse 1 = Churn

shap.plots.beeswarm(shap_values, max_display=15, show=False)
plt.title("SHAP: globale Einflussfaktoren (Test)")
plt.tight_layout()
plt.show()

# COMMAND ----------

# Lokale Erklärung: der riskanteste Testkunde
top_idx = int(np.argmax(test_scores[WINNER]))
print(f"Kunde {test.loc[top_idx, 'customer_id']} – Score {test_scores[WINNER][top_idx]:.2f}, "
      f"tatsächlich gekündigt: {bool(y_test[top_idx])}")
shap.plots.waterfall(shap_values[top_idx], max_display=10, show=False)
plt.tight_layout()
plt.show()

# COMMAND ----------

# Globale Wichtigkeit als Tabelle (mittlerer |SHAP|-Wert) – für den Business-Report
importance = (
    pd.DataFrame({"merkmal": feature_names, "mean_abs_shap": np.abs(shap_values.values).mean(axis=0)})
    .sort_values("mean_abs_shap", ascending=False)
    .head(10)
    .round(4)
)
display(importance)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Finales Modell trainieren & in Unity Catalog registrieren
# MAGIC Für den Produktivbetrieb wird auf **Train + Test** neu trainiert (mehr Daten, gleiche Konfiguration).
# MAGIC Der Schwellenwert wird als Tag an der Modellversion gespeichert.

# COMMAND ----------

from sklearn.base import clone

mlflow.set_registry_uri("databricks-uc")

X_all = pd.concat([X_train, X_test], ignore_index=True)
y_all = pd.concat([y_train, y_test], ignore_index=True)
final_pipe = clone(CANDIDATES[WINNER]).fit(X_all, y_all)

winner_metrics = test_results.set_index("model").loc[WINNER]

with mlflow.start_run(run_name=f"final_{WINNER}") as run:
    mlflow.log_params({
        "model_type": WINNER, "threshold": WINNER_THRESHOLD, "offer_cost": OFFER_COST,
        "success_rate": SUCCESS_RATE, "trained_on": "train+test",
    })
    mlflow.log_metrics({f"test_{k}": float(winner_metrics[k]) for k in
                        ["pr_auc", "roc_auc", "recall", "precision", "f1", "lift_top10", "netto_ersparnis"]})
    mlflow.log_table(importance, artifact_file="shap_importance.json")

    signature = infer_signature(X_all, final_pipe.predict_proba(X_all)[:, 1])
    model_info = mlflow.sklearn.log_model(
        final_pipe, name="model", signature=signature, input_example=X_all.head(3),
        serialization_format="cloudpickle", registered_model_name=MODEL_NAME,
    )

client = mlflow.MlflowClient()
version = model_info.registered_model_version
client.set_model_version_tag(MODEL_NAME, version, "threshold", str(WINNER_THRESHOLD))
client.set_model_version_tag(MODEL_NAME, version, "model_type", WINNER)
client.set_registered_model_alias(MODEL_NAME, "champion", version)
print(f"Registriert: {MODEL_NAME} v{version} mit Alias @champion")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Kontrolle: Modell über den Alias laden (so lädt es später die App)

# COMMAND ----------

loaded = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")
sample = X_test.head(5)
print(pd.DataFrame({
    "customer_id": test["customer_id"].head(5),
    "churn_proba": loaded.predict_proba(sample)[:, 1].round(3),
    "kampagne": loaded.predict_proba(sample)[:, 1] >= WINNER_THRESHOLD,
}))
