# 📉 Churn Prediction – End-to-End Data-Science-Projekt

Frühwarnsystem für Kundenabwanderung in der Telekommunikation: vom Business Understanding über ein
**Databricks Lakehouse (Medallion-Architektur)** und **MLflow** bis zur **Streamlit-App** mit
**CI/CD über GitHub Actions**.

> **Ergebnis:** Das Modell findet **86 % der Kündiger** und verdoppelt die Netto-Ersparnis einer
> Retention-Kampagne gegenüber „alle Kunden ansprechen“ (51.400 $ vs. 27.500 $ auf dem Test-Split).

## Business-Problem

Ein Telco-Anbieter verliert **26,5 %** seiner Kunden – rund **30 % des Monatsumsatzes**.
Ziel ist ein Risiko-Score pro Kunde, damit das Retention-Team die richtigen Kunden mit der
richtigen Maßnahme anspricht. Details: [`docs/01_business_understanding.md`](docs/01_business_understanding.md).

## Architektur

```
Telco-CSV ─► Bronze ─► Silver ─► Gold ─► Modell (MLflow, Unity Catalog @champion) ─► churn_scores
              (Databricks Lakehouse, Notebooks 01–06)                                   │
                                                                                         ▼
GitHub: src/churn (gleiche Logik als Python-Paket) ─► pytest ─► GitHub Actions ─► Streamlit-App
```

| Phase | Inhalt | Ort |
|---|---|---|
| 1 Business Understanding | KPIs, Erfolgsmetriken, Kostenannahmen | `docs/01_*` |
| 2 Daten & EDA | Bronze/Silver, Visualisierung | `databricks/01_*`, `02_*` |
| 3 Feature Engineering | Gold-Layer in SQL | `databricks/03_*`, `src/churn/features.py` |
| 4 Modellierung | 5 Modelle, 5-fache CV, MLflow Tracking | `databricks/04_*`, `docs/04_*` |
| 5 Evaluation | Kostenbasierter Schwellenwert, SHAP, Model Registry | `databricks/05_*` |
| 6 Scoring | Batch-Scoring mit Gründen pro Kunde | `databricks/06_*`, `src/churn/score.py` |
| 7 App | Streamlit: Übersicht, Was-wäre-wenn, Modell | `app/streamlit_app.py` |
| 8 CI/CD | Lint, Tests, Quality Gate, Deployment | `.github/workflows/` |

## Ergebnisse

| Modell (5-fache CV) | PR-AUC | ROC-AUC | Recall |
|---|---|---|---|
| **Logistische Regression (Champion)** | 0,649 | 0,845 | 0,799 |
| Random Forest | 0,650 | 0,845 | 0,777 |
| XGBoost | 0,641 | 0,838 | 0,738 |
| LightGBM | 0,635 | 0,836 | 0,723 |

Die Logistische Regression liegt gleichauf mit dem besten Modell (Abstand < CV-Streuung) und ist
erklärbar – deshalb Champion. Der Schwellenwert **0,39** maximiert die Netto-Ersparnis
(50 $ pro Angebot, 30 % Erfolgsquote, Jahresumsatz als Wert eines geretteten Kunden).

**Wichtigste Treiber:** kurze Kundendauer, Monatsvertrag, wenige Zusatzdienste, hohe Monatsgebühr,
Glasfaser ohne Schutzdienste, Zahlung per Electronic check.

## Schnellstart

```bash
pip install -r requirements-dev.txt
make train     # Modell trainieren → models/
make score     # aktive Kunden bewerten → data/processed/churn_scores.csv
make test      # 23 Tests inkl. Quality Gate und App-Smoke-Test
make app       # Streamlit-App starten
```

## Projektstruktur

```
├── app/streamlit_app.py      # Streamlit-App
├── databricks/               # Notebooks 01–06 (Lakehouse, MLflow, Registry)
├── data/raw/                 # IBM Telco Customer Churn
├── docs/                     # Business Understanding, Modellvergleich
├── models/                   # trainiertes Modell + Metadaten
├── src/churn/                # config, data, features, model, evaluate, explain, train, score
└── tests/                    # pytest
```

## Grenzen

- **Snapshot-Daten** ohne Zeitbezug: kein echtes Vorhersagefenster; aktive Kunden werden in-sample bewertet.
- **Kalibrierung:** `class_weight="balanced"` → Scores liegen über den echten Wahrscheinlichkeiten.
- **Kostenannahmen** sind Platzhalter und mit dem Fachbereich abzustimmen.
- **Kapazität:** 37 % der aktiven Kunden liegen über dem Schwellenwert – in der Praxis begrenzt die
  Teamkapazität die Liste (`priority_rank`).

## Daten

[IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d) – 7.043 Kunden, 21 Merkmale.

---
Mustafa Öztemiz · Bremen
