# Churn Prediction – Modellvergleich & Auswahl (Phase 4)

Stand: 12.10.2026 · Mustafa Öztemiz

## Kontext

Die Logistische Regression liegt gleichauf mit dem besten Modell; wir wählen sie als Champion. Verglichen wurden 5 Modelle per 5-facher stratifizierter Kreuzvalidierung auf dem Train-Split (5.644 Kunden, Churn-Rate 26,4 %). Der Test-Split bleibt bis Phase 5 unberührt.

| Modell | PR-AUC | ROC-AUC | Recall | Lift Top-10 % |
| --- | --- | --- | --- | --- |
| Random Forest | 0,650 | 0,845 | 0,777 | 2,85 |
| Logistische Regression | 0,649 | 0,845 | 0,799 | 2,81 |
| XGBoost | 0,641 | 0,838 | 0,738 | 2,83 |
| LightGBM | 0,635 | 0,836 | 0,723 | 2,83 |
| Dummy (Zufall) | 0,265 | 0,500 | 0,267 | 0,93 |

Werte aus dem lokalen Referenzlauf (Mittelwert über 5 Folds). Der Databricks-Lauf weicht leicht ab, weil `hash()` den Split anders bildet: PR-AUC dort ca. 0,66 (RF), 0,655 (LogReg), 0,65 (XGBoost), 0,64 (LightGBM).

## Frage 1: Wie weit liegt die Logistische Regression hinter den Boosting-Modellen?

Gar nicht – sie liegt vor beiden. Bei PR-AUC und ROC-AUC schlägt sie XGBoost und LightGBM, beim Recall ist sie mit ca. 0,80 klar die Beste: Sie erkennt rund 80 % der tatsächlichen Kündiger.

Gründe: Der Datensatz ist klein und die Zusammenhänge sind weitgehend linear (Vertragsart, tenure). Die stärksten Interaktionen wie `fiber_without_protection` haben wir im Feature Engineering bereits von Hand gebaut. Damit entfällt der Hauptvorteil von Boosting. Zudem sind die Boosting-Modelle noch nicht getunt.

## Frage 2: Lohnt sich die zusätzliche Komplexität?

Nein. Der Abstand zwischen Random Forest und Logistischer Regression beträgt nur ca. 0,005 PR-AUC, die Standardabweichung über die Folds aber ca. 0,025. Der Unterschied ist damit statistisch nicht signifikant: Beide Modelle liegen gleichauf.

Faustregel: Ist der Abstand der Mittelwerte kleiner als die Standardabweichung, nimm das einfachere Modell (Ockhams Rasiermesser).

Zusätzliche Vorteile der Logistischen Regression:

- **Erklärbarkeit:** Koeffizienten sind direkt interpretierbar, z. B. wie stark ein Zweijahresvertrag das Risiko senkt. Das unterstützt die Transparenzpflichten (EU AI Act, EuGH-Urteil 2025).
- **Leichtgewichtig:** läuft sofort in Streamlit und Databricks Apps.
- **Wartbar:** wenige Hyperparameter, wenig Überraschungen im Betrieb.

## Frage 3: Welches Modell ist am stabilsten?

Beim PR-AUC ist Random Forest am stabilsten, beim Recall die Logistische Regression; LightGBM schwankt beim Recall am stärksten. Insgesamt sind die Unterschiede klein, kein Modell ist auffällig instabil.

| Modell | Std. PR-AUC | Std. Recall |
| --- | --- | --- |
| Random Forest | **0,024** | 0,022 |
| Logistische Regression | 0,029 | **0,019** |
| XGBoost | 0,029 | 0,028 |
| LightGBM | 0,026 | 0,037 |

Werte aus dem lokalen Referenzlauf. Eigene Werte prüfen in der MLflow-UI: **Experiments → churn-prediction → 5 Runs markieren → Compare**, Metriken `cv_pr_auc_std` und `cv_recall_std`.

## Entscheidung und nächste Schritte

Wir gehen mit dem Champion-Challenger-Ansatz in Phase 5:

- **Champion: Logistische Regression** – Spitzenleistung, erklärbar, einfach.
- **Challenger: Random Forest** – höchster und stabilster PR-AUC.

Phase 5:

1. Beide Modelle einmalig auf dem Test-Split bewerten.
2. Schwellenwert anhand der Kampagnenkosten wählen (50 $ pro Angebot, 30 % Erfolgsquote, Jahresumsatz als Verlust).
3. Modell mit SHAP erklären.
4. Gewinner im Unity Catalog Model Registry registrieren.

Offene Frage: Soll XGBoost vorher noch eine Chance mit Hyperparameter-Tuning bekommen?
