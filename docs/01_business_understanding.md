# Phase 1 – Business Understanding: Churn Prediction (Telco)

## 1. Ausgangslage
Ein Telekommunikationsanbieter mit 7.043 Kunden verliert jeden Monat einen erheblichen Teil seiner Kunden.
Neukundengewinnung ist deutlich teurer als Kundenbindung. Ziel ist ein **Frühwarnsystem**, das pro Kunde
einen **Risiko-Score (0–1)** berechnet, damit das Retention-Team gefährdete Kunden gezielt ansprechen kann.

## 2. Ist-Analyse (aus den Daten)
| Kennzahl | Wert |
|---|---|
| Kunden gesamt | 7.043 |
| Kündiger (Churn = Yes) | 1.869 (26,5 %) |
| Monatsumsatz gesamt | ca. 456.000 $ |
| Monatsumsatz der Kündiger | ca. 139.000 $ (≈ 30 %) |
| Churn-Rate Monatsvertrag / 1 Jahr / 2 Jahre | 42,7 % / 11,3 % / 2,8 % |
| Churn-Rate Neukunden (≤ 6 Monate) | 52,9 % |

## 3. Business-Fragen
1. **Wer** kündigt wahrscheinlich? → individueller Risiko-Score
2. **Warum** kündigt ein Kunde? → Modellinterpretation (SHAP)
3. **Was** tun wir? → Maßnahmen je Segment (z. B. Rabatt für Wechsel in Jahresvertrag)

## 4. ML-Problemformulierung
- **Typ:** Binäre Klassifikation, Zielvariable `Churn` (Yes = 1)
- **Output:** Wahrscheinlichkeit `churn_proba` ∈ [0, 1]
- **Klassenverteilung:** unbalanciert (≈ 27 / 73) → Accuracy allein ist irreführend

## 5. Erfolgsmetriken
**Technisch (Modellgüte)**
- Primär: **PR-AUC** und **Recall** der Klasse Churn
- Sekundär: ROC-AUC, Precision, F1
- **Lift im Top-10 %-Segment** (wie viel mehr Kündiger als Zufallsauswahl)

**Business**
- Kostenbasierte Schwellenwertwahl:
  - Annahme: Retention-Angebot kostet **50 $** pro Kunde
  - Verlorener Kunde = **12 × MonthlyCharges** (Jahresumsatz)
  - Erfolgsquote des Angebots: **30 %**
- Ziel: **Netto-Ersparnis** der Kampagne maximieren, nicht nur die Accuracy

> Die Kostenannahmen sind Platzhalter und müssen mit dem Fachbereich abgestimmt werden.

## 6. Datenquellen (vorhanden vs. ideal)
| Kategorie | Im Datensatz | In der Praxis zusätzlich |
|---|---|---|
| Demografie | gender, SeniorCitizen, Partner, Dependents | Alter, Region |
| Vertrag | Contract, tenure, PaymentMethod, PaperlessBilling | Tarifwechsel, Vertragsende |
| Nutzung | Internet-/Telefon-/Streaming-Services | Nutzungsverlauf, Rückgang der Nutzung |
| Transaktionen | MonthlyCharges, TotalCharges | Mahnungen, Zahlungsverzug |
| Kundenservice | – | Tickets, Beschwerden, Sentiment |

## 7. Einschränkungen & Risiken
- **Snapshot-Daten:** kein Zeitbezug → kein echtes Vorhersagefenster (z. B. „Kündigung in 2 Monaten“)
- **Data Leakage:** `TotalCharges` ≈ `tenure × MonthlyCharges` – zulässig, aber prüfen
- **Fairness / EU AI Act:** Kein Hochrisiko-System, aber Logik und Merkmale dokumentieren;
  `gender` ggf. aus dem Modell ausschließen
- 11 fehlende Werte in `TotalCharges` (Kunden mit tenure = 0)

## 8. Übergabe an den Fachbereich (von Anfang an mitdenken)
- Output: wöchentliche Liste „Top-Risiko-Kunden“ + Hauptgründe pro Kunde
- Kanal: Streamlit-Dashboard (später: CRM-Export als CSV)
- Verantwortlich: Retention-Team

## 9. Projektplan
| Phase | Inhalt | Ergebnis |
|---|---|---|
| 1 | Business Understanding | dieses Dokument ✅ |
| 2 | EDA & Visualisierung | Notebook + Insights |
| 3 | Feature Engineering & Pipeline | `src/churn/features.py` |
| 4 | Modellierung (LogReg → RF → XGBoost/LightGBM) | Modellvergleich |
| 5 | Evaluation, Threshold, SHAP | Business-Report |
| 6 | MLOps (MLflow, Projektstruktur, Tests) | reproduzierbares Training |
| 7 | Streamlit-App | Score + Dashboard |
| 8 | CI/CD (GitHub Actions, Docker, Deploy) | Live-App |
