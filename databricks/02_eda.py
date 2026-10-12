# Databricks notebook source
# MAGIC %md
# MAGIC # Churn Prediction – Notebook 02: Explorative Datenanalyse (EDA)
# MAGIC
# MAGIC **Ziel:** Treiber der Kundenabwanderung verstehen, bevor wir modellieren.
# MAGIC
# MAGIC | Abschnitt | Frage |
# MAGIC |---|---|
# MAGIC | 1. Zielvariable | Wie unbalanciert ist `churn`? |
# MAGIC | 2. Numerische Merkmale | Unterscheiden sich tenure / Kosten zwischen Kündigern und Bestandskunden? |
# MAGIC | 3. Kategoriale Merkmale | Welche Gruppen haben die höchste Churn-Rate? |
# MAGIC | 4. Korrelation | Gibt es redundante Merkmale (Multikollinearität)? |
# MAGIC | 5. Insights | Was bedeutet das für Business und Modell? |

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TABLE = "workspace.churn.silver_customers"

# Silver-Tabelle nach pandas laden (7.043 Zeilen → passt problemlos in den Speicher)
df = spark.table(TABLE).toPandas()

COLOR_STAY, COLOR_CHURN = "#4C78A8", "#E45756"
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})

print(df.shape)
df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Zielvariable – Klassenverteilung

# COMMAND ----------

churn_counts = df["churn"].value_counts().sort_index()
churn_share = churn_counts / churn_counts.sum() * 100

fig, ax = plt.subplots(figsize=(5, 3.5))
bars = ax.bar(["Bleibt (0)", "Kündigt (1)"], churn_counts.values, color=[COLOR_STAY, COLOR_CHURN])
for bar, pct in zip(bars, churn_share.values):
    ax.annotate(f"{pct:.1f} %", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center", va="bottom", fontsize=10)
ax.set_title("Klassenverteilung der Zielvariable")
ax.set_ylabel("Anzahl Kunden")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Numerische Merkmale nach Churn

# COMMAND ----------

numeric_cols = ["tenure", "monthly_charges", "total_charges"]

fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
for ax, col in zip(axes, numeric_cols):
    ax.hist(df.loc[df["churn"] == 0, col], bins=30, alpha=0.6, density=True,
            color=COLOR_STAY, label="Bleibt")
    ax.hist(df.loc[df["churn"] == 1, col], bins=30, alpha=0.6, density=True,
            color=COLOR_CHURN, label="Kündigt")
    ax.set_title(col)
    ax.set_yticks([])
axes[0].legend()
fig.suptitle("Verteilung numerischer Merkmale (normiert)", y=1.02)
plt.tight_layout()
plt.show()

# Mittelwerte und Mediane im Vergleich
summary = df.groupby("churn")[numeric_cols].agg(["mean", "median"]).round(1)
summary.columns = [f"{col}_{stat}" for col, stat in summary.columns]  # MultiIndex abflachen
display(summary.reset_index())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Kategoriale Merkmale – Churn-Rate je Ausprägung
# MAGIC Die gestrichelte Linie zeigt die durchschnittliche Churn-Rate. Balken darüber = Risikogruppe.

# COMMAND ----------

categorical_cols = [
    "contract", "internet_service", "payment_method", "online_security",
    "tech_support", "paperless_billing", "senior_citizen", "partner",
    "dependents", "streaming_tv", "multiple_lines", "gender",
]
overall_rate = df["churn"].mean() * 100

fig, axes = plt.subplots(4, 3, figsize=(15, 14))
for ax, col in zip(axes.flat, categorical_cols):
    rates = df.groupby(col)["churn"].mean().mul(100).sort_values()
    colors = [COLOR_CHURN if r > overall_rate else COLOR_STAY for r in rates.values]
    ax.barh(rates.index.astype(str), rates.values, color=colors)
    ax.axvline(overall_rate, color="grey", linestyle="--", linewidth=1)
    for i, r in enumerate(rates.values):
        ax.text(r + 0.5, i, f"{r:.0f} %", va="center", fontsize=8)
    ax.set_title(col, fontsize=10)
    ax.set_xlim(0, 60)
fig.suptitle("Churn-Rate (%) je Kategorie", fontsize=14, y=1.0)
plt.tight_layout()
plt.show()

# COMMAND ----------

# Churn-Rate nach Kundendauer (tenure) als Linie: zeigt das Risiko über den Lebenszyklus
tenure_bins = pd.cut(df["tenure"], bins=range(0, 78, 6), include_lowest=True)
tenure_rate = df.groupby(tenure_bins, observed=True)["churn"].mean().mul(100)

fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot(range(len(tenure_rate)), tenure_rate.values, marker="o", color=COLOR_CHURN)
ax.set_xticks(range(len(tenure_rate)))
ax.set_xticklabels([str(b) for b in tenure_rate.index], rotation=45, fontsize=8)
ax.set_ylabel("Churn-Rate (%)")
ax.set_xlabel("tenure (Monate)")
ax.set_title("Churn-Rate sinkt mit zunehmender Kundendauer")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Korrelation numerischer Merkmale

# COMMAND ----------

corr = df[numeric_cols + ["senior_citizen", "churn"]].corr().round(2)

fig, ax = plt.subplots(figsize=(5.5, 4.5))
im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr)))
ax.set_xticklabels(corr.columns, rotation=45, ha="right")
ax.set_yticks(range(len(corr)))
ax.set_yticklabels(corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, corr.values[i, j], ha="center", va="center", fontsize=9)
fig.colorbar(im, ax=ax, shrink=0.8)
ax.set_title("Korrelationsmatrix")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Risikosegmente – Kombination der stärksten Treiber

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT contract,
# MAGIC        internet_service,
# MAGIC        CASE WHEN tenure <= 12 THEN 'Neukunde (≤12 M)' ELSE 'Bestand (>12 M)' END AS kundendauer,
# MAGIC        COUNT(*)                          AS kunden,
# MAGIC        ROUND(AVG(churn) * 100, 1)        AS churn_rate_pct,
# MAGIC        ROUND(SUM(CASE WHEN churn = 1 THEN monthly_charges END)) AS verlorener_monatsumsatz
# MAGIC FROM workspace.churn.silver_customers
# MAGIC GROUP BY 1, 2, 3
# MAGIC HAVING COUNT(*) >= 50
# MAGIC ORDER BY churn_rate_pct DESC
# MAGIC LIMIT 10

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Insights (nach dem Ausführen ergänzen)
# MAGIC
# MAGIC **Business**
# MAGIC 1. Monatsverträge sind der größte Risikofaktor → Anreize für Jahresverträge
# MAGIC 2. Die ersten 6–12 Monate sind kritisch → Onboarding-Programm für Neukunden
# MAGIC 3. Glasfaser-Kunden (Fiber optic) kündigen überdurchschnittlich → Preis/Leistung prüfen
# MAGIC 4. Fehlende Zusatzdienste (OnlineSecurity, TechSupport) erhöhen das Risiko → Bundles anbieten
# MAGIC 5. Electronic check als Zahlungsart korreliert mit Churn → Umstellung auf Lastschrift fördern
# MAGIC
# MAGIC **Modell**
# MAGIC - Klassen unbalanciert (≈ 27 %) → stratifizierter Split, PR-AUC, `class_weight` / `scale_pos_weight`
# MAGIC - `total_charges` ≈ `tenure × monthly_charges` → stark korreliert, für lineare Modelle ggf. weglassen
# MAGIC - `gender` zeigt kaum Unterschied → aus Fairness-Gründen nicht verwenden
# MAGIC - Viele binäre Dienste-Spalten → Feature „Anzahl Zusatzdienste“ bauen (Notebook 03)
