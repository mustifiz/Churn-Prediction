# Databricks notebook source
# MAGIC %md
# MAGIC # Churn Prediction – Notebook 03: Feature Engineering & Gold-Tabelle
# MAGIC
# MAGIC **Ziel:** Aus den EDA-Erkenntnissen fachlich sinnvolle Merkmale bauen und als
# MAGIC `workspace.churn.gold_churn_features` speichern – die einzige Datenquelle für Training und Scoring.
# MAGIC
# MAGIC | Neues Merkmal | Idee aus der EDA |
# MAGIC |---|---|
# MAGIC | `num_addon_services` | Kunden ohne Zusatzdienste kündigen häufiger |
# MAGIC | `has_protection` | OnlineSecurity / TechSupport senken das Risiko |
# MAGIC | `is_new_customer` | Die ersten 12 Monate sind kritisch |
# MAGIC | `contract_months` | Vertragslaufzeit als Zahl (1 / 12 / 24) |
# MAGIC | `is_auto_payment` | Automatische Zahlung = stärkere Bindung |
# MAGIC | `avg_monthly_spend` | Durchschnittlicher Monatsbetrag über die gesamte Laufzeit |
# MAGIC | `charge_increase` | Aktueller Preis höher als der historische Schnitt → Preiserhöhung? |
# MAGIC | `fiber_without_protection` | Risikokombination aus der Segmentanalyse |
# MAGIC
# MAGIC **Bewusst nicht im Gold-Layer:** One-Hot-Encoding und Skalierung.
# MAGIC Diese Schritte lernen Parameter aus den Trainingsdaten und gehören deshalb in die
# MAGIC sklearn-`Pipeline` (Notebook 04), sonst droht **Data Leakage**.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE workspace.churn.gold_churn_features
# MAGIC COMMENT 'Gold-Layer: Modellmerkmale für Churn Prediction (eine Zeile pro Kunde)'
# MAGIC AS
# MAGIC SELECT
# MAGIC   customer_id,
# MAGIC
# MAGIC   -- Demografie (gender bewusst ausgeschlossen: kein Signal, Fairness)
# MAGIC   senior_citizen,
# MAGIC   CASE WHEN partner    = 'Yes' THEN 1 ELSE 0 END AS has_partner,
# MAGIC   CASE WHEN dependents = 'Yes' THEN 1 ELSE 0 END AS has_dependents,
# MAGIC
# MAGIC   -- Vertrag
# MAGIC   tenure,
# MAGIC   contract,
# MAGIC   CASE contract WHEN 'Month-to-month' THEN 1 WHEN 'One year' THEN 12 ELSE 24 END AS contract_months,
# MAGIC   CASE WHEN tenure <= 12 THEN 1 ELSE 0 END                         AS is_new_customer,
# MAGIC   CASE WHEN paperless_billing = 'Yes' THEN 1 ELSE 0 END            AS paperless_billing,
# MAGIC   payment_method,
# MAGIC   CASE WHEN payment_method LIKE '%automatic%' THEN 1 ELSE 0 END    AS is_auto_payment,
# MAGIC
# MAGIC   -- Dienste
# MAGIC   internet_service,
# MAGIC   CASE WHEN phone_service = 'Yes' THEN 1 ELSE 0 END                AS has_phone,
# MAGIC   CASE WHEN multiple_lines = 'Yes' THEN 1 ELSE 0 END               AS has_multiple_lines,
# MAGIC   (  CASE WHEN online_security   = 'Yes' THEN 1 ELSE 0 END
# MAGIC    + CASE WHEN online_backup     = 'Yes' THEN 1 ELSE 0 END
# MAGIC    + CASE WHEN device_protection = 'Yes' THEN 1 ELSE 0 END
# MAGIC    + CASE WHEN tech_support      = 'Yes' THEN 1 ELSE 0 END
# MAGIC    + CASE WHEN streaming_tv      = 'Yes' THEN 1 ELSE 0 END
# MAGIC    + CASE WHEN streaming_movies  = 'Yes' THEN 1 ELSE 0 END )        AS num_addon_services,
# MAGIC   CASE WHEN online_security = 'Yes' OR tech_support = 'Yes' THEN 1 ELSE 0 END AS has_protection,
# MAGIC   CASE WHEN internet_service = 'Fiber optic'
# MAGIC         AND online_security = 'No' AND tech_support = 'No' THEN 1 ELSE 0 END AS fiber_without_protection,
# MAGIC
# MAGIC   -- Kosten
# MAGIC   monthly_charges,
# MAGIC   total_charges,
# MAGIC   ROUND(CASE WHEN tenure > 0 THEN total_charges / tenure ELSE monthly_charges END, 2) AS avg_monthly_spend,
# MAGIC   ROUND(monthly_charges
# MAGIC         - CASE WHEN tenure > 0 THEN total_charges / tenure ELSE monthly_charges END, 2) AS charge_increase,
# MAGIC
# MAGIC   -- Zielvariable
# MAGIC   churn,
# MAGIC
# MAGIC   -- Reproduzierbarer Train/Test-Split (≈ 80/20) per Hash der Kunden-ID:
# MAGIC   -- gleicher Kunde landet bei jedem Lauf im gleichen Split (Churn-Rate je Split unten prüfen)
# MAGIC   CASE WHEN pmod(hash(customer_id), 100) < 80 THEN 'train' ELSE 'test' END AS split
# MAGIC
# MAGIC FROM workspace.churn.silver_customers

# COMMAND ----------

# MAGIC %md
# MAGIC ## Qualitätschecks für den Gold-Layer

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Split-Größen und Churn-Rate je Split (sollte in beiden ≈ 26–27 % sein)
# MAGIC SELECT split,
# MAGIC        COUNT(*)                   AS kunden,
# MAGIC        ROUND(AVG(churn) * 100, 1) AS churn_rate_pct
# MAGIC FROM workspace.churn.gold_churn_features
# MAGIC GROUP BY split

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Wirken die neuen Merkmale? Churn-Rate je Anzahl Zusatzdienste
# MAGIC SELECT num_addon_services,
# MAGIC        COUNT(*)                   AS kunden,
# MAGIC        ROUND(AVG(churn) * 100, 1) AS churn_rate_pct
# MAGIC FROM workspace.churn.gold_churn_features
# MAGIC GROUP BY num_addon_services
# MAGIC ORDER BY num_addon_services

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Binäre Risikomerkmale im Vergleich
# MAGIC SELECT 'fiber_without_protection' AS merkmal, fiber_without_protection AS wert,
# MAGIC        COUNT(*) AS kunden, ROUND(AVG(churn) * 100, 1) AS churn_rate_pct
# MAGIC FROM workspace.churn.gold_churn_features GROUP BY 1, 2
# MAGIC UNION ALL
# MAGIC SELECT 'is_auto_payment', is_auto_payment, COUNT(*), ROUND(AVG(churn) * 100, 1)
# MAGIC FROM workspace.churn.gold_churn_features GROUP BY 1, 2
# MAGIC UNION ALL
# MAGIC SELECT 'is_new_customer', is_new_customer, COUNT(*), ROUND(AVG(churn) * 100, 1)
# MAGIC FROM workspace.churn.gold_churn_features GROUP BY 1, 2
# MAGIC ORDER BY merkmal, wert

# COMMAND ----------

# Technische Checks: keine Nullwerte, eindeutige Kunden-ID
from pyspark.sql import functions as F

gold = spark.table("workspace.churn.gold_churn_features")

null_total = gold.select(sum(F.sum(F.col(c).isNull().cast("int")) for c in gold.columns)).first()[0]
assert null_total == 0, f"{null_total} Nullwerte im Gold-Layer!"
assert gold.count() == gold.select("customer_id").distinct().count(), "Doppelte customer_id!"

print(f"OK – {gold.count()} Zeilen, {len(gold.columns)} Spalten, keine Nullwerte.")
gold.printSchema()
