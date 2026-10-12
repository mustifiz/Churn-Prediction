# Databricks notebook source
# MAGIC %md
# MAGIC # Churn Prediction – Notebook 01: Setup, Bronze & Silver
# MAGIC
# MAGIC **Ziel:** Rohdaten in die Lakehouse-Architektur (Medallion) laden und bereinigen.
# MAGIC
# MAGIC | Schicht | Tabelle | Inhalt |
# MAGIC |---|---|---|
# MAGIC | Bronze | `workspace.churn.bronze_customers` | Rohdaten 1:1 aus der CSV |
# MAGIC | Silver | `workspace.churn.silver_customers` | bereinigt, typisiert, Zielvariable 0/1 |
# MAGIC | Gold   | (Notebook 03) | Features für das Modell |
# MAGIC
# MAGIC **Ablauf:** Zelle 1 ausführen → CSV in das Volume hochladen → restliche Zellen ausführen.

# COMMAND ----------

# Konfiguration: Katalog, Schema und Volume
CATALOG = "workspace"
SCHEMA = "churn"
VOLUME = "raw"

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}")

RAW_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/Telco-Customer-Churn.csv"
print("Bitte die CSV hier hochladen:", RAW_PATH)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Schritt 2 – CSV hochladen
# MAGIC **Catalog → workspace → churn → Volumes → raw → „Upload to this volume“** → Datei `telco_churn.csv` auswählen.
# MAGIC Danach die nächste Zelle ausführen.

# COMMAND ----------

# Prüfen, ob die Datei im Volume liegt
display(dbutils.fs.ls(f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bronze – Rohdaten unverändert speichern

# COMMAND ----------

from pyspark.sql import functions as F

# Alle Spalten als String einlesen: Bronze = unveränderte Rohdaten
bronze_df = (
    spark.read.option("header", True)
    .option("inferSchema", False)
    .csv(RAW_PATH)
    .withColumn("_ingested_at", F.current_timestamp())
)

bronze_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.bronze_customers")
print("Bronze-Zeilen:", bronze_df.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Silver – Bereinigung & Typisierung
# MAGIC - `TotalCharges` enthält Leerzeichen bei Neukunden (tenure = 0) → 0.0
# MAGIC - Numerische Spalten casten
# MAGIC - Zielvariable `churn` als 0/1
# MAGIC - Spaltennamen in snake_case

# COMMAND ----------

import re

def to_snake_case(name: str) -> str:
    """Wandelt CamelCase-Spaltennamen in snake_case um."""
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    return name.lower()

bronze = spark.table(f"{CATALOG}.{SCHEMA}.bronze_customers")

silver_df = (
    bronze
    .withColumn("TotalCharges", F.when(F.trim("TotalCharges") == "", None).otherwise(F.col("TotalCharges")))
    .withColumn("TotalCharges", F.coalesce(F.col("TotalCharges").cast("double"), F.lit(0.0)))
    .withColumn("MonthlyCharges", F.col("MonthlyCharges").cast("double"))
    .withColumn("tenure", F.col("tenure").cast("int"))
    .withColumn("SeniorCitizen", F.col("SeniorCitizen").cast("int"))
    .withColumn("Churn", F.when(F.col("Churn") == "Yes", 1).otherwise(0))
    .drop("_ingested_at")
)

# Spalten umbenennen (customerID → customer_id usw.)
for col in silver_df.columns:
    silver_df = silver_df.withColumnRenamed(col, to_snake_case(col))

silver_df.write.mode("overwrite").option("overwriteSchema", True).saveAsTable(
    f"{CATALOG}.{SCHEMA}.silver_customers"
)
display(silver_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Datenqualitäts-Checks

# COMMAND ----------

silver = spark.table(f"{CATALOG}.{SCHEMA}.silver_customers")

# Fehlende Werte pro Spalte
null_counts = silver.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in silver.columns])
display(null_counts)

# Eindeutigkeit des Schlüssels prüfen
n_rows, n_ids = silver.count(), silver.select("customer_id").distinct().count()
assert n_rows == n_ids, "Doppelte customer_id gefunden!"
print(f"OK – {n_rows} Zeilen, keine Duplikate.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Business-KPIs (Phase 1 – Ist-Analyse)
# MAGIC Diese Zahlen kommen ins Business-Understanding-Dokument.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   COUNT(*)                                                AS kunden_gesamt,
# MAGIC   SUM(churn)                                              AS kuendiger,
# MAGIC   ROUND(AVG(churn) * 100, 1)                              AS churn_rate_pct,
# MAGIC   ROUND(SUM(monthly_charges))                             AS monatsumsatz_gesamt,
# MAGIC   ROUND(SUM(CASE WHEN churn = 1 THEN monthly_charges END)) AS monatsumsatz_verloren
# MAGIC FROM workspace.churn.silver_customers

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Churn-Rate nach Vertragsart (Visualisierung: „+“ → Bar Chart)
# MAGIC SELECT contract,
# MAGIC        COUNT(*)                   AS kunden,
# MAGIC        ROUND(AVG(churn) * 100, 1) AS churn_rate_pct
# MAGIC FROM workspace.churn.silver_customers
# MAGIC GROUP BY contract
# MAGIC ORDER BY churn_rate_pct DESC

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Churn-Rate nach Kundendauer (tenure in Monaten)
# MAGIC SELECT CASE
# MAGIC          WHEN tenure <= 6  THEN '01: 0-6'
# MAGIC          WHEN tenure <= 12 THEN '02: 7-12'
# MAGIC          WHEN tenure <= 24 THEN '03: 13-24'
# MAGIC          WHEN tenure <= 48 THEN '04: 25-48'
# MAGIC          ELSE '05: 49-72'
# MAGIC        END                        AS tenure_gruppe,
# MAGIC        COUNT(*)                   AS kunden,
# MAGIC        ROUND(AVG(churn) * 100, 1) AS churn_rate_pct
# MAGIC FROM workspace.churn.silver_customers
# MAGIC GROUP BY 1
# MAGIC ORDER BY 1
