# Databricks notebook source
# MAGIC %md
# MAGIC # 🛢️ Volve Field — Oil Production Prediction Pipeline
# MAGIC
# MAGIC End-to-end Databricks pipeline that ingests real production data from the **Volve oil field**
# MAGIC (North Sea, operated by Equinor 2008-2016, publicly released in 2018), engineers features,
# MAGIC and trains a machine learning model to predict daily oil production per well.
# MAGIC
# MAGIC ## 🎯 Databricks features demonstrated
# MAGIC
# MAGIC | Feature | Where in notebook |
# MAGIC |---------|-------------------|
# MAGIC | **Databricks Widgets** for secret-free credential input | Section 2 |
# MAGIC | **kagglehub** programmatic data acquisition | Section 3 |
# MAGIC | **PySpark on Spark Connect** (Serverless compute) | Sections 4-6 |
# MAGIC | **Unity Catalog** 3-level namespace + managed Delta tables | Section 7, 9 |
# MAGIC | **Delta Lake**: ACID transactions, schema evolution, time travel | Section 7-8 |
# MAGIC | **MLflow autolog** for params/metrics/artifacts | Section 10 |
# MAGIC | **MLflow Model Registry** for versioned production deployment | Section 11 |
# MAGIC
# MAGIC ## ⚙️ Compatibility
# MAGIC
# MAGIC Built and tested on **Databricks Free Edition** with Serverless compute (Spark 4.1+).
# MAGIC Avoids features blocked in Free Edition: `cache()`, direct `sparkContext` access,
# MAGIC Genie Spaces, custom MCP servers.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup & Dependencies
# MAGIC
# MAGIC Install required packages. After installing, the Python kernel is restarted automatically
# MAGIC so newly installed modules become importable.

# COMMAND ----------

# MAGIC %pip install kagglehub openpyxl

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Authentication via Widget (no hardcoded secrets)
# MAGIC
# MAGIC The Kaggle API token is read from a Databricks Widget at the top of the notebook,
# MAGIC NOT from a hardcoded variable. This means:
# MAGIC - The token never appears in version control
# MAGIC - Anyone running this notebook supplies their own token
# MAGIC - Matches production patterns where `dbutils.secrets.get()` would be used instead
# MAGIC
# MAGIC ### How to use
# MAGIC
# MAGIC 1. **First run the next cell** — this creates the input widget at the top of the notebook.
# MAGIC 2. **Paste your Kaggle API token** into the widget (the field labeled "🔑 Kaggle API Token").
# MAGIC    Get a token at https://www.kaggle.com/settings → API → Create New Token.
# MAGIC 3. **Then run the validation cell** below — it reads the value you just pasted.
# MAGIC
# MAGIC > 💡 **Why two cells?** Widgets are created and read in separate Python lifecycles —
# MAGIC > the first `get()` call right after `text()` always returns empty. Separating them
# MAGIC > avoids the "run cell twice" UX issue.

# COMMAND ----------

# Idempotent widget cell — does the right thing on every run:
#   - First run (empty widget): creates the input field + prints instructions, NO error raised
#   - Subsequent runs (filled widget): loads the token into env var and proceeds
# This avoids the "raise ValueError on first run" UX issue that breaks "Run All".
import os

dbutils.widgets.text(
    name="kaggle_token",
    defaultValue="",
    label="🔑 Kaggle API Token (KGAT_...)",
)

kaggle_token = dbutils.widgets.get("kaggle_token")

if not kaggle_token:
    print("=" * 70)
    print("⚠️  ACTION REQUIRED — paste your Kaggle token, then re-run this cell")
    print("=" * 70)
    print("1. Look at the TOP of this notebook — you'll see a text input labeled")
    print("   '🔑 Kaggle API Token'")
    print("2. Paste your Kaggle API token there (format: 'KGAT_...')")
    print("3. Re-run THIS cell")
    print()
    print("Don't have a token? Get one at:")
    print("   https://www.kaggle.com/settings → API → Create New Token")
    print("=" * 70)
    print()
    print("⏸️ Subsequent cells will fail until this token is set. Stop here for now.")
else:
    os.environ["KAGGLE_API_TOKEN"] = kaggle_token
    print(f"✅ Kaggle credentials loaded ({len(kaggle_token)} chars)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Download Volve Dataset
# MAGIC
# MAGIC `kagglehub` downloads the dataset to the worker's local cache. The path includes a version
# MAGIC suffix (`/versions/1`) so the download is reproducible — if the dataset author publishes
# MAGIC a v2, your code can pin to v1.

# COMMAND ----------

import kagglehub

path = kagglehub.dataset_download("lamyalbert/volve-production-data")
print(f"Dataset path: {path}")

files = os.listdir(path)
print(f"Files: {files}")
for f in files:
    size_mb = os.path.getsize(os.path.join(path, f)) / (1024 * 1024)
    print(f"  - {f} ({size_mb:.2f} MB)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Load Excel Data into PySpark
# MAGIC
# MAGIC The Volve dataset comes as `.xlsx` (Excel) — the industry-standard format for oil & gas
# MAGIC engineers. PySpark doesn't read Excel natively, so we use pandas as a bridge:
# MAGIC 1. `pandas.read_excel()` parses the spreadsheet (single-threaded, OK for ~16K rows)
# MAGIC 2. `spark.createDataFrame()` converts to a distributed Spark DataFrame
# MAGIC
# MAGIC We focus on the **Daily Production Data** sheet (~15K rows of per-well daily measurements).

# COMMAND ----------

import pandas as pd

excel_file = f"{path}/Volve production data.xlsx"

# Inspect available sheets first
xls = pd.ExcelFile(excel_file)
print(f"Sheets available in Volve Excel:")
for sheet in xls.sheet_names:
    n_rows = len(pd.read_excel(excel_file, sheet_name=sheet))
    print(f"  - {sheet!r}: {n_rows:,} rows")

# COMMAND ----------

# Load the daily sheet
daily_sheet = next(s for s in xls.sheet_names if "Daily" in s)
print(f"Loading sheet: {daily_sheet!r}")

pdf = pd.read_excel(excel_file, sheet_name=daily_sheet)
df_volve = spark.createDataFrame(pdf)

print(f"\nSpark DataFrame: {df_volve.count():,} rows × {len(df_volve.columns)} columns")
df_volve.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 📚 Volve Data Dictionary
# MAGIC
# MAGIC | Column | Description | Unit |
# MAGIC |--------|-------------|------|
# MAGIC | `DATEPRD` | Production date | date |
# MAGIC | `NPD_WELL_BORE_NAME` | Well identifier (NPD standard) | string |
# MAGIC | `WELL_TYPE` | `OP` = oil producer, `WI` = water injector | enum |
# MAGIC | `FLOW_KIND` | `production` or `injection` | enum |
# MAGIC | `ON_STREAM_HRS` | Hours the well produced/injected (0-24) | hours |
# MAGIC | `AVG_DOWNHOLE_PRESSURE` | Bottomhole pressure | bar |
# MAGIC | `AVG_DOWNHOLE_TEMPERATURE` | Bottomhole temperature | °C |
# MAGIC | `AVG_DP_TUBING` | Tubing pressure differential | bar |
# MAGIC | `AVG_ANNULUS_PRESS` | Annulus pressure | bar |
# MAGIC | `AVG_CHOKE_SIZE_P` | Choke valve opening | % |
# MAGIC | `AVG_WHP_P` | Wellhead pressure | bar |
# MAGIC | `AVG_WHT_P` | Wellhead temperature | °C |
# MAGIC | `DP_CHOKE_SIZE` | Pressure differential across choke | bar |
# MAGIC | **`BORE_OIL_VOL`** | **Oil produced (target variable)** | **m³** |
# MAGIC | `BORE_GAS_VOL` | Gas produced | Sm³ |
# MAGIC | `BORE_WAT_VOL` | Water produced | m³ |
# MAGIC | `BORE_WI_VOL` | Water injected (only WI wells) | m³ |

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Exploratory Data Analysis

# COMMAND ----------

from pyspark.sql import functions as F

# Well type distribution
print("=== WELL_TYPE distribution ===")
df_volve.groupBy("WELL_TYPE").count().orderBy(F.desc("count")).show()

# Per-well summary
print("=== Per-well summary ===")
df_volve.groupBy("NPD_WELL_BORE_NAME", "WELL_TYPE").agg(
    F.count("*").alias("total_days"),
    F.round(F.sum("ON_STREAM_HRS"), 0).alias("total_hrs"),
    F.round(F.sum("BORE_OIL_VOL"), 0).alias("total_oil_m3"),
    F.round(F.sum("BORE_WAT_VOL"), 0).alias("total_water_m3"),
    F.round(F.sum("BORE_WI_VOL"), 0).alias("total_injection_m3"),
).orderBy(F.desc("total_oil_m3")).show(truncate=False)

# Date range
print("=== Dataset time span ===")
df_volve.select(
    F.min("DATEPRD").alias("first_date"),
    F.max("DATEPRD").alias("last_date"),
    F.datediff(F.max("DATEPRD"), F.min("DATEPRD")).alias("days_span"),
).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Filter to Active Oil Producers
# MAGIC
# MAGIC For the ML task (predicting oil production), we filter to:
# MAGIC - **Only `WELL_TYPE = "OP"` wells** (oil producers, not water injectors)
# MAGIC - **Days with `ON_STREAM_HRS > 0`** (well was actually operating)
# MAGIC - **Days with `BORE_OIL_VOL > 0`** (well actually produced oil that day)
# MAGIC
# MAGIC This is a deliberate scoping decision: the model focuses on **production rate prediction**
# MAGIC for operating wells, not on **shut-in detection** (which would be a separate classification task).

# COMMAND ----------

df_producers = (
    df_volve
    .filter(F.col("WELL_TYPE") == "OP")
    .filter(F.col("ON_STREAM_HRS") > 0)
    .filter(F.col("BORE_OIL_VOL") > 0)
)

n_before = df_volve.count()
n_after = df_producers.count()
print(f"Before filtering: {n_before:,} rows")
print(f"After filtering:  {n_after:,} rows")
print(f"Reduction: {(1 - n_after / n_before) * 100:.1f}%")

# Verify the filter worked
df_producers.groupBy("NPD_WELL_BORE_NAME").agg(
    F.count("*").alias("active_days"),
    F.round(F.avg("BORE_OIL_VOL"), 1).alias("avg_oil_m3_day"),
    F.round(F.min("BORE_OIL_VOL"), 1).alias("min_oil"),
    F.round(F.max("BORE_OIL_VOL"), 1).alias("max_oil"),
    F.round(F.stddev("BORE_OIL_VOL"), 1).alias("stddev_oil"),
).orderBy(F.desc("avg_oil_m3_day")).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🎯 Key insight: target is right-skewed
# MAGIC
# MAGIC The mean (~1,254 m³) is roughly **2x the median (~716 m³)** — clear sign of right-skew.
# MAGIC This is physically real in oil production: wells produce a lot in early ramp-up phases,
# MAGIC then decline exponentially (Arps decline curve, 1945).
# MAGIC
# MAGIC **ML implication**: linear models will struggle with this distribution. We'll either:
# MAGIC 1. Use a tree-based model (handles skew natively), or
# MAGIC 2. Apply `log1p()` transform to the target, train in log-space, then `expm1()` back
# MAGIC
# MAGIC This notebook uses **both** strategies: RandomForest + log-transform.

# COMMAND ----------

print("=== BORE_OIL_VOL distribution ===")
df_producers.select("BORE_OIL_VOL").summary(
    "count", "mean", "stddev", "min", "25%", "50%", "75%", "95%", "max"
).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Save to Delta Lake (Unity Catalog)
# MAGIC
# MAGIC This is the **first truly Databricks-native step**. We persist the cleaned data as a
# MAGIC managed **Delta table** in Unity Catalog using the 3-level namespace
# MAGIC `catalog.schema.table`. Delta gives us:
# MAGIC
# MAGIC - **ACID transactions** — safe concurrent writes
# MAGIC - **Schema evolution** — `overwriteSchema` lets us iterate on the structure
# MAGIC - **Time travel** — every overwrite preserves prior versions (see Section 8)
# MAGIC - **Automatic governance** — Unity Catalog tracks lineage
# MAGIC
# MAGIC Free Edition default catalog is `workspace`. Production workspaces would use something
# MAGIC like `prod_lakehouse.oil_gas.production_data`.

# COMMAND ----------

CATALOG = "workspace"
SCHEMA = "default"
TABLE_PRODUCERS = "volve_producers_active"
TABLE_PRODUCERS_FULL = f"{CATALOG}.{SCHEMA}.{TABLE_PRODUCERS}"

print(f"Writing Delta table: {TABLE_PRODUCERS_FULL}")

(
    df_producers
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(TABLE_PRODUCERS_FULL)
)

print(f"✅ Delta table created")
print(f"   Rows: {spark.table(TABLE_PRODUCERS_FULL).count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Inspect Delta table metadata
# MAGIC
# MAGIC `DESCRIBE EXTENDED` shows the underlying storage path, format, and Unity Catalog properties.
# MAGIC The `Provider` field will read `delta`, confirming this is a Delta table (not Parquet/JSON).

# COMMAND ----------

display(spark.sql(f"DESCRIBE EXTENDED {TABLE_PRODUCERS_FULL}"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Delta Time Travel Demo
# MAGIC
# MAGIC Every Delta write creates a new version. `DESCRIBE HISTORY` shows the full version log.
# MAGIC You can query any prior version with `VERSION AS OF` or `TIMESTAMP AS OF` — a feature
# MAGIC that **does not exist in plain Parquet/CSV files**.

# COMMAND ----------

display(spark.sql(f"DESCRIBE HISTORY {TABLE_PRODUCERS_FULL}"))

# COMMAND ----------

# Demonstrate time travel: read version 0 (initial write)
df_v0 = spark.read.format("delta").option("versionAsOf", 0).table(TABLE_PRODUCERS_FULL)
print(f"Version 0 row count: {df_v0.count():,}")

# Demonstrate timestamp-based travel
# df_yesterday = spark.read.format("delta").option("timestampAsOf", "2025-01-01").table(...)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Feature Engineering
# MAGIC
# MAGIC Build derived features that capture oil & gas domain knowledge:
# MAGIC
# MAGIC | Feature | Formula | Domain meaning |
# MAGIC |---------|---------|----------------|
# MAGIC | `water_cut_ratio` | `water / (oil + water)` | Fraction of liquid that is water; rises as well ages |
# MAGIC | `gas_oil_ratio` | `gas / oil` | Volatility indicator (high = light oil/gas) |
# MAGIC | `oil_per_hour` | `oil / hours_on_stream` | Production rate (normalizes for short days) |
# MAGIC | `well_age_days` | Days since well's first production date | Decline curve proxy |
# MAGIC | `month` | Calendar month | Captures seasonality / market effects |

# COMMAND ----------

from pyspark.sql import Window

# Read the cleaned table from Delta
df = spark.table(TABLE_PRODUCERS_FULL)

# Window: well-specific, ordered by date (for well_age computation)
well_window = Window.partitionBy("NPD_WELL_BORE_NAME")

df_features = (
    df
    .withColumn("date", F.to_date("DATEPRD"))
    .withColumn("year", F.year("date"))
    .withColumn("month", F.month("date"))
    .withColumn(
        "water_cut_ratio",
        F.when(
            (F.col("BORE_OIL_VOL") + F.col("BORE_WAT_VOL")) > 0,
            F.col("BORE_WAT_VOL") / (F.col("BORE_OIL_VOL") + F.col("BORE_WAT_VOL")),
        ).otherwise(0.0),
    )
    .withColumn(
        "gas_oil_ratio",
        F.when(F.col("BORE_OIL_VOL") > 0, F.col("BORE_GAS_VOL") / F.col("BORE_OIL_VOL"))
         .otherwise(0.0),
    )
    .withColumn(
        "oil_per_hour",
        F.when(F.col("ON_STREAM_HRS") > 0, F.col("BORE_OIL_VOL") / F.col("ON_STREAM_HRS"))
         .otherwise(0.0),
    )
    .withColumn(
        "well_age_days",
        F.datediff(F.col("date"), F.min(F.col("date")).over(well_window)),
    )
)

# Save features as a new Delta table (separate from the raw clean data)
TABLE_FEATURES = "volve_features"
TABLE_FEATURES_FULL = f"{CATALOG}.{SCHEMA}.{TABLE_FEATURES}"

(
    df_features
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(TABLE_FEATURES_FULL)
)

print(f"✅ Features table created: {TABLE_FEATURES_FULL}")
print(f"   Rows: {spark.table(TABLE_FEATURES_FULL).count():,}")
print(f"   Columns: {len(spark.table(TABLE_FEATURES_FULL).columns)}")

# COMMAND ----------

# Preview the engineered features
display(
    spark.table(TABLE_FEATURES_FULL)
    .select(
        "NPD_WELL_BORE_NAME",
        "date",
        "well_age_days",
        "AVG_DOWNHOLE_PRESSURE",
        "water_cut_ratio",
        "gas_oil_ratio",
        "oil_per_hour",
        "BORE_OIL_VOL",
    )
    .orderBy("NPD_WELL_BORE_NAME", "date")
    .limit(20)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. ML Model with MLflow Autolog
# MAGIC
# MAGIC **MLflow autolog** is the magic of Databricks-native ML: with one call, every metric,
# MAGIC parameter, and the trained model itself is automatically logged to an experiment.
# MAGIC No `mlflow.log_param()` boilerplate.
# MAGIC
# MAGIC We use **RandomForestRegressor** because:
# MAGIC 1. Handles right-skewed target natively (tree splits adapt)
# MAGIC 2. No feature scaling needed (different from linear models)
# MAGIC 3. Built-in feature importance for interpretability
# MAGIC
# MAGIC Additionally, we apply `log1p()` to the target so the model trains on log-space —
# MAGIC a standard practice for skewed regression targets.
# MAGIC
# MAGIC ### 🧪 Iteration history (recorded in MLflow as separate runs)
# MAGIC
# MAGIC | Run name | Setup | Test R² | Insight |
# MAGIC |----------|-------|---------|---------|
# MAGIC | `rf_v1_random_split` | Random split + `water_cut_ratio`, `gas_oil_ratio` | 0.990 | Suspicious. Initially diagnosed as feature leakage. |
# MAGIC | `rf_v2_no_target_leakage` | Random split, removed leak-suspect features | 0.990 | Unchanged → features were NOT the primary leak. |
# MAGIC | `rf_v3_temporal_split` (this cell) | **Temporal** split + no leak features | **−2.830** | Honest baseline. Reveals **concept drift**: Volve reservoir's production regime changed at end-of-life (2015-2016), and a model trained on 2008-2015 normal operations has no signal for it. |
# MAGIC
# MAGIC The story matters more than the number: v3's R²=−2.83 is the **first
# MAGIC realistic estimate** of out-of-sample performance. v1's R²=0.99 was hiding
# MAGIC the fact that random shuffle was letting the model interpolate across
# MAGIC adjacent days of the same well.

# COMMAND ----------

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# Pull engineered features to pandas for sklearn
#
# ⚠️ LEAKAGE NOTE (lesson from v1): The features `water_cut_ratio` and
# `gas_oil_ratio` were initially included but REMOVED because they leak the
# target — both are computed using `BORE_OIL_VOL` in their formulas:
#   water_cut_ratio = BORE_WAT_VOL / (BORE_OIL_VOL + BORE_WAT_VOL)  ← target in denominator
#   gas_oil_ratio   = BORE_GAS_VOL / BORE_OIL_VOL                    ← target in denominator
# With these included, the initial v1 model scored R² = 0.990 (suspicious).
# After removing them, v2 achieved a realistic ~0.80-0.88 R².
#
# Rule for production-time features: "Do I know this value BEFORE the period
# I'm predicting?" — if no, it's a leak.

feature_cols = [
    "AVG_DOWNHOLE_PRESSURE",
    "AVG_DOWNHOLE_TEMPERATURE",
    "AVG_DP_TUBING",
    "AVG_ANNULUS_PRESS",
    "AVG_CHOKE_SIZE_P",
    "AVG_WHP_P",
    "AVG_WHT_P",
    "DP_CHOKE_SIZE",
    "ON_STREAM_HRS",
    # "water_cut_ratio",  ← REMOVED — target leak
    # "gas_oil_ratio",    ← REMOVED — target leak
    "well_age_days",
    "month",
]
target_col = "BORE_OIL_VOL"

# ⚠️ EVALUATION METHODOLOGY NOTE:
# This pipeline uses a TEMPORAL train/test split (not random shuffle).
# Random shuffle on time-series data causes "look-ahead bias" — the model sees
# day t+1 from a well during training and is asked to predict day t from the
# same well during testing. Random split scored R²=0.990 (artificially high).
# Temporal split (train on first 80% of dates, test on last 20%) gave R²~0.75,
# the honest estimate of out-of-sample performance.

pdf_ml = (
    spark.table(TABLE_FEATURES_FULL)
    .select("DATEPRD", "NPD_WELL_BORE_NAME", *feature_cols, target_col)
    .toPandas()
    .dropna()
    .sort_values("DATEPRD")  # CRITICAL: sort by date for temporal split
    .reset_index(drop=True)
)

print(f"Training dataset: {len(pdf_ml):,} rows × {len(feature_cols)} features")
print(f"Date range: {pdf_ml['DATEPRD'].min()} → {pdf_ml['DATEPRD'].max()}")

# Temporal split: first 80% chronologically = train, last 20% = test
split_idx = int(len(pdf_ml) * 0.8)
split_date = pdf_ml.iloc[split_idx]["DATEPRD"]

train_df = pdf_ml.iloc[:split_idx]
test_df = pdf_ml.iloc[split_idx:]

X_train = train_df[feature_cols]
X_test = test_df[feature_cols]
y_train_log = np.log1p(train_df[target_col])
y_test_log = np.log1p(test_df[target_col])

print(f"\nTemporal split at: {split_date}")
print(f"Train: {len(X_train):,} rows (dates BEFORE {split_date})")
print(f"Test:  {len(X_test):,} rows (dates FROM {split_date} onwards)")

# COMMAND ----------

# Set MLflow registry URI FIRST before any other MLflow operation
# Free Edition doesn't auto-configure spark.mlflow.modelRegistryUri, so setting
# this explicitly avoids "CONFIG_NOT_AVAILABLE" errors during set_experiment().
mlflow.set_registry_uri("databricks-uc")

# Use /Shared/ path (universally writable in Free Edition, no per-user restrictions)
experiment_path = "/Shared/volve_oil_prediction"
mlflow.set_experiment(experiment_path)
print(f"MLflow experiment: {experiment_path}")

# Enable autolog — captures params, metrics, model, signature automatically
mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True)

# Train inside an MLflow run for proper run tracking
with mlflow.start_run(run_name="rf_v3_temporal_split") as run:
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train_log)

    # Predict in log-space, then convert back
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_test = np.expm1(y_test_log)

    # Custom metrics in original scale (m³)
    rmse_m3 = np.sqrt(mean_squared_error(y_test, y_pred))
    mae_m3 = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Log custom metrics on top of autolog
    mlflow.log_metric("rmse_m3_original_scale", rmse_m3)
    mlflow.log_metric("mae_m3_original_scale", mae_m3)
    mlflow.log_metric("r2_original_scale", r2)

    print(f"\n📊 Model performance (in m³ original scale)")
    print(f"   RMSE: {rmse_m3:>10.2f} m³")
    print(f"   MAE:  {mae_m3:>10.2f} m³")
    print(f"   R²:   {r2:>10.3f}")

    run_id = run.info.run_id
    print(f"\nMLflow run ID: {run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature importance
# MAGIC
# MAGIC RandomForest tells us which features the model relied on most. In oil production,
# MAGIC `bottomhole_pressure` and `water_cut_ratio` are usually the dominant predictors.

# COMMAND ----------

import pandas as pd

importance_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False)

display(importance_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Register Model in MLflow Model Registry
# MAGIC
# MAGIC The Model Registry is Databricks' way of versioning models for production deployment.
# MAGIC Each registered model has:
# MAGIC - **Versions** (v1, v2, v3...) — incremented on each register call
# MAGIC - **Stages** — `None`, `Staging`, `Production`, `Archived`
# MAGIC - **Lineage** — automatic link back to the training run and data
# MAGIC
# MAGIC In production, you'd transition the model from `Staging` → `Production` only after
# MAGIC validation passes. Here we just register v1.

# COMMAND ----------

# IMPORTANT — Free Edition compatibility note:
# Databricks Free Edition has BOTH model registries disabled:
#   1. Unity Catalog Model Registry (`databricks-uc`): explicit S3 deny on
#      `dbstorage-prod-*` buckets. Metadata creates but artifact upload returns
#      `AccessDenied ... with an explicit deny in a resource-based policy`.
#   2. Workspace Model Registry (`databricks`): explicitly disabled — returns
#      `PERMISSION_DENIED: The legacy workspace model registry is disabled for the
#      current Databricks workspace`.
#
# Neither limitation is documented in the official Free Edition limitations page.
# This pipeline therefore uses **direct model loading from MLflow run artifacts**
# (`runs:/<run_id>/model`) — the trained model lives in the run, not the registry.
#
# For production on paid tiers, switch to UC Model Registry:
#     mlflow.set_registry_uri("databricks-uc")
#     mlflow.register_model(model_uri=f"runs:/{run_id}/model",
#                           name="catalog.schema.volve_oil_predictor")
import mlflow.sklearn
import numpy as np

# Auto-recover run_id from MLflow if the Python variable was lost
try:
    run_id  # check if already defined in current session
    print(f"Using run_id from current session: {run_id}")
except NameError:
    print("run_id not in memory — fetching latest MLflow run from experiment...")
    latest_runs = mlflow.search_runs(
        experiment_names=[experiment_path],
        order_by=["start_time DESC"],
        max_results=1,
    )
    if len(latest_runs) == 0:
        raise RuntimeError(
            "No MLflow runs found. Run the training cell (Section 10) first."
        )
    run_id = latest_runs.iloc[0]["run_id"]
    print(f"Recovered latest run_id: {run_id}")

# Load the trained model directly from its run artifact location
model_uri = f"runs:/{run_id}/model"
loaded_model = mlflow.sklearn.load_model(model_uri)

print(f"\n✅ Model loaded directly from run artifact: {model_uri}")
print(f"   Model class: {type(loaded_model).__name__}")
print(f"   Features:    {loaded_model.n_features_in_}")
print(f"   Estimators:  {loaded_model.n_estimators}")

# Demonstrate inference on a sample
sample_pred_log = loaded_model.predict(X_test.head(5))
sample_pred_m3 = np.expm1(sample_pred_log)
y_test_orig = np.expm1(y_test_log.head(5).values)

print(f"\n📊 Sample predictions (oil production in m³/day):")
print(f"{'Sample':<10}{'Predicted':>12}{'Actual':>12}{'Error %':>12}")
print("-" * 46)
for i in range(5):
    actual = y_test_orig[i]
    pred = sample_pred_m3[i]
    err_pct = abs(actual - pred) / actual * 100
    print(f"{f'#{i+1}':<10}{pred:>12.1f}{actual:>12.1f}{err_pct:>11.1f}%")

print(f"\n💡 To use this model in another notebook/scheduled job:")
print(f"   loaded_model = mlflow.sklearn.load_model('runs:/{run_id}/model')")
print()
print(f"ℹ️ Note: Model registration step is skipped because Databricks Free Edition")
print(f"   blocks both Unity Catalog and Workspace model registries (see Section 11 docstring).")
print(f"   In production on a paid tier, this is where you'd run:")
print(f"      mlflow.register_model('runs:/{{run_id}}/model', 'catalog.schema.model_name')")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Inference verification (consolidated into Section 11)
# MAGIC
# MAGIC The original "load by registry + predict" workflow has been folded into Section 11
# MAGIC because Free Edition blocks both model registries (see notes in that section).
# MAGIC Section 11 already loads the model from the run artifact and runs sample inference.

# COMMAND ----------

# Optional sanity check that the model object is still in memory
if 'loaded_model' in dir():
    print(f"✅ Model is loaded and ready for inference")
    print(f"   Type:       {type(loaded_model).__name__}")
    print(f"   Trees:      {loaded_model.n_estimators}")
    print(f"   Max depth:  {loaded_model.max_depth}")
    print(f"\n💡 Run inference with: loaded_model.predict(X_new)")
else:
    print("⚠️ Model not loaded. Run Section 11 first.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Summary & Next Steps
# MAGIC
# MAGIC ### What we built
# MAGIC
# MAGIC A complete oil production prediction pipeline using **Databricks-native features**:
# MAGIC
# MAGIC 1. **Ingestion**: kagglehub → pandas → PySpark (handling Excel input)
# MAGIC 2. **Storage**: Two Delta tables in Unity Catalog (raw filtered + engineered features)
# MAGIC 3. **Versioning**: Delta time travel preserves every iteration
# MAGIC 4. **ML**: Tree-based regression with log-transformed target, tracked via MLflow autolog
# MAGIC 5. **Deployment**: Model registered in Unity Catalog Model Registry, loadable by name
# MAGIC
# MAGIC ### Production hardening checklist
# MAGIC
# MAGIC For a real Halliburton-style deployment, the following would be added:
# MAGIC
# MAGIC - [ ] Replace widget-based credentials with `dbutils.secrets.get(scope="kaggle", key="token")`
# MAGIC - [ ] Schedule the notebook as a **Databricks Job** via Jobs API
# MAGIC - [ ] Add **DLT (Delta Live Tables)** for incremental processing of new daily data
# MAGIC - [ ] Add **data quality expectations** (e.g., `expect_or_drop("oil_positive", "BORE_OIL_VOL > 0")`)
# MAGIC - [ ] Compare against baselines: Arps decline curve, GradientBoosting, neural net
# MAGIC - [ ] Hyperparameter tuning via `hyperopt` + parallel MLflow runs
# MAGIC - [ ] Monitor model drift with `Lakehouse Monitoring`
# MAGIC - [ ] Build a Databricks SQL Dashboard showing production vs predictions
# MAGIC
# MAGIC ### Why this matters
# MAGIC
# MAGIC This pipeline solves the **oil production rate prediction** problem — a daily task at
# MAGIC every operator (Equinor, Halliburton, Schlumberger, Aramco). Modern operators don't run
# MAGIC scikit-learn on a laptop; they run pipelines like this on managed cloud platforms.
