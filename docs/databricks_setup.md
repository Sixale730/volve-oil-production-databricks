# Databricks Free Edition — Setup Guide

This document covers the complete setup flow for running this pipeline on **Databricks Free Edition**,
including known limitations and workarounds.

## 1. Create a Free Edition account

1. Go to https://www.databricks.com/learn/free-edition
2. Sign up with email (no credit card needed)
3. After verification, you'll land on the **Welcome to Databricks** screen

## 2. Free Edition vs Community Edition vs Paid

Databricks now has three tiers:

| Tier | Compute | Unity Catalog | MLflow | Cost | Status |
|------|---------|---------------|--------|------|--------|
| **Community Edition** (legacy) | Single dedicated cluster, manual start | No | Basic | Free | Deprecated, replaced by Free Edition |
| **Free Edition** (current) | Serverless (auto-scaling, auto-shutdown) | ✅ Yes | ✅ Full | Free | Active |
| **Paid** (Standard/Premium/Enterprise) | Serverless + dedicated + GPU | ✅ Full | ✅ Full + Registry | $$ | Production |

This notebook is built for **Free Edition**. It deliberately avoids features that only exist
in paid tiers (Genie Spaces, custom Databricks Apps, managed MCP servers).

## 3. Import the notebook

### Method A: Upload via UI

1. In Databricks: click **Workspace** in the left sidebar
2. Click your username folder (e.g., `Users/your-email@gmail.com`)
3. Click **Create** → **Import**
4. Drop or browse to `notebooks/01_volve_production_pipeline.py`
5. Format will auto-detect as **Source file** (because of the `# Databricks notebook source` header)
6. Click **Import**

### Method B: Sync via Git Repos

1. In Databricks: **Workspace** → **Repos** → **Add Repo**
2. Paste your GitHub URL
3. Authenticate with a GitHub Personal Access Token
4. Databricks clones the repo and you can edit notebooks in-place

### Troubleshooting imports

| Error | Likely cause | Fix |
|-------|--------------|-----|
| "Cannot import file" | Wrong format selected | Use **Source file** for `.py`, **Notebook** for `.ipynb` |
| "Invalid notebook source" | Missing header | First line of `.py` must be `# Databricks notebook source` |
| Import succeeds but no cells visible | File has Windows line endings | Convert to LF (`dos2unix` or `git config core.autocrlf input`) |

## 4. Generate a Kaggle API token

The notebook downloads the Volve dataset programmatically. You need a Kaggle API token:

1. Sign up at https://www.kaggle.com (free)
2. Go to **Settings** → scroll to **API** section
3. Click **Create New Token**
4. A file called `kaggle.json` downloads (or you'll see the token directly)
5. The token looks like `KGAT_abc123def456...` (newer format) or has `{"username": "...", "key": "..."}` (older format)

**Where to paste it**: in the notebook, after running the first 2-3 cells, a widget appears
at the top labeled `🔑 Kaggle API Token`. Paste the token there. **Never paste it into the code**.

## 5. Free Edition limitations to be aware of

### Outbound network restrictions

> "Outbound internet access is restricted to a limited set of trusted domains."
> — [official docs](https://docs.databricks.com/aws/en/getting-started/free-edition-limitations)

This means most external HTTP/HTTPS calls from your notebook will be blocked. **Kaggle is on the
allowlist** (which is why this pipeline works). If you try to call arbitrary APIs, expect failures.

### No `df.cache()` on Serverless

The classic PySpark trick of caching DataFrames in memory doesn't work on Serverless. Use Delta
tables for persistence instead — they're often faster anyway due to Photon engine + automatic caching.

### No direct `sparkContext` access

```python
spark.sparkContext.master  # ❌ Raises PySparkAttributeError on Serverless
spark.version              # ✅ Works fine
```

### One Databricks App per workspace (24-hour auto-shutdown)

This blocks deploying custom MCP servers on Databricks Apps. Use widgets for parameterization
instead.

### PAT permissions = Premium only

You can create Personal Access Tokens in Free Edition, but they're "all-or-nothing" — you can't
restrict their scope. This is fine for personal projects, but not for production multi-user setups.

## 6. Running the notebook

1. Open the imported notebook
2. Check that **Serverless** is selected as the compute (top-right dropdown)
3. Run cells 1-3 (install + restart Python + widget)
4. **Paste Kaggle token into the widget** that just appeared at the top
5. Click **Run All** (or step through cells individually)

### Expected timeline

| Cell range | Action | Time |
|------------|--------|------|
| 1-3 | Install packages, restart Python | 30-60 sec |
| 4-7 | Download Volve, load Excel | 30-90 sec (first time), 10 sec (cached) |
| 8-12 | EDA + filtering | 5 sec |
| 13-15 | Delta save + time travel | 5 sec |
| 16-19 | Feature engineering | 5 sec |
| 20-22 | Train ML + log to MLflow | 30 sec |
| 23-25 | Register model + inference | 5 sec |

**Total**: ~2-3 minutes end-to-end on Serverless.

## 7. Verify the artifacts after running

After successful execution, you should see:

### In Catalog Explorer (left sidebar → Catalog)

```
workspace/
└── default/
    ├── volve_producers_active   (Delta table, ~8K rows)
    └── volve_features            (Delta table, ~8K rows, more columns)
```

Right-click any table → **Open in editor** to browse rows, see schema, check version history.

### In MLflow Experiments (left sidebar → Experiments)

```
/Users/your-email@gmail.com/
└── volve_oil_prediction
    └── rf_baseline_log_target (run)
        ├── params: n_estimators=100, max_depth=15...
        ├── metrics: rmse_m3_original_scale, mae_m3...
        └── artifacts: model/ (the trained RandomForest)
```

Click the run → **Charts** tab to see metric comparisons across runs (when you train multiple).

### In Models (left sidebar → Models)

```
workspace.default.volve_oil_predictor
└── Version 1 (None stage)
    └── Source run: <run-id>
```

## 8. Next steps

Once the baseline pipeline runs successfully:

1. **Take screenshots** of the Catalog Explorer, MLflow runs UI, and Models UI for your portfolio
2. **Tweak hyperparameters** — try `n_estimators=500`, `max_depth=20`, etc. Each `mlflow.start_run()`
   creates a new run you can compare visually
3. **Add features** — try `lag_1_day_oil` (previous day's production), `30_day_avg_oil`, etc.
4. **Try other models** — XGBoost, LightGBM, gradient boosting. MLflow autolog supports them all.
5. **Document in the README** with metrics from your best run
