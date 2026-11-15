# Retraining Pipeline

cd terrasense/backend

## Step 1: Rebuild processed features & monthly training table
.venv/bin/python -m ml.build_dataset

## Step 2: Retrain LightGBM transition models, conformal quantiles, bundle & manifest
.venv/bin/python -m ml.train

## Step 3: Run leakage-free spatio-temporal validation & promotion gate
.venv/bin/python -m ml.evaluate

## Testing
Verify after training. Note that if the retrained model changes the baselin peredictions, the test_01_golden_regression_exact_preservation() in test_farmer_customization.py
will fail until golden_baseline_snapshot.json is regenerated against the new model.

.venv/bin/pytest tests -v

## Reload in running Flask server
The predict.py caches predictors and static zone in memory, so we would have to restart app.py.