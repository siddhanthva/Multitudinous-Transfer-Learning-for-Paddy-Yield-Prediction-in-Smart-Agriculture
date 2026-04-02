# Paddy Yield Prediction (Bi-LSTM vs Multi-Task Transfer Learning)

This project builds an end-to-end **data + modeling pipeline** to reproduce the workflow described in *“Multitudinous Transfer Learning Based Yield Prediction of Paddy by Enhancing Sustainable Smart Agriculture”* using the provided dataset `paddy_with_yield.csv`.

## What it does

- Loads the CSV (soil nutrients + weather + atmospheric gases + yield).
- Cleans data (type coercion, missing rows dropped).
- Feature engineering from `date` (year/month/day-of-year), drops `sl_no`.
- **Per-fold Min-Max scaling** (fit on train split only) with **5-fold cross-validation**.
- Trains and evaluates:
  - **Bi-LSTM baseline** (treats each sample’s feature vector as a short “sequence” so a Bi-LSTM can be applied robustly under shuffled K-Fold).
  - **MTL (multi-task transfer learning) model** (shared trunk + task heads for:
    - primary: `yield_kg_per_ha`
    - auxiliary: reconstruct `n` and `pH` from the same inputs)
- Produces fold-by-fold and aggregated metrics and writes predictions to `outputs/`.

## Requirements

You need Python 3.10+ installed and available as `python` on PATH.

Install dependencies:

```bash
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run (5-fold CV)

Use your CSV path directly:

```bash
python scripts\run_cv.py --data-path "d:\studies\int\paddy_with_yield.csv"
```

## Data profiling (sanity check)

To generate a quick dataset profile (shapes, columns, missing values, min/max), run:

```bash
python scripts\profile_data.py --data-path "d:\studies\int\paddy_with_yield.csv"
```

Outputs will be written under:

- `outputs\bilstm\`
- `outputs\mtl\`

Each contains:
- `metrics.json` (per-fold + mean/std)
- `predictions.csv` (y_true/y_pred per validation row)

## Notes on matching the paper

- The paper states **min-max scaling** and **K-fold CV** via Scikit-Learn.
- The paper’s reported metrics mix **scaled-space** and **original-space** values in different tables.
  - This pipeline reports both **scaled** and **original** yield metrics where applicable so you can compare either way.
- The MTL loss weights default to the paper-style weighting:
  - yield: 0.7, `n`: 0.15, `pH`: 0.15

