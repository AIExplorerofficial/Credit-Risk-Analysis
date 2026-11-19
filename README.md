# CreditView

CreditView is a local research interface around the saved models, preprocessing, policy rules, and evaluation data from `notebooks/02_training.ipynb`. It presents historical model estimates; it is not a lender or a source of loan offers.

## Run the application

From this project folder, install the pinned packages and start the local server:

```powershell
python -m pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:8000`. The app loads the saved notebook artifacts; it does not train or overwrite them at startup. To check the packaged artifact fingerprints, run `python run_all.py`.

## How to read the recommendations

1. For inputs within observed training support, the live recommendation uses the saved 23% threshold and conformal referral rule. A conformal set containing both outcomes is referred when estimated risk is within two percentage points of the threshold; otherwise risk below the threshold is labelled Approve and risk at or above it is labelled Reject.
2. Support is assessed independently using only rows in the notebook’s training split. Numeric values outside the observed training minimum/maximum and categories absent from training are flagged as extrapolations; values outside Tukey's 1.5×IQR fences but within observed limits are flagged as unusual. These bounds are derived at startup from the packaged training data, not manually selected cutoffs.
3. The model still returns a risk estimate for extrapolated inputs, but it does not issue an Approve or Reject recommendation. The final status is `Manual Review Required — Outside Model Support`; the estimate is marked provisional and may be unreliable. In-support unusual values receive a warning but retain the saved decision policy.
4. Historical portfolio evaluation retains the notebook's saved out-of-support policy so its reported cohort metrics remain reproducible. That retrospective rule is distinct from the live applicant recommendation described above.

The outcome is a model recommendation, not an actual credit decision. The notebook reports 88.95% empirical conformal coverage on its 2015 test set against a 90% target; CreditView preserves and displays this shortfall.

## Data and computation

- Single application, batch, and user-entered what-if scoring use one Python inference service, the saved preprocessor, calibrated LightGBM model, conformal artifact, and notebook decision rules.
- The 2014 `cal_B` half is used for threshold and cohort analysis. The 2015 out-of-time split is reserved for final evaluation and reporting.
- Historical examples are real rows from the notebook data. What-if inputs are supplied by the user. No generated applicant records or external data feeds are used.
- SHAP contributions use the notebook’s base LightGBM model and an observed training-split background. They describe model associations, not causes.
- No applicant-specific payment estimate is produced; the notebook does not define that calculation.

## Project contents

- `frontend/` — HTML, CSS, JavaScript, and locally bundled fonts.
- `backend/` — FastAPI routes and analytics calculated from the notebook artifacts and data.
- `creditview/` — Shared request schemas and notebook-backed inference service used by the API.
- `src/` — Original notebook computation modules imported by both the notebook and serving code.
- `notebooks/02_training.ipynb` — Authoritative training and evaluation reference.
- `notebook_runtime/` — Required frozen model artifacts, processed dataset splits, and checksum manifest. The active configuration is the root `config.yaml`.
- `tests/` — Parity and route regression checks, using the real packaged artifacts and observations.
- `config.yaml`, `requirements.txt`, `app.py`, `api.py`, and `run_all.py` — Active configuration, dependencies, app entry points, and read-only integrity check.

The large source CSV and superseded model bundle are not required to run the website; the processed notebook splits and frozen artifacts are included under `notebook_runtime/`.
