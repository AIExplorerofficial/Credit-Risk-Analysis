# CreditView

CreditView is a browser-computed research interface using the saved model, preprocessing, policy rules, and evaluation data from `notebooks/02_training.ipynb`. Applicant inputs and scoring stay on the visitor's device. The host serves the application and read-only runtime files; it does not load the model or score requests. CreditView presents historical model estimates, not loan offers.

## Run the application

From this project folder, install the pinned packages and start the lightweight static host:

```powershell
python -m pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:8000` (or `http://localhost:8000`). The browser downloads the frozen model and notebook-derived runtime bundle, checks their SHA-256 fingerprints, then performs scoring in a Web Worker. Starting the host does not import or load the Python ML service. The `/api/*` Python routes remain available in `backend/main.py` for local reference and regression tests; the website does not call them.

### Render deployment

The existing Python Web Service can keep `python app.py` as its start command. Change its build command to `pip install -r requirements-web.txt`. This installs only the lightweight static host dependencies; the server does not load LightGBM, SHAP, the processed dataset, or analytics into Render memory. Keep the service on the same repository and branch; push the changed files and let Render redeploy. The frontend reads `PORT` from Render automatically.

For a Render Static Site instead, use the repository root as the root directory, leave the build command empty, and set the publish directory to `frontend`. No Python runtime or start command is needed. Both deployment types serve the same browser-computed application.

If the notebook, saved model, configuration, or processed data are intentionally updated, regenerate the packaged runtime locally before pushing:

```powershell
python -B scripts/export_browser_runtime.py
```

That command exports the saved model and actual processed observations and refreshes the fixed notebook-governance report and SHA-256 manifest. It does not retrain the model. Do not run it as a Render build step.

## How to read the recommendations

1. For inputs within observed training support, the live recommendation uses the saved 23% threshold and conformal referral rule. A conformal set containing both outcomes is referred when estimated risk is within two percentage points of the threshold; otherwise risk below the threshold is labelled Approve and risk at or above it is labelled Reject.
2. Support is assessed independently using only rows in the notebook’s training split. Numeric values outside the observed training minimum/maximum and categories absent from training are flagged as extrapolations; values outside Tukey's 1.5×IQR fences but within observed limits are flagged as unusual. These bounds are derived at startup from the packaged training data, not manually selected cutoffs.
3. The model still returns a risk estimate for extrapolated inputs, but it does not issue an Approve or Reject recommendation. The final status is `Manual Review Required — Outside Model Support`; the estimate is marked provisional and may be unreliable. In-support unusual values receive a warning but retain the saved decision policy.
4. Historical portfolio evaluation retains the notebook's saved out-of-support policy so its reported cohort metrics remain reproducible. That retrospective rule is distinct from the live applicant recommendation described above.

The outcome is a model recommendation, not an actual credit decision. The notebook reports 88.95% empirical conformal coverage on its 2015 test set against a 90% target; CreditView preserves and displays this shortfall.

## Data and computation

- Single application, batch, and user-entered what-if scoring run in the visitor's browser using the saved preprocessor parameters, calibrated LightGBM trees, conformal quantile, support summary, and notebook decision rules. Applicant values are not sent to Render.
- Dashboard prediction summaries, portfolio filtering, and decision-boundary analysis operate in a browser Web Worker over the bundled notebook splits. The out-of-time test split remains read-only; the calibration cohort remains separate.
- Governance metrics and SHAP feature-importance output are fixed reports generated from the notebook artifacts by the local export command and served as static files. They are not recomputed by Render.
- Individual and What-if applicant SHAP reason factors are computed in the browser using an exact JavaScript port of SHAP's interventional TreeExplainer traversal over the notebook's saved, deterministic 200-row training background. Batch scoring omits per-applicant SHAP because the CSV results do not display those explanations; this avoids repeating the heavier calculation for every row.
- The 2014 `cal_B` half is used for threshold and cohort analysis. The 2015 out-of-time split is reserved for final evaluation and reporting.
- Historical examples are real rows from the notebook data. What-if inputs are supplied by the user. No generated applicant records or external data feeds are used.
- SHAP contributions use the notebook’s base LightGBM model and an observed training-split background. They describe model associations, not causes.
- No applicant-specific payment estimate is produced; the notebook does not define that calculation.

## Project contents

- `frontend/` — HTML, CSS, JavaScript, and locally bundled fonts.
- `backend/` — Python API and reference analytics retained for local regression and parity work; it is not the production scoring host.
- `frontend_host.py` and `app.py` — Lightweight static-file host used by the existing Render Web Service configuration.
- `requirements-web.txt` — Minimal dependencies for the Render file host; `requirements.txt` remains for local backend regression and notebook parity work.
- `frontend/runtime/` — Browser-ready saved model, actual notebook split bundle, governance report, and SHA-256 manifest.
- `creditview/` — Shared request schemas and notebook-backed inference service used by the API.
- `src/` — Original notebook computation modules imported by both the notebook and serving code.
- `notebooks/02_training.ipynb` — Authoritative training and evaluation reference.
- `notebook_runtime/` — Required frozen model artifacts, processed dataset splits, and checksum manifest. The active configuration is the root `config.yaml`.
- `tests/` — Parity and route regression checks, using the real packaged artifacts and observations.
- `config.yaml`, `requirements.txt`, `app.py`, `api.py`, and `run_all.py` — Active configuration, dependencies, app entry points, and read-only integrity check.

The large source CSV and superseded model bundle are not required to run the website; the processed notebook splits and frozen artifacts are included under `notebook_runtime/`.
