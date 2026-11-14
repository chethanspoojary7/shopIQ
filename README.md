# ShopPredict AI — Explainable Online Purchase Intention Classification

Predicts whether an online shopping session will result in a purchase
(0 = no purchase, 1 = purchase), with a probability, a tuned threshold,
and permutation-importance explanations. Decision support only — never
a guaranteed outcome.

## Design
Frontend follows the supplied Salesforce Invoices screenshot as the
authoritative layout: grey surround, pale-blue gradient inner card,
black pill navigation (Overview / Predict / Models / Insights, lime
active tab), three-stat card + month timeline, selected-model card,
filter row, and a black bottom panel with a model list on the left and
a detail panel on the right. Content is adapted to purchase prediction;
no Salesforce data, logos, or invoice content is carried over.

## Quick start
```bash
python -m venv .venv && .venv/bin/pip install -r backend/requirements.txt
# Dataset: place online_shoppers_intention.csv in data/raw/
# (already present in this delivery)
.venv/bin/python -m ml.train          # trains, writes artifacts/ + reports/
.venv/bin/python -m backend.app       # Flask API on http://127.0.0.1:5000
cd frontend && npm install && npm run dev   # http://127.0.0.1:5173
# or: npm run build && npm run preview
```
Frontend API base: `VITE_API_BASE_URL` (default `/api`; Vite proxies it to
the local Flask server, and Vercel routes it to the backend service).

## Vercel deployment
The root `vercel.json` defines separate `backend` (Flask) and `frontend`
(Vite) services. `/api/*` is routed to Flask and all other paths to the
frontend. The frontend calls the public `/api` path, so no service binding
is needed; there are no backend-to-backend calls. The backend service uses
the repository root because its Flask app depends on the shared `ml/` package
and `artifacts/`; the required model artifacts are allowlisted in `.gitignore`
and must be committed for deployments.

## API
- `GET /api/health` — status, model loaded, version
- `GET /api/metadata` — schema, categorical values, threshold, prediction moment
- `GET /api/metrics` — real test metrics, model comparison, curves, importance, leakage comparison
- `GET /api/analytics` — real dataset counts, rates by month/visitor/weekend
- `POST /api/predict` — `{"session": {...}}`, 16 served features.
  PageValues is accepted but ignored and never required (leakage-safe model).

## Results (real run, see reports/project_report.md)
- Selected: Random Forest, threshold 0.40 (validation-tuned).
- Test at threshold: accuracy 0.755, precision 0.308, recall 0.466,
  F1 0.371, ROC-AUC 0.754, PR-AUC 0.344, CM [[1685, 399], [204, 178]].
- Leakage check: XGBoost with PageValues validation PR-AUC 0.724 vs
  0.326 without — PageValues is excluded from the served model.

## Tests
```bash
.venv/bin/python -m pytest tests/ -q
```

## Structure
`ml/` pipeline, `backend/` Flask API + services, `frontend/` React+Vite+TS,
`artifacts/` trained model + metrics (8.3 MB compressed joblib — Random Forest,
200 trees), `reports/` report + comparison CSV + plots, `tests/` pytest.
Raw dataset is in `data/raw/` for reproducibility in this delivery;
`.gitignore` excludes venv, node_modules, and generated artifacts, while
allowing the model and metadata files required by the deployed API.
# shopIQ
