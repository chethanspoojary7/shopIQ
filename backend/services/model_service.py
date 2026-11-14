"""Model loading + prediction service. Loads artifacts once at import.

If artifacts are missing, endpoints return a clear setup message
instead of invented metrics — never a fabricated model.
"""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
ARTIFACTS = BASE / "artifacts"

_model = None
_metadata = None
_evaluation = None
_analytics = None


def _load_json(name):
    p = ARTIFACTS / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def load_all():
    global _model, _metadata, _evaluation, _analytics
    if _model is None:
        try:
            import joblib
            p = ARTIFACTS / "best_model.joblib"
            _model = joblib.load(p) if p.exists() else None
        except Exception:
            _model = None
    if _metadata is None:
        _metadata = _load_json("model_metadata.json")
    if _evaluation is None:
        _evaluation = _load_json("evaluation_results.json")
    if _analytics is None:
        _analytics = _load_json("analytics.json")
    return _model, _metadata, _evaluation, _analytics


def model_loaded():
    m, *_ = load_all()
    return m is not None


def validate_session(session):
    """Validate against the real schema. PageValues is accepted but
    ignored by the leakage-safe model; it is never required."""
    if not isinstance(session, dict):
        return "session must be an object"
    numeric = ["Administrative", "Administrative_Duration", "Informational",
               "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
               "BounceRates", "ExitRates", "SpecialDay", "OperatingSystems",
               "Browser", "Region", "TrafficType"]
    for f in numeric:
        if f not in session:
            return f"Missing field: {f}"
        try:
            float(session[f])
        except (TypeError, ValueError):
            return f"Field {f} must be numeric"
    if session.get("Month") not in ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]:
        return "Month must be one of Feb, Mar, May, June, Jul, Aug, Sep, Oct, Nov, Dec"
    if session.get("VisitorType") not in ["Returning_Visitor", "New_Visitor", "Other"]:
        return "VisitorType must be Returning_Visitor, New_Visitor or Other"
    if not isinstance(session.get("Weekend"), bool):
        return "Weekend must be true or false"
    for f in ["BounceRates", "ExitRates"]:
        if not (0 <= float(session[f]) <= 0.2 + 1e-9):
            return f"{f} must be between 0 and 0.2 in this dataset"
    if not (0 <= float(session["SpecialDay"]) <= 1):
        return "SpecialDay must be between 0 and 1"
    return None


def predict(session):
    model, metadata, _, _ = load_all()
    if model is None or metadata is None:
        return None, "Model artifacts not found. Run: python -m ml.train (from the project root, with the dataset in data/raw/)"
    import pandas as pd
    import sys
    sys.path.insert(0, str(BASE))
    from ml.preprocess import add_engineered
    from ml.explain import local_contribution_hint
    row = add_engineered(pd.DataFrame([session]))
    proba = float(model.predict_proba(row)[0][1])
    threshold = float(metadata["selected_threshold"])
    pred = int(proba >= threshold)
    hints = local_contribution_hint(session, metadata.get("feature_importance_permutation", []))
    return {
        "prediction": pred,
        "label": "Purchase likely" if pred == 1 else "No purchase likely",
        "purchase_probability": proba,
        "threshold": threshold,
        "model_version": metadata["model_version"],
        "selected_model": metadata["selected_model"],
        "explanation": hints,
        "note": "Decision support only — a prediction is not a guaranteed outcome.",
    }, None
