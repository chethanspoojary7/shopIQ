"""Single-session prediction using the saved pipeline (no retraining)."""
import joblib


def load_pipeline(path):
    return joblib.load(path)


def predict_session(pipeline, session_dict, threshold):
    """session_dict: raw feature dict (PageValues accepted but ignored
    by the leakage-safe model). Returns class, label, probability."""
    import pandas as pd
    from .preprocess import add_engineered
    row = add_engineered(pd.DataFrame([session_dict]))
    proba = float(pipeline.predict_proba(row)[0][1])
    pred = int(proba >= threshold)
    return {
        "prediction": pred,
        "label": "Purchase likely" if pred == 1 else "No purchase likely",
        "purchase_probability": proba,
        "threshold": float(threshold),
    }
