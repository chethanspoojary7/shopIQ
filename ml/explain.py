"""Explainability: permutation importance (model-agnostic fallback).

SHAP is not a hard dependency here: permutation importance is the
documented alternative in the spec when SHAP is incompatible or
unavailable, and it is computed from real model predictions on real
held-out data. Values are associations, not proven causes.
"""
import numpy as np
from sklearn.inspection import permutation_importance


def permutation_importance_for(pipeline, X, y, n_repeats=5, random_state=42):
    result = permutation_importance(
        pipeline, X, y, n_repeats=n_repeats, random_state=random_state, scoring="roc_auc"
    )
    cols = list(X.columns)
    items = sorted(
        [{"feature": c, "importance": float(m), "std": float(s)}
         for c, m, s in zip(cols, result.importances_mean, result.importances_std)],
        key=lambda d: d["importance"], reverse=True,
    )
    return items


def local_contribution_hint(session, importance):
    """Plain-language hint: which supplied inputs are above their
    typical purchase-associated direction. This is a descriptive hint
    built from the session itself, not a SHAP value, and is labelled
    as such in the API response."""
    hints = []
    pr = session.get("ProductRelated", 0)
    if pr >= 30:
        hints.append({"feature": "ProductRelated", "direction": "higher",
                      "note": "Many product pages viewed in this session."})
    if session.get("BounceRates", 1) <= 0.01:
        hints.append({"feature": "BounceRates", "direction": "lower",
                      "note": "Very low bounce rate in this session."})
    if session.get("ExitRates", 1) <= 0.02:
        hints.append({"feature": "ExitRates", "direction": "lower",
                      "note": "Low exit rate in this session."})
    if session.get("VisitorType") == "New_Visitor":
        hints.append({"feature": "VisitorType", "direction": "new",
                      "note": "New visitors purchase more often than returning visitors in this data (24.9% vs 13.9%)."})
    return hints
