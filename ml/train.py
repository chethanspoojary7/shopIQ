"""ShopPredict AI training.

Split (seed 42, stratified): test 20% held out untouched; remaining
split 85% train / 15% validation. Model selection + threshold tuning
use validation only; test is evaluated exactly once at the end.
Preprocessing is inside each Pipeline, fitted on train only.
Practical (served) models exclude PageValues; one full-feature
XGBoost is trained only for the leakage comparison.
"""
import json, time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .config import (ARTIFACTS, DATA_PROCESSED, DATA_RAW, MODEL_VERSION,
                     RANDOM_STATE, REPORTS, TEST_SIZE, VAL_SIZE_OF_REMAINDER)
from .evaluate import compute_metrics, curve_points
from .explain import permutation_importance_for
from .preprocess import build_preprocessor, prepare_xy, validate
from .utils import write_json

ARTIFACTS.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)


def make_models():
    neg_pos = 10422 / 1908  # documented from verified counts; recomputed below too
    return {
        "Dummy (stratified)": (DummyClassifier(strategy="stratified", random_state=RANDOM_STATE), {}),
        "Logistic Regression": (LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear"), {"classifier__C": [0.1, 1.0]}),
        "Decision Tree": (DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE), {"classifier__max_depth": [5, 10, None]}),
        "Random Forest": (RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1), {"classifier__n_estimators": [200], "classifier__max_depth": [10, None]}),
        "SVM (RBF)": (SVC(probability=True, class_weight="balanced", random_state=RANDOM_STATE), {"classifier__C": [1.0]}),
        "XGBoost": (XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1), {"classifier__n_estimators": [200], "classifier__max_depth": [4, 6], "classifier__learning_rate": [0.1]}),
    }


def main():
    df = pd.read_csv(DATA_RAW)
    quality = validate(df)
    print("Data quality:", quality)
    # EDA aggregates (real, from this file)
    df["_y"] = df["Revenue"].astype(int)
    analytics = {
        "total_sessions": int(len(df)),
        "purchase_sessions": int(df["_y"].sum()),
        "non_purchase_sessions": int(len(df) - df["_y"].sum()),
        "purchase_rate": float(df["_y"].mean()),
        "data_quality": quality,
        "purchase_rate_by_month": {k: float(v) for k, v in df.groupby("Month")["_y"].mean().items()},
        "purchase_rate_by_visitor": {k: float(v) for k, v in df.groupby("VisitorType")["_y"].mean().items()},
        "purchase_rate_by_weekend": {str(k): float(v) for k, v in df.groupby("Weekend")["_y"].mean().items()},
        "feature_means": {c: float(df[c].mean()) for c in df.select_dtypes(include="number").columns if c != "_y"},
        "pagevalues_mean_purchase": float(df[df["_y"] == 1]["PageValues"].mean()),
        "pagevalues_mean_no_purchase": float(df[df["_y"] == 0]["PageValues"].mean()),
        "month_counts": {k: int(v) for k, v in df["Month"].value_counts().items()},
        "visitor_counts": {k: int(v) for k, v in df["VisitorType"].value_counts().items()},
    }
    # Imbalance check: Logistic Regression, no weight vs balanced, on train/val
    X_all, y_all = prepare_xy(df, drop_pagevalues=True)
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X_all, y_all, test_size=TEST_SIZE, stratify=y_all, random_state=RANDOM_STATE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=VAL_SIZE_OF_REMAINDER, stratify=y_tmp, random_state=RANDOM_STATE)
    print(f"Split: train={len(X_train)} val={len(X_val)} test={len(X_test)}")
    print(f"Train purchase rate={y_train.mean():.4f} val={y_val.mean():.4f} test={y_test.mean():.4f}")

    def quick_logreg(weight):
        pipe = Pipeline([("pre", build_preprocessor(False)),
                         ("classifier", LogisticRegression(max_iter=1000, class_weight=weight, solver="liblinear"))])
        pipe.fit(X_train, y_train)
        p = pipe.predict_proba(X_val)[:, 1]
        return compute_metrics(y_val, (p >= 0.5).astype(int), p)
    imb = {"logreg_no_weight_val": quick_logreg(None), "logreg_balanced_val": quick_logreg("balanced")}
    print("Imbalance check (validation):", {k: (round(v["pr_auc"], 4), round(v["f1"], 4)) for k, v in imb.items()})

    models = make_models()
    # scale_pos_weight from actual train counts
    neg = int((y_train == 0).sum()); pos = int((y_train == 1).sum())
    models["XGBoost"][0].set_params(scale_pos_weight=neg / pos)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    comparison, fitted = [], {}
    for name, (clf, grid) in models.items():
        t0 = time.time()
        pipe = Pipeline([("pre", build_preprocessor(False)), ("classifier", clf)])
        search = GridSearchCV(pipe, grid, cv=cv, scoring="average_precision", n_jobs=-1) if grid else pipe
        search.fit(X_train, y_train)
        best = search.best_estimator_ if grid else search
        train_time = round(time.time() - t0, 2)
        val_proba = best.predict_proba(X_val)[:, 1]
        val_m = compute_metrics(y_val, (val_proba >= 0.5).astype(int), val_proba)
        print(f"{name}: val PR-AUC={val_m['pr_auc']:.4f} F1={val_m['f1']:.4f} ({train_time}s) params={getattr(search, 'best_params_', {})}")
        comparison.append({"model": name, "train_time_s": train_time,
                           "best_params": getattr(search, "best_params_", {}),
                           "validation": val_m})
        fitted[name] = best

    # Selection on validation only: PR-AUC, then F1
    selected_name = max(comparison, key=lambda r: (r["validation"]["pr_auc"], r["validation"]["f1"]))["model"]
    print("Selected (validation):", selected_name)
    selected = fitted[selected_name]
    # Threshold tuning on validation only, grid 0.05..0.90
    thresholds = []
    for t in np.arange(0.05, 0.91, 0.05):
        m = compute_metrics(y_val, (selected.predict_proba(X_val)[:, 1] >= t).astype(int), selected.predict_proba(X_val)[:, 1])
        thresholds.append({"threshold": round(float(t), 2), "precision": m["precision"], "recall": m["recall"], "f1": m["f1"]})
    best_t = max(thresholds, key=lambda d: d["f1"])
    threshold = float(best_t["threshold"])
    print("Selected threshold (validation F1):", best_t)

    # Leakage comparison: same XGBoost config, full features incl. PageValues
    full_pipe = Pipeline([("pre", build_preprocessor(True)),
                          ("classifier", XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1, n_estimators=200, max_depth=6, learning_rate=0.1, scale_pos_weight=neg / pos))])
    # Re-split the full-feature frame identically (same seed/sizes).
    Xf_all, _yf = prepare_xy(df, drop_pagevalues=False)
    Xf_tmp, Xf_test, yf_tmp, yf_test = train_test_split(Xf_all, _yf, test_size=TEST_SIZE, stratify=_yf, random_state=RANDOM_STATE)
    Xf_train, Xf_val, yf_train, yf_val = train_test_split(Xf_tmp, yf_tmp, test_size=VAL_SIZE_OF_REMAINDER, stratify=yf_tmp, random_state=RANDOM_STATE)
    full_pipe.fit(Xf_train, yf_train)
    full_val_proba = full_pipe.predict_proba(Xf_val)[:, 1]
    full_val_m = compute_metrics(yf_val, (full_val_proba >= 0.5).astype(int), full_val_proba)
    print(f"Leakage comparison XGBoost WITH PageValues: val PR-AUC={full_val_m['pr_auc']:.4f} vs practical (no PageValues) below")

    # Final: evaluate every practical model ONCE on test at 0.5 AND at
    # selected threshold for the selected model.
    test_rows = []
    for row in comparison:
        name = row["model"]; pipe = fitted[name]
        proba = pipe.predict_proba(X_test)[:, 1]
        test_rows.append({**row, "test": compute_metrics(y_test, (proba >= 0.5).astype(int), proba)})
    sel_test_proba = selected.predict_proba(X_test)[:, 1]
    sel_test_at_threshold = compute_metrics(y_test, (sel_test_proba >= threshold).astype(int), sel_test_proba)
    curves = curve_points(y_test, sel_test_proba)
    # Permutation importance on a test subsample (max 1500 rows, for speed)
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(len(X_test), size=min(1500, len(X_test)), replace=False)
    importance = permutation_importance_for(selected, X_test.iloc[idx], y_test[idx])

    metadata = {
        "project": "ShopPredict AI", "model_version": MODEL_VERSION,
        "selected_model": selected_name, "selected_threshold": threshold,
        "threshold_table": thresholds,
        "split": {"train": int(len(X_train)), "validation": int(len(X_val)), "test": int(len(X_test)), "seed": RANDOM_STATE,
                  "train_purchase_rate": float(y_train.mean()), "validation_purchase_rate": float(y_val.mean()), "test_purchase_rate": float(y_test.mean())},
        "features_served": [c for c in X_train.columns],
        "features_excluded": ["PageValues"],
        "prediction_moment": "During browsing, before checkout completes. PageValues excluded because it reflects completed-transaction value.",
        "leakage_comparison": {"xgboost_with_pagevalues_validation": full_val_m,
                               "xgboost_without_pagevalues_validation": next(r["validation"] for r in comparison if r["model"] == "XGBoost")},
        "imbalance_experiment": imb,
        "target_meanings": {"0": "No purchase recorded", "1": "Purchase recorded"},
        "feature_importance_permutation": importance,
        "curves": curves,
    }
    evaluation = {
        "selected_model": selected_name, "threshold": threshold,
        "test_at_threshold": sel_test_at_threshold,
        "test_at_0_5": next(r["test"] for r in test_rows if r["model"] == selected_name),
        "models": [{"model": r["model"], "train_time_s": r["train_time_s"], "best_params": r["best_params"],
                    "validation": r["validation"], "test": r["test"]} for r in test_rows],
    }
    joblib.dump(selected, ARTIFACTS / "best_model.joblib")
    joblib.dump(selected.named_steps["pre"], ARTIFACTS / "preprocessing_pipeline.joblib")
    write_json(ARTIFACTS / "model_metadata.json", metadata)
    write_json(ARTIFACTS / "evaluation_results.json", evaluation)
    write_json(ARTIFACTS / "analytics.json", analytics)
    pd.DataFrame([{"model": r["model"], **{f"test_{k}": v for k, v in r["test"].items() if isinstance(v, float)},
                   "train_time_s": r["train_time_s"]} for r in test_rows]).to_csv(REPORTS / "model_comparison.csv", index=False)
    # EDA plots (matplotlib, real data)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    (REPORTS / "plots").mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    months_order = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    rates = [analytics["purchase_rate_by_month"].get(m, 0) for m in months_order]
    ax.bar(months_order, rates); ax.set_title("Purchase rate by month"); ax.set_ylabel("Purchase rate")
    fig.tight_layout(); fig.savefig(REPORTS / "plots" / "purchase_rate_by_month.png", dpi=120); plt.close(fig)
    print("DONE selected:", selected_name, "threshold:", threshold)
    print("Test at threshold:", sel_test_at_threshold)


if __name__ == "__main__":
    main()
