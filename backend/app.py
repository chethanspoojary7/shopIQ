"""ShopPredict AI — Flask API. Run from project root: python -m backend.app
or: flask --app backend.app run --port 5000  (after training)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flask import Flask, jsonify, request
from flask_cors import CORS

from backend.services.model_service import load_all, model_loaded, predict, validate_session

app = Flask(__name__)
CORS(app, origins=["http://127.0.0.1:5173", "http://localhost:5173"])


def _setup_error():
    return jsonify({"error": "Model artifacts not found",
                    "setup": "Place online_shoppers_intention.csv in data/raw/ and run: python -m ml.train"}), 503


@app.get("/api/health")
def health():
    _, metadata, _, _ = load_all()
    return jsonify({"status": "ok", "model_loaded": model_loaded(),
                    "model_version": metadata["model_version"] if metadata else None,
                    "preprocessing_available": (Path(__file__).resolve().parents[1] / "artifacts" / "preprocessing_pipeline.joblib").exists()})


@app.get("/api/metadata")
def metadata():
    _, meta, _, _ = load_all()
    if not meta:
        return _setup_error()
    return jsonify({
        "project": "ShopPredict AI", "model_version": meta["model_version"],
        "selected_model": meta["selected_model"], "selected_threshold": meta["selected_threshold"],
        "prediction_moment": meta["prediction_moment"],
        "features_served": meta["features_served"], "features_excluded": meta["features_excluded"],
        "target_meanings": meta["target_meanings"],
        "categorical_values": {"Month": ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
                               "VisitorType": ["Returning_Visitor", "New_Visitor", "Other"],
                               "Weekend": [True, False]},
        "threshold_table": meta["threshold_table"],
    })


@app.get("/api/metrics")
def metrics():
    _, meta, evaluation, _ = load_all()
    if not evaluation or not meta:
        return _setup_error()
    return jsonify({"selected_model": evaluation["selected_model"], "threshold": evaluation["threshold"],
                    "test_at_threshold": evaluation["test_at_threshold"],
                    "test_at_0_5": evaluation["test_at_0_5"],
                    "models": evaluation["models"],
                    "feature_importance": meta.get("feature_importance_permutation", []),
                    "curves": meta.get("curves", {}),
                    "leakage_comparison": meta.get("leakage_comparison", {})})


@app.get("/api/analytics")
def analytics():
    _, _, _, data = load_all()
    if not data:
        return _setup_error()
    return jsonify(data)


@app.post("/api/predict")
def predict_route():
    body = request.get_json(silent=True) or {}
    session = body.get("session")
    err = validate_session(session) if session is not None else "Missing 'session' object"
    if err:
        return jsonify({"error": err}), 400
    result, setup_err = predict(session)
    if setup_err:
        return jsonify({"error": setup_err}), 503
    return jsonify(result)


@app.errorhandler(404)
def not_found(_e):
    return jsonify({"error": "Unknown endpoint"}), 404


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
