"""Light training checks — full training is ml.train, not repeated here."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml.evaluate import compute_metrics

def test_metrics_range():
    m = compute_metrics([0, 1, 1, 0], [0, 1, 0, 0], [0.1, 0.9, 0.4, 0.2])
    assert 0 <= m["accuracy"] <= 1 and 0 <= m["roc_auc"] <= 1
    assert m["confusion_matrix"] == [[2, 0], [1, 1]]

def test_artifacts_exist_after_training():
    # Only meaningful after a real run; skipped by pytest if not trained.
    import pytest
    p = Path(__file__).resolve().parents[1] / "artifacts" / "evaluation_results.json"
    if not p.exists():
        pytest.skip("training not run yet")
    ev = json.loads(p.read_text())
    assert ev["test_at_threshold"]["roc_auc"] > 0.5
    assert ev["selected_model"] in [m["model"] for m in ev["models"]]
