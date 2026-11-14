import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml.preprocess import add_engineered, build_preprocessor, validate

def test_engineered_features():
    df = pd.DataFrame([{"Administrative": 1, "Informational": 1, "ProductRelated": 8,
                        "Administrative_Duration": 10.0, "Informational_Duration": 20.0,
                        "ProductRelated_Duration": 70.0, "Weekend": False}])
    out = add_engineered(df)
    assert out["TotalPages"].iloc[0] == 10
    assert out["TotalDuration"].iloc[0] == 100.0
    assert abs(out["ProductShare"].iloc[0] - 0.8) < 1e-9

def test_validate_rejects_missing_columns():
    try:
        validate(pd.DataFrame([{"Administrative": 1}]))
        raise AssertionError("should have raised")
    except ValueError:
        pass

def test_preprocessor_fits_without_pagevalues():
    df = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "raw" / "online_shoppers_intention.csv").head(50)
    from ml.preprocess import prepare_xy
    X, y = prepare_xy(df, drop_pagevalues=True)
    assert "PageValues" not in X.columns
    assert len(y) == 50
    build_preprocessor(False).fit(X)
