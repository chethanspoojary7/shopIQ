import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.services.model_service import validate_session

SAMPLE = {"Administrative": 2, "Administrative_Duration": 45.0, "Informational": 1,
          "Informational_Duration": 30.0, "ProductRelated": 10, "ProductRelated_Duration": 300.0,
          "BounceRates": 0.02, "ExitRates": 0.04, "SpecialDay": 0.0, "Month": "May",
          "OperatingSystems": 2, "Browser": 2, "Region": 1, "TrafficType": 2,
          "VisitorType": "Returning_Visitor", "Weekend": False}

def test_valid_sample_passes():
    assert validate_session(SAMPLE) is None

def test_missing_field_rejected():
    s = dict(SAMPLE); del s["Month"]
    assert validate_session(s) is not None

def test_bad_month_rejected():
    s = dict(SAMPLE); s["Month"] = "January"
    assert validate_session(s) is not None

def test_weekend_must_be_bool():
    s = dict(SAMPLE); s["Weekend"] = "yes"
    assert validate_session(s) is not None

def test_pagevalues_not_required():
    assert "PageValues" not in SAMPLE
    assert validate_session(SAMPLE) is None
