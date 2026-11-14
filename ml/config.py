"""ShopPredict AI — shared configuration. No values here are model results."""
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
DATA_RAW = BASE / "data" / "raw" / "online_shoppers_intention.csv"
DATA_PROCESSED = BASE / "data" / "processed"
ARTIFACTS = BASE / "artifacts"
REPORTS = BASE / "reports"

RANDOM_STATE = 42
TEST_SIZE = 0.20
VAL_SIZE_OF_REMAINDER = 0.15  # validation = 15% of train+val, i.e. ~12% of all data
MODEL_VERSION = "1.0.0"

TARGET = "Revenue"
# Original 17 inputs, in dataset order.
ORIGINAL_FEATURES = [
    "Administrative", "Administrative_Duration", "Informational",
    "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "PageValues", "SpecialDay", "Month",
    "OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType",
    "Weekend",
]
# Leakage decision (see reports/project_report.md, Step 5):
# PageValues is strongly tied to a completed transaction's value and is
# not reliably available at the intended prediction moment — during
# browsing, before checkout completes. The practical model therefore
# EXCLUDES PageValues. A full-feature model is trained only as a
# leakage comparison, never served by /api/predict as required input.
LEAKAGE_FEATURE = "PageValues"
PRACTICAL_FEATURES = [f for f in ORIGINAL_FEATURES if f != LEAKAGE_FEATURE]

NUMERIC_FEATURES = [
    "Administrative", "Administrative_Duration", "Informational",
    "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "SpecialDay",
    # Engineered (Step 4) — all computable from legitimate inputs above:
    "TotalPages", "TotalDuration", "ProductShare", "ProductDurationShare",
]
CATEGORICAL_FEATURES = [
    "Month", "OperatingSystems", "Browser", "Region", "TrafficType",
    "VisitorType", "Weekend",
]
MONTHS = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
VISITOR_TYPES = ["Returning_Visitor", "New_Visitor", "Other"]
