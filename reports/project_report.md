# ShopPredict AI — Project Report

## How to explain this project (no tech talk)
ShopPredict AI looks at how someone browses an online shop — how many
product pages they open, how long they stay, whether they bounce quickly
— and estimates whether that visit will end in a purchase. It does not
guarantee a purchase; it ranks visits so a shop can understand which
browsing patterns usually lead to buying, and which usually do not.

## Dataset — verified from the supplied file, not assumed
- Source file supplied by the user: `online_shoppers_intention.csv`
  (UCI Online Shoppers Purchasing Intention Dataset, archive.ics.uci.edu/dataset/468).
- 12,330 sessions, 18 columns, 17 inputs + target `Revenue`.
- 10,422 no-purchase (84.5%), 1,908 purchase (15.5%, exact rate 15.47%).
- Missing values: 0. Duplicate rows: 125 (kept as-is; duplicates are
  genuine repeated sessions in this dataset, and removing them was not
  specified — they are reported here, not hidden).
- Months present: Feb, Mar, May, June, Jul, Aug, Sep, Oct, Nov, Dec
  (no January/April sessions). May is largest (3,364); November has the
  highest purchase rate (25.4%), February the lowest (1.6%).
- Visitor mix: Returning 10,551 (85.6%), New 1,694 (13.7%), Other 85 (0.7%).
  Purchase rates: New 24.9%, Other 18.8%, Returning 13.9% — new visitors
  purchase more often here, contrary to the usual assumption.

## Prediction moment and leakage — the most important decision
Intended moment: **during browsing, before checkout completes**.
`PageValues` averages 5.89 overall but is far higher for purchases, and
it reflects the value of pages already tied to a completed transaction.
It is therefore treated as leakage for this moment.

Evidence (validation, XGBoost, same config):
- With PageValues: PR-AUC 0.724
- Without PageValues: PR-AUC 0.326

That jump is the warning sign, not a win. The served model **excludes
PageValues**; `/api/predict` never requires it. A model that used it
would look impressive in a report and fail in real use, because the
value would not be available when the prediction is needed.

## Pipeline
- Stratified split, seed 42: train 8,384 / validation 1,480 / test 2,466.
  Purchase rates are 15.47% / 15.47% / 15.49% — proportions preserved.
- Preprocessing inside each Pipeline (fitted on train only):
  median imputation + StandardScaler for numerics, most-frequent +
  OneHot for categoricals (Month, VisitorType, Weekend, and the nominal
  codes OperatingSystems/Browser/Region/TrafficType).
- Engineered features, all from legitimate inputs:
  TotalPages = Administrative + Informational + ProductRelated;
  TotalDuration = sum of the three durations;
  ProductShare = ProductRelated / max(1, TotalPages);
  ProductDurationShare = ProductRelated_Duration / max(1, TotalDuration).

## Imbalance
Only 15.5% purchase, so accuracy misleads: the SVM reaches 84.3%
test accuracy while finding only 5.5% of real purchases.
Class-weighted models were used throughout. Check on validation
(Logistic Regression): no weight — PR-AUC 0.311, F1 0.057;
balanced — PR-AUC 0.312, F1 0.394. SMOTE was not used: class weights
already closed the gap, and resampling would duplicate minority
sessions without adding information.

## Model comparison (test, evaluated once, threshold 0.5)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Dummy (stratified) | 0.739 | 0.159 | 0.160 | 0.159 | 0.503 | 0.156 |
| Logistic Regression | 0.633 | 0.259 | 0.738 | 0.384 | 0.738 | 0.328 |
| Decision Tree | 0.648 | 0.267 | 0.730 | 0.391 | 0.730 | 0.283 |
| Random Forest | 0.811 | 0.375 | 0.325 | 0.348 | 0.754 | 0.344 |
| SVM (RBF) | 0.843 | 0.457 | 0.055 | 0.098 | 0.745 | 0.328 |
| XGBoost | 0.694 | 0.288 | 0.662 | 0.402 | 0.765 | 0.350 |

## Selection and threshold — decided on validation, not on test
Selection rule (fixed before test evaluation): highest validation
PR-AUC, then F1. Winner: **Random Forest** (validation PR-AUC 0.330,
F1 0.360), narrowly ahead of XGBoost (0.326). Threshold tuned on
validation only (grid 0.05–0.90, max F1): **0.40**
(validation precision 0.316, recall 0.528, F1 0.395).

Honest caveat: on the test set at 0.5, XGBoost's F1 (0.402) beats
Random Forest's (0.348). That does **not** retroactively change the
selection — picking the test winner after seeing test scores would be
test leakage. The served model is the validation winner, as decided.

## Final served model — test at threshold 0.40
- Random Forest, 200 trees, class_weight balanced.
- Accuracy 0.755, Precision 0.308, Recall 0.466, F1 0.371,
  ROC-AUC 0.754, PR-AUC 0.344, Balanced accuracy 0.637,
  Specificity 0.809, False-positive rate 0.191, False-negative rate 0.534.
- Confusion matrix [[TN 1685, FP 399], [FN 204, TP 178]]:
  of 382 real purchases, 178 were caught; of 577 flagged sessions,
  178 were real purchases. In plain terms: the model is useful for
  ranking and prioritising, but roughly 2 in 3 flagged sessions still
  do not purchase, and just over half of real purchases are missed.
  That is the honest limit of browsing behaviour alone, without
  PageValues.

## Explainability
Permutation importance on a 1,500-row test subsample (ROC-AUC drop):
Month 0.038, ExitRates 0.036, VisitorType 0.023,
ProductRelated_Duration 0.012, TotalDuration 0.011, then the rest.
SHAP was not installed as a hard dependency; permutation importance
is the spec's documented alternative. All explanations are
associations, not proven causes: a low exit rate is associated with
purchasing, it does not cause a purchase.

## Limitations
- No January/April data; month effects do not generalise to those months.
- Browsing-only features cap performance (see F1 above); adding
  checkout-stage signals would change the prediction moment and must
  be re-justified, not silently added.
- SVM probability uses the deprecated `probability=True` path in
  scikit-learn 1.9 (warning only; a future upgrade should switch to
  CalibratedClassifierCV).
- Live-browser visual check not run: localhost is unreachable from
  that browser (standing limitation, same as earlier projects).
  Frontend was verified by vite build + preview HTTP 200 instead.
