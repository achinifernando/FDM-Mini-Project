# Member 4 - Data Quality, Dataset Splitting and Baseline Preparation

## IT3051 Fundamentals of Data Mining Mini Project 2026

### Project

Next-Year PM2.5 Air-Quality Risk Classification

### Responsibility

Data-Quality Evaluation, Chronological Dataset Splitting and Baseline Model Preparation

---

## 1. Overview

My responsibility was to evaluate the quality of the ML-ready dataset, create leakage-safe chronological training, validation and testing datasets, and establish the initial baseline classification performance.

Main implementation files:

- `src/data/data_quality.py`
- `src/data/make_dataset.py`
- `src/models/train_model.py`

Detailed documentation:

- `docs/data_quality_summary.md`
- `docs/data_splitting_methodology.md`

Generated baseline evidence:

- `reports/model_comparison.csv`
- `reports/figures/baseline_confusion_matrix.png`

---

## 2. Dataset Summary

The input dataset used was:

`data/interim/ml_ready_base.csv`

Dataset characteristics:

- Records: 16,535
- Columns: 38
- Unique monitoring sites: 1,849
- Year range: 1997–2016
- Target: `pm25_risk_category_next_year`

---

## 3. Data-Quality Evaluation

The following checks were performed:

- Dataset dimensions and data types
- Missing values
- Fully duplicated rows
- Duplicated site-year records
- Infinite numerical values
- Missing target values
- Invalid numerical values
- Target-class distribution

Main findings:

- Fully duplicated rows: 0
- Duplicated site-year records: 0
- Infinite numerical values: 0
- Missing target values: 0
- `ten_percentile` contained 27 textual `unknown` values
- `p99_p10_range` contained 27 textual `unknown` values

The preprocessing pipeline converts textual missing-value markers to missing values and uses training-set statistics for imputation.

Detailed results are available in:

`docs/data_quality_summary.md`

---

## 4. Target-Class Preparation

The original target contained six categories. The `Hazardous` category contained only four records.

Following the group decision, the rare class was merged as follows:

`Hazardous → Very Unhealthy`

The final target contains five categories:

1. Good
2. Moderate
3. Unhealthy for Sensitive Groups
4. Unhealthy
5. Very Unhealthy

---

## 5. Chronological Dataset Splitting

A chronological split was used because the project predicts future-year PM2.5 risk categories.

| Dataset | Year Range | Records |
|---|---|---:|
| Training | 1997–2012 | 13,191 |
| Validation | 2013–2014 | 1,783 |
| Testing | 2015–2016 | 1,561 |
| Total | 1997–2016 | 16,535 |

No records were lost during splitting, and all five target classes were present in every split.

Detailed splitting decisions are documented in:

`docs/data_splitting_methodology.md`

---

## 6. Leakage Prevention

The following measures were used to prevent data leakage:

1. Records were divided chronologically.
2. Earlier years were used for training.
3. Later years were reserved for validation and testing.
4. The target variable was removed from the feature matrix.
5. Monitoring-site identifier columns were excluded from model predictors.
6. The preprocessing pipeline was fitted using training data only.
7. Training-set preprocessing settings were applied unchanged to validation and testing data.
8. Validation data was used for baseline selection.
9. Testing data was used only for final baseline evaluation.

Excluded identifier columns:

- `site_id`
- `state_code`
- `county_code`
- `site_num`

---

## 7. Monitoring-Site Overlap

Different years from the same monitoring sites can occur across the chronological splits.

Observed overlap:

- Training and validation shared sites: 867
- Training and testing shared sites: 798
- Validation and testing shared sites: 861

This does not indicate duplicated site-year records. The year ranges do not overlap, and site identifiers are excluded from the model predictors.

---

## 8. Baseline Models

Two baseline classifiers were trained.

### 8.1 Dummy Classifier

The Dummy Classifier used the `most_frequent` strategy. It provides a minimum reference by predicting the majority class.

### 8.2 Multinomial Logistic Regression

Multinomial Logistic Regression was used as the interpretable machine-learning
baseline. The `lbfgs` solver supports multinomial classification and is used
with the five target classes.

The following setting was used to reduce the effect of class imbalance:

```python
class_weight="balanced"
```

Macro F1-score was used as the primary model-selection metric.

---

## 9. Evaluation Metrics

The following metrics were calculated:

- Accuracy
- Balanced accuracy
- Macro precision
- Macro recall
- Macro F1-score
- Weighted F1-score
- Confusion matrix
- Per-class classification report (precision, recall, F1-score and support)

Accuracy was not used alone because the target dataset is imbalanced.
The confusion matrix and classification report are generated for the selected
model on the held-out test set.

---

## 10. Baseline Results

### Validation Results

| Model | Accuracy | Balanced Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|---:|
| Dummy Classifier | 0.7190 | 0.2000 | 0.1438 | 0.2000 | 0.1673 |
| Logistic Regression | 0.7291 | 0.4381 | 0.4071 | 0.4381 | 0.3497 |

Logistic Regression achieved the highest validation Macro F1-score and was selected as the best baseline.

The Dummy Classifier achieved relatively high accuracy because it predicted the majority class. However, its low balanced accuracy and Macro F1-score demonstrate why accuracy alone is misleading for this dataset.

### Test Results

| Model | Accuracy | Balanced Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7969 | 0.2894 | 0.3312 | 0.2894 | 0.2920 |

The selected Logistic Regression baseline achieved a test Macro F1-score of 0.2920.

---

## 11. Temporal Distribution Shift

The target-class distribution changed between the training and testing periods.

- `Moderate` represented approximately 51.92% of the training set.
- `Moderate` represented approximately 82.45% of the testing set.

This is a temporal distribution shift rather than a splitting error. It reflects real-world differences between historical and more recent observations.

This limitation should be considered when interpreting the final model results.

---

## 12. Generated Outputs

Dataset-splitting outputs:

- `data/processed/train_data.csv`
- `data/processed/validation_data.csv`
- `data/processed/test_data.csv`
- `data/processed/X_train.csv`
- `data/processed/X_validation.csv`
- `data/processed/X_test.csv`
- `data/processed/y_train.csv`
- `data/processed/y_validation.csv`
- `data/processed/y_test.csv`

Baseline outputs:

- `reports/model_comparison.csv`
- `reports/confusion_matrix.csv`
- `reports/classification_report.txt`
- `reports/figures/baseline_confusion_matrix.png`
- `models/baseline/preprocessor.joblib`
- `models/baseline/dummy_classifier.joblib`
- `models/baseline/multinomial_logistic_regression.joblib`
- `models/baseline/best_baseline_bundle.joblib`

Generated processed datasets and trained-model files are excluded from Git because they can be reproduced by running the scripts.

---

## 13. How to Run

Activate the virtual environment:

```powershell
.\venv\Scripts\Activate.ps1
```

Install project dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run the data-quality evaluation:

```powershell
python src/data/data_quality.py
```

Generate chronological splits:

```powershell
python src/data/make_dataset.py
```

Train and evaluate the baseline models:

```powershell
python src/models/train_model.py
```

---

## 14. Conclusion

This contribution established a reproducible and leakage-safe workflow from data-quality evaluation to baseline model assessment.

Logistic Regression outperformed the Dummy Classifier based on validation Macro F1-score and provides the initial baseline for comparison with more advanced models during later modelling stages.