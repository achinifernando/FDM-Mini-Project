# Member 3 - Preprocessing Pipeline

## IT3051 Fundamentals of Data Mining Mini Project 2026

### Project
Next-Year PM2.5 Air-Quality Risk Classification

### Responsibility
Preprocessing Pipeline

---

## 1. Overview

My responsibility in this project was to develop a reusable and
leakage-safe preprocessing pipeline for the machine-learning stage.

The pipeline takes the cleaned and feature-engineered dataset produced
by the Data Engineering stage and prepares the predictors for machine
learning.

Main implementation:

`src/preprocessing/preprocessing_pipeline.py`

Demonstration notebook:

`notebooks/12_preprocessing_pipeline.ipynb`

---

## 2. Input Dataset

The preprocessing pipeline uses:

`data/interim/ml_ready_base.csv`

Current dataset dimensions:

- 16,535 records
- 38 columns

Target variable:

`pm25_risk_category_next_year`

The target represents the PM2.5 air-quality risk category for the
following year.

---

## 3. Preprocessing Steps

### 3.1 Input Validation

The pipeline checks that:

- The dataframe is not empty.
- The required target column exists.
- Duplicate column names are not present.

---

### 3.2 Handling Textual Missing Values

EDA identified textual `unknown` markers in several variables.

The pipeline converts these markers to `NaN` so that missing values
can be handled consistently.

---

### 3.3 Data Type Correction

Some numerical variables can be loaded as object/string columns because
they contain textual `unknown` values.

The pipeline explicitly converts:

- `ten_percentile`
- `p99_p10_range`

to numerical values.

Invalid values are converted to missing values and are later handled by
the numerical imputation pipeline.

---

### 3.4 Duplicate Handling

The pipeline checks for exact duplicate rows.

No exact duplicate rows were identified in the current dataset, but the
functionality is included to keep the preprocessing workflow robust.

---

### 3.5 Target Separation

The target variable:

`pm25_risk_category_next_year`

is separated from the predictor variables before model preprocessing.

This prevents the target itself from being accidentally included as an
input feature.

---

### 3.6 Identifier Exclusion

The following monitoring-site identifier fields are excluded from model
predictors:

- `site_id`
- `state_code`
- `county_code`
- `site_num`

These fields are retained in the original dataframe when required for
grouping or splitting but are not treated as ordinary model predictors.

---

## 4. Numerical Preprocessing

The current model feature set contains 29 numerical predictors.

Numerical preprocessing consists of:

1. Median missing-value imputation
2. Standard scaling

### Median Imputation

Median imputation was selected because it is less sensitive to extreme
values than mean imputation.

### Standard Scaling

`StandardScaler` is used to place numerical features on comparable
scales.

This is particularly important for scale-sensitive algorithms such as:

- Logistic Regression
- Support Vector Machine
- k-Nearest Neighbours

---

## 5. Categorical Preprocessing

The current model feature set contains four categorical predictors:

- `state_name`
- `county_name`
- `city_name`
- `cbsa_name`

Categorical preprocessing consists of:

1. Most-frequent missing-value imputation
2. One-hot encoding

`OneHotEncoder(handle_unknown="ignore")` is used so that previously
unseen categories in validation/test data do not cause the pipeline
to fail.

---

## 6. ColumnTransformer

The numerical and categorical preprocessing pipelines are combined
using scikit-learn's `ColumnTransformer`.

Pipeline structure:

Numerical Features:

`Median Imputer -> StandardScaler`

Categorical Features:

`Most-Frequent Imputer -> OneHotEncoder`

This creates one reusable preprocessing component that can be used
consistently during model training and prediction.

---

## 7. Data Leakage Prevention

Data leakage prevention is a major design requirement because this
project predicts the following year's PM2.5 risk category.

The preprocessing pipeline is therefore NOT fitted on the complete
dataset.

After the chronological train/validation/test split is created, the
pipeline must be fitted using training data only.

Example:

```python
preprocessor = fit_preprocessor(X_train)

X_train_processed = transform_features(
    preprocessor,
    X_train
)

X_val_processed = transform_features(
    preprocessor,
    X_val
)

X_test_processed = transform_features(
    preprocessor,
    X_test
)