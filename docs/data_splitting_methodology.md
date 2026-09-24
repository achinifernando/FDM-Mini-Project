# Data Quality Evaluation and Dataset Splitting Methodology

## 1. Dataset Used

The input dataset used for data-quality evaluation and dataset splitting was:

`data/interim/ml_ready_base.csv`

This dataset was produced after the data engineering and feature engineering stages.

- Number of records: 16,535
- Number of columns: 38
- Number of unique monitoring sites: 1,849
- Year range: 1997–2016
- Target variable: `pm25_risk_category_next_year`

The objective of the project is to predict the next-year PM2.5 air-quality risk category using historical air-quality and monitoring-site information.

## 2. Data-Quality Evaluation

A data-quality evaluation was conducted before splitting the dataset. The following checks were performed:

- Dataset dimensions
- Column names and data types
- Missing values
- Fully duplicated rows
- Duplicated site-year records
- Infinite numerical values
- Missing target values
- Target-class distribution
- Invalid or non-numeric values in numerical columns

### 2.1 Missing and Invalid Values

No standard missing values were detected in the dataset.

However, 27 records contained the text value `unknown` in each of the following numerical columns:

- `ten_percentile`
- `p99_p10_range`

Although these values were not detected as standard null values, they represent invalid values in numerical features.

These values must be converted to `NaN` during preprocessing. The missing values must then be imputed using a statistic calculated only from the training dataset, such as the training median.

The same fitted training median must be applied to the validation and testing datasets to prevent data leakage.

### 2.2 Duplicate Records

The following duplicate checks were performed:

- Fully duplicated rows: 0
- Duplicated site-year records: 0

Therefore, no records were removed as duplicates.

### 2.3 Additional Invalid-Value Checks

The dataset contained:

- Infinite numerical values: 0
- Missing target values: 0

Apart from the 27 `unknown` values identified in the two numerical columns, no other invalid values were found during this evaluation.

## 3. Original Target-Class Distribution

Before merging the rare class, the target variable contained six categories:

| Target Class | Number of Records | Percentage |
|---|---:|---:|
| Moderate | 9,418 | 56.96% |
| Unhealthy for Sensitive Groups | 5,578 | 33.73% |
| Unhealthy | 1,398 | 8.45% |
| Good | 86 | 0.52% |
| Very Unhealthy | 51 | 0.31% |
| Hazardous | 4 | 0.02% |

The dataset has a significant class imbalance. The `Moderate` class is the majority class, while the `Hazardous`, `Very Unhealthy`, and `Good` classes contain comparatively few records.

Therefore, model evaluation should not depend only on accuracy. Metrics such as macro precision, macro recall, macro F1-score, balanced accuracy, and the confusion matrix should also be considered.

## 4. Rare-Class Merging

The `Hazardous` category contained only four records. This number was insufficient to represent the category reliably across the training, validation, and testing datasets.

Following the group leader's decision, the `Hazardous` category was merged into the `Very Unhealthy` category.

The transformation was:

`Hazardous → Very Unhealthy`

After merging, the final target contained five categories:

1. Good
2. Moderate
3. Unhealthy for Sensitive Groups
4. Unhealthy
5. Very Unhealthy

The `Very Unhealthy` category contained 55 records after the merge.

## 5. Dataset Splitting Strategy

A chronological split was used instead of a random split because the project predicts the next-year PM2.5 risk category.

In a real-world prediction setting, a model is trained using historical observations and evaluated using future observations. A random split could place future-year records in the training dataset and older records in the testing dataset.

This would produce an unrealistic evaluation and could introduce look-ahead leakage. Therefore, the records were divided according to their year.

| Dataset | Year Range | Number of Records |
|---|---|---:|
| Training set | 1997–2012 | 13,191 |
| Validation set | 2013–2014 | 1,783 |
| Testing set | 2015–2016 | 1,561 |
| Total | 1997–2016 | 16,535 |

The sum of records in all three datasets is equal to the number of records in the original ML-ready dataset. Therefore, no records were lost during splitting.

The splitting process also verified that the year ranges of the training, validation, and testing datasets did not overlap.

## 6. Generated Dataset Files

The splitting script generated the following files inside `data/processed/`:

- `train_data.csv`
- `validation_data.csv`
- `test_data.csv`
- `X_train.csv`
- `X_validation.csv`
- `X_test.csv`
- `y_train.csv`
- `y_validation.csv`
- `y_test.csv`

The `X` files contain the input features, while the `y` files contain the target variable.

Generated processed datasets are excluded from Git because they can be reproduced by running:

```bash
python src/data/make_dataset.py
```

## 7. Temporal Distribution Shift

A change in the target-class distribution was observed between the training and testing periods.

For example:

- `Moderate` represented approximately 51.92% of the training dataset.
- `Moderate` represented approximately 82.45% of the testing dataset.

This difference indicates a temporal distribution shift. The distribution of the air-quality risk categories changed over time.

This is not a dataset-splitting error. It reflects a real difference between historical and more recent observations.

The chronological split was retained because it provides a more realistic evaluation of the model's ability to predict future-year PM2.5 risk categories.

This temporal distribution shift should be considered when interpreting the final model results and should be documented as a project limitation.

## 8. Data-Leakage Prevention

The following steps were taken to prevent or reduce data leakage:

1. Records were split chronologically using the `year` column.
2. Earlier years were used for training, while later years were reserved for validation and testing.
3. The target column was separated from the input features.
4. The splitting script verified that the year ranges did not overlap.
5. Validation and testing data must not be used to calculate preprocessing statistics.
6. Missing-value imputation, scaling, encoding, and other fitted preprocessing operations must be learned only from the training dataset.
7. The preprocessing operations fitted on the training dataset must be applied unchanged to the validation and testing datasets.
8. The validation dataset may be used for model selection and hyperparameter tuning.
9. The final testing dataset must be reserved for the final unbiased model evaluation.

This approach ensures that information from future years does not influence the model during training.

## 9. Implementation Files

The following files were created for this part of the project:

- `src/data/data_quality.py` — performs data-quality checks and generates the data-quality report.
- `src/data/make_dataset.py` — performs rare-class merging and chronological dataset splitting.
- `reports/data_quality_summary.txt` — contains the generated data-quality results.
- `docs/data_splitting_methodology.md` — documents the data-quality evaluation and dataset-splitting methodology.

## 10. Summary

The ML-ready dataset was evaluated for missing values, duplicate records, invalid values, and target-class imbalance before splitting.

The rare `Hazardous` category was merged with the `Very Unhealthy` category, resulting in five final target classes.

A chronological splitting strategy was selected to support realistic next-year PM2.5 risk prediction and prevent future-year information from entering the training dataset.

The resulting training, validation, and testing datasets preserve the time order of the observations and provide a suitable foundation for preprocessing, model development, and final evaluation.