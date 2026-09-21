# Member 1 – Data Engineering and Feature Engineering

## Project
Air Quality Risk Classification Using EPA Historical Monitoring Data

## Module
IT3051 – Fundamentals of Data Mining

---

## 1. Member 1 Responsibility

Member 1 is responsible for the Data Engineering and Feature Engineering stage
of the project.

The main objective of this stage is to transform the raw EPA Air Quality Annual
Summary dataset into a validated, cleaned, site-year-level, leakage-safe,
feature-engineered dataset that can be used by the remaining team members for:

- Exploratory Data Analysis (EDA)
- Data preprocessing
- Dataset splitting
- Machine-learning model development
- Model evaluation

The Member 1 pipeline includes:

1. Raw data loading
2. Schema and data-quality validation
3. PM2.5 record extraction
4. Data cleaning
5. Pollutant-standard deduplication
6. Site-year aggregation
7. Next-year target engineering
8. Historical feature engineering
9. Target-leakage prevention
10. Final dataset validation

---

# 2. Source Dataset

## Dataset

EPA Air Quality Annual Summary

## Original Source

U.S. Environmental Protection Agency (EPA) Air Quality System (AQS)

## Dataset Used

Kaggle EPA Air Quality dataset

Raw file:

data/raw/air_quality.csv

## Raw Dataset Size

2,038,710 rows × 55 columns

The project focuses on historical PM2.5 monitoring information.

EPA parameter code used:

88101 – PM2.5 mass concentration under local conditions

The usable PM2.5 records in the supplied dataset cover the period:

1997–2017

---

# 3. Project Structure for Member 1

The Member 1 implementation uses the following structure:

src/
├── data/
│   ├── __init__.py
│   ├── load_data.py
│   ├── validate_data.py
│   └── clean_data.py
│
└── features/
    ├── __init__.py
    ├── target_engineering.py
    └── feature_engineering.py

Generated outputs:

data/
├── raw/
│   └── air_quality.csv
│
└── interim/
    ├── pm25_extracted.csv
    ├── pm25_aggregated.csv
    ├── pm25_with_target.csv
    └── ml_ready_base.csv

reports/
└── data_quality_report.json

Documentation:

MEMBER1_README.md

---

# 4. Data Loading

File:

src/data/load_data.py

The loader is designed to support the large EPA dataset while preserving
important monitoring-site identifiers.

Main responsibilities:

- Load the raw EPA CSV dataset.
- Preserve state, county, and monitoring-site codes as strings.
- Prevent loss of leading zeros in geographic identifiers.
- Support loading selected columns.
- Support chunk-based loading for large files.
- Provide a reusable iterator for memory-conscious processing.

The full raw dataset contains:

2,038,710 rows × 55 columns

For large-file processing, the pipeline supports chunk sizes such as:

250,000 rows per chunk

---

# 5. Data Validation

File:

src/data/validate_data.py

Before cleaning, the raw EPA dataset is validated to confirm that the required
variables are available and that major data-quality problems can be identified.

Validation includes:

- Required-column validation
- Missing-value reporting
- Full-row duplicate detection
- Numeric conversion checks
- Latitude range validation
- Longitude range validation
- Observation-percentage validation
- Negative observation-count detection
- Negative valid-day-count detection

A data-quality report is generated at:

reports/data_quality_report.json

The validation stage confirmed that the required EPA fields used by the
Member 1 pipeline are available.

---

# 6. PM2.5 Extraction and Cleaning

File:

src/data/clean_data.py

The raw EPA dataset contains records for multiple pollutants and monitoring
configurations.

The Member 1 pipeline selects the records relevant to the PM2.5 prediction
problem.

## PM2.5 Parameter

The following EPA parameter code is retained:

88101

This prevents unrelated pollutants and other PM2.5-related parameter codes from
being included in the modelling dataset.

---

# 7. Identifier Normalization

The following monitoring-site identifiers are normalized:

- state_code
- county_code
- site_num

They are stored as strings and padded to consistent lengths.

A unique monitoring-site identifier is then constructed using:

state_code-county_code-site_num

This produces:

site_id

The site identifier is used throughout target engineering and historical
feature engineering to ensure that observations are matched within the same
monitoring site.

---

# 8. Numeric Data Cleaning

Relevant measurement and monitoring variables are converted to numeric format.

Examples include:

- latitude
- longitude
- year
- observation_count
- observation_percent
- valid_day_count
- arithmetic_mean
- arithmetic_standard_dev
- first_max_value
- 99th percentile
- 98th percentile
- 95th percentile
- 90th percentile
- 75th percentile
- median
- 10th percentile

Invalid geographic or monitoring records are filtered using appropriate
validity checks.

Examples:

Latitude:

-90 ≤ latitude ≤ 90

Longitude:

-180 ≤ longitude ≤ 180

Observation percentage:

0 ≤ observation_percent ≤ 100

Negative observation counts and negative valid-day counts are not retained as
valid monitoring information.

Negative PM2.5 concentration statistics are treated as missing because negative
mass concentrations are not meaningful for this project.

---

# 9. PM2.5 Units

The cleaning stage retains PM2.5 records reported in micrograms per cubic meter.

The unit filtering is based on records containing:

Micrograms/cubic meter

This keeps the concentration scale consistent for downstream feature and target
engineering.

---

# 10. Sample-Duration Filtering

The engineered target is based on a 24-hour PM2.5 concentration statistic.

Therefore, the pipeline retains:

- 24 HOUR
- 24-HR BLK AVG

The pipeline excludes 1-hour summaries from the target dataset.

Final retained sample-duration counts after global cleaning:

24 HOUR:
20,522 records

24-HR BLK AVG:
2,760 records

Total cleaned PM2.5 extracted records:

23,282 rows

---

# 11. Pollutant-Standard Deduplication

EPA annual-summary data may represent monitoring information under multiple
pollutant standards.

To prevent duplicated representations from influencing the site-year dataset,
a pollutant-standard priority rule is applied.

Configured priority:

1. PM25 24-hour 2012
2. PM25 24-hour 2006
3. PM25 Annual 2012
4. PM25 Annual 2006

Deduplication uses the following combination:

- site_id
- year
- POC
- sample_duration

Separate POCs are retained because different POCs may represent different
monitoring configurations at the same physical site.

The priority rule is first applied while processing individual chunks.

Importantly, it is applied again after all chunks are concatenated.

This global deduplication step prevents competing pollutant-standard records
located in different raw CSV chunks from surviving simply because they were
processed separately.

After global standard deduplication, the retained pollutant-standard
representation is:

PM25 24-hour 2012:
23,282 records

---

# 12. Cleaned PM2.5 Extraction Result

Output:

data/interim/pm25_extracted.csv

Final shape:

23,282 rows × 38 columns

Sample durations:

24 HOUR:
20,522

24-HR BLK AVG:
2,760

This dataset represents the cleaned PM2.5 records before site-year aggregation.

---

# 13. Site-Year Aggregation

EPA data can contain multiple monitoring records for the same physical site and
calendar year.

For downstream machine-learning purposes, these records are collapsed into one
site-year representation.

Output:

data/interim/pm25_aggregated.csv

Final shape:

18,638 rows × 24 columns

Year range:

1997–2017

Unique monitoring sites:

1,944

Duplicate site-year rows:

0

---

# 14. Site-Year Statistics

The site-year aggregation retains historical PM2.5 and monitoring-quality
statistics.

These include:

- PM2.5 mean
- Arithmetic standard deviation
- Maximum reported PM2.5 concentration
- 99th percentile
- 98th percentile
- 95th percentile
- 90th percentile
- 75th percentile
- Median
- 10th percentile
- Observation count
- Observation percentage
- Valid-day count
- Latitude
- Longitude
- State information
- County information
- City information
- CBSA information

The PM2.5 mean is calculated using an observation-count-weighted aggregation
across the retained monitoring records.

High-concentration statistics are preserved using aggregation rules suitable for
retaining high-pollution information at the site-year level.

---

# 15. Target Engineering

File:

src/features/target_engineering.py

The target variable is:

pm25_risk_category_next_year

The objective is to predict the PM2.5 health-risk category for the following
calendar year at the same monitoring site.

For a monitoring site S and prediction year t:

Features:
Site S at year t

Target:
Observed risk category for Site S at exactly year t+1

For example:

Site A – 2010 → Site A – 2011 target

The pipeline does not treat a non-consecutive observation such as:

2010 → 2012

as a one-year-ahead target.

This ensures that the prediction task represents an actual next-calendar-year
classification problem.

---

# 16. Target Concentration Statistic

The target is based on:

first_max_value

after PM2.5 cleaning, 24-hour record filtering, pollutant-standard
deduplication, and site-year aggregation.

This represents the maximum reported 24-hour PM2.5 concentration statistic
retained for the site-year representation.

The next year's value is temporarily stored as:

next_year_pm25_target_concentration

This value is used only for constructing the categorical supervised target.

IMPORTANT:

The resulting target is an engineered annual prediction indicator based on a
24-hour PM2.5 concentration statistic.

It should NOT be described as an official annual EPA AQI value.

---

# 17. PM2.5 Risk Categories

The project uses the following AQI-style PM2.5 concentration categories:

Good

Moderate

Unhealthy for Sensitive Groups

Unhealthy

Very Unhealthy

Hazardous

Configured concentration boundaries:

Good:
≤ 9.0 µg/m³

Moderate:
> 9.0 to ≤ 35.4 µg/m³

Unhealthy for Sensitive Groups:
> 35.4 to ≤ 55.4 µg/m³

Unhealthy:
> 55.4 to ≤ 125.4 µg/m³

Very Unhealthy:
> 125.4 to ≤ 225.4 µg/m³

Hazardous:
> 225.4 µg/m³

These categories are used to construct the project's engineered next-year
health-risk classification target.

---

# 18. Final Target Dataset

Output:

data/interim/pm25_with_target.csv

Rows with valid next-year targets:

16,535

Prediction years:

1997–2016

The prediction period ends in 2016 because a feature-year observation requires
an actual following-year observation to construct the target.

The target concentration range is:

0.1–505.6 µg/m³

---

# 19. Final Target Distribution

The final target distribution after corrected global pollutant-standard
deduplication is:

Moderate:
9,418

Unhealthy for Sensitive Groups:
5,578

Unhealthy:
1,398

Good:
86

Very Unhealthy:
51

Hazardous:
4

Total supervised rows:

16,535

The target is strongly imbalanced.

In particular, the following classes contain relatively few observations:

- Good
- Very Unhealthy
- Hazardous

The original engineered target is retained by Member 1.

Any later decision to merge rare classes, use class weighting, resampling, or
apply another imbalance-handling strategy should be evaluated and documented
during the modelling stage.

---

# 20. Feature Engineering

File:

src/features/feature_engineering.py

Historical features are generated using information from the current prediction
year t and earlier years only.

No information from year t+1 is intentionally used as a model predictor.

The following engineered features are created.

---

## 20.1 Previous-Year PM2.5

Feature:

pm25_lag1

Definition:

PM2.5 mean from exactly year t-1 for the same monitoring site.

The feature is created only when the immediately previous available site record
corresponds to the previous calendar year.

---

## 20.2 Two-Year Lag

Feature:

pm25_lag2

Definition:

PM2.5 mean from the same site two calendar years before the prediction year,
subject to the historical availability rules used by the feature pipeline.

---

## 20.3 One-Year Absolute Change

Feature:

pm25_change_1yr

Definition:

pm25_mean(t) - pm25_lag1

This measures the recent absolute PM2.5 trend.

---

## 20.4 One-Year Percentage Change

Feature:

pm25_pct_change_1yr

Definition:

(pm25_mean(t) - pm25_mean(t-1)) / pm25_mean(t-1)

Division by zero is prevented by treating a zero denominator as missing.

---

## 20.5 Two-Year Rolling Mean

Feature:

pm25_rolling_2yr_mean

Definition:

Mean of:

- Current-year PM2.5 mean
- Previous-year PM2.5 mean

The feature is available when exact previous-year history exists.

---

## 20.6 Three-Year Rolling Mean

Feature:

pm25_rolling_3yr_mean

Uses:

- t
- t-1
- t-2

The feature requires the historical values used by the three-year feature
construction to be available.

---

## 20.7 Three-Year Rolling Standard Deviation

Feature:

pm25_rolling_3yr_std

This measures variation across the three-year PM2.5 history used by the
feature-engineering pipeline.

---

## 20.8 Pollution Spread

Feature:

p95_minus_median

Definition:

95th percentile - median

This represents the difference between higher pollution levels and the typical
site-year concentration.

---

## 20.9 Distribution Range

Feature:

p99_p10_range

Definition:

99th percentile - 10th percentile

This provides an additional measure of PM2.5 concentration spread.

---

## 20.10 Maximum-to-Mean Ratio

Feature:

max_to_mean_ratio

Definition:

first_max_value / pm25_mean

This compares the maximum reported PM2.5 concentration with the typical annual
level represented by the aggregated PM2.5 mean.

---

## 20.11 Observation Density

Feature:

observations_per_valid_day

Definition:

observation_count / valid_day_count

This provides information about monitoring density.

---

## 20.12 Previous-Year Availability

Feature:

has_previous_year

Values:

1 – Exact previous-year PM2.5 history is available

0 – Exact previous-year PM2.5 history is unavailable

---

## 20.13 Historical Availability

Feature:

has_three_year_history

This indicates whether the historical information required by the pipeline's
longer-history feature construction is available.

---

# 21. Target-Leakage Prevention

Preventing target leakage is a critical requirement of the Member 1 pipeline.

The following temporary column contains the actual future PM2.5 concentration
used to create the supervised target:

next_year_pm25_target_concentration

This variable must never be supplied to the machine-learning model.

Therefore, before the final feature dataset is saved, the pipeline explicitly
removes:

next_year_pm25_target_concentration

It also removes legacy target-related variables if present:

next_year_pm25_mean

target_risk

A final runtime validation confirms that the future target concentration does
not remain in the feature dataset.

Final result:

Future target concentration present:
False

---

# 22. Final ML-Ready Base Dataset

Output:

data/interim/ml_ready_base.csv

Final shape:

16,535 rows × 38 columns

Prediction years:

1997–2016

This is the main Member 1 output intended for downstream preprocessing and
machine-learning work.

---

# 23. Final Dataset Validation

The final ML-ready base dataset was validated after completing the full
Member 1 pipeline.

Validation results:

Duplicate site-year rows:
0

Duplicate full rows:
0

Missing target values:
0

Future target leakage column present:
False

Infinite numeric values:
0

Total missing values:
22,468

These results confirm that:

- Each supervised row represents a unique site-year.
- Duplicate full records are not present.
- Every supervised row has a target.
- The future concentration used to construct the target is not retained as a
  predictor.
- Infinite numeric values are not present.

---

# 24. Missing Values in the Final Dataset

Some missing values remain intentionally because historical monitoring coverage
is not complete for every monitoring site.

Major missing-value counts include:

pm25_rolling_3yr_std:
3,766

pm25_rolling_3yr_mean:
3,766

pm25_lag2:
3,766

pm25_pct_change_1yr:
1,996

pm25_lag1:
1,996

pm25_change_1yr:
1,996

pm25_rolling_2yr_mean:
1,996

city_name:
1,967

cbsa_name:
1,165

ten_percentile:
27

p99_p10_range:
27

These values are not automatically filled during Member 1 data engineering.

Historical lag and rolling-feature missingness can occur when a monitoring site
does not have the required consecutive historical observations.

Missing contextual variables such as city_name and cbsa_name originate from the
available source information.

Missing-value treatment should be performed in the downstream preprocessing
pipeline using an explicitly documented strategy.

---

# 25. Final Generated Files

## Raw Input

data/raw/air_quality.csv

Raw EPA dataset.

---

## Data Quality Report

reports/data_quality_report.json

Contains validation and data-quality information for the raw dataset.

---

## Cleaned PM2.5 Dataset

data/interim/pm25_extracted.csv

Shape:

23,282 × 38

Contains cleaned PM2.5 monitoring records after 24-hour filtering and global
pollutant-standard deduplication.

---

## Site-Year Aggregated Dataset

data/interim/pm25_aggregated.csv

Shape:

18,638 × 24

Contains one aggregated representation for each retained monitoring site and
year.

---

## Target-Engineered Dataset

data/interim/pm25_with_target.csv

Contains site-year features together with the engineered next-year PM2.5 risk
target and temporary future concentration used during target construction.

This file is an intermediate dataset and should not be supplied directly to a
model without removing the future concentration field.

---

## Final ML-Ready Base Dataset

data/interim/ml_ready_base.csv

Shape:

16,535 × 38

This is the final Member 1 handover dataset.

The future target concentration has been removed from this file.

---

# 26. Pipeline Execution Order

Run all commands from the project root directory.

## Step 1 – Test Data Loading

python -m src.data.load_data

Expected raw shape:

2,038,710 × 55

---

## Step 2 – Validate Raw Dataset

python -m src.data.validate_data

Output:

reports/data_quality_report.json

---

## Step 3 – Clean and Aggregate PM2.5 Data

python -m src.data.clean_data

Final expected output:

PM2.5 extracted:
(23282, 38)

Site-year aggregated:
(18638, 24)

Year range:
1997–2017

Unique sites:
1,944

Duplicate site-year rows:
0

---

## Step 4 – Create Next-Year Target

python -m src.features.target_engineering

Expected supervised rows:

16,535

Expected prediction years:

1997–2016

Expected target distribution:

Moderate:
9,418

Unhealthy for Sensitive Groups:
5,578

Unhealthy:
1,398

Good:
86

Very Unhealthy:
51

Hazardous:
4

---

## Step 5 – Generate Final Features

python -m src.features.feature_engineering

Expected final shape:

16,535 × 38

Expected leakage check:

Future target concentration present:
False

Expected target column:

pm25_risk_category_next_year

---

# 27. Handover to Member 2 – Exploratory Data Analysis

Member 2 can use the cleaned/aggregated datasets and final Member 1 output to
perform EDA.

Recommended areas include:

- PM2.5 distribution
- Target-class distribution
- Yearly pollution trends
- Geographic patterns
- Missing-value patterns
- Observation completeness
- Extreme concentration behaviour
- Feature relationships
- Class imbalance analysis

Member 2 should be informed that the target distribution is strongly imbalanced.

---

# 28. Handover to Member 3 – Preprocessing

The main input for downstream preprocessing is:

data/interim/ml_ready_base.csv

Member 3 should determine and document appropriate strategies for:

- Missing numerical values
- Missing categorical values
- Categorical encoding
- Feature scaling where required
- Model-specific preprocessing
- Class imbalance handling where appropriate

Missing lag and rolling values should not automatically be interpreted as data
errors because many are caused by incomplete monitoring histories.

---

# 29. Handover for Dataset Splitting and Modelling

The dataset represents a temporal prediction problem.

Therefore, downstream evaluation should preserve the chronological nature of
the prediction task.

The project should avoid introducing future information into earlier training
periods.

Where required by the project methodology, site grouping should also be
considered when evaluating generalization.

The final target is:

pm25_risk_category_next_year

The future concentration used to create this target is NOT included in the
final ML-ready dataset.

---

# 30. Important Limitations

## 30.1 Incomplete Monitoring Histories

Not every monitoring site has a complete consecutive annual history.

This causes expected missing values in lag and rolling features.

---

## 30.2 Target-Class Imbalance

The target distribution is strongly imbalanced.

The Hazardous class contains only:

4 observations

The Very Unhealthy class contains:

51 observations

The Good class contains:

86 observations

This should be considered carefully during modelling and evaluation.

---

## 30.3 Engineered Risk Target

The target is an engineered AQI-style next-year health-risk indicator based on
a maximum reported 24-hour PM2.5 concentration statistic.

It should not be described as an official annual EPA AQI.

---

## 30.4 Historical Dataset

The usable PM2.5 records in this implementation cover:

1997–2017

The final supervised prediction years are:

1997–2016

Therefore, the dataset represents historical monitoring conditions and should
not be interpreted as current air-quality conditions.

---

## 30.5 Contextual Missing Values

Some location-related fields, including city_name and cbsa_name, contain
missing values.

These should be handled during downstream preprocessing rather than filled with
unsupported information during data engineering.

---

# 31. Final Member 1 Results Summary

Raw EPA dataset:

2,038,710 × 55

Cleaned PM2.5 extracted dataset:

23,282 × 38

Site-year aggregated dataset:

18,638 × 24

Unique sites before supervised target pairing:

1,944

Final supervised ML-ready dataset:

16,535 × 38

Prediction years:

1997–2016

Duplicate site-year rows:

0

Duplicate full rows:

0

Missing target values:

0

Infinite numeric values:

0

Future target concentration present:

False

Final target:

pm25_risk_category_next_year

Final target distribution:

- Moderate: 9,418
- Unhealthy for Sensitive Groups: 5,578
- Unhealthy: 1,398
- Good: 86
- Very Unhealthy: 51
- Hazardous: 4

---

# 32. Member 1 Final Deliverable

The Member 1 pipeline transforms the raw EPA annual air-quality dataset into a
cleaned, validated, site-year-level and feature-engineered dataset for
next-year PM2.5 risk classification.

The completed pipeline provides:

- Memory-conscious raw data loading
- Schema validation
- Data-quality reporting
- PM2.5 extraction
- 24-hour record filtering
- Global pollutant-standard deduplication
- Site-year aggregation
- Exact next-year target matching
- AQI-style engineered risk classification
- Historical lag features
- Trend features
- Rolling historical features
- Distribution-based features
- Monitoring-quality features
- Explicit target-leakage prevention
- Final dataset integrity validation

The final handover dataset is:

data/interim/ml_ready_base.csv

This dataset is ready for the downstream EDA, preprocessing, splitting,
modelling, and evaluation stages of the project.