# Air Quality Risk Classification Using EPA Historical Monitoring Data

## IT3051 - Fundamentals of Data Mining | Mini Project 2026

### Group: DataBytes

| Student ID | Name |
|:---|:---|
| IT23688490 | Fernando A A I |
| IT23714434 | Rajasekara L.G.P.B |
| IT23715974 | Marasinghe M.A.P.I |
| IT23709966 | Edirisingha E A P K |

## Project Overview

This project provides an annual planning estimate of the next-year PM2.5 risk category for an EPA monitoring site. The app is organized into Location Information, Air Quality Measurements, Monitoring Information, and Prediction Result. Staff can select an available EPA site-year or enter annual statistics manually, see an interactive measurement profile and predicted-category scale, set an outdoor-activity alert threshold, and optionally upload a CSV for regional predictions of up to 200 sites.

The category is an engineered project indicator based on the following year's maximum reported daily PM2.5 concentration. It is not an official AQI forecast, health advisory, or causal explanation of the model. The app uses the saved model as-is and does not retrain or recalibrate it. A model probability estimate is shown only if the saved model provides one; it is not a guarantee of accuracy.

## Dataset

- **Source:** [Kaggle - EPA Air Quality](https://www.kaggle.com/datasets/epa/air-quality)
- **Original Source:** US EPA Air Quality System (AQS)
- **Time Range:** 1987-2017 (PM2.5: 1998-2017)
- **Target:** Next-year PM2.5 risk category (Good, Moderate, Unhealthy for Sensitive Groups, Unhealthy, Very Unhealthy)

## Quick Start

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate
# Windows PowerShell: .\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Launch the prediction app
python app/backend/server.py
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser. Select a state, county, city, metro area, monitoring site, and available year from the cascading menus; readings and coordinates from `data/interim/ml_ready_base.csv` are filled in and may be edited. Alternatively, switch to manual entry. Optional recent annual means can be provided when available; missing history is explicitly represented in the model features. The annual profile updates with the entered percentile readings, and the result diagram highlights the predicted category.

The three alert settings change only the activity-guidance threshold, not the predicted category:

- **General public:** alert at Unhealthy or higher.
- **Sensitive groups:** alert at Unhealthy for Sensitive Groups or higher.
- **High precaution:** alert at Moderate or higher.

For a regional overview, use **Download CSV template**, fill one row per site, and upload the CSV. The API accepts up to 200 site rows per batch, reports invalid rows individually, and returns a downloadable results CSV. The batch endpoint is `POST /api/predict/batch` with `Content-Type: text/csv` and optional `X-Sensitivity: general|sensitive|high`.

The server loads the fitted preprocessing and prediction pipeline from `models/best_model.joblib` and uses the feature engineering in `src/features/feature_engineering.py`. The repository identifies the EPA Air Quality System as the original source and documents a 1987–2017 dataset range, with the model-ready dropdown records spanning 1997–2016. The repository does not contain the exact BigQuery project, table, extraction query, or query-level date filter; those details must be recovered from the original extraction before claiming that the BigQuery build can be reproduced exactly. Run the backend tests with `python -m pytest tests/test_prediction_app.py`.
