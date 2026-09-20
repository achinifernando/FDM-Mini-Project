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

This project builds a machine learning system to predict the next-year PM2.5 air quality risk category for EPA monitoring sites across the United States.

## Dataset

- **Source:** [Kaggle - EPA Air Quality](https://www.kaggle.com/datasets/epa/air-quality)
- **Original Source:** US EPA Air Quality System (AQS)
- **Time Range:** 1987-2017 (PM2.5: 1998-2017)
- **Target:** Next-year PM2.5 risk category (Good, Moderate, Unhealthy for Sensitive Groups, Unhealthy, Very Unhealthy)

## Quick Start

```bash
# Clone repository
git clone https://github.com/your-username/air-quality-risk-classification.git
cd air-quality-risk-classification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run full pipeline
python scripts/run_pipeline.py

# Launch application
streamlit run app/frontend/streamlit_app.py
