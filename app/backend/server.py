"""Local HTTP server and prediction API for the PM2.5 risk model."""

from __future__ import annotations

import json
import math
import sys
import csv
import io
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.feature_engineering import (
    HISTORY_FEATURES,
    add_history_features,
    prepare_model_features,
)


FRONTEND_DIR = PROJECT_ROOT / "app" / "frontend"
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
HOST = "127.0.0.1"
PORT = 8000
MAX_REQUEST_BYTES = 16_384
MAX_BATCH_ROWS = 200
_MODEL: Any | None = None
_SITE_DATA: pd.DataFrame | None = None
NUMERIC_FEATURES = (
    "year",
    "latitude",
    "longitude",
    "observation_count",
    "observation_percent",
    "valid_day_count",
    "arithmetic_standard_dev",
    "first_max_value",
    "ninety_nine_percentile",
    "ninety_eight_percentile",
    "ninety_five_percentile",
    "ninety_percentile",
    "seventy_five_percentile",
    "fifty_percentile",
    "ten_percentile",
    "pm25_mean",
    *HISTORY_FEATURES,
    "p95_minus_median",
    "p99_p10_range",
    "max_to_mean_ratio",
    "observations_per_valid_day",
    "has_previous_year",
    "has_three_year_history",
)
TEXT_FEATURES = ("state_name", "county_name", "city_name", "cbsa_name")
LOCATION_COLUMNS = (
    "site_id",
    "state_name",
    "county_name",
    "city_name",
    "cbsa_name",
    "year",
    "latitude",
    "longitude",
    "observation_count",
    "observation_percent",
    "valid_day_count",
    "arithmetic_standard_dev",
    "first_max_value",
    "ninety_nine_percentile",
    "ninety_eight_percentile",
    "ninety_five_percentile",
    "ninety_percentile",
    "seventy_five_percentile",
    "fifty_percentile",
    "ten_percentile",
    "pm25_mean",
    "pm25_lag1",
    "pm25_lag2",
    "has_previous_year",
    "has_three_year_history",
)

NUMERIC_LIMITS = {
    "year": (1980, 2100),
    "latitude": (-90, 90),
    "longitude": (-180, 180),
    "observation_count": (1, 366),
    "observation_percent": (0, 100),
    "valid_day_count": (1, 366),
    "arithmetic_standard_dev": (0, 1000),
    "first_max_value": (0, 2000),
    "ninety_nine_percentile": (0, 2000),
    "ninety_eight_percentile": (0, 2000),
    "ninety_five_percentile": (0, 2000),
    "ninety_percentile": (0, 2000),
    "seventy_five_percentile": (0, 2000),
    "fifty_percentile": (0, 2000),
    "ten_percentile": (0, 2000),
    "pm25_mean": (0, 2000),
    "previous_year_pm25_mean": (0, 2000),
    "two_years_ago_pm25_mean": (0, 2000),
}
BATCH_REQUIRED_COLUMNS = {
    "site_id",
    *TEXT_FEATURES,
    *(
        field for field in NUMERIC_LIMITS
        if field not in {"previous_year_pm25_mean", "two_years_ago_pm25_mean"}
    ),
}

CATEGORY_DETAILS = {
    "Good": {
        "range": "Up to 9 µg/m³",
        "description": "The model places the next-year site in the lowest project risk category.",
        "style": "success",
    },
    "Moderate": {
        "range": "Above 9 to 35.4 µg/m³",
        "description": "The model places the next-year site in the moderate project risk category.",
        "style": "warning",
    },
    "Unhealthy for Sensitive Groups": {
        "range": "Above 35.4 to 55.4 µg/m³",
        "description": "The model places the next-year site in the sensitive-groups project risk category.",
        "style": "warning",
    },
    "Unhealthy": {
        "range": "Above 55.4 to 125.4 µg/m³",
        "description": "The model places the next-year site in the unhealthy project risk category.",
        "style": "error",
    },
    "Very Unhealthy": {
        "range": "Above 125.4 to 225.4 µg/m³",
        "description": "The model places the next-year site in the highest represented project risk category.",
        "style": "error",
    },
}

SENSITIVITY_LEVELS = {
    "general": {
        "label": "General public",
        "minimum_class": "Unhealthy",
    },
    "sensitive": {
        "label": "Sensitive groups",
        "minimum_class": "Unhealthy for Sensitive Groups",
    },
    "high": {
        "label": "High precaution",
        "minimum_class": "Moderate",
    },
}

CATEGORY_ORDER = [
    "Good",
    "Moderate",
    "Unhealthy for Sensitive Groups",
    "Unhealthy",
    "Very Unhealthy",
]

ACTIVITY_GUIDANCE = {
    "Good": "Usual outdoor activities are generally reasonable.",
    "Moderate": "Most people can continue usual activities; people who are unusually sensitive should consider taking it easier outdoors.",
    "Unhealthy for Sensitive Groups": "People more sensitive to air pollution should reduce prolonged or strenuous outdoor activity.",
    "Unhealthy": "Everyone should reduce prolonged or strenuous outdoor activity.",
    "Very Unhealthy": "Avoid prolonged or strenuous outdoor activity where possible.",
}


class InputError(ValueError):
    """Raised when a prediction request contains invalid values."""


def _number(payload: dict[str, Any], field: str, *, optional: bool = False) -> float | None:
    value = payload.get(field)
    if optional and (value is None or value == ""):
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InputError(f"Enter a valid number for {field.replace('_', ' ')}.")
    number = float(value)
    if not math.isfinite(number):
        raise InputError(f"Enter a finite number for {field.replace('_', ' ')}.")
    minimum, maximum = NUMERIC_LIMITS[field]
    if number < minimum or number > maximum:
        raise InputError(
            f"{field.replace('_', ' ').capitalize()} must be between "
            f"{minimum:g} and {maximum:g}."
        )
    if field == "year" and not number.is_integer():
        raise InputError("Year must be a whole number.")
    if field in {"observation_count", "valid_day_count"} and not number.is_integer():
        raise InputError(f"{field.replace('_', ' ').capitalize()} must be a whole number.")
    return number


def _text(payload: dict[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise InputError(f"Enter {field.replace('_', ' ')}.")
    value = value.strip()
    if len(value) > 100:
        raise InputError(f"{field.replace('_', ' ').capitalize()} must be 100 characters or fewer.")
    return value


def _load_site_data() -> pd.DataFrame:
    """Load the project's real monitoring-site/year records once for form options."""
    global _SITE_DATA
    if _SITE_DATA is None:
        data_path = PROJECT_ROOT / "data" / "interim" / "ml_ready_base.csv"
        if not data_path.is_file():
            raise FileNotFoundError(f"Monitoring site data not found: {data_path}")
        _SITE_DATA = pd.read_csv(
            data_path,
            usecols=list(LOCATION_COLUMNS),
            dtype={
                "site_id": "string",
                "state_name": "string",
                "county_name": "string",
                "city_name": "string",
                "cbsa_name": "string",
            },
            low_memory=False,
        )
    return _SITE_DATA


def _site_options(filters: dict[str, str]) -> dict[str, list[str]]:
    """Return cascading dropdown choices restricted to matching real sites."""
    data = _load_site_data()
    filtered = data
    for field in ("state_name", "county_name", "city_name", "cbsa_name", "site_id"):
        value = filters.get(field, "").strip()
        if value:
            filtered = filtered.loc[filtered[field] == value]

    choices = {
        "state_name": data["state_name"].dropna().unique().tolist(),
        "county_name": [],
        "city_name": [],
        "cbsa_name": [],
        "site_id": [],
        "year": [],
    }
    if filters.get("state_name"):
        choices["county_name"] = (
            filtered["county_name"].dropna().unique().tolist()
            if not filters.get("county_name")
            else data.loc[
                data["state_name"].eq(filters["state_name"]),
                "county_name",
            ].dropna().unique().tolist()
        )
    if filters.get("county_name"):
        county_data = data.loc[
            data["state_name"].eq(filters.get("state_name"))
            & data["county_name"].eq(filters["county_name"])
        ]
        choices["city_name"] = county_data["city_name"].dropna().unique().tolist()
    if filters.get("city_name"):
        city_data = data.loc[
            data["state_name"].eq(filters.get("state_name"))
            & data["county_name"].eq(filters.get("county_name"))
            & data["city_name"].eq(filters["city_name"])
        ]
        choices["cbsa_name"] = city_data["cbsa_name"].dropna().unique().tolist()
    if filters.get("cbsa_name"):
        metro_data = data.loc[
            data["state_name"].eq(filters.get("state_name"))
            & data["county_name"].eq(filters.get("county_name"))
            & data["city_name"].eq(filters.get("city_name"))
            & data["cbsa_name"].eq(filters["cbsa_name"])
        ]
        choices["site_id"] = sorted(metro_data["site_id"].dropna().unique().tolist())
    if filters.get("site_id"):
        site_data = data.loc[data["site_id"].eq(filters["site_id"])]
        choices["year"] = sorted(
            site_data["year"].dropna().astype(int).astype(str).unique().tolist()
        )
    for field in choices:
        choices[field] = sorted(str(value) for value in choices[field])
    return choices


def _site_record(site_id: str, year: str) -> dict[str, Any] | None:
    """Return the feature inputs for one real site-year observation."""
    data = _load_site_data()
    record = data.loc[
        data["site_id"].eq(site_id) & data["year"].eq(int(year))
    ]
    if record.empty:
        return None
    row = record.iloc[0]
    values = row.to_dict()
    values["previous_year_pm25_mean"] = (
        values["pm25_lag1"]
        if values["has_previous_year"] == 1
        else None
    )
    values["two_years_ago_pm25_mean"] = (
        values["pm25_lag2"]
        if values["has_three_year_history"] == 1
        else None
    )
    return {
        key: (
            None if pd.isna(value)
            else int(value) if key in {"year", "observation_count", "valid_day_count"}
            else float(value) if isinstance(value, (int, float))
            else str(value)
        )
        for key, value in values.items()
    }


def _activity_assessment(category: str, sensitivity: str) -> dict[str, str | bool]:
    if sensitivity not in SENSITIVITY_LEVELS:
        raise InputError("Choose General public, Sensitive groups, or High precaution sensitivity.")
    minimum = SENSITIVITY_LEVELS[sensitivity]["minimum_class"]
    alert = CATEGORY_ORDER.index(category) >= CATEGORY_ORDER.index(minimum)
    if alert:
        guidance = ACTIVITY_GUIDANCE[category]
    else:
        guidance = "No activity alert at this sensitivity setting; follow routine local air-quality updates."
    return {
        "sensitivity": sensitivity,
        "sensitivity_label": SENSITIVITY_LEVELS[sensitivity]["label"],
        "activity_alert": alert,
        "activity_guidance": guidance,
    }


def _explain_prediction(features: pd.DataFrame) -> list[str]:
    row = features.iloc[0]
    factors: list[str] = []
    has_previous = bool(row["has_previous_year"])
    has_three_year_history = bool(row["has_three_year_history"])
    if has_three_year_history:
        earlier = float(row["pm25_lag2"])
        current = float(row["pm25_mean"])
        if earlier > 0:
            percent_change = (current - earlier) / earlier * 100
            if percent_change >= 10:
                factors.append(
                    f"The annual mean rose {percent_change:.0f}% over the available two-year history."
                )
            elif percent_change <= -10:
                factors.append(
                    f"The annual mean fell {abs(percent_change):.0f}% over the available two-year history."
                )
            else:
                factors.append("The annual mean was broadly steady over the available two-year history.")
    elif has_previous:
        previous = float(row["pm25_lag1"])
        current = float(row["pm25_mean"])
        direction = "increased" if current > previous else "decreased" if current < previous else "was unchanged"
        factors.append(
            f"The annual mean {direction} from {previous:.1f} to {current:.1f} µg/m³ versus the previous year."
        )
    else:
        factors.append("No complete annual history was supplied; this prediction uses the available current-year readings.")

    peak = float(row["first_max_value"])
    if peak > 35.4:
        factors.append(
            f"The highest daily reading in the input year was {peak:.1f} µg/m³, above the project's 35.4 µg/m³ reference point."
        )
    else:
        factors.append(
            f"The highest daily reading in the input year was {peak:.1f} µg/m³."
        )
    return factors[:2]


def _predict_one(payload: dict[str, Any], model: Any) -> dict[str, Any]:
    sensitivity = payload.get("sensitivity", "sensitive")
    if not isinstance(sensitivity, str):
        raise InputError("Sensitivity must be one of the available settings.")
    features = build_features(payload)
    expected = getattr(model, "feature_names_in_", None)
    if expected is None or set(features.columns) != set(expected):
        raise RuntimeError("Prediction inputs do not match the trained model feature schema.")
    features = features.loc[:, list(expected)]
    prediction = str(model.predict(features)[0])
    details = CATEGORY_DETAILS.get(prediction)
    if details is None:
        raise RuntimeError(f"The model returned an unsupported category: {prediction}")
    result: dict[str, Any] = {
        "category": prediction,
        **details,
        "confidence": None,
        "explanation": _explain_prediction(features),
    }
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        class_index = list(model.classes_).index(prediction)
        result["confidence"] = round(float(probabilities[class_index]) * 100, 1)
        result["confidence_label"] = "Model probability estimate"
    else:
        result["confidence_label"] = "Not available from this saved model"
    result.update(_activity_assessment(prediction, sensitivity))
    return result


def build_features(payload: dict[str, Any]) -> pd.DataFrame:
    """Validate input and apply the project's training-time feature engineering."""
    row: dict[str, Any] = {
        **{field: _text(payload, field) for field in TEXT_FEATURES},
        "site_id": _text(payload, "site_id"),
    }
    numeric_fields = tuple(
        field for field in NUMERIC_LIMITS
        if field not in {"previous_year_pm25_mean", "two_years_ago_pm25_mean"}
    )
    row.update({field: _number(payload, field) for field in numeric_fields})
    ordered_measurements = (
        "ten_percentile",
        "fifty_percentile",
        "seventy_five_percentile",
        "ninety_percentile",
        "ninety_five_percentile",
        "ninety_eight_percentile",
        "ninety_nine_percentile",
        "first_max_value",
    )
    if any(
        row[lower] > row[higher]
        for lower, higher in zip(ordered_measurements, ordered_measurements[1:])
    ):
        raise InputError(
            "Check the percentile readings: each higher percentile must be "
            "at least as large as the one before it."
        )
    current_year = int(row["year"])
    previous = _number(payload, "previous_year_pm25_mean", optional=True)
    two_years_ago = _number(payload, "two_years_ago_pm25_mean", optional=True)
    if two_years_ago is not None and previous is None:
        raise InputError(
            "Enter the previous-year mean before adding the two-years-ago mean."
        )

    records = [row]
    if previous is not None:
        records.append(
            {
                "site_id": row["site_id"],
                "year": current_year - 1,
                "pm25_mean": previous,
            }
        )
    if previous is not None and two_years_ago is not None:
        records.append(
            {
                "site_id": row["site_id"],
                "year": current_year - 2,
                "pm25_mean": two_years_ago,
            }
        )

    engineered = add_history_features(pd.DataFrame(records))
    prepared = prepare_model_features(engineered)
    features = prepared.loc[prepared["year"] == current_year].drop(
        columns=["site_id"]
    )
    features.loc[:, NUMERIC_FEATURES] = features.loc[
        :, NUMERIC_FEATURES
    ].apply(pd.to_numeric, errors="coerce")
    return features


def load_model() -> Any:
    """Load the final fitted pipeline and fail early if its schema is invalid."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)
    steps = getattr(model, "named_steps", {})
    if "preprocessor" not in steps or "classifier" not in steps:
        raise RuntimeError(
            "The saved model must contain its fitted preprocessor and classifier."
        )
    if not hasattr(steps["preprocessor"], "transformers_"):
        raise RuntimeError("The saved model preprocessor has not been fitted.")
    if not hasattr(model, "feature_names_in_"):
        raise RuntimeError("The saved model does not declare its input feature schema.")
    expected_features = set(NUMERIC_FEATURES) | set(TEXT_FEATURES)
    if set(model.feature_names_in_) != expected_features:
        raise RuntimeError(
            "The saved model's input feature schema does not match the prediction service."
        )
    if not set(model.classes_).issubset(CATEGORY_DETAILS):
        raise RuntimeError("The saved model contains unsupported prediction classes.")
    return model


def predict(payload: dict[str, Any], model: Any | None = None) -> dict[str, Any]:
    """Return the model's category, optional probability estimate, and activity alert."""
    global _MODEL
    if model is None:
        if _MODEL is None:
            _MODEL = load_model()
        model = _MODEL
    return _predict_one(payload, model)


def predict_batch(rows: list[dict[str, Any]], sensitivity: str, model: Any | None = None) -> list[dict[str, Any]]:
    """Predict a bounded batch and report row validation errors independently."""
    global _MODEL
    if not isinstance(rows, list) or not rows:
        raise InputError("Upload a CSV file containing at least one data row.")
    if len(rows) > MAX_BATCH_ROWS:
        raise InputError(f"Batch size cannot exceed {MAX_BATCH_ROWS} rows.")
    if not isinstance(sensitivity, str) or sensitivity not in SENSITIVITY_LEVELS:
        raise InputError("Choose a valid sensitivity setting.")
    if model is None:
        if _MODEL is None:
            _MODEL = load_model()
        model = _MODEL
    results: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            results.append({"row": index, "error": "Each CSV row must contain named values."})
            continue
        row_payload = {**row, "sensitivity": sensitivity}
        for field in NUMERIC_LIMITS:
            value = row_payload.get(field)
            if isinstance(value, str) and value.strip():
                try:
                    row_payload[field] = float(value)
                except ValueError:
                    pass
        try:
            results.append(
                {
                    "row": index,
                    "site_id": row.get("site_id", ""),
                    "year": row.get("year", ""),
                    "state_name": row.get("state_name", ""),
                    **_predict_one(row_payload, model),
                }
            )
        except InputError as error:
            results.append(
                {
                    "row": index,
                    "site_id": row.get("site_id", ""),
                    "year": row.get("year", ""),
                    "state_name": row.get("state_name", ""),
                    "error": str(error),
                }
            )
    return results


def parse_batch_csv(content: str) -> list[dict[str, str]]:
    """Parse and validate the required headers of an uploaded site CSV."""
    try:
        reader = csv.DictReader(io.StringIO(content))
        headers = reader.fieldnames
        if not headers:
            raise InputError("The uploaded CSV needs a header row.")
        normalized = [header.strip() for header in headers if header]
        if len(normalized) != len(set(normalized)):
            raise InputError("The uploaded CSV has duplicate column names.")
        missing = sorted(BATCH_REQUIRED_COLUMNS - set(normalized))
        if missing:
            raise InputError(
                "CSV is missing required columns: " + ", ".join(missing)
            )
        rows = [
            {str(key).strip(): (value or "").strip() for key, value in row.items() if key}
            for row in reader
            if row and any(value and value.strip() for value in row.values())
        ]
    except csv.Error as error:
        raise InputError(f"Could not read the uploaded CSV: {error}") from error
    if not rows:
        raise InputError("The uploaded CSV contains no data rows.")
    if len(rows) > MAX_BATCH_ROWS:
        raise InputError(f"Batch size cannot exceed {MAX_BATCH_ROWS} rows.")
    return rows


class PredictionHandler(SimpleHTTPRequestHandler):
    """Serve the frontend and handle its same-origin JSON API requests."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(FRONTEND_DIR), **kwargs)

    def _send_json(self, status: int, data: dict[str, Any]) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed_url = urlsplit(self.path)
        if parsed_url.path == "/api/health":
            self._send_json(200, {"status": "ok"})
            return
        if parsed_url.path == "/api/options":
            parameters = parse_qs(parsed_url.query)
            filters = {
                field: parameters[field][0]
                for field in (
                    "state_name",
                    "county_name",
                    "city_name",
                    "cbsa_name",
                    "site_id",
                )
                if parameters.get(field)
            }
            try:
                self._send_json(200, _site_options(filters))
            except (FileNotFoundError, OSError, ValueError) as error:
                self.log_error("Could not load site dropdown options: %s", error)
                self._send_json(503, {"error": "Monitoring site options are unavailable."})
            return
        if parsed_url.path == "/api/site-record":
            parameters = parse_qs(parsed_url.query)
            site_id = parameters.get("site_id", [""])[0]
            year = parameters.get("year", [""])[0]
            if not site_id or not year or not year.isdigit():
                self._send_json(400, {"error": "Choose a monitoring site and reading year."})
                return
            try:
                record = _site_record(site_id, year)
            except (FileNotFoundError, OSError, ValueError) as error:
                self.log_error("Could not load selected site record: %s", error)
                self._send_json(503, {"error": "Monitoring site readings are unavailable."})
                return
            if record is None:
                self._send_json(404, {"error": "No readings were found for that site and year."})
            else:
                self._send_json(200, record)
            return
        super().do_GET()

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        if path not in {"/api/predict", "/api/predict/batch"}:
            self._send_json(404, {"error": "Endpoint not found."})
            return
        if path == "/api/predict/batch":
            content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
            if content_type not in {"text/csv", "application/json"}:
                self._send_json(415, {"error": "Upload a CSV file or send a JSON batch."})
                return
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._send_json(400, {"error": "Invalid request size."})
                return
            if content_length <= 0 or content_length > 2_000_000:
                self._send_json(413, {"error": "Batch request must be between 1 byte and 2 MB."})
                return
            try:
                raw_body = self.rfile.read(content_length)
                if content_type == "text/csv":
                    rows = parse_batch_csv(raw_body.decode("utf-8-sig"))
                    sensitivity = self.headers.get("X-Sensitivity", "sensitive")
                else:
                    data = json.loads(raw_body)
                    if not isinstance(data, dict):
                        raise InputError("Batch request must be a JSON object.")
                    rows = data.get("rows")
                    sensitivity = data.get("sensitivity", "sensitive")
                results = predict_batch(rows, sensitivity)
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._send_json(400, {"error": "The batch request is not valid UTF-8 CSV or JSON."})
                return
            except InputError as error:
                self._send_json(400, {"error": str(error)})
                return
            except (FileNotFoundError, OSError, TypeError, ValueError, RuntimeError) as error:
                self.log_error("Batch prediction failed: %s", error)
                self._send_json(500, {"error": "Batch prediction failed. Check that the trained model is available."})
                return
            self._send_json(200, {"results": results, "count": len(results)})
            return
        if not self.headers.get("Content-Type", "").startswith("application/json"):
            self._send_json(415, {"error": "Send the form data as JSON."})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(400, {"error": "Invalid request size."})
            return
        if content_length <= 0 or content_length > MAX_REQUEST_BYTES:
            self._send_json(413, {"error": "Request must be between 1 byte and 16 KB."})
            return
        try:
            payload = json.loads(self.rfile.read(content_length))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(400, {"error": "Request body must be valid JSON."})
            return
        if not isinstance(payload, dict):
            self._send_json(400, {"error": "Request body must be a JSON object."})
            return
        try:
            result = predict(payload)
        except InputError as error:
            self._send_json(400, {"error": str(error)})
            return
        except (FileNotFoundError, OSError, TypeError, ValueError, RuntimeError) as error:
            self.log_error("Prediction failed: %s", error)
            self._send_json(500, {"error": "Prediction failed. Check that the trained model is available."})
            return
        self._send_json(200, result)


def main() -> None:
    global _MODEL
    _MODEL = load_model()
    server = ThreadingHTTPServer((HOST, PORT), PredictionHandler)
    print(f"PM2.5 risk predictor running at http://{HOST}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping prediction server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
