import csv
import io
import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from app.backend.server import (
    InputError,
    PredictionHandler,
    BATCH_REQUIRED_COLUMNS,
    MAX_BATCH_ROWS,
    _activity_assessment,
    _site_options,
    _site_record,
    build_features,
    load_model,
    parse_batch_csv,
    predict,
    predict_batch,
)


@pytest.fixture
def site_reading():
    return {
        "site_id": "55-059-0019",
        "state_name": "California",
        "county_name": "Los Angeles",
        "city_name": "Pasadena",
        "cbsa_name": "Los Angeles-Long Beach-Anaheim CA",
        "year": 2016,
        "latitude": 34.1478,
        "longitude": -118.1445,
        "observation_count": 320,
        "observation_percent": 88,
        "valid_day_count": 320,
        "arithmetic_standard_dev": 5.4,
        "first_max_value": 42,
        "ninety_nine_percentile": 36,
        "ninety_eight_percentile": 30,
        "ninety_five_percentile": 25,
        "ninety_percentile": 20,
        "seventy_five_percentile": 15,
        "fifty_percentile": 9,
        "ten_percentile": 3,
        "pm25_mean": 8.2,
    }


def test_build_features_derives_trends_and_missing_history(site_reading):
    row = build_features(site_reading).iloc[0]

    assert row["pm25_lag1"] == 0
    assert row["pm25_change_1yr"] == 0
    assert row["has_previous_year"] == 0
    assert row["has_three_year_history"] == 0
    assert row["p95_minus_median"] == 16
    assert row["p99_p10_range"] == 33


def test_build_features_uses_available_site_history(site_reading):
    site_reading["previous_year_pm25_mean"] = 7
    site_reading["two_years_ago_pm25_mean"] = 9

    row = build_features(site_reading).iloc[0]

    assert row["pm25_change_1yr"] == pytest.approx(1.2)
    assert row["pm25_pct_change_1yr"] == pytest.approx(1.2 / 7)
    assert row["pm25_rolling_2yr_mean"] == pytest.approx(7.6)
    assert row["pm25_rolling_3yr_mean"] == pytest.approx(8.0666666667)
    assert row["has_previous_year"] == 1
    assert row["has_three_year_history"] == 1


def test_build_features_rejects_two_year_history_with_a_gap(site_reading):
    site_reading["two_years_ago_pm25_mean"] = 9

    with pytest.raises(InputError, match="previous-year mean"):
        build_features(site_reading)


@pytest.mark.parametrize(
    "field",
    [
        "state_name",
        "county_name",
        "city_name",
        "cbsa_name",
        "site_id",
        "year",
        "pm25_mean",
    ],
)
def test_build_features_reports_missing_required_fields(site_reading, field):
    del site_reading[field]

    with pytest.raises(InputError):
        build_features(site_reading)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("year", 2016.5),
        ("latitude", 91),
        ("longitude", -181),
        ("observation_percent", 101),
        ("pm25_mean", -1),
        ("observation_count", 0),
    ],
)
def test_build_features_rejects_out_of_range_values(site_reading, field, value):
    site_reading[field] = value

    with pytest.raises(InputError):
        build_features(site_reading)


def test_build_features_rejects_inconsistent_percentile_order(site_reading):
    site_reading["fifty_percentile"] = 28

    with pytest.raises(InputError, match="percentile readings"):
        build_features(site_reading)


def test_predict_uses_fitted_model_and_returns_known_category(site_reading):
    result = predict(site_reading)

    assert result["category"] in {
        "Good",
        "Moderate",
        "Unhealthy for Sensitive Groups",
        "Unhealthy",
        "Very Unhealthy",
    }
    assert result["range"]
    assert result["description"]


def test_predict_returns_confidence_activity_guidance_and_evidence(site_reading):
    result = predict(site_reading)

    assert 0 <= result["confidence"] <= 100
    assert result["sensitivity_label"] == "Sensitive groups"
    assert isinstance(result["activity_alert"], bool)
    assert result["activity_guidance"]
    assert result["explanation"]
    assert any("highest daily reading" in factor for factor in result["explanation"])


def test_predict_works_when_saved_model_has_no_probability_method(site_reading):
    fitted_model = load_model()

    class PredictOnlyModel:
        feature_names_in_ = fitted_model.feature_names_in_
        classes_ = fitted_model.classes_

        def predict(self, features):
            return fitted_model.predict(features)

    result = predict(site_reading, model=PredictOnlyModel())

    assert result["category"] in fitted_model.classes_
    assert result["confidence"] is None
    assert result["confidence_label"] == "Not available from this saved model"


def test_sensitivity_changes_alert_threshold_not_prediction_category():
    general = _activity_assessment("Moderate", "general")
    sensitive = _activity_assessment("Moderate", "sensitive")
    precautionary = _activity_assessment("Moderate", "high")

    assert general["activity_alert"] is False
    assert sensitive["activity_alert"] is False
    assert precautionary["activity_alert"] is True


def test_predict_batch_keeps_row_level_validation_errors(site_reading):
    invalid_reading = {**site_reading, "pm25_mean": -1}

    results = predict_batch([site_reading, invalid_reading], "sensitive")

    assert len(results) == 2
    assert "category" in results[0]
    assert results[1]["row"] == 2
    assert "error" in results[1]


def test_predict_batch_enforces_row_limit():
    with pytest.raises(InputError, match="cannot exceed"):
        predict_batch([{}] * (MAX_BATCH_ROWS + 1), "sensitive")


def test_parse_batch_csv_rejects_missing_columns():
    with pytest.raises(InputError, match="missing required columns"):
        parse_batch_csv("site_id,year\nsite-1,2016\n")


def test_parse_batch_csv_rejects_duplicate_normalized_headers():
    with pytest.raises(InputError, match="duplicate column names"):
        parse_batch_csv("site_id, site_id\nsite-1,site-2\n")


def test_parse_batch_csv_reads_rows_with_quoted_commas(site_reading):
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=sorted(BATCH_REQUIRED_COLUMNS))
    writer.writeheader()
    writer.writerow(site_reading)

    rows = parse_batch_csv(output.getvalue())

    assert len(rows) == 1
    assert rows[0]["city_name"] == "Pasadena"
    assert rows[0]["site_id"] == site_reading["site_id"]


def test_prediction_features_match_saved_model_schema(site_reading):
    model = load_model()

    assert set(build_features(site_reading).columns) == set(model.feature_names_in_)


@pytest.fixture
def api_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), PredictionHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def test_prediction_endpoint_returns_model_result(site_reading, api_server):
    request = Request(
        f"{api_server}/api/predict",
        data=json.dumps(site_reading).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urlopen(request) as response:
        result = json.load(response)

    assert response.status == 200
    assert result["category"] in load_model().classes_
    assert result["range"]
    assert "confidence" in result
    assert "activity_guidance" in result


def test_frontend_has_four_sections_and_interactive_diagrams():
    frontend = Path(__file__).resolve().parents[1] / "app" / "frontend" / "index.html"
    markup = frontend.read_text(encoding="utf-8")

    headings = [
        "Location Information",
        "Air Quality Measurements",
        "Monitoring Information",
        "Prediction Result",
    ]
    positions = [markup.index(heading) for heading in headings]

    assert positions == sorted(positions)
    assert 'id="measurement-chart"' in markup
    assert 'id="risk-band"' in markup
    assert "Calibrated probability" not in markup


def test_batch_prediction_endpoint_returns_site_results(site_reading, api_server):
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=sorted(BATCH_REQUIRED_COLUMNS))
    writer.writeheader()
    writer.writerow(site_reading)
    request = Request(
        f"{api_server}/api/predict/batch",
        data=output.getvalue().encode(),
        headers={"Content-Type": "text/csv", "X-Sensitivity": "general"},
        method="POST",
    )

    with urlopen(request) as response:
        result = json.load(response)

    assert response.status == 200
    assert result["count"] == 1
    assert result["results"][0]["site_id"] == site_reading["site_id"]
    assert result["results"][0]["sensitivity"] == "general"


def test_batch_prediction_endpoint_rejects_missing_csv_columns(api_server):
    request = Request(
        f"{api_server}/api/predict/batch",
        data=b"site_id,year\nsite-1,2016\n",
        headers={"Content-Type": "text/csv"},
        method="POST",
    )

    with pytest.raises(HTTPError) as error:
        urlopen(request)

    assert error.value.code == 400
    assert "missing required columns" in json.load(error.value)["error"]


def test_prediction_endpoint_rejects_missing_required_input(api_server):
    request = Request(
        f"{api_server}/api/predict",
        data=b'{"state_name":"California"}',
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with pytest.raises(HTTPError) as error:
        urlopen(request)

    assert error.value.code == 400
    assert "county name" in json.load(error.value)["error"]


def test_prediction_endpoint_rejects_malformed_json(api_server):
    request = Request(
        f"{api_server}/api/predict",
        data=b'{"state_name":',
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with pytest.raises(HTTPError) as error:
        urlopen(request)

    assert error.value.code == 400
    assert json.load(error.value)["error"] == "Request body must be valid JSON."


def test_site_options_follow_selected_location():
    filters = {
        "state_name": "California",
        "county_name": "Los Angeles",
        "city_name": "Pasadena",
        "cbsa_name": "Los Angeles-Long Beach-Anaheim CA",
    }
    options = _site_options(filters)

    assert options["state_name"]
    assert "Pasadena" in options["city_name"]
    assert options["site_id"]
    filters["site_id"] = options["site_id"][0]
    assert _site_options(filters)["year"]


def test_site_record_returns_measurements_and_history():
    record = _site_record("55-059-0019", "1997")

    assert record is not None
    assert record["site_id"] == "55-059-0019"
    assert record["year"] == 1997
    assert record["pm25_mean"] is not None
    assert "latitude" in record
    assert "previous_year_pm25_mean" in record


def test_site_options_endpoint_returns_dataset_values(api_server):
    with urlopen(f"{api_server}/api/options") as response:
        options = json.load(response)

    assert response.status == 200
    assert "California" in options["state_name"]


def test_site_record_endpoint_returns_selected_reading(api_server):
    with urlopen(
        f"{api_server}/api/site-record?site_id=55-059-0019&year=1997"
    ) as response:
        record = json.load(response)

    assert response.status == 200
    assert record["site_id"] == "55-059-0019"
    assert record["year"] == 1997
