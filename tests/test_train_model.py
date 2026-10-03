import pandas as pd

from src.models import train_model


def test_calculate_metrics_returns_requested_classification_scores():
    y_true = pd.Series(
        ["Good", "Good", "Moderate", "Unhealthy", "Very Unhealthy"]
    )
    y_pred = pd.Series(
        ["Good", "Moderate", "Moderate", "Unhealthy", "Unhealthy"]
    )

    metrics = train_model.calculate_metrics(
        model_name="Multinomial Logistic Regression",
        dataset_name="test",
        y_true=y_true,
        y_pred=y_pred,
    )

    assert metrics["accuracy"] == 0.6
    assert metrics["macro_precision"] == 0.4
    assert metrics["macro_recall"] == 0.5
    assert metrics["macro_f1"] == 0.4
    assert metrics["weighted_f1"] == 8 / 15


def test_save_confusion_matrix_writes_csv_and_plot(tmp_path, monkeypatch):
    monkeypatch.setattr(train_model, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(train_model, "FIGURES_DIR", tmp_path)
    y_true = pd.Series(["Good", "Moderate"])
    y_pred = pd.Series(["Good", "Unhealthy"])

    train_model.save_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        model_name="Multinomial Logistic Regression",
    )

    matrix = pd.read_csv(
        tmp_path / "confusion_matrix.csv",
        index_col=0,
    )
    assert matrix.loc["Good", "Good"] == 1
    assert matrix.loc["Moderate", "Unhealthy"] == 1
    assert (tmp_path / "baseline_confusion_matrix.png").exists()


def test_save_classification_report_writes_per_class_metrics(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(train_model, "REPORTS_DIR", tmp_path)

    train_model.save_classification_report(
        y_true=pd.Series(["Good", "Moderate"]),
        y_pred=pd.Series(["Good", "Unhealthy"]),
        model_name="Multinomial Logistic Regression",
        dataset_name="test",
    )

    report = (tmp_path / "classification_report.txt").read_text(
        encoding="utf-8"
    )
    assert "precision" in report
    assert "recall" in report
    assert "f1-score" in report
    assert "support" in report
    assert "Good" in report
