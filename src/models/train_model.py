"""
Baseline model training for the Next-Year PM2.5
Air-Quality Risk Classification project.

This script:
1. Loads the chronological train, validation and test datasets.
2. Applies deterministic preprocessing preparation.
3. Fits the preprocessing pipeline using training data only.
4. Trains Dummy Classifier and Logistic Regression baselines.
5. Compares models using validation data.
6. Evaluates the best baseline using the final test data.
7. Saves metrics, confusion matrix and model artifacts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)


# ------------------------------------------------------------
# Project paths
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Member 3 preprocessing functions
from src.preprocessing.preprocessing_pipeline import (  # noqa: E402
    fit_preprocessor,
    get_transformed_feature_names,
    prepare_base_dataframe,
    prepare_features_and_target,
    transform_features,
)


PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
BASELINE_MODEL_DIR = PROJECT_ROOT / "models" / "baseline"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

TRAIN_FILE = PROCESSED_DATA_DIR / "train_data.csv"
VALIDATION_FILE = PROCESSED_DATA_DIR / "validation_data.csv"
TEST_FILE = PROCESSED_DATA_DIR / "test_data.csv"

TARGET_COLUMN = "pm25_risk_category_next_year"

TARGET_CLASSES = [
    "Good",
    "Moderate",
    "Unhealthy for Sensitive Groups",
    "Unhealthy",
    "Very Unhealthy",
]


# ------------------------------------------------------------
# Load one dataset split
# ------------------------------------------------------------

def load_split(file_path: Path) -> pd.DataFrame:
    """
    Load a processed dataset split.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Required dataset was not found: {file_path}\n"
            "Run 'python src/data/make_dataset.py' first."
        )

    dataframe = pd.read_csv(file_path, low_memory=False)

    print(
        f"Loaded {file_path.name}: "
        f"{dataframe.shape[0]:,} rows x "
        f"{dataframe.shape[1]} columns"
    )

    return dataframe


# ------------------------------------------------------------
# Prepare one dataset split
# ------------------------------------------------------------

def prepare_split(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Apply deterministic cleaning and separate features and target.

    This function does not calculate medians, scaling statistics
    or encoding vocabularies.
    """

    prepared_dataframe = prepare_base_dataframe(dataframe)

    X, y = prepare_features_and_target(
        prepared_dataframe,
        target_column=TARGET_COLUMN,
    )

    return X, y


# ------------------------------------------------------------
# Calculate evaluation metrics
# ------------------------------------------------------------

def calculate_metrics(
    model_name: str,
    dataset_name: str,
    y_true: pd.Series,
    y_pred,
) -> dict:
    """
    Calculate classification metrics suitable for imbalanced data.
    """

    return {
        "model": model_name,
        "dataset": dataset_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            y_pred,
        ),
        "macro_precision": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "macro_recall": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
    }


# ------------------------------------------------------------
# Print evaluation metrics
# ------------------------------------------------------------

def print_metrics(metrics: dict) -> None:
    """
    Display model evaluation results.
    """

    print("\n" + "=" * 60)
    print(
        f"{metrics['model']} - "
        f"{metrics['dataset'].upper()} RESULTS"
    )
    print("=" * 60)

    print(f"Accuracy:          {metrics['accuracy']:.4f}")
    print(
        f"Balanced accuracy: "
        f"{metrics['balanced_accuracy']:.4f}"
    )
    print(
        f"Macro precision:   "
        f"{metrics['macro_precision']:.4f}"
    )
    print(f"Macro recall:      {metrics['macro_recall']:.4f}")
    print(f"Macro F1-score:    {metrics['macro_f1']:.4f}")


# ------------------------------------------------------------
# Save confusion matrix
# ------------------------------------------------------------

def save_confusion_matrix(
    y_true: pd.Series,
    y_pred,
    model_name: str,
) -> None:
    """
    Save the test-set confusion matrix for the best baseline.
    """

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=TARGET_CLASSES,
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=TARGET_CLASSES,
    )

    figure, axis = plt.subplots(figsize=(12, 9))

    display.plot(
        ax=axis,
        cmap="Blues",
        xticks_rotation=35,
        colorbar=False,
    )

    axis.set_title(
        f"{model_name} - Test Set Confusion Matrix"
    )

    figure.tight_layout()

    output_path = (
        FIGURES_DIR / "baseline_confusion_matrix.png"
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(f"Confusion matrix saved to: {output_path}")


# ------------------------------------------------------------
# Main baseline workflow
# ------------------------------------------------------------

def main() -> None:
    """
    Run the complete baseline-model preparation workflow.
    """

    print("\n" + "=" * 60)
    print("MEMBER 4 - BASELINE MODEL PREPARATION")
    print("=" * 60)

    # Create output folders if they do not already exist.
    BASELINE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load the chronological splits created by Member 4.
    train_dataframe = load_split(TRAIN_FILE)
    validation_dataframe = load_split(VALIDATION_FILE)
    test_dataframe = load_split(TEST_FILE)

    # Prepare each split independently.
    X_train, y_train = prepare_split(train_dataframe)
    X_validation, y_validation = prepare_split(
        validation_dataframe
    )
    X_test, y_test = prepare_split(test_dataframe)

    print("\nPrepared feature shapes:")
    print(f"Training:   {X_train.shape}")
    print(f"Validation: {X_validation.shape}")
    print(f"Testing:    {X_test.shape}")

    # Confirm the final five-class target.
    all_classes = set(y_train.unique())

    if "Hazardous" in all_classes:
        raise ValueError(
            "Hazardous class is still present in training data. "
            "Run the approved rare-class merging and splitting "
            "script again."
        )

    if all_classes != set(TARGET_CLASSES):
        raise ValueError(
            "Training target classes do not match the expected "
            f"five classes. Found: {sorted(all_classes)}"
        )

    # Fit preprocessing using TRAINING DATA ONLY.
    preprocessor = fit_preprocessor(X_train)

    X_train_processed = transform_features(
        preprocessor,
        X_train,
    )
    X_validation_processed = transform_features(
        preprocessor,
        X_validation,
    )
    X_test_processed = transform_features(
        preprocessor,
        X_test,
    )

    feature_names = get_transformed_feature_names(
        preprocessor
    )

    print("\nTransformed feature shapes:")
    print(f"Training:   {X_train_processed.shape}")
    print(f"Validation: {X_validation_processed.shape}")
    print(f"Testing:    {X_test_processed.shape}")
    print(
        f"Total transformed features: "
        f"{len(feature_names):,}"
    )

    # Define baseline models.
    models = {
        "Dummy Classifier": DummyClassifier(
            strategy="most_frequent",
        ),
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
        ),
    }

    validation_results = []
    trained_models = {}

    # Train and validate each baseline model.
    for model_name, model in models.items():
        print("\n" + "-" * 60)
        print(f"Training: {model_name}")
        print("-" * 60)

        model.fit(
            X_train_processed,
            y_train,
        )

        validation_predictions = model.predict(
            X_validation_processed
        )

        metrics = calculate_metrics(
            model_name=model_name,
            dataset_name="validation",
            y_true=y_validation,
            y_pred=validation_predictions,
        )

        validation_results.append(metrics)
        trained_models[model_name] = model

        print_metrics(metrics)

    # Select the best baseline using validation Macro F1.
    best_validation_result = max(
        validation_results,
        key=lambda result: result["macro_f1"],
    )

    best_model_name = best_validation_result["model"]
    best_model = trained_models[best_model_name]

    print("\n" + "=" * 60)
    print(f"BEST VALIDATION BASELINE: {best_model_name}")
    print(
        "Selection metric: Macro F1-score = "
        f"{best_validation_result['macro_f1']:.4f}"
    )
    print("=" * 60)

    # Evaluate the selected baseline once on the test set.
    test_predictions = best_model.predict(
        X_test_processed
    )

    test_result = calculate_metrics(
        model_name=best_model_name,
        dataset_name="test",
        y_true=y_test,
        y_pred=test_predictions,
    )

    print_metrics(test_result)

    # Save model comparison results.
    all_results = validation_results + [test_result]

    results_dataframe = pd.DataFrame(all_results)

    results_path = REPORTS_DIR / "model_comparison.csv"

    results_dataframe.to_csv(
        results_path,
        index=False,
    )

    print(f"\nModel results saved to: {results_path}")

    # Save the confusion matrix for the best baseline.
    save_confusion_matrix(
        y_true=y_test,
        y_pred=test_predictions,
        model_name=best_model_name,
    )

    # Save fitted preprocessing pipeline and trained models.
    preprocessor_path = (
        BASELINE_MODEL_DIR / "preprocessor.joblib"
    )

    joblib.dump(
        preprocessor,
        preprocessor_path,
    )

    print(f"Preprocessor saved to: {preprocessor_path}")

    for model_name, model in trained_models.items():
        safe_name = (
            model_name.lower()
            .replace(" ", "_")
        )

        model_path = (
            BASELINE_MODEL_DIR / f"{safe_name}.joblib"
        )

        joblib.dump(
            model,
            model_path,
        )

        print(f"Model saved to: {model_path}")

    # Save a combined bundle for the selected baseline.
    best_bundle_path = (
        BASELINE_MODEL_DIR / "best_baseline_bundle.joblib"
    )

    best_bundle = {
        "model_name": best_model_name,
        "preprocessor": preprocessor,
        "model": best_model,
        "target_classes": TARGET_CLASSES,
        "feature_names": feature_names,
        "selection_metric": "macro_f1",
    }

    joblib.dump(
        best_bundle,
        best_bundle_path,
    )

    print(f"Best baseline bundle saved to: {best_bundle_path}")

    print("\n" + "=" * 60)
    print("BASELINE MODEL PREPARATION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()