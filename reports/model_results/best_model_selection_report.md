# Best Model Selection Report

## Decision

**Selected model: SVM (RBF)**

- Parameters: `{'C': 5.0, 'gamma': 'scale', 'class_weight': 'balanced'}`
- Selection rule: validation macro_f1; ties broken by qwk, balanced_accuracy, accuracy.
- Validation Macro F1: 0.4364 | QWK: 0.4850 | Balanced accuracy: 0.4300 | Accuracy: 0.7549
- Lead over runner-up (Random Forest): +0.0273 macro_f1

## Validation comparison (all tuned models)

| rank | model | macro_f1 | qwk | balanced_accuracy | accuracy | weighted_f1 | macro_precision | macro_recall | fit_seconds | converged |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | SVM (RBF) | 0.4364 | 0.4850 | 0.4300 | 0.7549 | 0.7198 | 0.5032 | 0.4300 | 32.5288 | True |
| 2 | Random Forest | 0.4091 | 0.5165 | 0.4172 | 0.7650 | 0.7229 | 0.6158 | 0.4172 | 6.6846 | True |
| 3 | Gradient Boosting | 0.3884 | 0.4827 | 0.3774 | 0.7650 | 0.7259 | 0.4516 | 0.3774 | 100.6766 | True |
| 4 | LightGBM | 0.3825 | 0.4615 | 0.3404 | 0.7656 | 0.7203 | 0.5769 | 0.3404 | 4.4692 | True |
| 5 | Logistic Regression | 0.3485 | 0.4340 | 0.4629 | 0.7213 | 0.6851 | 0.4008 | 0.4629 | 785.3510 | False |

## Test performance of the selected model

| model | accuracy | balanced_accuracy | macro_precision | macro_recall | macro_f1 | weighted_f1 | qwk |
|---|---|---|---|---|---|---|---|
| SVM (RBF) | 0.7924 | 0.2968 | 0.3593 | 0.2968 | 0.2945 | 0.7786 | 0.4014 |

The test set was used once, after the model was chosen on validation results.

## Method notes

- All models used the same chronological train/validation/test split.
- The preprocessor was fitted on training data only.
- Models were rebuilt from the tuned hyperparameters with random_state=42.
- Test labels were not used for tuning or model selection.