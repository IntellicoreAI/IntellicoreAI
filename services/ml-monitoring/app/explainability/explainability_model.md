# Model Explainability — ML Monitoring

## 1. Purpose

The explainability module helps interpret the predictions of the credit-risk XGBoost classifier. It provides global feature-importance analysis using SHAP and individual prediction explanations using LIME.

These explanations support model transparency and investigation. They do not establish causation or guarantee that a prediction is correct.

## 2. SHAP — Global Explainability

SHAP (SHapley Additive exPlanations) estimates how features contribute to model outputs.

The SHAP implementation:
- Loads the trained XGBoost model and feature names.
- Loads the processed test dataset.
- Selects a reproducible sample of up to 5,000 records.
- Calculates SHAP values for the selected records.
- Aggregates feature importance using mean absolute SHAP values.
- Saves numerical results and visualization artifacts.

### SHAP artifacts

Output directory: `data/artifacts/xgboost/shap/`

| Artifact | Purpose |
|---|---|
| `feature_importance.json` | Aggregated feature-importance results |
| `sample_shap_values.csv` | Per-record SHAP values for the selected sample |
| `summary_bar.png` | Global feature-importance visualization |
| `summary_beeswarm.png` | Distribution of SHAP contributions across records |
| `shap_explainability_result.md` | Generated explainability summary |

### Interpretation

A higher mean absolute SHAP value indicates a greater average contribution to the model output magnitude within the evaluated sample. The sign of an individual SHAP value indicates the direction of contribution relative to the model's SHAP baseline, on the output scale used by the explainer.

Feature importance should not be interpreted as proof that a feature causes credit delinquency.

## 3. LIME — Local Explainability

LIME (Local Interpretable Model-agnostic Explanations) approximates a model's behavior around an individual input using an interpretable local surrogate model.

The LIME implementation:
- Loads the trained XGBoost classifier and feature names.
- Uses processed test data as the reference data for perturbations.
- Validates that an input contains the expected numeric, finite features.
- Generates a local explanation for the serious-delinquency class.
- Records the predicted class, class probabilities, local contributions, surrogate score, and interpretation note.
- Saves the result as JSON.

### LIME artifact

Output directory: `data/artifacts/xgboost/lime/`

| Artifact | Purpose |
|---|---|
| `sample_explanation.json` | Example local explanation for one test record |

### Interpretation

LIME contributions describe the local surrogate's approximation of the classifier near the selected record. They are not causal effects and may vary with sampling settings and the local neighborhood.

The surrogate score indicates local approximation quality under LIME's scoring method. It is not the XGBoost model's accuracy or ROC-AUC.

## 4. Input Validation

The LIME implementation checks that:
- All expected features are present.
- Unexpected features are rejected.
- Values are numeric and finite.
- Boolean values are not accepted as numeric features.
- The reference dataset is non-empty, finite, and has columns in the expected order.

The SHAP implementation validates feature alignment and produces numerical explanations for the selected records.

## 5. Reproducibility and Validation

Random seeds are configured for reproducible sampling and explanation generation where supported.

Unit tests cover:
- SHAP sample reproducibility and output shapes.
- Feature-importance ranking and feature alignment.
- SHAP output with three-dimensional values.
- LIME input validation.
- LIME explanation structure and probability values.
- LIME behavior with empty reference data.
- JSON artifact saving.

A numerical SHAP additivity check was also performed on 20 records. The maximum absolute difference between the reconstructed raw model output and the model's raw score was approximately `2.38 × 10⁻⁶`.

This numerical check validates the tested records, not every possible input or the complete production pipeline.

## 6. Running the Explainability Modules

Run commands from the `services/ml-monitoring` directory.

### SHAP

```powershell
python -m app.explainability.shap_explainer
```

### LIME

```powershell
python -m app.explainability.lime_explainer
```

### Explainability tests

```powershell
python -m pytest tests\test_explainability.py -v
```

## 7. Limitations

- SHAP and LIME explain model behavior; neither proves that a prediction is correct.
- Explanations are not causal analyses.
- Feature importance does not establish fairness or absence of bias.
- LIME is a local approximation and may not fully reproduce the underlying model.
- Explainability results should be interpreted alongside data quality, drift, performance, and fairness monitoring results.
- A successful unit-test run does not by itself establish production readiness.
