# ML Monitoring Model

## 1. Overview

The ML Monitoring component of IntelliCoreAI is responsible for detecting changes in the behavior and input data of machine-learning models.

For the current implementation, the monitored ML workload is the XGBoost credit-risk model trained using the Give Me Some Credit dataset.

The monitoring pipeline currently includes:

- Establishing a reference baseline
- Monitoring prediction telemetry
- Detecting feature-level data drift
- Detecting prediction drift
- Monitoring model performance degradation
- Validating incoming telemetry through data-quality checks
- Generating unified alerts from monitoring reports

The current implementation produces structured monitoring reports that can be consumed by the Diagnosis Engine.

Additional work, including live production telemetry validation, continuous operation, and further diagnostic integration, remains outside the currently verified scope.

---

## 2. Current ML Monitoring Architecture

The current monitoring flow is:

```text
                    TRAINING DATA
                         |
                         v
                   Preprocessed Data
                         |
                         v
                  Baseline Generation
                         |
              +----------+-----------+
              |                      |
              v                      v
         baseline.json       Trained XGBoost Model
                                     |
                                     v
                                  Prediction
                                     |
                                     v
                                   Telemetry
                                     |
              +----------------------+----------------------+
              |                      |                      |
              v                      v                      v
          Data Drift           Prediction Drift         Performance
          Monitoring            Monitoring              Monitoring
              |                      |                      |
              v                      v                      v
        Drift Report          Prediction Drift          Performance
                                  Report                  Report

                          Data Quality Monitor
                                   |
                                   v
                           Data Quality Report

              +----------------------+----------------------+
                                     |
                                     v
                            Unified Alert Engine
                                     |
                                     v
                            unified_alert.json
```

The monitoring system compares current telemetry and model behavior against the appropriate reference baselines. Each monitoring component generates an individual report, and the Unified Alert Engine consolidates those reports into a single alert.

---

## 33. Prediction Drift Monitoring

Prediction drift monitors whether the distribution of model outputs changes over time.

Data drift focuses on changes in model inputs, while prediction drift focuses on changes in the model's outputs.

For IntelliCoreAI, prediction drift is measured using the model's predicted probability for the positive class.

The monitored prediction characteristics include:

- Mean prediction probability
- Median prediction probability
- Standard deviation
- Minimum and maximum probability
- Prediction probability quantiles
- Predicted class distribution
- Positive prediction rate

---

## 34. Prediction Drift Reference

A separate prediction baseline is generated using the trained XGBoost model on the processed test dataset.

Flow:

```text
Test Dataset
    |
    v
Trained XGBoost Model
    |
    v
Reference Prediction Probabilities
    |
    v
Prediction Baseline
```

The prediction baseline is stored at:

`data/artifacts/monitoring/prediction_baseline.json`

This is kept separate from `baseline.json`, which contains input-feature statistics.

---

## 35. Prediction Drift Detection

Current prediction probabilities are obtained from:

`data/telemetry/prediction_events.jsonl`

The reference and current prediction-probability distributions are compared using the two-sample Kolmogorov-Smirnov (KS) test.

The significance level is:

`alpha = 0.05`

Decision rule:

- p-value less than 0.05: prediction drift detected
- p-value greater than or equal to 0.05: no statistically significant prediction drift

---

## 36. Normal Prediction Drift Validation

The normal telemetry window contained:

`1,000 prediction events`

Reference prediction statistics:

- Mean probability: `0.068733`
- Median probability: `0.021513`
- Positive prediction rate: `2.3060%`

Current telemetry:

- Mean probability: `0.066360`
- Positive prediction rate: `2.2000%`

KS test:

- KS statistic: `0.026926`
- p-value: `0.475953`

Since:

`0.475953 > 0.05`

the normal telemetry window was classified as:

`NO SIGNIFICANT PREDICTION DRIFT`

---

## 37. Controlled Prediction Drift Test

A separate controlled telemetry file was created to validate whether the detector can identify artificial prediction drift.

The original telemetry file was not modified.

Controlled test file:

`data/telemetry/prediction_drift_test_events.jsonl`

The prediction probability for each event was deliberately shifted upward by:

`+0.25`

with probabilities capped at `0.99`.

Predicted classes were recalculated using the `0.50` classification threshold.

---

## 38. Controlled Prediction Drift Result

The controlled drift experiment produced:

Reference mean probability:

`0.068733`

Current mean probability:

`0.316143`

Reference positive prediction rate:

`2.3060%`

Current positive prediction rate:

`7.0000%`

KS statistic:

`0.927206`

p-value:

`0.000000`

Since the p-value was below `0.05`, the detector correctly produced:

`PREDICTION DRIFT DETECTED`

This validates the prediction-drift detection implementation for the controlled experiment.

---

## 39. Model Performance Monitoring

Prediction drift does not necessarily mean that the model's predictive performance has deteriorated.

Therefore, IntelliCoreAI separately monitors actual model performance using ground-truth labels.

The performance monitor evaluates:

- Accuracy
- Precision
- F1 Score
- Recall
- ROC-AUC
- PR-AUC
- Log Loss
- Brier Score

---

## 40. Performance Reference Evaluation

The trained XGBoost model was evaluated against the processed test dataset.

Test dataset:

- Rows: `29,879`
- Negative samples: `27,877`
- Positive samples: `2,002`

The baseline performance was:

| Metric | Value |
|---|---:|
| Accuracy | 0.937046 |
| Precision | 0.587808 |
| Recall | 0.202298 |
| F1 Score | 0.301003 |
| ROC-AUC | 0.864874 |
| PR-AUC | 0.403800 |
| Log Loss | 0.177291 |
| Brier Score | 0.048824 |

Classification threshold:

`0.50`

---

## 41. Performance Confusion Matrix

The baseline confusion matrix was:

|  | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 27,593 | 284 |
| Actual 1 | 1,597 | 405 |

The corresponding interpretation is:

- True Negatives: `27,593`
- False Positives: `284`
- False Negatives: `1,597`
- True Positives: `405`

The confusion matrix is stored at:

`data/artifacts/monitoring/performance_confusion_matrix.png`

---

## 42. Why Multiple Performance Metrics Are Used

The Give Me Some Credit dataset is highly imbalanced, with approximately 93.3% negative and 6.7% positive observations.

Therefore, accuracy alone is not sufficient to evaluate model performance.

For IntelliCoreAI, additional metrics are monitored to provide a broader view of model behavior.

PR-AUC is particularly useful for evaluating performance under class imbalance.

Probability-based metrics such as Log Loss and Brier Score are also monitored because the model produces prediction probabilities rather than only binary classes.

---

## 43. Controlled Performance Degradation

A controlled degradation experiment was created to validate performance-degradation monitoring.

The experiment intentionally degraded 30% of model predictions.

The original dataset and trained model were not modified.

The controlled experiment produced the following results:

| Metric | Baseline | Degraded | Change |
|---|---:|---:|---:|
| Accuracy | 0.937046 | 0.655444 | -30.05% |
| Precision | 0.587808 | 0.032578 | -94.46% |
| Recall | 0.202298 | 0.144356 | -28.64% |
| F1 Score | 0.301003 | 0.053159 | -82.34% |
| ROC-AUC | 0.864874 | 0.632136 | -26.91% |
| PR-AUC | 0.403800 | 0.116777 | -71.08% |
| Log Loss | 0.177291 | 0.494050 | +178.67% |
| Brier Score | 0.048824 | 0.184450 | +277.79% |

The controlled degradation demonstrates that substantial deterioration in model behavior can be observed across multiple metrics.

---

## 44. Performance Degradation Detection

The performance degradation detector compares current model metrics against the established performance baseline.

A relative-change threshold of:

`20%`

is currently used.

For metrics where higher values indicate better performance:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- PR-AUC

a decrease of at least 20% is considered degradation.

For metrics where lower values indicate better performance:

- Log Loss
- Brier Score

an increase of at least 20% is considered degradation.

---

## 45. Controlled Performance Degradation Detection Result

The controlled experiment produced degradation in all eight monitored metrics.

Detected degraded metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- PR-AUC
- Log Loss
- Brier Score

The detector therefore produced:

`PERFORMANCE DEGRADATION DETECTED`

The generated report is stored at:

`data/artifacts/monitoring/performance_degradation_report.json`

---

## 46. Current ML Monitoring Architecture

The current IntelliCoreAI ML monitoring pipeline consists of four monitoring dimensions and a unified alert-generation stage.

The pipeline is:

Model → Prediction → Telemetry → Monitoring → Unified Alert Engine

### 46.1 Data Drift

Checks whether input feature distributions have changed relative to the reference baseline.

**Method:** Two-sample KS test.

**Output:** `drift_report.json`

### 46.2 Prediction Drift

Checks whether the distribution of model prediction probabilities has changed relative to the prediction baseline.

**Method:** Two-sample KS test.

**Output:** `prediction_drift_report.json`

### 46.3 Model Performance

Checks whether predictive performance has deteriorated relative to the established baseline.

**Method:** Relative metric comparison using the configured degradation threshold.

**Output:** `performance_degradation_report.json`

### 46.4 Data Quality

Validates prediction telemetry for:

- Telemetry structure
- Schema consistency
- Numeric validity
- Domain-rule violations
- Missing values
- Duplicate events
- Data freshness

**Output:** `data_quality_report.json`

### 46.5 Unified Alert Engine

Combines the four monitoring reports into a single structured alert with an overall status and severity.

**Output:** `unified_alert.json`

---

## 47. Current Monitoring Flow

```text
                         XGBoost Model
                               |
                               v
                           Prediction
                               |
                               v
                            Telemetry
                               |
          +--------------------+--------------------+
          |                    |                    |
          v                    v                    v
      Data Drift         Prediction Drift       Performance
      Monitoring          Monitoring            Monitoring
          |                    |                    |
          v                    v                    v
     Drift Report       Prediction Drift       Performance
                            Report               Report

                     Data Quality Monitor
                               |
                               v
                      Data Quality Report

          +--------------------+--------------------+
                               |
                               v
                       Unified Alert Engine
                               |
                               v
                      unified_alert.json
```

The Unified Alert Engine reads:

- `drift_report.json`
- `prediction_drift_report.json`
- `performance_degradation_report.json`
- `data_quality_report.json`

It consolidates the monitoring results into `unified_alert.json`, which can be consumed by the Diagnosis Engine for subsequent investigation.

---

## 48. Unified Alert Engine

The Unified Alert Engine combines the outputs of the individual ML monitoring components into a single structured alert. It evaluates monitoring reports, identifies conditions that require attention, and assigns an overall alert severity.

The current severity rules are:

- **HIGH:** Performance degradation is detected, or at least two alert conditions are detected.
- **MEDIUM:** Exactly one alert condition is detected.
- **INFO:** No alert-triggering conditions are detected.

Performance degradation always results in HIGH severity under the current implementation.

A failed data-quality check triggers an alert. A data-quality warning alone does not trigger an alert under the current implementation.

These rules represent the project's initial alert policy and are not a production risk policy.

### 48.1 Overall Alert Status

The Alert Engine produces one of two overall statuses:

| Status | Meaning |
|---|---|
| `alert` | At least one alert-triggering condition has been detected. |
| `no_alert` | No alert-triggering conditions have been detected. |

### 48.2 Unified Alert Output

The unified alert is stored at:

`data/artifacts/monitoring/unified_alert.json`

The report contains:

- Alert ID and generation timestamp
- Model name and version
- Overall alert status and severity
- Individual monitoring results
- A summary of detected alert conditions

---

## 49. Current Verification Results

The current implementation has passed the following verification checks:

- All 44 automated tests passed.
- The Alert Engine successfully loaded all four monitoring reports.
- The generated `unified_alert.json` was parsed successfully as valid JSON.
- The output contained four monitoring categories.
- Severity aggregation produced the expected result for the available monitoring reports.

The most recent verification produced:

| Result | Observed value |
|---|---|
| Overall status | `alert` |
| Severity | `HIGH` |
| Alert conditions | 3 |
| Data drift | Not detected |
| Prediction drift | Detected |
| Performance degradation | Detected |
| Data quality | Failed |

These values describe the current monitoring artifacts. They do not establish that the same conditions exist in a live production environment.

---

## 50. Current Data Limitation

The current telemetry file contains 1,000 historical prediction events. Its latest event timestamp is:

`2026-09-05T16:07:50.147882+00:00`

The configured maximum event age is 15 minutes.

When the data-quality monitor ran on October 13, 2026, the freshness check failed because the latest event was older than the permitted age.

This failure is expected for the current historical dataset. The freshness threshold has not been weakened, and event timestamps have not been modified to make the check pass.

The implementation validates the available telemetry and generates monitoring reports and alerts. Validation against continuously arriving production telemetry remains a separate step.

---

## 51. Current Scope and Next Steps

The current ML Monitoring implementation includes:

1. Input data drift detection
2. Prediction drift detection
3. Model performance degradation detection
4. Prediction telemetry data-quality validation
5. Unified alert generation
6. Automated tests for the implemented monitoring components

The current implementation is functionally verified against the available datasets and automated tests.

Remaining work should be tracked separately, including:

- Validation against continuously arriving telemetry
- Production deployment and operational validation
- Integration testing with the Diagnosis Engine
- Any additional monitoring components required by the wider IntelliCoreAI platform