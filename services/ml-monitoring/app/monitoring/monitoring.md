# IntelliCore AI - Unified ML Monitoring
## 48. Unified Alert Engine

The Unified Alert Engine combines the outputs of the individual ML monitoring components into a single structured alert. It evaluates monitoring reports, identifies conditions that require attention, and assigns an overall alert severity.

The Alert Engine currently consumes reports for:

- Data drift
- Prediction drift
- Model performance degradation
- Data quality

The purpose of this component is to provide a consolidated monitoring result that downstream components, including the Diagnosis Engine, can consume without independently processing each monitoring report.

### 48.1 Monitoring Flow

The current ML Monitoring pipeline follows this flow:

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

The monitoring components operate independently and produce their own reports. The Unified Alert Engine combines the results of these reports into one structured output.

### 48.2 Alert Generation

The Alert Engine reads the following monitoring reports:

| Monitoring component | Input report |
|---|---|
| Data drift | `drift_report.json` |
| Prediction drift | `prediction_drift_report.json` |
| Performance degradation | `performance_degradation_report.json` |
| Data quality | `data_quality_report.json` |

The engine extracts the detection status and relevant details from each report.

It then generates an alert containing:

- A unique alert identifier
- The alert generation timestamp
- Model name and version
- Overall alert status
- Alert severity
- Individual monitoring results
- A consolidated summary of detected conditions

The generated alert can be consumed by the Diagnosis Engine for subsequent investigation.

### 48.3 Alert Severity

The current severity rules are:

- **HIGH:** Performance degradation is detected, or at least two alert conditions are detected.
- **MEDIUM:** Exactly one alert condition is detected.
- **INFO:** No alert-triggering monitoring conditions are detected.

Performance degradation always results in HIGH severity under the current implementation.

A failed data-quality check also triggers an alert. A data-quality warning alone does not trigger an alert under the current implementation.

These rules represent the project's initial alert policy and should not be interpreted as a production risk policy.

### 48.4 Individual Monitoring Alerts

The unified alert contains four individual monitoring results.

#### Data Drift

Indicates whether a statistically significant change in input feature distributions has been detected.

The report includes the number of drifted features, the total number of monitored features, the drift rate, and the names of drifted features.

**Source:** `drift_report.json`

#### Prediction Drift

Indicates whether a statistically significant change in model prediction distributions has been detected.

The report includes the KS statistic, p-value, and difference in positive prediction rates when these values are available.

**Source:** `prediction_drift_report.json`

#### Performance Degradation

Indicates whether model performance has deteriorated relative to the established performance baseline.

The report includes the number of degraded metrics, the total number of metrics evaluated, and the names of degraded metrics.

The monitored metrics include:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Log loss
- Brier score

**Source:** `performance_degradation_report.json`

#### Data Quality

Indicates whether the prediction telemetry has passed the configured data-quality checks.

The current checks cover:

- Telemetry structure
- Schema validation
- Numeric-value validation
- Domain-rule validation
- Missing values
- Duplicate events
- Data freshness

The unified alert includes the statuses of individual checks and lists any failed or warning checks.

**Source:** `data_quality_report.json`

### 48.5 Overall Alert Status

The Alert Engine produces one of two overall statuses:

| Status | Meaning |
|---|---|
| `alert` | At least one alert-triggering condition has been detected. |
| `no_alert` | No alert-triggering conditions have been detected. |

A data-quality warning without any other alert-triggering condition does not change the overall status to `alert`.

### 48.6 Output

The unified alert is stored at:

`data/artifacts/monitoring/unified_alert.json`

The report follows a structured JSON format containing the model information, overall status, severity, individual monitoring alerts, and summary.

A simplified representation of the output is:

```json
{
    "alert_id": "ml-monitoring-YYYYMMDDHHMMSS",
    "generated_at": "UTC timestamp",
    "model": {
        "model_name": "xgboost_credit_risk",
        "model_version": "1.0"
    },
    "overall_status": "alert",
    "severity": "HIGH",
    "alerts": [
        {
            "type": "data_drift",
            "status": "not_detected"
        },
        {
            "type": "prediction_drift",
            "status": "detected"
        },
        {
            "type": "performance_degradation",
            "status": "detected"
        },
        {
            "type": "data_quality",
            "status": "failed"
        }
    ],
    "summary": {
        "data_drift_detected": false,
        "prediction_drift_detected": true,
        "performance_degradation_detected": true,
        "data_quality_failed": true,
        "data_quality_warning": false,
        "total_alert_conditions": 3
    }
}
```

This is an illustrative example of the report structure. Actual values depend on the monitoring reports available when the Alert Engine runs.

### 48.7 Current Verification Results

The current implementation has passed the following verification checks:

- All 44 automated tests passed.
- The Alert Engine successfully loaded all four monitoring reports.
- The generated `unified_alert.json` was parsed successfully as valid JSON.
- The alert output contained four monitoring categories.
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

These are results from the current monitoring artifacts, not evidence that the model is experiencing the same conditions in a live production environment.

### 48.8 Current Data Limitation

The current telemetry file contains 1,000 historical prediction events. Its latest event timestamp is:

`2026-09-05T16:07:50.147882+00:00`

The configured maximum event age is 15 minutes. When the data-quality monitor ran on October 13, 2026, the freshness check failed because the latest event was older than the permitted age.

This failure is expected for the current historical dataset. The freshness threshold has not been weakened, and the event timestamps have not been modified to make the check pass.

The current implementation validates the available telemetry and generates monitoring alerts. Validation against continuously arriving production telemetry remains a separate step.

### 48.9 Current Scope

The current ML Monitoring implementation includes:

1. Input data drift detection
2. Prediction drift detection
3. Model performance degradation detection
4. Prediction telemetry data-quality validation
5. Unified alert generation
6. Automated tests for the implemented monitoring components

The Unified Alert Engine completes the consolidation stage of the current ML Monitoring pipeline. Its output provides a single monitoring summary for subsequent diagnosis.

Live production validation, continuous telemetry operation, and any additional deployment or integration work should be tracked separately from the functionality verified in the current implementation.



### 48.8 Alert Response and Escalation Policy

The following is the proposed operational policy for the
IntelliCore AI ML Monitoring module. The current implementation
assigns severity and generates unified alert reports; it does
not automatically notify responders or perform timed escalation.

| Severity | Trigger | Required response | Escalation |
|---|---|---|---|
| INFO | No alert-triggering conditions | Record the monitoring result and continue routine monitoring | No escalation |
| MEDIUM | Exactly one alert-triggering condition, excluding performance degradation | Review the affected monitoring report and investigate the condition | Escalate to HIGH priority if the condition persists, recurs, or is associated with additional failures |
| HIGH | Performance degradation or two or more alert-triggering conditions | Prioritize investigation of the affected model, verify monitoring evidence, and assess potential impact | Notify or refer the issue to the responsible project lead or model owner through the project's agreed communication process |

#### Data-Quality Handling

- A data-quality failure is an alert-triggering condition.
- A data-quality warning alone does not trigger an alert under the current implementation.
- A warning should still be reviewed when investigating a related alert.
- Failed freshness checks must not be interpreted as proof of model degradation.

#### Recovery and Resolution

An alert may be considered resolved after the underlying condition has been investigated and subsequent valid monitoring results no longer indicate that condition.

- Data drift: subsequent drift checks no longer identify significant drift.
- Prediction drift: subsequent prediction-distribution checks no longer identify significant drift.
- Performance degradation: a comparison using reliable ground-truth outcomes no longer meets the configured degradation criterion.
- Data quality: the relevant checks pass, including freshness when applicable.

A single successful check should not automatically be treated as proof of sustained recovery. The responder should consider the evidence and whether the condition is recurring.

The current implementation generates monitoring reports and determines alert severity. Automatic notifications, timed escalation, alert acknowledgement, incident assignment, and persistent recovery tracking are not claimed unless separately implemented and tested.

### 48.9 Monitoring Policy Limitations

1. INFO, MEDIUM, and HIGH are project-defined severity levels, not calibrated production risk classifications.
2. Drift indicates a distribution change; it does not by itself prove that model performance has deteriorated.
3. Performance degradation requires reliable labels or ground-truth outcomes. Controlled degradation experiments validate the detector but do not establish real-world degradation.
4. Historical telemetry that exceeds the configured freshness limit should remain flagged as stale rather than having its timestamps artificially updated.
5. Escalation and recovery procedures in this section are operational guidance; the current alert engine does not automatically execute them.
