# IntelliCore AI — ML Preprocessing & Model Comparison

## 1. Overview

This document records the preprocessing, exploratory analysis, model training, evaluation, and model-selection process performed for the ML component of IntelliCore AI.

The objective of this experiment is to establish a reliable machine-learning workload that can later be integrated with the IntelliCore AI ML Monitoring system.

The **Give Me Some Credit** dataset is used as the first representative fintech ML workload.

> **Important:** IntelliCore AI is not a credit-risk-only platform. Credit-risk prediction is used as one representative fintech ML use case for developing and demonstrating the ML monitoring capabilities of IntelliCore AI.

---

# 2. Objective

The objectives of this experiment are:

1. Inspect the selected dataset.
2. Understand its structure and data quality.
3. Remove unnecessary columns.
4. Identify and handle duplicate records.
5. Analyze missing values.
6. Handle missing numerical values.
7. Analyze feature ranges and extreme observations.
8. Analyze target-class imbalance.
9. Prepare training and testing datasets.
10. Compare multiple ML algorithms.
11. Evaluate the models using classification metrics.
12. Select a representative model for IntelliCore AI.
13. Prepare the selected model for the subsequent ML monitoring stage.

The models evaluated are:

- Linear Regression
- Logistic Regression
- Random Forest
- XGBoost

---

# 3. Dataset Information

## 3.1 Dataset Used

**Give Me Some Credit**

Dataset location:

```text
data/sample/GiveMeSomeCredit_2011/
```

Files available:

```text
data/sample/GiveMeSomeCredit_2011/
├── sampleEntry.csv
├── cs-test.csv
├── Data Dictionary.xls
└── cs-training.csv
```

---

# 4. Dataset Selection

The dataset contains information related to borrowers and their financial/credit characteristics.

The training dataset was selected because it contains the labeled target variable:

```text
SeriousDlqin2yrs
```

The other files serve different purposes.

### `cs-training.csv`

Used for supervised ML training because it contains the target variable.

### `cs-test.csv`

Contains competition test observations but does not contain usable target labels for supervised evaluation.

Therefore, it is not used for the current supervised training experiment.

### `sampleEntry.csv`

Contains the expected competition submission format:

```text
Id
Probability
```

It is not used for model training.

### `Data Dictionary.xls`

Contains descriptions of the dataset features and is used as a reference for understanding the variables.

---

# 5. Initial Dataset Inspection

The original `cs-training.csv` contains:

```text
Rows: 150,000
Columns: 12
```

The original columns are:

```text
Unnamed: 0
SeriousDlqin2yrs
RevolvingUtilizationOfUnsecuredLines
age
NumberOfTime30-59DaysPastDueNotWorse
DebtRatio
MonthlyIncome
NumberOfOpenCreditLinesAndLoans
NumberOfTimes90DaysLate
NumberRealEstateLoansOrLines
NumberOfTime60-89DaysPastDueNotWorse
NumberOfDependents
```

All predictive variables are numerical.

---

# 6. Data Cleaning

## 6.1 Removing the Index Column

The dataset contains the column:

```text
Unnamed: 0
```

This column represents an exported row/index identifier and does not represent a meaningful predictive feature.

Therefore, it was removed before model training.

### Result

```text
Original columns: 12
Columns after removing Unnamed: 0: 11
```

The resulting dataset contains:

- 10 predictive features
- 1 target variable

---

# 7. Duplicate Analysis

Duplicate records were checked after removing the unnecessary index column.

The preprocessing experiment identified:

```text
Duplicate rows: 609
```

These duplicate observations were removed.

### Result

```text
Rows before duplicate removal: 150,000
Duplicate rows removed: 609
Rows after duplicate removal: 149,391
```

Therefore, the final dataset used for the experiment contains:

```text
149,391 rows
```

---

# 8. Missing Value Analysis

Missing values were identified in two numerical features.

| Feature            | Missing Values |
| ------------------ | -------------: |
| MonthlyIncome      |         29,221 |
| NumberOfDependents |          3,828 |

The remaining features did not contain missing values requiring imputation.

### Missing-value percentages

For the dataset after duplicate removal:

| Feature            | Missing Values | Approx. Percentage |
| ------------------ | -------------: | -----------------: |
| MonthlyIncome      |         29,221 |             19.56% |
| NumberOfDependents |          3,828 |              2.56% |

`MonthlyIncome` therefore has a relatively high proportion of missing observations and requires appropriate treatment.

---

# 9. Missing Value Imputation

Both missing-value features are numerical.

Median imputation was selected because the financial variables contain highly skewed values and extreme observations.

The median is less sensitive to extreme values than the mean.

The selected strategy is:

```text
MonthlyIncome
        ↓
Median imputation

NumberOfDependents
        ↓
Median imputation
```

---

# 10. Data Leakage Consideration

The initial preprocessing experiment performed imputation before the train/test split.

For the final research implementation, this should be corrected.

The correct procedure is:

```text
Raw Dataset
     ↓
Train/Test Split
     ↓
Training Data
     ↓
Fit Median Imputer
     ↓
Transform Training Data
     ↓
Transform Test Data
using training medians
```

The test dataset must not influence the statistics used to preprocess the training data.

This prevents preprocessing leakage.

---

# 11. Post-Imputation Validation

After imputation, the preprocessing experiment verified that there were no remaining missing values.

```text
Missing values after imputation: 0
```

Additional validation was performed for numerical validity:

```text
NaN values: 0
Infinite values: 0
```

Therefore, the processed dataset contains valid numerical input values.

---

# 12. Feature Analysis

After removing the index column and separating the target variable, the dataset contains **10 predictive features**.

The features are:

```text
1. RevolvingUtilizationOfUnsecuredLines
2. age
3. NumberOfTime30-59DaysPastDueNotWorse
4. DebtRatio
5. MonthlyIncome
6. NumberOfOpenCreditLinesAndLoans
7. NumberOfTimes90DaysLate
8. NumberRealEstateLoansOrLines
9. NumberOfTime60-89DaysPastDueNotWorse
10. NumberOfDependents
```

---

# 13. Feature Range Analysis

The preprocessing experiment identified a wide range of values across the numerical features.

Important maximum values observed include:

| Feature                              |   Maximum |
| ------------------------------------ | --------: |
| RevolvingUtilizationOfUnsecuredLines |    50,708 |
| DebtRatio                            |   329,664 |
| MonthlyIncome                        | 3,008,750 |
| NumberOfTime30-59DaysPastDueNotWorse |        98 |
| NumberOfTimes90DaysLate              |        98 |
| NumberOfTime60-89DaysPastDueNotWorse |        98 |
| NumberOfOpenCreditLinesAndLoans      |        58 |
| NumberRealEstateLoansOrLines         |        54 |
| NumberOfDependents                   |        20 |
| age                                  |       109 |

These ranges show that the dataset contains significant variation between observations.

---

# 14. Extreme Value Analysis

Several features contain extremely large observations.

Examples:

```text
RevolvingUtilizationOfUnsecuredLines → 50,708
DebtRatio                         → 329,664
MonthlyIncome                     → 3,008,750
```

These values were not automatically removed.

## Reason

Extreme observations in financial datasets may represent:

- unusual financial behavior
- highly leveraged customers
- unusual income levels
- potential data-entry anomalies
- potentially important predictive signals

Since IntelliCore AI is intended to monitor ML systems for anomalies and distribution changes, extreme observations may also be useful later for monitoring experiments.

Therefore, the current approach is:

> **Identify extreme observations first rather than blindly deleting them.**

A separate outlier-treatment experiment can be performed later if required.

---

# 15. Target Variable Analysis

The target variable is:

```text
SeriousDlqin2yrs
```

It represents a binary classification problem.

```text
0 → No serious delinquency
1 → Serious delinquency
```

After duplicate removal:

```text
Total observations: 149,391

Class 0: 139,382
Class 1: 10,009
```

---

# 16. Class Distribution

The target distribution is:

| Class     |       Count | Percentage |
| --------- | ----------: | ---------: |
| 0         |     139,382 |     93.30% |
| 1         |      10,009 |      6.70% |
| **Total** | **149,391** |   **100%** |

The dataset is therefore significantly imbalanced.

The positive class represents approximately:

```text
6.70%
```

of the observations.

---

# 17. Impact of Class Imbalance

Because the positive class is relatively rare, accuracy alone is not sufficient to evaluate model performance.

A model that predicts the majority class most of the time could achieve high accuracy while performing poorly at identifying the minority class.

Therefore, the experiment evaluates:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

Additional metrics such as PR-AUC should be included in the final research evaluation.

---

# 18. Feature and Target Separation

The target variable was separated from the predictive features.

```python
X = features
y = SeriousDlqin2yrs
```

The resulting feature matrix contains:

```text
X → 149,391 rows × 10 features
y → 149,391 target values
```

---

# 19. Train-Test Split

The data was divided into:

```text
Training set → 80%
Testing set  → 20%
```

The split uses:

```text
random_state = 42
```

and stratification based on the target variable.

This ensures that the class distribution remains approximately consistent between the training and testing sets.

### Result

```text
Training data:
X_train → 119,512 × 10
y_train → 119,512

Testing data:
X_test → 29,879 × 10
y_test → 29,879
```

The positive-class proportion was approximately:

```text
6.70%
```

in both partitions.

---

# 20. Model Comparison

Four models were evaluated:

1. Linear Regression
2. Logistic Regression
3. Random Forest
4. XGBoost

The purpose of this comparison was to understand how different modeling approaches perform on the same dataset.

---

# 21. Linear Regression

Linear Regression was included as a simple baseline.

Although the target is binary and classification algorithms are more appropriate, Linear Regression provides a basic reference point.

StandardScaler was applied to the input features for this model.

### Results

```text
RMSE      = 0.2436
MAE       = 0.1206
R²        = 0.0510

Accuracy  = 0.9330
Precision = 0.4923
Recall    = 0.0160
F1        = 0.0310
ROC-AUC   = 0.6939
```

### Interpretation

The model achieves relatively high accuracy because the dataset is dominated by class 0.

However:

```text
Recall = 1.60%
F1     = 3.10%
```

indicate poor identification of the positive class.

Therefore, Linear Regression is not suitable as the selected model.

---

# 22. Logistic Regression

Logistic Regression was included as a more appropriate linear baseline for binary classification.

StandardScaler was applied to the features.

### Results

```text
Accuracy  = 0.9331
Precision = 0.5130
Recall    = 0.0395
F1        = 0.0733
ROC-AUC   = 0.6990
```

### Interpretation

Logistic Regression performs slightly better than Linear Regression.

However, the recall remains very low:

```text
Recall = 3.95%
```

This indicates that the model identifies only a small proportion of positive-class observations at the default classification threshold.

It therefore does not provide the strongest candidate for the current ML workload.

---

# 23. Random Forest

Random Forest was evaluated as a non-linear ensemble tree-based model.

The configuration used was:

```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
```

The `class_weight="balanced"` setting was used to account for class imbalance.

### Results

```text
Accuracy  = 0.8156
Precision = 0.2293
Recall    = 0.7423
F1        = 0.3503
ROC-AUC   = 0.8582
```

### Interpretation

Random Forest achieves substantially higher recall than the other models:

```text
Recall = 74.23%
```

However, its precision is considerably lower:

```text
Precision = 22.93%
```

This means the model identifies many positive-class observations but also produces a relatively large number of false positive predictions.

Its ROC-AUC of:

```text
0.8582
```

shows strong discriminatory ability, but it is slightly lower than XGBoost.

---

# 24. XGBoost

XGBoost was evaluated as the advanced gradient-boosting model.

The configuration used was:

```python
XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)
```

### Results

```text
Accuracy  = 0.9372
Precision = 0.5915
Recall    = 0.2018
F1        = 0.3009
ROC-AUC   = 0.8651
```

### Interpretation

XGBoost achieved the strongest results among the models evaluated in the initial experiment.

It achieved the highest:

```text
Accuracy
Precision
ROC-AUC
```

Its ROC-AUC was:

```text
0.8651
```

which was higher than:

```text
Random Forest     → 0.8582
Logistic Regression → 0.6990
Linear Regression   → 0.6939
```

XGBoost also achieved the highest precision:

```text
59.15%
```

However, Random Forest achieved higher recall and F1 at the default 0.5 threshold.

This demonstrates that model performance depends on the evaluation objective and classification threshold.

---

# 25. Overall Model Comparison

The complete initial comparison is:

| Model               |   Accuracy |  Precision |     Recall |         F1 |    ROC-AUC |
| ------------------- | ---------: | ---------: | ---------: | ---------: | ---------: |
| Linear Regression   |     93.30% |     49.23% |      1.60% |      3.10% |     69.39% |
| Logistic Regression |     93.31% |     51.30% |      3.95% |      7.33% |     69.90% |
| Random Forest       |     81.56% |     22.93% | **74.23%** | **35.03%** |     85.82% |
| XGBoost             | **93.72%** | **59.15%** |     20.18% |     30.09% | **86.51%** |

---

# 26. Best Model by Individual Metric

### Accuracy

```text
XGBoost → 93.72%
```

### Precision

```text
XGBoost → 59.15%
```

### Recall

```text
Random Forest → 74.23%
```

### F1-score

```text
Random Forest → 35.03%
```

### ROC-AUC

```text
XGBoost → 86.51%
```

---

# 27. Model Comparison Interpretation

The results show two different performance characteristics.

## Random Forest

Random Forest is highly sensitive to the positive class at the default threshold.

```text
High Recall
Low Precision
```

It identifies many positive-class observations but generates more false positives.

## XGBoost

XGBoost provides:

```text
High Precision
High Accuracy
Highest ROC-AUC
```

while its recall at the default threshold is lower than Random Forest.

This indicates that XGBoost provides stronger overall ranking/discrimination performance, while Random Forest is more aggressive in classifying observations as positive.

---

# 28. Why Accuracy Alone Was Not Used

Accuracy is:

```text
Correct predictions / Total predictions
```

Because approximately 93.3% of the observations belong to class 0, a model can obtain high accuracy without identifying many class-1 observations.

Therefore, the experiment does not select a model based on accuracy alone.

ROC-AUC, precision, recall, F1-score, and eventually PR-AUC are considered.

---

# 29. Model Selection

Based on the initial model comparison, **XGBoost is selected as the model for the first IntelliCore AI ML workload.**

The primary reasons are:

1. Highest ROC-AUC
2. Highest precision
3. Highest accuracy
4. Strong non-linear modeling capability
5. Suitable for tabular numerical data
6. Produces prediction probabilities that can be used for monitoring

The selected model achieved:

```text
Accuracy  = 93.72%
Precision = 59.15%
Recall    = 20.18%
F1        = 30.09%
ROC-AUC   = 86.51%
```

---

# 30. Important Scope of Model Selection

The selection of XGBoost does **not** mean:

> XGBoost is the best model for every fintech application.

IntelliCore AI is designed as a general ML/AI reliability and monitoring platform for fintech systems.

Possible ML workloads include:

```text
Credit Risk
Fraud Detection
AML Detection
Customer Risk Scoring
Financial Forecasting
Churn Prediction
Transaction Classification
```

The Give Me Some Credit dataset represents only one possible workload.

Therefore, the correct research interpretation is:

> **XGBoost was selected as the representative model for the first IntelliCore AI ML experiment based on its performance on the selected dataset.**

---

# 31. Threshold Consideration

The reported precision, recall, and F1 values are based on the default classification threshold:

```text
0.5
```

However, XGBoost produces probability estimates.

For example:

```text
Prediction probability = 0.73
```

can be converted into a positive prediction using a selected threshold.

Different thresholds can produce different precision-recall trade-offs.

Therefore, further evaluation should test thresholds such as:

```text
0.2
0.3
0.4
0.5
0.6
0.7
```

This will be performed before considering the model experiment completely finalized.

---

# 32. Additional Evaluation Required

Because the dataset is imbalanced, the final XGBoost experiment should include additional metrics.

These include:

### PR-AUC

Precision-Recall AUC is particularly useful when the positive class is relatively rare.

### Confusion Matrix

Used to examine:

```text
True Positives
True Negatives
False Positives
False Negatives
```

### Log Loss

Measures the quality of predicted probabilities.

### Calibration

Checks whether predicted probabilities correspond reasonably to observed outcomes.

### Brier Score

Provides another measure of probabilistic prediction quality.

These evaluations will strengthen the research validity of the final model.

---

# 33. Numerical Stability Observations

During the initial experiment, Linear Regression and Logistic Regression generated numerical runtime warnings related to matrix operations.

Examples included:

```text
divide by zero encountered in matmul
overflow encountered in matmul
invalid value encountered in matmul
```

These warnings did not occur with the XGBoost model.

The warnings should be investigated before using the linear-model results as final benchmark results.

Potential factors include:

- feature scale
- feature magnitude
- multicollinearity
- numerical conditioning
- solver behavior
- extreme feature values

The XGBoost result is not affected by these particular linear-model warnings.

---

# 34. Preprocessing Leakage Correction

The initial preprocessing experiment performed median imputation before the train/test split.

For the final implementation, preprocessing must be changed to:

```text
Raw Dataset
      ↓
Remove index
      ↓
Remove duplicates
      ↓
Separate target
      ↓
Train/Test Split
      ↓
Fit imputer on training data
      ↓
Transform training data
      ↓
Transform test data
      ↓
Train XGBoost
```

This ensures that no information from the test set is used during training.

The final reported research metrics should ideally be regenerated using this leakage-free implementation.

---

# 35. Reproducibility

The experiment uses:

```text
random_state = 42
```

for reproducibility.

The primary environment includes:

```text
Python
pandas
NumPy
scikit-learn
XGBoost
```

The exact package versions should be recorded when the final experiment is executed.

---

# 36. Current Data Pipeline

The current ML preparation pipeline is:

```text
cs-training.csv
        ↓
Initial inspection
        ↓
Remove Unnamed: 0
        ↓
Remove 609 duplicates
        ↓
149,391 observations
        ↓
Missing-value analysis
        ↓
Median imputation
        ↓
NaN / infinite validation
        ↓
Feature-target separation
        ↓
80/20 stratified split
        ↓
Model comparison
        ↓
XGBoost selected
```

---

# 37. IntelliCore AI Integration

The model itself is only one component of the IntelliCore AI system.

The next stage is to use the trained model to generate prediction events.

The overall architecture becomes:

```text
                  Dataset
                     ↓
              Preprocessing
                     ↓
                 XGBoost
                     ↓
                Prediction
                     ↓
             Prediction Event
                     ↓
          ┌──────────────────────┐
          │ IntelliCore AI       │
          │ ML Monitoring        │
          └──────────┬───────────┘
                     ↓
       ┌─────────────┼──────────────┐
       ↓             ↓              ↓
  Data Drift   Prediction Drift  Performance
       ↓             ↓              ↓
       └─────────────┼──────────────┘
                     ↓
                 Diagnosis
                     ↓
                   Alerts
                     ↓
                 Dashboard
```

---

# 38. What Will Be Monitored

Once the XGBoost model is integrated with IntelliCore, the system can monitor:

## Data Drift

Changes in the distribution of incoming features.

## Feature Drift

Changes in individual feature distributions.

## Prediction Drift

Changes in the model's prediction distribution.

## Model Performance

Changes in metrics such as:

```text
Precision
Recall
F1
ROC-AUC
PR-AUC
Log Loss
```

when ground-truth labels become available.

## Anomalies

Unusual inputs or prediction behavior.

## Explainability

SHAP-based analysis can help determine which features are contributing to model predictions.

---

# 39. Research Flow

The complete research workflow is:

```text
Dataset Selection
       ↓
Dataset Inspection
       ↓
Data Cleaning
       ↓
Missing Value Analysis
       ↓
Imputation
       ↓
Feature Analysis
       ↓
Extreme Value Analysis
       ↓
Target Distribution
       ↓
Train/Test Split
       ↓
Model Comparison
       ↓
XGBoost Selection
       ↓
Leakage-Free Final Training
       ↓
PR-AUC / Confusion Matrix
       ↓
Threshold Analysis
       ↓
Calibration
       ↓
SHAP Explainability
       ↓
Model Artifact Storage
       ↓
Prediction Generation
       ↓
ML Monitoring
       ↓
Drift Detection
       ↓
Diagnosis
       ↓
Alerts / Dashboard
```

---

# 40. Current Status

| Stage                      | Status              |
| -------------------------- | ------------------- |
| Dataset selection          | ✅ Complete         |
| Dataset inspection         | ✅ Complete         |
| Index removal              | ✅ Complete         |
| Duplicate analysis         | ✅ Complete         |
| Missing-value analysis     | ✅ Complete         |
| Initial imputation         | ✅ Complete         |
| Feature analysis           | ✅ Complete         |
| Extreme-value analysis     | ✅ Complete         |
| Target analysis            | ✅ Complete         |
| Train/test split           | ✅ Complete         |
| Model comparison           | ✅ Complete         |
| XGBoost selection          | ✅ Complete         |
| Leakage-free preprocessing | 🔄 To finalize      |
| Final XGBoost evaluation   | 🔄 Pending          |
| PR-AUC                     | ⏳ Pending          |
| Confusion matrix           | ⏳ Pending          |
| Threshold analysis         | ⏳ Pending          |
| Calibration                | ⏳ Pending          |
| SHAP                       | ⏳ Pending          |
| Model artifact storage     | ⏳ Pending          |
| Prediction telemetry       | ⏳ Pending          |
| ML Monitoring              | ⏳ Next major stage |

---

# 41. Conclusion

The preprocessing experiment successfully transformed the original dataset from:

```text
150,000 rows
12 columns
```

into:

```text
149,391 rows
11 columns
10 predictive features
1 binary target
```

after removing the unnecessary index column and 609 duplicate observations.

Missing values were identified primarily in:

```text
MonthlyIncome
NumberOfDependents
```

and median imputation was selected as the treatment strategy.

The target distribution was found to be highly imbalanced:

```text
Class 0 → 93.30%
Class 1 → 6.70%
```

Four models were evaluated.

The initial comparison showed that:

```text
Random Forest
→ Highest Recall and F1

XGBoost
→ Highest Accuracy, Precision and ROC-AUC
```

XGBoost achieved the highest ROC-AUC:

```text
0.8651
```

and was therefore selected as the representative model for the first IntelliCore AI ML workload.

However, the experiment is **not yet considered the final research evaluation**.

The next stage is to regenerate the final XGBoost experiment using leakage-free preprocessing and perform:

```text
PR-AUC
Confusion Matrix
Threshold Analysis
Calibration
Error Analysis
SHAP Explainability
```

After that, the trained model will provide prediction telemetry for the IntelliCore AI ML Monitoring subsystem.

---

# 42. Key Research Statement

> **IntelliCore AI uses the Give Me Some Credit dataset as a representative fintech ML workload. After preprocessing and comparative evaluation of Linear Regression, Logistic Regression, Random Forest, and XGBoost, XGBoost was selected for the initial IntelliCore ML pipeline because it achieved the highest ROC-AUC (0.8651), precision (0.5915), and accuracy (0.9372) in the initial experiment. This selection is specific to the current experimental workload and is not intended to establish XGBoost as universally optimal for fintech applications.**
