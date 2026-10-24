
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.utils.validation import check_is_fitted


class IsolationForestAnomalyDetector:
    """Detect unusual input-feature patterns using Isolation Forest.

    Fit the detector on a representative reference dataset, then use it
    to score new prediction inputs. An anomaly flag indicates an unusual
    feature pattern, not fraud or an incorrect model prediction.
    """

    def __init__(
        self,
        feature_names: Sequence[str],
        contamination: float = 0.05,
        n_estimators: int = 100,
        random_state: int = 42,
    ) -> None:
        if not feature_names:
            raise ValueError("feature_names must not be empty.")

        if len(set(feature_names)) != len(feature_names):
            raise ValueError("feature_names must not contain duplicates.")

        if not 0.0 < contamination <= 0.5:
            raise ValueError("contamination must be greater than 0 and at most 0.5.")

        if n_estimators < 1:
            raise ValueError("n_estimators must be at least 1.")

        self.feature_names = list(feature_names)
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state

        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
        )

    def _validate_records(
        self,
        records: pd.DataFrame | Sequence[dict[str, Any]],
    ) -> pd.DataFrame:
        """Validate and order feature records before scoring."""
        if isinstance(records, pd.DataFrame):
            frame = records.copy()
        else:
            if isinstance(records, (str, bytes)):
                raise ValueError("records must be a DataFrame or a sequence of dictionaries.")

            try:
                frame = pd.DataFrame(list(records))
            except (TypeError, ValueError) as exc:
                raise ValueError("records must contain feature dictionaries.") from exc

        if frame.empty:
            raise ValueError("At least one record is required.")

        if not frame.columns.is_unique:
            raise ValueError("Feature names must not contain duplicates.")

        missing = sorted(set(self.feature_names) - set(frame.columns))
        unexpected = sorted(set(frame.columns) - set(self.feature_names))

        if missing:
            raise ValueError(f"Missing required features: {missing}")

        if unexpected:
            raise ValueError(f"Unexpected features: {unexpected}")

        frame = frame.loc[:, self.feature_names]

        for feature in self.feature_names:
            if (
                pd.api.types.is_bool_dtype(frame[feature])
                or not pd.api.types.is_numeric_dtype(frame[feature])
            ):
                raise ValueError(f"Feature '{feature}' must contain numeric values.")

        values = frame.to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError("Feature values must not contain NaN or infinity.")

        return frame.astype(float)

    def fit(
        self,
        reference_records: pd.DataFrame | Sequence[dict[str, Any]],
    ) -> "IsolationForestAnomalyDetector":
        """Fit on representative reference feature records."""
        frame = self._validate_records(reference_records)

        if len(frame) < 2:
            raise ValueError("At least two reference records are required.")

        self.model.fit(frame)
        self.is_fitted_ = True
        return self

    def detect(
        self,
        records: pd.DataFrame | Sequence[dict[str, Any]],
        record_ids: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Score records and return individual results plus a summary."""
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("Detector must be fitted before calling detect().")

        check_is_fitted(self.model)

        frame = self._validate_records(records)

        if record_ids is not None:
            if len(record_ids) != len(frame):
                raise ValueError("record_ids length must match the number of records.")

            if any(not isinstance(record_id, str) or not record_id.strip()
                   for record_id in record_ids):
                raise ValueError("Each record ID must be a non-empty string.")

        # Isolation Forest returns -1 for anomalies and 1 for inliers.
        predictions = self.model.predict(frame)

        # Lower decision_function values are more anomalous.
        # Negating the values makes larger scores indicate greater unusualness.
        scores = -self.model.decision_function(frame)

        results = []
        for index, (prediction, score) in enumerate(zip(predictions, scores)):
            result = {
                "record_index": index,
                "anomaly_score": float(score),
                "is_anomaly": bool(prediction == -1),
            }

            if record_ids is not None:
                result["record_id"] = record_ids[index]

            results.append(result)

        anomaly_count = sum(item["is_anomaly"] for item in results)
        total_records = len(results)

        return {
            "detector": "isolation_forest",
            "contamination": self.contamination,
            "total_records": total_records,
            "anomaly_count": anomaly_count,
            "anomaly_rate": float(anomaly_count / total_records),
            "results": results,
        }
