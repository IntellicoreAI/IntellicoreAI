
import numpy as np
import pandas as pd
import pytest

from app.explainability.shap_explainer import (
    calculate_feature_importance,
    calculate_shap_values,
    create_sample,
)


class FakeExplanation:
    def __init__(self, values):
        self.values = values


class FakeTreeExplainer:
    def __init__(self, model):
        self.model = model

    def __call__(self, X):
        # One SHAP contribution per row and feature.
        return FakeExplanation(
            np.ones((len(X), X.shape[1]), dtype=float)
        )


def test_create_sample_uses_reproducible_sample(monkeypatch):
    import app.explainability.shap_explainer as module

    monkeypatch.setattr(module, "SHAP_SAMPLE_SIZE", 5)

    X = pd.DataFrame({
        "feature_a": range(20),
        "feature_b": range(20, 40),
    })

    first = create_sample(X)
    second = create_sample(X)

    assert len(first) == 5
    assert first.equals(second)
    assert set(first.columns) == {"feature_a", "feature_b"}


def test_create_sample_keeps_small_dataset_unchanged():
    X = pd.DataFrame({
        "feature_a": [1, 2],
        "feature_b": [3, 4],
    })

    sample = create_sample(X)

    assert sample.equals(X)
    assert sample is not X


def test_calculate_shap_values_returns_expected_shape(monkeypatch):
    import app.explainability.shap_explainer as module

    monkeypatch.setattr(module.shap, "TreeExplainer", FakeTreeExplainer)

    X = pd.DataFrame({
        "feature_a": [1.0, 2.0, 3.0],
        "feature_b": [4.0, 5.0, 6.0],
    })

    values = calculate_shap_values(object(), X)

    assert values.shape == (3, 2)
    assert np.isfinite(values).all()


def test_calculate_feature_importance_ranks_features():
    values = np.array([
        [1.0, -3.0, 2.0],
        [-1.0, 1.0, 4.0],
    ])

    result = calculate_feature_importance(
        values,
        ["feature_a", "feature_b", "feature_c"],
    )

    assert result.iloc[0]["feature"] == "feature_c"
    assert result.iloc[0]["mean_abs_shap"] == pytest.approx(3.0)
    assert result.iloc[1]["feature"] == "feature_b"
    assert result.iloc[1]["mean_abs_shap"] == pytest.approx(2.0)
    assert result.iloc[2]["feature"] == "feature_a"
    assert result.iloc[2]["mean_abs_shap"] == pytest.approx(1.0)


def test_calculate_feature_importance_rejects_mismatched_features():
    values = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    with pytest.raises(ValueError):
        calculate_feature_importance(values, ["feature_a"])


def test_calculate_shap_values_handles_three_dimensional_output(monkeypatch):
    import app.explainability.shap_explainer as module

    class ThreeDimensionalExplainer:
        def __init__(self, model):
            pass

        def __call__(self, X):
            return FakeExplanation(
                np.ones((len(X), X.shape[1], 2), dtype=float)
            )

    monkeypatch.setattr(
        module.shap,
        "TreeExplainer",
        ThreeDimensionalExplainer,
    )

    X = pd.DataFrame({
        "feature_a": [1.0, 2.0],
        "feature_b": [3.0, 4.0],
    })

    values = calculate_shap_values(object(), X)

    assert values.shape == (2, 2)
    assert np.isfinite(values).all()


from app.explainability.lime_explainer import (
    explain_instance,
    save_explanation,
    validate_instance,
)


def test_validate_instance_returns_ordered_dataframe():
    features = ["age", "DebtRatio"]
    instance = {"DebtRatio": 0.5, "age": 40}

    result = validate_instance(instance, features)

    assert list(result.columns) == features
    assert result.iloc[0].tolist() == [40.0, 0.5]


@pytest.mark.parametrize(
    "instance",
    [
        {"age": 40},
        {"age": 40, "DebtRatio": 0.5, "extra": 1},
        {"age": float("nan"), "DebtRatio": 0.5},
        {"age": float("inf"), "DebtRatio": 0.5},
        {"age": True, "DebtRatio": 0.5},
        {"age": "not-a-number", "DebtRatio": 0.5},
    ],
)
def test_validate_instance_rejects_invalid_inputs(instance):
    with pytest.raises(ValueError):
        validate_instance(instance, ["age", "DebtRatio"])


def test_explain_instance_returns_expected_structure():
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(42)
    training_data = pd.DataFrame(
        rng.normal(size=(100, 2)),
        columns=["feature_a", "feature_b"],
    )
    labels = (training_data["feature_a"] > 0).astype(int)

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(training_data, labels)

    instance = training_data.iloc[0].to_dict()

    result = explain_instance(
        model=model,
        training_data=training_data,
        instance=instance,
        feature_names=list(training_data.columns),
        num_features=2,
        num_samples=200,
        random_state=42,
    )

    assert result["method"] == "LIME"
    assert result["prediction"] in (0, 1)
    assert set(result["class_probabilities"]) == {"0", "1"}
    assert sum(result["class_probabilities"].values()) == pytest.approx(1.0)
    assert len(result["local_explanation"]) <= 2
    assert all(
        np.isfinite(item["contribution"])
        for item in result["local_explanation"]
    )
    assert np.isfinite(result["score"])


def test_explain_instance_rejects_empty_training_data():
    from sklearn.dummy import DummyClassifier

    training_data = pd.DataFrame(columns=["feature_a"])
    model = DummyClassifier()

    with pytest.raises(ValueError, match="training_data must not be empty"):
        explain_instance(
            model=model,
            training_data=training_data,
            instance={"feature_a": 1.0},
            feature_names=["feature_a"],
        )


def test_save_explanation_writes_json(tmp_path):
    import json

    result = {
        "method": "LIME",
        "prediction": 0,
        "score": 0.8,
    }

    output_path = save_explanation(result, tmp_path / "explanation.json")

    assert output_path.exists()
    assert json.loads(output_path.read_text(encoding="utf-8")) == result
