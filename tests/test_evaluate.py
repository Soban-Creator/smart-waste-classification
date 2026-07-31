"""Tests for reusable model-evaluation utilities."""

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluate import (
    PredictionResults,
    calculate_evaluation_metrics,
    calculate_multiclass_roc_auc,
    calculate_per_class_accuracy,
    collect_predictions,
    evaluate_model,
    validate_class_names,
)


class FixedClassifier(nn.Module):
    """Return input values directly as two-class logits."""

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        return inputs


def create_evaluation_loader() -> DataLoader:
    """Create deterministic logits and labels."""

    logits = torch.tensor(
        [
            [4.0, 1.0],
            [1.0, 4.0],
            [3.0, 2.0],
            [2.0, 3.0],
        ],
        dtype=torch.float32,
    )

    targets = torch.tensor(
        [0, 1, 0, 1],
        dtype=torch.long,
    )

    return DataLoader(
        TensorDataset(
            logits,
            targets,
        ),
        batch_size=2,
        shuffle=False,
    )


def test_validate_class_names_accepts_valid_names() -> None:
    """Valid class names should become a tuple."""

    result = validate_class_names(
        ["cardboard", "glass"]
    )

    assert result == (
        "cardboard",
        "glass",
    )


@pytest.mark.parametrize(
    "invalid_names",
    [
        ["only_one"],
        ["cardboard", ""],
        ["glass", "glass"],
    ],
)
def test_validate_class_names_rejects_invalid_values(
    invalid_names: list[str],
) -> None:
    """Insufficient, empty, and duplicate names should fail."""

    with pytest.raises(ValueError):
        validate_class_names(invalid_names)


def test_collect_predictions_returns_expected_arrays() -> None:
    """Prediction collection should preserve sample order."""

    model = FixedClassifier()
    loader = create_evaluation_loader()

    results = collect_predictions(
        model=model,
        data_loader=loader,
        device="cpu",
    )

    assert results.sample_count == 4
    assert results.true_labels.shape == (4,)
    assert results.predicted_labels.shape == (4,)
    assert results.probabilities.shape == (4, 2)
    assert results.logits.shape == (4, 2)

    assert np.array_equal(
        results.true_labels,
        np.array([0, 1, 0, 1]),
    )

    assert np.array_equal(
        results.predicted_labels,
        np.array([0, 1, 0, 1]),
    )


def test_calculate_per_class_accuracy() -> None:
    """Per-class accuracy should use confusion-matrix rows."""

    matrix = np.array(
        [
            [8, 2],
            [1, 9],
        ]
    )

    result = calculate_per_class_accuracy(
        confusion_matrix_values=matrix,
        class_names=("class_a", "class_b"),
    )

    assert result["class_a"] == pytest.approx(0.8)
    assert result["class_b"] == pytest.approx(0.9)


def test_perfect_predictions_produce_perfect_metrics() -> None:
    """Correct predictions should yield metric values of one."""

    results = PredictionResults(
        true_labels=np.array([0, 1, 0, 1]),
        predicted_labels=np.array([0, 1, 0, 1]),
        probabilities=np.array(
            [
                [0.9, 0.1],
                [0.1, 0.9],
                [0.8, 0.2],
                [0.2, 0.8],
            ]
        ),
        logits=np.array(
            [
                [3.0, 1.0],
                [1.0, 3.0],
                [2.5, 1.0],
                [1.0, 2.5],
            ]
        ),
        average_inference_time_seconds=0.001,
        sample_count=4,
    )

    metrics = calculate_evaluation_metrics(
        prediction_results=results,
        class_names=("class_a", "class_b"),
    )

    assert metrics.accuracy == pytest.approx(1.0)
    assert metrics.macro_precision == pytest.approx(1.0)
    assert metrics.macro_recall == pytest.approx(1.0)
    assert metrics.macro_f1 == pytest.approx(1.0)
    assert metrics.weighted_f1 == pytest.approx(1.0)
    assert metrics.roc_auc_ovr_macro == pytest.approx(1.0)


def test_multiclass_roc_auc_returns_none_when_class_missing() -> None:
    """ROC-AUC should be skipped when not every class is present."""

    true_labels = np.array([0, 0, 1, 1])

    probabilities = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.7, 0.2, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.8, 0.1],
        ]
    )

    result = calculate_multiclass_roc_auc(
        true_labels=true_labels,
        probabilities=probabilities,
        number_of_classes=3,
    )

    assert result is None


def test_evaluate_model_combines_collection_and_metrics() -> None:
    """The public evaluation function should return both objects."""

    model = FixedClassifier()
    loader = create_evaluation_loader()

    results, metrics = evaluate_model(
        model=model,
        data_loader=loader,
        class_names=("class_a", "class_b"),
        device="cpu",
    )

    assert results.sample_count == 4
    assert metrics.sample_count == 4
    assert metrics.accuracy == pytest.approx(1.0)


def test_metrics_to_dict_is_serializable() -> None:
    """Metrics should convert arrays into ordinary lists."""

    results = PredictionResults(
        true_labels=np.array([0, 1]),
        predicted_labels=np.array([0, 1]),
        probabilities=np.array(
            [
                [0.9, 0.1],
                [0.1, 0.9],
            ]
        ),
        logits=np.array(
            [
                [2.0, 0.0],
                [0.0, 2.0],
            ]
        ),
        average_inference_time_seconds=0.002,
        sample_count=2,
    )

    metrics = calculate_evaluation_metrics(
        prediction_results=results,
        class_names=("class_a", "class_b"),
    )

    dictionary = metrics.to_dict()

    assert isinstance(
        dictionary["confusion_matrix"],
        list,
    )

    assert dictionary["sample_count"] == 2
    