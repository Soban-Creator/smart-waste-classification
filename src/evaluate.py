"""Reusable evaluation utilities for image-classification models.

This module collects predictions and calculates classification metrics
for the baseline CNN and future transfer-learning models.

Training logic remains in ``src.train``.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from torch import Tensor, nn
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class PredictionResults:
    """Raw prediction data collected from one dataset split."""

    true_labels: np.ndarray
    predicted_labels: np.ndarray
    probabilities: np.ndarray
    logits: np.ndarray
    average_inference_time_seconds: float
    sample_count: int


@dataclass(frozen=True)
class EvaluationMetrics:
    """Calculated multiclass classification metrics."""

    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_precision: float
    weighted_recall: float
    weighted_f1: float
    roc_auc_ovr_macro: float | None
    confusion_matrix: np.ndarray
    per_class_accuracy: dict[str, float]
    classification_report: dict[str, Any]
    average_inference_time_seconds: float
    sample_count: int

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable representation."""

        return {
            "accuracy": self.accuracy,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
            "weighted_precision": self.weighted_precision,
            "weighted_recall": self.weighted_recall,
            "weighted_f1": self.weighted_f1,
            "roc_auc_ovr_macro": self.roc_auc_ovr_macro,
            "confusion_matrix": self.confusion_matrix.tolist(),
            "per_class_accuracy": self.per_class_accuracy,
            "classification_report": self.classification_report,
            "average_inference_time_seconds": (
                self.average_inference_time_seconds
            ),
            "sample_count": self.sample_count,
        }


def validate_class_names(
    class_names: tuple[str, ...] | list[str],
) -> tuple[str, ...]:
    """Validate class names used during metric calculation."""

    if not isinstance(class_names, (tuple, list)):
        raise TypeError(
            "class_names must be a tuple or list of strings."
        )

    validated_names = tuple(class_names)

    if len(validated_names) < 2:
        raise ValueError(
            "At least two class names are required."
        )

    if not all(
        isinstance(class_name, str)
        and class_name.strip()
        for class_name in validated_names
    ):
        raise ValueError(
            "Every class name must be a non-empty string."
        )

    if len(set(validated_names)) != len(validated_names):
        raise ValueError(
            "Class names must be unique."
        )

    return validated_names


def collect_predictions(
    *,
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device | str,
) -> PredictionResults:
    """Collect logits, probabilities, predictions, and targets."""

    if not isinstance(model, nn.Module):
        raise TypeError(
            "model must be a torch.nn.Module."
        )

    selected_device = torch.device(device)

    model.to(selected_device)
    model.eval()

    all_true_labels: list[Tensor] = []
    all_predicted_labels: list[Tensor] = []
    all_probabilities: list[Tensor] = []
    all_logits: list[Tensor] = []

    total_inference_time = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(
                selected_device,
                non_blocking=True,
            )

            batch_start = time.perf_counter()

            logits = model(images)

            if selected_device.type == "cuda":
                torch.cuda.synchronize()

            batch_duration = (
                time.perf_counter() - batch_start
            )

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            predicted_labels = probabilities.argmax(
                dim=1
            )

            batch_size = targets.shape[0]

            total_inference_time += batch_duration
            total_samples += int(batch_size)

            all_true_labels.append(
                targets.detach().cpu()
            )

            all_predicted_labels.append(
                predicted_labels.detach().cpu()
            )

            all_probabilities.append(
                probabilities.detach().cpu()
            )

            all_logits.append(
                logits.detach().cpu()
            )

    if total_samples == 0:
        raise ValueError(
            "The evaluation DataLoader produced no samples."
        )

    true_labels = torch.cat(
        all_true_labels
    ).numpy()

    predicted_labels = torch.cat(
        all_predicted_labels
    ).numpy()

    probabilities = torch.cat(
        all_probabilities
    ).numpy()

    logits_array = torch.cat(
        all_logits
    ).numpy()

    return PredictionResults(
        true_labels=true_labels,
        predicted_labels=predicted_labels,
        probabilities=probabilities,
        logits=logits_array,
        average_inference_time_seconds=(
            total_inference_time / total_samples
        ),
        sample_count=total_samples,
    )


def calculate_per_class_accuracy(
    *,
    confusion_matrix_values: np.ndarray,
    class_names: tuple[str, ...],
) -> dict[str, float]:
    """Calculate accuracy independently for every class."""

    if confusion_matrix_values.shape != (
        len(class_names),
        len(class_names),
    ):
        raise ValueError(
            "Confusion-matrix dimensions must match "
            "the number of classes."
        )

    per_class_accuracy: dict[str, float] = {}

    for class_index, class_name in enumerate(
        class_names
    ):
        total_true_samples = int(
            confusion_matrix_values[
                class_index,
                :,
            ].sum()
        )

        correct_predictions = int(
            confusion_matrix_values[
                class_index,
                class_index,
            ]
        )

        if total_true_samples == 0:
            per_class_accuracy[class_name] = 0.0
        else:
            per_class_accuracy[class_name] = (
                correct_predictions
                / total_true_samples
            )

    return per_class_accuracy


def calculate_multiclass_roc_auc(
    *,
    true_labels: np.ndarray,
    probabilities: np.ndarray,
    number_of_classes: int,
) -> float | None:
    """Calculate macro one-vs-rest ROC-AUC when possible."""

    if probabilities.ndim != 2:
        raise ValueError(
            "probabilities must be a two-dimensional array."
        )

    if probabilities.shape[1] != number_of_classes:
        raise ValueError(
            "Probability columns must match the number of classes."
        )

    present_classes = np.unique(true_labels)

    if len(present_classes) != number_of_classes:
        return None

    binary_targets = label_binarize(
        true_labels,
        classes=np.arange(number_of_classes),
    )

    try:
        return float(
            roc_auc_score(
                binary_targets,
                probabilities,
                average="macro",
                multi_class="ovr",
            )
        )
    except ValueError:
        return None


def calculate_evaluation_metrics(
    *,
    prediction_results: PredictionResults,
    class_names: tuple[str, ...] | list[str],
) -> EvaluationMetrics:
    """Calculate mandatory multiclass evaluation metrics."""

    validated_class_names = validate_class_names(
        class_names
    )

    number_of_classes = len(
        validated_class_names
    )

    expected_labels = np.arange(
        number_of_classes
    )

    if (
        prediction_results.probabilities.ndim != 2
        or prediction_results.probabilities.shape[1]
        != number_of_classes
    ):
        raise ValueError(
            "Prediction probabilities do not match "
            "the number of class names."
        )

    accuracy = float(
        accuracy_score(
            prediction_results.true_labels,
            prediction_results.predicted_labels,
        )
    )

    (
        macro_precision,
        macro_recall,
        macro_f1,
        _,
    ) = precision_recall_fscore_support(
        prediction_results.true_labels,
        prediction_results.predicted_labels,
        labels=expected_labels,
        average="macro",
        zero_division=0,
    )

    (
        weighted_precision,
        weighted_recall,
        weighted_f1,
        _,
    ) = precision_recall_fscore_support(
        prediction_results.true_labels,
        prediction_results.predicted_labels,
        labels=expected_labels,
        average="weighted",
        zero_division=0,
    )

    matrix = confusion_matrix(
        prediction_results.true_labels,
        prediction_results.predicted_labels,
        labels=expected_labels,
    )

    per_class_accuracy = calculate_per_class_accuracy(
        confusion_matrix_values=matrix,
        class_names=validated_class_names,
    )

    report = classification_report(
        prediction_results.true_labels,
        prediction_results.predicted_labels,
        labels=expected_labels,
        target_names=validated_class_names,
        output_dict=True,
        zero_division=0,
    )

    roc_auc = calculate_multiclass_roc_auc(
        true_labels=prediction_results.true_labels,
        probabilities=prediction_results.probabilities,
        number_of_classes=number_of_classes,
    )

    return EvaluationMetrics(
        accuracy=accuracy,
        macro_precision=float(macro_precision),
        macro_recall=float(macro_recall),
        macro_f1=float(macro_f1),
        weighted_precision=float(
            weighted_precision
        ),
        weighted_recall=float(
            weighted_recall
        ),
        weighted_f1=float(
            weighted_f1
        ),
        roc_auc_ovr_macro=roc_auc,
        confusion_matrix=matrix,
        per_class_accuracy=per_class_accuracy,
        classification_report=report,
        average_inference_time_seconds=(
            prediction_results
            .average_inference_time_seconds
        ),
        sample_count=prediction_results.sample_count,
    )


def evaluate_model(
    *,
    model: nn.Module,
    data_loader: DataLoader,
    class_names: tuple[str, ...] | list[str],
    device: torch.device | str,
) -> tuple[PredictionResults, EvaluationMetrics]:
    """Collect predictions and calculate all evaluation metrics."""

    prediction_results = collect_predictions(
        model=model,
        data_loader=data_loader,
        device=device,
    )

    metrics = calculate_evaluation_metrics(
        prediction_results=prediction_results,
        class_names=class_names,
    )

    return prediction_results, metrics


__all__ = [
    "EvaluationMetrics",
    "PredictionResults",
    "calculate_evaluation_metrics",
    "calculate_multiclass_roc_auc",
    "calculate_per_class_accuracy",
    "collect_predictions",
    "evaluate_model",
    "validate_class_names",
]
