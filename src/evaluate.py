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
    """Calculated binary or multiclass classification metrics."""

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
        """Return a JSON-serializable dictionary representation."""

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
    """Validate class names used during metric calculation.

    Args:
        class_names:
            Ordered class names matching model-output indices.

    Returns:
        The validated class names as a tuple.

    Raises:
        TypeError:
            If ``class_names`` is not a tuple or list.
        ValueError:
            If fewer than two classes are provided, names are empty,
            or names are duplicated.
    """

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
    """Collect logits, probabilities, predictions, and true labels.

    Args:
        model:
            PyTorch classification model.
        data_loader:
            Evaluation DataLoader.
        device:
            PyTorch device used for inference.

    Returns:
        Collected prediction arrays and timing information.

    Raises:
        TypeError:
            If ``model`` is not a PyTorch module.
        ValueError:
            If the DataLoader produces no samples or model output
            dimensions are invalid.
    """

    if not isinstance(model, nn.Module):
        raise TypeError(
            "model must be a torch.nn.Module."
        )

    if not isinstance(data_loader, DataLoader):
        raise TypeError(
            "data_loader must be a torch.utils.data.DataLoader."
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

            if not isinstance(targets, Tensor):
                raise TypeError(
                    "DataLoader targets must be PyTorch tensors."
                )

            batch_start = time.perf_counter()

            logits = model(images)

            if selected_device.type == "cuda":
                torch.cuda.synchronize()

            batch_duration = (
                time.perf_counter() - batch_start
            )

            if logits.ndim != 2:
                raise ValueError(
                    "Model output must have shape "
                    "[batch_size, number_of_classes]."
                )

            if logits.shape[0] != targets.shape[0]:
                raise ValueError(
                    "Model outputs and targets must have "
                    "matching batch sizes."
                )

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            predicted_labels = probabilities.argmax(
                dim=1
            )

            batch_size = int(targets.shape[0])

            total_inference_time += batch_duration
            total_samples += batch_size

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
    """Calculate accuracy independently for every class.

    Per-class accuracy is calculated from each row of the confusion
    matrix:

    ``correct predictions / all true samples in that class``

    Args:
        confusion_matrix_values:
            Square confusion matrix.
        class_names:
            Ordered class names.

    Returns:
        Mapping from class name to per-class accuracy.
    """

    if not isinstance(
        confusion_matrix_values,
        np.ndarray,
    ):
        raise TypeError(
            "confusion_matrix_values must be a NumPy array."
        )

    expected_shape = (
        len(class_names),
        len(class_names),
    )

    if confusion_matrix_values.shape != expected_shape:
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
    """Calculate ROC-AUC for binary or multiclass classification.

    Binary classification uses the probability of class index 1.

    Multiclass classification uses macro-averaged one-vs-rest ROC-AUC.

    ``None`` is returned when every expected class is not represented
    or when ROC-AUC cannot be calculated safely.

    Args:
        true_labels:
            One-dimensional array of true class indices.
        probabilities:
            Two-dimensional probability array with one column per class.
        number_of_classes:
            Total expected number of classes.

    Returns:
        ROC-AUC value, or ``None`` when unavailable.
    """

    if not isinstance(true_labels, np.ndarray):
        raise TypeError(
            "true_labels must be a NumPy array."
        )

    if not isinstance(probabilities, np.ndarray):
        raise TypeError(
            "probabilities must be a NumPy array."
        )

    if true_labels.ndim != 1:
        raise ValueError(
            "true_labels must be a one-dimensional array."
        )

    if probabilities.ndim != 2:
        raise ValueError(
            "probabilities must be a two-dimensional array."
        )

    if probabilities.shape[0] != true_labels.shape[0]:
        raise ValueError(
            "Probabilities and true labels must contain "
            "the same number of samples."
        )

    if probabilities.shape[1] != number_of_classes:
        raise ValueError(
            "Probability columns must match the number of classes."
        )

    if number_of_classes < 2:
        raise ValueError(
            "number_of_classes must be at least 2."
        )

    if not np.isfinite(probabilities).all():
        raise ValueError(
            "probabilities must contain only finite values."
        )

    present_classes = np.unique(true_labels)

    if len(present_classes) != number_of_classes:
        return None

    try:
        if number_of_classes == 2:
            return float(
                roc_auc_score(
                    true_labels,
                    probabilities[:, 1],
                )
            )

        binary_targets = label_binarize(
            true_labels,
            classes=np.arange(number_of_classes),
        )

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


def validate_prediction_results(
    *,
    prediction_results: PredictionResults,
    number_of_classes: int,
) -> None:
    """Validate prediction arrays before metric calculation."""

    if not isinstance(
        prediction_results,
        PredictionResults,
    ):
        raise TypeError(
            "prediction_results must be a PredictionResults object."
        )

    if prediction_results.true_labels.ndim != 1:
        raise ValueError(
            "true_labels must be one-dimensional."
        )

    if prediction_results.predicted_labels.ndim != 1:
        raise ValueError(
            "predicted_labels must be one-dimensional."
        )

    if prediction_results.probabilities.ndim != 2:
        raise ValueError(
            "probabilities must be two-dimensional."
        )

    if prediction_results.logits.ndim != 2:
        raise ValueError(
            "logits must be two-dimensional."
        )

    sample_count = prediction_results.sample_count

    if sample_count <= 0:
        raise ValueError(
            "sample_count must be greater than zero."
        )

    if prediction_results.true_labels.shape[0] != sample_count:
        raise ValueError(
            "true_labels length does not match sample_count."
        )

    if prediction_results.predicted_labels.shape[0] != sample_count:
        raise ValueError(
            "predicted_labels length does not match sample_count."
        )

    if prediction_results.probabilities.shape != (
        sample_count,
        number_of_classes,
    ):
        raise ValueError(
            "Prediction probabilities do not match "
            "sample_count and number_of_classes."
        )

    if prediction_results.logits.shape != (
        sample_count,
        number_of_classes,
    ):
        raise ValueError(
            "Prediction logits do not match "
            "sample_count and number_of_classes."
        )

    if not np.isfinite(
        prediction_results.probabilities
    ).all():
        raise ValueError(
            "Prediction probabilities must be finite."
        )

    if not np.isfinite(
        prediction_results.logits
    ).all():
        raise ValueError(
            "Prediction logits must be finite."
        )

    if prediction_results.average_inference_time_seconds < 0:
        raise ValueError(
            "Average inference time cannot be negative."
        )


def calculate_evaluation_metrics(
    *,
    prediction_results: PredictionResults,
    class_names: tuple[str, ...] | list[str],
) -> EvaluationMetrics:
    """Calculate mandatory binary or multiclass metrics.

    Metrics include:

    - accuracy
    - macro precision
    - macro recall
    - macro F1-score
    - weighted precision
    - weighted recall
    - weighted F1-score
    - ROC-AUC where available
    - confusion matrix
    - per-class accuracy
    - classification report
    """

    validated_class_names = validate_class_names(
        class_names
    )

    number_of_classes = len(
        validated_class_names
    )

    validate_prediction_results(
        prediction_results=prediction_results,
        number_of_classes=number_of_classes,
    )

    expected_labels = np.arange(
        number_of_classes
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
    """Collect predictions and calculate all evaluation metrics.

    Args:
        model:
            Trained PyTorch model.
        data_loader:
            Validation or test DataLoader.
        class_names:
            Ordered class names matching model outputs.
        device:
            PyTorch inference device.

    Returns:
        A pair containing raw prediction results and calculated metrics.
    """

    validated_class_names = validate_class_names(
        class_names
    )

    prediction_results = collect_predictions(
        model=model,
        data_loader=data_loader,
        device=device,
    )

    metrics = calculate_evaluation_metrics(
        prediction_results=prediction_results,
        class_names=validated_class_names,
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
    "validate_prediction_results",
]
