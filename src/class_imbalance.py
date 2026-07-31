"""Class-imbalance utilities for multiclass image classification.

This module calculates class counts and inverse-frequency class weights
for use with PyTorch's weighted cross-entropy loss.

Class weights must be calculated from the training split only. Validation
and test labels must never influence training decisions.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Final

import torch
from torch import Tensor
from torchvision.datasets import ImageFolder


WEIGHT_NORMALIZATION_EPSILON: Final[float] = 1e-12


def validate_class_counts(
    class_counts: Sequence[int],
) -> tuple[int, ...]:
    """Validate class-frequency counts.

    Args:
        class_counts:
            Number of training samples belonging to each class.

    Returns:
        The validated counts as an immutable tuple.

    Raises:
        TypeError:
            If the input is not a sequence of integers.
        ValueError:
            If the sequence is empty or contains non-positive counts.
    """

    if isinstance(class_counts, (str, bytes)):
        raise TypeError(
            "class_counts must be a sequence of integers."
        )

    try:
        counts = tuple(class_counts)
    except TypeError as error:
        raise TypeError(
            "class_counts must be a sequence of integers."
        ) from error

    if not counts:
        raise ValueError(
            "class_counts must contain at least one class."
        )

    if not all(
        isinstance(count, int)
        and not isinstance(count, bool)
        for count in counts
    ):
        raise TypeError(
            "Every class count must be an integer."
        )

    if any(count <= 0 for count in counts):
        raise ValueError(
            "Every class count must be greater than zero."
        )

    return counts


def calculate_balanced_class_weights(
    class_counts: Sequence[int],
    *,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Calculate normalized inverse-frequency class weights.

    The unnormalized weight for class ``i`` is:

    ``total_samples / (number_of_classes * class_count_i)``

    The resulting weights are normalized so their mean equals one.
    This keeps the general scale of the loss stable while assigning
    larger penalties to underrepresented classes.

    Args:
        class_counts:
            Number of training samples in each class, ordered according
            to the dataset's class-index mapping.
        dtype:
            Desired PyTorch floating-point dtype.

    Returns:
        One weight per class as a one-dimensional tensor.
    """

    counts = validate_class_counts(class_counts)

    count_tensor = torch.tensor(
        counts,
        dtype=torch.float64,
    )

    total_samples = count_tensor.sum()
    number_of_classes = count_tensor.numel()

    weights = total_samples / (
        number_of_classes * count_tensor
    )

    mean_weight = weights.mean().clamp_min(
        WEIGHT_NORMALIZATION_EPSILON
    )

    normalized_weights = weights / mean_weight

    return normalized_weights.to(dtype=dtype)


def get_imagefolder_class_counts(
    dataset: ImageFolder,
) -> tuple[int, ...]:
    """Count samples for each class in a Torchvision ImageFolder.

    Args:
        dataset:
            Loaded training dataset.

    Returns:
        Counts ordered according to ``dataset.class_to_idx``.

    Raises:
        TypeError:
            If the dataset is not an ImageFolder.
        ValueError:
            If the dataset has no classes or samples.
    """

    if not isinstance(dataset, ImageFolder):
        raise TypeError(
            "dataset must be a torchvision.datasets.ImageFolder."
        )

    number_of_classes = len(dataset.classes)

    if number_of_classes == 0:
        raise ValueError(
            "The dataset does not contain any classes."
        )

    if len(dataset.targets) == 0:
        raise ValueError(
            "The dataset does not contain any samples."
        )

    counts = torch.bincount(
        torch.tensor(
            dataset.targets,
            dtype=torch.long,
        ),
        minlength=number_of_classes,
    )

    return tuple(
        int(count)
        for count in counts.tolist()
    )


def get_training_class_weights(
    train_directory: Path | str,
) -> tuple[Tensor, dict[str, int]]:
    """Calculate class weights from a processed training directory.

    Args:
        train_directory:
            Directory arranged in ImageFolder format:

            ``train/class_name/image.jpg``

    Returns:
        A tuple containing:

        1. class-weight tensor ordered by class index
        2. class-to-index mapping

    Raises:
        TypeError:
            If the path is not a string or Path.
        FileNotFoundError:
            If the directory does not exist.
        ValueError:
            If the path is not a directory or contains no valid classes.
    """

    if not isinstance(train_directory, (str, Path)):
        raise TypeError(
            "train_directory must be a string or pathlib.Path."
        )

    directory = Path(train_directory)

    if not directory.exists():
        raise FileNotFoundError(
            f"Training directory does not exist: {directory}"
        )

    if not directory.is_dir():
        raise ValueError(
            f"Expected a training directory but received: {directory}"
        )

    dataset = ImageFolder(
        root=directory,
    )

    class_counts = get_imagefolder_class_counts(dataset)
    class_weights = calculate_balanced_class_weights(
        class_counts
    )

    return class_weights, dataset.class_to_idx.copy()


__all__ = [
    "calculate_balanced_class_weights",
    "get_imagefolder_class_counts",
    "get_training_class_weights",
    "validate_class_counts",
]
