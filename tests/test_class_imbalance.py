"""Tests for class-imbalance utilities."""

from pathlib import Path

import pytest
import torch
from PIL import Image
from torchvision.datasets import ImageFolder

from src.class_imbalance import (
    calculate_balanced_class_weights,
    get_imagefolder_class_counts,
    get_training_class_weights,
    validate_class_counts,
)


def create_imagefolder_dataset(
    root: Path,
    class_sizes: dict[str, int],
) -> Path:
    """Create a small temporary ImageFolder dataset."""

    for class_name, number_of_images in class_sizes.items():
        class_directory = root / class_name
        class_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        for image_index in range(number_of_images):
            image = Image.new(
                mode="RGB",
                size=(20, 20),
                color=(
                    20 + image_index,
                    40,
                    60,
                ),
            )

            image.save(
                class_directory
                / f"image_{image_index}.jpg"
            )

    return root


def test_validate_class_counts_accepts_positive_integers() -> None:
    """Valid class counts should be returned as a tuple."""

    result = validate_class_counts(
        [10, 20, 30]
    )

    assert result == (10, 20, 30)


@pytest.mark.parametrize(
    "invalid_counts",
    [
        [],
        [0, 10],
        [-1, 10],
    ],
)
def test_validate_class_counts_rejects_invalid_values(
    invalid_counts: list[int],
) -> None:
    """Empty and non-positive class counts should be rejected."""

    with pytest.raises(ValueError):
        validate_class_counts(invalid_counts)


@pytest.mark.parametrize(
    "invalid_counts",
    [
        "10,20",
        [10.5, 20],
        [True, 20],
    ],
)
def test_validate_class_counts_rejects_invalid_types(
    invalid_counts: object,
) -> None:
    """Non-integer class counts should be rejected."""

    with pytest.raises(TypeError):
        validate_class_counts(
            invalid_counts  # type: ignore[arg-type]
        )


def test_balanced_dataset_receives_equal_weights() -> None:
    """Equal class frequencies should produce equal weights."""

    weights = calculate_balanced_class_weights(
        [100, 100, 100]
    )

    assert torch.allclose(
        weights,
        torch.ones(3),
    )


def test_minority_class_receives_larger_weight() -> None:
    """A less frequent class should receive a larger weight."""

    weights = calculate_balanced_class_weights(
        [100, 50, 25]
    )

    assert weights[2] > weights[1] > weights[0]


def test_class_weights_have_mean_one() -> None:
    """Normalized class weights should have a mean of one."""

    weights = calculate_balanced_class_weights(
        [100, 50, 25]
    )

    assert torch.isclose(
        weights.mean(),
        torch.tensor(1.0),
    )


def test_class_weights_use_float32_by_default() -> None:
    """Weights should use the standard model-training dtype."""

    weights = calculate_balanced_class_weights(
        [100, 50]
    )

    assert weights.dtype == torch.float32


def test_imagefolder_class_counts_follow_class_order(
    tmp_path: Path,
) -> None:
    """Counts should match ImageFolder's alphabetical class order."""

    dataset_path = create_imagefolder_dataset(
        root=tmp_path / "train",
        class_sizes={
            "glass": 2,
            "cardboard": 3,
            "trash": 1,
        },
    )

    dataset = ImageFolder(
        root=dataset_path,
    )

    counts = get_imagefolder_class_counts(dataset)

    assert dataset.classes == [
        "cardboard",
        "glass",
        "trash",
    ]

    assert counts == (
        3,
        2,
        1,
    )


def test_get_training_class_weights_returns_mapping(
    tmp_path: Path,
) -> None:
    """Directory-based calculation should return weights and mapping."""

    dataset_path = create_imagefolder_dataset(
        root=tmp_path / "train",
        class_sizes={
            "glass": 2,
            "cardboard": 4,
        },
    )

    weights, class_to_idx = get_training_class_weights(
        dataset_path
    )

    assert weights.shape == (2,)
    assert class_to_idx == {
        "cardboard": 0,
        "glass": 1,
    }

    assert weights[1] > weights[0]


def test_get_training_class_weights_rejects_missing_directory(
    tmp_path: Path,
) -> None:
    """A missing training directory should be rejected."""

    with pytest.raises(FileNotFoundError):
        get_training_class_weights(
            tmp_path / "missing"
        )
        