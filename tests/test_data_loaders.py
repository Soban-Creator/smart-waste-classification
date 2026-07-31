"""Tests for PyTorch dataset and DataLoader utilities."""

from pathlib import Path

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.data_loaders import (
    DataLoaderBundle,
    create_data_loaders,
    create_imagefolder_datasets,
    validate_dataset_directory,
    validate_non_negative_integer,
    validate_positive_integer,
)


def create_split_directory(
    root: Path,
    split_name: str,
    class_sizes: dict[str, int],
) -> Path:
    """Create a temporary ImageFolder-compatible split."""

    split_directory = root / split_name

    for class_name, image_count in class_sizes.items():
        class_directory = (
            split_directory / class_name
        )

        class_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        for image_index in range(image_count):
            image = Image.new(
                mode="RGB",
                size=(40, 30),
                color=(
                    30 + image_index,
                    60,
                    90,
                ),
            )

            image.save(
                class_directory
                / f"image_{image_index}.jpg"
            )

    return split_directory


def create_complete_test_splits(
    root: Path,
) -> tuple[Path, Path, Path]:
    """Create train, validation, and test folders."""

    class_sizes = {
        "cardboard": 4,
        "glass": 4,
    }

    train_directory = create_split_directory(
        root,
        "train",
        class_sizes,
    )

    validation_directory = create_split_directory(
        root,
        "validation",
        class_sizes,
    )

    test_directory = create_split_directory(
        root,
        "test",
        class_sizes,
    )

    return (
        train_directory,
        validation_directory,
        test_directory,
    )


def test_create_imagefolder_datasets_returns_three_datasets(
    tmp_path: Path,
) -> None:
    """All three dataset splits should load successfully."""

    (
        train_directory,
        validation_directory,
        test_directory,
    ) = create_complete_test_splits(tmp_path)

    (
        train_dataset,
        validation_dataset,
        test_dataset,
    ) = create_imagefolder_datasets(
        train_directory,
        validation_directory,
        test_directory,
    )

    assert len(train_dataset) == 8
    assert len(validation_dataset) == 8
    assert len(test_dataset) == 8

    assert train_dataset.classes == [
        "cardboard",
        "glass",
    ]


def test_create_data_loaders_returns_bundle(
    tmp_path: Path,
) -> None:
    """The loader factory should return a structured bundle."""

    (
        train_directory,
        validation_directory,
        test_directory,
    ) = create_complete_test_splits(tmp_path)

    bundle = create_data_loaders(
        train_directory,
        validation_directory,
        test_directory,
        batch_size=2,
    )

    assert isinstance(bundle, DataLoaderBundle)
    assert isinstance(bundle.train_loader, DataLoader)
    assert isinstance(bundle.validation_loader, DataLoader)
    assert isinstance(bundle.test_loader, DataLoader)


def test_data_loader_batch_has_expected_shape(
    tmp_path: Path,
) -> None:
    """A training batch should contain transformed images and labels."""

    (
        train_directory,
        validation_directory,
        test_directory,
    ) = create_complete_test_splits(tmp_path)

    bundle = create_data_loaders(
        train_directory,
        validation_directory,
        test_directory,
        batch_size=2,
    )

    images, labels = next(
        iter(bundle.train_loader)
    )

    assert images.shape == (
        2,
        3,
        224,
        224,
    )

    assert labels.shape == (2,)
    assert images.dtype == torch.float32
    assert labels.dtype == torch.int64


def test_class_metadata_is_consistent(
    tmp_path: Path,
) -> None:
    """The bundle should expose the ImageFolder class mapping."""

    (
        train_directory,
        validation_directory,
        test_directory,
    ) = create_complete_test_splits(tmp_path)

    bundle = create_data_loaders(
        train_directory,
        validation_directory,
        test_directory,
        batch_size=2,
    )

    assert bundle.class_names == (
        "cardboard",
        "glass",
    )

    assert bundle.class_to_idx == {
        "cardboard": 0,
        "glass": 1,
    }


def test_validation_and_test_loaders_are_not_shuffled(
    tmp_path: Path,
) -> None:
    """Evaluation loaders should preserve deterministic sample order."""

    (
        train_directory,
        validation_directory,
        test_directory,
    ) = create_complete_test_splits(tmp_path)

    bundle = create_data_loaders(
        train_directory,
        validation_directory,
        test_directory,
        batch_size=2,
    )

    first_validation_batch = next(
        iter(bundle.validation_loader)
    )

    second_validation_batch = next(
        iter(bundle.validation_loader)
    )

    first_images, first_labels = first_validation_batch
    second_images, second_labels = second_validation_batch

    assert torch.equal(
        first_images,
        second_images,
    )

    assert torch.equal(
        first_labels,
        second_labels,
    )


def test_mismatched_class_mappings_are_rejected(
    tmp_path: Path,
) -> None:
    """All dataset splits must contain the same class folders."""

    train_directory = create_split_directory(
        tmp_path,
        "train",
        {
            "cardboard": 2,
            "glass": 2,
        },
    )

    validation_directory = create_split_directory(
        tmp_path,
        "validation",
        {
            "cardboard": 2,
            "glass": 2,
        },
    )

    test_directory = create_split_directory(
        tmp_path,
        "test",
        {
            "cardboard": 2,
            "metal": 2,
        },
    )

    with pytest.raises(
        ValueError,
        match="Training and test class mappings",
    ):
        create_imagefolder_datasets(
            train_directory,
            validation_directory,
            test_directory,
        )


def test_missing_directory_is_rejected(
    tmp_path: Path,
) -> None:
    """A missing dataset directory should raise a clear error."""

    missing_directory = (
        tmp_path / "missing"
    )

    with pytest.raises(FileNotFoundError):
        validate_dataset_directory(
            missing_directory,
            directory_name="Training directory",
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        0,
        -1,
    ],
)
def test_positive_integer_validation_rejects_non_positive_values(
    invalid_value: int,
) -> None:
    """Batch size must be greater than zero."""

    with pytest.raises(ValueError):
        validate_positive_integer(
            invalid_value,
            parameter_name="batch_size",
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        1.5,
        "2",
        True,
    ],
)
def test_positive_integer_validation_rejects_invalid_types(
    invalid_value: object,
) -> None:
    """Batch size must use an integer type."""

    with pytest.raises(TypeError):
        validate_positive_integer(
            invalid_value,  # type: ignore[arg-type]
            parameter_name="batch_size",
        )


def test_non_negative_integer_accepts_zero() -> None:
    """Zero workers should be valid on Windows."""

    result = validate_non_negative_integer(
        0,
        parameter_name="number_of_workers",
    )

    assert result == 0


def test_negative_worker_count_is_rejected() -> None:
    """The worker count cannot be negative."""

    with pytest.raises(ValueError):
        validate_non_negative_integer(
            -1,
            parameter_name="number_of_workers",
        )
        