"""Reusable PyTorch dataset and DataLoader utilities.

This module creates training, validation, and test DataLoaders from the
processed TrashNet directory structure.

Training data uses stochastic augmentation. Validation and test data use
deterministic preprocessing.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from src.augmentation import get_training_transform
from src.data_processing import get_evaluation_transform


DEFAULT_BATCH_SIZE: Final[int] = 32
DEFAULT_NUMBER_OF_WORKERS: Final[int] = 0
DEFAULT_RANDOM_SEED: Final[int] = 42


@dataclass(frozen=True)
class DataLoaderBundle:
    """Container holding datasets, loaders, and class metadata."""

    train_dataset: ImageFolder
    validation_dataset: ImageFolder
    test_dataset: ImageFolder

    train_loader: DataLoader
    validation_loader: DataLoader
    test_loader: DataLoader

    class_names: tuple[str, ...]
    class_to_idx: dict[str, int]


def validate_positive_integer(
    value: int,
    *,
    parameter_name: str,
) -> int:
    """Validate an integer that must be greater than zero.

    Args:
        value:
            Value to validate.
        parameter_name:
            Name used in the error message.

    Returns:
        The validated integer.

    Raises:
        TypeError:
            If the value is not an integer.
        ValueError:
            If the value is not greater than zero.
    """

    if (
        not isinstance(value, int)
        or isinstance(value, bool)
    ):
        raise TypeError(
            f"{parameter_name} must be an integer."
        )

    if value <= 0:
        raise ValueError(
            f"{parameter_name} must be greater than zero."
        )

    return value


def validate_non_negative_integer(
    value: int,
    *,
    parameter_name: str,
) -> int:
    """Validate an integer that must be zero or greater."""

    if (
        not isinstance(value, int)
        or isinstance(value, bool)
    ):
        raise TypeError(
            f"{parameter_name} must be an integer."
        )

    if value < 0:
        raise ValueError(
            f"{parameter_name} must be zero or greater."
        )

    return value


def validate_dataset_directory(
    directory: Path | str,
    *,
    directory_name: str,
) -> Path:
    """Validate a processed dataset directory.

    Args:
        directory:
            Directory containing class subfolders.
        directory_name:
            Human-readable name used in error messages.

    Returns:
        The validated path.

    Raises:
        TypeError:
            If the directory is not a string or Path.
        FileNotFoundError:
            If the directory does not exist.
        ValueError:
            If the path is not a directory.
    """

    if not isinstance(directory, (str, Path)):
        raise TypeError(
            f"{directory_name} must be a string or pathlib.Path."
        )

    path = Path(directory)

    if not path.exists():
        raise FileNotFoundError(
            f"{directory_name} does not exist: {path}"
        )

    if not path.is_dir():
        raise ValueError(
            f"{directory_name} must be a directory: {path}"
        )

    return path


def seed_data_loader_worker(
    worker_id: int,
) -> None:
    """Seed a DataLoader worker for reproducible random operations.

    PyTorch provides each worker with an initial seed. That seed is reduced
    to the range accepted by NumPy-style random generators.

    Args:
        worker_id:
            Worker identifier supplied automatically by PyTorch.
    """

    del worker_id

    worker_seed = torch.initial_seed() % (2**32)

    torch.manual_seed(worker_seed)


def create_imagefolder_datasets(
    train_directory: Path | str,
    validation_directory: Path | str,
    test_directory: Path | str,
) -> tuple[ImageFolder, ImageFolder, ImageFolder]:
    """Create training, validation, and test ImageFolder datasets.

    Training images receive stochastic augmentation. Validation and test
    images receive deterministic evaluation preprocessing.

    Args:
        train_directory:
            Processed training directory.
        validation_directory:
            Processed validation directory.
        test_directory:
            Processed test directory.

    Returns:
        Training, validation, and test datasets.

    Raises:
        ValueError:
            If class names or class-index mappings differ across splits.
    """

    validated_train_directory = validate_dataset_directory(
        train_directory,
        directory_name="Training directory",
    )

    validated_validation_directory = validate_dataset_directory(
        validation_directory,
        directory_name="Validation directory",
    )

    validated_test_directory = validate_dataset_directory(
        test_directory,
        directory_name="Test directory",
    )

    training_dataset = ImageFolder(
        root=validated_train_directory,
        transform=get_training_transform(),
    )

    validation_dataset = ImageFolder(
        root=validated_validation_directory,
        transform=get_evaluation_transform(),
    )

    test_dataset = ImageFolder(
        root=validated_test_directory,
        transform=get_evaluation_transform(),
    )

    if training_dataset.class_to_idx != validation_dataset.class_to_idx:
        raise ValueError(
            "Training and validation class mappings do not match."
        )

    if training_dataset.class_to_idx != test_dataset.class_to_idx:
        raise ValueError(
            "Training and test class mappings do not match."
        )

    if len(training_dataset) == 0:
        raise ValueError(
            "The training dataset contains no images."
        )

    if len(validation_dataset) == 0:
        raise ValueError(
            "The validation dataset contains no images."
        )

    if len(test_dataset) == 0:
        raise ValueError(
            "The test dataset contains no images."
        )

    return (
        training_dataset,
        validation_dataset,
        test_dataset,
    )


def create_data_loaders(
    train_directory: Path | str,
    validation_directory: Path | str,
    test_directory: Path | str,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    number_of_workers: int = DEFAULT_NUMBER_OF_WORKERS,
    random_seed: int = DEFAULT_RANDOM_SEED,
    pin_memory: bool | None = None,
) -> DataLoaderBundle:
    """Create reproducible DataLoaders for all dataset splits.

    Args:
        train_directory:
            Processed training directory.
        validation_directory:
            Processed validation directory.
        test_directory:
            Processed test directory.
        batch_size:
            Number of samples in one batch.
        number_of_workers:
            Number of worker processes used for loading data.
            Zero loads data in the main process and is the safest
            default on Windows.
        random_seed:
            Seed used for reproducible training shuffling.
        pin_memory:
            Whether DataLoaders should use pinned CPU memory.
            When omitted, it is enabled only when CUDA is available.

    Returns:
        A DataLoaderBundle containing datasets, loaders, and metadata.
    """

    validated_batch_size = validate_positive_integer(
        batch_size,
        parameter_name="batch_size",
    )

    validated_worker_count = validate_non_negative_integer(
        number_of_workers,
        parameter_name="number_of_workers",
    )

    validated_random_seed = validate_non_negative_integer(
        random_seed,
        parameter_name="random_seed",
    )

    if pin_memory is not None and not isinstance(
        pin_memory,
        bool,
    ):
        raise TypeError(
            "pin_memory must be a boolean or None."
        )

    (
        training_dataset,
        validation_dataset,
        test_dataset,
    ) = create_imagefolder_datasets(
        train_directory=train_directory,
        validation_directory=validation_directory,
        test_directory=test_directory,
    )

    use_pin_memory = (
        torch.cuda.is_available()
        if pin_memory is None
        else pin_memory
    )

    training_generator = torch.Generator()
    training_generator.manual_seed(
        validated_random_seed
    )

    common_loader_arguments = {
        "batch_size": validated_batch_size,
        "num_workers": validated_worker_count,
        "pin_memory": use_pin_memory,
    }

    if validated_worker_count > 0:
        common_loader_arguments["persistent_workers"] = True
        common_loader_arguments[
            "worker_init_fn"
        ] = seed_data_loader_worker

    training_loader = DataLoader(
        training_dataset,
        shuffle=True,
        generator=training_generator,
        drop_last=False,
        **common_loader_arguments,
    )

    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        drop_last=False,
        **common_loader_arguments,
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        drop_last=False,
        **common_loader_arguments,
    )

    return DataLoaderBundle(
        train_dataset=training_dataset,
        validation_dataset=validation_dataset,
        test_dataset=test_dataset,
        train_loader=training_loader,
        validation_loader=validation_loader,
        test_loader=test_loader,
        class_names=tuple(training_dataset.classes),
        class_to_idx=training_dataset.class_to_idx.copy(),
    )


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_NUMBER_OF_WORKERS",
    "DEFAULT_RANDOM_SEED",
    "DataLoaderBundle",
    "create_data_loaders",
    "create_imagefolder_datasets",
    "seed_data_loader_worker",
    "validate_dataset_directory",
    "validate_non_negative_integer",
    "validate_positive_integer",
]
