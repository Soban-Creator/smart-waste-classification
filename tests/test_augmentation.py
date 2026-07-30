"""Tests for training image augmentation."""

import torch
from PIL import Image

from src.augmentation import get_training_transform


def test_training_transform_returns_tensor() -> None:
    """The training transform should return a PyTorch tensor."""

    image = Image.new(
        mode="RGB",
        size=(100, 100),
        color=(120, 50, 30),
    )

    transform = get_training_transform()

    tensor = transform(image)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32


def test_training_transform_is_callable() -> None:
    """The training transform should be callable."""

    transform = get_training_transform()

    assert callable(transform)


def test_custom_image_size_is_supported() -> None:
    """The training transform should support custom output sizes."""

    image = Image.new(
        mode="RGB",
        size=(150, 120),
        color=(50, 80, 120),
    )

    transform = get_training_transform(
        image_size=(128, 160),
    )

    tensor = transform(image)

    assert tensor.shape == (3, 128, 160)


def test_training_transform_output_is_float_tensor() -> None:
    """The output tensor should contain floating-point values."""

    image = Image.new(
        mode="RGB",
        size=(120, 90),
        color=(200, 120, 80),
    )

    transform = get_training_transform()

    tensor = transform(image)

    assert tensor.dtype == torch.float32


def test_training_transform_contains_expected_steps() -> None:
    """The composed transform should contain the expected operations."""

    transform = get_training_transform()

    transform_names = [
        type(step).__name__
        for step in transform.transforms
    ]

    expected = [
        "RandomHorizontalFlip",
        "RandomRotation",
        "ColorJitter",
        "Resize",
        "ToTensor",
        "Normalize",
    ]

    assert transform_names == expected