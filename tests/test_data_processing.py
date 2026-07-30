"""Tests for reusable image preprocessing utilities."""

from pathlib import Path

import pytest
import torch
from PIL import Image
from torchvision import transforms

from src.data_processing import (
    DEFAULT_IMAGE_SIZE,
    get_evaluation_transform,
    load_and_preprocess_image,
    prepare_pil_image,
    preprocess_image,
    validate_image_path,
    validate_image_size,
)


def create_test_image(
    path: Path,
    mode: str = "RGB",
    size: tuple[int, int] = (80, 60),
) -> Path:
    """Create and save a small temporary image for testing."""

    if mode == "RGBA":
        color = (120, 80, 40, 180)
    elif mode == "L":
        color = 128
    else:
        color = (120, 80, 40)

    image = Image.new(
        mode=mode,
        size=size,
        color=color,
    )

    image.save(path)

    return path


def test_default_image_size_is_valid() -> None:
    """The default model input size should pass validation."""

    result = validate_image_size(DEFAULT_IMAGE_SIZE)

    assert result == (224, 224)


@pytest.mark.parametrize(
    "invalid_size",
    [
        [224, 224],
        (224,),
        (224, 224, 224),
        ("224", 224),
        (224.0, 224),
        (True, 224),
    ],
)
def test_validate_image_size_rejects_invalid_types(
    invalid_size: object,
) -> None:
    """Malformed image-size configurations should be rejected."""

    with pytest.raises(TypeError):
        validate_image_size(invalid_size)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "invalid_size",
    [
        (0, 224),
        (224, 0),
        (-1, 224),
        (224, -1),
    ],
)
def test_validate_image_size_rejects_non_positive_values(
    invalid_size: tuple[int, int],
) -> None:
    """Zero and negative dimensions should be rejected."""

    with pytest.raises(ValueError):
        validate_image_size(invalid_size)


def test_evaluation_transform_is_deterministic() -> None:
    """The same image should produce the same evaluation tensor."""

    image = Image.new(
        mode="RGB",
        size=(100, 70),
        color=(100, 150, 200),
    )

    transform = get_evaluation_transform()

    first_tensor = transform(image)
    second_tensor = transform(image)

    assert torch.equal(
        first_tensor,
        second_tensor,
    )


def test_preprocess_image_returns_expected_shape() -> None:
    """An RGB image should become a normalized model-input tensor."""

    image = Image.new(
        mode="RGB",
        size=(100, 70),
        color=(100, 150, 200),
    )

    tensor = preprocess_image(image)

    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32


@pytest.mark.parametrize(
    "mode",
    [
        "RGB",
        "RGBA",
        "L",
    ],
)
def test_prepare_pil_image_converts_supported_modes_to_rgb(
    mode: str,
) -> None:
    """Different Pillow image modes should become RGB images."""

    if mode == "RGBA":
        color = (10, 20, 30, 128)
    elif mode == "L":
        color = 128
    else:
        color = (10, 20, 30)

    image = Image.new(
        mode=mode,
        size=(20, 20),
        color=color,
    )

    prepared_image = prepare_pil_image(image)

    assert prepared_image.mode == "RGB"


def test_preprocess_image_rejects_non_image_input() -> None:
    """Non-Pillow values should not be accepted."""

    with pytest.raises(TypeError):
        preprocess_image("not an image")  # type: ignore[arg-type]


def test_validate_image_path_accepts_jpeg(
    tmp_path: Path,
) -> None:
    """An existing JPEG image should pass path validation."""

    image_path = create_test_image(
        tmp_path / "sample.jpg"
    )

    result = validate_image_path(image_path)

    assert result == image_path


def test_validate_image_path_accepts_uppercase_extension(
    tmp_path: Path,
) -> None:
    """Extension validation should be case-insensitive."""

    image_path = create_test_image(
        tmp_path / "sample.JPG"
    )

    result = validate_image_path(image_path)

    assert result == image_path


def test_validate_image_path_rejects_missing_file(
    tmp_path: Path,
) -> None:
    """A nonexistent file should raise FileNotFoundError."""

    missing_path = tmp_path / "missing.jpg"

    with pytest.raises(FileNotFoundError):
        validate_image_path(missing_path)


def test_validate_image_path_rejects_directory(
    tmp_path: Path,
) -> None:
    """A directory should not be accepted as an image file."""

    with pytest.raises(ValueError):
        validate_image_path(tmp_path)


def test_validate_image_path_rejects_unsupported_extension(
    tmp_path: Path,
) -> None:
    """Unsupported file extensions should be rejected."""

    text_path = tmp_path / "notes.txt"
    text_path.write_text(
        "This is not an image.",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        validate_image_path(text_path)


def test_load_and_preprocess_image_returns_expected_tensor(
    tmp_path: Path,
) -> None:
    """A valid image file should load and preprocess successfully."""

    image_path = create_test_image(
        tmp_path / "sample.png"
    )

    tensor = load_and_preprocess_image(image_path)

    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32


def test_load_and_preprocess_image_rejects_corrupted_image(
    tmp_path: Path,
) -> None:
    """A corrupted file with an image extension should be rejected."""

    corrupted_path = tmp_path / "corrupted.jpg"
    corrupted_path.write_bytes(
        b"This is not valid JPEG data."
    )

    with pytest.raises(
        ValueError,
        match="Unable to decode image file",
    ):
        load_and_preprocess_image(corrupted_path)


def test_custom_image_size_is_applied() -> None:
    """A caller should be able to request another valid output size."""

    image = Image.new(
        mode="RGB",
        size=(90, 60),
        color=(30, 60, 90),
    )

    tensor = preprocess_image(
        image=image,
        image_size=(128, 160),
    )

    assert tensor.shape == (3, 128, 160)


def test_preprocess_image_accepts_reusable_transform() -> None:
    """A prebuilt transform should be reusable across calls."""

    image = Image.new(
        mode="RGB",
        size=(90, 60),
        color=(30, 60, 90),
    )

    transform = transforms.Compose(
        [
            transforms.Resize(
                (64, 64),
                antialias=True,
            ),
            transforms.ToTensor(),
        ]
    )

    tensor = preprocess_image(
        image=image,
        transform=transform,
    )

    assert tensor.shape == (3, 64, 64)