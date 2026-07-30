"""Deterministic image preprocessing utilities.

This module defines reusable preprocessing for validation, testing,
and production inference.

Random training augmentation must remain in ``src/augmentation.py``.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Final, TypeAlias

from PIL import Image, ImageOps, UnidentifiedImageError
from torch import Tensor
from torchvision import transforms


ImageSize: TypeAlias = tuple[int, int]
ImageTransform: TypeAlias = Callable[[Image.Image], Tensor]

DEFAULT_IMAGE_SIZE: Final[ImageSize] = (224, 224)

IMAGENET_MEAN: Final[tuple[float, float, float]] = (
    0.485,
    0.456,
    0.406,
)

IMAGENET_STD: Final[tuple[float, float, float]] = (
    0.229,
    0.224,
    0.225,
)

SUPPORTED_IMAGE_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {
        ".jpg",
        ".jpeg",
        ".png",
    }
)


def validate_image_size(image_size: ImageSize) -> ImageSize:
    """Validate an image size written as ``(height, width)``.

    Args:
        image_size:
            Requested output image size.

    Returns:
        The validated image size.

    Raises:
        TypeError:
            If the value is not a two-item tuple of integers.
        ValueError:
            If either dimension is zero or negative.
    """

    if not isinstance(image_size, tuple):
        raise TypeError(
            "image_size must be a tuple containing two integers."
        )

    if len(image_size) != 2:
        raise TypeError(
            "image_size must contain exactly two dimensions."
        )

    if not all(
        isinstance(dimension, int)
        and not isinstance(dimension, bool)
        for dimension in image_size
    ):
        raise TypeError(
            "Image height and width must both be integers."
        )

    if any(dimension <= 0 for dimension in image_size):
        raise ValueError(
            "Image height and width must be greater than zero."
        )

    return image_size


def get_evaluation_transform(
    image_size: ImageSize = DEFAULT_IMAGE_SIZE,
) -> transforms.Compose:
    """Create deterministic preprocessing for evaluation and inference.

    The pipeline performs resizing, tensor conversion, and ImageNet
    normalization. It contains no random operations.

    Args:
        image_size:
            Output size written as ``(height, width)``.

    Returns:
        A deterministic Torchvision transformation pipeline.
    """

    validated_size = validate_image_size(image_size)

    return transforms.Compose(
        [
            transforms.Resize(
                validated_size,
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )


def validate_image_path(image_path: Path | str) -> Path:
    """Validate the path and extension of an image file.

    Args:
        image_path:
            Path to a JPEG or PNG image.

    Returns:
        The validated path.

    Raises:
        TypeError:
            If ``image_path`` is not a string or ``Path``.
        FileNotFoundError:
            If the path does not exist.
        ValueError:
            If the path is not a file or has an unsupported extension.
    """

    if not isinstance(image_path, (str, Path)):
        raise TypeError(
            "image_path must be a string or pathlib.Path object."
        )

    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Image file does not exist: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Expected an image file but received: {path}"
        )

    if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        supported_formats = ", ".join(
            sorted(SUPPORTED_IMAGE_EXTENSIONS)
        )

        raise ValueError(
            f"Unsupported image format '{path.suffix}'. "
            f"Supported formats: {supported_formats}."
        )

    return path


def prepare_pil_image(image: Image.Image) -> Image.Image:
    """Prepare a Pillow image for model preprocessing.

    EXIF orientation metadata is applied before the image is converted
    to RGB. This ensures photographs are not accidentally processed
    sideways and guarantees a three-channel model input.

    Args:
        image:
            Pillow image to prepare.

    Returns:
        An EXIF-corrected RGB image.

    Raises:
        TypeError:
            If ``image`` is not a Pillow image.
    """

    if not isinstance(image, Image.Image):
        raise TypeError(
            "image must be an instance of PIL.Image.Image."
        )

    oriented_image = ImageOps.exif_transpose(image)

    return oriented_image.convert("RGB")


def preprocess_image(
    image: Image.Image,
    image_size: ImageSize = DEFAULT_IMAGE_SIZE,
    transform: ImageTransform | None = None,
) -> Tensor:
    """Convert a Pillow image into a normalized model-input tensor.

    Args:
        image:
            Pillow image to preprocess.
        image_size:
            Output size written as ``(height, width)``.
        transform:
            Optional deterministic transform. When omitted, the standard
            evaluation transform is created from ``image_size``.

    Returns:
        A normalized tensor with shape ``[3, height, width]``.
    """

    prepared_image = prepare_pil_image(image)

    selected_transform = (
        transform
        if transform is not None
        else get_evaluation_transform(image_size)
    )

    tensor = selected_transform(prepared_image)

    if not isinstance(tensor, Tensor):
        raise TypeError(
            "The image transform must return a PyTorch Tensor."
        )

    return tensor


def load_and_preprocess_image(
    image_path: Path | str,
    image_size: ImageSize = DEFAULT_IMAGE_SIZE,
    transform: ImageTransform | None = None,
) -> Tensor:
    """Load an image file and apply deterministic preprocessing.

    Args:
        image_path:
            Path to a JPEG or PNG image.
        image_size:
            Output size written as ``(height, width)``.
        transform:
            Optional deterministic transformation pipeline.

    Returns:
        A normalized tensor with shape ``[3, height, width]``.

    Raises:
        ValueError:
            If Pillow cannot identify or decode the image.
    """

    validated_path = validate_image_path(image_path)

    try:
        with Image.open(validated_path) as image:
            image.load()

            return preprocess_image(
                image=image,
                image_size=image_size,
                transform=transform,
            )

    except (UnidentifiedImageError, OSError) as error:
        raise ValueError(
            f"Unable to decode image file: {validated_path}"
        ) from error


__all__ = [
    "DEFAULT_IMAGE_SIZE",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "SUPPORTED_IMAGE_EXTENSIONS",
    "ImageSize",
    "ImageTransform",
    "get_evaluation_transform",
    "load_and_preprocess_image",
    "prepare_pil_image",
    "preprocess_image",
    "validate_image_path",
    "validate_image_size",
]