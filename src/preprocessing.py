"""
Production image preprocessing and upload validation.

The preprocessing values are read from the serialized model package so that
training-time and inference-time normalization remain consistent.
"""

from __future__ import annotations

from io import BytesIO
from typing import Iterable

from PIL import Image, UnidentifiedImageError
from torchvision import transforms


SUPPORTED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
}

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}

DEFAULT_MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024
MINIMUM_IMAGE_WIDTH = 32
MINIMUM_IMAGE_HEIGHT = 32


class ImageValidationError(ValueError):
    """Raised when an uploaded image does not satisfy API requirements."""


def validate_image_bytes(
    image_bytes: bytes,
    *,
    filename: str | None = None,
    content_type: str | None = None,
    maximum_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
) -> Image.Image:
    """
    Validate and decode an uploaded JPEG or PNG image.

    Parameters
    ----------
    image_bytes:
        Raw uploaded file contents.

    filename:
        Original filename supplied by the client.

    content_type:
        MIME type supplied by the client.

    maximum_file_size_bytes:
        Maximum accepted file size.

    Returns
    -------
    PIL.Image.Image
        Valid RGB image.

    Raises
    ------
    ImageValidationError
        When the file is empty, unsupported, corrupted, too large,
        or has invalid dimensions.
    """
    if not image_bytes:
        raise ImageValidationError(
            "The uploaded file is empty."
        )

    if len(image_bytes) > maximum_file_size_bytes:
        maximum_megabytes = (
            maximum_file_size_bytes
            / (1024 ** 2)
        )

        raise ImageValidationError(
            "The uploaded image exceeds the "
            f"{maximum_megabytes:.0f} MB limit."
        )

    if content_type and (
        content_type
        not in SUPPORTED_CONTENT_TYPES
    ):
        raise ImageValidationError(
            "Unsupported image type. "
            "Only JPEG and PNG files are accepted."
        )

    if filename:
        normalized_filename = filename.lower()

        if not any(
            normalized_filename.endswith(extension)
            for extension in SUPPORTED_EXTENSIONS
        ):
            raise ImageValidationError(
                "Unsupported filename extension. "
                "Only .jpg, .jpeg, and .png are accepted."
            )

    try:
        image = Image.open(
            BytesIO(image_bytes)
        )

        # Force Pillow to decode the complete file.
        image.load()

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
    ) as error:
        raise ImageValidationError(
            "The uploaded file is not a valid "
            "or readable image."
        ) from error

    if image.format not in {
        "JPEG",
        "PNG",
    }:
        raise ImageValidationError(
            "Unsupported image format. "
            "Only JPEG and PNG images are accepted."
        )

    width, height = image.size

    if (
        width < MINIMUM_IMAGE_WIDTH
        or height < MINIMUM_IMAGE_HEIGHT
    ):
        raise ImageValidationError(
            "The image dimensions are too small. "
            f"Minimum size is "
            f"{MINIMUM_IMAGE_WIDTH} × "
            f"{MINIMUM_IMAGE_HEIGHT} pixels."
        )

    return image.convert("RGB")


def build_inference_transform(
    *,
    image_size: Iterable[int],
    normalization_mean: Iterable[float],
    normalization_std: Iterable[float],
    resize_shorter_side: int = 256,
) -> transforms.Compose:
    """
    Build deterministic production preprocessing.

    This must match the evaluation preprocessing used during training.
    """
    image_height, image_width = tuple(
        int(value)
        for value in image_size
    )

    return transforms.Compose(
        [
            transforms.Resize(
                resize_shorter_side
            ),
            transforms.CenterCrop(
                (
                    image_height,
                    image_width,
                )
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=list(
                    normalization_mean
                ),
                std=list(
                    normalization_std
                ),
            ),
        ]
    )