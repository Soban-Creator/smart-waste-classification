"""Training image augmentation utilities.

This module defines stochastic image transformations used only during
model training.

Validation, testing, and production inference must use
``src.data_processing`` instead.
"""

from typing import Final

from torchvision import transforms

from src.data_processing import (
    DEFAULT_IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    validate_image_size,
)

# ============================================================================
# Augmentation Configuration
# ============================================================================

HORIZONTAL_FLIP_PROBABILITY: Final[float] = 0.5

ROTATION_DEGREES: Final[int] = 15

COLOR_JITTER_BRIGHTNESS: Final[float] = 0.20

COLOR_JITTER_CONTRAST: Final[float] = 0.20

COLOR_JITTER_SATURATION: Final[float] = 0.20

COLOR_JITTER_HUE: Final[float] = 0.05


# ============================================================================
# Training Transformation
# ============================================================================

def get_training_transform(
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> transforms.Compose:
    """Create the stochastic preprocessing pipeline used for training.

    The training pipeline intentionally contains random image
    transformations to improve model generalization.

    Transform order:

    1. Random horizontal flip
    2. Random rotation
    3. Random color jitter
    4. Resize
    5. Convert to tensor
    6. Normalize using ImageNet statistics

    Args:
        image_size:
            Output image size as ``(height, width)``.

    Returns:
        A Torchvision Compose object containing the training
        augmentation pipeline.
    """

    validated_size = validate_image_size(image_size)

    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(
                p=HORIZONTAL_FLIP_PROBABILITY,
            ),
            transforms.RandomRotation(
                degrees=ROTATION_DEGREES,
            ),
            transforms.ColorJitter(
                brightness=COLOR_JITTER_BRIGHTNESS,
                contrast=COLOR_JITTER_CONTRAST,
                saturation=COLOR_JITTER_SATURATION,
                hue=COLOR_JITTER_HUE,
            ),
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


__all__ = [
    "get_training_transform",
]