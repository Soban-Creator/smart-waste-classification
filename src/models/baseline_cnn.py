"""Baseline convolutional neural network for TrashNet classification.

The model is intentionally small and trained from scratch. Its purpose is
to provide a minimum performance benchmark before transfer-learning models
are evaluated.
"""

from typing import Final

import torch
from torch import Tensor, nn


DEFAULT_NUMBER_OF_CLASSES: Final[int] = 6
DEFAULT_DROPOUT_PROBABILITY: Final[float] = 0.30


def validate_number_of_classes(
    number_of_classes: int,
) -> int:
    """Validate the requested number of output classes.

    Args:
        number_of_classes:
            Number of classification categories.

    Returns:
        The validated class count.

    Raises:
        TypeError:
            If the class count is not an integer.
        ValueError:
            If fewer than two classes are requested.
    """

    if (
        not isinstance(number_of_classes, int)
        or isinstance(number_of_classes, bool)
    ):
        raise TypeError(
            "number_of_classes must be an integer."
        )

    if number_of_classes < 2:
        raise ValueError(
            "number_of_classes must be at least 2."
        )

    return number_of_classes


def validate_dropout_probability(
    dropout_probability: float,
) -> float:
    """Validate a dropout probability.

    Args:
        dropout_probability:
            Probability of setting a classifier feature to zero.

    Returns:
        The validated probability.

    Raises:
        TypeError:
            If the probability is not numeric.
        ValueError:
            If the probability is outside the interval [0, 1).
    """

    if (
        not isinstance(dropout_probability, (int, float))
        or isinstance(dropout_probability, bool)
    ):
        raise TypeError(
            "dropout_probability must be numeric."
        )

    probability = float(dropout_probability)

    if not 0.0 <= probability < 1.0:
        raise ValueError(
            "dropout_probability must satisfy "
            "0.0 <= probability < 1.0."
        )

    return probability


class ConvolutionBlock(nn.Module):
    """Convolution, batch normalization, activation, and pooling block."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
    ) -> None:
        """Initialize a convolutional feature-extraction block."""

        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),
        )

    def forward(
        self,
        inputs: Tensor,
    ) -> Tensor:
        """Extract features from an input tensor."""

        return self.layers(inputs)


class BaselineCNN(nn.Module):
    """Small CNN trained from scratch for waste classification."""

    def __init__(
        self,
        number_of_classes: int = DEFAULT_NUMBER_OF_CLASSES,
        dropout_probability: float = DEFAULT_DROPOUT_PROBABILITY,
    ) -> None:
        """Initialize the baseline CNN.

        Args:
            number_of_classes:
                Number of output waste categories.
            dropout_probability:
                Dropout probability used before the final classifier.
        """

        super().__init__()

        validated_class_count = validate_number_of_classes(
            number_of_classes
        )

        validated_dropout = validate_dropout_probability(
            dropout_probability
        )

        self.number_of_classes = validated_class_count

        self.features = nn.Sequential(
            ConvolutionBlock(
                input_channels=3,
                output_channels=32,
            ),
            ConvolutionBlock(
                input_channels=32,
                output_channels=64,
            ),
            ConvolutionBlock(
                input_channels=64,
                output_channels=128,
            ),
        )

        self.global_average_pool = nn.AdaptiveAvgPool2d(
            output_size=(1, 1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(
                p=validated_dropout
            ),
            nn.Linear(
                in_features=128,
                out_features=validated_class_count,
            ),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize trainable layers using standard CNN practices."""

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Linear):
                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=0.01,
                )
                nn.init.zeros_(module.bias)

    def forward(
        self,
        inputs: Tensor,
    ) -> Tensor:
        """Return raw class scores for a batch of images.

        Args:
            inputs:
                Image tensor with shape
                ``[batch_size, 3, height, width]``.

        Returns:
            Raw logits with shape
            ``[batch_size, number_of_classes]``.

        Raises:
            ValueError:
                If the input is not a four-dimensional RGB batch.
        """

        if inputs.ndim != 4:
            raise ValueError(
                "Expected a 4D tensor with shape "
                "[batch_size, channels, height, width]."
            )

        if inputs.shape[1] != 3:
            raise ValueError(
                "Expected RGB input with exactly 3 channels."
            )

        features = self.features(inputs)
        pooled_features = self.global_average_pool(features)
        logits = self.classifier(pooled_features)

        return logits


def count_trainable_parameters(
    model: nn.Module,
) -> int:
    """Return the number of trainable model parameters."""

    if not isinstance(model, nn.Module):
        raise TypeError(
            "model must be an instance of torch.nn.Module."
        )

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def create_baseline_cnn(
    number_of_classes: int = DEFAULT_NUMBER_OF_CLASSES,
    dropout_probability: float = DEFAULT_DROPOUT_PROBABILITY,
) -> BaselineCNN:
    """Create a configured baseline CNN instance."""

    return BaselineCNN(
        number_of_classes=number_of_classes,
        dropout_probability=dropout_probability,
    )


__all__ = [
    "BaselineCNN",
    "ConvolutionBlock",
    "DEFAULT_DROPOUT_PROBABILITY",
    "DEFAULT_NUMBER_OF_CLASSES",
    "count_trainable_parameters",
    "create_baseline_cnn",
    "validate_dropout_probability",
    "validate_number_of_classes",
]