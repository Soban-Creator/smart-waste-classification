"""Tests for the baseline convolutional neural network."""

import pytest
import torch
from torch import nn

from src.models.baseline_cnn import (
    BaselineCNN,
    ConvolutionBlock,
    count_trainable_parameters,
    create_baseline_cnn,
    validate_dropout_probability,
    validate_number_of_classes,
)


def test_baseline_cnn_is_pytorch_module() -> None:
    """The baseline model should inherit from PyTorch Module."""

    model = create_baseline_cnn()

    assert isinstance(model, nn.Module)
    assert isinstance(model, BaselineCNN)


def test_baseline_cnn_returns_expected_output_shape() -> None:
    """A batch should produce one logit vector per image."""

    model = create_baseline_cnn()

    inputs = torch.randn(
        4,
        3,
        224,
        224,
    )

    outputs = model(inputs)

    assert outputs.shape == (4, 6)
    assert outputs.dtype == torch.float32


def test_baseline_cnn_supports_different_batch_sizes() -> None:
    """The architecture should not depend on a fixed batch size."""

    model = create_baseline_cnn()

    for batch_size in [1, 2, 8]:
        inputs = torch.randn(
            batch_size,
            3,
            224,
            224,
        )

        outputs = model(inputs)

        assert outputs.shape == (
            batch_size,
            6,
        )


def test_baseline_cnn_supports_valid_spatial_sizes() -> None:
    """Adaptive pooling should support different valid input sizes."""

    model = create_baseline_cnn()

    inputs = torch.randn(
        2,
        3,
        128,
        160,
    )

    outputs = model(inputs)

    assert outputs.shape == (2, 6)


def test_baseline_cnn_supports_custom_class_count() -> None:
    """The classifier should support another valid class count."""

    model = create_baseline_cnn(
        number_of_classes=10
    )

    inputs = torch.randn(
        2,
        3,
        224,
        224,
    )

    outputs = model(inputs)

    assert outputs.shape == (2, 10)
    assert model.number_of_classes == 10


def test_baseline_cnn_rejects_non_batch_input() -> None:
    """A single unbatched CHW tensor should be rejected."""

    model = create_baseline_cnn()

    invalid_inputs = torch.randn(
        3,
        224,
        224,
    )

    with pytest.raises(
        ValueError,
        match="Expected a 4D tensor",
    ):
        model(invalid_inputs)


def test_baseline_cnn_rejects_non_rgb_input() -> None:
    """A batch without three channels should be rejected."""

    model = create_baseline_cnn()

    grayscale_batch = torch.randn(
        2,
        1,
        224,
        224,
    )

    with pytest.raises(
        ValueError,
        match="exactly 3 channels",
    ):
        model(grayscale_batch)


def test_model_has_trainable_parameters() -> None:
    """The baseline CNN should contain trainable parameters."""

    model = create_baseline_cnn()

    parameter_count = count_trainable_parameters(
        model
    )

    assert parameter_count > 0


def test_parameter_count_is_reasonably_small() -> None:
    """The baseline should remain a compact model."""

    model = create_baseline_cnn()

    parameter_count = count_trainable_parameters(
        model
    )

    assert parameter_count < 1_000_000


def test_convolution_block_reduces_spatial_dimensions() -> None:
    """Pooling should halve width and height."""

    block = ConvolutionBlock(
        input_channels=3,
        output_channels=32,
    )

    inputs = torch.randn(
        2,
        3,
        224,
        224,
    )

    outputs = block(inputs)

    assert outputs.shape == (
        2,
        32,
        112,
        112,
    )


@pytest.mark.parametrize(
    "invalid_class_count",
    [
        0,
        1,
        -1,
    ],
)
def test_validate_number_of_classes_rejects_small_values(
    invalid_class_count: int,
) -> None:
    """Classification requires at least two categories."""

    with pytest.raises(ValueError):
        validate_number_of_classes(
            invalid_class_count
        )


@pytest.mark.parametrize(
    "invalid_class_count",
    [
        6.0,
        "6",
        True,
    ],
)
def test_validate_number_of_classes_rejects_invalid_types(
    invalid_class_count: object,
) -> None:
    """The class count must be an integer."""

    with pytest.raises(TypeError):
        validate_number_of_classes(
            invalid_class_count  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "invalid_probability",
    [
        -0.1,
        1.0,
        1.5,
    ],
)
def test_validate_dropout_rejects_invalid_range(
    invalid_probability: float,
) -> None:
    """Dropout must be at least zero and lower than one."""

    with pytest.raises(ValueError):
        validate_dropout_probability(
            invalid_probability
        )


def test_dropout_is_active_only_during_training() -> None:
    """Dropout should behave differently in train and evaluation modes."""

    model = create_baseline_cnn(
        dropout_probability=0.5
    )

    inputs = torch.randn(
        2,
        3,
        224,
        224,
    )

    model.eval()

    with torch.no_grad():
        first_output = model(inputs)
        second_output = model(inputs)

    assert torch.equal(
        first_output,
        second_output,
    )
    