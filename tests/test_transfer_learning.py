"""Tests for the transfer-learning model factory."""

import pytest
import torch
from torch import nn

from src.models.transfer_learning import (
    SUPPORTED_MODEL_NAMES,
    count_parameters,
    create_transfer_learning_model,
    get_trainable_parameter_names,
    validate_dropout_probability,
    validate_model_name,
    validate_number_of_classes,
    validate_training_mode,
)


@pytest.mark.parametrize(
    "model_name",
    SUPPORTED_MODEL_NAMES,
)
def test_all_required_models_can_be_created(
    model_name: str,
) -> None:
    """Every manual-required model should be constructible."""

    model = create_transfer_learning_model(
        model_name=model_name,
        number_of_classes=6,
        pretrained=False,
        training_mode="classifier",
    )

    assert isinstance(model, nn.Module)


@pytest.mark.parametrize(
    "model_name",
    SUPPORTED_MODEL_NAMES,
)
def test_all_models_return_six_logits(
    model_name: str,
) -> None:
    """Each architecture should produce six class scores."""

    model = create_transfer_learning_model(
        model_name=model_name,
        number_of_classes=6,
        pretrained=False,
        training_mode="classifier",
    )

    model.eval()

    inputs = torch.randn(
        1,
        3,
        64,
        64,
    )

    with torch.no_grad():
        outputs = model(inputs)

    assert outputs.shape == (1, 6)


@pytest.mark.parametrize(
    "model_name",
    SUPPORTED_MODEL_NAMES,
)
def test_classifier_mode_freezes_most_parameters(
    model_name: str,
) -> None:
    """Classifier mode should leave only the head trainable."""

    model = create_transfer_learning_model(
        model_name=model_name,
        pretrained=False,
        training_mode="classifier",
    )

    total, trainable = count_parameters(model)

    assert total > 0
    assert trainable > 0
    assert trainable < total


@pytest.mark.parametrize(
    "model_name",
    SUPPORTED_MODEL_NAMES,
)
def test_last_block_mode_trains_more_parameters(
    model_name: str,
) -> None:
    """Fine-tuning should enable more parameters than head training."""

    classifier_model = create_transfer_learning_model(
        model_name=model_name,
        pretrained=False,
        training_mode="classifier",
    )

    fine_tuning_model = create_transfer_learning_model(
        model_name=model_name,
        pretrained=False,
        training_mode="last_block",
    )

    _, classifier_trainable = count_parameters(
        classifier_model
    )

    _, fine_tuning_trainable = count_parameters(
        fine_tuning_model
    )

    assert fine_tuning_trainable > classifier_trainable


@pytest.mark.parametrize(
    "model_name",
    SUPPORTED_MODEL_NAMES,
)
def test_full_mode_makes_every_parameter_trainable(
    model_name: str,
) -> None:
    """Full mode should enable every parameter."""

    model = create_transfer_learning_model(
        model_name=model_name,
        pretrained=False,
        training_mode="full",
    )

    total, trainable = count_parameters(model)

    assert trainable == total


def test_resnet_classifier_names_are_trainable() -> None:
    """ResNet classifier mode should train only fc parameters."""

    model = create_transfer_learning_model(
        model_name="resnet50",
        pretrained=False,
        training_mode="classifier",
    )

    trainable_names = get_trainable_parameter_names(
        model
    )

    assert trainable_names
    assert all(
        name.startswith("fc.")
        for name in trainable_names
    )


@pytest.mark.parametrize(
    "alias, expected",
    [
        ("resnet50", "resnet50"),
        ("efficientnet-b0", "efficientnet_b0"),
        ("DenseNet121", "densenet121"),
        ("mobilenet_v3_large", "mobilenet_v3"),
    ],
)
def test_model_name_aliases(
    alias: str,
    expected: str,
) -> None:
    """Common model-name spellings should be normalized."""

    assert validate_model_name(alias) == expected


def test_invalid_model_name_is_rejected() -> None:
    """Architectures outside the approved factory should fail."""

    with pytest.raises(ValueError):
        validate_model_name("resnet18")


@pytest.mark.parametrize(
    "invalid_mode",
    [
        "frozen",
        "partial",
        "unknown",
    ],
)
def test_invalid_training_mode_is_rejected(
    invalid_mode: str,
) -> None:
    """Unknown training modes should fail clearly."""

    with pytest.raises(ValueError):
        validate_training_mode(invalid_mode)


@pytest.mark.parametrize(
    "invalid_count",
    [
        0,
        1,
        -1,
    ],
)
def test_invalid_class_counts_are_rejected(
    invalid_count: int,
) -> None:
    """Classification requires at least two classes."""

    with pytest.raises(ValueError):
        validate_number_of_classes(
            invalid_count
        )


@pytest.mark.parametrize(
    "invalid_probability",
    [
        -0.1,
        1.0,
        1.2,
    ],
)
def test_invalid_dropout_is_rejected(
    invalid_probability: float,
) -> None:
    """Dropout must be within the supported interval."""

    with pytest.raises(ValueError):
        validate_dropout_probability(
            invalid_probability
        )
        