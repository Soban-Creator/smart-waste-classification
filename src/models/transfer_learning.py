"""Transfer-learning model factory for waste classification.

This module creates and configures the four pretrained architectures
required by the client:

- ResNet50
- EfficientNet-B0
- DenseNet121
- MobileNetV3-Large

The factory supports three training modes:

- ``classifier``: train only the replacement classifier
- ``last_block``: train the final feature block and classifier
- ``full``: train the complete model
"""

from __future__ import annotations

from typing import Final, Literal, TypeAlias

from torch import Tensor, nn
from torchvision.models import (
    DenseNet121_Weights,
    EfficientNet_B0_Weights,
    MobileNet_V3_Large_Weights,
    ResNet50_Weights,
    densenet121,
    efficientnet_b0,
    mobilenet_v3_large,
    resnet50,
)


ModelName: TypeAlias = Literal[
    "resnet50",
    "efficientnet_b0",
    "densenet121",
    "mobilenet_v3",
]

TrainingMode: TypeAlias = Literal[
    "classifier",
    "last_block",
    "full",
]


SUPPORTED_MODEL_NAMES: Final[tuple[str, ...]] = (
    "resnet50",
    "efficientnet_b0",
    "densenet121",
    "mobilenet_v3",
)

SUPPORTED_TRAINING_MODES: Final[tuple[str, ...]] = (
    "classifier",
    "last_block",
    "full",
)

DEFAULT_NUMBER_OF_CLASSES: Final[int] = 6
DEFAULT_DROPOUT_PROBABILITY: Final[float] = 0.20


def validate_model_name(
    model_name: str,
) -> ModelName:
    """Validate and normalize a transfer-learning model name."""

    if not isinstance(model_name, str):
        raise TypeError(
            "model_name must be a string."
        )

    normalized_name = (
        model_name
        .strip()
        .lower()
        .replace("-", "_")
    )

    aliases = {
        "efficientnetb0": "efficientnet_b0",
        "efficientnet_b0": "efficientnet_b0",
        "mobilenetv3": "mobilenet_v3",
        "mobilenet_v3_large": "mobilenet_v3",
        "mobilenet_v3": "mobilenet_v3",
        "resnet50": "resnet50",
        "densenet121": "densenet121",
    }

    resolved_name = aliases.get(normalized_name)

    if resolved_name not in SUPPORTED_MODEL_NAMES:
        supported = ", ".join(
            SUPPORTED_MODEL_NAMES
        )

        raise ValueError(
            f"Unsupported model '{model_name}'. "
            f"Supported models: {supported}."
        )

    return resolved_name  # type: ignore[return-value]


def validate_training_mode(
    training_mode: str,
) -> TrainingMode:
    """Validate a transfer-learning training mode."""

    if not isinstance(training_mode, str):
        raise TypeError(
            "training_mode must be a string."
        )

    normalized_mode = (
        training_mode
        .strip()
        .lower()
        .replace("-", "_")
    )

    if normalized_mode not in SUPPORTED_TRAINING_MODES:
        supported = ", ".join(
            SUPPORTED_TRAINING_MODES
        )

        raise ValueError(
            f"Unsupported training mode '{training_mode}'. "
            f"Supported modes: {supported}."
        )

    return normalized_mode  # type: ignore[return-value]


def validate_number_of_classes(
    number_of_classes: int,
) -> int:
    """Validate the number of classification categories."""

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
    """Validate a dropout probability in the interval [0, 1)."""

    if (
        not isinstance(dropout_probability, (int, float))
        or isinstance(dropout_probability, bool)
    ):
        raise TypeError(
            "dropout_probability must be numeric."
        )

    probability = float(
        dropout_probability
    )

    if not 0.0 <= probability < 1.0:
        raise ValueError(
            "dropout_probability must satisfy "
            "0.0 <= probability < 1.0."
        )

    return probability


def freeze_all_parameters(
    model: nn.Module,
) -> None:
    """Disable gradient calculation for every model parameter."""

    for parameter in model.parameters():
        parameter.requires_grad = False


def unfreeze_module(
    module: nn.Module,
) -> None:
    """Enable gradient calculation for every parameter in a module."""

    for parameter in module.parameters():
        parameter.requires_grad = True


def get_classifier_module(
    model: nn.Module,
    model_name: ModelName,
) -> nn.Module:
    """Return the replacement classification head."""

    if model_name == "resnet50":
        return model.fc  # type: ignore[attr-defined]

    if model_name == "efficientnet_b0":
        return model.classifier  # type: ignore[attr-defined]

    if model_name == "densenet121":
        return model.classifier  # type: ignore[attr-defined]

    if model_name == "mobilenet_v3":
        return model.classifier  # type: ignore[attr-defined]

    raise ValueError(
        f"Unsupported model name: {model_name}"
    )


def replace_classifier(
    *,
    model: nn.Module,
    model_name: ModelName,
    number_of_classes: int,
    dropout_probability: float,
) -> None:
    """Replace the ImageNet classifier with a TrashNet classifier."""

    if model_name == "resnet50":
        input_features = model.fc.in_features  # type: ignore[attr-defined]

        model.fc = nn.Sequential(  # type: ignore[attr-defined]
            nn.Dropout(
                p=dropout_probability
            ),
            nn.Linear(
                input_features,
                number_of_classes,
            ),
        )

        return

    if model_name == "efficientnet_b0":
        input_features = (
            model.classifier[1].in_features  # type: ignore[attr-defined]
        )

        model.classifier[0] = nn.Dropout(  # type: ignore[attr-defined]
            p=dropout_probability,
            inplace=True,
        )

        model.classifier[1] = nn.Linear(  # type: ignore[attr-defined]
            input_features,
            number_of_classes,
        )

        return

    if model_name == "densenet121":
        input_features = (
            model.classifier.in_features  # type: ignore[attr-defined]
        )

        model.classifier = nn.Sequential(  # type: ignore[attr-defined]
            nn.Dropout(
                p=dropout_probability
            ),
            nn.Linear(
                input_features,
                number_of_classes,
            ),
        )

        return

    if model_name == "mobilenet_v3":
        input_features = (
            model.classifier[3].in_features  # type: ignore[attr-defined]
        )

        model.classifier[2] = nn.Dropout(  # type: ignore[attr-defined]
            p=dropout_probability,
            inplace=True,
        )

        model.classifier[3] = nn.Linear(  # type: ignore[attr-defined]
            input_features,
            number_of_classes,
        )

        return

    raise ValueError(
        f"Unsupported model name: {model_name}"
    )


def unfreeze_last_feature_block(
    model: nn.Module,
    model_name: ModelName,
) -> None:
    """Unfreeze the final feature-extraction stage."""

    if model_name == "resnet50":
        unfreeze_module(
            model.layer4  # type: ignore[attr-defined]
        )
        return

    if model_name == "efficientnet_b0":
        feature_blocks = model.features  # type: ignore[attr-defined]

        unfreeze_module(feature_blocks[-2])
        unfreeze_module(feature_blocks[-1])
        return

    if model_name == "densenet121":
        features = model.features  # type: ignore[attr-defined]

        unfreeze_module(
            features.denseblock4
        )
        unfreeze_module(
            features.norm5
        )
        return

    if model_name == "mobilenet_v3":
        feature_blocks = model.features  # type: ignore[attr-defined]

        unfreeze_module(feature_blocks[-2])
        unfreeze_module(feature_blocks[-1])
        return

    raise ValueError(
        f"Unsupported model name: {model_name}"
    )


def configure_trainable_layers(
    *,
    model: nn.Module,
    model_name: ModelName,
    training_mode: TrainingMode,
) -> None:
    """Configure frozen and trainable layers.

    Modes:

    ``classifier``
        Freeze the feature extractor and train only the classifier.

    ``last_block``
        Train the final feature block and classifier.

    ``full``
        Train every model parameter.
    """

    if training_mode == "full":
        for parameter in model.parameters():
            parameter.requires_grad = True
        return

    freeze_all_parameters(model)

    classifier = get_classifier_module(
        model,
        model_name,
    )

    unfreeze_module(classifier)

    if training_mode == "last_block":
        unfreeze_last_feature_block(
            model,
            model_name,
        )


def create_transfer_learning_model(
    *,
    model_name: str,
    number_of_classes: int = DEFAULT_NUMBER_OF_CLASSES,
    pretrained: bool = True,
    dropout_probability: float = (
        DEFAULT_DROPOUT_PROBABILITY
    ),
    training_mode: str = "classifier",
) -> nn.Module:
    """Create a configured transfer-learning model.

    Args:
        model_name:
            Required architecture name.
        number_of_classes:
            Number of TrashNet output categories.
        pretrained:
            Whether to load ImageNet pretrained weights.
        dropout_probability:
            Dropout used in the replacement classifier.
        training_mode:
            ``classifier``, ``last_block``, or ``full``.

    Returns:
        Configured PyTorch model.
    """

    validated_model_name = validate_model_name(
        model_name
    )

    validated_class_count = validate_number_of_classes(
        number_of_classes
    )

    validated_dropout = validate_dropout_probability(
        dropout_probability
    )

    validated_training_mode = validate_training_mode(
        training_mode
    )

    if not isinstance(pretrained, bool):
        raise TypeError(
            "pretrained must be a boolean."
        )

    if validated_model_name == "resnet50":
        weights = (
            ResNet50_Weights.DEFAULT
            if pretrained
            else None
        )

        model = resnet50(
            weights=weights
        )

    elif validated_model_name == "efficientnet_b0":
        weights = (
            EfficientNet_B0_Weights.DEFAULT
            if pretrained
            else None
        )

        model = efficientnet_b0(
            weights=weights
        )

    elif validated_model_name == "densenet121":
        weights = (
            DenseNet121_Weights.DEFAULT
            if pretrained
            else None
        )

        model = densenet121(
            weights=weights
        )

    elif validated_model_name == "mobilenet_v3":
        weights = (
            MobileNet_V3_Large_Weights.DEFAULT
            if pretrained
            else None
        )

        model = mobilenet_v3_large(
            weights=weights
        )

    else:
        raise ValueError(
            f"Unsupported model: {validated_model_name}"
        )

    replace_classifier(
        model=model,
        model_name=validated_model_name,
        number_of_classes=validated_class_count,
        dropout_probability=validated_dropout,
    )

    configure_trainable_layers(
        model=model,
        model_name=validated_model_name,
        training_mode=validated_training_mode,
    )

    return model


def count_parameters(
    model: nn.Module,
) -> tuple[int, int]:
    """Return total and trainable parameter counts."""

    if not isinstance(model, nn.Module):
        raise TypeError(
            "model must be a torch.nn.Module."
        )

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return (
        total_parameters,
        trainable_parameters,
    )


def get_trainable_parameter_names(
    model: nn.Module,
) -> tuple[str, ...]:
    """Return names of parameters currently enabled for training."""

    return tuple(
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )


def validate_model_output(
    *,
    model: nn.Module,
    inputs: Tensor,
    number_of_classes: int,
) -> Tensor:
    """Run and validate a model forward pass."""

    outputs = model(inputs)

    expected_shape = (
        inputs.shape[0],
        number_of_classes,
    )

    if outputs.shape != expected_shape:
        raise ValueError(
            f"Expected model output shape {expected_shape}, "
            f"received {tuple(outputs.shape)}."
        )

    return outputs


__all__ = [
    "DEFAULT_DROPOUT_PROBABILITY",
    "DEFAULT_NUMBER_OF_CLASSES",
    "ModelName",
    "SUPPORTED_MODEL_NAMES",
    "SUPPORTED_TRAINING_MODES",
    "TrainingMode",
    "configure_trainable_layers",
    "count_parameters",
    "create_transfer_learning_model",
    "freeze_all_parameters",
    "get_classifier_module",
    "get_trainable_parameter_names",
    "replace_classifier",
    "unfreeze_last_feature_block",
    "unfreeze_module",
    "validate_dropout_probability",
    "validate_model_name",
    "validate_model_output",
    "validate_number_of_classes",
    "validate_training_mode",
]