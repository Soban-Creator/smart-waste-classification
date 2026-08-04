"""
Load and reconstruct the final production classifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from torchvision.models import (
    mobilenet_v3_large,
    resnet50,
)

from src.preprocessing import (
    build_inference_transform,
)


@dataclass
class LoadedModel:
    """Container for the active production model."""

    model: nn.Module
    package: dict[str, Any]
    transform: Any
    device: torch.device


def build_model_architecture(
    *,
    architecture: str,
    number_of_classes: int,
    configuration: dict[str, Any],
) -> nn.Module:
    """
    Reconstruct the selected architecture.

    Supported models are limited to the two final candidates used by
    the project.
    """
    dropout_rate = float(
        configuration.get(
            "dropout_rate",
            0.30,
        )
    )

    if architecture == "mobilenet_v3_large":
        model = mobilenet_v3_large(
            weights=None
        )

        input_features = (
            model.classifier[3].in_features
        )

        model.classifier[2] = nn.Dropout(
            p=dropout_rate,
            inplace=True,
        )

        model.classifier[3] = nn.Linear(
            in_features=input_features,
            out_features=number_of_classes,
        )

        return model

    if architecture == "resnet50":
        model = resnet50(
            weights=None
        )

        input_features = model.fc.in_features

        model.fc = nn.Sequential(
            nn.Dropout(
                p=dropout_rate,
            ),
            nn.Linear(
                in_features=input_features,
                out_features=number_of_classes,
            ),
        )

        return model

    raise ValueError(
        "Unsupported production architecture: "
        f"{architecture}"
    )


def load_production_model(
    model_path: Path,
    *,
    device: torch.device | None = None,
) -> LoadedModel:
    """
    Load the complete production package.

    The model package must include architecture, model weights,
    class mapping, image size, and normalization values.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            "Production model was not found at:\n"
            f"{model_path.resolve()}"
        )

    selected_device = (
        device
        if device is not None
        else torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    )

    package = torch.load(
        model_path,
        map_location=selected_device,
        weights_only=False,
    )

    required_keys = {
        "architecture",
        "model_name",
        "model_version",
        "model_state_dict",
        "configuration",
        "class_names",
        "class_to_index",
        "image_size",
        "normalization_mean",
        "normalization_std",
    }

    missing_keys = (
        required_keys
        - set(package.keys())
    )

    if missing_keys:
        raise KeyError(
            "The model package is missing: "
            + ", ".join(
                sorted(missing_keys)
            )
        )

    class_names = list(
        package["class_names"]
    )

    model = build_model_architecture(
        architecture=package[
            "architecture"
        ],
        number_of_classes=len(
            class_names
        ),
        configuration=package[
            "configuration"
        ],
    )

    model.load_state_dict(
        package["model_state_dict"]
    )

    model = model.to(
        selected_device
    )

    model.eval()

    preprocessing = package.get(
        "preprocessing",
        {},
    )

    transform = build_inference_transform(
        image_size=package[
            "image_size"
        ],
        normalization_mean=package[
            "normalization_mean"
        ],
        normalization_std=package[
            "normalization_std"
        ],
        resize_shorter_side=int(
            preprocessing.get(
                "resize_shorter_side",
                256,
            )
        ),
    )

    return LoadedModel(
        model=model,
        package=package,
        transform=transform,
        device=selected_device,
    )