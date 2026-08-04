"""
High-level production inference service.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import torch

from PIL import Image

from src.gradcam import (
    create_gradcam_base64,
)

from src.model_loader import (
    LoadedModel,
)


@dataclass
class PredictionOutput:
    """Structured internal prediction result."""

    predicted_class: str
    predicted_class_index: int
    confidence: float
    class_probabilities: dict[str, float]
    gradcam_base64: str
    timestamp: str


class WastePredictionService:
    """
    Reusable prediction interface used by the FastAPI application.
    """

    def __init__(
        self,
        loaded_model: LoadedModel,
    ) -> None:
        self.loaded_model = loaded_model

        self.model = loaded_model.model
        self.package = loaded_model.package
        self.transform = loaded_model.transform
        self.device = loaded_model.device

        self.class_names = list(
            self.package["class_names"]
        )

        self.architecture = str(
            self.package["architecture"]
        )

    def predict(
        self,
        image: Image.Image,
    ) -> PredictionOutput:
        """Classify one image and generate its Grad-CAM overlay."""
        input_tensor = (
            self.transform(image)
            .unsqueeze(0)
            .to(self.device)
        )

        self.model.eval()

        with torch.inference_mode():
            logits = self.model(
                input_tensor
            )

            probabilities_tensor = (
                torch.softmax(
                    logits,
                    dim=1,
                )
            )

            predicted_class_index = int(
                probabilities_tensor.argmax(
                    dim=1
                ).item()
            )

            confidence = float(
                probabilities_tensor[
                    0,
                    predicted_class_index,
                ].item()
            )

            probabilities = (
                probabilities_tensor[
                    0
                ]
                .detach()
                .cpu()
                .tolist()
            )

        # Grad-CAM requires gradients, so it is generated
        # outside torch.inference_mode().
        gradcam_base64 = (
            create_gradcam_base64(
                model=self.model,
                architecture=self.architecture,
                input_tensor=input_tensor,
                original_image=image,
                predicted_class_index=(
                    predicted_class_index
                ),
            )
        )

        class_probabilities = {
            class_name: float(
                probabilities[class_index]
            )
            for class_index, class_name
            in enumerate(
                self.class_names
            )
        }

        return PredictionOutput(
            predicted_class=(
                self.class_names[
                    predicted_class_index
                ]
            ),
            predicted_class_index=(
                predicted_class_index
            ),
            confidence=confidence,
            class_probabilities=(
                class_probabilities
            ),
            gradcam_base64=(
                gradcam_base64
            ),
            timestamp=(
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
        )

    def get_model_information(
        self,
    ) -> dict[str, Any]:
        """Return safe public metadata for the active model."""
        return {
            "model_name": (
                self.package[
                    "model_name"
                ]
            ),
            "model_version": (
                self.package[
                    "model_version"
                ]
            ),
            "architecture": (
                self.architecture
            ),
            "class_names": (
                self.class_names
            ),
            "image_size": (
                self.package[
                    "image_size"
                ]
            ),
            "saved_at": (
                self.package.get(
                    "saved_at"
                )
            ),
            "framework": (
                self.package.get(
                    "framework",
                    "PyTorch",
                )
            ),
            "pytorch_version": (
                self.package.get(
                    "pytorch_version"
                )
            ),
            "torchvision_version": (
                self.package.get(
                    "torchvision_version"
                )
            ),
        }
    