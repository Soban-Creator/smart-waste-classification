"""
Production Grad-CAM generation.
"""

from __future__ import annotations

import base64

from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from PIL import Image


def get_target_layer(
    model: nn.Module,
    architecture: str,
) -> nn.Module:
    """Return the final convolutional layer used for Grad-CAM."""
    if architecture == "mobilenet_v3_large":
        return model.features[-1][0]

    if architecture == "resnet50":
        return model.layer4[-1].conv3

    raise ValueError(
        "Grad-CAM is not configured for "
        f"{architecture}."
    )


class GradCAM:
    """Generate a Grad-CAM heatmap using PyTorch hooks."""

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
    ) -> None:
        self.model = model
        self.target_layer = target_layer

        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None

        self.forward_handle = (
            target_layer.register_forward_hook(
                self._capture_activations
            )
        )

        self.backward_handle = (
            target_layer.register_full_backward_hook(
                self._capture_gradients
            )
        )

    def _capture_activations(
        self,
        module,
        inputs,
        output,
    ) -> None:
        self.activations = output.detach()

    def _capture_gradients(
        self,
        module,
        gradient_input,
        gradient_output,
    ) -> None:
        self.gradients = (
            gradient_output[0].detach()
        )

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class_index: int,
        output_size: tuple[int, int],
    ) -> np.ndarray:
        """Generate a normalized Grad-CAM heatmap."""
        self.model.zero_grad(
            set_to_none=True
        )

        logits = self.model(
            input_tensor
        )

        target_score = logits[
            0,
            target_class_index,
        ]

        target_score.backward()

        if (
            self.activations is None
            or self.gradients is None
        ):
            raise RuntimeError(
                "Grad-CAM hooks did not capture "
                "activations and gradients."
            )

        weights = self.gradients.mean(
            dim=(2, 3),
            keepdim=True,
        )

        heatmap = (
            weights * self.activations
        ).sum(
            dim=1,
            keepdim=True,
        )

        heatmap = torch.relu(
            heatmap
        )

        heatmap = (
            nn.functional.interpolate(
                heatmap,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )
        )

        heatmap_array = (
            heatmap[0, 0]
            .detach()
            .cpu()
            .numpy()
        )

        heatmap_array -= (
            heatmap_array.min()
        )

        maximum_value = (
            heatmap_array.max()
        )

        if maximum_value > 0:
            heatmap_array /= maximum_value

        return heatmap_array

    def close(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()


def create_gradcam_base64(
    *,
    model: nn.Module,
    architecture: str,
    input_tensor: torch.Tensor,
    original_image: Image.Image,
    predicted_class_index: int,
) -> str:
    """
    Generate a PNG Grad-CAM overlay and return it as base64.
    """
    target_layer = get_target_layer(
        model,
        architecture,
    )

    gradcam = GradCAM(
        model=model,
        target_layer=target_layer,
    )

    try:
        heatmap = gradcam.generate(
            input_tensor=input_tensor,
            target_class_index=(
                predicted_class_index
            ),
            output_size=(
                original_image.height,
                original_image.width,
            ),
        )

    finally:
        gradcam.close()

    image_array = np.asarray(
        original_image.convert("RGB")
    )

    figure, axis = plt.subplots(
        figsize=(5, 5)
    )

    axis.imshow(
        image_array
    )

    axis.imshow(
        heatmap,
        cmap="jet",
        alpha=0.45,
    )

    axis.axis("off")

    figure.tight_layout(
        pad=0
    )

    buffer = BytesIO()

    figure.savefig(
        buffer,
        format="png",
        dpi=150,
        bbox_inches="tight",
        pad_inches=0,
    )

    plt.close(
        figure
    )

    buffer.seek(0)

    return base64.b64encode(
        buffer.read()
    ).decode("utf-8")
