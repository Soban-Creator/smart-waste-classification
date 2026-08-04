"""Tests for Streamlit frontend helper behavior."""

from __future__ import annotations

import base64

from io import BytesIO

from PIL import Image

from app.streamlit_app import (
    build_probability_table,
    decode_gradcam_image,
    format_class_name,
)


def create_base64_png() -> str:
    """Create a valid Base64 PNG for testing."""
    image = Image.new(
        mode="RGB",
        size=(32, 32),
        color=(100, 150, 200),
    )

    buffer = BytesIO()

    image.save(
        buffer,
        format="PNG",
    )

    return base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")


def test_format_class_name() -> None:
    assert (
        format_class_name(
            "mobilenet_class"
        )
        == "Mobilenet Class"
    )


def test_probability_table_sorted() -> None:
    table = build_probability_table(
        {
            "plastic": 0.20,
            "glass": 0.70,
            "paper": 0.10,
        }
    )

    assert (
        table.iloc[0]["Class"]
        == "Glass"
    )

    assert (
        table.iloc[0]["Probability"]
        == 0.70
    )


def test_decode_gradcam_image() -> None:
    encoded_image = create_base64_png()

    decoded_image = decode_gradcam_image(
        encoded_image
    )

    assert decoded_image.size == (
        32,
        32,
    )

    assert decoded_image.mode == "RGB"


def test_invalid_gradcam_rejected() -> None:
    try:
        decode_gradcam_image(
            "not-valid-base64"
        )

    except ValueError:
        return

    raise AssertionError(
        "Invalid Base64 should raise ValueError."
    )
