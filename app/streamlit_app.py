"""
Streamlit frontend for the Smart Waste Classification System.

The frontend communicates only with the FastAPI backend. It does not load or
reimplement the PyTorch model.
"""

from __future__ import annotations

import base64
import os

from io import BytesIO
from typing import Any

import pandas as pd
import requests
import streamlit as st

from PIL import Image, UnidentifiedImageError


# =============================================================================
# Application configuration
# =============================================================================

DEFAULT_API_URL = "http://127.0.0.1:8000"

API_BASE_URL = os.getenv(
    "API_BASE_URL",
    DEFAULT_API_URL,
).rstrip("/")

HEALTH_ENDPOINT = (
    f"{API_BASE_URL}/health"
)

MODEL_INFO_ENDPOINT = (
    f"{API_BASE_URL}/model-info"
)

PREDICT_ENDPOINT = (
    f"{API_BASE_URL}/predict"
)

BATCH_PREDICT_ENDPOINT = (
    f"{API_BASE_URL}/batch-predict"
)

REQUEST_TIMEOUT_SECONDS = 180

SUPPORTED_FILE_TYPES = [
    "jpg",
    "jpeg",
    "png",
]

MAX_BATCH_FILES = 5


# =============================================================================
# Page configuration
# =============================================================================

st.set_page_config(
    page_title=(
        "Smart Waste Classification"
    ),
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# Styling
# =============================================================================

st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0;
        }

        .main-subtitle {
            color: #666666;
            font-size: 1.05rem;
            margin-top: 0.25rem;
            margin-bottom: 1.5rem;
        }

        .prediction-card {
            padding: 1.2rem;
            border: 1px solid rgba(128, 128, 128, 0.25);
            border-radius: 12px;
            margin-bottom: 1rem;
        }

        .class-label {
            font-size: 1.6rem;
            font-weight: 700;
        }

        .small-note {
            color: #777777;
            font-size: 0.9rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# API helpers
# =============================================================================

def get_api_health() -> dict[str, Any] | None:
    """
    Request the FastAPI health endpoint.

    Returns None when the API cannot be reached.
    """
    try:
        response = requests.get(
            HEALTH_ENDPOINT,
            timeout=10,
        )

        response.raise_for_status()

        return response.json()

    except (
        requests.RequestException,
        ValueError,
    ):
        return None


def get_model_information() -> dict[str, Any] | None:
    """
    Request information about the active production model.
    """
    try:
        response = requests.get(
            MODEL_INFO_ENDPOINT,
            timeout=10,
        )

        response.raise_for_status()

        return response.json()

    except (
        requests.RequestException,
        ValueError,
    ):
        return None


def parse_api_error(
    response: requests.Response,
) -> str:
    """
    Extract a useful error message from an API response.
    """
    try:
        payload = response.json()

    except ValueError:
        return (
            response.text.strip()
            or "The API returned an unknown error."
        )

    detail = payload.get(
        "detail"
    )

    if isinstance(detail, str):
        return detail

    if isinstance(detail, list):
        messages = []

        for item in detail:
            if isinstance(item, dict):
                message = item.get(
                    "msg",
                    str(item),
                )
            else:
                message = str(item)

            messages.append(
                message
            )

        return "; ".join(
            messages
        )

    return str(
        payload
    )


def validate_uploaded_image(
    uploaded_file,
) -> Image.Image:
    """
    Validate an uploaded file locally before sending it to the API.

    The backend remains the authoritative validator.
    """
    if uploaded_file is None:
        raise ValueError(
            "Please upload an image."
        )

    try:
        uploaded_file.seek(0)

        image = Image.open(
            uploaded_file
        )

        image.load()

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
    ) as error:
        raise ValueError(
            "The selected file is not a valid "
            "JPEG or PNG image."
        ) from error

    uploaded_file.seek(0)

    return image.convert(
        "RGB"
    )


def decode_gradcam_image(
    gradcam_base64: str,
) -> Image.Image:
    """
    Decode a Base64 Grad-CAM image returned by the API.
    """
    if not gradcam_base64:
        raise ValueError(
            "The API returned an empty "
            "Grad-CAM result."
        )

    try:
        image_bytes = (
            base64.b64decode(
                gradcam_base64,
                validate=True,
            )
        )

        gradcam_image = Image.open(
            BytesIO(
                image_bytes
            )
        )

        gradcam_image.load()

        return gradcam_image.convert(
            "RGB"
        )

    except (
        ValueError,
        UnidentifiedImageError,
        OSError,
    ) as error:
        raise ValueError(
            "The Grad-CAM response could "
            "not be decoded as an image."
        ) from error


def request_single_prediction(
    uploaded_file,
) -> dict[str, Any]:
    """
    Send one image to POST /predict.
    """
    uploaded_file.seek(0)

    file_bytes = (
        uploaded_file.read()
    )

    files = {
        "file": (
            uploaded_file.name,
            file_bytes,
            uploaded_file.type
            or "application/octet-stream",
        )
    }

    try:
        response = requests.post(
            PREDICT_ENDPOINT,
            files=files,
            timeout=(
                REQUEST_TIMEOUT_SECONDS
            ),
        )

    except requests.Timeout as error:
        raise RuntimeError(
            "The prediction request timed out. "
            "The model may still be processing."
        ) from error

    except requests.ConnectionError as error:
        raise RuntimeError(
            "The FastAPI backend could not be reached. "
            "Confirm that Uvicorn is running on "
            f"{API_BASE_URL}."
        ) from error

    except requests.RequestException as error:
        raise RuntimeError(
            "The prediction request failed."
        ) from error

    if response.status_code != 200:
        raise RuntimeError(
            parse_api_error(
                response
            )
        )

    return response.json()


def request_batch_prediction(
    uploaded_files: list,
) -> dict[str, Any]:
    """
    Send up to five images to POST /batch-predict.

    The current backend exposes file_1 through file_5 as separate fields.
    """
    multipart_files = {}

    for file_index, uploaded_file in enumerate(
        uploaded_files,
        start=1,
    ):
        uploaded_file.seek(0)

        multipart_files[
            f"file_{file_index}"
        ] = (
            uploaded_file.name,
            uploaded_file.read(),
            uploaded_file.type
            or "application/octet-stream",
        )

    try:
        response = requests.post(
            BATCH_PREDICT_ENDPOINT,
            files=multipart_files,
            timeout=(
                REQUEST_TIMEOUT_SECONDS
            ),
        )

    except requests.Timeout as error:
        raise RuntimeError(
            "The batch request timed out."
        ) from error

    except requests.ConnectionError as error:
        raise RuntimeError(
            "The FastAPI backend could not be reached. "
            "Confirm that Uvicorn is running."
        ) from error

    except requests.RequestException as error:
        raise RuntimeError(
            "The batch request failed."
        ) from error

    if response.status_code != 200:
        raise RuntimeError(
            parse_api_error(
                response
            )
        )

    return response.json()


# =============================================================================
# Display helpers
# =============================================================================

def format_class_name(
    class_name: str,
) -> str:
    """
    Convert an API class value into a display label.
    """
    return class_name.replace(
        "_",
        " ",
    ).title()


def build_probability_table(
    probabilities: dict[str, float],
) -> pd.DataFrame:
    """
    Create a sorted probability table.
    """
    probability_table = pd.DataFrame(
        [
            {
                "Class": format_class_name(
                    class_name
                ),
                "Probability": float(
                    probability
                ),
            }
            for class_name, probability
            in probabilities.items()
        ]
    )

    return probability_table.sort_values(
        "Probability",
        ascending=False,
    ).reset_index(
        drop=True
    )


def display_probability_results(
    probabilities: dict[str, float],
) -> None:
    """
    Display probabilities as a chart and percentage table.
    """
    probability_table = (
        build_probability_table(
            probabilities
        )
    )

    chart_table = (
        probability_table
        .set_index(
            "Class"
        )
    )

    st.bar_chart(
        chart_table,
        y="Probability",
        horizontal=True,
    )

    display_table = (
        probability_table.copy()
    )

    display_table[
        "Probability"
    ] = display_table[
        "Probability"
    ].map(
        lambda value: (
            f"{value:.2%}"
        )
    )

    st.dataframe(
        display_table,
        use_container_width=True,
        hide_index=True,
    )


def display_single_prediction(
    result: dict[str, Any],
) -> None:
    """
    Display one complete API prediction.
    """
    predicted_class = (
        format_class_name(
            result[
                "predicted_class"
            ]
        )
    )

    confidence = float(
        result["confidence"]
    )

    st.success(
        "Classification completed successfully."
    )

    metric_column_1, metric_column_2 = (
        st.columns(2)
    )

    with metric_column_1:
        st.metric(
            label="Predicted Category",
            value=predicted_class,
        )

    with metric_column_2:
        st.metric(
            label="Confidence",
            value=(
                f"{confidence:.2%}"
            ),
        )

    st.progress(
        min(
            max(
                confidence,
                0.0,
            ),
            1.0,
        ),
        text=(
            f"Prediction confidence: "
            f"{confidence:.2%}"
        ),
    )

    probability_column, gradcam_column = (
        st.columns(
            [1, 1]
        )
    )

    with probability_column:
        st.subheader(
            "Class Probabilities"
        )

        display_probability_results(
            result[
                "class_probabilities"
            ]
        )

    with gradcam_column:
        st.subheader(
            "Grad-CAM Explanation"
        )

        try:
            gradcam_image = (
                decode_gradcam_image(
                    result[
                        "gradcam_base64"
                    ]
                )
            )

            st.image(
                gradcam_image,
                caption=(
                    "Highlighted regions influenced "
                    "the model's prediction."
                ),
                use_container_width=True,
            )

        except ValueError as error:
            st.error(
                str(error)
            )

    with st.expander(
        "Technical prediction details"
    ):
        st.write(
            {
                "filename": (
                    result.get(
                        "filename"
                    )
                ),
                "predicted_class_index": (
                    result.get(
                        "predicted_class_index"
                    )
                ),
                "timestamp": (
                    result.get(
                        "timestamp"
                    )
                ),
            }
        )


def display_batch_result(
    result: dict[str, Any],
    item_number: int,
) -> None:
    """
    Display one result from a batch response.
    """
    predicted_class = (
        format_class_name(
            result[
                "predicted_class"
            ]
        )
    )

    confidence = float(
        result[
            "confidence"
        ]
    )

    with st.expander(
        (
            f"{item_number}. "
            f"{result['filename']} — "
            f"{predicted_class} "
            f"({confidence:.2%})"
        ),
        expanded=(
            item_number == 1
        ),
    ):
        metric_column_1, metric_column_2 = (
            st.columns(2)
        )

        metric_column_1.metric(
            "Predicted Category",
            predicted_class,
        )

        metric_column_2.metric(
            "Confidence",
            f"{confidence:.2%}",
        )

        image_column, probability_column = (
            st.columns(2)
        )

        with image_column:
            try:
                gradcam_image = (
                    decode_gradcam_image(
                        result[
                            "gradcam_base64"
                        ]
                    )
                )

                st.image(
                    gradcam_image,
                    caption=(
                        "Grad-CAM explanation"
                    ),
                    use_container_width=True,
                )

            except ValueError as error:
                st.error(
                    str(error)
                )

        with probability_column:
            display_probability_results(
                result[
                    "class_probabilities"
                ]
            )


# =============================================================================
# Sidebar
# =============================================================================

st.sidebar.title(
    "System Status"
)

api_health = get_api_health()

if api_health is None:
    st.sidebar.error(
        "API unavailable"
    )

    st.sidebar.caption(
        "Start the FastAPI backend before "
        "using classification."
    )

    api_is_healthy = False

else:
    api_is_healthy = bool(
        api_health.get(
            "model_loaded",
            False,
        )
    )

    if api_is_healthy:
        st.sidebar.success(
            "API and model are ready"
        )

    else:
        st.sidebar.warning(
            "API is running, but the model "
            "is unavailable"
        )

    st.sidebar.write(
        f"**Status:** "
        f"{api_health.get('status')}"
    )

    st.sidebar.write(
        f"**Active model:** "
        f"{api_health.get('model_name')}"
    )

st.sidebar.divider()

st.sidebar.subheader(
    "API Configuration"
)

st.sidebar.code(
    API_BASE_URL
)

if st.sidebar.button(
    "Refresh API Status",
    use_container_width=True,
):
    st.rerun()

model_information = (
    get_model_information()
    if api_is_healthy
    else None
)

if model_information:
    with st.sidebar.expander(
        "Active Model Information"
    ):
        st.write(
            f"**Model:** "
            f"{model_information.get('model_name')}"
        )

        st.write(
            f"**Version:** "
            f"{model_information.get('model_version')}"
        )

        st.write(
            f"**Architecture:** "
            f"{model_information.get('architecture')}"
        )

        st.write(
            "**Classes:**"
        )

        for class_name in (
            model_information.get(
                "class_names",
                [],
            )
        ):
            st.write(
                f"- {format_class_name(class_name)}"
            )

        st.write(
            f"**Input size:** "
            f"{model_information.get('image_size')}"
        )


# =============================================================================
# Main page
# =============================================================================

st.markdown(
    '<p class="main-title">♻️ Smart Waste Classification</p>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <p class="main-subtitle">
        Upload a waste image to classify it as cardboard, glass, metal,
        paper, plastic, or trash. The result includes prediction confidence
        and a Grad-CAM explanation.
    </p>
    """,
    unsafe_allow_html=True,
)

single_tab, batch_tab, guide_tab = st.tabs(
    [
        "Single Classification",
        "Batch Classification",
        "How It Works",
    ]
)


# =============================================================================
# Single prediction tab
# =============================================================================

with single_tab:
    st.header(
        "Classify One Waste Image"
    )

    uploaded_image = st.file_uploader(
        "Upload a JPEG or PNG image",
        type=SUPPORTED_FILE_TYPES,
        accept_multiple_files=False,
        key="single_image_upload",
        help=(
            "Use an image containing one dominant "
            "waste object."
        ),
    )

    if uploaded_image is not None:
        try:
            preview_image = (
                validate_uploaded_image(
                    uploaded_image
                )
            )

            preview_column, instruction_column = (
                st.columns(
                    [1.1, 0.9]
                )
            )

            with preview_column:
                st.image(
                    preview_image,
                    caption=(
                        uploaded_image.name
                    ),
                    use_container_width=True,
                )

            with instruction_column:
                st.info(
                    "The backend will resize, crop, "
                    "normalize, classify, and generate "
                    "Grad-CAM for this image."
                )

                st.write(
                    f"**Filename:** "
                    f"{uploaded_image.name}"
                )

                st.write(
                    f"**File type:** "
                    f"{uploaded_image.type}"
                )

                st.write(
                    f"**Dimensions:** "
                    f"{preview_image.width} × "
                    f"{preview_image.height}"
                )

        except ValueError as error:
            st.error(
                str(error)
            )

    classify_button = st.button(
        "Classify Image",
        type="primary",
        use_container_width=True,
        disabled=(
            uploaded_image is None
            or not api_is_healthy
        ),
    )

    if classify_button:
        try:
            validate_uploaded_image(
                uploaded_image
            )

            with st.spinner(
                "Classifying image and generating "
                "Grad-CAM explanation..."
            ):
                prediction_result = (
                    request_single_prediction(
                        uploaded_image
                    )
                )

            st.session_state[
                "single_prediction_result"
            ] = prediction_result

        except (
            ValueError,
            RuntimeError,
        ) as error:
            st.error(
                str(error)
            )

    if (
        "single_prediction_result"
        in st.session_state
    ):
        st.divider()

        display_single_prediction(
            st.session_state[
                "single_prediction_result"
            ]
        )


# =============================================================================
# Batch prediction tab
# =============================================================================

with batch_tab:
    st.header(
        "Classify Multiple Waste Images"
    )

    st.caption(
        "Upload between one and five images. "
        "Each image is processed independently."
    )

    batch_uploads = st.file_uploader(
        "Upload JPEG or PNG images",
        type=SUPPORTED_FILE_TYPES,
        accept_multiple_files=True,
        key="batch_image_upload",
    )

    batch_uploads = (
        batch_uploads
        or []
    )

    if len(
        batch_uploads
    ) > MAX_BATCH_FILES:
        st.error(
            f"Upload no more than "
            f"{MAX_BATCH_FILES} images."
        )

    elif batch_uploads:
        preview_columns = st.columns(
            min(
                len(batch_uploads),
                3,
            )
        )

        for file_index, uploaded_file in enumerate(
            batch_uploads
        ):
            try:
                preview_image = (
                    validate_uploaded_image(
                        uploaded_file
                    )
                )

                with preview_columns[
                    file_index
                    % len(preview_columns)
                ]:
                    st.image(
                        preview_image,
                        caption=(
                            uploaded_file.name
                        ),
                        use_container_width=True,
                    )

            except ValueError as error:
                st.error(
                    f"{uploaded_file.name}: "
                    f"{error}"
                )

    batch_button = st.button(
        "Classify Batch",
        type="primary",
        use_container_width=True,
        disabled=(
            not batch_uploads
            or len(batch_uploads)
            > MAX_BATCH_FILES
            or not api_is_healthy
        ),
    )

    if batch_button:
        try:
            for uploaded_file in (
                batch_uploads
            ):
                validate_uploaded_image(
                    uploaded_file
                )

            with st.spinner(
                "Processing batch and generating "
                "Grad-CAM explanations..."
            ):
                batch_result = (
                    request_batch_prediction(
                        batch_uploads
                    )
                )

            st.session_state[
                "batch_prediction_result"
            ] = batch_result

        except (
            ValueError,
            RuntimeError,
        ) as error:
            st.error(
                str(error)
            )

    if (
        "batch_prediction_result"
        in st.session_state
    ):
        batch_result = (
            st.session_state[
                "batch_prediction_result"
            ]
        )

        st.divider()

        summary_column_1, summary_column_2, summary_column_3 = (
            st.columns(3)
        )

        summary_column_1.metric(
            "Total Files",
            batch_result[
                "total_files"
            ],
        )

        summary_column_2.metric(
            "Successful",
            batch_result[
                "successful_predictions"
            ],
        )

        summary_column_3.metric(
            "Failed",
            batch_result[
                "failed_predictions"
            ],
        )

        for item_number, result in enumerate(
            batch_result.get(
                "results",
                [],
            ),
            start=1,
        ):
            display_batch_result(
                result,
                item_number,
            )

        batch_errors = (
            batch_result.get(
                "errors",
                [],
            )
        )

        if batch_errors:
            st.subheader(
                "Files That Could Not Be Processed"
            )

            st.dataframe(
                pd.DataFrame(
                    batch_errors
                ),
                use_container_width=True,
                hide_index=True,
            )


# =============================================================================
# Explanation tab
# =============================================================================

with guide_tab:
    st.header(
        "How the System Works"
    )

    st.markdown(
        """
        1. **Image upload**  
           The frontend accepts JPEG or PNG images.

        2. **API validation**  
           FastAPI checks the filename, MIME type, image format,
           dimensions, and file integrity.

        3. **Preprocessing**  
           The image is converted to RGB, resized, center-cropped,
           converted to a tensor, and normalized using the same
           configuration used during model evaluation.

        4. **Classification**  
           The serialized production model predicts one of six classes:
           cardboard, glass, metal, paper, plastic, or trash.

        5. **Confidence and probabilities**  
           Softmax probabilities are returned for every class.

        6. **Grad-CAM explanation**  
           Grad-CAM highlights spatial regions that influenced the
           predicted category.
        """
    )

    st.warning(
        "Grad-CAM shows influential image regions, "
        "but it does not prove that the model reasons "
        "like a human."
    )

    st.info(
        "This application is designed for images "
        "containing one dominant waste object. "
        "Multi-object detection is outside the "
        "current project scope."
    )


# =============================================================================
# Footer
# =============================================================================

st.divider()

st.caption(
    "Smart Waste Classification System · "
    "Streamlit frontend connected to FastAPI"
)
