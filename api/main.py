"""
FastAPI backend for the Smart Waste Classification System.

Available endpoints
-------------------
GET  /health
GET  /model-info
POST /predict
POST /batch-predict

The batch endpoint uses separate UploadFile fields because some combinations
of FastAPI, Pydantic, and Swagger UI incorrectly render list[UploadFile] as
array<string> text boxes instead of file selectors.
"""

from __future__ import annotations

import logging
import os
import time

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    BatchPredictionItem,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)

from src.model_loader import load_production_model
from src.predict import WastePredictionService
from src.preprocessing import (
    ImageValidationError,
    validate_image_bytes,
)


# =============================================================================
# Project configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "final_waste_classifier.pt"
)

MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        str(DEFAULT_MODEL_PATH),
    )
)

LOG_LEVEL = os.getenv(
    "LOG_LEVEL",
    "INFO",
).upper()


# =============================================================================
# Logging
# =============================================================================

logging.basicConfig(
    level=LOG_LEVEL,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    ),
)

logger = logging.getLogger(
    "smart-waste-classification-api"
)


# =============================================================================
# Global prediction service
# =============================================================================

prediction_service: WastePredictionService | None = None


# =============================================================================
# Application startup and shutdown
# =============================================================================

@asynccontextmanager
async def lifespan(
    app: FastAPI,
):
    """
    Load the production model once during application startup.

    The loaded model is shared by all prediction requests.
    """
    global prediction_service

    logger.info(
        "Application startup initiated."
    )

    try:
        loaded_model = load_production_model(
            MODEL_PATH
        )

        prediction_service = (
            WastePredictionService(
                loaded_model
            )
        )

        logger.info(
            "Production model loaded successfully."
        )

        logger.info(
            "Model name: %s",
            loaded_model.package.get(
                "model_name"
            ),
        )

        logger.info(
            "Architecture: %s",
            loaded_model.package.get(
                "architecture"
            ),
        )

        logger.info(
            "Model path: %s",
            MODEL_PATH.resolve(),
        )

    except Exception:
        prediction_service = None

        logger.exception(
            "Production model loading failed."
        )

    yield

    prediction_service = None

    logger.info(
        "Application shutdown completed."
    )


# =============================================================================
# FastAPI application
# =============================================================================

app = FastAPI(
    title="Smart Waste Classification API",
    version="1.0.0",
    description=(
        "Classifies waste images using the final "
        "production deep-learning model and returns "
        "Grad-CAM explanations."
    ),
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Shared helpers
# =============================================================================

def require_prediction_service(
) -> WastePredictionService:
    """
    Return the active prediction service.

    Raises
    ------
    HTTPException
        If the production model is unavailable.
    """
    if prediction_service is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "The production model is not available. "
                "Check the API startup logs and verify "
                "that final_waste_classifier.pt exists."
            ),
        )

    return prediction_service


def build_prediction_response(
    *,
    filename: str,
    prediction,
) -> PredictionResponse:
    """
    Convert an internal prediction result into the public API schema.
    """
    return PredictionResponse(
        filename=filename,
        predicted_class=(
            prediction.predicted_class
        ),
        predicted_class_index=(
            prediction.predicted_class_index
        ),
        confidence=(
            prediction.confidence
        ),
        class_probabilities=(
            prediction.class_probabilities
        ),
        gradcam_base64=(
            prediction.gradcam_base64
        ),
        timestamp=(
            prediction.timestamp
        ),
    )


async def process_uploaded_file(
    *,
    upload_file: UploadFile,
    service: WastePredictionService,
) -> PredictionResponse:
    """
    Validate and classify one uploaded image.

    The caller remains responsible for closing the UploadFile.
    """
    filename = (
        upload_file.filename
        or "uploaded_image"
    )

    image_bytes = await upload_file.read()

    image = validate_image_bytes(
        image_bytes,
        filename=filename,
        content_type=(
            upload_file.content_type
        ),
    )

    prediction = service.predict(
        image
    )

    return build_prediction_response(
        filename=filename,
        prediction=prediction,
    )


# =============================================================================
# System endpoints
# =============================================================================

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Check API health",
)
def health_check() -> HealthResponse:
    """
    Report whether the API is running and the model is loaded.
    """
    model_loaded = (
        prediction_service is not None
    )

    model_name = None

    if prediction_service is not None:
        model_name = str(
            prediction_service.package.get(
                "model_name"
            )
        )

    return HealthResponse(
        status=(
            "healthy"
            if model_loaded
            else "degraded"
        ),
        model_loaded=model_loaded,
        model_name=model_name,
    )


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    tags=["Model"],
    summary="Get active model information",
)
def model_information(
) -> ModelInfoResponse:
    """
    Return safe public metadata for the active model.
    """
    service = require_prediction_service()

    return ModelInfoResponse(
        **service.get_model_information()
    )


# =============================================================================
# Single-image prediction
# =============================================================================

@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Classify one waste image",
)
async def predict_image(
    file: UploadFile = File(
        ...,
        description=(
            "JPEG or PNG waste image."
        ),
    ),
) -> PredictionResponse:
    """
    Classify one JPEG or PNG image.

    The response contains the predicted class, confidence, probabilities,
    Grad-CAM image, and timestamp.
    """
    service = require_prediction_service()

    request_start_time = (
        time.perf_counter()
    )

    filename = (
        file.filename
        or "uploaded_image"
    )

    try:
        result = await process_uploaded_file(
            upload_file=file,
            service=service,
        )

        duration_seconds = (
            time.perf_counter()
            - request_start_time
        )

        logger.info(
            "Prediction completed | "
            "filename=%s | "
            "class=%s | "
            "confidence=%.4f | "
            "duration=%.3fs",
            filename,
            result.predicted_class,
            result.confidence,
            duration_seconds,
        )

        return result

    except ImageValidationError as error:
        logger.warning(
            "Image validation failed | "
            "filename=%s | reason=%s",
            filename,
            error,
        )

        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error

    except HTTPException:
        raise

    except Exception as error:
        logger.exception(
            "Prediction failed | "
            "filename=%s",
            filename,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Prediction could not be completed."
            ),
        ) from error

    finally:
        await file.close()


# =============================================================================
# Batch prediction
# =============================================================================

@app.post(
    "/batch-predict",
    response_model=BatchPredictionResponse,
    tags=["Prediction"],
    summary="Classify up to five waste images",
)
async def batch_predict(
    file_1: UploadFile = File(
        ...,
        description="First JPEG or PNG image.",
    ),
    file_2: UploadFile | None = File(
        None,
        description="Optional second JPEG or PNG image.",
    ),
    file_3: UploadFile | None = File(
        None,
        description="Optional third JPEG or PNG image.",
    ),
    file_4: UploadFile | None = File(
        None,
        description="Optional fourth JPEG or PNG image.",
    ),
    file_5: UploadFile | None = File(
        None,
        description="Optional fifth JPEG or PNG image.",
    ),
) -> BatchPredictionResponse:
    """
    Classify between one and five uploaded images.

    Empty optional upload fields are ignored. Invalid files are reported
    individually and do not cancel successful predictions.
    """
    service = require_prediction_service()

    submitted_files = [
        file_1,
        file_2,
        file_3,
        file_4,
        file_5,
    ]

    # Swagger submits unused optional fields as empty multipart entries.
    # Keep only fields that contain a real filename.
    uploaded_files = [
        upload_file
        for upload_file in submitted_files
        if (
            upload_file is not None
            and upload_file.filename is not None
            and upload_file.filename.strip() != ""
        )
    ]

    if not uploaded_files:
        raise HTTPException(
            status_code=422,
            detail="At least one image must be uploaded.",
        )

    results: list[BatchPredictionItem] = []
    errors: list[dict[str, str]] = []

    batch_start_time = time.perf_counter()

    for upload_file in uploaded_files:
        filename = (
            upload_file.filename
            or "uploaded_image"
        )

        try:
            prediction_response = (
                await process_uploaded_file(
                    upload_file=upload_file,
                    service=service,
                )
            )

            # Convert the single-prediction schema into the exact schema
            # required by BatchPredictionResponse.
            batch_item = BatchPredictionItem(
                **prediction_response.model_dump()
            )

            results.append(batch_item)

            logger.info(
                "Batch item completed | "
                "filename=%s | "
                "class=%s | "
                "confidence=%.4f",
                filename,
                batch_item.predicted_class,
                batch_item.confidence,
            )

        except ImageValidationError as error:
            logger.warning(
                "Batch image validation failed | "
                "filename=%s | reason=%s",
                filename,
                error,
            )

            errors.append(
                {
                    "filename": filename,
                    "error": str(error),
                }
            )

        except Exception as error:
            logger.exception(
                "Batch prediction failed | "
                "filename=%s",
                filename,
            )

            errors.append(
                {
                    "filename": filename,
                    "error": (
                        "Prediction could not be completed."
                    ),
                }
            )

        finally:
            await upload_file.close()

    batch_duration_seconds = (
        time.perf_counter()
        - batch_start_time
    )

    logger.info(
        "Batch prediction completed | "
        "total=%d | successful=%d | "
        "failed=%d | duration=%.3fs",
        len(uploaded_files),
        len(results),
        len(errors),
        batch_duration_seconds,
    )

    return BatchPredictionResponse(
        total_files=len(uploaded_files),
        successful_predictions=len(results),
        failed_predictions=len(errors),
        results=results,
        errors=errors,
    )
