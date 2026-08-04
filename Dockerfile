# syntax=docker/dockerfile:1

# =============================================================================
# Base Python image
# =============================================================================

FROM python:3.12-slim


# =============================================================================
# Runtime configuration
# =============================================================================

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONPATH=/app


# =============================================================================
# Application directory
# =============================================================================

WORKDIR /app


# =============================================================================
# System dependencies
# =============================================================================

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*


# =============================================================================
# Python dependencies
# =============================================================================

COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip \
    && python -m pip install -r /app/requirements.txt


# =============================================================================
# Application source
# =============================================================================

COPY api /app/api
COPY app /app/app
COPY src /app/src
COPY models/final_waste_classifier.pt \
     /app/models/final_waste_classifier.pt

# Copy metadata only when it exists in your project.
COPY models/final_waste_classifier_metadata.json \
     /app/models/final_waste_classifier_metadata.json


# =============================================================================
# Non-root runtime user
# =============================================================================

RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app

USER appuser


# =============================================================================
# Default service
# =============================================================================

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
