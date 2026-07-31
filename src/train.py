"""Reusable PyTorch model-training utilities.

This module contains the generic training and validation logic used by
the baseline CNN and future transfer-learning models.

Model architecture definitions remain in ``src.models``.
Final model evaluation remains in ``src.evaluate``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import random
import time
from typing import Any, Final

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader


DEFAULT_RANDOM_SEED: Final[int] = 42
DEFAULT_NUMBER_OF_EPOCHS: Final[int] = 20
DEFAULT_EARLY_STOPPING_PATIENCE: Final[int] = 5
DEFAULT_MINIMUM_IMPROVEMENT: Final[float] = 0.0


@dataclass(frozen=True)
class EpochMetrics:
    """Metrics collected during one complete dataset pass."""

    loss: float
    accuracy: float
    correct_predictions: int
    sample_count: int
    duration_seconds: float


@dataclass
class TrainingHistory:
    """Training and validation metrics collected across epochs."""

    train_loss: list[float] = field(default_factory=list)
    train_accuracy: list[float] = field(default_factory=list)
    validation_loss: list[float] = field(default_factory=list)
    validation_accuracy: list[float] = field(default_factory=list)
    learning_rates: list[float] = field(default_factory=list)
    epoch_durations: list[float] = field(default_factory=list)

    best_epoch: int | None = None
    best_validation_loss: float | None = None
    stopped_early: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable dictionary representation."""

        return asdict(self)


@dataclass
class EarlyStopping:
    """Track validation loss and decide when training should stop."""

    patience: int = DEFAULT_EARLY_STOPPING_PATIENCE
    minimum_improvement: float = DEFAULT_MINIMUM_IMPROVEMENT

    best_loss: float = float("inf")
    epochs_without_improvement: int = 0
    best_epoch: int | None = None

    def __post_init__(self) -> None:
        """Validate early-stopping configuration."""

        validate_positive_integer(
            self.patience,
            parameter_name="patience",
        )

        if not isinstance(
            self.minimum_improvement,
            (int, float),
        ) or isinstance(
            self.minimum_improvement,
            bool,
        ):
            raise TypeError(
                "minimum_improvement must be numeric."
            )

        if self.minimum_improvement < 0:
            raise ValueError(
                "minimum_improvement must be zero or greater."
            )

        self.minimum_improvement = float(
            self.minimum_improvement
        )

    def update(
        self,
        validation_loss: float,
        epoch_number: int,
    ) -> tuple[bool, bool]:
        """Update state using the latest validation loss.

        Args:
            validation_loss:
                Average validation loss for the current epoch.
            epoch_number:
                One-based epoch number.

        Returns:
            A pair containing:

            1. whether the loss improved
            2. whether training should stop
        """

        if not isinstance(
            validation_loss,
            (int, float),
        ) or isinstance(validation_loss, bool):
            raise TypeError(
                "validation_loss must be numeric."
            )

        if not np.isfinite(validation_loss):
            raise ValueError(
                "validation_loss must be finite."
            )

        improvement_threshold = (
            self.best_loss - self.minimum_improvement
        )

        improved = validation_loss < improvement_threshold

        if improved:
            self.best_loss = float(validation_loss)
            self.best_epoch = epoch_number
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        should_stop = (
            self.epochs_without_improvement
            >= self.patience
        )

        return improved, should_stop


def validate_positive_integer(
    value: int,
    *,
    parameter_name: str,
) -> int:
    """Validate an integer that must be greater than zero."""

    if (
        not isinstance(value, int)
        or isinstance(value, bool)
    ):
        raise TypeError(
            f"{parameter_name} must be an integer."
        )

    if value <= 0:
        raise ValueError(
            f"{parameter_name} must be greater than zero."
        )

    return value


def validate_non_negative_integer(
    value: int,
    *,
    parameter_name: str,
) -> int:
    """Validate an integer that must be zero or greater."""

    if (
        not isinstance(value, int)
        or isinstance(value, bool)
    ):
        raise TypeError(
            f"{parameter_name} must be an integer."
        )

    if value < 0:
        raise ValueError(
            f"{parameter_name} must be zero or greater."
        )

    return value


def set_random_seed(
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> None:
    """Configure common random generators for reproducibility."""

    validated_seed = validate_non_negative_integer(
        random_seed,
        parameter_name="random_seed",
    )

    random.seed(validated_seed)
    np.random.seed(validated_seed)
    torch.manual_seed(validated_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(validated_seed)


def select_device(
    preferred_device: str | None = None,
) -> torch.device:
    """Select the available PyTorch computation device.

    Args:
        preferred_device:
            Optional explicit device such as ``"cpu"`` or ``"cuda"``.

    Returns:
        Selected PyTorch device.

    Raises:
        TypeError:
            If the preferred device is not a string or None.
        ValueError:
            If CUDA is requested but unavailable.
    """

    if preferred_device is not None and not isinstance(
        preferred_device,
        str,
    ):
        raise TypeError(
            "preferred_device must be a string or None."
        )

    if preferred_device is None:
        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    normalized_device = preferred_device.strip().lower()

    if normalized_device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(
                "CUDA was requested but is not available."
            )

        return torch.device("cuda")

    if normalized_device == "cpu":
        return torch.device("cpu")

    raise ValueError(
        "preferred_device must be either 'cpu', 'cuda', or None."
    )


def create_classification_loss(
    *,
    class_weights: Tensor | None = None,
    device: torch.device | str = "cpu",
) -> nn.CrossEntropyLoss:
    """Create multiclass cross-entropy loss.

    Args:
        class_weights:
            Optional one-dimensional class-weight tensor.
        device:
            Device on which the loss weights should be stored.

    Returns:
        Configured CrossEntropyLoss instance.

    Raises:
        TypeError:
            If class weights are not a tensor.
        ValueError:
            If class weights are not valid.
    """

    selected_device = torch.device(device)

    if class_weights is None:
        return nn.CrossEntropyLoss()

    if not isinstance(class_weights, Tensor):
        raise TypeError(
            "class_weights must be a PyTorch Tensor or None."
        )

    if class_weights.ndim != 1:
        raise ValueError(
            "class_weights must be one-dimensional."
        )

    if class_weights.numel() < 2:
        raise ValueError(
            "class_weights must contain at least two values."
        )

    if not torch.isfinite(class_weights).all():
        raise ValueError(
            "class_weights must contain only finite values."
        )

    if torch.any(class_weights <= 0):
        raise ValueError(
            "Every class weight must be greater than zero."
        )

    return nn.CrossEntropyLoss(
        weight=class_weights.to(
            device=selected_device,
            dtype=torch.float32,
        )
    )


def calculate_batch_accuracy(
    logits: Tensor,
    targets: Tensor,
) -> tuple[int, int]:
    """Calculate correct predictions and sample count for a batch."""

    if logits.ndim != 2:
        raise ValueError(
            "logits must have shape "
            "[batch_size, number_of_classes]."
        )

    if targets.ndim != 1:
        raise ValueError(
            "targets must have shape [batch_size]."
        )

    if logits.shape[0] != targets.shape[0]:
        raise ValueError(
            "logits and targets must have matching batch sizes."
        )

    predictions = logits.argmax(dim=1)

    correct_predictions = int(
        (predictions == targets).sum().item()
    )

    sample_count = int(targets.shape[0])

    return correct_predictions, sample_count


def run_training_epoch(
    *,
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device | str,
) -> EpochMetrics:
    """Train a model for one complete epoch."""

    if not isinstance(model, nn.Module):
        raise TypeError(
            "model must be a torch.nn.Module."
        )

    selected_device = torch.device(device)

    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    epoch_start = time.perf_counter()

    for images, targets in data_loader:
        images = images.to(
            selected_device,
            non_blocking=True,
        )

        targets = targets.to(
            selected_device,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        batch_size = targets.shape[0]

        total_loss += float(
            loss.detach().item()
        ) * batch_size

        correct_count, sample_count = (
            calculate_batch_accuracy(
                logits.detach(),
                targets,
            )
        )

        total_correct += correct_count
        total_samples += sample_count

    if total_samples == 0:
        raise ValueError(
            "The training DataLoader produced no samples."
        )

    duration = time.perf_counter() - epoch_start

    return EpochMetrics(
        loss=total_loss / total_samples,
        accuracy=total_correct / total_samples,
        correct_predictions=total_correct,
        sample_count=total_samples,
        duration_seconds=duration,
    )


def run_validation_epoch(
    *,
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device | str,
) -> EpochMetrics:
    """Evaluate a model for one complete validation epoch."""

    if not isinstance(model, nn.Module):
        raise TypeError(
            "model must be a torch.nn.Module."
        )

    selected_device = torch.device(device)

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    epoch_start = time.perf_counter()

    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(
                selected_device,
                non_blocking=True,
            )

            targets = targets.to(
                selected_device,
                non_blocking=True,
            )

            logits = model(images)
            loss = criterion(logits, targets)

            batch_size = targets.shape[0]

            total_loss += float(
                loss.item()
            ) * batch_size

            correct_count, sample_count = (
                calculate_batch_accuracy(
                    logits,
                    targets,
                )
            )

            total_correct += correct_count
            total_samples += sample_count

    if total_samples == 0:
        raise ValueError(
            "The validation DataLoader produced no samples."
        )

    duration = time.perf_counter() - epoch_start

    return EpochMetrics(
        loss=total_loss / total_samples,
        accuracy=total_correct / total_samples,
        correct_predictions=total_correct,
        sample_count=total_samples,
        duration_seconds=duration,
    )


def save_training_checkpoint(
    *,
    checkpoint_path: Path | str,
    model: nn.Module,
    optimizer: Optimizer,
    epoch_number: int,
    validation_loss: float,
    validation_accuracy: float,
    class_names: tuple[str, ...],
    training_history: TrainingHistory,
    model_name: str,
) -> Path:
    """Save a reusable PyTorch training checkpoint."""

    path = Path(checkpoint_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "model_name": model_name,
        "epoch": epoch_number,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "validation_loss": validation_loss,
        "validation_accuracy": validation_accuracy,
        "class_names": class_names,
        "training_history": training_history.to_dict(),
    }

    torch.save(
        checkpoint,
        path,
    )

    return path


def get_current_learning_rate(
    optimizer: Optimizer,
) -> float:
    """Return the first optimizer parameter-group learning rate."""

    if not optimizer.param_groups:
        raise ValueError(
            "The optimizer contains no parameter groups."
        )

    return float(
        optimizer.param_groups[0]["lr"]
    )


def train_model(
    *,
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device | str,
    number_of_epochs: int = DEFAULT_NUMBER_OF_EPOCHS,
    checkpoint_path: Path | str,
    class_names: tuple[str, ...],
    model_name: str,
    early_stopping_patience: int = (
        DEFAULT_EARLY_STOPPING_PATIENCE
    ),
    minimum_improvement: float = (
        DEFAULT_MINIMUM_IMPROVEMENT
    ),
    scheduler: LRScheduler | None = None,
) -> TrainingHistory:
    """Train and validate a classification model.

    The checkpoint is overwritten whenever validation loss improves.
    Training stops when validation loss fails to improve for the
    configured number of epochs.
    """

    validated_epoch_count = validate_positive_integer(
        number_of_epochs,
        parameter_name="number_of_epochs",
    )

    selected_device = torch.device(device)

    model.to(selected_device)

    early_stopping = EarlyStopping(
        patience=early_stopping_patience,
        minimum_improvement=minimum_improvement,
    )

    history = TrainingHistory()

    for epoch_index in range(validated_epoch_count):
        epoch_number = epoch_index + 1

        train_metrics = run_training_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=selected_device,
        )

        validation_metrics = run_validation_epoch(
            model=model,
            data_loader=validation_loader,
            criterion=criterion,
            device=selected_device,
        )

        current_learning_rate = get_current_learning_rate(
            optimizer
        )

        history.train_loss.append(
            train_metrics.loss
        )

        history.train_accuracy.append(
            train_metrics.accuracy
        )

        history.validation_loss.append(
            validation_metrics.loss
        )

        history.validation_accuracy.append(
            validation_metrics.accuracy
        )

        history.learning_rates.append(
            current_learning_rate
        )

        history.epoch_durations.append(
            train_metrics.duration_seconds
            + validation_metrics.duration_seconds
        )

        improved, should_stop = early_stopping.update(
            validation_metrics.loss,
            epoch_number,
        )

        if improved:
            history.best_epoch = epoch_number
            history.best_validation_loss = (
                validation_metrics.loss
            )

            save_training_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch_number=epoch_number,
                validation_loss=validation_metrics.loss,
                validation_accuracy=(
                    validation_metrics.accuracy
                ),
                class_names=class_names,
                training_history=history,
                model_name=model_name,
            )

        if scheduler is not None:
            scheduler.step()

        print(
            f"Epoch {epoch_number:02d}/"
            f"{validated_epoch_count:02d} | "
            f"Train loss: {train_metrics.loss:.4f} | "
            f"Train accuracy: "
            f"{train_metrics.accuracy:.4f} | "
            f"Validation loss: "
            f"{validation_metrics.loss:.4f} | "
            f"Validation accuracy: "
            f"{validation_metrics.accuracy:.4f} | "
            f"LR: {current_learning_rate:.6f}"
        )

        if should_stop:
            history.stopped_early = True

            print(
                "Early stopping activated at "
                f"epoch {epoch_number}."
            )

            break

    return history


__all__ = [
    "DEFAULT_EARLY_STOPPING_PATIENCE",
    "DEFAULT_MINIMUM_IMPROVEMENT",
    "DEFAULT_NUMBER_OF_EPOCHS",
    "DEFAULT_RANDOM_SEED",
    "EarlyStopping",
    "EpochMetrics",
    "TrainingHistory",
    "calculate_batch_accuracy",
    "create_classification_loss",
    "get_current_learning_rate",
    "run_training_epoch",
    "run_validation_epoch",
    "save_training_checkpoint",
    "select_device",
    "set_random_seed",
    "train_model",
    "validate_non_negative_integer",
    "validate_positive_integer",
]
