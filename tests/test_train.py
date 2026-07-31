"""Tests for reusable model-training utilities."""

from pathlib import Path

import pytest
import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

from src.train import (
    EarlyStopping,
    TrainingHistory,
    calculate_batch_accuracy,
    create_classification_loss,
    run_training_epoch,
    run_validation_epoch,
    save_training_checkpoint,
    select_device,
    set_random_seed,
    train_model,
)


class TinyClassifier(nn.Module):
    """Small network used only for fast training tests."""

    def __init__(self) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4, 2),
        )

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(inputs)


def create_tiny_data_loader(
    *,
    batch_size: int = 4,
) -> DataLoader:
    """Create a small deterministic classification dataset."""

    inputs = torch.tensor(
        [
            [[[0.0, 0.0], [0.0, 0.0]]],
            [[[0.1, 0.0], [0.0, 0.1]]],
            [[[1.0, 1.0], [1.0, 1.0]]],
            [[[0.9, 1.0], [1.0, 0.9]]],
            [[[0.0, 0.1], [0.1, 0.0]]],
            [[[1.0, 0.9], [0.9, 1.0]]],
            [[[0.2, 0.0], [0.0, 0.2]]],
            [[[0.8, 1.0], [1.0, 0.8]]],
        ],
        dtype=torch.float32,
    )

    targets = torch.tensor(
        [0, 0, 1, 1, 0, 1, 0, 1],
        dtype=torch.long,
    )

    dataset = TensorDataset(
        inputs,
        targets,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )


def test_select_device_returns_cpu_when_requested() -> None:
    """Explicit CPU selection should work."""

    device = select_device("cpu")

    assert device == torch.device("cpu")


def test_select_device_rejects_unknown_device() -> None:
    """Unsupported device names should be rejected."""

    with pytest.raises(ValueError):
        select_device("quantum")


def test_random_seed_reproduces_torch_values() -> None:
    """Resetting the same seed should reproduce random tensors."""

    set_random_seed(42)
    first_tensor = torch.rand(4)

    set_random_seed(42)
    second_tensor = torch.rand(4)

    assert torch.equal(
        first_tensor,
        second_tensor,
    )


def test_classification_loss_accepts_valid_weights() -> None:
    """Positive class weights should configure cross-entropy."""

    class_weights = torch.tensor(
        [0.5, 1.5],
        dtype=torch.float32,
    )

    criterion = create_classification_loss(
        class_weights=class_weights,
        device="cpu",
    )

    assert isinstance(
        criterion,
        nn.CrossEntropyLoss,
    )

    assert torch.equal(
        criterion.weight,
        class_weights,
    )


def test_classification_loss_rejects_non_positive_weights() -> None:
    """Zero and negative class weights should be rejected."""

    with pytest.raises(ValueError):
        create_classification_loss(
            class_weights=torch.tensor(
                [1.0, 0.0]
            )
        )


def test_calculate_batch_accuracy() -> None:
    """Predicted classes should be compared with labels."""

    logits = torch.tensor(
        [
            [2.0, 0.1],
            [0.2, 1.8],
            [1.5, 0.3],
        ]
    )

    targets = torch.tensor(
        [0, 1, 1]
    )

    correct, total = calculate_batch_accuracy(
        logits,
        targets,
    )

    assert correct == 2
    assert total == 3


def test_training_epoch_updates_model_parameters() -> None:
    """A training epoch should modify trainable parameters."""

    set_random_seed(42)

    model = TinyClassifier()
    loader = create_tiny_data_loader()

    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(
        model.parameters(),
        lr=0.1,
    )

    parameters_before = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]

    metrics = run_training_epoch(
        model=model,
        data_loader=loader,
        criterion=criterion,
        optimizer=optimizer,
        device="cpu",
    )

    parameters_after = list(
        model.parameters()
    )

    changed_parameters = [
        not torch.equal(before, after)
        for before, after in zip(
            parameters_before,
            parameters_after,
        )
    ]

    assert any(changed_parameters)
    assert metrics.sample_count == 8
    assert 0.0 <= metrics.accuracy <= 1.0
    assert metrics.loss >= 0.0


def test_validation_epoch_does_not_update_parameters() -> None:
    """Validation must not modify model parameters."""

    model = TinyClassifier()
    loader = create_tiny_data_loader()

    criterion = nn.CrossEntropyLoss()

    parameters_before = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]

    metrics = run_validation_epoch(
        model=model,
        data_loader=loader,
        criterion=criterion,
        device="cpu",
    )

    parameters_after = list(
        model.parameters()
    )

    for before, after in zip(
        parameters_before,
        parameters_after,
    ):
        assert torch.equal(
            before,
            after,
        )

    assert metrics.sample_count == 8


def test_early_stopping_tracks_improvement() -> None:
    """Improving validation loss should reset patience."""

    early_stopping = EarlyStopping(
        patience=2
    )

    improved, should_stop = early_stopping.update(
        1.0,
        epoch_number=1,
    )

    assert improved
    assert not should_stop
    assert early_stopping.best_epoch == 1

    improved, should_stop = early_stopping.update(
        0.8,
        epoch_number=2,
    )

    assert improved
    assert not should_stop
    assert early_stopping.best_epoch == 2


def test_early_stopping_activates_after_patience() -> None:
    """Repeated non-improvement should stop training."""

    early_stopping = EarlyStopping(
        patience=2
    )

    early_stopping.update(
        1.0,
        epoch_number=1,
    )

    _, should_stop = early_stopping.update(
        1.1,
        epoch_number=2,
    )

    assert not should_stop

    _, should_stop = early_stopping.update(
        1.2,
        epoch_number=3,
    )

    assert should_stop


def test_checkpoint_is_saved(
    tmp_path: Path,
) -> None:
    """Checkpoint saving should create a readable file."""

    model = TinyClassifier()
    optimizer = SGD(
        model.parameters(),
        lr=0.1,
    )

    history = TrainingHistory(
        train_loss=[1.0],
        validation_loss=[0.9],
    )

    checkpoint_path = (
        tmp_path / "model.pt"
    )

    result_path = save_training_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        epoch_number=1,
        validation_loss=0.9,
        validation_accuracy=0.5,
        class_names=("class_a", "class_b"),
        training_history=history,
        model_name="tiny_classifier",
    )

    assert result_path.exists()

    checkpoint = torch.load(
        result_path,
        map_location="cpu",
        weights_only=False,
    )

    assert checkpoint["epoch"] == 1
    assert checkpoint["model_name"] == "tiny_classifier"
    assert checkpoint["class_names"] == (
        "class_a",
        "class_b",
    )

    assert "model_state_dict" in checkpoint
    assert "optimizer_state_dict" in checkpoint


def test_train_model_creates_best_checkpoint(
    tmp_path: Path,
) -> None:
    """The full training loop should save its best state."""

    set_random_seed(42)

    model = TinyClassifier()

    train_loader = create_tiny_data_loader()
    validation_loader = create_tiny_data_loader()

    criterion = nn.CrossEntropyLoss()

    optimizer = SGD(
        model.parameters(),
        lr=0.1,
    )

    checkpoint_path = (
        tmp_path / "best_model.pt"
    )

    history = train_model(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        criterion=criterion,
        optimizer=optimizer,
        device="cpu",
        number_of_epochs=3,
        checkpoint_path=checkpoint_path,
        class_names=("class_a", "class_b"),
        model_name="tiny_classifier",
        early_stopping_patience=3,
    )

    assert checkpoint_path.exists()
    assert len(history.train_loss) >= 1
    assert len(history.validation_loss) >= 1
    assert history.best_epoch is not None
    assert history.best_validation_loss is not None
    