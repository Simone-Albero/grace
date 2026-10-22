"""PyTorch MLP classifier: losses, class weights, loaders, training."""

__all__ = [
    "LOSSES",
    "MODELS",
    "build_loader",
    "class_weights",
    "predict",
    "train",
]

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class CrossEntropyLoss(nn.Module):
    """Cross-entropy with class weights and label smoothing.

    The loss is averaged over the batch.

    Attributes:
        label_smoothing: The smoothing of the target distribution.
        class_weight: The weight of each class, a buffer; None for equal
            weights.
    """

    def __init__(
        self,
        *,
        class_weight: Sequence[float] | None = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.label_smoothing = label_smoothing
        if class_weight is not None:
            self.register_buffer(
                "class_weight", torch.tensor(class_weight, dtype=torch.float32)
            )
        else:
            self.class_weight = None

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        """Return the batch's mean weighted cross-entropy.

        Args:
            logits: The class logits, shape (n, n_classes).
            target: The class ids, shape (n,).

        Returns:
            The scalar loss.
        """
        # Reducing by hand divides by the batch size; reduction="mean"
        # would divide by the sum of the weights of the batch.
        return F.cross_entropy(
            logits,
            target,
            weight=self.class_weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        ).mean()


class FocalLoss(nn.Module):
    """Focal loss alpha_t * (1 - p_t)^gamma * CE, batch mean.

    The CE term is label-smoothed and p_t is not.

    Attributes:
        gamma: The focusing exponent; 0 gives the plain weighted CE.
        label_smoothing: The smoothing of the CE term.
        class_weight: The alpha of each class, a buffer; None for equal
            weights.
    """

    def __init__(
        self,
        *,
        class_weight: Sequence[float] | None = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        if class_weight is not None:
            self.register_buffer(
                "class_weight", torch.tensor(class_weight, dtype=torch.float32)
            )
        else:
            self.class_weight = None

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        """Return the batch's mean focal loss.

        Args:
            logits: The class logits, shape (n, n_classes).
            target: The class ids, shape (n,).

        Returns:
            The scalar loss.
        """
        ce_loss = F.cross_entropy(
            logits, target, reduction="none", label_smoothing=self.label_smoothing
        )
        p_t = torch.softmax(logits, dim=1).gather(1, target.unsqueeze(1)).squeeze(1)
        loss = (1 - p_t) ** self.gamma * ce_loss
        if self.class_weight is not None:
            loss = self.class_weight.gather(0, target) * loss
        return loss.mean()


LOSSES: dict[str, type[nn.Module]] = {
    "cross_entropy": CrossEntropyLoss,
    "focal": FocalLoss,
}


class MLPClassifier(nn.Module):
    """Multi-layer perceptron over numerical and categorical features.

    Categorical codes go through embeddings.

    Attributes:
        embeddings: One embedding per categorical column; code 0 is a
            missing or unknown category.
        body: The hidden layers.
        head: The output layer, one logit per class.
    """

    def __init__(
        self,
        num_numerical_features: int,
        num_classes: int,
        hidden_dims: Sequence[int],
        *,
        cardinalities: Sequence[int] = (),
        max_emb_dim: int = 50,
        dropout: float = 0.0,
    ) -> None:
        # The order of every layer creation and init below fixes what
        # the global seed draws, and the registration order fixes the
        # sum in clip_grad_norm_.
        super().__init__()
        dims = [
            min(max_emb_dim, int(cardinality**0.5)) for cardinality in cardinalities
        ]
        self.embeddings = nn.ModuleList(
            nn.Embedding(cardinality + 1, emb_dim, padding_idx=0)
            for cardinality, emb_dim in zip(cardinalities, dims)
        )
        layers: list[nn.Module] = []
        width = num_numerical_features + sum(dims)
        for dim in hidden_dims[:-1]:
            layers += [
                nn.Linear(width, dim, bias=False),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
            ]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            width = dim
        layers.append(nn.Linear(width, hidden_dims[-1], bias=False))
        self.body = nn.Sequential(*layers)
        for module in self.body:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
        self.head = nn.Linear(hidden_dims[-1], num_classes)

    def forward(self, x_num: Tensor, x_cat: Tensor) -> Tensor:
        """Return the class logits of the rows.

        Args:
            x_num: The numerical features, shape (n, num_numerical).
            x_cat: The categorical codes, shape (n, num_categorical).

        Returns:
            The logits, shape (n, num_classes).
        """
        embedded = [layer(x_cat[:, i]) for i, layer in enumerate(self.embeddings)]
        return self.head(self.body(torch.cat([x_num, *embedded], dim=1)))


MODELS: dict[str, type[nn.Module]] = {"mlp": MLPClassifier}


def class_weights(labels: pd.Series) -> dict[int, float]:
    """Return each class's weight, a log-damped inverse frequency.

    Args:
        labels: The class id of every training row.

    Returns:
        The weight by class id, scaled to a maximum of 1.
    """
    counts = labels.value_counts()
    weights = len(labels) / (len(counts) * counts)
    log_weights = np.log1p(weights)
    return (log_weights / log_weights.max()).to_dict()


def _tensors(
    rows: pd.DataFrame, num_cols: list[str], cat_cols: list[str]
) -> tuple[Tensor, Tensor]:
    """Return the numerical (float32) and categorical (long) tensors."""
    x_num = (
        torch.tensor(rows[num_cols].to_numpy(), dtype=torch.float32)
        if num_cols
        else torch.empty(len(rows), 0, dtype=torch.float32)
    )
    x_cat = (
        torch.tensor(rows[cat_cols].to_numpy(), dtype=torch.long)
        if cat_cols
        else torch.empty(len(rows), 0, dtype=torch.long)
    )
    return x_num, x_cat


def build_loader(
    rows: pd.DataFrame,
    *,
    num_cols: list[str],
    cat_cols: list[str],
    label_col: str,
    seed: int,
    **params: object,
) -> DataLoader:
    """Return batches of (numerical, categorical, label) tensors.

    Args:
        rows: The encoded rows.
        num_cols: The numerical feature names.
        cat_cols: The categorical feature names.
        label_col: The label column.
        seed: The seed of the loader's own generator.
        **params: The keyword arguments of `DataLoader`, such as
            `batch_size` and `shuffle`.

    Returns:
        The loader.
    """
    labels = torch.tensor(rows[label_col].to_numpy(), dtype=torch.long)
    # Its own generator: a loader draws a seed each time it is iterated,
    # shuffling or not, and from the global stream that draw would land
    # between two epochs' dropout draws.
    return DataLoader(
        TensorDataset(*_tensors(rows, num_cols, cat_cols), labels),
        generator=torch.Generator().manual_seed(seed),
        **params,
    )


def _mean(values: Sequence[float]) -> float:
    """Return the left-to-right mean; sum() would round differently."""
    total = 0.0
    for value in values:
        total += value
    return total / len(values)


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    epoch: int,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    max_grad_norm: float,
    device: torch.device,
) -> list[float]:
    """Run one training epoch and return the loss of every step."""
    model.train()
    losses = []
    for batch in loader:
        x_num, x_cat, y = (tensor.to(device, non_blocking=True) for tensor in batch)
        optimizer.zero_grad()
        loss = loss_fn(model(x_num, x_cat), y)
        if not torch.isfinite(loss):
            raise ValueError(
                f"Training loss is {loss.item()} at epoch {epoch}: lower "
                "optimizer.params.lr or scheduler.params.max_lr, or "
                "fit.training.max_grad_norm."
            )
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
    return losses


def _val_loss(
    model: nn.Module,
    loader: DataLoader,
    *,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    """Return the mean loss of the model over the loader's batches."""
    model.eval()
    with torch.no_grad():
        losses = []
        for batch in loader:
            x_num, x_cat, y = (tensor.to(device, non_blocking=True) for tensor in batch)
            losses.append(loss_fn(model(x_num, x_cat), y).item())
    return _mean(losses)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epochs: int,
    max_grad_norm: float,
    patience: int,
    min_delta: float,
    device: torch.device,
) -> list[float]:
    """Train with early stopping on val loss, keeping the best epoch.

    Args:
        model: The model, trained in place; it ends with the weights of
            the first epoch of least val loss.
        train_loader: The train batches.
        val_loader: The validation batches.
        loss_fn: The loss, for both splits.
        optimizer: The optimizer, stepped once per batch.
        scheduler: The learning-rate schedule, stepped once per batch.
        epochs: The most epochs to run.
        max_grad_norm: The gradient norm clipped to.
        patience: The epochs without a val-loss gain of `min_delta`
            that stop the training.
        min_delta: The smallest val-loss gain that resets the patience.
        device: The device of the batches.

    Returns:
        The train loss of every step, over all the epochs run.

    Raises:
        ValueError: If a train loss is not finite.
    """
    history: list[float] = []
    best_loss, best_epoch, best_state = None, 0, None
    stop_score, stale = None, 0
    for epoch in range(1, epochs + 1):
        losses = _train_epoch(
            model,
            train_loader,
            epoch=epoch,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            max_grad_norm=max_grad_norm,
            device=device,
        )
        history += losses
        val_loss = _val_loss(model, val_loader, loss_fn=loss_fn, device=device)
        logger.info("Epoch [%d] Train Loss: %.6f", epoch, _mean(losses))
        logger.info("Epoch [%d] Val Loss: %.6f", epoch, val_loss)

        # Strict < keeps the earliest of tied best epochs.
        if best_state is None or val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = {
                name: tensor.detach().clone()
                for name, tensor in model.state_dict().items()
            }
        score = -val_loss
        if stop_score is None:
            stop_score = score
        elif score <= stop_score + min_delta:
            stop_score = max(stop_score, score)
            stale += 1
            if stale >= patience:
                logger.info("Early stopping after epoch %d", epoch)
                break
        else:
            stop_score, stale = score, 0

    model.load_state_dict(best_state)
    logger.info("Best epoch %d (val loss %.6f) restored.", best_epoch, best_loss)
    return history


def predict(
    model: nn.Module,
    rows: pd.DataFrame,
    *,
    num_cols: list[str],
    cat_cols: list[str],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict the class and class probabilities of every row.

    Args:
        model: The trained model.
        rows: The encoded rows.
        num_cols: The numerical feature names.
        cat_cols: The categorical feature names.
        device: The device of the model.

    Returns:
        The predicted class ids and the probabilities, shape
        (n, n_classes), from one forward pass.
    """
    model.eval()
    with torch.no_grad():
        logits = model(
            *(tensor.to(device) for tensor in _tensors(rows, num_cols, cat_cols))
        )
    probabilities = F.softmax(logits.cpu(), dim=1)
    return probabilities.argmax(dim=1).numpy(), probabilities.numpy()
