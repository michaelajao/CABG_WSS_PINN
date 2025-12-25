"""
Utility Functions for PINN Training.

This module provides helper functions and classes used throughout the
PINN training pipeline.

Attributes:
    EPSILON (float): Small constant for numerical stability (1e-10).

Classes:
    EarlyStopping: Callback to stop training when loss plateaus.

Functions:
    compute_nrmse: Calculate Normalized Root Mean Squared Error.
"""

from typing import Optional

import numpy as np

# =============================================================================
# CONSTANTS
# =============================================================================

EPSILON: float = 1e-10  # Small constant for numerical stability


# =============================================================================
# EARLY STOPPING
# =============================================================================

class EarlyStopping:
    """
    Early stopping callback to prevent overfitting.

    Monitors a metric (typically loss) and stops training if no improvement
    is seen for a specified number of epochs (patience). Uses relative
    tolerance for robustness across different loss scales.

    Attributes:
        patience (int): Number of epochs to wait before stopping.
        min_delta (float): Minimum relative improvement required (default 0.1%).
        monitor (str): Name of metric being monitored.
        best_value (float): Best metric value seen so far.
        counter (int): Epochs since last improvement.
        best_epoch (int): Epoch where best value was achieved.
        should_stop (bool): Whether training should stop.

    Example:
        >>> stopper = EarlyStopping(patience=50, min_delta=0.001)
        >>> for epoch in range(max_epochs):
        ...     loss = train_one_epoch()
        ...     if stopper(loss, epoch):
        ...         break
    """

    def __init__(
        self,
        patience: int = 50,
        min_delta: float = 0.001,
        monitor: str = 'loss'
    ) -> None:
        """
        Initialize early stopping with relative tolerance.

        Args:
            patience: Number of epochs to wait before stopping.
            min_delta: Minimum relative improvement to reset patience counter.
                Default 0.001 = 0.1% improvement required.
                E.g., loss must drop from 0.0033 to < 0.00330 to count.
            monitor: Name of metric being monitored (for logging).
        """
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.best_value: float = float('inf')
        self.counter: int = 0
        self.best_epoch: int = 0
        self.should_stop: bool = False

    def __call__(self, value: float, epoch: int) -> bool:
        """
        Check if training should stop using relative tolerance.

        Args:
            value: Current metric value.
            epoch: Current epoch number.

        Returns:
            True if training should stop, False otherwise.
        """
        # Use relative improvement: new_value < best_value * (1 - min_delta)
        # This makes the threshold adaptive to the current loss scale
        threshold = self.best_value * (1.0 - self.min_delta)

        if value < threshold:
            self.best_value = value
            self.counter = 0
            self.best_epoch = epoch
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                print(f"\n  Early stopping triggered at epoch {epoch + 1}")
                print(f"  No improvement for {self.patience} epochs")
                print(
                    f"  Best {self.monitor}: {self.best_value:.6f} "
                    f"at epoch {self.best_epoch + 1}"
                )

        return self.should_stop

    def reset(self) -> None:
        """Reset the early stopping state for a new training run."""
        self.best_value = float('inf')
        self.counter = 0
        self.best_epoch = 0
        self.should_stop = False


# =============================================================================
# METRICS
# =============================================================================

def compute_nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Normalized Root Mean Squared Error.

    NRMSE = RMSE / (max(y_true) - min(y_true))

    This normalizes the error by the data range, making it easier to
    compare across datasets with different scales. A value of 0.01
    indicates the RMSE is 1% of the data range.

    Args:
        y_true: Ground truth values with shape (N,).
        y_pred: Predicted values with shape (N,).

    Returns:
        NRMSE value (unitless, typically 0-0.1 for good predictions).

    Raises:
        ValueError: If input arrays have different shapes.

    Example:
        >>> y_true = np.array([0, 5, 10])
        >>> y_pred = np.array([0.5, 5.5, 9.5])
        >>> nrmse = compute_nrmse(y_true, y_pred)  # ~0.05 (5% of range)
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )

    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    data_range = np.max(y_true) - np.min(y_true) + EPSILON
    return rmse / data_range
