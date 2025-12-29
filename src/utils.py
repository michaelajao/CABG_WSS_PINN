"""
Utility Functions for PINN Training.

This module provides helper functions and classes used throughout the
PINN training pipeline.

Attributes:
    EPSILON (float): Small constant for numerical stability (1e-10).

Classes:
    EarlyStopping: Callback to stop training when loss plateaus.
    ReLoBRaLo: Self-adaptive loss weighting for multi-objective PINNs.

Functions:
    compute_normalised_rmse: Calculate Normalised Root Mean Squared Error.
"""

from typing import Optional, List, Union

import numpy as np
from tqdm import tqdm

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
                tqdm.write(f"\n  Early stopping triggered at epoch {epoch + 1}")
                tqdm.write(f"  No improvement for {self.patience} epochs")
                tqdm.write(
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
# ADAPTIVE LOSS WEIGHTING
# =============================================================================

class ReLoBRaLo:
    """
    Relative Loss Balancing with Random Lookback (ReLoBRaLo).

    Self-adaptive loss weighting for multi-objective PINNs based on:
    Bischof & Kraus, "Multi-Objective Loss Balancing for Physics-Informed
    Deep Learning", arXiv:2110.09813

    The method balances loss terms based on their relative changes using
    exponential moving averages with random lookback for stability.

    Weight update formula:
        lambda_i(t) = alpha * lambda_i(t-1) + (1-alpha) * lambda_hat_i(t)

    where lambda_hat uses softmax over loss ratios:
        lambda_hat_i = softmax(L_i(t-tau) / L_i(t) / T)
        tau ~ Bernoulli(rho) selects lookback (initial vs previous)

    IMPORTANT: This implementation supports minimum weight bounds to prevent
    data losses (WSS, velocity) from collapsing. Without bounds, physics
    losses can dominate and the model won't fit the actual data.

    Attributes:
        num_losses: Number of loss terms being balanced.
        alpha: EMA decay rate (higher = more stable, slower adaptation).
        T: Temperature for softmax (lower = more aggressive balancing).
        rho: Probability of using random lookback to initial losses.
        weights: Current loss weights (sum to num_losses).
        min_weights: Minimum allowed weight per loss term.
        max_weights: Maximum allowed weight per loss term.

    Example:
        >>> # Ensure data losses (indices 0,1) never drop below 1.0
        >>> balancer = ReLoBRaLo(
        ...     num_losses=5, alpha=0.999, T=1.0,
        ...     min_weights=[1.0, 1.0, 0.1, 0.1, 0.1]
        ... )
        >>> for epoch in range(epochs):
        ...     losses = [wss_loss, vel_loss, ns_loss, cont_loss, wss_phys]
        ...     weights = balancer.update(losses)
        ...     total_loss = sum(w * l for w, l in zip(weights, losses))
    """

    def __init__(
        self,
        num_losses: int,
        alpha: float = 0.999,
        T: float = 1.0,
        rho: float = 0.99,
        min_weights: Optional[List[float]] = None,
        max_weights: Optional[List[float]] = None
    ) -> None:
        """
        Initialize ReLoBRaLo adaptive loss weighting.

        Args:
            num_losses: Number of loss terms to balance.
            alpha: EMA decay rate (0.99-0.999 recommended).
                Higher values = more stable but slower adaptation.
            T: Temperature for softmax (default 1.0).
                Lower values = more aggressive balancing.
                Higher values = more uniform weights.
            rho: Probability of using lookback to initial losses (0.95-0.99).
                Helps escape local minima by occasionally referencing start.
            min_weights: Minimum weight per loss term (length = num_losses).
                Use to prevent important losses from being ignored.
                Default: [0.1] * num_losses (no loss completely ignored).
            max_weights: Maximum weight per loss term (length = num_losses).
                Use to prevent any single loss from dominating.
                Default: [10.0] * num_losses.
        """
        self.num_losses = num_losses
        self.alpha = alpha
        self.T = T
        self.rho = rho

        # Set minimum/maximum weight bounds
        if min_weights is not None:
            if len(min_weights) != num_losses:
                raise ValueError(
                    f"min_weights length ({len(min_weights)}) must match "
                    f"num_losses ({num_losses})"
                )
            self.min_weights = np.array(min_weights, dtype=np.float64)
        else:
            self.min_weights = np.full(num_losses, 0.1, dtype=np.float64)

        if max_weights is not None:
            if len(max_weights) != num_losses:
                raise ValueError(
                    f"max_weights length ({len(max_weights)}) must match "
                    f"num_losses ({num_losses})"
                )
            self.max_weights = np.array(max_weights, dtype=np.float64)
        else:
            self.max_weights = np.full(num_losses, 10.0, dtype=np.float64)

        # Initialize weights uniformly (sum to num_losses for scale preservation)
        self.weights = np.ones(num_losses, dtype=np.float64)

        # Loss history
        self.init_losses: Optional[np.ndarray] = None
        self.prev_losses: Optional[np.ndarray] = None
        self.iteration: int = 0

    def update(self, losses: List[Union[float, 'torch.Tensor']]) -> np.ndarray:
        """
        Update weights based on current loss values.

        Uses numerically stable computation with:
        - Log-space softmax to prevent overflow
        - Clipping to prevent extreme values
        - EMA smoothing for stability

        Args:
            losses: List of current loss values (one per term).
                Can be floats or PyTorch tensors.

        Returns:
            Updated weights as numpy array (length = num_losses).
            Weights sum to num_losses to preserve loss scale.
        """
        # Convert to numpy, handle tensors
        loss_arr = np.array([
            l.item() if hasattr(l, 'item') else float(l)
            for l in losses
        ], dtype=np.float64)

        # Clip to prevent numerical issues
        loss_arr = np.clip(loss_arr, 1e-10, 1e10)

        # First iteration: store initial losses, return uniform weights
        if self.iteration == 0:
            self.init_losses = loss_arr.copy()
            self.prev_losses = loss_arr.copy()
            self.iteration += 1
            return self.weights.copy()

        # Second iteration: no EMA yet, just compute lambda_hat
        if self.iteration == 1:
            alpha_eff = 0.0  # Skip EMA, use raw lambda_hat
        else:
            alpha_eff = self.alpha

        # Random lookback: compare to initial or previous losses
        use_lookback = np.random.rand() < self.rho
        if use_lookback and self.iteration > 1:
            ref_losses = self.init_losses
        else:
            ref_losses = self.prev_losses

        # Compute loss ratios: L_ref / L_current (higher ratio = loss decreased)
        # Losses that decreased more get higher weights (to balance)
        ratios = ref_losses / (loss_arr + EPSILON)

        # Numerically stable softmax in log-space
        log_ratios = np.log(ratios + EPSILON) / self.T
        log_ratios_stable = log_ratios - np.max(log_ratios)  # Subtract max
        exp_ratios = np.exp(log_ratios_stable)
        lambda_hat = self.num_losses * exp_ratios / (exp_ratios.sum() + EPSILON)

        # Exponential moving average update
        self.weights = alpha_eff * self.weights + (1.0 - alpha_eff) * lambda_hat

        # Ensure weights sum to num_losses (scale preservation)
        self.weights = self.num_losses * self.weights / (self.weights.sum() + EPSILON)

        # Apply per-loss weight bounds (prevents data losses from collapsing)
        self.weights = np.clip(self.weights, self.min_weights, self.max_weights)
        self.weights = self.num_losses * self.weights / (self.weights.sum() + EPSILON)

        # Update history
        self.prev_losses = loss_arr.copy()
        self.iteration += 1

        return self.weights.copy()

    def get_weights(self) -> np.ndarray:
        """Return current weights without updating."""
        return self.weights.copy()

    def reset(self) -> None:
        """Reset to initial state for new training run."""
        self.weights = np.ones(self.num_losses, dtype=np.float64)
        self.init_losses = None
        self.prev_losses = None
        self.iteration = 0


# =============================================================================
# METRICS
# =============================================================================

def compute_normalised_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Normalised Root Mean Squared Error.

    NRMSE = RMSE / (max(y_true) - min(y_true))

    This normalises the error by the data range, making it easier to
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
        >>> nrmse = compute_normalised_rmse(y_true, y_pred)  # ~0.05 (5% of range)
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )

    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    data_range = np.max(y_true) - np.min(y_true) + EPSILON
    return rmse / data_range
