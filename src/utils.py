"""
Utility Functions for PINN Training

This module provides helper functions and classes used throughout the
PINN training pipeline.

Components:
    - EarlyStopping: Callback to stop training when loss plateaus
    - compute_nrmse: Normalized RMSE calculation
"""

import numpy as np


class EarlyStopping:
    """
    Early stopping callback to prevent overfitting.
    
    Monitors a metric (typically loss) and stops training if no improvement
    is seen for a specified number of epochs (patience).
    
    Uses relative tolerance for robustness across different loss scales.
    
    Attributes:
        patience: Number of epochs to wait before stopping
        min_delta: Minimum relative change to qualify as improvement (default 0.1%)
        best_value: Best metric value seen so far
        counter: Epochs since last improvement
        best_epoch: Epoch where best value was achieved
    """
    
    def __init__(self, patience: int = 50, min_delta: float = 0.001, 
                 monitor: str = 'loss'):
        """
        Initialize early stopping with relative tolerance.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum relative improvement to reset patience counter.
                      Default 0.001 = 0.1% improvement required.
                      E.g., loss must drop from 0.0033 to < 0.00330 to count as improvement.
            monitor: Name of metric being monitored (for logging)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.best_value = float('inf')
        self.counter = 0
        self.best_epoch = 0
        self.should_stop = False
    
    def __call__(self, value: float, epoch: int) -> bool:
        """
        Check if training should stop using relative tolerance.
        
        Args:
            value: Current metric value
            epoch: Current epoch
            
        Returns:
            True if training should stop
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
                print(f"  Best {self.monitor}: {self.best_value:.6f} at epoch {self.best_epoch + 1}")
        
        return self.should_stop

def compute_nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Normalized Root Mean Squared Error.
    
    NRMSE = RMSE / (max(y_true) - min(y_true))
    
    This normalizes the error by the data range, making it easier to 
    compare across datasets with different scales.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        NRMSE value (unitless, typically 0-1 for good predictions)
    """
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    data_range = np.max(y_true) - np.min(y_true) + 1e-10
    return rmse / data_range
