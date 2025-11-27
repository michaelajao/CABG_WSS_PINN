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
    
    Attributes:
        patience: Number of epochs to wait before stopping
        min_delta: Minimum change to qualify as improvement
        best_value: Best metric value seen so far
        counter: Epochs since last improvement
        best_epoch: Epoch where best value was achieved
    """
    
    def __init__(self, patience: int = 50, min_delta: float = 1e-6, 
                 monitor: str = 'loss'):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum improvement to reset patience counter
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
        Check if training should stop.
        
        Args:
            value: Current metric value
            epoch: Current epoch
            
        Returns:
            True if training should stop
        """
        if value < self.best_value - self.min_delta:
            self.best_value = value
            self.counter = 0
            self.best_epoch = epoch
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                print(f"\n  Early stopping at epoch {epoch + 1}")
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
