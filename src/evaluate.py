"""
Evaluation Module for PINN Model Assessment

This module provides functions to evaluate trained PINN models against
CFD ground truth data. Metrics are computed for WSS prediction accuracy.

Metrics Computed:
    - RMSE: Root Mean Squared Error (Pa)
    - MAE: Mean Absolute Error (Pa)
    - NRMSE: Normalized RMSE (RMSE / data range)
    - R²: Coefficient of Determination
    - Pearson: Pearson correlation coefficient
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from typing import Dict

from src.config import DEVICE
from src.dataset import PatientDataset
from src.physics import compute_wss_from_velocity
from src.utils import compute_nrmse


def evaluate_model(model: nn.Module, loader: DataLoader, dataset: PatientDataset,
                   compute_wss: bool, coord_scale: torch.Tensor) -> Dict:
    """
    Evaluate trained PINN model on WSS prediction.
    
    Args:
        model: Trained PINN model
        loader: DataLoader with evaluation data
        dataset: PatientDataset (needed for inverse scaling)
        compute_wss: If True, compute WSS from velocity gradients.
                    If False, use network's direct WSS prediction.
        coord_scale: Scale factors for gradient computation
        
    Returns:
        Dictionary of evaluation metrics:
            - RMSE, MAE, NRMSE: Error metrics in Pa
            - R2: Coefficient of determination
            - Pearson: Correlation coefficient
            - n_points: Number of evaluation points
    """
    model.eval()
    
    all_true, all_pred, all_coords = [], [], []
    
    with torch.no_grad():
        for batch in loader:
            coords = batch['coords'].to(DEVICE)
            coords_raw = batch['coords_raw'].numpy()
            wss_raw = batch['wss_raw'].numpy().flatten()
            normals = batch['normals'].to(DEVICE)
            has_wss = batch['has_wss'].numpy().squeeze().astype(bool)
            
            if compute_wss:
                # Compute WSS from velocity
                if has_wss.any():
                    wss_pred = compute_wss_from_velocity(
                        model, coords[has_wss], normals[has_wss], coord_scale
                    ).cpu().numpy().flatten()
                else:
                    continue
            else:
                # Get predicted WSS
                outputs = model(coords)
                wss_pred_scaled = outputs['wss'].cpu().numpy()
                wss_pred = dataset.scaler_y.inverse_transform(wss_pred_scaled).flatten()
                wss_pred = wss_pred[has_wss]
            
            if has_wss.any():
                all_true.append(wss_raw[has_wss])
                all_pred.append(wss_pred)
                all_coords.append(coords_raw[has_wss])
    
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    
    # Metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    corr, _ = pearsonr(y_true, y_pred)
    nrmse = compute_nrmse(y_true, y_pred)
    
    metrics = {
        'RMSE': float(rmse),
        'MAE': float(mae),
        'R2': float(r2),
        'Pearson': float(corr),
        'NRMSE': float(nrmse),
        'n_points': int(len(y_true))
    }
    
    print(f"  RMSE: {rmse:.4f} Pa | NRMSE: {nrmse:.4f} | R²: {r2:.4f} | Pearson: {corr:.4f}")
    
    return metrics
