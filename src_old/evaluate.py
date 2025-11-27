"""
Evaluation module for PINN

Contains functions for model evaluation, metrics computation, and prediction.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Tuple, Optional
import json
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score
)

from .config import DEVICE, RESULTS_PATH, HIGH_RISK_WSS_THRESHOLD, COLUMN_MAPPINGS
from .dataset import HemodynamicsDataset
from .utils import get_all_datasets, parse_cfd_file, get_column_safe, extract_vessel_type


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    dataset: HemodynamicsDataset,
    verbose: bool = True
) -> Tuple[Dict, pd.DataFrame]:
    """
    Comprehensive model evaluation on test set

    Computes regression metrics (MSE, MAE, R²) and classification metrics
    for high-risk zone detection.

    Args:
        model: Trained PINN model
        test_loader: Test data loader
        dataset: Full dataset (needed for inverse transform)
        verbose: Whether to print results

    Returns:
        Tuple of (metrics_dict, results_dataframe)
    """
    if verbose:
        print("\n[MODEL EVALUATION]")

    model.eval()
    model = model.to(DEVICE)

    all_preds = []
    all_true = []
    all_coords = []

    # Collect predictions
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating", disable=not verbose):
            coords = batch['coords'].to(DEVICE)
            wss_true = batch['wss_raw'].cpu().numpy()
            has_wss = batch['has_wss'].cpu().numpy()

            # Predict
            outputs = model(coords)
            wss_pred_scaled = outputs['wss'].cpu().numpy()

            # Inverse transform to get original scale
            wss_pred = dataset.scaler_y.inverse_transform(
                wss_pred_scaled.reshape(-1, 1)
            ).flatten()

            # Only keep predictions for points with WSS labels (wall points)
            wall_mask = has_wss.astype(bool)
            all_preds.append(wss_pred[wall_mask])
            all_true.append(wss_true.flatten()[wall_mask])
            all_coords.append(batch['coords_raw'].cpu().numpy()[wall_mask])

    # Concatenate results
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_true)
    coords = np.vstack(all_coords)

    # =========================================================================
    # Regression Metrics
    # =========================================================================
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    r2 = r2_score(y_true, y_pred)
    corr, _ = pearsonr(y_true, y_pred)

    # =========================================================================
    # High-Risk Classification Metrics
    # =========================================================================
    y_true_high_risk = (y_true > HIGH_RISK_WSS_THRESHOLD).astype(int)
    y_pred_high_risk = (y_pred > HIGH_RISK_WSS_THRESHOLD).astype(int)

    accuracy = accuracy_score(y_true_high_risk, y_pred_high_risk)
    precision = precision_score(y_true_high_risk, y_pred_high_risk, zero_division=0)
    recall = recall_score(y_true_high_risk, y_pred_high_risk, zero_division=0)
    f1 = f1_score(y_true_high_risk, y_pred_high_risk, zero_division=0)

    # ROC-AUC (use continuous predictions)
    if len(np.unique(y_true_high_risk)) > 1:
        auc = roc_auc_score(y_true_high_risk, y_pred)
    else:
        auc = None

    # Compile metrics
    metrics = {
        'MSE': float(mse),
        'RMSE': float(rmse),
        'MAE': float(mae),
        'MAPE': float(mape),
        'R2': float(r2),
        'Pearson_Correlation': float(corr),
        'High_Risk_Threshold': float(HIGH_RISK_WSS_THRESHOLD),
        'High_Risk_Accuracy': float(accuracy),
        'High_Risk_Precision': float(precision),
        'High_Risk_Recall': float(recall),
        'High_Risk_F1': float(f1),
        'High_Risk_AUC': float(auc) if auc is not None else None
    }

    # =========================================================================
    # Print Results
    # =========================================================================
    if verbose:
        print(f"  RMSE: {rmse:.4f} Pa | MAE: {mae:.4f} Pa | R²: {r2:.4f}")
        print(f"  High-Risk F1: {f1:.4f} | Accuracy: {accuracy:.4f}")

    # =========================================================================
    # Create Results DataFrame
    # =========================================================================
    results_df = pd.DataFrame({
        'X': coords[:, 0],
        'Y': coords[:, 1],
        'Z': coords[:, 2],
        'WSS_True': y_true,
        'WSS_Pred': y_pred,
        'Error': y_pred - y_true,
        'Abs_Error': np.abs(y_pred - y_true),
        'Percent_Error': np.abs((y_pred - y_true) / (y_true + 1e-10)) * 100,
        'High_Risk_True': y_true_high_risk,
        'High_Risk_Pred': y_pred_high_risk
    })

    return metrics, results_df


def save_evaluation_results(
    metrics: Dict,
    results_df: pd.DataFrame,
    save_path: Path = RESULTS_PATH
):
    """
    Save evaluation results to disk

    Args:
        metrics: Dictionary of metrics
        results_df: DataFrame with predictions
        save_path: Directory to save results
    """
    # Save metrics JSON
    with open(save_path / "evaluation_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    # Save predictions CSV
    results_df.to_csv(save_path / "test_predictions.csv", index=False)


def save_metrics_text(
    metrics: Dict,
    results_df: pd.DataFrame,
    save_path: Path = RESULTS_PATH,
    filename: str = "metrics_summary.txt"
):
    """Save human-readable metrics summary (including NRMSE) to a TXT file.

    Computes NRMSE on the provided results_df (typically test set).
    """
    if 'WSS_True' in results_df.columns and 'WSS_Pred' in results_df.columns:
        y_true = results_df['WSS_True'].values
        y_pred = results_df['WSS_Pred'].values
        nrmse = _nrmse(y_true, y_pred)
    else:
        nrmse = None

    lines = []
    lines.append("Evaluation Metrics Summary\n")
    if nrmse is not None:
        lines.append(f"NRMSE\t{nrmse:.6f}")
    lines.append(f"MSE\t{metrics.get('MSE', float('nan')):.6f}")
    lines.append(f"RMSE\t{metrics.get('RMSE', float('nan')):.6f}")
    lines.append(f"MAE\t{metrics.get('MAE', float('nan')):.6f}")
    lines.append(f"MAPE_percent\t{metrics.get('MAPE', float('nan')):.2f}")
    lines.append(f"R2\t{metrics.get('R2', float('nan')):.6f}")
    lines.append(f"Pearson_r\t{metrics.get('Pearson_Correlation', float('nan')):.6f}")
    lines.append("")
    lines.append(f"High_Risk_Threshold_Pa\t{metrics.get('High_Risk_Threshold', float('nan')):.3f}")
    lines.append(f"High_Risk_Accuracy\t{metrics.get('High_Risk_Accuracy', float('nan')):.6f}")
    lines.append(f"High_Risk_Precision\t{metrics.get('High_Risk_Precision', float('nan')):.6f}")
    lines.append(f"High_Risk_Recall\t{metrics.get('High_Risk_Recall', float('nan')):.6f}")
    lines.append(f"High_Risk_F1\t{metrics.get('High_Risk_F1', float('nan')):.6f}")
    auc = metrics.get('High_Risk_AUC', None)
    if auc is not None:
        lines.append(f"High_Risk_AUC\t{auc:.6f}")

    save_path.mkdir(parents=True, exist_ok=True)
    txt_path = save_path / filename
    with open(txt_path, 'w') as f:
        f.write('\n'.join(lines))


def predict_on_new_data(
    model: nn.Module,
    coords: np.ndarray,
    scaler_X,
    scaler_y,
    batch_size: int = 8192
) -> np.ndarray:
    """
    Make predictions on new coordinate data

    Args:
        model: Trained PINN model
        coords: Spatial coordinates (N, 3) - [x, y, z]
        scaler_X: Fitted MinMaxScaler for coordinates
        scaler_y: Fitted MinMaxScaler for WSS
        batch_size: Batch size for prediction

    Returns:
        Predicted WSS values (N,)
    """
    model.eval()
    model = model.to(DEVICE)

    # Scale coordinates
    coords_scaled = scaler_X.transform(coords)

    predictions = []

    # Process in batches
    for i in range(0, len(coords_scaled), batch_size):
        batch_coords = coords_scaled[i:i+batch_size]
        batch_tensor = torch.FloatTensor(batch_coords).to(DEVICE)

        with torch.no_grad():
            outputs = model(batch_tensor)
            wss_pred_scaled = outputs['wss'].cpu().numpy()

            # Inverse transform
            wss_pred = scaler_y.inverse_transform(
                wss_pred_scaled.reshape(-1, 1)
            ).flatten()

            predictions.append(wss_pred)

    return np.concatenate(predictions)


def compute_error_statistics(results_df: pd.DataFrame) -> Dict:
    """
    Compute detailed error statistics

    Args:
        results_df: DataFrame with 'Error', 'Abs_Error', 'Percent_Error' columns

    Returns:
        Dictionary with error statistics
    """
    stats = {
        'error_mean': float(results_df['Error'].mean()),
        'error_std': float(results_df['Error'].std()),
        'error_median': float(results_df['Error'].median()),
        'error_q25': float(results_df['Error'].quantile(0.25)),
        'error_q75': float(results_df['Error'].quantile(0.75)),
        'abs_error_mean': float(results_df['Abs_Error'].mean()),
        'abs_error_median': float(results_df['Abs_Error'].median()),
        'abs_error_max': float(results_df['Abs_Error'].max()),
        'abs_error_q95': float(results_df['Abs_Error'].quantile(0.95)),
        'percent_error_mean': float(results_df['Percent_Error'].mean()),
        'percent_error_median': float(results_df['Percent_Error'].median())
    }

    return stats


def _nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-12
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    return float(rmse / (np.max(y_true) - np.min(y_true) + eps))


def compute_and_save_per_case_nrmse(
    model: nn.Module,
    scaler_X,
    scaler_y,
    save_path: Path = RESULTS_PATH,
    cases: Optional[Dict[str, Dict]] = None,
) -> Dict[str, Dict[str, float]]:
    """Compute NRMSE per patient case and vessel region; save to TXT.

    Returns a dictionary mapping case_id -> { 'n_points': int, 'nrmse': float, 'vessel': str }.
    Also appends aggregated NRMSE per vessel type.
    """
    if cases is None:
        cases = get_all_datasets()

    results: Dict[str, Dict[str, float]] = {}
    vessel_groups: Dict[str, Dict[str, list]] = {}

    lines = []
    lines.append("Per-case NRMSE (PINN vs Full CFD)\n")
    lines.append("CaseID\tVessel\tPoints\tNRMSE\n")

    for case_id, paths in cases.items():
        main_path = paths.get('main')
        if main_path is None or not main_path.exists():
            continue

        df = parse_cfd_file(main_path)
        if df is None or len(df) == 0:
            continue

        # Resolve columns
        cx = get_column_safe(df, 'X', COLUMN_MAPPINGS)
        cy = get_column_safe(df, 'Y', COLUMN_MAPPINGS)
        cz = get_column_safe(df, 'Z', COLUMN_MAPPINGS)
        cw = get_column_safe(df, 'WSS', COLUMN_MAPPINGS)
        if None in (cx, cy, cz, cw):
            continue

        coords = df[[cx, cy, cz]].values
        y_true = df[cw].values.astype(float)

        # Predict
        y_pred = predict_on_new_data(
            model,
            coords=coords,
            scaler_X=scaler_X,
            scaler_y=scaler_y,
            batch_size=8192
        )

        score = _nrmse(y_true, y_pred)
        vessel = extract_vessel_type(main_path.name)

        results[case_id] = {'n_points': int(len(y_true)), 'nrmse': float(score), 'vessel': vessel}
        lines.append(f"{case_id}\t{vessel}\t{len(y_true)}\t{score:.6f}")

        # Group by vessel
        vessel_groups.setdefault(vessel, {'y_true': [], 'y_pred': []})
        vessel_groups[vessel]['y_true'].append(y_true)
        vessel_groups[vessel]['y_pred'].append(y_pred)

    # Aggregated by vessel
    if vessel_groups:
        lines.append("\nAggregated NRMSE by Vessel\n")
        lines.append("Vessel\tPoints\tNRMSE\n")
        for vessel, data in vessel_groups.items():
            if len(data['y_true']) == 0:
                continue
            yt = np.concatenate(data['y_true'])
            yp = np.concatenate(data['y_pred'])
            score = _nrmse(yt, yp)
            lines.append(f"{vessel}\t{len(yt)}\t{score:.6f}")

    # Write to file
    save_path.mkdir(parents=True, exist_ok=True)
    txt_path = save_path / "per_case_nrmse.txt"
    with open(txt_path, 'w') as f:
        f.write('\n'.join(lines))

    return results


def load_trained_model(
    model_class: type,
    checkpoint_path: Path,
    device: torch.device = DEVICE
) -> nn.Module:
    """
    Load trained model from checkpoint

    Args:
        model_class: PINN model class
        checkpoint_path: Path to checkpoint file
        device: Device to load model to

    Returns:
        Loaded model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = model_class().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"\n  Loaded model from epoch: {checkpoint['epoch']+1}")
    print(f"  Validation loss: {checkpoint['val_loss']:.6f}")

    return model
