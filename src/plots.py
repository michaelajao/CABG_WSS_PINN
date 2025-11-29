import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict

from src.config import DEVICE
from src.dataset import PatientDataset
from src.utils import compute_nrmse

# =========================================
# Publication-Quality Plot Settings
# =========================================

# Use Seaborn's "paper" style for clean and professional aesthetics
plt.style.use("seaborn-v0_8-paper")

# Update rcParams for publication-quality plots
plt.rcParams.update(
    {
        # General Figure Settings
        "font.size": 12,
        "figure.figsize": [7, 4],
        "text.usetex": False,
        "figure.facecolor": "white",
        "figure.autolayout": True,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.format": "png",
        "savefig.bbox": "tight",
        
        # Axes and Titles
        "axes.labelsize": 12,
        "axes.titlesize": 16,
        "axes.facecolor": "white",
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.formatter.use_mathtext": True,
        "axes.formatter.useoffset": False,
        
        # Legend Settings
        "legend.fontsize": 12,
        "legend.loc": "best",
    }
)


def plot_training_history(history: Dict, patient_id: str, save_path: Path):
    """
    Plot training loss curve (total loss only).
    
    Args:
        history: Dictionary with 'train_loss' list
        patient_id: Patient identifier
        save_path: Directory to save figure
    """
    # Create loss subfolder
    loss_path = save_path / 'loss'
    loss_path.mkdir(parents=True, exist_ok=True)
    
    train_loss = history['train_loss']
    epochs = range(1, len(train_loss) + 1)
    
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, train_loss, linewidth=2, color='#1f77b4')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.set_title(f'Total Loss - Patient {patient_id}')
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(loss_path / f'{patient_id}_total_loss.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_loss_components(history: Dict, patient_id: str, save_path: Path):
    """
    Plot individual loss components as separate figures saved in loss/ subfolder.
    
    Args:
        history: Dictionary containing loss component histories
        patient_id: Patient identifier
        save_path: Directory to save figure
    """
    # Create loss subfolder
    loss_path = save_path / 'loss'
    loss_path.mkdir(parents=True, exist_ok=True)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Data loss
    if 'data_loss' in history:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(epochs, history['data_loss'], linewidth=2, color='#2ca02c')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Data Loss')
        ax.set_title(f'Data Loss (MSE) - Patient {patient_id}')
        ax.set_yscale('log')
        plt.tight_layout()
        plt.savefig(loss_path / f'{patient_id}_data_loss.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # WSS loss
    if 'wss_loss' in history:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(epochs, history['wss_loss'], linewidth=2, color='#d62728')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('WSS Loss')
        ax.set_title(f'WSS Prediction Loss - Patient {patient_id}')
        ax.set_yscale('log')
        plt.tight_layout()
        plt.savefig(loss_path / f'{patient_id}_wss_loss.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Navier-Stokes loss
    if 'ns_loss' in history:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(epochs, history['ns_loss'], linewidth=2, color='#9467bd')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Navier-Stokes Loss')
        ax.set_title(f'Navier-Stokes Residual - Patient {patient_id}')
        ax.set_yscale('log')
        plt.tight_layout()
        plt.savefig(loss_path / f'{patient_id}_ns_loss.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Continuity loss
    if 'cont_loss' in history:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(epochs, history['cont_loss'], linewidth=2, color='#ff7f0e')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Continuity Loss')
        ax.set_title(f'Continuity Residual - Patient {patient_id}')
        ax.set_yscale('log')
        plt.tight_layout()
        plt.savefig(loss_path / f'{patient_id}_cont_loss.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Velocity loss
    if 'vel_loss' in history:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(epochs, history['vel_loss'], linewidth=2, color='#17becf')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Velocity Loss')
        ax.set_title(f'Velocity Loss - Patient {patient_id}')
        ax.set_yscale('log')
        plt.tight_layout()
        plt.savefig(loss_path / f'{patient_id}_vel_loss.png', dpi=300, bbox_inches='tight')
        plt.close()


def _create_comparison_plot(coords: np.ndarray, true_vals: np.ndarray, pred_vals: np.ndarray,
                            x_idx: int, y_idx: int, xlabel: str, ylabel: str,
                            cmap_main: str, cmap_error: str, vmin: float, vmax: float,
                            error_vmax: float, unit: str, title_prefix: str = '') -> tuple:
    """
    Helper to create side-by-side comparison plots: True vs Predicted vs Error.
    
    Returns:
        Tuple of (fig, axes, rmse, nrmse)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    rmse = np.sqrt(np.mean((pred_vals - true_vals) ** 2))
    nrmse = compute_nrmse(true_vals, pred_vals)
    
    x_plot = coords[:, x_idx] * 1000
    y_plot = coords[:, y_idx] * 1000
    
    # True values (CFD)
    sc1 = axes[0].scatter(x_plot, y_plot, c=true_vals, cmap=cmap_main, s=0.3, vmin=vmin, vmax=vmax)
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    axes[0].set_title('CFD')
    axes[0].set_aspect('equal')
    plt.colorbar(sc1, ax=axes[0], shrink=0.7, label=f'{title_prefix} ({unit})')
    
    # Predicted values (PINN)
    sc2 = axes[1].scatter(x_plot, y_plot, c=pred_vals, cmap=cmap_main, s=0.3, vmin=vmin, vmax=vmax)
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel(ylabel)
    axes[1].set_title('PINN')
    axes[1].set_aspect('equal')
    plt.colorbar(sc2, ax=axes[1], shrink=0.7, label=f'{title_prefix} ({unit})')
    
    # Absolute error
    error = np.abs(pred_vals - true_vals)
    sc3 = axes[2].scatter(x_plot, y_plot, c=error, cmap=cmap_error, s=0.3, vmin=0, vmax=error_vmax)
    axes[2].set_xlabel(xlabel)
    axes[2].set_ylabel(ylabel)
    axes[2].set_title(f'Absolute Error\nNRMSE={nrmse:.2%}')
    axes[2].set_aspect('equal')
    plt.colorbar(sc3, ax=axes[2], shrink=0.7, label=f'Absolute Error ({unit})')
    
    return fig, axes, rmse, nrmse


def plot_wss_comparison(coords: np.ndarray, wss_true: np.ndarray, wss_pred: np.ndarray,
                        patient_id: str, save_path: Path, view: str,
                        x_idx: int, y_idx: int, xlabel: str, ylabel: str):
    """Create side-by-side WSS comparison: CFD vs PINN vs Error."""
    # Use 99th percentile for colorbar to handle outliers while showing full range
    vmax = np.percentile(np.concatenate([wss_true, wss_pred]), 99)
    error_vmax = np.percentile(np.abs(wss_pred - wss_true), 99)
    
    fig, axes, rmse, nrmse = _create_comparison_plot(
        coords, wss_true, wss_pred, x_idx, y_idx, xlabel, ylabel,
        cmap_main='jet', cmap_error='Reds', vmin=0, vmax=vmax,
        error_vmax=error_vmax, unit='Pa', title_prefix='WSS'
    )
    
    plt.tight_layout()
    plt.savefig(save_path / f'{patient_id}_WSS_{view}.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_velocity_comparison(coords: np.ndarray, vel_true: np.ndarray, vel_pred: np.ndarray,
                             patient_id: str, save_path: Path, view: str,
                             x_idx: int, y_idx: int, xlabel: str, ylabel: str,
                             component: str, comp_idx: int):
    """Create side-by-side velocity comparison: CFD vs PINN vs Error."""
    v_true = vel_true[:, comp_idx]
    v_pred = vel_pred[:, comp_idx]
    
    vmax = max(abs(v_true.min()), abs(v_true.max()), abs(v_pred.min()), abs(v_pred.max()))
    
    fig, axes, rmse, nrmse = _create_comparison_plot(
        coords, v_true, v_pred, x_idx, y_idx, xlabel, ylabel,
        cmap_main='RdBu_r', cmap_error='Reds', vmin=-vmax, vmax=vmax,
        error_vmax=vmax * 0.3, unit='m/s', title_prefix=component
    )
    
    plt.tight_layout()
    plt.savefig(save_path / f'{patient_id}_vel_{component}_{view}.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_vessel_wss_comparison(coords: np.ndarray, wss_true: np.ndarray, wss_pred: np.ndarray,
                               patient_id: str, vessel_name: str, save_path: Path, 
                               view: str, x_idx: int, y_idx: int, xlabel: str, ylabel: str):
    """Create side-by-side WSS comparison for a specific vessel."""
    # Use 99th percentile for colorbar to handle outliers while showing full range
    vmax = np.percentile(np.concatenate([wss_true, wss_pred]), 99)
    error_vmax = np.percentile(np.abs(wss_pred - wss_true), 99)
    
    fig, axes, rmse, nrmse = _create_comparison_plot(
        coords, wss_true, wss_pred, x_idx, y_idx, xlabel, ylabel,
        cmap_main='jet', cmap_error='Reds', vmin=0, vmax=vmax,
        error_vmax=error_vmax, unit='Pa', title_prefix='WSS'
    )
    # Use larger points for vessel-specific plots
    for ax in axes:
        for coll in ax.collections:
            coll.set_sizes([0.5])
    
    plt.tight_layout()
    plt.savefig(save_path / f'{patient_id}_{vessel_name}_WSS_{view}.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_per_vessel_plots(model: nn.Module, per_vessel_data: Dict[str, Dict[str, np.ndarray]],
                              dataset: PatientDataset, patient_id: str, save_path: Path):
    """
    Generate WSS comparison plots for each individual vessel.
    """
    model.eval()
    
    views = [
        ('XY', 0, 1, 'X (mm)', 'Y (mm)'),
        ('XZ', 0, 2, 'X (mm)', 'Z (mm)'),
        ('YZ', 1, 2, 'Y (mm)', 'Z (mm)')
    ]
    
    for vessel_name, vessel_data in per_vessel_data.items():
        if vessel_name == 'Combined':
            continue
        
        # Get wall points with WSS
        has_wss = vessel_data['has_wss']
        if not has_wss.any():
            continue
        
        coords_raw = vessel_data['X'][has_wss]
        wss_true = vessel_data['y'][has_wss]
        
        # Skip if too few points
        if len(coords_raw) < 100:
            print(f"    Skipping {vessel_name}: only {len(coords_raw)} points")
            continue
        
        # Scale coordinates for model input
        coords_scaled = dataset.scaler_X.transform(coords_raw)
        coords_tensor = torch.FloatTensor(coords_scaled).to(DEVICE)
        
        # Get predictions
        with torch.no_grad():
            outputs = model(coords_tensor)
            wss_pred_scaled = outputs['wss'].cpu().numpy()
            wss_pred = dataset.scaler_y.inverse_transform(wss_pred_scaled).flatten()
        
        # Compute metrics for this vessel
        rmse = np.sqrt(np.mean((wss_pred - wss_true) ** 2))
        nrmse = compute_nrmse(wss_true, wss_pred)
        r2 = 1 - np.sum((wss_pred - wss_true)**2) / (np.sum((wss_true - np.mean(wss_true))**2) + 1e-10)
        
        print(f"    {vessel_name}: {len(coords_raw):,} points | RMSE={rmse:.4f} Pa | NRMSE={nrmse:.4f} | R²={r2:.4f}")
        
        # Generate plots for each view
        for view, x_idx, y_idx, xlabel, ylabel in views:
            plot_vessel_wss_comparison(coords_raw, wss_true, wss_pred,
                                       patient_id, vessel_name, save_path,
                                       view, x_idx, y_idx, xlabel, ylabel)


def generate_all_plots(model: nn.Module, data_loader: DataLoader, dataset: PatientDataset,
                       patient_id: str, save_path: Path, metrics: Dict,
                       per_vessel_data: Dict[str, Dict[str, np.ndarray]] = None,
                       history: Dict = None):
    """Generate all publication-quality comparison plots, including per-vessel and training history."""
    model.eval()
    
    # Generate loss plots if history provided
    if history is not None:
        print("  Generating training history plots...")
        plot_training_history(history, patient_id, save_path)
        if any(k in history for k in ['data_loss', 'ns_loss', 'wss_loss']):
            plot_loss_components(history, patient_id, save_path)
    
    all_coords, all_wss_true, all_wss_pred = [], [], []
    all_velocity_true, all_velocity_pred = [], []
    all_coords_full = []
    
    with torch.no_grad():
        for batch in data_loader:
            coords = batch['coords'].to(DEVICE)
            coords_raw = batch['coords_raw'].numpy()
            wss_raw = batch['wss_raw'].numpy().flatten()
            vel_scaled = batch['velocity'].numpy()
            has_wss = batch['has_wss'].numpy().squeeze().astype(bool)
            
            outputs = model(coords)
            
            # WSS prediction
            wss_pred_scaled = outputs['wss'].cpu().numpy()
            wss_pred = dataset.scaler_y.inverse_transform(wss_pred_scaled).flatten()
            
            # Velocity prediction (inverse transform using MinMaxScaler attributes)
            vel_pred_scaled = torch.cat([outputs['u'], outputs['v'], outputs['w']], dim=1).cpu().numpy()
            vel_pred = dataset.scaler_vel.inverse_transform(vel_pred_scaled)
            vel_true = dataset.scaler_vel.inverse_transform(vel_scaled)
            
            if has_wss.any():
                all_coords.append(coords_raw[has_wss])
                all_wss_true.append(wss_raw[has_wss])
                all_wss_pred.append(wss_pred[has_wss])
            
            all_coords_full.append(coords_raw)
            all_velocity_true.append(vel_true)
            all_velocity_pred.append(vel_pred)
    
    # Concatenate
    coords_wall = np.concatenate(all_coords)
    wss_true = np.concatenate(all_wss_true)
    wss_pred = np.concatenate(all_wss_pred)
    coords_full = np.concatenate(all_coords_full)
    vel_true = np.concatenate(all_velocity_true)
    vel_pred = np.concatenate(all_velocity_pred)
    
    # View configurations
    views = [
        ('XY', 0, 1, 'X (mm)', 'Y (mm)'),
        ('XZ', 0, 2, 'X (mm)', 'Z (mm)'),
        ('YZ', 1, 2, 'Y (mm)', 'Z (mm)')
    ]
    
    print("  Generating WSS comparison plots...")
    for view, x_idx, y_idx, xlabel, ylabel in views:
        plot_wss_comparison(coords_wall, wss_true, wss_pred, patient_id, save_path,
                           view, x_idx, y_idx, xlabel, ylabel)
    
    print("  Generating velocity comparison plots...")
    for comp, comp_idx in [('u', 0), ('v', 1), ('w', 2)]:
        for view, x_idx, y_idx, xlabel, ylabel in views:
            plot_velocity_comparison(coords_full, vel_true, vel_pred, patient_id, save_path,
                                    view, x_idx, y_idx, xlabel, ylabel, comp, comp_idx)
    
    # Generate per-vessel WSS plots if vessel data available
    if per_vessel_data is not None and len(per_vessel_data) > 1:
        print(f"  Generating per-vessel comparison plots ({len(per_vessel_data)} vessels)...")
        generate_per_vessel_plots(model, per_vessel_data, dataset, patient_id, save_path)

