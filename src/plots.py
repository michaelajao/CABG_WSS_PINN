"""
Visualization Module for PINN Results

This module generates publication-quality plots for analyzing PINN
performance on coronary artery hemodynamics prediction.

Plot Types:
    - Training History: Loss curves over epochs
    - WSS Comparison: CFD vs PINN vs Absolute Error
    - Velocity Fields: Component-wise comparison (u, v, w)
    - Per-Vessel Analysis: Individual vessel WSS plots

All plots use consistent styling following scientific publication standards:
    - 300 DPI resolution
    - Clean axes without top/right spines
    - Colorbar scales based on data percentiles (robust to outliers)
    - Metric annotations (NRMSE, R²) on relevant plots
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict

from src.config import DEVICE, PRIMARY_VESSELS
from src.dataset import PatientDataset
from src.utils import compute_nrmse

# =============================================================================
# PUBLICATION-QUALITY PLOT SETTINGS
# =============================================================================

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
    axes[2].set_title('Absolute Error')
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


def plot_per_vessel_wss(model: nn.Module, per_vessel_data: Dict[str, Dict[str, np.ndarray]],
                        dataset, patient_id: str, save_path: Path):
    """
    Generate WSS comparison plots for each individual vessel (excluding Aorta unless specified).
    Each vessel gets XY, XZ, and YZ view plots showing ONLY that vessel's data.
    
    Args:
        model: Trained PINN model
        per_vessel_data: Dictionary with vessel names as keys and data dicts as values
        dataset: PatientDataset (for scalers)
        patient_id: Patient identifier
        save_path: Directory to save figures
    """
    if per_vessel_data is None or len(per_vessel_data) == 0:
        print("  No per-vessel data available for plotting")
        return
    
    # Define which vessels to plot per patient (based on actual files available)
    # Exclude Aorta from per-vessel plots except for 0073 where it's explicitly primary
    # Get vessels to plot for this patient (PRIMARY_VESSELS imported from config)
    vessels_to_plot = PRIMARY_VESSELS.get(patient_id, [])
    if not vessels_to_plot:
        # Fallback: plot all vessels except Aorta
        vessels_to_plot = [v for v in per_vessel_data.keys() if v.lower() != 'aorta']
    
    model.eval()
    
    # View configurations - all three views
    views = [
        ('XY', 0, 1, 'X (mm)', 'Y (mm)'),
        ('XZ', 0, 2, 'X (mm)', 'Z (mm)'),
        ('YZ', 1, 2, 'Y (mm)', 'Z (mm)')
    ]
    
    for vessel_name in vessels_to_plot:
        if vessel_name not in per_vessel_data:
            print(f"  Skipping {vessel_name}: not in loaded data")
            continue
            
        vessel_data = per_vessel_data[vessel_name]
        
        # Get vessel coordinates and WSS
        coords_raw = vessel_data['X']
        wss_raw = vessel_data['y']
        has_wss = vessel_data['has_wss']
        
        # Only plot wall points (where we have WSS)
        if not has_wss.any():
            print(f"  Skipping {vessel_name}: no wall points")
            continue
        
        wall_coords = coords_raw[has_wss]
        wss_true = wss_raw[has_wss]
        
        # Scale coordinates for model input
        coords_scaled = dataset.scaler_X.transform(wall_coords)
        coords_tensor = torch.FloatTensor(coords_scaled).to(DEVICE)
        
        # Get predictions
        with torch.no_grad():
            outputs = model(coords_tensor)
            wss_pred_scaled = outputs['wss'].cpu().numpy()
            wss_pred = dataset.scaler_y.inverse_transform(wss_pred_scaled).flatten()
        
        # Clean vessel name for filename (replace spaces with underscores)
        vessel_filename = vessel_name.replace(' ', '_').replace('/', '_')
        
        # Calculate NRMSE once for this vessel
        from src.utils import compute_nrmse
        vessel_nrmse = compute_nrmse(wss_true, wss_pred)
        print(f"    {vessel_name}: NRMSE={vessel_nrmse:.2%}")
        
        # Generate plot for each view
        for view, x_idx, y_idx, xlabel, ylabel in views:
            # Use 99th percentile for colorbar
            vmax = np.percentile(np.concatenate([wss_true, wss_pred]), 99)
            error_vmax = np.percentile(np.abs(wss_pred - wss_true), 99)
            
            fig, axes, rmse, nrmse = _create_comparison_plot(
                wall_coords, wss_true, wss_pred, x_idx, y_idx, xlabel, ylabel,
                cmap_main='jet', cmap_error='Reds', vmin=0, vmax=vmax,
                error_vmax=error_vmax, unit='Pa', title_prefix='WSS'
            )
            
            plt.tight_layout()
            plt.savefig(save_path / f'{patient_id}_{vessel_filename}_WSS_{view}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()


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
    
    # Generate per-vessel WSS plots
    if per_vessel_data is not None and len(per_vessel_data) > 0:
        print("  Generating per-vessel WSS plots...")
        plot_per_vessel_wss(model, per_vessel_data, dataset, patient_id, save_path)


# =============================================================================
# FULL PATIENT ANATOMY VISUALIZATION
# =============================================================================

def plot_full_patient_wss(patient_id: str, vessel_data: list, df_aorta: np.ndarray,
                          save_path: Path, planes: list = None, plane_names: list = None):
    """
    Generate WSS comparison plots for full patient anatomy.
    
    Shows Wall Shear Stress on all vessel walls combined:
    - Aorta: gray background (no WSS coloring)
    - All vessels: colored by WSS (Pa)
    
    Creates three-column figure: CFD | PINN | Absolute Error
    
    Args:
        patient_id: Patient identifier
        vessel_data: List of vessel dictionaries, each containing:
            - 'name': Vessel name (e.g., 'LCA', 'RCA')
            - 'coords': (N, 3) coordinates in meters
            - 'wss_true': (N,) ground truth WSS values
            - 'wss_pred': (N,) predicted WSS values
        df_aorta: Aorta coordinates (N, 3) for gray background, or None
        save_path: Directory to save figures
        planes: List of (x_col, y_col) tuples for 2D projections
        plane_names: Corresponding names ['XY', 'XZ', 'YZ']
        
    Outputs:
        For each plane: {patient_id}_full_patient_wss_{plane}.png
    """
    if planes is None:
        planes = [(0, 1), (0, 2), (1, 2)]  # XY, XZ, YZ
    if plane_names is None:
        plane_names = ['XY', 'XZ', 'YZ']
    
    # Create output directory
    full_patient_path = save_path / 'full_patient'
    full_patient_path.mkdir(parents=True, exist_ok=True)
    
    # Combine all vessel data
    all_coords = []
    all_wss_true = []
    all_wss_pred = []
    
    for vdata in vessel_data:
        all_coords.append(vdata['coords'])
        all_wss_true.append(vdata['wss_true'])
        all_wss_pred.append(vdata['wss_pred'])
    
    if len(all_coords) == 0:
        print("  No vessel data for full patient visualization")
        return
    
    coords = np.vstack(all_coords)
    wss_true = np.concatenate(all_wss_true)
    wss_pred = np.concatenate(all_wss_pred)
    wss_error = np.abs(wss_pred - wss_true)
    
    # Calculate metrics
    nrmse = compute_nrmse(wss_true, wss_pred)
    
    # Color scale based on 99th percentile (robust to outliers)
    vmax = np.percentile(np.concatenate([wss_true, wss_pred]), 99)
    error_vmax = np.percentile(wss_error, 99)
    
    for (x_idx, y_idx), plane_name in zip(planes, plane_names):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Convert to mm for display
        x_plot = coords[:, x_idx] * 1000
        y_plot = coords[:, y_idx] * 1000
        
        # Plot aorta background (gray) if available
        if df_aorta is not None and len(df_aorta) > 0:
            aorta_x = df_aorta[:, x_idx] * 1000
            aorta_y = df_aorta[:, y_idx] * 1000
            for ax in axes:
                ax.scatter(aorta_x, aorta_y, c='gray', s=0.1, alpha=0.5, zorder=0)
        
        # CFD ground truth
        sc1 = axes[0].scatter(x_plot, y_plot, c=wss_true, cmap='jet', s=0.5, vmin=0, vmax=vmax)
        axes[0].set_title(f'CFD Ground Truth', fontsize=14)
        axes[0].set_xlabel(f'{["X", "Y", "Z"][x_idx]} (mm)')
        axes[0].set_ylabel(f'{["X", "Y", "Z"][y_idx]} (mm)')
        axes[0].set_aspect('equal')
        plt.colorbar(sc1, ax=axes[0], shrink=0.7, label='WSS (Pa)')
        
        # PINN prediction
        sc2 = axes[1].scatter(x_plot, y_plot, c=wss_pred, cmap='jet', s=0.5, vmin=0, vmax=vmax)
        axes[1].set_title(f'PINN Prediction', fontsize=14)
        axes[1].set_xlabel(f'{["X", "Y", "Z"][x_idx]} (mm)')
        axes[1].set_ylabel(f'{["X", "Y", "Z"][y_idx]} (mm)')
        axes[1].set_aspect('equal')
        plt.colorbar(sc2, ax=axes[1], shrink=0.7, label='WSS (Pa)')
        
        # Absolute error
        sc3 = axes[2].scatter(x_plot, y_plot, c=wss_error, cmap='Reds', s=0.5, vmin=0, vmax=error_vmax)
        axes[2].set_title('Absolute Error', fontsize=14)
        axes[2].set_xlabel(f'{["X", "Y", "Z"][x_idx]} (mm)')
        axes[2].set_ylabel(f'{["X", "Y", "Z"][y_idx]} (mm)')
        axes[2].set_aspect('equal')
        plt.colorbar(sc3, ax=axes[2], shrink=0.7, label='|Error| (Pa)')
        
        plt.tight_layout()
        plt.savefig(full_patient_path / f'{patient_id}_full_patient_wss_{plane_name}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"  Full patient WSS plots saved to {full_patient_path}")


