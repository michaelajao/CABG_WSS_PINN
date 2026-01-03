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
from pathlib import Path
from typing import Dict

from src.config import DEVICE, PRIMARY_VESSELS
from src.dataset import PatientData, load_aorta_data, load_full_anatomy
from src.utils import compute_normalised_rmse

# =============================================================================
# PUBLICATION-QUALITY PLOT SETTINGS
# =============================================================================

# Use Seaborn's "paper" style for clean and professional aesthetics
plt.style.use("seaborn-v0_8-paper")

# Update rcParams for publication-quality plots
plt.rcParams.update({
    # -------------------------------------
    # General Figure Settings
    # -------------------------------------
    'font.size': 10,                        # Base font size for all text elements
    'font.family': 'sans-serif',
    'figure.dpi': 300,                      # High resolution for display
    'savefig.dpi':300,                     # High resolution for saved figures
    'savefig.bbox': 'tight',                # Minimize whitespace around figure
    'savefig.pad_inches': 0.02,             # Small padding for tight layout
    'savefig.format': 'png',                # Default format (overridden per plot)
    'figure.autolayout': True,              # Automatically adjust subplot params
    
    # -------------------------------------
    # Axes and Background
    # -------------------------------------
    'figure.facecolor': 'white',            # White background for research papers
    'axes.facecolor': 'white',              # White axes background
    'savefig.facecolor': 'white',           # White saved figure background
    'savefig.transparent': False,           # Opaque background (not transparent)
    'axes.spines.top': False,               # Remove top spine for cleaner look
    'axes.spines.right': False,             # Remove right spine for cleaner look
    
    # -------------------------------------
    # Text and Labels
    # -------------------------------------
    'axes.titlesize': 12,                   # Plot title font size
    'axes.labelsize': 10,                   # Axis label font size
    'xtick.labelsize': 9,                   # X-axis tick label size
    'ytick.labelsize': 9,                   # Y-axis tick label size
    'legend.fontsize': 9,                   # Legend text size
    'legend.loc': 'best',                   # Auto-place legend optimally
    
    # -------------------------------------
    # Formatting
    # -------------------------------------
    'axes.formatter.use_mathtext': True,    # LaTeX-style math formatting
    'axes.formatter.useoffset': False,      # Disable offset in tick labels
    'image.cmap': 'viridis',                # Default colormap for images
})


# # Update rcParams for publication-quality plots
# plt.rcParams.update(
#     {
#         # General Figure Settings
#         "font.size": 12,
#         "figure.figsize": [7, 4],
#         "text.usetex": False,
#         "figure.facecolor": "white",
#         "figure.autolayout": True,
#         "figure.dpi": 300,
#         "savefig.dpi": 300,
#         "savefig.format": "png",
#         "savefig.bbox": "tight",
        
#         # Axes and Titles
#         "axes.labelsize": 12,
#         "axes.titlesize": 16,
#         "axes.facecolor": "white",
#         "axes.grid": False,
#         "axes.spines.top": False,
#         "axes.spines.right": False,
#         "axes.formatter.use_mathtext": True,
#         "axes.formatter.useoffset": False,
        
#         # Legend Settings
#         "legend.fontsize": 12,
#         "legend.loc": "best",
#     }
# )


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

    # WSS Physics loss
    if 'wss_physics_loss' in history:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(epochs, history['wss_physics_loss'], linewidth=2, color='#8c564b')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('WSS Physics Loss')
        ax.set_title(f'WSS Physics Residual - Patient {patient_id}')
        ax.set_yscale('log')
        plt.tight_layout()
        plt.savefig(loss_path / f'{patient_id}_wss_physics_loss.png', dpi=300, bbox_inches='tight')
        plt.close()


def plot_adaptive_weights(history: Dict, patient_id: str, save_path: Path):
    """
    Plot adaptive loss weights (ReLoBRaLo) over training epochs.

    Shows how each loss term weight evolves during training, indicating
    which losses the optimizer prioritizes at different stages.

    Args:
        history: Dictionary containing weight histories (weight_wss, etc.)
        patient_id: Patient identifier
        save_path: Directory to save figure
    """
    # Check if adaptive weights exist in history
    weight_keys = ['weight_wss', 'weight_vel', 'weight_ns', 'weight_cont', 'weight_wss_physics']
    if not any(k in history for k in weight_keys):
        return  # No adaptive weights to plot

    # Create loss subfolder
    loss_path = save_path / 'loss'
    loss_path.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history.get('train_loss', [])) + 1)

    # Plot all weights on one figure
    fig, ax = plt.subplots(figsize=(10, 5))

    # Color scheme matching the individual loss plots
    colors = {
        'weight_wss': '#d62728',       # Red (matches wss_loss)
        'weight_vel': '#17becf',       # Cyan (matches vel_loss)
        'weight_ns': '#9467bd',        # Purple (matches ns_loss)
        'weight_cont': '#ff7f0e',      # Orange (matches cont_loss)
        'weight_wss_physics': '#8c564b'  # Brown (matches wss_physics_loss)
    }

    labels = {
        'weight_wss': 'WSS',
        'weight_vel': 'Velocity',
        'weight_ns': 'Navier-Stokes',
        'weight_cont': 'Continuity',
        'weight_wss_physics': 'WSS Physics'
    }

    for key in weight_keys:
        if key in history and len(history[key]) > 0:
            ax.plot(epochs, history[key], linewidth=2, color=colors[key], label=labels[key])

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss Weight')
    ax.set_title(f'Adaptive Loss Weights (ReLoBRaLo) - Patient {patient_id}')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Uniform (1.0)')
    ax.legend(loc='best', ncol=2)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(loss_path / f'{patient_id}_adaptive_weights.png', dpi=300, bbox_inches='tight')
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
    nrmse = compute_normalised_rmse(true_vals, pred_vals)
    
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


def plot_velocity_comparison(coords: np.ndarray, vel_true: np.ndarray,
                              vel_pred: np.ndarray, patient_id: str,
                              save_path: Path, view: str, x_idx: int,
                              y_idx: int, xlabel: str, ylabel: str,
                              component: str = 'magnitude'):
    """
    Create side-by-side velocity comparison: CFD vs PINN vs Absolute Error.

    Plots velocity field using mesh-based scatter approach (same as WSS) for
    visual consistency in publications. Velocity is plotted for ALL points
    (wall + interior) to validate the full flow field prediction.

    Args:
        coords: (N, 3) coordinates in meters
        vel_true: (N, 3) ground truth velocity [m/s]
        vel_pred: (N, 3) predicted velocity [m/s]
        patient_id: Patient identifier
        save_path: Directory to save figure
        view: View name ('XY', 'XZ', 'YZ')
        x_idx, y_idx: Column indices for 2D projection
        xlabel, ylabel: Axis labels
        component: 'magnitude', 'u', 'v', or 'w'

    Returns:
        Tuple of (rmse, nrmse) for the plotted component
    """
    # Create velocity subfolder for organization
    vel_path = save_path / 'velocity'
    vel_path.mkdir(parents=True, exist_ok=True)

    # Extract appropriate values based on component
    if component == 'magnitude':
        true_vals = np.linalg.norm(vel_true, axis=1)
        pred_vals = np.linalg.norm(vel_pred, axis=1)
        unit = 'm/s'
        title = 'Velocity Magnitude'
        cmap = 'viridis'
        vmin = 0  # Magnitude is always positive
        vmax = np.percentile(np.concatenate([true_vals, pred_vals]), 99)
    else:
        comp_idx = {'u': 0, 'v': 1, 'w': 2}[component]
        true_vals = vel_true[:, comp_idx]
        pred_vals = vel_pred[:, comp_idx]
        unit = 'm/s'
        title = f'Velocity {component}'
        cmap = 'RdBu_r'  # Diverging colormap for signed values
        # Symmetric colorbar around zero for components
        max_abs = np.percentile(np.abs(np.concatenate([true_vals, pred_vals])), 99)
        vmin = -max_abs
        vmax = max_abs

    # Error colorbar scaling
    error_vmax = np.percentile(np.abs(pred_vals - true_vals), 99)

    fig, axes, rmse, nrmse = _create_comparison_plot(
        coords, true_vals, pred_vals, x_idx, y_idx, xlabel, ylabel,
        cmap_main=cmap, cmap_error='Reds', vmin=vmin, vmax=vmax,
        error_vmax=error_vmax, unit=unit, title_prefix=title
    )

    plt.tight_layout()
    filename = f'{patient_id}_vel_{component}_{view}.png'
    plt.savefig(vel_path / filename, dpi=300, bbox_inches='tight')
    plt.close()

    return rmse, nrmse


def plot_error_histogram(wss_true: np.ndarray, wss_pred: np.ndarray,
                         patient_id: str, save_path: Path):
    """
    Plot histogram of WSS prediction errors for statistical validation.

    Generates two histograms:
    - Signed error distribution (shows bias)
    - Absolute error distribution (shows MAE)

    Args:
        wss_true: (N,) ground truth WSS [Pa]
        wss_pred: (N,) predicted WSS [Pa]
        patient_id: Patient identifier
        save_path: Directory to save figure
    """
    errors = wss_pred - wss_true
    abs_errors = np.abs(errors)

    # Compute statistics
    mae = np.mean(abs_errors)
    rmse = np.sqrt(np.mean(errors**2))
    bias = np.mean(errors)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Signed error distribution
    axes[0].hist(errors, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
    axes[0].axvline(bias, color='orange', linestyle='-', linewidth=2,
                    label=f'Bias: {bias:.4f} Pa')
    axes[0].set_xlabel('Error (Pa)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Error Distribution (Signed)')
    axes[0].legend()

    # Absolute error distribution
    axes[1].hist(abs_errors, bins=50, color='coral', edgecolor='black', alpha=0.7)
    axes[1].axvline(mae, color='red', linestyle='--', linewidth=2,
                    label=f'MAE: {mae:.4f} Pa')
    axes[1].set_xlabel('Absolute Error (Pa)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Absolute Error Distribution')
    axes[1].legend()

    # Add text box with statistics
    textstr = f'MAE: {mae:.4f} Pa\nRMSE: {rmse:.4f} Pa\nBias: {bias:.4f} Pa'
    fig.text(0.99, 0.98, textstr, transform=fig.transFigure, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path / f'{patient_id}_error_histogram.png',
                dpi=300, bbox_inches='tight')
    plt.close()


def plot_per_vessel_wss(model: nn.Module, per_vessel_data: Dict[str, Dict[str, np.ndarray]],
                        dataset, patient_id: str, save_path: Path):
    """
    Generate WSS comparison plots for each individual vessel (excluding Aorta unless specified).
    Each vessel gets XY, XZ, and YZ view plots showing ONLY that vessel's data.

    Args:
        model: Trained PINN model
        per_vessel_data: Dictionary with vessel names as keys and data dicts as values
        dataset: PatientData (for scalers)
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
        
        # Scale coordinates for model input (uniform scaling)
        coords_scaled = (wall_coords - dataset.coord_offset) / dataset.L_ref
        coords_tensor = torch.FloatTensor(coords_scaled).to(DEVICE)
        
        # Get predictions and denormalize to physical units
        with torch.no_grad():
            outputs = model(coords_tensor)
            wss_pred_nondim = outputs['wss'].cpu().numpy().flatten()
            # WSS: tau = tau* * T_ref
            wss_pred = wss_pred_nondim * dataset.T_ref
        
        # Clean vessel name for filename (replace spaces with underscores)
        vessel_filename = vessel_name.replace(' ', '_').replace('/', '_')
        
        # Calculate NRMSE once for this vessel
        vessel_nrmse = compute_normalised_rmse(wss_true, wss_pred)
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


def generate_all_plots(model: nn.Module, dataset: PatientData,
                       patient_id: str, save_path: Path, metrics: Dict,
                       per_vessel_data: Dict[str, Dict[str, np.ndarray]] = None,
                       history: Dict = None, batch_size: int = 4096):
    """Generate all publication-quality comparison plots, including per-vessel and training history."""
    model.eval()

    # Generate loss plots if history provided
    if history is not None:
        print("  Generating training history plots...")
        plot_training_history(history, patient_id, save_path)
        if any(k in history for k in ['data_loss', 'ns_loss', 'wss_loss']):
            plot_loss_components(history, patient_id, save_path)
        # Plot adaptive weights if they exist
        if any(k in history for k in ['weight_wss', 'weight_vel', 'weight_ns']):
            print("  Generating adaptive weights plot...")
            plot_adaptive_weights(history, patient_id, save_path)

    # Get data masks and references
    has_wss = dataset.has_wss.cpu().numpy()
    n_samples = len(dataset)

    all_coords, all_wss_true, all_wss_pred = [], [], []
    all_velocity_true, all_velocity_pred = [], []
    all_coords_full = []

    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            batch = dataset.get_batch_sequential(start, batch_size)
            coords = batch['coords']
            coords_raw = batch['coords_raw']
            wss_raw = batch['wss_raw']
            vel_nondim = batch['velocity'].cpu().numpy()
            batch_has_wss = batch['has_wss'].cpu().numpy()

            outputs = model(coords)

            # WSS prediction: tau = tau* * T_ref
            wss_pred_nondim = outputs['wss'].cpu().numpy().flatten()
            wss_pred = wss_pred_nondim * dataset.T_ref

            # Velocity prediction: u = u* * U_ref
            vel_pred_nondim = torch.cat([outputs['u'], outputs['v'], outputs['w']], dim=1).cpu().numpy()
            vel_pred = vel_pred_nondim * dataset.U_ref
            vel_true = vel_nondim * dataset.U_ref  # Ground truth is also non-dimensional

            if batch_has_wss.any():
                all_coords.append(coords_raw[batch_has_wss])
                all_wss_true.append(wss_raw[batch_has_wss])
                all_wss_pred.append(wss_pred[batch_has_wss])

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

    # Generate velocity magnitude comparison plots (validates full flow field)
    print("  Generating velocity magnitude plots...")
    for view, x_idx, y_idx, xlabel, ylabel in views:
        plot_velocity_comparison(coords_full, vel_true, vel_pred, patient_id,
                                save_path, view, x_idx, y_idx, xlabel, ylabel,
                                component='magnitude')

    # Generate per-vessel WSS plots
    if per_vessel_data is not None and len(per_vessel_data) > 0:
        print("  Generating per-vessel WSS plots...")
        plot_per_vessel_wss(model, per_vessel_data, dataset, patient_id, save_path)

        # Generate full patient anatomy visualization
        print("  Generating full patient anatomy plots...")
        _generate_full_patient_from_per_vessel(
            model, per_vessel_data, dataset, patient_id, save_path
        )


def _generate_full_patient_from_per_vessel(
    model: nn.Module,
    per_vessel_data: Dict[str, Dict[str, np.ndarray]],
    dataset: PatientData,
    patient_id: str,
    save_path: Path
):
    """
    Generate full patient anatomy visualization from per-vessel data.

    Helper function called by generate_all_plots.
    """
    # Prepare vessel data for full patient plot
    vessel_data_list = []
    
    for vessel_name, vdata in per_vessel_data.items():
        if vessel_name.lower() == 'aorta':
            continue
        
        has_wss = vdata['has_wss']
        if not has_wss.any():
            continue
        
        # Get wall coordinates
        wall_coords = vdata['X'][has_wss]
        wss_true = vdata['y'][has_wss]
        
        # Get predictions (uniform scaling)
        coords_scaled = (wall_coords - dataset.coord_offset) / dataset.L_ref
        coords_tensor = torch.FloatTensor(coords_scaled).to(DEVICE)
        
        with torch.no_grad():
            outputs = model(coords_tensor)
            wss_pred_nondim = outputs['wss'].cpu().numpy().flatten()
            wss_pred = wss_pred_nondim * dataset.T_ref
        
        vessel_data_list.append({
            'name': vessel_name,
            'coords': wall_coords,
            'wss_true': wss_true,
            'wss_pred': wss_pred
        })
    
    # Load complete anatomy for grey background (not just aorta)
    df_aorta = load_full_anatomy(patient_id)
    if df_aorta is None:
        # Fallback to aorta-only if full anatomy not available
        df_aorta = load_aorta_data(patient_id)
    
    # Generate full patient plots
    if len(vessel_data_list) > 0:
        plot_full_patient_wss(patient_id, vessel_data_list, df_aorta, save_path)


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
    nrmse = compute_normalised_rmse(wss_true, wss_pred)
    
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
        axes[0].set_title('CFD')
        axes[0].set_xlabel(f'{["X", "Y", "Z"][x_idx]} (mm)')
        axes[0].set_ylabel(f'{["X", "Y", "Z"][y_idx]} (mm)')
        axes[0].set_aspect('equal')
        plt.colorbar(sc1, ax=axes[0], shrink=0.7, label='WSS (Pa)')

        # PINN prediction
        sc2 = axes[1].scatter(x_plot, y_plot, c=wss_pred, cmap='jet', s=0.5, vmin=0, vmax=vmax)
        axes[1].set_title('PINN')
        axes[1].set_xlabel(f'{["X", "Y", "Z"][x_idx]} (mm)')
        axes[1].set_ylabel(f'{["X", "Y", "Z"][y_idx]} (mm)')
        axes[1].set_aspect('equal')
        plt.colorbar(sc2, ax=axes[1], shrink=0.7, label='WSS (Pa)')

        # Absolute error
        sc3 = axes[2].scatter(x_plot, y_plot, c=wss_error, cmap='Reds', s=0.5, vmin=0, vmax=error_vmax)
        axes[2].set_title('Absolute Error')
        axes[2].set_xlabel(f'{["X", "Y", "Z"][x_idx]} (mm)')
        axes[2].set_ylabel(f'{["X", "Y", "Z"][y_idx]} (mm)')
        axes[2].set_aspect('equal')
        plt.colorbar(sc3, ax=axes[2], shrink=0.7, label='|Error| (Pa)')
        
        plt.tight_layout()
        plt.savefig(full_patient_path / f'{patient_id}_full_patient_wss_{plane_name}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"  Full patient WSS plots saved to {full_patient_path}")


