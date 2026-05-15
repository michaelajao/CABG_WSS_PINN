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

import csv
import math
import re
import sys
from pathlib import Path
from statistics import mean
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from src.config import DEVICE, PATIENT_DATA, PRIMARY_VESSELS
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
    # Convert Pa to dynes/cm^2 for presentation to match the CFD results.
    wss_true = wss_true * 10.0
    wss_pred = wss_pred * 10.0

    # Use 99th percentile for colorbar to handle outliers while showing full range
    vmax = np.percentile(np.concatenate([wss_true, wss_pred]), 99)
    error_vmax = np.percentile(np.abs(wss_pred - wss_true), 99)

    fig, axes, rmse, nrmse = _create_comparison_plot(
        coords, wss_true, wss_pred, x_idx, y_idx, xlabel, ylabel,
        cmap_main='jet', cmap_error='Reds', vmin=0, vmax=vmax,
        error_vmax=error_vmax, unit='dynes/cm^2', title_prefix='WSS'
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
    
    # Vessels-to-plot per patient (PRIMARY_VESSELS imported from config).
    # Patients are keyed by public paper labels (H1..D3); each entry is a list
    # of vessel names that exist for that patient.
    vessels_to_plot = PRIMARY_VESSELS[patient_id]
    
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

        wss_true_plot = wss_true * 10.0
        wss_pred_plot = wss_pred * 10.0
        
        # Clean vessel name for filename (replace spaces with underscores)
        vessel_filename = vessel_name.replace(' ', '_').replace('/', '_')
        
        # Calculate NRMSE once for this vessel
        vessel_nrmse = compute_normalised_rmse(wss_true, wss_pred)
        print(f"    {vessel_name}: NRMSE={vessel_nrmse:.2%}")
        
        # Generate plot for each view
        for view, x_idx, y_idx, xlabel, ylabel in views:
            # Use 99th percentile for colorbar
            vmax = np.percentile(np.concatenate([wss_true_plot, wss_pred_plot]), 99)
            error_vmax = np.percentile(np.abs(wss_pred_plot - wss_true_plot), 99)
            
            fig, axes, rmse, nrmse = _create_comparison_plot(
                wall_coords, wss_true_plot, wss_pred_plot, x_idx, y_idx, xlabel, ylabel,
                cmap_main='jet', cmap_error='Reds', vmin=0, vmax=vmax,
                error_vmax=error_vmax, unit='dynes/cm^2', title_prefix='WSS'
            )
            
            plt.tight_layout()
            plt.savefig(save_path / f'{patient_id}_{vessel_filename}_WSS_{view}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()


def generate_all_plots(model: nn.Module, dataset: PatientData,
                       patient_id: str, save_path: Path, metrics: Dict,
                       per_vessel_data: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
                       history: Optional[Dict] = None, batch_size: int = 4096):
    """Generate all publication-quality comparison plots, including per-vessel and training history."""
    model.eval()

    # Generate loss plots if history provided
    if history is not None:
        print("  Generating training history plots...")
        plot_training_history(history, patient_id, save_path)
        if any(k in history for k in ['data_loss', 'ns_loss', 'wss_loss']):
            plot_loss_components(history, patient_id, save_path)

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

def plot_full_patient_wss(patient_id: str, vessel_data: list,
                          df_aorta: 'np.ndarray | None',
                          save_path: Path,
                          planes: 'list | None' = None,
                          plane_names: 'list | None' = None):
    """
    Generate WSS comparison plots for full patient anatomy.
    
    Shows Wall Shear Stress on all vessel walls combined:
    - Aorta: gray background (no WSS coloring)
    - All vessels: colored by WSS (dynes/cm^2)
    
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

    # Convert Pa to dynes/cm^2 for presentation to match the CFD results.
    wss_true = wss_true * 10.0
    wss_pred = wss_pred * 10.0
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
        plt.colorbar(sc1, ax=axes[0], shrink=0.7, label='WSS (dynes/cm^2)')

        # PINN prediction
        sc2 = axes[1].scatter(x_plot, y_plot, c=wss_pred, cmap='jet', s=0.5, vmin=0, vmax=vmax)
        axes[1].set_title('PINN')
        axes[1].set_xlabel(f'{["X", "Y", "Z"][x_idx]} (mm)')
        axes[1].set_ylabel(f'{["X", "Y", "Z"][y_idx]} (mm)')
        axes[1].set_aspect('equal')
        plt.colorbar(sc2, ax=axes[1], shrink=0.7, label='WSS (dynes/cm^2)')

        # Absolute error
        sc3 = axes[2].scatter(x_plot, y_plot, c=wss_error, cmap='Reds', s=0.5, vmin=0, vmax=error_vmax)
        axes[2].set_title('Absolute Error')
        axes[2].set_xlabel(f'{["X", "Y", "Z"][x_idx]} (mm)')
        axes[2].set_ylabel(f'{["X", "Y", "Z"][y_idx]} (mm)')
        axes[2].set_aspect('equal')
        plt.colorbar(sc3, ax=axes[2], shrink=0.7, label='|Error| (dynes/cm^2)')
        
        plt.tight_layout()
        plt.savefig(full_patient_path / f'{patient_id}_full_patient_wss_{plane_name}.png',
                   dpi=300, bbox_inches='tight')
        plt.close()

    print(f"  Full patient WSS plots saved to {full_patient_path}")


# =============================================================================
# Holdout-summary figure + LaTeX-table patch
# =============================================================================
# These two paper outputs are driven from a single CSV produced by
# scripts/run_holdout_eval.py. Run as:
#     python -m src.plots --rheology newtonian
#     python -m src.plots --rheology carreau_yasuda
# By default the LaTeX table tab:pinn_holdout(_cy) is also patched in place.
# Pass --no-update-table to skip the LaTeX patch and only render the figure.

# Patient ordering and category labels are derived from the central registry
# in src.config; CATEGORY_COLOUR is the only thing local to the holdout figure.
_HOLDOUT_ROW_ORDER = list(PATIENT_DATA.keys())
_CATEGORY_DISPLAY = {'healthy': 'Healthy', 'svg': 'SVG', 'diseased': 'Diseased'}
_HOLDOUT_LABEL_TO_CATEGORY = {
    label: _CATEGORY_DISPLAY[entry['category']]
    for label, entry in PATIENT_DATA.items()
}
_HOLDOUT_CATEGORY_COLOUR = {
    'Healthy':  '#2E7D32',  # green
    'SVG':      '#1565C0',  # blue
    'Diseased': '#C62828',  # red
}
_HOLDOUT_RHEOLOGY_COLOUR = {
    'Newtonian': '#2F5597',
    'Carreau--Yasuda': '#D67E2C',
}
_HOLDOUT_TABLE_LABELS = {
    'newtonian': 'tab:pinn_holdout',
    'carreau_yasuda': 'tab:pinn_holdout_cy',
}
# Columns rendered in each LaTeX table row: (csv_column, decimals, percentise).
_HOLDOUT_TABLE_COLUMNS = (
    ('NRMSE_train', 2, True),
    ('NRMSE_holdout', 2, True),
    ('R2_train', 3, False),
    ('R2_holdout', 3, False),
    ('pearson_holdout', 3, False),
)


def _holdout_read_metric(row, key, percentise=False):
    raw = row.get(key, '')
    if raw in ('', None):
        return float('nan')
    value = float(raw)
    return value * 100 if percentise else value


def _holdout_read_csv(csv_path: Path) -> Dict[str, dict]:
    """Return mapping public_label -> metrics dict (NRMSE/R^2/r)."""
    rows: Dict[str, dict] = {}
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            label = r.get('patient_id')
            if label not in PATIENT_DATA:
                continue
            rows[label] = {
                'NRMSE_train':     _holdout_read_metric(r, 'NRMSE_train',     percentise=True),
                'NRMSE_holdout':   _holdout_read_metric(r, 'NRMSE_holdout',   percentise=True),
                'R2_train':        _holdout_read_metric(r, 'R2_train'),
                'R2_holdout':      _holdout_read_metric(r, 'R2_holdout'),
                'pearson_holdout': _holdout_read_metric(r, 'pearson_holdout'),
            }
    return rows


def render_holdout_figure(rows: Dict[str, dict], out_dir: Path, stem: str) -> None:
    """Render the 2-panel holdout figure (NRMSE bars + R^2 bars) to PDF + PNG.

    The bars at the held-out value are coloured by anatomical category. A short
    black tick on each NRMSE bar marks the train value so any train/holdout gap
    is visible at a glance.
    """
    labels = [lbl for lbl in _HOLDOUT_ROW_ORDER if lbl in rows]
    nrmse_tr = [rows[lbl]['NRMSE_train']   for lbl in labels]
    nrmse_ho = [rows[lbl]['NRMSE_holdout'] for lbl in labels]
    r2_ho    = [rows[lbl]['R2_holdout']    for lbl in labels]
    colours  = [_HOLDOUT_CATEGORY_COLOUR[_HOLDOUT_LABEL_TO_CATEGORY[lbl]]
                for lbl in labels]

    x = np.arange(len(labels))
    bar_width = 0.62

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.0))

    def _draw(ax, vals_hold, vals_train=None):
        ax.bar(x, vals_hold, bar_width, color=colours,
               edgecolor='black', linewidth=0.7)
        if vals_train is not None:
            tick_half = bar_width * 0.45
            for xi, vt in zip(x, vals_train):
                ax.hlines(vt, xi - tick_half, xi + tick_half,
                          colors='black', linewidth=1.4)

    _draw(axL, nrmse_ho, nrmse_tr)
    axL.set_xticks(x)
    axL.set_xticklabels(labels)
    axL.set_ylabel('WSS NRMSE (%)')
    axL.set_title('(a) Per-patient WSS NRMSE')
    axL.grid(axis='y', alpha=0.3, linewidth=0.5)
    axL.set_axisbelow(True)

    # R^2 train and holdout values are within ~0.01 of each other, which is
    # invisible at the 0.85--1.00 scale; show only the holdout bar to keep the
    # panel readable. Exact train/hold values are in the LaTeX table.
    _draw(axR, r2_ho)
    axR.set_xticks(x)
    axR.set_xticklabels(labels)
    axR.set_ylabel(r'$R^2$')
    axR.set_title(r'(b) Per-patient $R^2$ (held-out)')
    axR.set_ylim(0.85, 1.00)
    axR.grid(axis='y', alpha=0.3, linewidth=0.5)
    axR.set_axisbelow(True)

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_handles = [
        Patch(facecolor=c, edgecolor='black', linewidth=0.8, label=k)
        for k, c in _HOLDOUT_CATEGORY_COLOUR.items()
    ]
    legend_handles.append(
        Line2D([0], [0], color='black', linewidth=1.4, label='Train value')
    )
    fig.legend(handles=legend_handles, loc='lower center', ncol=4,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=(0, 0.05, 1, 1))

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f'{stem}.pdf'
    png_path = out_dir / f'{stem}.png'
    fig.savefig(pdf_path, bbox_inches='tight')
    fig.savefig(png_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'  Wrote {pdf_path}')
    print(f'  Wrote {png_path}')


def render_holdout_comparison_figure(newtonian_rows: Dict[str, dict],
                                     cy_rows: Dict[str, dict],
                                     out_dir: Path,
                                     stem: str = 'pinn_holdout_comparison') -> None:
    """Render a compact Newtonian-vs-Carreau--Yasuda holdout comparison.

    The manuscript table carries train-vs-holdout detail. This figure focuses
    on the held-out metrics so it remains readable as a publication figure.
    """
    labels = [
        lbl for lbl in _HOLDOUT_ROW_ORDER
        if lbl in newtonian_rows and lbl in cy_rows
    ]
    x = np.arange(len(labels))
    bar_width = 0.36

    nrmse_newt = [newtonian_rows[lbl]['NRMSE_holdout'] for lbl in labels]
    nrmse_cy = [cy_rows[lbl]['NRMSE_holdout'] for lbl in labels]
    r2_newt = [newtonian_rows[lbl]['R2_holdout'] for lbl in labels]
    r2_cy = [cy_rows[lbl]['R2_holdout'] for lbl in labels]

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11.2, 4.2), sharex=True)

    def _category_spans(ax):
        start = 0
        while start < len(labels):
            category = _HOLDOUT_LABEL_TO_CATEGORY[labels[start]]
            end = start
            while end + 1 < len(labels) and _HOLDOUT_LABEL_TO_CATEGORY[labels[end + 1]] == category:
                end += 1
            ax.axvspan(
                start - 0.5, end + 0.5,
                color=_HOLDOUT_CATEGORY_COLOUR[category],
                alpha=0.07,
                linewidth=0,
                zorder=0,
            )
            ax.text(
                (start + end) / 2, 1.02, category,
                transform=ax.get_xaxis_transform(),
                ha='center', va='bottom', fontsize=8,
                color=_HOLDOUT_CATEGORY_COLOUR[category],
            )
            start = end + 1

    def _draw_pair(ax, vals_newt, vals_cy, ylabel, title):
        _category_spans(ax)
        ax.bar(
            x - bar_width / 2, vals_newt, bar_width,
            color=_HOLDOUT_RHEOLOGY_COLOUR['Newtonian'],
            edgecolor='black', linewidth=0.6, label='Newtonian', zorder=2,
        )
        ax.bar(
            x + bar_width / 2, vals_cy, bar_width,
            color=_HOLDOUT_RHEOLOGY_COLOUR['Carreau--Yasuda'],
            edgecolor='black', linewidth=0.6, label='Carreau--Yasuda', zorder=2,
        )
        ax.set_ylabel(ylabel)
        ax.set_title(title, pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3, linewidth=0.5, zorder=1)
        ax.set_axisbelow(True)

    _draw_pair(ax_l, nrmse_newt, nrmse_cy, 'Held-out WSS NRMSE (%)',
               '(a) Held-out error')
    ax_l.set_ylim(0, max(nrmse_newt + nrmse_cy) * 1.18)

    _draw_pair(ax_r, r2_newt, r2_cy, r'Held-out $R^2$',
               r'(b) Held-out $R^2$')
    ax_r.set_ylim(0, 1.0)
    ax_r.axhline(0.8, color='0.45', linestyle='--', linewidth=0.8, zorder=1)

    handles, labels_legend = ax_l.get_legend_handles_labels()
    fig.legend(handles, labels_legend, loc='lower center', ncol=2,
               frameon=False, bbox_to_anchor=(0.5, -0.03))
    plt.tight_layout(rect=(0, 0.06, 1, 1))

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f'{stem}.pdf'
    png_path = out_dir / f'{stem}.png'
    fig.savefig(pdf_path, bbox_inches='tight')
    fig.savefig(png_path, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(f'  Wrote {pdf_path}')
    print(f'  Wrote {png_path}')


def _holdout_fmt_cell(value, decimals: int) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return '---'
    return f'{value:.{decimals}f}'


def _holdout_row_tex(label: str, metrics: dict) -> str:
    cells = ' & '.join(
        _holdout_fmt_cell(metrics.get(col), decimals)
        for col, decimals, _ in _HOLDOUT_TABLE_COLUMNS
    )
    return f'{label} & {_HOLDOUT_LABEL_TO_CATEGORY[label]} & {cells} \\\\'


def _holdout_build_table_block(rows: Dict[str, dict]) -> str:
    lines = [_holdout_row_tex(lbl, rows.get(lbl, {})) for lbl in _HOLDOUT_ROW_ORDER]
    means = {}
    for col, _, _ in _HOLDOUT_TABLE_COLUMNS:
        values = [
            r[col] for r in rows.values()
            if col in r and not (isinstance(r[col], float) and math.isnan(r[col]))
        ]
        means[col] = mean(values) if values else float('nan')
    mean_cells = ' & '.join(
        f'\\textbf{{{_holdout_fmt_cell(means[col], decimals)}}}'
        for col, decimals, _ in _HOLDOUT_TABLE_COLUMNS
    )
    lines.append(f'\\hline\n\\textbf{{Mean}} & -- & {mean_cells} \\\\')
    return '\n'.join(lines)


def patch_holdout_latex_table(tex_path: Path, label_target: str,
                              rows: Dict[str, dict]) -> None:
    """Patch the data rows + Mean row of the named holdout table in main.tex."""
    pattern = re.compile(
        rf'(\\label\{{{re.escape(label_target)}\}}.*?'
        r'\n & & Train & Holdout & Train & Holdout & Holdout \\\\\n\\hline\n)'
        r'(.*?)'
        r'(\n\\hline\n\\end\{tabular\})',
        re.DOTALL,
    )
    tex = tex_path.read_text(encoding='utf-8')
    match = pattern.search(tex)
    if not match:
        sys.exit(
            f'Could not locate Table {label_target} body in {tex_path}; '
            'check the table layout has not changed.'
        )
    block = _holdout_build_table_block(rows)
    new_tex = tex[: match.start(2)] + block + tex[match.end(2):]
    tex_path.write_text(new_tex, encoding='utf-8')
    print(f'  Patched {label_target} in {tex_path} ({len(rows)} rows).')


def _holdout_main(argv=None):
    """CLI entry point: ``python -m src.plots --rheology {newtonian|carreau_yasuda}``."""
    import argparse
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description='Generate the holdout figure (PDF + PNG) and patch the '
                    'corresponding LaTeX table in main.tex.'
    )
    parser.add_argument('--rheology', choices=sorted(_HOLDOUT_TABLE_LABELS),
                        default='newtonian')
    parser.add_argument('--csv', default=None,
                        help='Override CSV path (default: '
                             'reports/metrics/holdout_summary_<rheology>.csv).')
    parser.add_argument('--out-name', default=None,
                        help='Output filename stem under doc/CABG_Paper/figures/ '
                             '(default: pinn_holdout_summary[_<rheology>]).')
    parser.add_argument('--tex', default=str(repo_root / 'doc/CABG_Paper/main.tex'),
                        help='Path to main.tex (default: doc/CABG_Paper/main.tex).')
    parser.add_argument('--no-update-table', action='store_true',
                        help='Skip the LaTeX-table patch and render only the figure.')
    args = parser.parse_args(argv)

    csv_path = (
        Path(args.csv) if args.csv
        else repo_root / f'reports/metrics/holdout_summary_{args.rheology}.csv'
    )
    if not csv_path.exists():
        sys.exit(f'CSV not found: {csv_path}')
    rows = _holdout_read_csv(csv_path)
    if not rows:
        sys.exit(f'No rows in {csv_path} match patients in PATIENT_DATA.')

    if args.out_name:
        stem = args.out_name
    elif args.rheology == 'newtonian':
        stem = 'pinn_holdout_summary'
    else:
        stem = f'pinn_holdout_summary_{args.rheology}'

    print(f'[plots.holdout] rheology={args.rheology}, CSV={csv_path}, rows={len(rows)}')
    render_holdout_figure(rows, repo_root / 'doc/CABG_Paper/figures', stem)

    if not args.no_update_table:
        tex_path = Path(args.tex)
        if not tex_path.exists():
            sys.exit(f'TeX file not found: {tex_path}')
        patch_holdout_latex_table(tex_path, _HOLDOUT_TABLE_LABELS[args.rheology], rows)


if __name__ == '__main__':
    _holdout_main()
