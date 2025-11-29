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
    """Create side-by-side velocity comparison: CFD vs PINN vs Error.
    
    Note: Most points are wall points with velocity=0 (no-slip BC).
    Streamline (interior) points show the actual flow field.
    """
    v_true = vel_true[:, comp_idx]
    v_pred = vel_pred[:, comp_idx]
    
    vmax = max(abs(v_true.min()), abs(v_true.max()), abs(v_pred.min()), abs(v_pred.max()))
    error_vmax = np.percentile(np.abs(v_pred - v_true), 99)
    
    fig, axes, rmse, nrmse = _create_comparison_plot(
        coords, v_true, v_pred, x_idx, y_idx, xlabel, ylabel,
        cmap_main='RdBu_r', cmap_error='Reds', vmin=-vmax, vmax=vmax,
        error_vmax=error_vmax, unit='m/s', title_prefix=component
    )
    
    plt.tight_layout()
    plt.savefig(save_path / f'{patient_id}_vel_{component}_{view}.png', dpi=300, bbox_inches='tight')
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
    PRIMARY_VESSELS = {
        'H-12': ['LCA'],
        'H-09': ['RCA'],
        'D-10': ['LCA', 'RCA'],
        '0149': ['G1', 'G2', 'G3'],  # No LCA/RCA files available
        '0073': ['LCA', 'RCA', 'Aorta'],  # Aorta explicitly in Primary_Vessels
        '0156': ['G2', 'G3'],  # No LCA/RCA files available
        '0148': ['G2'],  # No RCA file available
        '0150': ['G3'],  # No LCA/RCA files available
        'ND2': ['LCA'],
    }
    
    # Get vessels to plot for this patient
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
    
    print("  Generating velocity comparison plots...")
    for comp, comp_idx in [('u', 0), ('v', 1), ('w', 2)]:
        for view, x_idx, y_idx, xlabel, ylabel in views:
            plot_velocity_comparison(coords_full, vel_true, vel_pred, patient_id, save_path,
                                    view, x_idx, y_idx, xlabel, ylabel, comp, comp_idx)
    
    # Generate per-vessel WSS plots
    if per_vessel_data is not None and len(per_vessel_data) > 0:
        print("  Generating per-vessel WSS plots...")
        plot_per_vessel_wss(model, per_vessel_data, dataset, patient_id, save_path)

