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
        "savefig.format": "pdf",
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
    Plot training loss curve.
    
    Args:
        history: Dictionary with 'train_loss' list
        patient_id: Patient identifier
        save_path: Directory to save figure
    """
    train_loss = history['train_loss']
    epochs = range(1, len(train_loss) + 1)
    
    fig, ax = plt.subplots(figsize=(7, 4))
    
    ax.plot(epochs, train_loss, label='Training Loss', linewidth=2, color='#1f77b4')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'Training History - Patient {patient_id}')
    ax.legend()
    ax.set_yscale('log')  # Log scale often better for loss curves
    
    plt.savefig(save_path / f'{patient_id}_training_loss.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(save_path / f'{patient_id}_training_loss.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_loss_components(history: Dict, patient_id: str, save_path: Path):
    """
    Plot individual loss components over training.
    
    Args:
        history: Dictionary containing loss component histories
        patient_id: Patient identifier
        save_path: Directory to save figure
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Data loss
    if 'data_loss' in history:
        axes[0, 0].plot(epochs, history['data_loss'], linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Data Loss')
        axes[0, 0].set_title('Data Loss (MSE)')
        axes[0, 0].set_yscale('log')
    
    # Physics losses
    if 'ns_loss' in history:
        axes[0, 1].plot(epochs, history['ns_loss'], linewidth=2, label='Navier-Stokes')
        if 'cont_loss' in history:
            axes[0, 1].plot(epochs, history['cont_loss'], linewidth=2, label='Continuity')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Physics Loss')
        axes[0, 1].set_title('Physics Residuals')
        axes[0, 1].legend()
        axes[0, 1].set_yscale('log')
    
    # WSS loss
    if 'wss_loss' in history:
        axes[1, 0].plot(epochs, history['wss_loss'], linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('WSS Loss')
        axes[1, 0].set_title('WSS Prediction Loss')
        axes[1, 0].set_yscale('log')
    
    # Total loss
    axes[1, 1].plot(epochs, history['train_loss'], linewidth=2, color='#1f77b4')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Total Loss')
    axes[1, 1].set_title('Total Loss')
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path / f'{patient_id}_loss_components.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(save_path / f'{patient_id}_loss_components.png', dpi=300, bbox_inches='tight')
    plt.close()

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
        "savefig.format": "pdf",
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

def plot_wss_comparison(coords: np.ndarray, wss_true: np.ndarray, wss_pred: np.ndarray,
                        patient_id: str, save_path: Path, view: str,
                        x_idx: int, y_idx: int, xlabel: str, ylabel: str):
    """
    Create side-by-side WSS comparison: CFD vs PINN vs Error with NRMSE.
    
    Args:
        coords: Spatial coordinates (N, 3)
        wss_true: Ground truth WSS (N,)
        wss_pred: Predicted WSS (N,)
        patient_id: Patient identifier
        save_path: Directory to save figure
        view: View name (XY, XZ, YZ)
        x_idx, y_idx: Coordinate indices for the view
        xlabel, ylabel: Axis labels
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Compute metrics
    rmse = np.sqrt(np.mean((wss_pred - wss_true) ** 2))
    nrmse = compute_nrmse(wss_true, wss_pred)
    
    # Common colorbar limits for CFD and PINN
    vmax = min(5, max(wss_true.max(), wss_pred.max()))
    
    # CFD (Ground Truth)
    sc1 = axes[0].scatter(coords[:, x_idx] * 1000, coords[:, y_idx] * 1000,
                          c=wss_true, cmap='jet', s=0.3, vmin=0, vmax=vmax)
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    axes[0].set_title('CFD')
    axes[0].set_aspect('equal')
    plt.colorbar(sc1, ax=axes[0], shrink=0.7, label='WSS (Pa)')
    
    # PINN Prediction
    sc2 = axes[1].scatter(coords[:, x_idx] * 1000, coords[:, y_idx] * 1000,
                          c=wss_pred, cmap='jet', s=0.3, vmin=0, vmax=vmax)
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel(ylabel)
    axes[1].set_title('PINN')
    axes[1].set_aspect('equal')
    plt.colorbar(sc2, ax=axes[1], shrink=0.7, label='WSS (Pa)')
    
    # Absolute Error with metrics
    error = np.abs(wss_pred - wss_true)
    sc3 = axes[2].scatter(coords[:, x_idx] * 1000, coords[:, y_idx] * 1000,
                          c=error, cmap='Reds', s=0.3, vmin=0, vmax=2)
    axes[2].set_xlabel(xlabel)
    axes[2].set_ylabel(ylabel)
    axes[2].set_title(f'|Error|\nNRMSE={nrmse:.4f}, RMSE={rmse:.2f} Pa')
    axes[2].set_aspect('equal')
    plt.colorbar(sc3, ax=axes[2], shrink=0.7, label='|Error| (Pa)')
    
    plt.tight_layout()
    plt.savefig(save_path / f'{patient_id}_WSS_{view}.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_velocity_comparison(coords: np.ndarray, vel_true: np.ndarray, vel_pred: np.ndarray,
                             patient_id: str, save_path: Path, view: str,
                             x_idx: int, y_idx: int, xlabel: str, ylabel: str,
                             component: str, comp_idx: int):
    """Create side-by-side velocity comparison: CFD vs PINN vs Error with NRMSE."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    v_true = vel_true[:, comp_idx]
    v_pred = vel_pred[:, comp_idx]
    
    # Compute metrics
    rmse = np.sqrt(np.mean((v_pred - v_true) ** 2))
    nrmse = compute_nrmse(v_true, v_pred)
    
    # Symmetric colorbar
    vmax = max(abs(v_true.min()), abs(v_true.max()), abs(v_pred.min()), abs(v_pred.max()))
    vmin = -vmax
    
    # CFD
    sc1 = axes[0].scatter(coords[:, x_idx] * 1000, coords[:, y_idx] * 1000,
                          c=v_true, cmap='RdBu_r', s=0.3, vmin=vmin, vmax=vmax)
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    axes[0].set_title('CFD')
    axes[0].set_aspect('equal')
    plt.colorbar(sc1, ax=axes[0], shrink=0.7, label=f'{component} (m/s)')
    
    # PINN
    sc2 = axes[1].scatter(coords[:, x_idx] * 1000, coords[:, y_idx] * 1000,
                          c=v_pred, cmap='RdBu_r', s=0.3, vmin=vmin, vmax=vmax)
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel(ylabel)
    axes[1].set_title('PINN')
    axes[1].set_aspect('equal')
    plt.colorbar(sc2, ax=axes[1], shrink=0.7, label=f'{component} (m/s)')
    
    # Error with metrics
    error = np.abs(v_pred - v_true)
    sc3 = axes[2].scatter(coords[:, x_idx] * 1000, coords[:, y_idx] * 1000,
                          c=error, cmap='Reds', s=0.3, vmin=0, vmax=vmax * 0.3)
    axes[2].set_xlabel(xlabel)
    axes[2].set_ylabel(ylabel)
    axes[2].set_title(f'|Error|\nNRMSE={nrmse:.4f}, RMSE={rmse:.4f} m/s')
    axes[2].set_aspect('equal')
    plt.colorbar(sc3, ax=axes[2], shrink=0.7, label=f'|Error| (m/s)')
    
    plt.tight_layout()
    plt.savefig(save_path / f'{patient_id}_vel_{component}_{view}.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_vessel_wss_comparison(coords: np.ndarray, wss_true: np.ndarray, wss_pred: np.ndarray,
                               patient_id: str, vessel_name: str, save_path: Path, 
                               view: str, x_idx: int, y_idx: int, xlabel: str, ylabel: str):
    """
    Create side-by-side WSS comparison for a specific vessel: CFD vs PINN vs Error.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Compute metrics
    rmse = np.sqrt(np.mean((wss_pred - wss_true) ** 2))
    nrmse = compute_nrmse(wss_true, wss_pred)
    
    # Common colorbar limits for CFD and PINN
    vmax = min(5, max(wss_true.max(), wss_pred.max()))
    
    # CFD (Ground Truth)
    sc1 = axes[0].scatter(coords[:, x_idx] * 1000, coords[:, y_idx] * 1000,
                          c=wss_true, cmap='jet', s=0.5, vmin=0, vmax=vmax)
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    axes[0].set_title('CFD')
    axes[0].set_aspect('equal')
    plt.colorbar(sc1, ax=axes[0], shrink=0.7, label='WSS (Pa)')
    
    # PINN Prediction
    sc2 = axes[1].scatter(coords[:, x_idx] * 1000, coords[:, y_idx] * 1000,
                          c=wss_pred, cmap='jet', s=0.5, vmin=0, vmax=vmax)
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel(ylabel)
    axes[1].set_title('PINN')
    axes[1].set_aspect('equal')
    plt.colorbar(sc2, ax=axes[1], shrink=0.7, label='WSS (Pa)')
    
    # Absolute Error with metrics
    error = np.abs(wss_pred - wss_true)
    sc3 = axes[2].scatter(coords[:, x_idx] * 1000, coords[:, y_idx] * 1000,
                          c=error, cmap='Reds', s=0.5, vmin=0, vmax=2)
    axes[2].set_xlabel(xlabel)
    axes[2].set_ylabel(ylabel)
    axes[2].set_title(f'|Error|\nNRMSE={nrmse:.4f}, RMSE={rmse:.2f} Pa')
    axes[2].set_aspect('equal')
    plt.colorbar(sc3, ax=axes[2], shrink=0.7, label='|Error| (Pa)')
    
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

