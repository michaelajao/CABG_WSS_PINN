"""
Visualization module for PINN

Contains all plotting functions for:
- Training history visualization
- Prediction results and error analysis
- 3D spatial visualizations
- 2D plane cuts
- Distribution analysis
- Newtonian vs Non-Newtonian comparisons
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
from typing import Dict, Optional, List
from sklearn.metrics import confusion_matrix

from .config import (
    MODEL_PATH, RESULTS_PATH, FIGURES_PATH, VISUALIZATIONS_3D_PATH,
    PLANES_2D_PATH, PLOT_STYLE, PLOT_DPI, MAX_3D_POINTS,
    HIGH_RISK_WSS_THRESHOLD
)
from .utils import sample_dataframe

# Set plotting style
plt.style.use(PLOT_STYLE)
sns.set_palette("husl")


# =============================================================================
# TRAINING VISUALIZATION
# =============================================================================

def plot_training_history(
    history_path: Optional[Path] = None,
    save_path: Path = RESULTS_PATH
):
    """
    Plot training history from saved JSON file

    Args:
        history_path: Path to training_history.json (if None, uses MODEL_PATH)
        save_path: Directory to save figure
    """
    if history_path is None:
        history_path = MODEL_PATH / "training_history.json"

    print("\n[PLOT] Training history...")

    with open(history_path, 'r') as f:
        history = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('PINN Training History', fontsize=16, fontweight='bold')

    epochs = range(1, len(history['train_loss']) + 1)

    # Total loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')

    # Data loss (WSS)
    axes[0, 1].plot(epochs, history['data_loss_wss'], 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('WSS Loss')
    axes[0, 1].set_title('Data Fitting Loss (WSS)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_yscale('log')

    # Physics losses
    axes[1, 0].plot(epochs, history['physics_loss_nse'], 'purple', label='Navier-Stokes', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('NSE Residual')
    axes[1, 0].set_title('Navier-Stokes Equation Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')

    axes[1, 1].plot(epochs, history['physics_loss_cont'], 'orange', label='Continuity', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Continuity Residual')
    axes[1, 1].set_title('Continuity Equation Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')

    plt.tight_layout()
    plt.savefig(save_path / "training_history.png", dpi=PLOT_DPI, bbox_inches='tight')
    print(f"  Saved: {save_path / 'training_history.png'}")
    plt.close()


# =============================================================================
# PREDICTION RESULTS VISUALIZATION
# =============================================================================

def plot_prediction_results(
    results_df: pd.DataFrame,
    save_path: Path = RESULTS_PATH
):
    """
    Plot prediction vs ground truth with various error analyses

    Args:
        results_df: DataFrame with 'WSS_True', 'WSS_Pred', 'Error', etc.
        save_path: Directory to save figure
    """
    print("\n[PLOT] Prediction results...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('PINN Prediction Results', fontsize=16, fontweight='bold')

    # Scatter plot: True vs Predicted
    axes[0, 0].scatter(results_df['WSS_True'], results_df['WSS_Pred'],
                      alpha=0.3, s=1, c='blue')
    max_val = max(results_df['WSS_True'].max(), results_df['WSS_Pred'].max())
    axes[0, 0].plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect prediction')
    axes[0, 0].set_xlabel('True WSS (Pa)')
    axes[0, 0].set_ylabel('Predicted WSS (Pa)')
    axes[0, 0].set_title('Prediction vs Ground Truth')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Error distribution
    axes[0, 1].hist(results_df['Error'], bins=100, color='purple', alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Prediction Error (Pa)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title(f'Error Distribution\nMean: {results_df["Error"].mean():.3f} Pa')
    axes[0, 1].grid(True, alpha=0.3)

    # Absolute error distribution
    axes[0, 2].hist(results_df['Abs_Error'], bins=100, color='orange', alpha=0.7, edgecolor='black')
    axes[0, 2].set_xlabel('Absolute Error (Pa)')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title(f'Absolute Error\nMAE: {results_df["Abs_Error"].mean():.3f} Pa')
    axes[0, 2].grid(True, alpha=0.3)

    # Bland-Altman plot
    mean_wss = (results_df['WSS_True'] + results_df['WSS_Pred']) / 2
    diff = results_df['Error']
    axes[1, 0].scatter(mean_wss, diff, alpha=0.3, s=1)
    axes[1, 0].axhline(0, color='red', linestyle='--', linewidth=2)
    axes[1, 0].axhline(diff.mean(), color='blue', linestyle='--', linewidth=2,
                      label=f'Mean: {diff.mean():.3f}')
    axes[1, 0].axhline(diff.mean() + 1.96*diff.std(), color='gray', linestyle='--', linewidth=1)
    axes[1, 0].axhline(diff.mean() - 1.96*diff.std(), color='gray', linestyle='--', linewidth=1,
                      label='+/- 1.96 SD')
    axes[1, 0].set_xlabel('Mean WSS (Pa)')
    axes[1, 0].set_ylabel('Difference (Pred - True) [Pa]')
    axes[1, 0].set_title('Bland-Altman Plot')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Residual vs Predicted
    axes[1, 1].scatter(results_df['WSS_Pred'], results_df['Error'],
                      alpha=0.3, s=1, c='green')
    axes[1, 1].axhline(0, color='red', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('Predicted WSS (Pa)')
    axes[1, 1].set_ylabel('Residual (Pa)')
    axes[1, 1].set_title('Residual Plot')
    axes[1, 1].grid(True, alpha=0.3)

    # High-risk classification confusion matrix
    true_high_risk = results_df['WSS_True'] > HIGH_RISK_WSS_THRESHOLD
    pred_high_risk = results_df['WSS_Pred'] > HIGH_RISK_WSS_THRESHOLD

    cm = confusion_matrix(true_high_risk, pred_high_risk)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 2],
               xticklabels=['Safe', 'High-risk'], yticklabels=['Safe', 'High-risk'])
    axes[1, 2].set_xlabel('Predicted')
    axes[1, 2].set_ylabel('True')
    axes[1, 2].set_title(f'High-Risk Classification\n(Threshold: {HIGH_RISK_WSS_THRESHOLD} Pa)')

    plt.tight_layout()
    plt.savefig(save_path / "prediction_results.png", dpi=PLOT_DPI, bbox_inches='tight')
    print(f"  Saved: {save_path / 'prediction_results.png'}")
    plt.close()


# =============================================================================
# SPATIAL VISUALIZATION
# =============================================================================

def plot_spatial_error_distribution(
    results_df: pd.DataFrame,
    save_path: Path = RESULTS_PATH
):
    """
    Create 3D scatter plot of spatial error distribution

    Args:
        results_df: DataFrame with 'X', 'Y', 'Z', 'Abs_Error' columns
        save_path: Directory to save HTML file
    """
    print("\n[PLOT] Spatial error distribution (3D)...")

    # Sample if too large
    df_viz = sample_dataframe(results_df, MAX_3D_POINTS)

    fig = go.Figure(data=[go.Scatter3d(
        x=df_viz['X'] * 1000,  # Convert to mm
        y=df_viz['Y'] * 1000,
        z=df_viz['Z'] * 1000,
        mode='markers',
        marker=dict(
            size=2,
            color=df_viz['Abs_Error'],
            colorscale='Jet',
            colorbar=dict(title="Abs Error (Pa)"),
            cmin=0,
            cmax=df_viz['Abs_Error'].quantile(0.95)
        ),
        text=[f'Error: {e:.3f} Pa' for e in df_viz['Abs_Error']],
        hoverinfo='text'
    )])

    fig.update_layout(
        title='Spatial Distribution of Prediction Errors',
        scene=dict(
            xaxis_title='X (mm)',
            yaxis_title='Y (mm)',
            zaxis_title='Z (mm)',
            aspectmode='data'
        ),
        width=1000,
        height=800
    )

    fig.write_html(save_path / "spatial_error_distribution.html")
    print(f"  Saved: {save_path / 'spatial_error_distribution.html'}")


def create_comparison_3d(
    results_df: pd.DataFrame,
    save_path: Path = RESULTS_PATH
):
    """
    Create side-by-side 3D comparison of ground truth vs predictions

    Args:
        results_df: DataFrame with columns 'X', 'Y', 'Z', 'WSS_True', 'WSS_Pred'
        save_path: Directory to save HTML file
    """
    print("\n[PLOT] 3D comparison (CFD vs PINN)...")

    # Sample data
    df_viz = sample_dataframe(results_df, MAX_3D_POINTS)

    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Ground Truth WSS', 'PINN Predicted WSS'),
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]]
    )

    # Ground truth
    fig.add_trace(
        go.Scatter3d(
            x=df_viz['X'] * 1000,
            y=df_viz['Y'] * 1000,
            z=df_viz['Z'] * 1000,
            mode='markers',
            marker=dict(
                size=2,
                color=df_viz['WSS_True'],
                colorscale='Jet',
                colorbar=dict(title="WSS (Pa)", x=0.45),
                cmin=0,
                cmax=df_viz['WSS_True'].quantile(0.95)
            ),
            name='Ground Truth'
        ),
        row=1, col=1
    )

    # Predictions
    fig.add_trace(
        go.Scatter3d(
            x=df_viz['X'] * 1000,
            y=df_viz['Y'] * 1000,
            z=df_viz['Z'] * 1000,
            mode='markers',
            marker=dict(
                size=2,
                color=df_viz['WSS_Pred'],
                colorscale='Jet',
                colorbar=dict(title="WSS (Pa)", x=1.0),
                cmin=0,
                cmax=df_viz['WSS_Pred'].quantile(0.95)
            ),
            name='PINN Prediction'
        ),
        row=1, col=2
    )

    fig.update_layout(
        title_text='PINN vs CFD Ground Truth Comparison',
        height=600,
        showlegend=False
    )

    fig.write_html(save_path / "comparison_3d.html")
    print(f"  Saved: {save_path / 'comparison_3d.html'}")


# =============================================================================
# DISTRIBUTION PLOTS
# =============================================================================

def plot_wss_distributions(
    datasets: Dict[str, pd.DataFrame],
    save_path: Path = FIGURES_PATH
):
    """
    Plot WSS distribution comparison across datasets

    Args:
        datasets: Dictionary mapping names to DataFrames with WSS data
        save_path: Directory to save figure
    """
    print("\n[PLOT] WSS distributions...")

    n_datasets = len(datasets)
    if n_datasets == 0:
        return

    fig, axes = plt.subplots(2, n_datasets, figsize=(6*n_datasets, 10))
    if n_datasets == 1:
        axes = axes.reshape(-1, 1)

    fig.suptitle('Wall Shear Stress Distribution Analysis', fontsize=16, fontweight='bold')

    colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12', '#9b59b6']

    for idx, (name, df) in enumerate(datasets.items()):
        wss_data = df['WSS']

        # Histogram
        axes[0, idx].hist(wss_data, bins=100, color=colors[idx % len(colors)],
                         alpha=0.7, edgecolor='black')
        axes[0, idx].axvline(HIGH_RISK_WSS_THRESHOLD, color='red', linestyle='--',
                            linewidth=2, label=f'{HIGH_RISK_WSS_THRESHOLD} Pa threshold')
        axes[0, idx].set_xlabel('WSS (Pa)', fontsize=10)
        axes[0, idx].set_ylabel('Frequency', fontsize=10)
        axes[0, idx].set_title(f'{name}\nMean: {wss_data.mean():.3f} Pa', fontsize=11)
        axes[0, idx].legend()
        axes[0, idx].grid(True, alpha=0.3)

        # Box plot
        axes[1, idx].boxplot([wss_data], vert=True, patch_artist=True,
                            boxprops=dict(facecolor=colors[idx % len(colors)], alpha=0.7))
        axes[1, idx].axhline(HIGH_RISK_WSS_THRESHOLD, color='red', linestyle='--', linewidth=2)
        axes[1, idx].set_ylabel('WSS (Pa)', fontsize=10)
        axes[1, idx].set_title(f'Median: {wss_data.median():.3f} Pa', fontsize=11)
        axes[1, idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path / "wss_distributions.png", dpi=PLOT_DPI, bbox_inches='tight')
    print(f"  Saved: {save_path / 'wss_distributions.png'}")
    plt.close()


# =============================================================================
# 2D PLANE CUTS
# =============================================================================

def create_2d_plane_cuts(
    df: pd.DataFrame,
    patient_name: str,
    save_path: Path = PLANES_2D_PATH
):
    """
    Create XY and XZ plane cuts showing WSS distribution

    Args:
        df: DataFrame with 'X', 'Y', 'Z', 'WSS' columns
        patient_name: Name for the plot title
        save_path: Directory to save figure
    """
    required_cols = ['X', 'Y', 'Z', 'WSS']
    if not all(col in df.columns for col in required_cols):
        print(f"  [WARNING] Missing required columns for {patient_name}")
        return

    X, Y, Z = df['X'].values, df['Y'].values, df['Z'].values
    wss_data = df['WSS'].values

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'{patient_name} - WSS Distribution', fontsize=16, fontweight='bold')

    # XY Plane
    scatter = axes[0].scatter(X, Y, c=wss_data, cmap='viridis', s=1, alpha=0.9)
    axes[0].set_xlabel('X (mm)', fontsize=12)
    axes[0].set_ylabel('Y (mm)', fontsize=12)
    axes[0].set_title('XY Plane', fontsize=14)
    axes[0].set_aspect('equal', adjustable='box')
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[0], label='WSS (Pa)')

    # XZ Plane
    scatter = axes[1].scatter(X, Z, c=wss_data, cmap='viridis', s=1, alpha=0.9)
    axes[1].set_xlabel('X (mm)', fontsize=12)
    axes[1].set_ylabel('Z (mm)', fontsize=12)
    axes[1].set_title('XZ Plane', fontsize=14)
    axes[1].set_aspect('equal', adjustable='box')
    axes[1].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[1], label='WSS (Pa)')

    plt.tight_layout()

    safe_name = patient_name.replace(' ', '_').replace('/', '_')
    filename = save_path / f"{safe_name}_wss_planes.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  Saved: {filename.name}")
    plt.close()


# =============================================================================
# NEWTONIAN VS NON-NEWTONIAN COMPARISON
# =============================================================================

def compare_newtonian_models(
    df_newt: pd.DataFrame,
    df_non_newt: pd.DataFrame,
    patient_name: str,
    save_path: Path = FIGURES_PATH
):
    """
    Compare Newtonian vs Non-Newtonian WSS results

    Args:
        df_newt: DataFrame with Newtonian WSS
        df_non_newt: DataFrame with Non-Newtonian WSS
        patient_name: Patient identifier
        save_path: Directory to save figure
    """
    print(f"\n[PLOT] Newtonian vs Non-Newtonian comparison ({patient_name})...")

    # Calculate differences
    min_len = min(len(df_newt), len(df_non_newt))
    wss_newt = df_newt['WSS'].values[:min_len]
    wss_non_newt = df_non_newt['WSS'].values[:min_len]

    wss_diff = wss_non_newt - wss_newt
    wss_percent_diff = (wss_diff / (wss_non_newt + 1e-10)) * 100

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f'Newtonian vs Non-Newtonian Comparison ({patient_name})',
                fontsize=16, fontweight='bold')

    # Scatter plot
    axes[0, 0].scatter(wss_newt, wss_non_newt, alpha=0.3, s=1)
    max_val = max(wss_newt.max(), wss_non_newt.max())
    axes[0, 0].plot([0, max_val], [0, max_val], 'r--', label='y=x')
    axes[0, 0].set_xlabel('Newtonian WSS (Pa)')
    axes[0, 0].set_ylabel('Non-Newtonian WSS (Pa)')
    axes[0, 0].set_title('WSS Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Difference distribution
    axes[0, 1].hist(wss_diff, bins=100, color='purple', alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('WSS Difference (Non-Newt - Newt) [Pa]')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title(f'Difference Distribution\nMean: {wss_diff.mean():.3f} Pa')
    axes[0, 1].grid(True, alpha=0.3)

    # Percentage difference
    axes[1, 0].hist(wss_percent_diff, bins=100, color='orange', alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Percentage Difference (%)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title(f'Percentage Difference\nMean: {wss_percent_diff.mean():.2f}%')
    axes[1, 0].grid(True, alpha=0.3)

    # Bland-Altman plot
    mean_wss = (wss_newt + wss_non_newt) / 2
    axes[1, 1].scatter(mean_wss, wss_diff, alpha=0.3, s=1)
    axes[1, 1].axhline(0, color='red', linestyle='--', linewidth=2)
    axes[1, 1].axhline(wss_diff.mean(), color='blue', linestyle='--', linewidth=2,
                      label=f'Mean diff: {wss_diff.mean():.3f}')
    axes[1, 1].axhline(wss_diff.mean() + 1.96*wss_diff.std(), color='gray',
                      linestyle='--', linewidth=1, label='+/- 1.96 SD')
    axes[1, 1].axhline(wss_diff.mean() - 1.96*wss_diff.std(), color='gray',
                      linestyle='--', linewidth=1)
    axes[1, 1].set_xlabel('Mean WSS (Pa)')
    axes[1, 1].set_ylabel('Difference (Pa)')
    axes[1, 1].set_title('Bland-Altman Plot')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path / f"newtonian_comparison_{patient_name}.png",
                dpi=PLOT_DPI, bbox_inches='tight')
    print(f"  Saved: newtonian_comparison_{patient_name}.png")
    plt.close()


# =============================================================================
# CFD vs PINN COMPARISON (3x1) FOR PLANES AND STREAMLINES
# =============================================================================

def _compute_nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Normalized RMSE (NRMSE) using range of true values.

    NRMSE = RMSE / (max(y_true) - min(y_true) + eps)
    """
    eps = 1e-12
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    return rmse / (np.max(y_true) - np.min(y_true) + eps)


def plot_plane_comparison(
    results_df: pd.DataFrame,
    patient_name: str,
    plane: str = 'XY',
    save_path: Path = FIGURES_PATH,
    sample_points: Optional[int] = None
):
    """Create 3x1 plane comparison plot: CFD (True) vs PINN (Pred) vs Error.

    Args:
        results_df: DataFrame with columns 'X','Y','Z','WSS_True','WSS_Pred'.
        patient_name: Identifier for file naming.
        plane: 'XY' or 'XZ'.
        save_path: Directory to save figure.
        sample_points: Optional subsampling to reduce plotting load.
    """
    required = {'X','Y','Z','WSS_True','WSS_Pred'}
    if not required.issubset(results_df.columns):
        print("  Warning: results_df missing required columns for plane comparison")
        return

    df = results_df.copy()
    if sample_points and len(df) > sample_points:
        df = df.sample(sample_points, random_state=42)

    # Select plane axes
    if plane.upper() == 'XY':
        x_vals, y_vals = df['X'].values, df['Y'].values
        x_label, y_label = 'X (m)', 'Y (m)'
        plane_label = 'XY'
    elif plane.upper() == 'XZ':
        x_vals, y_vals = df['X'].values, df['Z'].values
        x_label, y_label = 'X (m)', 'Z (m)'
        plane_label = 'XZ'
    elif plane.upper() == 'YZ':
        x_vals, y_vals = df['Y'].values, df['Z'].values
        x_label, y_label = 'Y (m)', 'Z (m)'
        plane_label = 'YZ'
    else:
        print(f"  [WARNING] Unsupported plane '{plane}'. Use 'XY', 'XZ', or 'YZ'.")
        return

    y_true = df['WSS_True'].values
    y_pred = df['WSS_Pred'].values
    abs_err = np.abs(y_pred - y_true)
    nrmse = _compute_nrmse(y_true, y_pred)

    vmax_true = np.quantile(y_true, 0.95)
    vmax_pred = np.quantile(y_pred, 0.95)
    vmax_err = np.quantile(abs_err, 0.95)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'{patient_name} - {plane_label} Plane CFD vs PINN (NRMSE={nrmse:.4f})',
                 fontsize=16, fontweight='bold')

    # CFD ground truth
    sc0 = axes[0].scatter(x_vals, y_vals, c=y_true, cmap='viridis', s=2, alpha=0.9,
                          vmin=0, vmax=vmax_true)
    axes[0].set_title('CFD WSS (Ground Truth)')
    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel(y_label)
    axes[0].set_aspect('equal', adjustable='box')
    plt.colorbar(sc0, ax=axes[0], label='WSS (Pa)')

    # PINN prediction
    sc1 = axes[1].scatter(x_vals, y_vals, c=y_pred, cmap='viridis', s=2, alpha=0.9,
                          vmin=0, vmax=vmax_pred)
    axes[1].set_title('PINN Predicted WSS')
    axes[1].set_xlabel(x_label)
    axes[1].set_ylabel(y_label)
    axes[1].set_aspect('equal', adjustable='box')
    plt.colorbar(sc1, ax=axes[1], label='WSS (Pa)')

    # Absolute Error
    sc2 = axes[2].scatter(x_vals, y_vals, c=abs_err, cmap='inferno', s=2, alpha=0.9,
                          vmin=0, vmax=vmax_err)
    axes[2].set_title('Absolute Error (Pa)')
    axes[2].set_xlabel(x_label)
    axes[2].set_ylabel(y_label)
    axes[2].set_aspect('equal', adjustable='box')
    plt.colorbar(sc2, ax=axes[2], label='|Error| (Pa)')

    for ax in axes:
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    safe_name = patient_name.replace(' ', '_').replace('/', '_')
    filename = save_path / f"{safe_name}_{plane_label}_plane_comparison.png"
    plt.savefig(filename, dpi=PLOT_DPI, bbox_inches='tight')
    print(f"  Saved: {filename.name}")
    plt.close()


def plot_streamline_comparison(
    cfd_streamline_df: pd.DataFrame,
    pinn_results_df: pd.DataFrame,
    patient_name: str,
    save_path: Path = FIGURES_PATH,
    sample_points: Optional[int] = None
):
    """Create 3x1 streamline comparison: CFD vs PINN vs Error with NRMSE.

    Performs nearest-neighbor mapping from streamline CFD points to PINN
    prediction coordinates.

    Args:
        cfd_streamline_df: Raw CFD streamline DataFrame (must contain X,Y,Z,WSS).
        pinn_results_df: DataFrame with columns 'X','Y','Z','WSS_Pred'. If it
            also has 'WSS_True' will use it for direct alignment check.
        patient_name: Identifier for figure naming.
        save_path: Directory to save figure.
        sample_points: Optional subsampling of CFD streamline for speed.
    """
    cols_needed_stream = ['X','Y','Z','WSS']
    # Attempt column resolution using common aliases
    alias_map = {
        'X': ['X [ m ]','X','x'],
        'Y': ['Y [ m ]','Y','y'],
        'Z': ['Z [ m ]','Z','z'],
        'WSS': ['Wall Shear [ Pa ]','Wall Shear','WSS','wss']
    }
    def resolve(df, target):
        for cand in alias_map[target]:
            if cand in df.columns:
                return cand
        return None

    # Resolve columns in CFD streamline
    resolved_cols = {t: resolve(cfd_streamline_df, t) for t in alias_map}
    if any(v is None for v in resolved_cols.values()):
        print("  Warning: Unable to resolve required columns in CFD streamline DataFrame")
        return

    stream_df = cfd_streamline_df[[resolved_cols['X'], resolved_cols['Y'], resolved_cols['Z'], resolved_cols['WSS']]].copy()
    stream_df.columns = ['X','Y','Z','WSS_True']

    if sample_points and len(stream_df) > sample_points:
        stream_df = stream_df.sample(sample_points, random_state=42).sort_index()

    # Prepare PINN coordinates
    if not {'X','Y','Z','WSS_Pred'}.issubset(pinn_results_df.columns):
        print("  Warning: pinn_results_df missing required columns for streamline comparison")
        return

    pinn_coords = pinn_results_df[['X','Y','Z']].values
    pinn_wss_pred = pinn_results_df['WSS_Pred'].values

    # Nearest neighbor mapping
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(pinn_coords)
        dists, idxs = tree.query(stream_df[['X','Y','Z']].values)
    except Exception:
        # Fallback naive argmin (slower)
        print("  Warning: scipy.spatial.cKDTree unavailable, using slow nearest-neighbor fallback")
        coords_stream = stream_df[['X','Y','Z']].values
        idxs = []
        for pt in coords_stream:
            d = np.linalg.norm(pinn_coords - pt, axis=1)
            idxs.append(np.argmin(d))
        idxs = np.array(idxs)
        dists = np.zeros_like(idxs, dtype=float)

    stream_df['WSS_Pred'] = pinn_wss_pred[idxs]

    # Compute arc length (in meters converted to mm for plotting if desired)
    pts = stream_df[['X','Y','Z']].values
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])  # meters
    stream_df['ArcLength'] = arc

    y_true = stream_df['WSS_True'].values
    y_pred = stream_df['WSS_Pred'].values
    err = y_pred - y_true
    abs_err = np.abs(err)
    nrmse = _compute_nrmse(y_true, y_pred)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'{patient_name} - Streamline CFD vs PINN (NRMSE={nrmse:.4f})',
                 fontsize=16, fontweight='bold')

    axes[0].plot(stream_df['ArcLength'], y_true, color='blue', linewidth=1)
    axes[0].set_title('CFD WSS (True)')
    axes[0].set_xlabel('Arc Length (m)')
    axes[0].set_ylabel('WSS (Pa)')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(stream_df['ArcLength'], y_pred, color='green', linewidth=1)
    axes[1].set_title('PINN WSS (Predicted)')
    axes[1].set_xlabel('Arc Length (m)')
    axes[1].set_ylabel('WSS (Pa)')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(stream_df['ArcLength'], err, color='purple', linewidth=1)
    axes[2].axhline(0, color='red', linestyle='--', linewidth=1)
    axes[2].set_title('Error (Pred - True) [Pa]')
    axes[2].set_xlabel('Arc Length (m)')
    axes[2].set_ylabel('Error (Pa)')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    safe_name = patient_name.replace(' ', '_').replace('/', '_')
    filename = save_path / f"{safe_name}_streamline_comparison.png"
    plt.savefig(filename, dpi=PLOT_DPI, bbox_inches='tight')
    print(f"  Saved: {filename.name}")
    plt.close()

