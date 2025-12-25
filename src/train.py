"""
Training Module for Physics-Informed Neural Networks.

This module implements the complete training pipeline for PINNs applied to
coronary artery hemodynamics. It handles data loading, model initialization,
loss computation, optimization, and evaluation.

Training Pipeline:
    1. Load patient-specific CFD data (wall surface + streamlines)
    2. Initialize PINN model with selected architecture
    3. Train with combined data and physics losses
    4. Evaluate against CFD ground truth
    5. Generate visualization plots

Loss Components:
    - Data Loss: MSE on velocity and WSS predictions
    - Physics Loss: Navier-Stokes and continuity residuals
    - WSS Physics: Consistency between predicted WSS and velocity gradients

Features:
    - Automatic mixed-precision training (if available)
    - Gradient clipping for stable training
    - Early stopping to prevent overfitting
    - Cosine annealing learning rate schedule

Attributes:
    LOSS_WEIGHTS (dict): Weight factors for each loss component.
    CHAR_VELOCITY (float): Characteristic velocity scale (m/s).
    CHAR_LENGTH (float): Characteristic length scale (m).
"""

import json
import time
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import DEVICE, FIGURES_PATH, MODELS_PATH, PATIENT_DATA, RESULTS_PATH, RHO
from src.dataset import CollocationSampler, PatientDataset, load_patient_data
from src.evaluate import evaluate_model
from src.model import KANPINN, FourierPINN, MultiResNetPINN, VanillaPINN
from src.physics import continuity_residual, navier_stokes_residual, wss_physics_residual
from src.plots import generate_all_plots
from src.utils import EarlyStopping

# Enable cuDNN autotuner for faster convolutions on your specific GPU
torch.backends.cudnn.benchmark = True

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

# Loss weights for combining different loss components
LOSS_WEIGHTS: Dict[str, float] = {
    'wss': 1.0,           # Wall shear stress prediction
    'velocity': 0.1,      # Velocity field prediction
    'navier_stokes': 1.0, # Navier-Stokes residual
    'continuity': 1.0,    # Continuity equation residual
    'wss_physics': 0.1,   # WSS physics constraint
}

# Characteristic scales for non-dimensionalization
CHAR_VELOCITY: float = 0.1   # Characteristic velocity (m/s)
CHAR_LENGTH: float = 0.05    # Characteristic length (m)


# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def train_patient(
    patient_id: str,
    epochs: int = 500,
    batch_size: int = 4096,
    learning_rate: float = 1e-4,
    n_collocation: int = 2048,
    patience: int = 50,
    hidden_dim: int = 256,
    num_blocks: int = 4,
    grad_clip: float = 1.0,
    arch: str = 'vanilla',
    kan_grid_size: int = 5,
    kan_spline_order: int = 3,
    num_frequencies: int = 64,
    fourier_scale: float = 10.0,
    verbose: bool = True
) -> Tuple[nn.Module, Dict]:
    """
    Train per-patient PINN with all improvements applied.

    This function implements the complete training pipeline including:
    - Data loading and preprocessing
    - Model initialization based on architecture choice
    - Training loop with combined data and physics losses
    - Early stopping and learning rate scheduling
    - Model evaluation and visualization

    Args:
        patient_id: Patient identifier from PATIENT_DATA registry.
        epochs: Maximum number of training epochs.
        batch_size: Number of samples per training batch.
        learning_rate: Initial learning rate for AdamW optimizer.
        n_collocation: Number of collocation points per batch for physics.
        patience: Number of epochs without improvement before early stopping.
        hidden_dim: Width of hidden layers in the network.
        num_blocks: Number of ResNet blocks (or KAN layers).
        grad_clip: Maximum gradient norm for clipping (0 to disable).
        arch: Architecture type. Options: 'vanilla', 'fourier', 'multi', 'kan'.
        kan_grid_size: KAN B-spline grid size (only for arch='kan').
        kan_spline_order: KAN B-spline order (only for arch='kan').
        num_frequencies: Number of Fourier frequencies (only for arch='fourier').
        fourier_scale: Scale of Fourier frequency matrix (only for arch='fourier').
        verbose: If True, print progress bars and status updates.

    Returns:
        Tuple containing:
            - model: Trained PyTorch model (best checkpoint loaded).
            - results: Dictionary with metrics, history, timing, and config.

    Raises:
        KeyError: If patient_id is not found in PATIENT_DATA.

    Example:
        >>> model, results = train_patient('H-12', epochs=100, arch='fourier')
        >>> print(f"R²: {results['metrics']['R2']:.4f}")
    """
    # Setup per-patient output folders
    patient_models = MODELS_PATH / patient_id
    patient_figures = FIGURES_PATH / patient_id
    patient_results = RESULTS_PATH / patient_id

    for path in [patient_models, patient_figures, patient_results]:
        path.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"TRAINING PINN FOR: {patient_id}")
    print(f"Category: {PATIENT_DATA[patient_id]['category']}")
    print("=" * 80)

    # Load data
    print("\n[LOADING DATA]")
    data, per_vessel = load_patient_data(patient_id)

    # Create dataset and dataloader
    dataset = PatientDataset(data)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    print(f"  Dataset: {len(dataset):,} points")

    # Create mesh-based collocation sampler
    print("\n[COLLOCATION SAMPLER]")
    collocation_sampler = CollocationSampler(
        coords=data['X'],
        has_wss=data['has_wss'],
        prefer_interior=True
    )
    print(f"  Wall points: {len(collocation_sampler.wall_indices):,}")
    print(f"  Interior points: {len(collocation_sampler.interior_indices):,}")
    print(f"  Sampling: {n_collocation} points/batch FROM MESH")

    # Initialize model based on architecture
    model, arch_name = _create_model(
        arch=arch,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        num_frequencies=num_frequencies,
        fourier_scale=fourier_scale,
        kan_grid_size=kan_grid_size,
        kan_spline_order=kan_spline_order
    )

    # Count parameters
    num_params = model.count_parameters()

    print("\n[MODEL]")
    if arch == 'kan':
        print(
            f"  Architecture: {arch_name} "
            f"({num_blocks} layers, {hidden_dim} dim, grid={kan_grid_size})"
        )
    elif arch == 'fourier':
        print(
            f"  Architecture: {arch_name} "
            f"({num_blocks} blocks, {hidden_dim} dim, {num_frequencies} freqs)"
        )
    else:
        print(f"  Architecture: {arch_name} ({num_blocks} blocks, {hidden_dim} dim)")
    print(f"  Parameters: {num_params:,}")

    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-5
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-6
    )

    # Physics scaling
    coord_scale = torch.tensor(
        dataset.scaler_X.data_range_,
        dtype=torch.float32,
        device=DEVICE
    ).view(1, 3)

    # Non-dimensional scales for physics residuals
    nse_scale = (RHO * CHAR_VELOCITY**2) / CHAR_LENGTH
    cont_scale = CHAR_VELOCITY / CHAR_LENGTH

    # Early stopping to prevent overfitting
    early_stopper = EarlyStopping(patience=patience, monitor='loss')

    # Training history
    history = _init_history()
    best_loss = float('inf')

    print(f"\n[TRAINING] Max {epochs} epochs, early stopping patience={patience}")
    print("-" * 80)

    # Start training timer
    train_start_time = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_losses = {k: 0.0 for k in history.keys() if k != 'lr'}

        pbar = tqdm(
            loader,
            desc=f"Epoch {epoch+1:3d}/{epochs}",
            disable=not verbose,
            leave=False
        )

        for batch in pbar:
            # Move batch to device
            coords = batch['coords'].to(DEVICE, non_blocking=True)
            wss_true = batch['wss'].to(DEVICE, non_blocking=True)
            vel_true = batch['velocity'].to(DEVICE, non_blocking=True)
            normals = batch['normals'].to(DEVICE, non_blocking=True)
            has_wss = batch['has_wss'].to(DEVICE, non_blocking=True).squeeze()

            # More efficient than zero_grad()
            optimizer.zero_grad(set_to_none=True)

            # Forward pass
            outputs = model(coords)
            vel_pred = torch.cat(
                [outputs['u'], outputs['v'], outputs['w']],
                dim=1
            )

            # Compute losses
            losses = _compute_losses(
                outputs=outputs,
                vel_pred=vel_pred,
                vel_true=vel_true,
                wss_true=wss_true,
                has_wss=has_wss,
                coords=coords,
                normals=normals,
                coord_scale=coord_scale,
                model=model,
                collocation_sampler=collocation_sampler,
                dataset=dataset,
                n_collocation=n_collocation,
                nse_scale=nse_scale,
                cont_scale=cont_scale
            )

            # Weighted total loss
            loss_total = (
                LOSS_WEIGHTS['wss'] * losses['wss'] +
                LOSS_WEIGHTS['velocity'] * losses['velocity'] +
                LOSS_WEIGHTS['navier_stokes'] * losses['navier_stokes'] +
                LOSS_WEIGHTS['continuity'] * losses['continuity'] +
                LOSS_WEIGHTS['wss_physics'] * losses['wss_physics']
            )

            # Backward with gradient clipping
            loss_total.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            # Accumulate losses
            epoch_losses['train_loss'] += loss_total.item()
            epoch_losses['data_loss'] += (
                losses['wss'].item() + losses['velocity'].item()
            )
            epoch_losses['wss_loss'] += losses['wss'].item()
            epoch_losses['vel_loss'] += losses['velocity'].item()
            epoch_losses['ns_loss'] += losses['navier_stokes'].item()
            epoch_losses['cont_loss'] += losses['continuity'].item()
            epoch_losses['wss_physics_loss'] += losses['wss_physics'].item()

            pbar.set_postfix({
                'Loss': f'{loss_total.item():.4f}',
                'WSS': f'{losses["wss"].item():.4f}'
            })

        scheduler.step()

        # Average losses
        n_batches = len(loader)
        for k in epoch_losses:
            epoch_losses[k] /= n_batches
            history[k].append(epoch_losses[k])
        history['lr'].append(optimizer.param_groups[0]['lr'])

        # Save best model
        if epoch_losses['train_loss'] < best_loss:
            best_loss = epoch_losses['train_loss']
            _save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=best_loss,
                hidden_dim=hidden_dim,
                num_blocks=num_blocks,
                path=patient_models / f'pinn_{patient_id}_best.pth'
            )

        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            lr = optimizer.param_groups[0]['lr']
            print(
                f"Epoch {epoch+1:3d} | "
                f"Loss: {epoch_losses['train_loss']:.4f} | "
                f"WSS: {epoch_losses['wss_loss']:.4f} | "
                f"Physics: {epoch_losses['ns_loss']:.2e} | "
                f"LR: {lr:.2e}"
            )

        # Check early stopping condition
        if early_stopper(epoch_losses['train_loss'], epoch):
            break

    # Calculate training time
    train_end_time = time.time()
    total_train_time = train_end_time - train_start_time
    epochs_trained = len(history['train_loss'])
    time_per_epoch = total_train_time / epochs_trained if epochs_trained > 0 else 0

    print("\n[TIMING]")
    print(f"  Total training time: {total_train_time:.1f}s ({total_train_time/60:.2f} min)")
    print(f"  Time per epoch: {time_per_epoch:.2f}s")
    print(f"  Epochs trained: {epochs_trained}")

    # Load best model
    # Note: weights_only=False is required to load optimizer state and config.
    # Only use on trusted checkpoint files.
    checkpoint = torch.load(
        patient_models / f'pinn_{patient_id}_best.pth',
        weights_only=False
    )
    model.load_state_dict(checkpoint['model_state_dict'])

    # Evaluation
    print("\n[EVALUATION]")
    metrics = evaluate_model(model, loader, dataset, coord_scale)

    # Generate plots
    print("\n[GENERATING PLOTS]")
    generate_all_plots(
        model, loader, dataset, patient_id,
        patient_figures, metrics, per_vessel, history
    )
    print(f"  Saved to: {patient_figures}")

    # Save results
    results = _compile_results(
        patient_id=patient_id,
        metrics=metrics,
        history=history,
        total_train_time=total_train_time,
        time_per_epoch=time_per_epoch,
        epochs_trained=epochs_trained,
        arch=arch,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks
    )

    # Save readable results file
    _save_results_file(
        path=patient_results / f'{patient_id}_results.txt',
        patient_id=patient_id,
        metrics=metrics,
        history=history,
        arch_name=arch_name,
        model=model,
        epochs_trained=epochs_trained,
        total_train_time=total_train_time,
        time_per_epoch=time_per_epoch
    )

    # Save history separately (can be large)
    with open(patient_results / f'{patient_id}_history.json', 'w') as f:
        json.dump(history, f)

    print(f"\n  Results saved to: {patient_results}")

    return model, results


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _create_model(
    arch: str,
    hidden_dim: int,
    num_blocks: int,
    num_frequencies: int,
    fourier_scale: float,
    kan_grid_size: int,
    kan_spline_order: int
) -> Tuple[nn.Module, str]:
    """
    Create and initialize a PINN model based on architecture choice.

    Args:
        arch: Architecture type ('vanilla', 'fourier', 'multi', 'kan').
        hidden_dim: Width of hidden layers.
        num_blocks: Number of blocks/layers.
        num_frequencies: Fourier frequencies (for 'fourier' arch).
        fourier_scale: Fourier scale (for 'fourier' arch).
        kan_grid_size: KAN grid size (for 'kan' arch).
        kan_spline_order: KAN spline order (for 'kan' arch).

    Returns:
        Tuple of (model, architecture_name).
    """
    if arch == 'vanilla':
        model = VanillaPINN(
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            head_layers=2,
            predict_wss=True
        ).to(DEVICE)
        arch_name = 'VanillaPINN'
    elif arch == 'fourier':
        model = FourierPINN(
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            predict_wss=True,
            num_frequencies=num_frequencies,
            fourier_scale=fourier_scale
        ).to(DEVICE)
        arch_name = 'FourierPINN'
    elif arch == 'kan':
        model = KANPINN(
            in_dim=3,
            out_dim=5,
            hidden_dim=hidden_dim,
            num_layers=num_blocks,
            grid_size=kan_grid_size,
            spline_order=kan_spline_order,
            predict_wss=True
        ).to(DEVICE)
        arch_name = 'KANPINN'
    else:  # 'multi'
        model = MultiResNetPINN(
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            predict_wss=True
        ).to(DEVICE)
        arch_name = 'MultiResNetPINN'

    return model, arch_name


def _init_history() -> Dict[str, list]:
    """
    Initialize training history dictionary.

    Returns:
        Dictionary with empty lists for each tracked metric.
    """
    return {
        'train_loss': [],
        'data_loss': [],
        'wss_loss': [],
        'vel_loss': [],
        'ns_loss': [],
        'cont_loss': [],
        'wss_physics_loss': [],
        'lr': []
    }


def _compute_losses(
    outputs: Dict[str, torch.Tensor],
    vel_pred: torch.Tensor,
    vel_true: torch.Tensor,
    wss_true: torch.Tensor,
    has_wss: torch.Tensor,
    coords: torch.Tensor,
    normals: torch.Tensor,
    coord_scale: torch.Tensor,
    model: nn.Module,
    collocation_sampler: CollocationSampler,
    dataset: PatientDataset,
    n_collocation: int,
    nse_scale: float,
    cont_scale: float
) -> Dict[str, torch.Tensor]:
    """
    Compute all loss components for a single batch.

    Args:
        outputs: Model output dictionary.
        vel_pred: Predicted velocity tensor.
        vel_true: Ground truth velocity tensor.
        wss_true: Ground truth WSS tensor.
        has_wss: Boolean mask for wall points.
        coords: Input coordinates.
        normals: Surface normal vectors.
        coord_scale: Coordinate scaling factors.
        model: The PINN model.
        collocation_sampler: Sampler for collocation points.
        dataset: The patient dataset (for scaling).
        n_collocation: Number of collocation points.
        nse_scale: Navier-Stokes normalization scale.
        cont_scale: Continuity normalization scale.

    Returns:
        Dictionary with loss values for each component.
    """
    # Velocity loss (all points)
    loss_vel = nn.MSELoss()(vel_pred, vel_true)

    # WSS loss (wall points only)
    if has_wss.any():
        loss_wss = nn.MSELoss()(outputs['wss'][has_wss], wss_true[has_wss])
        # WSS physics constraint
        wss_res = wss_physics_residual(
            model, coords[has_wss], normals[has_wss], coord_scale
        )
        loss_wss_physics = (wss_res**2).mean()
    else:
        loss_wss = torch.tensor(0.0, device=DEVICE)
        loss_wss_physics = torch.tensor(0.0, device=DEVICE)

    # Physics at data points
    f_u, f_v, f_w = navier_stokes_residual(model, coords, coord_scale)
    cont = continuity_residual(model, coords, coord_scale)
    loss_nse_data = (f_u**2 + f_v**2 + f_w**2).mean() / (nse_scale**2)
    loss_cont_data = (cont**2).mean() / (cont_scale**2)

    # Physics at mesh-based collocation points
    colloc_raw = collocation_sampler.sample(n_collocation)
    colloc_scaled = dataset.scaler_X.transform(colloc_raw)
    colloc_tensor = torch.FloatTensor(colloc_scaled).to(DEVICE)

    f_u_c, f_v_c, f_w_c = navier_stokes_residual(model, colloc_tensor, coord_scale)
    cont_c = continuity_residual(model, colloc_tensor, coord_scale)
    loss_nse_colloc = (f_u_c**2 + f_v_c**2 + f_w_c**2).mean() / (nse_scale**2)
    loss_cont_colloc = (cont_c**2).mean() / (cont_scale**2)

    # Average data and collocation physics losses
    loss_nse = 0.5 * loss_nse_data + 0.5 * loss_nse_colloc
    loss_cont = 0.5 * loss_cont_data + 0.5 * loss_cont_colloc

    return {
        'wss': loss_wss,
        'velocity': loss_vel,
        'navier_stokes': loss_nse,
        'continuity': loss_cont,
        'wss_physics': loss_wss_physics
    }


def _save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    loss: float,
    hidden_dim: int,
    num_blocks: int,
    path
) -> None:
    """
    Save model checkpoint.

    Args:
        model: The trained model.
        optimizer: The optimizer with current state.
        epoch: Current epoch number.
        loss: Current best loss value.
        hidden_dim: Model hidden dimension.
        num_blocks: Number of model blocks.
        path: Path to save the checkpoint.
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': {
            'hidden_dim': hidden_dim,
            'num_blocks': num_blocks
        }
    }, path)


def _compile_results(
    patient_id: str,
    metrics: Dict,
    history: Dict,
    total_train_time: float,
    time_per_epoch: float,
    epochs_trained: int,
    arch: str,
    hidden_dim: int,
    num_blocks: int
) -> Dict:
    """
    Compile training results into a dictionary.

    Args:
        patient_id: Patient identifier.
        metrics: Evaluation metrics dictionary.
        history: Training history dictionary.
        total_train_time: Total training time in seconds.
        time_per_epoch: Average time per epoch.
        epochs_trained: Number of epochs completed.
        arch: Architecture name.
        hidden_dim: Model hidden dimension.
        num_blocks: Number of model blocks.

    Returns:
        Complete results dictionary.
    """
    return {
        'patient_id': patient_id,
        'category': PATIENT_DATA[patient_id]['category'],
        'metrics': metrics,
        'history': history,
        'timing': {
            'total_seconds': total_train_time,
            'seconds_per_epoch': time_per_epoch,
            'epochs_trained': epochs_trained
        },
        'config': {
            'arch': arch,
            'epochs_trained': epochs_trained,
            'hidden_dim': hidden_dim,
            'num_blocks': num_blocks
        }
    }


def _save_results_file(
    path,
    patient_id: str,
    metrics: Dict,
    history: Dict,
    arch_name: str,
    model: nn.Module,
    epochs_trained: int,
    total_train_time: float,
    time_per_epoch: float
) -> None:
    """
    Save human-readable results file.

    Args:
        path: Output file path.
        patient_id: Patient identifier.
        metrics: Evaluation metrics.
        history: Training history.
        arch_name: Architecture name string.
        model: The trained model.
        epochs_trained: Number of epochs completed.
        total_train_time: Total training time.
        time_per_epoch: Average time per epoch.
    """
    with open(path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write(f"PINN TRAINING RESULTS - PATIENT {patient_id}\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Patient ID: {patient_id}\n")
        f.write(f"Category: {PATIENT_DATA[patient_id]['category']}\n\n")

        f.write("-" * 40 + "\n")
        f.write("EVALUATION METRICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"  RMSE:     {metrics['RMSE']:.4f} Pa\n")
        f.write(f"  MAE:      {metrics['MAE']:.4f} Pa\n")
        f.write(f"  NRMSE:    {metrics['NRMSE']:.4f}\n")
        f.write(f"  R²:       {metrics['R2']:.4f}\n\n")

        f.write("-" * 40 + "\n")
        f.write("TRAINING SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Architecture: {arch_name}\n")
        f.write(f"  Parameters:  {model.count_parameters():,}\n")
        f.write(f"  Epochs:      {epochs_trained}\n")
        f.write(f"  Final Loss:  {history['train_loss'][-1]:.6f}\n")
        f.write(f"  Best Loss:   {min(history['train_loss']):.6f}\n\n")

        f.write("-" * 40 + "\n")
        f.write("TIMING\n")
        f.write("-" * 40 + "\n")
        f.write(
            f"  Total time:      {total_train_time:.1f}s "
            f"({total_train_time/60:.2f} min)\n"
        )
        f.write(f"  Time per epoch:  {time_per_epoch:.2f}s\n")
