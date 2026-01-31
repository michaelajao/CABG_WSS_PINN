"""
Training Module for Physics-Informed Neural Networks.

This module implements the training pipeline for FourierPINN applied to
coronary artery hemodynamics. It handles data loading, model initialization,
loss computation, optimization, and evaluation.

Training Pipeline:
    1. Load patient-specific CFD data (wall surface + streamlines)
    2. Initialize FourierPINN model
    3. Train with combined data and physics losses
    4. Evaluate against CFD ground truth
    5. Generate visualization plots

Loss Components:
    - Data Loss: MSE on velocity and WSS predictions
    - Physics Loss: Navier-Stokes and continuity residuals
    - WSS Physics: Consistency between predicted WSS and velocity gradients

For experimental training methods (TRUE PINN mode), see the experimental/ folder.
"""

import json
import time
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.config import DEVICE, FIGURES_PATH, MODELS_PATH, PATIENT_DATA, RESULTS_PATH
from src.dataset import CollocationSamplerGPU, PatientData, load_patient_data
from src.evaluate import evaluate_model
from src.model import FourierPINN
from src.physics import (
    compute_continuity_residual,
    compute_navier_stokes_residual,
    compute_wss_physics_residual,
)
from src.plots import generate_all_plots
from src.utils import EarlyStopping

# Enable cuDNN autotuner for faster convolutions
torch.backends.cudnn.benchmark = True


# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

# Loss weights used in the published paper
LOSS_WEIGHTS: Dict[str, float] = {
    'wss': 10.0,          # WSS data fitting
    'velocity': 10.0,     # Velocity data fitting
    'navier_stokes': 1.0, # Physics constraint
    'continuity': 1.0,    # Physics constraint
    'wss_physics': 1.0,   # Enforces tau = mu*du/dn consistency
}


# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def train_patient(
    patient_id: str,
    epochs: int = 500,
    batch_size: int = 8192,
    learning_rate: float = 2e-4,
    num_collocation_points: int = 4096,
    patience: int = 100,
    hidden_dim: int = 48,
    num_blocks: int = 6,
    grad_clip: float = 1.0,
    num_frequencies: int = 64,
    fourier_scale: float = 10.0,
    verbose: bool = True
) -> Tuple[nn.Module, Dict]:
    """
    Train FourierPINN model for a specific patient.

    This is the training function used in the published paper.

    Args:
        patient_id: Patient identifier from PATIENT_DATA registry.
        epochs: Maximum number of training epochs (default 500).
        batch_size: Number of samples per training batch (default 8192).
        learning_rate: Initial learning rate for AdamW (default 2e-4).
        num_collocation_points: Collocation points per batch for physics (default 4096).
        patience: Epochs without improvement before early stopping (default 100).
        hidden_dim: Width of hidden layers (default 48).
        num_blocks: Number of ResNet blocks (default 6).
        grad_clip: Maximum gradient norm for clipping (default 1.0).
        num_frequencies: Fourier frequencies (default 64).
        fourier_scale: Fourier scale sigma (default 10.0).
        verbose: Print progress bars and status updates.

    Returns:
        Tuple of (trained_model, results_dict).
    """
    # Setup output folders
    patient_models = MODELS_PATH / patient_id
    patient_figures = FIGURES_PATH / patient_id
    patient_results = RESULTS_PATH / patient_id

    for path in [patient_models, patient_figures, patient_results]:
        path.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"TRAINING FOURIERPINN FOR: {patient_id}")
    print(f"Category: {PATIENT_DATA[patient_id]['category']}")
    print("=" * 80)

    # Load data
    print("\n[LOADING DATA]")
    data, per_vessel = load_patient_data(patient_id)

    # Create dataset with GPU tensors
    print("\n[CREATING DATASET]")
    dataset = PatientData(data, device=DEVICE)
    print(f"  Total points: {len(dataset):,}")
    n_batches = (len(dataset) + batch_size - 1) // batch_size

    print("\n[DEVICE INFO]")
    print(f"  Device: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Create collocation sampler
    collocation_sampler = CollocationSamplerGPU(
        coords=data['X'],
        has_wss=data['has_wss'],
        coord_offset=dataset.coord_offset,
        L_ref=dataset.L_ref,
        device=DEVICE,
        prefer_interior=True
    )

    print("\n[COLLOCATION SAMPLER]")
    print(f"  Wall points: {len(collocation_sampler.wall_indices):,}")
    print(f"  Interior points: {len(collocation_sampler.interior_indices):,}")
    print(f"  Sampling: {num_collocation_points} points/batch")

    # Initialize model
    model = FourierPINN(
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        num_frequencies=num_frequencies,
        fourier_scale=fourier_scale
    ).to(DEVICE)

    print("\n[MODEL]")
    print(f"  Architecture: FourierPINN")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Residual blocks: {num_blocks}")
    print(f"  Fourier frequencies: {num_frequencies}")
    print(f"  Fourier scale: {fourier_scale}")
    print(f"  Parameters: {model.count_parameters():,}")

    # Optimizer with weight decay for regularization
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # Learning rate scheduler: cosine annealing
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=learning_rate * 0.01
    )

    # Get reference scales
    ref_scales = dataset.get_reference_scales()
    L_ref = ref_scales['L_ref']
    U_ref = ref_scales['U_ref']
    Re = ref_scales['Re']
    T_ref = ref_scales['T_ref']
    T_ref_physics = ref_scales['T_ref_physics']
    wss_scale_factor = T_ref_physics / T_ref

    print("\n[NON-DIMENSIONAL SCALES]")
    print(f"  L_ref = {L_ref*1000:.2f} mm")
    print(f"  U_ref = {U_ref:.4f} m/s")
    print(f"  Re = {Re:.1f}")
    print(f"  T_ref (data) = {T_ref:.2f} Pa")

    # Early stopping
    early_stopper = EarlyStopping(patience=patience, monitor='loss')

    # Training history
    history = {
        'train_loss': [], 'wss_loss': [], 'vel_loss': [],
        'ns_loss': [], 'cont_loss': [], 'lr': []
    }
    best_loss = float('inf')

    print(f"\n[TRAINING] {epochs} epochs, patience={patience}")
    print(f"  LR schedule: cosine annealing (min={learning_rate*0.01:.2e})")
    print(f"  Loss weights: WSS={LOSS_WEIGHTS['wss']}, Vel={LOSS_WEIGHTS['velocity']}, "
          f"NS={LOSS_WEIGHTS['navier_stokes']}, Cont={LOSS_WEIGHTS['continuity']}, "
          f"WSS_phy={LOSS_WEIGHTS['wss_physics']}")
    print("-" * 80)

    train_start = time.time()

    for epoch in range(epochs):
        model.train()

        # Resample collocation points for this epoch
        collocation_sampler.resample_for_epoch(epoch, num_collocation_points * n_batches)

        # Shuffle data for this epoch
        dataset.shuffle_for_epoch(epoch)

        epoch_losses = {k: 0.0 for k in ['total', 'wss', 'vel', 'ns', 'cont', 'wss_phy']}

        pbar = tqdm(range(n_batches), desc=f"Epoch {epoch+1:3d}/{epochs}",
                    disable=not verbose, leave=False)

        for batch_idx in pbar:
            # Get batch (data already on GPU)
            batch = dataset.get_batch(batch_idx, batch_size)
            coords = batch['coords']
            wss_true = batch['wss']
            vel_true = batch['velocity']
            normals = batch['normals']
            has_wss = batch['has_wss']

            optimizer.zero_grad(set_to_none=True)

            # Forward pass
            outputs = model(coords)
            vel_pred = torch.cat([outputs['u'], outputs['v'], outputs['w']], dim=1)

            # Compute losses
            losses = _compute_losses(
                outputs, vel_pred, vel_true, wss_true, has_wss,
                coords, normals, model,
                collocation_sampler, num_collocation_points, batch_idx,
                Re, wss_scale_factor
            )

            # Total loss with static weights
            loss_total = (
                LOSS_WEIGHTS['wss'] * losses['wss'] +
                LOSS_WEIGHTS['velocity'] * losses['vel'] +
                LOSS_WEIGHTS['navier_stokes'] * losses['ns'] +
                LOSS_WEIGHTS['continuity'] * losses['cont'] +
                LOSS_WEIGHTS['wss_physics'] * losses['wss_phy']
            )

            # Backward
            loss_total.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            # Accumulate
            epoch_losses['total'] += loss_total.item()
            epoch_losses['wss'] += losses['wss'].item()
            epoch_losses['vel'] += losses['vel'].item()
            epoch_losses['ns'] += losses['ns'].item()
            epoch_losses['cont'] += losses['cont'].item()

            pbar.set_postfix({'Loss': f'{loss_total.item():.4f}'})

        scheduler.step()

        # Average and record
        for k in epoch_losses:
            epoch_losses[k] /= n_batches

        history['train_loss'].append(epoch_losses['total'])
        history['wss_loss'].append(epoch_losses['wss'])
        history['vel_loss'].append(epoch_losses['vel'])
        history['ns_loss'].append(epoch_losses['ns'])
        history['cont_loss'].append(epoch_losses['cont'])
        history['lr'].append(optimizer.param_groups[0]['lr'])

        # Save best
        if epoch_losses['total'] < best_loss:
            best_loss = epoch_losses['total']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': best_loss,
                'config': {
                    'arch': 'fourier',
                    'hidden_dim': hidden_dim,
                    'num_blocks': num_blocks,
                    'num_frequencies': num_frequencies,
                    'fourier_scale': fourier_scale
                }
            }, patient_models / f'pinn_{patient_id}_best.pth')

        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            tqdm.write(
                f"Epoch {epoch+1:3d} | Loss: {epoch_losses['total']:.4f} | "
                f"WSS: {epoch_losses['wss']:.4f} | Vel: {epoch_losses['vel']:.4f} | "
                f"NS: {epoch_losses['ns']:.2e} | Cont: {epoch_losses['cont']:.2e}"
            )

        if early_stopper(epoch_losses['total'], epoch):
            break

    train_time = time.time() - train_start
    epochs_done = len(history['train_loss'])

    print(f"\n[TIMING] {train_time:.1f}s total, {train_time/epochs_done:.2f}s/epoch")

    # Load best model
    ckpt = torch.load(patient_models / f'pinn_{patient_id}_best.pth', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    # Evaluate
    print("\n[EVALUATION]")
    metrics = evaluate_model(model, dataset)

    # Generate plots
    print("\n[GENERATING PLOTS]")
    generate_all_plots(model, dataset, patient_id,
                       patient_figures, metrics, per_vessel, history)

    # Save results
    results = {
        'patient_id': patient_id,
        'metrics': metrics,
        'history': history,
        'timing': {'total': train_time, 'per_epoch': train_time/epochs_done}
    }

    with open(patient_results / f'{patient_id}_history.json', 'w') as f:
        json.dump(history, f)

    print(f"\nResults saved to: {patient_results}")

    return model, results


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _compute_losses(outputs, vel_pred, vel_true, wss_true, has_wss,
                    coords, normals, model,
                    collocation_sampler, num_collocation_points, batch_idx,
                    Re, wss_scale_factor=1.0):
    """Compute all loss components."""
    # Velocity loss
    loss_vel = nn.MSELoss()(vel_pred, vel_true)

    # WSS loss
    if has_wss.any():
        loss_wss = nn.MSELoss()(outputs['wss'][has_wss], wss_true[has_wss])
        wss_res = compute_wss_physics_residual(
            model, coords[has_wss], normals[has_wss], wss_scale_factor
        )
        loss_wss_phy = (wss_res**2).mean()
    else:
        loss_wss = torch.tensor(0.0, device=coords.device)
        loss_wss_phy = torch.tensor(0.0, device=coords.device)

    # Physics loss at collocation points
    if collocation_sampler:
        colloc = collocation_sampler.sample_batch(batch_idx, num_collocation_points)
    else:
        colloc = coords

    f_u, f_v, f_w = compute_navier_stokes_residual(model, colloc, Re)
    cont = compute_continuity_residual(model, colloc)

    loss_ns = (f_u**2 + f_v**2 + f_w**2).mean()
    loss_cont = (cont**2).mean()

    return {
        'wss': loss_wss,
        'vel': loss_vel,
        'ns': loss_ns,
        'cont': loss_cont,
        'wss_phy': loss_wss_phy
    }
