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
from src.dataset import (
    GPUCollocationSampler, GPUDataCache,
    PatientDataset, load_patient_data
)
from src.evaluate import evaluate_model
from src.model import (
    KANPINN, FourierPINN, MultiResNetPINN, PirateNetPINN, VanillaPINN
)
from src.physics import (
    continuity_residual_nondim,
    navier_stokes_residual_nondim,
    wss_physics_residual_nondim,
)
from src.plots import generate_all_plots
from src.utils import EarlyStopping, ReLoBRaLo

# Enable cuDNN autotuner for faster convolutions
torch.backends.cudnn.benchmark = True

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

# Loss weights - BALANCED for this problem:
# - WSS physics constraint is DISABLED (scale mismatch makes it harmful)
# - NS/continuity get moderate weight to enforce physics without hurting data fit
LOSS_WEIGHTS: Dict[str, float] = {
    'wss': 1.0,           # Primary target - WSS prediction
    'velocity': 0.1,      # Supporting data loss
    'navier_stokes': 0.1,   # Light physics regularization
    'continuity': 0.1,      # Light physics regularization
    'wss_physics': 0.0,     # DISABLED - scale mismatch (T_ref_physics/T_ref=0.002) makes it harmful
}


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
    arch: str = 'fourier',
    kan_grid_size: int = 5,
    kan_spline_order: int = 3,
    num_frequencies: int = 64,
    fourier_scale: float = 10.0,
    adaptive_weights: bool = False,
    verbose: bool = True
) -> Tuple[nn.Module, Dict]:
    """
    Train per-patient PINN model.

    Args:
        patient_id: Patient identifier from PATIENT_DATA registry.
        epochs: Maximum number of training epochs.
        batch_size: Number of samples per training batch.
        learning_rate: Initial learning rate for Adam optimizer.
        n_collocation: Number of collocation points per batch for physics.
        patience: Epochs without improvement before early stopping.
        hidden_dim: Width of hidden layers.
        num_blocks: Number of ResNet blocks (or KAN layers).
        grad_clip: Maximum gradient norm for clipping (0 to disable).
        arch: Architecture: 'vanilla', 'fourier', 'pirate', 'multi', 'kan'.
        kan_grid_size: KAN B-spline grid size (for arch='kan').
        kan_spline_order: KAN B-spline order (for arch='kan').
        num_frequencies: Fourier frequencies (for arch='fourier'/'pirate').
        fourier_scale: Fourier scale (for arch='fourier'/'pirate').
        adaptive_weights: Use ReLoBRaLo adaptive loss weighting.
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
    print(f"TRAINING PINN FOR: {patient_id}")
    print(f"Category: {PATIENT_DATA[patient_id]['category']}")
    print("=" * 80)

    # Load data
    print("\n[LOADING DATA]")
    data, per_vessel = load_patient_data(patient_id)

    # Create dataset
    dataset = PatientDataset(data)
    print(f"  Dataset: {len(dataset):,} points")

    # Check if using CUDA
    is_cuda = (DEVICE.type == 'cuda')

    print("\n[GPU OPTIMIZATION]")
    print(f"  Device: {DEVICE}")
    if is_cuda:
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"  GPU Data Cache: enabled")
        print(f"  GPU Collocation: enabled")

    # Create GPU data cache and collocation sampler
    if is_cuda:
        print("\n[GPU DATA CACHE]")
        gpu_data = GPUDataCache(dataset, device=DEVICE)
        n_batches = (len(dataset) + batch_size - 1) // batch_size

        collocation_sampler = GPUCollocationSampler(
            coords=data['X'],
            has_wss=data['has_wss'],
            scaler_X=dataset.scaler_X,
            device=DEVICE,
            prefer_interior=True
        )
    else:
        # CPU fallback
        gpu_data = None
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        n_batches = len(loader)
        collocation_sampler = None

    print("\n[COLLOCATION SAMPLER]")
    if collocation_sampler:
        print(f"  Type: GPU-native")
        print(f"  Wall points: {len(collocation_sampler.wall_indices):,}")
        print(f"  Interior points: {len(collocation_sampler.interior_indices):,}")
    print(f"  Sampling: {n_collocation} points/batch")

    # Initialize model
    model, arch_name = _create_model(
        arch=arch,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        num_frequencies=num_frequencies,
        fourier_scale=fourier_scale,
        kan_grid_size=kan_grid_size,
        kan_spline_order=kan_spline_order
    )

    print("\n[MODEL]")
    print(f"  Architecture: {arch_name}")
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

    coord_scale = torch.tensor(
        dataset.scaler_X.data_range_,
        dtype=torch.float32,
        device=DEVICE
    ).view(1, 3)

    print("\n[NON-DIMENSIONAL SCALES]")
    print(f"  L_ref = {L_ref*1000:.2f} mm")
    print(f"  U_ref = {U_ref:.4f} m/s")
    print(f"  Re = {Re:.1f}")
    print(f"  T_ref (WSS) = {T_ref:.4f} Pa")

    # Early stopping
    early_stopper = EarlyStopping(patience=patience, monitor='loss')

    # Adaptive weights (optional)
    # Order: [wss, vel, ns, cont, wss_phy]
    loss_balancer = None
    if adaptive_weights:
        loss_balancer = ReLoBRaLo(
            num_losses=5, alpha=0.999, T=1.0, rho=0.99,
            # Ensure physics losses don't get ignored (they start small)
            min_weights=[0.5, 0.1, 1.0, 1.0, 1.0],  # [wss, vel, ns, cont, wss_phy]
            max_weights=[5.0, 2.0, 20.0, 20.0, 50.0]  # Allow physics to grow
        )
        print("\n[ADAPTIVE WEIGHTING] ReLoBRaLo enabled")
        print("  Min weights: [wss=0.5, vel=0.1, ns=1.0, cont=1.0, wss_phy=1.0]")

    # Training history
    history = {
        'train_loss': [], 'wss_loss': [], 'vel_loss': [],
        'ns_loss': [], 'cont_loss': [], 'lr': []
    }
    best_loss = float('inf')

    print(f"\n[TRAINING] {epochs} epochs, patience={patience}")
    print(f"  LR schedule: cosine annealing (min={learning_rate*0.01:.2e})")
    if not adaptive_weights:
        print(f"  Loss weights: WSS={LOSS_WEIGHTS['wss']}, Vel={LOSS_WEIGHTS['velocity']}, "
              f"NS={LOSS_WEIGHTS['navier_stokes']}, Cont={LOSS_WEIGHTS['continuity']}, "
              f"WSS_phy={LOSS_WEIGHTS['wss_physics']}")
    print("-" * 80)

    train_start = time.time()

    for epoch in range(epochs):
        model.train()

        # Resample collocation points for this epoch
        if collocation_sampler:
            collocation_sampler.resample_for_epoch(epoch, n_collocation * n_batches)

        # Shuffle data
        if gpu_data:
            gpu_data.shuffle_for_epoch(epoch)

        epoch_losses = {k: 0.0 for k in ['total', 'wss', 'vel', 'ns', 'cont', 'wss_phy']}

        pbar = tqdm(range(n_batches), desc=f"Epoch {epoch+1:3d}/{epochs}",
                    disable=not verbose, leave=False)

        for batch_idx in pbar:
            # Get batch
            if gpu_data:
                batch = gpu_data.get_batch(batch_idx, batch_size)
                coords = batch['coords']
                wss_true = batch['wss']
                vel_true = batch['velocity']
                normals = batch['normals']
                has_wss = batch['has_wss']
            else:
                batch = next(iter(loader))
                coords = batch['coords'].to(DEVICE)
                wss_true = batch['wss'].to(DEVICE)
                vel_true = batch['velocity'].to(DEVICE)
                normals = batch['normals'].to(DEVICE)
                has_wss = batch['has_wss'].to(DEVICE).squeeze()

            optimizer.zero_grad(set_to_none=True)

            # Forward pass
            outputs = model(coords)
            vel_pred = torch.cat([outputs['u'], outputs['v'], outputs['w']], dim=1)

            # Compute losses
            losses = _compute_losses(
                outputs, vel_pred, vel_true, wss_true, has_wss,
                coords, normals, coord_scale, model,
                collocation_sampler, n_collocation, batch_idx,
                L_ref, Re, T_ref, T_ref_physics
            )

            # Total loss
            if loss_balancer:
                loss_list = [losses['wss'], losses['vel'], losses['ns'],
                             losses['cont'], losses['wss_phy']]
                w = loss_balancer.update(loss_list)
                loss_total = sum(w[i] * loss_list[i] for i in range(5))
            else:
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
                'config': {'arch': arch, 'hidden_dim': hidden_dim, 'num_blocks': num_blocks}
            }, patient_models / f'pinn_{patient_id}_best.pth')

        # Print progress with all loss components
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
    if gpu_data:
        # Create a simple loader for evaluation
        eval_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    else:
        eval_loader = loader
    metrics = evaluate_model(model, eval_loader, dataset, coord_scale)

    # Generate plots
    print("\n[GENERATING PLOTS]")
    generate_all_plots(model, eval_loader, dataset, patient_id,
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

def _create_model(arch, hidden_dim, num_blocks, num_frequencies,
                  fourier_scale, kan_grid_size, kan_spline_order):
    """Create PINN model based on architecture choice."""
    if arch == 'vanilla':
        model = VanillaPINN(hidden_dim=hidden_dim, num_blocks=num_blocks).to(DEVICE)
        name = 'VanillaPINN'
    elif arch == 'fourier':
        model = FourierPINN(hidden_dim=hidden_dim, num_blocks=num_blocks,
                           num_frequencies=num_frequencies, fourier_scale=fourier_scale).to(DEVICE)
        name = 'FourierPINN'
    elif arch == 'pirate':
        model = PirateNetPINN(hidden_dim=hidden_dim, num_blocks=num_blocks,
                             num_frequencies=num_frequencies, fourier_scale=fourier_scale).to(DEVICE)
        name = 'PirateNetPINN'
    elif arch == 'kan':
        model = KANPINN(hidden_dim=hidden_dim, num_layers=num_blocks,
                       grid_size=kan_grid_size, spline_order=kan_spline_order).to(DEVICE)
        name = 'KANPINN'
    else:  # multi
        model = MultiResNetPINN(hidden_dim=hidden_dim, num_blocks=num_blocks).to(DEVICE)
        name = 'MultiResNetPINN'
    return model, name


def _compute_losses(outputs, vel_pred, vel_true, wss_true, has_wss,
                    coords, normals, coord_scale, model,
                    collocation_sampler, n_collocation, batch_idx,
                    L_ref, Re, T_ref, T_ref_physics):
    """Compute all loss components."""
    # Velocity loss
    loss_vel = nn.MSELoss()(vel_pred, vel_true)

    # WSS loss
    if has_wss.any():
        loss_wss = nn.MSELoss()(outputs['wss'][has_wss], wss_true[has_wss])
        wss_res = wss_physics_residual_nondim(
            model, coords[has_wss], normals[has_wss], coord_scale,
            L_ref, T_ref, T_ref_physics
        )
        loss_wss_phy = (wss_res**2).mean()
    else:
        loss_wss = torch.tensor(0.0, device=coords.device)
        loss_wss_phy = torch.tensor(0.0, device=coords.device)

    # Physics loss at collocation points
    if collocation_sampler:
        colloc = collocation_sampler.sample_batch(batch_idx, n_collocation)
    else:
        colloc = coords  # Fallback: use data points

    f_u, f_v, f_w = navier_stokes_residual_nondim(model, colloc, coord_scale, L_ref, Re)
    cont = continuity_residual_nondim(model, colloc, coord_scale, L_ref)

    loss_ns = (f_u**2 + f_v**2 + f_w**2).mean()
    loss_cont = (cont**2).mean()

    return {
        'wss': loss_wss,
        'vel': loss_vel,
        'ns': loss_ns,
        'cont': loss_cont,
        'wss_phy': loss_wss_phy
    }
