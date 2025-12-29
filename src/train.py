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
    PatientDataset, load_patient_data,
    sample_sparse_data_indices
)
from src.evaluate import evaluate_model
from src.model import (
    KANPINN, FourierPINN, MultiResNetPINN, PirateNetPINN, VanillaPINN
)
from src.physics import (
    compute_wss_from_gradients,
    continuity_residual_nondim,
    navier_stokes_residual_nondim,
    wss_physics_residual_nondim,
)
from src.plots import generate_all_plots
from src.utils import EarlyStopping, ReLoBRaLo

# Enable cuDNN autotuner for faster convolutions
torch.backends.cudnn.benchmark = True


# =============================================================================
# DERIVED WSS EVALUATION
# =============================================================================

def evaluate_model_derive_wss(
    model: nn.Module,
    dataset,
    coord_scale: torch.Tensor,
    L_ref: float,
    T_ref_physics: float,
    T_ref: float,
    batch_size: int = 4096
) -> Dict:
    """
    Evaluate model with WSS derived from velocity gradients.

    This implements the physics-based WSS computation:
        tau = mu * |du_tangential/dn|

    Used when derive_wss=True to match the example code approach where
    WSS is not directly predicted but computed from the velocity field.

    Args:
        model: Trained PINN model.
        dataset: PatientDataset with wall points.
        coord_scale: Coordinate scaling tensor.
        L_ref: Reference length scale.
        T_ref_physics: Physics-based WSS scale (mu * U_ref / L_ref).
        T_ref: Data-driven WSS scale for output.
        batch_size: Batch size for evaluation.

    Returns:
        Dictionary with RMSE, MAE, NRMSE, R2 metrics.
    """
    import numpy as np
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

    model.eval()

    # Get wall points (where we have ground truth WSS)
    wall_mask = dataset.has_wss
    wall_indices = np.where(wall_mask)[0]

    # Get wall data
    wall_coords = torch.tensor(
        dataset.X_scaled[wall_indices],
        dtype=torch.float32, device=DEVICE
    )
    wall_normals = torch.tensor(
        dataset.normals[wall_indices],
        dtype=torch.float32, device=DEVICE
    )
    wall_wss_true = dataset.y[wall_indices]  # In Pa (physical units)

    # Compute WSS from velocity gradients in batches
    # NOTE: Cannot use torch.no_grad() here because we need gradients
    # to compute du/dn for WSS derivation
    wss_computed_list = []
    n_wall = len(wall_indices)

    for i in range(0, n_wall, batch_size):
        end_idx = min(i + batch_size, n_wall)
        batch_coords = wall_coords[i:end_idx].clone()
        batch_normals = wall_normals[i:end_idx]

        # Need gradients for WSS computation
        batch_coords.requires_grad_(True)

        # Compute WSS from velocity gradients
        wss_batch = compute_wss_from_gradients(
            model, batch_coords, batch_normals, coord_scale,
            L_ref, T_ref_physics, T_ref
        )
        wss_computed_list.append(wss_batch.detach())

    # Concatenate and convert to physical units
    wss_computed = torch.cat(wss_computed_list, dim=0).cpu().numpy().flatten()
    wss_computed_pa = wss_computed * T_ref  # Convert from non-dim to Pa

    # Compute metrics
    valid_mask = ~np.isnan(wall_wss_true) & ~np.isnan(wss_computed_pa)
    wss_true_valid = wall_wss_true[valid_mask]
    wss_pred_valid = wss_computed_pa[valid_mask]

    rmse = np.sqrt(mean_squared_error(wss_true_valid, wss_pred_valid))
    mae = mean_absolute_error(wss_true_valid, wss_pred_valid)
    r2 = r2_score(wss_true_valid, wss_pred_valid)

    wss_range = wss_true_valid.max() - wss_true_valid.min()
    nrmse = rmse / wss_range if wss_range > 0 else 0.0

    metrics = {
        'RMSE': float(rmse),
        'MAE': float(mae),
        'NRMSE': float(nrmse),
        'R2': float(r2),
        'wss_method': 'derived_from_velocity'
    }

    print(f"\n  [DERIVED WSS METRICS]")
    print(f"    RMSE:  {rmse:.4f} Pa")
    print(f"    MAE:   {mae:.4f} Pa")
    print(f"    NRMSE: {nrmse*100:.2f}%")
    print(f"    R²:    {r2:.4f}")

    return metrics


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


# =============================================================================
# TRUE PINN TRAINING (SPARSE DATA + STRONG PHYSICS)
# =============================================================================

# Loss weights for TRUE PINN mode (following the example code):
# - Physics (Navier-Stokes) evaluated on ALL collocation points (weight = 1.0)
# - BC (no-slip at walls) enforced strongly (Lambda_BC = 20.0)
# - Data (sparse measurements) fitted strongly (Lambda_data = 20.0)
TRUE_PINN_WEIGHTS: Dict[str, float] = {
    'navier_stokes': 1.0,      # Physics - primary driver
    'continuity': 1.0,         # Physics - mass conservation
    'bc_noslip': 20.0,         # Boundary condition - no slip at walls
    'data_velocity': 20.0,     # Sparse velocity data
    'data_wss': 20.0,          # Sparse WSS data
}


def train_patient_true_pinn(
    patient_id: str,
    epochs: int = 5000,
    batch_size: int = 512,
    learning_rate: float = 5e-4,
    n_collocation: int = 4096,
    patience: int = 500,
    hidden_dim: int = 200,
    num_blocks: int = 8,
    grad_clip: float = 1.0,
    arch: str = 'vanilla',
    sample_every_n: int = 200,
    lr_step_size: int = 800,
    lr_decay: float = 0.5,
    num_frequencies: int = 64,
    fourier_scale: float = 10.0,
    derive_wss: bool = False,
    verbose: bool = True
) -> Tuple[nn.Module, Dict]:
    """
    Train PINN using TRUE PINN paradigm: sparse data + strong physics.

    This follows the classic PINN approach where:
    - Physics (Navier-Stokes) is the PRIMARY learning signal
    - Only SPARSE data points are used for data fitting
    - Boundary conditions (no-slip) are strongly enforced
    - The model learns to satisfy physics everywhere, not just fit data

    Key differences from train_patient():
    - Sparse data sampling (every Nth point)
    - Higher physics/BC/data weights (Lambda = 20)
    - Step LR decay (not cosine)
    - Longer training (5000+ epochs)
    - No-slip BC loss at wall points

    Args:
        patient_id: Patient identifier from PATIENT_DATA registry.
        epochs: Maximum training epochs (default 5000).
        batch_size: Batch size for collocation points (default 512).
        learning_rate: Starting learning rate (default 5e-4).
        n_collocation: Collocation points per batch for physics (default 4096).
        patience: Early stopping patience (default 500).
        hidden_dim: Hidden layer width (default 200, like example).
        num_blocks: Number of hidden layers (default 8, like example).
        grad_clip: Gradient clipping norm (default 1.0).
        arch: Architecture choice (default 'vanilla' like example).
        sample_every_n: Sample every Nth point for sparse data (default 200).
        lr_step_size: Epochs between LR decay (default 800).
        lr_decay: LR decay factor (default 0.5).
        num_frequencies: Fourier frequencies for FourierPINN (default 64).
        fourier_scale: Fourier scale for FourierPINN (default 10.0).
        derive_wss: If True, derive WSS from velocity gradients (like example code)
            instead of using direct WSS prediction. Only velocity data is used
            for training, and WSS is computed post-hoc from tau = mu * |du/dn|.
        verbose: Print progress.

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
    print(f"TRUE PINN TRAINING FOR: {patient_id}")
    print(f"Category: {PATIENT_DATA[patient_id]['category']}")
    print("=" * 80)
    if derive_wss:
        print("\n>>> TRUE PINN MODE: Sparse Data + Strong Physics <<<")
        print(">>> WSS MODE: Derived from velocity gradients (tau = mu * |du/dn|) <<<")
    else:
        print("\n>>> TRUE PINN MODE: Sparse Data + Strong Physics <<<")
        print(">>> WSS MODE: Direct prediction (network output) <<<")

    # Load ALL data (for physics evaluation on full domain)
    print("\n[LOADING DATA]")
    data, per_vessel = load_patient_data(patient_id)

    # Create full dataset (for scaling and evaluation)
    full_dataset = PatientDataset(data)
    print(f"  Full dataset: {len(full_dataset):,} points")

    # Sample SPARSE data indices for data loss
    sparse_wall_idx, sparse_interior_idx = sample_sparse_data_indices(
        data, sample_every_n=sample_every_n
    )

    # Pre-compute sparse data tensors (moved to GPU once)
    sparse_wall_coords = torch.tensor(
        full_dataset.scaler_X.transform(data['X'][sparse_wall_idx]),
        dtype=torch.float32, device=DEVICE
    )
    sparse_wall_wss = torch.tensor(
        data['y'][sparse_wall_idx] / full_dataset.T_ref,
        dtype=torch.float32, device=DEVICE
    ).view(-1, 1)

    sparse_interior_coords = torch.tensor(
        full_dataset.scaler_X.transform(data['X'][sparse_interior_idx]),
        dtype=torch.float32, device=DEVICE
    ) if len(sparse_interior_idx) > 0 else None
    sparse_interior_vel = torch.tensor(
        data['velocity'][sparse_interior_idx] / full_dataset.U_ref,
        dtype=torch.float32, device=DEVICE
    ) if len(sparse_interior_idx) > 0 else None

    # Wall points for BC (ALL wall points, not just sparse)
    wall_indices = data['has_wss']
    all_wall_coords = torch.tensor(
        full_dataset.scaler_X.transform(data['X'][wall_indices]),
        dtype=torch.float32, device=DEVICE
    )

    # Create collocation sampler for physics (on FULL mesh)
    is_cuda = (DEVICE.type == 'cuda')
    print("\n[GPU OPTIMIZATION]")
    print(f"  Device: {DEVICE}")
    if is_cuda:
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    collocation_sampler = GPUCollocationSampler(
        coords=data['X'],
        has_wss=data['has_wss'],
        scaler_X=full_dataset.scaler_X,
        device=DEVICE,
        prefer_interior=True
    )

    print("\n[COLLOCATION SAMPLER]")
    print(f"  All mesh points: {len(data['X']):,} (for physics)")
    print(f"  Collocation per batch: {n_collocation}")

    # Initialize model (vanilla MLP like example, or Fourier)
    model, arch_name = _create_model(
        arch=arch,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        num_frequencies=num_frequencies,
        fourier_scale=fourier_scale,
        kan_grid_size=5,
        kan_spline_order=3
    )

    print("\n[MODEL]")
    print(f"  Architecture: {arch_name}")
    print(f"  Parameters: {model.count_parameters():,}")

    # Optimizer (AdamW like example)
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.99),
        eps=1e-15
    )

    # Step LR scheduler (like example code)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=lr_step_size, gamma=lr_decay
    )

    # Get reference scales
    ref_scales = full_dataset.get_reference_scales()
    L_ref = ref_scales['L_ref']
    U_ref = ref_scales['U_ref']
    Re = ref_scales['Re']
    T_ref = ref_scales['T_ref']

    coord_scale = torch.tensor(
        full_dataset.scaler_X.data_range_,
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

    # Training history
    history = {
        'train_loss': [], 'physics_loss': [], 'bc_loss': [],
        'data_loss': [], 'lr': []
    }
    best_loss = float('inf')

    # Number of batches per epoch (based on collocation points)
    total_collocation = len(data['X'])
    n_batches = max((total_collocation + n_collocation - 1) // n_collocation, 1)

    print(f"\n[TRAINING] {epochs} epochs, patience={patience}")
    print(f"  LR schedule: step decay (step={lr_step_size}, gamma={lr_decay})")
    print(f"  Batches per epoch: {n_batches}")
    print(f"  Loss weights:")
    print(f"    Physics (NS): {TRUE_PINN_WEIGHTS['navier_stokes']}")
    print(f"    Physics (Cont): {TRUE_PINN_WEIGHTS['continuity']}")
    print(f"    BC (no-slip): {TRUE_PINN_WEIGHTS['bc_noslip']}")
    print(f"    Data (velocity): {TRUE_PINN_WEIGHTS['data_velocity']}")
    if derive_wss:
        print(f"    Data (WSS): 0.0 (derived from velocity gradients)")
    else:
        print(f"    Data (WSS): {TRUE_PINN_WEIGHTS['data_wss']}")
    print("-" * 80)

    train_start = time.time()

    for epoch in range(epochs):
        model.train()

        # Resample collocation points for this epoch
        collocation_sampler.resample_for_epoch(epoch, n_collocation * n_batches)

        epoch_losses = {k: 0.0 for k in ['total', 'physics', 'bc', 'data']}

        pbar = tqdm(range(n_batches), desc=f"Epoch {epoch+1:3d}/{epochs}",
                    disable=not verbose, leave=False)

        for batch_idx in pbar:
            optimizer.zero_grad(set_to_none=True)

            # =====================================================================
            # 1. PHYSICS LOSS: Navier-Stokes on collocation points
            # =====================================================================
            colloc = collocation_sampler.sample_batch(batch_idx, n_collocation)

            f_u, f_v, f_w = navier_stokes_residual_nondim(
                model, colloc, coord_scale, L_ref, Re
            )
            cont = continuity_residual_nondim(model, colloc, coord_scale, L_ref)

            loss_ns = (f_u**2 + f_v**2 + f_w**2).mean()
            loss_cont = (cont**2).mean()

            loss_physics = (
                TRUE_PINN_WEIGHTS['navier_stokes'] * loss_ns +
                TRUE_PINN_WEIGHTS['continuity'] * loss_cont
            )

            # =====================================================================
            # 2. BC LOSS: No-slip at wall (u=v=w=0)
            # =====================================================================
            # Sample a batch of wall points for BC
            n_bc = min(batch_size, len(all_wall_coords))
            bc_idx = torch.randint(0, len(all_wall_coords), (n_bc,), device=DEVICE)
            bc_coords = all_wall_coords[bc_idx]

            bc_outputs = model(bc_coords)
            bc_u = bc_outputs['u']
            bc_v = bc_outputs['v']
            bc_w = bc_outputs['w']

            # No-slip: u=v=w=0 at walls
            loss_bc = (
                (bc_u**2).mean() +
                (bc_v**2).mean() +
                (bc_w**2).mean()
            )
            loss_bc = TRUE_PINN_WEIGHTS['bc_noslip'] * loss_bc

            # =====================================================================
            # 3. DATA LOSS: Sparse measurements
            # =====================================================================
            if derive_wss:
                # DERIVE WSS MODE: Only use velocity data, no direct WSS fitting
                # WSS will be computed from velocity gradients during evaluation
                loss_data_wss = torch.tensor(0.0, device=DEVICE)
            else:
                # DIRECT WSS MODE: Use WSS data directly
                wss_outputs = model(sparse_wall_coords)
                loss_data_wss = nn.MSELoss()(wss_outputs['wss'], sparse_wall_wss)

            # Velocity data (sparse interior points)
            if sparse_interior_coords is not None and len(sparse_interior_coords) > 0:
                vel_outputs = model(sparse_interior_coords)
                vel_pred = torch.cat([vel_outputs['u'], vel_outputs['v'], vel_outputs['w']], dim=1)
                loss_data_vel = nn.MSELoss()(vel_pred, sparse_interior_vel)
            else:
                loss_data_vel = torch.tensor(0.0, device=DEVICE)

            loss_data = (
                TRUE_PINN_WEIGHTS['data_wss'] * loss_data_wss +
                TRUE_PINN_WEIGHTS['data_velocity'] * loss_data_vel
            )

            # =====================================================================
            # Total loss
            # =====================================================================
            loss_total = loss_physics + loss_bc + loss_data

            # Backward
            loss_total.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            # Accumulate
            epoch_losses['total'] += loss_total.item()
            epoch_losses['physics'] += loss_physics.item()
            epoch_losses['bc'] += loss_bc.item()
            epoch_losses['data'] += loss_data.item()

            pbar.set_postfix({'Loss': f'{loss_total.item():.4f}'})

        scheduler.step()

        # Average and record
        for k in epoch_losses:
            epoch_losses[k] /= n_batches

        history['train_loss'].append(epoch_losses['total'])
        history['physics_loss'].append(epoch_losses['physics'])
        history['bc_loss'].append(epoch_losses['bc'])
        history['data_loss'].append(epoch_losses['data'])
        history['lr'].append(optimizer.param_groups[0]['lr'])

        # Save best
        if epoch_losses['total'] < best_loss:
            best_loss = epoch_losses['total']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': best_loss,
                'config': {
                    'arch': arch, 'hidden_dim': hidden_dim,
                    'num_blocks': num_blocks, 'mode': 'true_pinn',
                    'sample_every_n': sample_every_n,
                    'derive_wss': derive_wss
                }
            }, patient_models / f'pinn_{patient_id}_true_pinn_best.pth')

        # Print progress
        if (epoch + 1) % 50 == 0 or epoch == 0:
            tqdm.write(
                f"Epoch {epoch+1:4d} | Loss: {epoch_losses['total']:.4f} | "
                f"Physics: {epoch_losses['physics']:.4f} | BC: {epoch_losses['bc']:.4f} | "
                f"Data: {epoch_losses['data']:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}"
            )

        if early_stopper(epoch_losses['total'], epoch):
            break

    train_time = time.time() - train_start
    epochs_done = len(history['train_loss'])

    print(f"\n[TIMING] {train_time:.1f}s total, {train_time/epochs_done:.2f}s/epoch")

    # Load best model
    ckpt = torch.load(patient_models / f'pinn_{patient_id}_true_pinn_best.pth', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    # Evaluate on FULL dataset (the true test - physics should fill in the gaps)
    print("\n[EVALUATION ON FULL DATASET]")
    print("  (Model trained on sparse data, evaluated on ALL points)")
    eval_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)

    if derive_wss:
        # DERIVE WSS MODE: Compute WSS from velocity gradients
        print("  WSS computed from velocity gradients: tau = mu * |du/dn|")
        metrics = evaluate_model_derive_wss(
            model, full_dataset, coord_scale,
            L_ref, full_dataset.T_ref_physics, T_ref,
            batch_size=batch_size
        )
    else:
        # DIRECT WSS MODE: Use model's WSS output
        metrics = evaluate_model(model, eval_loader, full_dataset, coord_scale)

    # Generate plots
    print("\n[GENERATING PLOTS]")
    generate_all_plots(model, eval_loader, full_dataset, patient_id,
                       patient_figures, metrics, per_vessel, history)

    # Save results
    results = {
        'patient_id': patient_id,
        'mode': 'true_pinn',
        'derive_wss': derive_wss,
        'sparse_ratio': 1.0 / sample_every_n,
        'metrics': metrics,
        'history': history,
        'timing': {'total': train_time, 'per_epoch': train_time/epochs_done}
    }

    with open(patient_results / f'{patient_id}_true_pinn_history.json', 'w') as f:
        json.dump(history, f)

    print(f"\nResults saved to: {patient_results}")

    return model, results
