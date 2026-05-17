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
"""

import json
import time
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src import config as _config
from src.config import DEVICE, FIGURES_PATH, MODELS_PATH, PATIENT_DATA, RESULTS_PATH
from src.dataset import CollocationSamplerGPU, PatientData, load_patient_data
from src.evaluate import evaluate_model
from src.model import FourierPINN
from src.physics import (
    compute_physics_residuals_fused,
    compute_wss_physics_residual,
)
from src.plots import generate_all_plots
from src.utils import EarlyStopping

# Enable cuDNN autotuner for faster convolutions
torch.backends.cudnn.benchmark = True


# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

# Clinical-priority factors that scale the gradient-norm-balanced loss
# weights computed once per patient by `compute_gradnorm_balanced_weights`.
# All 1.0 = pure GradNorm balancing; >1.0 boosts a term's effective weight.
LOSS_PRIORITY: Dict[str, float] = {
    'wss': 2.0,           # clinical target -- twice the gradient pull of others
    'velocity': 1.0,
    'navier_stokes': 1.0,
    'continuity': 1.0,
    'wss_physics': 1.0,
}


def compute_gradnorm_balanced_weights(
    model: nn.Module,
    init_batch: Dict[str, torch.Tensor],
    collocation_sampler,
    num_collocation_points: int,
    Re: float,
    wss_scale_factor: float,
    rheology: str,
    cy_params: Dict[str, float],
    U_ref: float,
    L_ref: float,
    priorities: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Per-term gradient-norm balancing (one-shot at training start).

    Computes the gradient norm of each loss term on a single batch, then sets
    the loss weight inversely proportional to that norm so each weighted term
    contributes equal magnitude to the parameter gradient. A relative
    priority dict (default LOSS_PRIORITY) lets the WSS data term carry more
    weight than the rest; physics terms remain balanced relative to each
    other. Returns a dict matching the LOSS_PRIORITY schema.
    """
    if priorities is None:
        priorities = LOSS_PRIORITY

    coords = init_batch['coords']
    wss_true = init_batch['wss']
    vel_true = init_batch['velocity']
    normals = init_batch['normals']
    has_wss = init_batch['has_wss']

    outputs = model(coords)
    vel_pred = torch.cat([outputs['u'], outputs['v'], outputs['w']], dim=1)

    losses = _compute_losses(
        outputs, vel_pred, vel_true, wss_true, has_wss,
        coords, normals, model,
        collocation_sampler, num_collocation_points, batch_idx=0,
        Re=Re, wss_scale_factor=wss_scale_factor,
        rheology=rheology, cy_params=cy_params,
        U_ref=U_ref, L_ref=L_ref,
    )

    # Map _compute_losses keys -> LOSS_PRIORITY keys.
    name_map = {
        'wss': 'wss', 'vel': 'velocity', 'ns': 'navier_stokes',
        'cont': 'continuity', 'wss_phy': 'wss_physics',
    }

    grad_norms: Dict[str, float] = {}
    params = list(model.parameters())
    for short_name, weight_key in name_map.items():
        loss = losses[short_name]
        if loss.detach().item() == 0.0:
            grad_norms[weight_key] = 1.0
            continue
        grads = torch.autograd.grad(
            loss, params, retain_graph=True, allow_unused=True,
        )
        sq = sum((g.detach() ** 2).sum().item() for g in grads if g is not None)
        grad_norms[weight_key] = max(sq ** 0.5, 1e-10)

    # Each weighted term should produce gradient magnitude proportional to
    # its priority. Normalise so the average target equals the average
    # observed gradient norm (keeps the overall update magnitude similar to
    # an unweighted run).
    target_avg = sum(grad_norms.values()) / len(grad_norms)
    weights = {
        key: priorities[key] * target_avg / grad_norms[key]
        for key in grad_norms
    }
    return weights


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
    holdout_fraction: float = 0.0,
    holdout_seed: int = 0,
    verbose: bool = True,
    output_tag: Optional[str] = None
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
        output_tag: Optional sub-namespace inserted between the rheology and
            patient folders for models/figures/results. Used by the sensitivity
            sweep (``output_tag='_sensitivity'``) so its many short re-trainings
            of a patient land in their own folder and never overwrite the
            holdout run's per-patient figures/timing. ``None`` (default)
            preserves the original ``<base>/<rheology>/<patient>`` layout used
            by the paper holdout sweep.

    Returns:
        Tuple of (trained_model, results_dict).
    """
    # Setup output folders. Models/figures/results are namespaced by rheology
    # so a Newtonian and a Carreau-Yasuda run on the SAME patient land in
    # separate subtrees and never overwrite each other. An optional output_tag
    # adds a further sub-namespace (e.g. '_sensitivity') so auxiliary sweeps do
    # not clobber the holdout run's per-patient artifacts.
    rheology_tag = _config.RHEOLOGY  # "newtonian" or "carreau_yasuda"
    if output_tag:
        patient_models = MODELS_PATH / rheology_tag / output_tag / patient_id
        patient_figures = FIGURES_PATH / rheology_tag / output_tag / patient_id
        patient_results = RESULTS_PATH / rheology_tag / output_tag / patient_id
    else:
        patient_models = MODELS_PATH / rheology_tag / patient_id
        patient_figures = FIGURES_PATH / rheology_tag / patient_id
        patient_results = RESULTS_PATH / rheology_tag / patient_id

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
    dataset = PatientData(
        data, device=DEVICE,
        holdout_fraction=holdout_fraction,
        holdout_seed=holdout_seed,
    )
    print(f"  Total points: {len(dataset):,}")
    if dataset.num_holdout > 0:
        print(
            f"  Spatial holdout: {dataset.num_train:,} train / "
            f"{dataset.num_holdout:,} held-out "
            f"({100*dataset.holdout_fraction:.0f}%, seed={dataset.holdout_seed})"
        )
    n_batches = (dataset.num_train + batch_size - 1) // batch_size

    print("\n[DEVICE INFO]")
    print(f"  Device: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        # GPU efficiency: enable tensor-core FP32 matmul (TF32) and cuDNN
        # autotuning. Lossless within ~1e-3 for inference and a no-op when
        # CUDA is unavailable.
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')

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

    # Gradient-norm balanced loss weights (one-shot at training start). Each
    # weight is set inversely proportional to the term's gradient norm on a
    # representative first batch, then scaled by LOSS_PRIORITY so the WSS data
    # term carries more pull than the rest. This replaces the previous static
    # 10:1 hand-picked weights and is the principled answer to R2-8.
    dataset.shuffle_for_epoch(0)
    collocation_sampler.resample_for_epoch(0, num_collocation_points * n_batches)
    init_batch = dataset.get_batch(0, batch_size)
    loss_weights = compute_gradnorm_balanced_weights(
        model, init_batch, collocation_sampler, num_collocation_points,
        Re=Re, wss_scale_factor=wss_scale_factor,
        rheology=_config.RHEOLOGY, cy_params=_config.CY_PARAMS,
        U_ref=U_ref, L_ref=L_ref,
    )

    print(f"\n[TRAINING] {epochs} epochs, patience={patience}")
    print(f"  LR schedule: cosine annealing (min={learning_rate*0.01:.2e})")
    print(
        "  Loss weights (gradnorm-balanced): "
        f"WSS={loss_weights['wss']:.3g}, Vel={loss_weights['velocity']:.3g}, "
        f"NS={loss_weights['navier_stokes']:.3g}, Cont={loss_weights['continuity']:.3g}, "
        f"WSS_phy={loss_weights['wss_physics']:.3g}"
    )
    print("-" * 80)

    train_start = time.time()

    for epoch in range(epochs):
        model.train()

        # Resample collocation points for this epoch
        collocation_sampler.resample_for_epoch(epoch, num_collocation_points * n_batches)

        # Shuffle data for this epoch
        dataset.shuffle_for_epoch(epoch)

        # Accumulate on-device to avoid forced GPU<->CPU syncs in the hot loop.
        loss_keys = ['total', 'wss', 'vel', 'ns', 'cont', 'wss_phy']
        epoch_loss_dev = {
            k: torch.zeros(1, device=DEVICE) for k in loss_keys
        }

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
                Re, wss_scale_factor,
                rheology=_config.RHEOLOGY, cy_params=_config.CY_PARAMS,
                U_ref=U_ref, L_ref=L_ref,
            )

            loss_total = (
                loss_weights['wss'] * losses['wss'] +
                loss_weights['velocity'] * losses['vel'] +
                loss_weights['navier_stokes'] * losses['ns'] +
                loss_weights['continuity'] * losses['cont'] +
                loss_weights['wss_physics'] * losses['wss_phy']
            )

            # Backward
            loss_total.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            # Accumulate ON DEVICE (detach to drop the autograd graph) -- a
            # single CPU sync per epoch instead of 5 per batch.
            with torch.no_grad():
                epoch_loss_dev['total'] += loss_total.detach()
                epoch_loss_dev['wss'] += losses['wss'].detach()
                epoch_loss_dev['vel'] += losses['vel'].detach()
                epoch_loss_dev['ns'] += losses['ns'].detach()
                epoch_loss_dev['cont'] += losses['cont'].detach()
                epoch_loss_dev['wss_phy'] += losses['wss_phy'].detach()

        scheduler.step()

        # Single sync per epoch.
        epoch_losses = {
            k: float(v.item() / n_batches) for k, v in epoch_loss_dev.items()
        }

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

    # Evaluate. With a non-zero holdout fraction we report metrics on the
    # training subset and the held-out subset separately so reviewers can
    # distinguish interpolation from prediction (Physics of Fluids R1-5/R2-6).
    print("\n[EVALUATION]")
    if dataset.num_holdout > 0:
        metrics_train = evaluate_model(model, dataset, split="train")
        metrics_holdout = evaluate_model(model, dataset, split="holdout")
        metrics = dict(metrics_holdout)  # primary metrics = held-out (predictive)
        metrics['train'] = metrics_train
        metrics['holdout'] = metrics_holdout
    else:
        metrics = evaluate_model(model, dataset, split="all")

    print("\n[GENERATING PLOTS]")
    generate_all_plots(model, dataset, patient_id,
                       patient_figures, metrics, per_vessel, history)

    # Benchmark full-field inference time (Physics of Fluids R1-7).
    inf_start = time.perf_counter()
    with torch.no_grad():
        all_coords = dataset.coords
        n_inf = all_coords.shape[0]
        bs_inf = 16384
        for s in range(0, n_inf, bs_inf):
            _ = model(all_coords[s:s + bs_inf])
        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - inf_start

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    peak_gpu_mb = (
        torch.cuda.max_memory_allocated() / 1e6 if DEVICE.type == 'cuda' else 0.0
    )

    timing_payload = {
        'patient_id': patient_id,
        'train_seconds': float(train_time),
        'epochs_done': int(epochs_done),
        'seconds_per_epoch': float(train_time / max(epochs_done, 1)),
        'inference_seconds_full_field': float(inference_seconds),
        'inference_points': int(n_inf),
        'n_params': int(n_params),
        'peak_gpu_mb': float(peak_gpu_mb),
        'holdout_fraction': float(dataset.holdout_fraction),
        'num_train': int(dataset.num_train),
        'num_holdout': int(dataset.num_holdout),
        'loss_weights': {k: float(v) for k, v in loss_weights.items()},
        'loss_priorities': {k: float(v) for k, v in LOSS_PRIORITY.items()},
    }
    with open(patient_results / 'timing.json', 'w') as f:
        json.dump(timing_payload, f, indent=2)
    print(
        f"[TIMING] {train_time:.1f}s train, "
        f"{inference_seconds:.3f}s full-field inference "
        f"({n_inf:,} points), peak GPU {peak_gpu_mb:.0f} MB"
    )

    # Save results
    results = {
        'patient_id': patient_id,
        'metrics': metrics,
        'history': history,
        'timing': timing_payload,
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
                    Re, wss_scale_factor=1.0,
                    rheology="newtonian", cy_params=None,
                    U_ref=None, L_ref=None):
    """Compute all loss components."""
    # Velocity loss
    loss_vel = nn.MSELoss()(vel_pred, vel_true)

    # WSS loss
    if has_wss.any():
        loss_wss = nn.MSELoss()(outputs['wss'][has_wss], wss_true[has_wss])
        wss_res = compute_wss_physics_residual(
            model, coords[has_wss], normals[has_wss], wss_scale_factor,
            rheology=rheology, cy_params=cy_params,
            U_ref=U_ref, L_ref=L_ref,
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

    f_u, f_v, f_w, cont = compute_physics_residuals_fused(
        model, colloc, Re,
        rheology=rheology, cy_params=cy_params,
        U_ref=U_ref, L_ref=L_ref,
    )

    loss_ns = (f_u**2 + f_v**2 + f_w**2).mean()
    loss_cont = (cont**2).mean()

    return {
        'wss': loss_wss,
        'vel': loss_vel,
        'ns': loss_ns,
        'cont': loss_cont,
        'wss_phy': loss_wss_phy
    }
