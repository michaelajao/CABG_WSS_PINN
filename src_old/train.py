"""
Training module for PINN

Contains training loop, early stopping, and model checkpoint management.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from typing import Dict, Optional
from pathlib import Path

from .config import (
    DEVICE, MODEL_PATH, LEARNING_RATE, EPOCHS,
    LR_SCHEDULER_FACTOR, LR_SCHEDULER_PATIENCE, EARLY_STOPPING_PATIENCE,
    PHYSICS_WEIGHT_NSE, PHYSICS_WEIGHT_CONT, DATA_WEIGHT,
    USE_COLLOCATION_POINTS, COLLOCATION_POINTS_PER_BATCH
)
from .physics import navier_stokes_residual, continuity_residual
from .model import ResNetPINN
import numpy as np


def train_pinn(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    coord_scaler: Optional[object] = None,
    train_data_dict: Optional[Dict] = None,
    epochs: int = EPOCHS,
    learning_rate: float = LEARNING_RATE,
    physics_weight_nse: float = PHYSICS_WEIGHT_NSE,
    physics_weight_cont: float = PHYSICS_WEIGHT_CONT,
    data_weight: float = DATA_WEIGHT,
    velocity_weight: float = 0.0,
    use_collocation: bool = USE_COLLOCATION_POINTS,
    n_collocation: int = COLLOCATION_POINTS_PER_BATCH,
    save_path: Path = MODEL_PATH,
    verbose: bool = True,
    init_method: Optional[str] = None
) -> Dict:
    """
    Train Physics-Informed Neural Network

    Args:
        model: PINN model
        train_loader: Training data loader
        val_loader: Validation data loader
        coord_scaler: StandardScaler for coordinates (for physics chain rule)
        train_data_dict: Training data dictionary for collocation point sampling
        epochs: Number of training epochs
        learning_rate: Initial learning rate
        physics_weight_nse: Weight for Navier-Stokes loss
        physics_weight_cont: Weight for continuity loss
        data_weight: Weight for data fitting loss (WSS)
        velocity_weight: Weight for velocity supervision loss
        use_collocation: Whether to use collocation points
        n_collocation: Number of collocation points per batch
        save_path: Directory to save model checkpoints
        verbose: Whether to print training progress
        init_method: Weight initialization method

    Returns:
        Dictionary containing training history
    """
    if verbose:
        print("\n" + "="*80)
        print("TRAINING STARTED")
        print("="*80)
        print(f"  Training Batches: {len(train_loader)}")
        print(f"  Validation Batches: {len(val_loader)}")
        if use_collocation:
            print(f"  Collocation Points per Batch: {n_collocation}")
        print("="*80)

    # Move model to device
    model = model.to(DEVICE)

    # Setup collocation points if enabled
    collocation_bounds = None
    if use_collocation and train_data_dict is not None:
        X_train = train_data_dict['X']
        collocation_bounds = {
            'x_min': X_train[:, 0].min(),
            'x_max': X_train[:, 0].max(),
            'y_min': X_train[:, 1].min(),
            'y_max': X_train[:, 1].max(),
            'z_min': X_train[:, 2].min(),
            'z_max': X_train[:, 2].max()
        }
        if verbose:
            print(f"\n[COLLOCATION POINTS]")
            print(f"  Domain: X=[{collocation_bounds['x_min']:.4f}, {collocation_bounds['x_max']:.4f}], Y=[{collocation_bounds['y_min']:.4f}, {collocation_bounds['y_max']:.4f}], Z=[{collocation_bounds['z_min']:.4f}, {collocation_bounds['z_max']:.4f}]")
            print(f"  Sampling: {n_collocation} points/batch")
    elif use_collocation and train_data_dict is None:
        if verbose:
            print("\n[WARNING] Collocation points requested but train_data_dict not provided. Disabling.")
        use_collocation = False

    # Extract coordinate scale factors for physics chain rule
    # MinMaxScaler: x_scaled = (x - min) / (max - min)
    # ∂/∂x_physical = (max - min) * ∂/∂x_scaled
    coord_scale = None
    if coord_scaler is not None:
        # For MinMaxScaler: data_range_ = max - min for each feature
        coord_scale = torch.tensor(coord_scaler.data_range_, dtype=torch.float32, device=DEVICE).view(1, 3)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=LR_SCHEDULER_FACTOR,
        patience=LR_SCHEDULER_PATIENCE
    )

    # =========================================================================
    # Physics Loss Normalization (Option 2)
    # =========================================================================
    # Compute characteristic scales from data
    # Velocity scale: typical velocity magnitude (from streamline data)
    U_char = 0.1  # m/s (inlet velocity)
    # Length scale: domain size
    L_char = 0.05  # m (typical vessel diameter ~50mm)
    # Density and viscosity
    from .config import RHO, MU

    # Characteristic scales for normalization
    # NSE residual scale: ρU²/L ≈ 1060 * 0.1² / 0.05 ≈ 21.2 Pa/m
    nse_scale = (RHO * U_char**2) / L_char
    # Continuity residual scale: U/L ≈ 0.1 / 0.05 = 2.0 1/s
    cont_scale = U_char / L_char

    if verbose:
        print(f"\n[PHYSICS NORMALIZATION]")
        print(f"  NSE scale: {nse_scale:.2f} Pa/m")
        print(f"  Continuity scale: {cont_scale:.2f} 1/s")

    # Loss weights (fixed for simplicity)
    if verbose:
        print(f"\n[LOSS WEIGHTS]")
        print(f"  WSS={data_weight:.2f}, Vel={velocity_weight:.2f}, NSE={physics_weight_nse:.2f}, Cont={physics_weight_cont:.2f}")

    # Loss history
    history = {
        'train_loss': [],
        'val_loss': [],
        'data_loss_wss': [],
        'data_loss_velocity': [],
        'physics_loss_nse': [],
        'physics_loss_cont': [],
        'learning_rate': [],
        'adaptive_weights': []
    }

    # Early stopping
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0

    # Training loop
    for epoch in range(epochs):
        # =====================================================================
        # Training Phase
        # =====================================================================
        model.train()
        train_loss_epoch = 0
        wss_loss_epoch = 0
        velocity_loss_epoch = 0
        physics_nse_epoch = 0
        physics_cont_epoch = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", disable=not verbose)

        for batch in pbar:
            coords = batch['coords'].to(DEVICE)
            wss_true = batch['wss'].to(DEVICE)
            velocity_true = batch['velocity'].to(DEVICE)  # Velocity from streamlines or wall
            has_wss = batch['has_wss'].to(DEVICE).squeeze()  # Boolean mask for WSS availability

            optimizer.zero_grad()

            # Forward pass
            outputs = model(coords)
            wss_pred = outputs['wss']
            u_pred = outputs['u']
            v_pred = outputs['v']
            w_pred = outputs['w']

            # Data loss: WSS supervision (only where has_wss=True)
            if has_wss.any():
                loss_wss = nn.MSELoss()(wss_pred[has_wss], wss_true[has_wss])
            else:
                loss_wss = torch.tensor(0.0, device=DEVICE)

            # Data loss: Velocity supervision (all points: interior + wall)
            velocity_pred = torch.cat([u_pred, v_pred, w_pred], dim=1)
            loss_velocity = nn.MSELoss()(velocity_pred, velocity_true)

            # Physics losses (use SCALED coordinates with scale factors)
            f_u, f_v, f_w = navier_stokes_residual(model, coords, coord_scale)
            loss_nse_raw = (f_u**2 + f_v**2 + f_w**2).mean()

            cont = continuity_residual(model, coords, coord_scale)
            loss_cont_raw = (cont**2).mean()

            # Collocation points: additional physics enforcement at random locations
            if use_collocation and collocation_bounds is not None:
                # Sample random points in domain
                colloc_coords_np = np.random.uniform(
                    low=[collocation_bounds['x_min'], collocation_bounds['y_min'], collocation_bounds['z_min']],
                    high=[collocation_bounds['x_max'], collocation_bounds['y_max'], collocation_bounds['z_max']],
                    size=(n_collocation, 3)
                ).astype(np.float32)
                
                # Apply same scaling as training data
                if coord_scaler is not None:
                    colloc_coords_scaled = coord_scaler.transform(colloc_coords_np)
                else:
                    colloc_coords_scaled = colloc_coords_np
                
                colloc_coords_tensor = torch.FloatTensor(colloc_coords_scaled).to(DEVICE)
                
                # Compute physics residuals at collocation points
                f_u_c, f_v_c, f_w_c = navier_stokes_residual(model, colloc_coords_tensor, coord_scale)
                cont_c = continuity_residual(model, colloc_coords_tensor, coord_scale)
                
                # Add to physics losses
                loss_nse_raw += (f_u_c**2 + f_v_c**2 + f_w_c**2).mean()
                loss_cont_raw += (cont_c**2).mean()

            # Normalize physics losses by characteristic scales
            loss_nse = loss_nse_raw / (nse_scale**2 + 1e-10)
            loss_cont = loss_cont_raw / (cont_scale**2 + 1e-10)

            # Total loss with fixed weights
            loss_total = (
                data_weight * loss_wss +
                velocity_weight * loss_velocity +
                physics_weight_nse * loss_nse +
                physics_weight_cont * loss_cont
            )

            # Backward pass
            loss_total.backward()
            optimizer.step()

            # Accumulate losses
            train_loss_epoch += loss_total.item()
            wss_loss_epoch += loss_wss.item()
            velocity_loss_epoch += loss_velocity.item()
            physics_nse_epoch += loss_nse.item()
            physics_cont_epoch += loss_cont.item()

            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss_total.item():.4f}',
                'WSS': f'{loss_wss.item():.4f}',
                'Vel': f'{loss_velocity.item():.4f}',
                'NSE': f'{loss_nse.item():.4e}',
                'Cont': f'{loss_cont.item():.4e}'
            })

        # Average training losses
        n_batches = len(train_loader)
        train_loss_epoch /= n_batches
        wss_loss_epoch /= n_batches
        velocity_loss_epoch /= n_batches
        physics_nse_epoch /= n_batches
        physics_cont_epoch /= n_batches

        # =====================================================================
        # Validation Phase
        # =====================================================================
        model.eval()
        val_loss_epoch = 0

        with torch.no_grad():
            for batch in val_loader:
                coords = batch['coords'].to(DEVICE)
                wss_true = batch['wss'].to(DEVICE)
                has_wss = batch['has_wss'].to(DEVICE).squeeze()

                outputs = model(coords)
                wss_pred = outputs['wss']

                # Only compute loss where WSS is available
                if has_wss.any():
                    loss = nn.MSELoss()(wss_pred[has_wss], wss_true[has_wss])
                    val_loss_epoch += loss.item()

        val_loss_epoch /= len(val_loader)

        # Update learning rate scheduler
        scheduler.step(val_loss_epoch)

        # Save to history
        history['train_loss'].append(train_loss_epoch)
        history['val_loss'].append(val_loss_epoch)
        history['data_loss_wss'].append(wss_loss_epoch)
        history['data_loss_velocity'].append(velocity_loss_epoch)
        history['physics_loss_nse'].append(physics_nse_epoch)
        history['physics_loss_cont'].append(physics_cont_epoch)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])

        # Print epoch summary
        if verbose and (epoch + 1) % 10 == 0:
            print(f"\n[Epoch {epoch+1}/{epochs}]")
            print(f"  Train Loss: {train_loss_epoch:.6f} | Val Loss: {val_loss_epoch:.6f}")
            print(f"  WSS: {wss_loss_epoch:.6f} | Vel: {velocity_loss_epoch:.6f} | NSE: {physics_nse_epoch:.4e} | Cont: {physics_cont_epoch:.4e}")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.4e}")

        # =====================================================================
        # Model Checkpointing and Early Stopping
        # =====================================================================
        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            best_epoch = epoch
            patience_counter = 0

            # Build model configuration dictionary
            model_config = {
                'arch': 'resnet' if isinstance(model, ResNetPINN) else 'pinn',
                'activation': model.activation_name if hasattr(model, 'activation_name') else 'tanh',
            }

            # Add activation beta if it exists
            if hasattr(model, 'activation_beta'):
                model_config['activation_beta'] = model.activation_beta

            # Add init method if used
            if init_method is not None:
                model_config['init_method'] = init_method

            # Add architecture-specific parameters
            if isinstance(model, ResNetPINN):
                model_config['res_width'] = model.width
                model_config['res_blocks'] = model.blocks
            else:  # PINN
                model_config['layers'] = model.layer_sizes

            # Save best model with architecture metadata
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss_epoch,
                'train_loss': train_loss_epoch,
                'history': history,
                'model_config': model_config
            }

            torch.save(checkpoint, save_path / "best_pinn_model.pth")

            if verbose and (epoch + 1) % 10 == 0:
                print(f"  [SAVED] Best model (val_loss: {val_loss_epoch:.6f})")

        else:
            patience_counter += 1

        # Early stopping check
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            if verbose:
                print(f"\n{'='*80}")
                print(f"Early stopping at epoch {epoch+1}")
                print(f"Best model from epoch {best_epoch+1} (val_loss: {best_val_loss:.6f})")
                print(f"{'='*80}")
            break

    # Save final training history
    with open(save_path / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)

    if verbose:
        print(f"\n[TRAINING COMPLETE] Best Val Loss: {best_val_loss:.6f} | Total Epochs: {len(history['train_loss'])}")

    return history


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    load_optimizer: bool = False,
    device: torch.device = DEVICE
) -> Dict:
    """
    Load model checkpoint with architecture validation

    Args:
        model: PINN model
        checkpoint_path: Path to checkpoint file
        load_optimizer: Whether to return optimizer state
        device: Device to load model to

    Returns:
        Dictionary with checkpoint information
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Try to load state dict with informative error message
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except RuntimeError as e:
        print("\n" + "="*80)
        print("ERROR: Architecture Mismatch")
        print("="*80)
        print("\nThe checkpoint was saved with a different model architecture.")

        if 'model_config' in checkpoint:
            print("\nCheckpoint was trained with:")
            config = checkpoint['model_config']
            print(f"  Architecture: {config['arch'].upper()}")
            print(f"  Activation: {config.get('activation', 'unknown')}")
            if config['arch'] == 'resnet':
                print(f"  ResNet Width: {config.get('res_width', 'unknown')}")
                print(f"  ResNet Blocks: {config.get('res_blocks', 'unknown')}")
            else:
                print(f"  Layers: {config.get('layers', 'unknown')}")
            print("\nPlease rebuild model with matching architecture.")
        else:
            print("\nCheckpoint lacks architecture metadata (legacy format).")
            print("Please manually specify architecture parameters.")

        print("="*80)
        raise

    model.to(device)

    info = {
        'epoch': checkpoint['epoch'],
        'val_loss': checkpoint['val_loss'],
        'train_loss': checkpoint['train_loss']
    }

    if load_optimizer:
        info['optimizer_state_dict'] = checkpoint['optimizer_state_dict']
        if 'scheduler_state_dict' in checkpoint:
            info['scheduler_state_dict'] = checkpoint['scheduler_state_dict']

    return info
