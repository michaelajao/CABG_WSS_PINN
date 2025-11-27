import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import time
from typing import Tuple, Dict

from src.config import DEVICE, MODELS_PATH, FIGURES_PATH, RESULTS_PATH, PATIENT_DATA, RHO
from src.dataset import load_patient_data, PatientDataset, CollocationSampler
from src.model import MultiResNetPINN, SharedTrunkPINN
from src.physics import (
    navier_stokes_residual, continuity_residual, 
    compute_wss_from_velocity, wss_physics_residual
)
from src.utils import EarlyStopping
from src.evaluate import evaluate_model
from src.plots import generate_all_plots

def train_patient(
    patient_id: str,
    epochs: int = 500,
    batch_size: int = 4096,
    learning_rate: float = 1e-4,
    n_collocation: int = 2048,
    patience: int = 50,
    compute_wss: bool = False,
    hidden_dim: int = 256,
    num_blocks: int = 4,
    grad_clip: float = 1.0,
    arch: str = 'shared',
    verbose: bool = True
) -> Tuple[nn.Module, Dict]:
    """
    Train per-patient PINN with all fixes applied.
    
    Args:
        patient_id: Patient identifier
        epochs: Maximum training epochs
        batch_size: Batch size
        learning_rate: Initial learning rate
        n_collocation: Number of collocation points per batch
        patience: Early stopping patience
        compute_wss: If True, compute WSS from velocity gradients
                    If False, predict WSS with separate network
        hidden_dim: Hidden layer dimension
        num_blocks: Number of ResNet blocks
        grad_clip: Gradient clipping value
        arch: Architecture type ('shared' or 'multi')
        verbose: Print progress
        
    Returns:
        Trained model and results dictionary
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
    print(f"WSS Mode: {'COMPUTED from velocity' if compute_wss else 'PREDICTED by network'}")
    print("=" * 80)
    
    # Load data
    print("\n[LOADING DATA]")
    data, per_vessel = load_patient_data(patient_id)
    
    # Create dataset and dataloader
    dataset = PatientDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                       num_workers=0, pin_memory=True)
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
    if arch == 'shared':
        model = SharedTrunkPINN(
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            head_layers=2,
            predict_wss=not compute_wss
        ).to(DEVICE)
        arch_name = 'SharedTrunkPINN'
    else:  # 'multi'
        model = MultiResNetPINN(
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            predict_wss=not compute_wss
        ).to(DEVICE)
        arch_name = 'MultiResNetPINN'
    
    print(f"\n[MODEL]")
    print(f"  Architecture: {arch_name} ({num_blocks} blocks, {hidden_dim} dim)")
    print(f"  Parameters: {model.count_parameters():,}")
    print(f"  WSS Output: {'Computed' if compute_wss else 'Predicted'}")
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    # Physics scaling
    coord_scale = torch.tensor(
        dataset.scaler_X.data_range_, dtype=torch.float32, device=DEVICE
    ).view(1, 3)
    
    # Characteristic scales for normalization
    U_char, L_char = 0.1, 0.05  # m/s, m
    nse_scale = (RHO * U_char**2) / L_char
    cont_scale = U_char / L_char
    
    # Early stopping to prevent overfitting
    early_stopper = EarlyStopping(patience=patience, monitor='loss')
    
    # Training history
    history = {
        'total_loss': [], 'wss_loss': [], 'vel_loss': [],
        'nse_loss': [], 'cont_loss': [], 'wss_physics_loss': [],
        'lr': []
    }
    best_loss = float('inf')
    
    print(f"\n[TRAINING] Max {epochs} epochs, early stopping patience={patience}")
    print("-" * 80)
    
    # Start training timer
    train_start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        epoch_losses = {k: 0.0 for k in history.keys() if k != 'lr'}
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1:3d}/{epochs}", 
                   disable=not verbose, leave=False)
        
        for batch in pbar:
            coords = batch['coords'].to(DEVICE)
            wss_true = batch['wss'].to(DEVICE)
            vel_true = batch['velocity'].to(DEVICE)
            normals = batch['normals'].to(DEVICE)
            has_wss = batch['has_wss'].to(DEVICE).squeeze()
            
            optimizer.zero_grad()
            
            outputs = model(coords)
            vel_pred = torch.cat([outputs['u'], outputs['v'], outputs['w']], dim=1)
            
            # === DATA LOSSES ===
            
            # Velocity loss (all points)
            loss_vel = nn.MSELoss()(vel_pred, vel_true)
            
            # WSS loss (wall points only)
            if compute_wss:
                # Compute WSS from velocity gradients
                if has_wss.any():
                    wss_computed = compute_wss_from_velocity(
                        model, coords[has_wss], normals[has_wss], coord_scale
                    )
                    # Scale for comparison
                    wss_computed_scaled = dataset.scaler_y.transform(
                        wss_computed.detach().cpu().numpy()
                    )
                    wss_computed_scaled = torch.FloatTensor(wss_computed_scaled).to(DEVICE)
                    loss_wss = nn.MSELoss()(wss_computed_scaled, wss_true[has_wss])
                else:
                    loss_wss = torch.tensor(0.0, device=DEVICE)
                loss_wss_physics = torch.tensor(0.0, device=DEVICE)
            else:
                # Predict WSS directly
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
            
            # === PHYSICS LOSSES ===
            
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
            
            loss_nse = 0.5 * loss_nse_data + 0.5 * loss_nse_colloc
            loss_cont = 0.5 * loss_cont_data + 0.5 * loss_cont_colloc
            
            # === TOTAL LOSS ===
            # Weights: WSS (primary) > Velocity > Physics > WSS constraint
            loss_total = (
                1.0 * loss_wss +
                0.1 * loss_vel +
                1.0 * loss_nse +
                1.0 * loss_cont +
                0.1 * loss_wss_physics
            )
            
            # Backward with gradient clipping
            loss_total.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            
            # Accumulate losses
            epoch_losses['total_loss'] += loss_total.item()
            epoch_losses['wss_loss'] += loss_wss.item()
            epoch_losses['vel_loss'] += loss_vel.item()
            epoch_losses['nse_loss'] += loss_nse.item()
            epoch_losses['cont_loss'] += loss_cont.item()
            epoch_losses['wss_physics_loss'] += loss_wss_physics.item()
            
            pbar.set_postfix({
                'Loss': f'{loss_total.item():.4f}',
                'WSS': f'{loss_wss.item():.4f}'
            })
        
        scheduler.step()
        
        # Average losses
        n_batches = len(loader)
        for k in epoch_losses:
            epoch_losses[k] /= n_batches
            history[k].append(epoch_losses[k])
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Save best model
        if epoch_losses['total_loss'] < best_loss:
            best_loss = epoch_losses['total_loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'config': {
                    'hidden_dim': hidden_dim,
                    'num_blocks': num_blocks,
                    'compute_wss': compute_wss
                }
            }, patient_models / f'pinn_{patient_id}_best.pth')
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:3d} | Loss: {epoch_losses['total_loss']:.4f} | "
                  f"WSS: {epoch_losses['wss_loss']:.4f} | "
                  f"Physics: {epoch_losses['nse_loss']:.2e} | LR: {lr:.2e}")
        
        # Check early stopping condition
        if early_stopper(epoch_losses['total_loss'], epoch):
            break
    
    # Calculate training time
    train_end_time = time.time()
    total_train_time = train_end_time - train_start_time
    epochs_trained = len(history['total_loss'])
    time_per_epoch = total_train_time / epochs_trained if epochs_trained > 0 else 0
    
    print(f"\n[TIMING]")
    print(f"  Total training time: {total_train_time:.1f}s ({total_train_time/60:.2f} min)")
    print(f"  Time per epoch: {time_per_epoch:.2f}s")
    print(f"  Epochs trained: {epochs_trained}")
    
    # Load best model
    checkpoint = torch.load(patient_models / f'pinn_{patient_id}_best.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluation
    print("\n[EVALUATION]")
    metrics = evaluate_model(model, loader, dataset, compute_wss, coord_scale)
    
    # Generate plots
    print("\n[GENERATING PLOTS]")
    generate_all_plots(model, loader, dataset, patient_id, patient_figures, metrics, per_vessel)
    print(f"  Saved to: {patient_figures}")
    
    # Save results
    results = {
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
            'compute_wss': compute_wss,
            'hidden_dim': hidden_dim,
            'num_blocks': num_blocks
        }
    }
    
    # Save as readable text file (User request from previous turn)
    results_file = patient_results / f'{patient_id}_results.txt'
    with open(results_file, 'w') as f:
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
        f.write(f"  R²:       {metrics['R2']:.4f}\n")
        f.write(f"  Pearson:  {metrics['Pearson']:.4f}\n\n")
        
        f.write("-" * 40 + "\n")
        f.write("TRAINING SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Architecture: {arch_name}\n")
        f.write(f"  Parameters:  {model.count_parameters():,}\n")
        f.write(f"  Epochs:      {epochs_trained}\n")
        f.write(f"  Final Loss:  {history['total_loss'][-1]:.6f}\n")
        f.write(f"  Best Loss:   {min(history['total_loss']):.6f}\n\n")
        
        f.write("-" * 40 + "\n")
        f.write("TIMING\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Total time:      {total_train_time:.1f}s ({total_train_time/60:.2f} min)\n")
        f.write(f"  Time per epoch:  {time_per_epoch:.2f}s\n")
    
    # Save history separately (can be large)
    with open(patient_results / f'{patient_id}_history.json', 'w') as f:
        json.dump(history, f)
    
    print(f"\n  Results saved to: {patient_results}")
    
    return model, results

