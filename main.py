"""
Physics-Informed Neural Network (PINN) for Wall Shear Stress Prediction

This script provides the command-line interface for training PINNs on 
patient-specific coronary artery CFD data. The model learns to predict 
wall shear stress (WSS) and velocity fields while satisfying the 
incompressible Navier-Stokes equations.

Usage:
    python main.py --patient 0156 --epochs 500
    python main.py --patient all --epochs 1000 --arch shared
    python main.py --help

Author: [Your Name]
"""

import argparse
import torch
import numpy as np
import random
import json
from pathlib import Path
from src.config import DEVICE, PATIENT_DATA, RESULTS_PATH
from src.train import train_patient


def main():
    """Main entry point for PINN training."""
    parser = argparse.ArgumentParser(
        description='Physics-Informed Neural Network for WSS Prediction in Coronary Arteries',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Patient selection
    parser.add_argument('--patient', type=str, default='0073',
                        help=f"Patient ID or 'all'. Available: {list(PATIENT_DATA.keys())}")
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=500, 
                        help='Maximum training epochs')
    parser.add_argument('--batch-size', type=int, default=4096, 
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-4, 
                        help='Initial learning rate')
    parser.add_argument('--patience', type=int, default=50, 
                        help='Early stopping patience (epochs without improvement)')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                        help='Gradient clipping value (0 to disable)')
    
    # Model architecture
    parser.add_argument('--arch', type=str, default='shared', 
                        choices=['shared', 'multi', 'kan'],
                        help='Architecture: shared (efficient), multi (separate networks), or kan (experimental)')
    parser.add_argument('--hidden-dim', type=int, default=256, 
                        help='Hidden layer dimension')
    parser.add_argument('--num-blocks', type=int, default=4, 
                        help='Number of ResNet blocks (or KAN layers for --arch kan)')
    
    # KAN-specific parameters
    parser.add_argument('--kan-grid-size', type=int, default=5,
                        help='KAN: B-spline grid size (only for --arch kan)')
    parser.add_argument('--kan-spline-order', type=int, default=3,
                        help='KAN: B-spline order (only for --arch kan)')
    
    # Physics configuration
    parser.add_argument('--collocation', type=int, default=2048, 
                        help='Collocation points per batch for physics loss')
    parser.add_argument('--compute-wss', action='store_true',
                        help='Compute WSS from velocity gradients (instead of predicting directly)')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("PINN TRAINING FOR CORONARY ARTERY WSS PREDICTION")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Set random seeds for reproducibility
    seed = 42
    random.seed(seed)  # Python random module
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch current GPU
    torch.cuda.manual_seed_all(seed)  # PyTorch all GPUs
    
    # Select patients to train
    patients = list(PATIENT_DATA.keys()) if args.patient.lower() == 'all' else [args.patient]
    
    # Train each patient
    all_results = {}
    for pid in patients:
        if pid not in PATIENT_DATA:
            print(f"\nWarning: Unknown patient {pid}")
            continue
        
        model, results = train_patient(
            patient_id=pid,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            n_collocation=args.collocation,
            patience=args.patience,
            compute_wss=args.compute_wss,
            hidden_dim=args.hidden_dim,
            num_blocks=args.num_blocks,
            grad_clip=args.grad_clip,
            arch=args.arch,
            kan_grid_size=args.kan_grid_size,
            kan_spline_order=args.kan_spline_order
        )
        all_results[pid] = results['metrics']
    
    # Summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE - SUMMARY")
    print("=" * 80)
    print(f"{'Patient':<10} {'RMSE (Pa)':<12} {'NRMSE':<10} {'MAE (Pa)':<12} {'R²':<10} {'Pearson':<10}")
    print("-" * 80)
    for pid, m in all_results.items():
        print(f"{pid:<10} {m['RMSE']:<12.4f} {m['NRMSE']:<10.4f} {m['MAE']:<12.4f} {m['R2']:<10.4f} {m['Pearson']:<10.4f}")
    
    # Save summary as readable text file
    summary_path = RESULTS_PATH / "training_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("PINN TRAINING SUMMARY - ALL PATIENTS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{'Patient':<10} {'RMSE (Pa)':<12} {'NRMSE':<10} {'MAE (Pa)':<12} {'R²':<10} {'Pearson':<10}\n")
        f.write("-" * 70 + "\n")
        for pid, m in all_results.items():
            f.write(f"{pid:<10} {m['RMSE']:<12.4f} {m['NRMSE']:<10.4f} {m['MAE']:<12.4f} {m['R2']:<10.4f} {m['Pearson']:<10.4f}\n")
    print(f"\nSummary saved to: {summary_path}")

if __name__ == '__main__':
    main()
