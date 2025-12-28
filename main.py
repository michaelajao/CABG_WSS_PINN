"""
Physics-Informed Neural Network (PINN) for Wall Shear Stress Prediction.

This script provides the command-line interface for training PINNs on
patient-specific coronary artery CFD data. The model learns to predict
wall shear stress (WSS) and velocity fields while satisfying the
incompressible Navier-Stokes equations.

Usage:
    python main.py train --patient 0156 --epochs 500
    python main.py train --patient all --epochs 1000
    python main.py --help
"""

import argparse
import random
import traceback

import numpy as np
import torch

from src.config import DEVICE, PATIENT_DATA
from src.train import train_patient

# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_SEED: int = 42

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main() -> None:
    """
    Main entry point for PINN training.
    """
    parser = argparse.ArgumentParser(
        description='Physics-Informed Neural Network for WSS Prediction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train PINN model')

    # Patient selection
    train_parser.add_argument(
        '--patient',
        type=str,
        nargs='+',
        default=['H-12'],
        help=f"Patient ID(s) or 'all'. Available: {list(PATIENT_DATA.keys())}"
    )

    # Random seed
    train_parser.add_argument(
        '--seed',
        type=int,
        default=DEFAULT_SEED,
        help='Random seed for reproducibility'
    )

    # Training hyperparameters
    train_parser.add_argument(
        '--epochs',
        type=int,
        default=500,
        help='Maximum training epochs'
    )
    train_parser.add_argument(
        '--batch-size',
        type=int,
        default=4096,
        help='Training batch size'
    )
    train_parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Initial learning rate'
    )
    train_parser.add_argument(
        '--patience',
        type=int,
        default=50,
        help='Early stopping patience'
    )
    train_parser.add_argument(
        '--n-collocation',
        type=int,
        default=2048,
        help='Number of collocation points per batch'
    )
    train_parser.add_argument(
        '--grad-clip',
        type=float,
        default=1.0,
        help='Gradient clipping value (0 to disable)'
    )

    # Model architecture
    train_parser.add_argument(
        '--arch',
        type=str,
        default='fourier',
        choices=['vanilla', 'fourier', 'pirate', 'multi', 'kan'],
        help='Model architecture'
    )
    train_parser.add_argument(
        '--hidden-dim',
        type=int,
        default=256,
        help='Hidden layer dimension'
    )
    train_parser.add_argument(
        '--num-blocks',
        type=int,
        default=4,
        help='Number of ResNet blocks'
    )
    train_parser.add_argument(
        '--num-frequencies',
        type=int,
        default=64,
        help='Number of Fourier frequencies (for fourier arch)'
    )
    train_parser.add_argument(
        '--fourier-scale',
        type=float,
        default=10.0,
        help='Fourier frequency scale (for fourier arch)'
    )
    train_parser.add_argument(
        '--kan-grid-size',
        type=int,
        default=5,
        help='KAN grid size (for kan arch)'
    )
    train_parser.add_argument(
        '--kan-spline-order',
        type=int,
        default=3,
        help='KAN spline order (for kan arch)'
    )

    # Adaptive loss weighting
    train_parser.add_argument(
        '--adaptive-weights',
        action='store_true',
        help='Use ReLoBRaLo adaptive loss weighting (auto-balances loss terms)'
    )


    # Verbosity
    train_parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show progress bars'
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == 'train':
        run_training(args)


def run_training(args) -> None:
    """Run training for specified patient(s)."""
    # Print header
    print("\n" + "=" * 80)
    print("PINN TRAINING FOR CORONARY ARTERY WSS PREDICTION")
    print("=" * 80)
    print(f"Device: {DEVICE}")

    # Set random seeds
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print(f"Random seed: {seed}")

    # Select patients
    if 'all' in [p.lower() for p in args.patient]:
        patients = list(PATIENT_DATA.keys())
    else:
        patients = []
        for p in args.patient:
            if p not in PATIENT_DATA:
                print(f"Error: Patient '{p}' not found.")
                print(f"Available patients: {list(PATIENT_DATA.keys())}")
                return
            patients.append(p)

    print(f"Patients to train: {patients}")
    print(f"Architecture: {args.arch}")

    # Train each patient
    results = {}
    for pid in patients:
        try:
            model, result = train_patient(
                patient_id=pid,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                n_collocation=args.n_collocation,
                patience=args.patience,
                hidden_dim=args.hidden_dim,
                num_blocks=args.num_blocks,
                grad_clip=args.grad_clip,
                arch=args.arch,
                kan_grid_size=args.kan_grid_size,
                kan_spline_order=args.kan_spline_order,
                num_frequencies=args.num_frequencies,
                fourier_scale=args.fourier_scale,
                adaptive_weights=args.adaptive_weights,
                verbose=args.verbose
            )
            results[pid] = result

        except Exception as e:
            print(f"Error training patient {pid}: {e}")
            traceback.print_exc()
            continue

    # Print summary
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)

    for pid, result in results.items():
        metrics = result['metrics']
        print(f"\nPatient {pid}:")
        print(f"  RMSE:  {metrics['RMSE']:.4f}")
        print(f"  NRMSE: {metrics['NRMSE']:.4f}")
        print(f"  R2:    {metrics['R2']:.4f}")

    print("\n" + "=" * 80)
    print("ALL TASKS COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
