"""
Physics-Informed Neural Network (PINN) for Wall Shear Stress Prediction.

This script provides the command-line interface for training PINNs on
patient-specific coronary artery CFD data. The model learns to predict
wall shear stress (WSS) and velocity fields while satisfying the
incompressible Navier-Stokes equations.

Usage:
    python main.py --patient 0156 --epochs 500
    python main.py --patient all --epochs 1000 --arch multi
    python main.py --help

Example:
    # Train single patient with Fourier features
    python main.py --patient H-12 --epochs 500 --arch fourier

    # Train all patients with default settings
    python main.py --patient all --epochs 500 --batch-size 4096
"""

import argparse
import random
import sys
from typing import Dict, Optional

import numpy as np
import torch

from src.config import DEVICE, PATIENT_DATA, RESULTS_PATH
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

    Parses command-line arguments, sets random seeds for reproducibility,
    and trains PINN models for specified patients. Results are saved to
    the configured results directory.

    Returns:
        None

    Raises:
        SystemExit: If argument parsing fails.
    """
    parser = argparse.ArgumentParser(
        description=(
            'Physics-Informed Neural Network for WSS Prediction '
            'in Coronary Arteries'
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Patient selection
    parser.add_argument(
        '--patient',
        type=str,
        default='0073',
        help=f"Patient ID or 'all'. Available: {list(PATIENT_DATA.keys())}"
    )

    # Random seed
    parser.add_argument(
        '--seed',
        type=int,
        default=DEFAULT_SEED,
        help='Random seed for reproducibility'
    )

    # Training hyperparameters
    parser.add_argument(
        '--epochs',
        type=int,
        default=500,
        help='Maximum training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4096,
        help='Training batch size'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Initial learning rate'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=50,
        help='Early stopping patience (epochs without improvement)'
    )
    parser.add_argument(
        '--grad-clip',
        type=float,
        default=1.0,
        help='Gradient clipping value (0 to disable)'
    )

    # Model architecture
    parser.add_argument(
        '--arch',
        type=str,
        default='vanilla',
        choices=['vanilla', 'fourier', 'multi', 'kan'],
        help=(
            'Architecture: vanilla (simple MLP), fourier (with Fourier features), '
            'multi (separate networks), kan (experimental)'
        )
    )
    parser.add_argument(
        '--hidden-dim',
        type=int,
        default=256,
        help='Hidden layer dimension'
    )
    parser.add_argument(
        '--num-blocks',
        type=int,
        default=4,
        help='Number of ResNet blocks (or KAN layers for --arch kan)'
    )

    # Fourier-specific parameters (for --arch fourier)
    parser.add_argument(
        '--num-frequencies',
        type=int,
        default=64,
        help='Number of Fourier frequencies (only for --arch fourier)'
    )
    parser.add_argument(
        '--fourier-scale',
        type=float,
        default=10.0,
        help='Fourier frequency scale (only for --arch fourier)'
    )

    # KAN-specific parameters
    parser.add_argument(
        '--kan-grid-size',
        type=int,
        default=5,
        help='KAN: B-spline grid size (only for --arch kan)'
    )
    parser.add_argument(
        '--kan-spline-order',
        type=int,
        default=3,
        help='KAN: B-spline order (only for --arch kan)'
    )

    # Physics configuration
    parser.add_argument(
        '--collocation',
        type=int,
        default=2048,
        help='Collocation points per batch for physics loss'
    )

    args = parser.parse_args()

    # Print header
    print("\n" + "=" * 80)
    print("PINN TRAINING FOR CORONARY ARTERY WSS PREDICTION")
    print("=" * 80)
    print(f"Device: {DEVICE}")

    # Print GPU info with error handling
    if torch.cuda.is_available():
        try:
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        except RuntimeError as e:
            print(f"GPU available but device info unavailable: {e}")

    # Set random seeds for reproducibility
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print(f"Random seed: {seed}")

    # Validate and select patients
    if args.patient.lower() == 'all':
        patients = list(PATIENT_DATA.keys())
    else:
        patients = [args.patient]

    # Validate patient IDs before training
    for pid in patients:
        if pid not in PATIENT_DATA:
            print(f"\nError: Unknown patient '{pid}'")
            print(f"Available patients: {list(PATIENT_DATA.keys())}")
            sys.exit(1)

    # Train each patient
    all_results: Dict[str, Dict] = {}
    for pid in patients:
        model, results = train_patient(
            patient_id=pid,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            n_collocation=args.collocation,
            patience=args.patience,
            hidden_dim=args.hidden_dim,
            num_blocks=args.num_blocks,
            grad_clip=args.grad_clip,
            arch=args.arch,
            kan_grid_size=args.kan_grid_size,
            kan_spline_order=args.kan_spline_order,
            num_frequencies=args.num_frequencies,
            fourier_scale=args.fourier_scale
        )
        all_results[pid] = results['metrics']

    # Print summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE - SUMMARY")
    print("=" * 80)
    print(
        f"{'Patient':<10} {'RMSE (Pa)':<12} {'NRMSE':<10} "
        f"{'MAE (Pa)':<12} {'R²':<10}"
    )
    print("-" * 70)
    for pid, m in all_results.items():
        print(
            f"{pid:<10} {m['RMSE']:<12.4f} {m['NRMSE']:<10.4f} "
            f"{m['MAE']:<12.4f} {m['R2']:<10.4f}"
        )

    # Save summary with error handling
    try:
        RESULTS_PATH.mkdir(parents=True, exist_ok=True)
        summary_path = RESULTS_PATH / "training_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("PINN TRAINING SUMMARY - ALL PATIENTS\n")
            f.write("=" * 70 + "\n\n")
            f.write(
                f"{'Patient':<10} {'RMSE (Pa)':<12} {'NRMSE':<10} "
                f"{'MAE (Pa)':<12} {'R²':<10}\n"
            )
            f.write("-" * 70 + "\n")
            for pid, m in all_results.items():
                f.write(
                    f"{pid:<10} {m['RMSE']:<12.4f} {m['NRMSE']:<10.4f} "
                    f"{m['MAE']:<12.4f} {m['R2']:<10.4f}\n"
                )
        print(f"\nSummary saved to: {summary_path}")
    except IOError as e:
        print(f"\nWarning: Could not save summary file: {e}")


if __name__ == '__main__':
    main()
