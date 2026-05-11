"""
Physics-Informed Neural Network (PINN) for Wall Shear Stress Prediction.

This script provides the command-line interface for training FourierPINN on
patient-specific coronary artery CFD data. The model learns to predict
wall shear stress (WSS) and velocity fields while satisfying the
incompressible Navier-Stokes equations.

Usage:
    python main.py train --patient H4 --epochs 500
    python main.py train --patient BG4 --epochs 500 --rheology newtonian
    python main.py train --patient H2 --epochs 500 --rheology carreau_yasuda
    python main.py train --patient all --epochs 500
    python main.py --help
"""

import argparse
import random

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
    """Main entry point for PINN training."""
    parser = argparse.ArgumentParser(
        description='Physics-Informed Neural Network for WSS Prediction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train FourierPINN model')

    # Patient selection (uses public paper labels: H1..H4, BG1..BG5, D1..D3)
    train_parser.add_argument(
        '--patient',
        type=str,
        nargs='+',
        default=['H4'],
        help=f"Patient label(s) or 'all'. Available: {list(PATIENT_DATA.keys())}"
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
        default=8192,
        help='Training batch size'
    )
    train_parser.add_argument(
        '--lr',
        type=float,
        default=2e-4,
        help='Initial learning rate'
    )
    train_parser.add_argument(
        '--patience',
        type=int,
        default=100,
        help='Early stopping patience'
    )
    train_parser.add_argument(
        '--num-collocation-points',
        type=int,
        default=4096,
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
        '--hidden-dim',
        type=int,
        default=48,
        help='Hidden layer dimension'
    )
    train_parser.add_argument(
        '--num-blocks',
        type=int,
        default=6,
        help='Number of residual blocks'
    )
    train_parser.add_argument(
        '--num-frequencies',
        type=int,
        default=64,
        help='Number of Fourier frequencies'
    )
    train_parser.add_argument(
        '--fourier-scale',
        type=float,
        default=10.0,
        help='Fourier frequency scale (sigma)'
    )

    # Rheology selection (newtonian / carreau_yasuda)
    train_parser.add_argument(
        '--rheology',
        type=str,
        choices=['newtonian', 'carreau_yasuda'],
        default=None,
        help='Override the rheology used in the physics loss (default: src/config.py)'
    )

    # Spatial holdout (Physics of Fluids R1-5 / R2-6)
    train_parser.add_argument(
        '--holdout-fraction',
        type=float,
        default=0.20,
        help='Fraction of mesh points withheld per patient for evaluation '
             '(default: 0.20). 0 disables the split.'
    )
    train_parser.add_argument(
        '--holdout-seed',
        type=int,
        default=0,
        help='Seed for the per-patient spatial holdout split (default: 0).'
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
    print("FOURIERPINN TRAINING FOR CORONARY ARTERY WSS PREDICTION")
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

    # Apply CLI rheology override before training so train_patient picks it up.
    if getattr(args, 'rheology', None) is not None:
        import src.config as _cfg
        _cfg.RHEOLOGY = args.rheology
        print(f"Rheology override: {_cfg.RHEOLOGY}")

    # Guard against the AE-9 contradiction: a carreau_yasuda physics loss is
    # only valid against patients whose CFD ground truth is also Carreau-
    # Yasuda. Filter the patient list against PATIENT_DATA[label]['carreau_yasuda'].
    import src.config as _cfg_chk
    if _cfg_chk.RHEOLOGY == "carreau_yasuda":
        cy_available = set(_cfg_chk.CY_AVAILABLE_LABELS)
        missing = [p for p in patients if p not in cy_available]
        if missing:
            raise SystemExit(
                f"ERROR: --rheology carreau_yasuda requested for patient(s) "
                f"{missing}, but no Carreau-Yasuda CFD ground truth exists for "
                f"them. Patients with CY data: {sorted(cy_available)}. "
                f"Either re-run with --rheology newtonian or restrict "
                f"--patient to the CY-eligible subset."
            )

    print(f"Patients to train: {patients}")
    print(f"Architecture: FourierPINN ({args.num_blocks} blocks, {args.hidden_dim} hidden)")

    # Train each patient. Failures propagate so real bugs surface immediately
    # instead of silently skipping a patient and continuing.
    results = {}
    for pid in patients:
        model, result = train_patient(
            patient_id=pid,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            num_collocation_points=args.num_collocation_points,
            patience=args.patience,
            hidden_dim=args.hidden_dim,
            num_blocks=args.num_blocks,
            grad_clip=args.grad_clip,
            num_frequencies=args.num_frequencies,
            fourier_scale=args.fourier_scale,
            holdout_fraction=args.holdout_fraction,
            holdout_seed=args.holdout_seed,
            verbose=args.verbose
        )
        results[pid] = result

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
