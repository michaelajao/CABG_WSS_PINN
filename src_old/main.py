"""
Physics-Informed Neural Networks (PINNs) for Wall Shear Stress Prediction
Main CLI Entry Point

This script provides a command-line interface for training, evaluating, and
making predictions with the PINN model.

Usage:
    python main.py train [options]
    python main.py evaluate [options]
    python main.py eda [options]

Author: Research Team
Date: November 8, 2025
"""

import argparse
import numpy as np
import pandas as pd
import sys
from pathlib import Path

from src_old import config
from src_old.model import PINN, ResNetPINN
from src_old.dataset import (
    load_and_prepare_data, create_dataloaders,
    load_patient_level_data, create_patient_level_dataloaders
)
from src_old.train import train_pinn, load_checkpoint
from src_old.evaluate import (
    evaluate_model,
    save_evaluation_results,
    save_metrics_text,
    predict_on_new_data,
    compute_and_save_per_case_nrmse
)
from src_old.plots import (
    plot_training_history,
    plot_prediction_results,
    create_comparison_3d,
    plot_plane_comparison,
    plot_streamline_comparison
)


def train_command(args):
    """Execute training command"""
    print("\n" + "="*80)
    print("PHYSICS-INFORMED NEURAL NETWORK - TRAINING")
    print("="*80)

    # Override config with CLI arguments if provided
    if args.device:
        import torch
        config.DEVICE = torch.device(args.device)

    if args.seed is not None:
        import torch
        config.RANDOM_SEED = args.seed
        torch.manual_seed(args.seed)

    # Print configuration
    if args.verbose:
        print(f"\n[DEVICE CONFIGURATION]")
        config.get_device_info()

    # Suppress verbose diff-style hyperparameter printing; keep output lean

    # Parse patient selection from CLI arguments
    train_patients = None
    val_patients = None
    test_patients = None

    if hasattr(args, 'train_patients') and args.train_patients:
        train_patients = [p.strip() for p in args.train_patients.split(',')]
    if hasattr(args, 'val_patients') and args.val_patients:
        val_patients = [p.strip() for p in args.val_patients.split(',')]
    if hasattr(args, 'test_patients') and args.test_patients:
        test_patients = [p.strip() for p in args.test_patients.split(',')]

    # Load data with patient-level splitting (prevents spatial data leakage)
    # Use hybrid loading if streamlines are requested
    if args.use_streamlines:
        from src_old.dataset import load_patient_level_hybrid_data
        train_data, val_data, test_data = load_patient_level_hybrid_data(
            train_patients=train_patients,
            val_patients=val_patients,
            test_patients=test_patients,
            verbose=args.verbose
        )
    else:
        train_data, val_data, test_data = load_patient_level_data(
            train_patients=train_patients,
            val_patients=val_patients,
            test_patients=test_patients,
            verbose=args.verbose
        )

    # Create data loaders with proper scaler fitting
    train_loader, val_loader, test_loader, full_dataset = create_patient_level_dataloaders(
        train_data, val_data, test_data,
        batch_size=args.batch_size,
        verbose=args.verbose
    )

    # Initialize model
    # Removed architecture header for cleaner output

    # Build layers from CLI if provided
    def parse_hidden(arg: str):
        return [int(x) for x in arg.split(',') if x.strip()]

    hidden_layers = None
    if getattr(args, 'hidden', None):
        hidden_layers = parse_hidden(args.hidden)
    elif getattr(args, 'depth', None) and getattr(args, 'width', None):
        hidden_layers = [args.width] * args.depth

    if args.arch == 'resnet':
        model = ResNetPINN(
            width=args.res_width,
            blocks=args.res_blocks,
            activation=args.activation,
            activation_beta=args.activation_beta
        ).to(config.DEVICE)
    else:
        if hidden_layers:
            layers = [3] + hidden_layers + [5]
            model = PINN(layers=layers, activation=args.activation, activation_beta=args.activation_beta).to(config.DEVICE)
        else:
            model = PINN(activation=args.activation, activation_beta=args.activation_beta).to(config.DEVICE)
    
    if args.init_method:
        from src_old.model import initialize_weights
        initialize_weights(model, args.init_method)  # Silent initialization
    
    model.print_architecture()

    # Train model
    history = train_pinn(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        coord_scaler=full_dataset.scaler_X,  # Pass scaler for physics chain rule
        train_data_dict=train_data,  # Pass for collocation point sampling
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        physics_weight_nse=args.nse_weight,
        physics_weight_cont=args.cont_weight,
        data_weight=args.data_weight,
        velocity_weight=args.velocity_weight,
        use_collocation=args.use_collocation if hasattr(args, 'use_collocation') else config.USE_COLLOCATION_POINTS,
        n_collocation=args.n_collocation if hasattr(args, 'n_collocation') else config.COLLOCATION_POINTS_PER_BATCH,
        verbose=args.verbose,
        init_method=args.init_method
    )

    # Plot training history
    if args.plot:
        plot_training_history()

    # Automatically run evaluation and visualization after training
    print("\n" + "="*80)
    print("RUNNING EVALUATION")
    print("="*80)

    # Load best model for evaluation
    info = load_checkpoint(model, config.MODEL_PATH / "best_pinn_model.pth")

    # Evaluate model
    metrics, results_df = evaluate_model(
        model=model,
        test_loader=test_loader,
        dataset=full_dataset,
        verbose=args.verbose
    )

    # Save results
    print("\n[SAVING RESULTS]")
    save_evaluation_results(metrics, results_df)
    save_metrics_text(metrics, results_df)

    # Generate visualizations
    if args.plot:
        print("\n[GENERATING VISUALIZATIONS]")

        plot_prediction_results(results_df)

        # Predict on full CFD grid (filter to wall points only for visualization)
        # full_dataset.y contains NaN for interior points without WSS
        wall_mask = ~np.isnan(full_dataset.y)
        coords_full = full_dataset.X[wall_mask]
        y_true_full = full_dataset.y[wall_mask]
        
        y_pred_full = predict_on_new_data(
            model,
            coords=coords_full,
            scaler_X=full_dataset.scaler_X,
            scaler_y=full_dataset.scaler_y,
            batch_size=8192
        )
        results_full_df = pd.DataFrame({
            'X': coords_full[:,0],
            'Y': coords_full[:,1],
            'Z': coords_full[:,2],
            'WSS_True': y_true_full,
            'WSS_Pred': y_pred_full,
        })

        # 3D visualizations
        create_comparison_3d(results_full_df)

        # Compute and save full-field NRMSE
        yt = results_full_df['WSS_True'].values
        yp = results_full_df['WSS_Pred'].values
        rmse_full = float(np.sqrt(np.mean((yp - yt) ** 2)))
        rng = float(np.max(yt) - np.min(yt) + 1e-12)
        nrmse_full = rmse_full / rng
        with open(config.RESULTS_PATH / 'metrics_summary.txt', 'a') as f:
            f.write("\n\nFull-field Comparison\n")
            f.write(f"NRMSE_full\t{nrmse_full:.6f}\n")

        # Plane comparisons (XY, XZ, YZ)
        plot_plane_comparison(results_full_df, patient_name='Evaluation', plane='XY')
        plot_plane_comparison(results_full_df, patient_name='Evaluation', plane='XZ')
        plot_plane_comparison(results_full_df, patient_name='Evaluation', plane='YZ')

        # Per-case NRMSE metrics
        compute_and_save_per_case_nrmse(
            model,
            scaler_X=full_dataset.scaler_X,
            scaler_y=full_dataset.scaler_y,
            save_path=config.RESULTS_PATH
        )

    # Summary
    print("\n" + "="*80)
    print("TRAINING AND EVALUATION COMPLETE")
    print("="*80)
    print(f"\nResults Summary:")
    print(f"  MAE: {metrics['MAE']:.4f} Pa")
    print(f"  RMSE: {metrics['RMSE']:.4f} Pa")
    print(f"  R2: {metrics['R2']:.4f}")
    print(f"\n  High-Risk Classification (>{config.HIGH_RISK_WSS_THRESHOLD} Pa):")
    print(f"    Accuracy: {metrics['High_Risk_Accuracy']:.4f}")
    print(f"    Precision: {metrics['High_Risk_Precision']:.4f}")
    print(f"    Recall: {metrics['High_Risk_Recall']:.4f}")
    print(f"    F1-Score: {metrics['High_Risk_F1']:.4f}")
    print("="*80)

    return model, history, test_loader, full_dataset


def evaluate_command(args):
    """Execute evaluation command"""
    print("\n" + "="*80)
    print("PINN MODEL EVALUATION")
    print("="*80)

    # Override device if specified
    if args.device:
        import torch
        config.DEVICE = torch.device(args.device)

    # Load checkpoint first to auto-detect architecture
    checkpoint_path = Path(args.model_path) if args.model_path else config.MODEL_PATH / "best_pinn_model.pth"

    if not checkpoint_path.exists():
        print(f"\nError: No trained model found at {checkpoint_path}")
        print(f"Please run: python {Path(__file__).name} train")
        sys.exit(1)

    print("\n[LOADING MODEL]")
    import torch
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Auto-detect architecture from checkpoint metadata
    if 'model_config' in checkpoint:
        model_config = checkpoint['model_config']
        arch = model_config['arch']
        activation = model_config.get('activation', 'tanh')
        activation_beta = model_config.get('activation_beta', None)

        # Build model based on saved configuration
        if arch == 'resnet':
            model = ResNetPINN(
                width=model_config['res_width'],
                blocks=model_config['res_blocks'],
                activation=activation,
                activation_beta=activation_beta
            ).to(config.DEVICE)
        else:  # pinn
            layers = model_config['layers']
            model = PINN(
                layers=layers,
                activation=activation,
                activation_beta=activation_beta
            ).to(config.DEVICE)
    else:
        # Fallback: Use CLI arguments (legacy checkpoints)
        print("  [WARNING] Using CLI arguments (legacy checkpoint)")

        def parse_hidden(arg: str):
            return [int(x) for x in arg.split(',') if x.strip()]

        hidden_layers = None
        if getattr(args, 'hidden', None):
            hidden_layers = parse_hidden(args.hidden)
        elif getattr(args, 'depth', None) and getattr(args, 'width', None):
            hidden_layers = [args.width] * args.depth

        if args.arch == 'resnet':
            model = ResNetPINN(
                width=args.res_width,
                blocks=args.res_blocks,
                activation=args.activation,
                activation_beta=args.activation_beta
            ).to(config.DEVICE)
        else:
            if hidden_layers:
                layers = [3] + hidden_layers + [5]
                model = PINN(layers=layers, activation=args.activation, activation_beta=args.activation_beta).to(config.DEVICE)
            else:
                model = PINN(activation=args.activation, activation_beta=args.activation_beta).to(config.DEVICE)

        if args.init_method:
            from src_old.model import initialize_weights
            initialize_weights(model, args.init_method)

    info = load_checkpoint(model, checkpoint_path)
    print(f"  Loaded: Epoch {info['epoch']+1} | Val Loss: {info['val_loss']:.6f}")

    # Load test data with patient-level splitting
    print("\n[LOADING DATA]")

    train_data, val_data, test_data = load_patient_level_data(verbose=False)
    train_loader, val_loader, test_loader, full_dataset = create_patient_level_dataloaders(
        train_data, val_data, test_data,
        verbose=False
    )

    print(f"  Test samples: {len(test_loader.dataset):,}")

    # Evaluate model
    metrics, results_df = evaluate_model(
        model=model,
        test_loader=test_loader,
        dataset=full_dataset,
        verbose=args.verbose
    )

    # Save results
    print("\n[SAVING RESULTS]")

    save_evaluation_results(metrics, results_df)
    # Save human-readable metrics TXT (includes NRMSE on test set)
    save_metrics_text(metrics, results_df)

    # Generate visualizations
    if args.plot:
        print("\n[GENERATING VISUALIZATIONS]")

        plot_training_history()
        plot_prediction_results(results_df)

        # Predict on full CFD grid
        coords_full = full_dataset.X
        y_true_full = full_dataset.y
        y_pred_full = predict_on_new_data(
            model,
            coords=coords_full,
            scaler_X=full_dataset.scaler_X,
            scaler_y=full_dataset.scaler_y,
            batch_size=8192
        )
        results_full_df = __import__('pandas').DataFrame({
            'X': coords_full[:,0],
            'Y': coords_full[:,1],
            'Z': coords_full[:,2],
            'WSS_True': y_true_full,
            'WSS_Pred': y_pred_full,
        })

        # 3D visualizations
        create_comparison_3d(results_full_df)

        # Compute and save full-field NRMSE
        yt = results_full_df['WSS_True'].values
        yp = results_full_df['WSS_Pred'].values
        rmse_full = float(np.sqrt(np.mean((yp - yt) ** 2)))
        rng = float(np.max(yt) - np.min(yt) + 1e-12)
        nrmse_full = rmse_full / rng
        with open(config.RESULTS_PATH / 'metrics_summary.txt', 'a') as f:
            f.write("\n\nFull-field Comparison\n")
            f.write(f"NRMSE_full\t{nrmse_full:.6f}\n")

        # Plane cuts (XY, XZ, YZ)
        plot_plane_comparison(results_full_df, patient_name='Evaluation', plane='XY')
        plot_plane_comparison(results_full_df, patient_name='Evaluation', plane='XZ')
        plot_plane_comparison(results_full_df, patient_name='Evaluation', plane='YZ')

        # Streamline comparison if available
        stream_files = [f for f in config.PINNS_PATH.glob('*Streamlines*.csv')]
        if stream_files:
            cfd_stream_df = pd.read_csv(stream_files[0], skiprows=5)
            plot_streamline_comparison(cfd_stream_df, results_full_df, patient_name='Evaluation')

        # Per-case NRMSE metrics
        compute_and_save_per_case_nrmse(
            model,
            scaler_X=full_dataset.scaler_X,
            scaler_y=full_dataset.scaler_y,
            save_path=config.RESULTS_PATH
        )

    # Summary
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)

    print(f"\nResults Summary:")
    print(f"  MAE: {metrics['MAE']:.4f} Pa")
    print(f"  RMSE: {metrics['RMSE']:.4f} Pa")
    print(f"  R2: {metrics['R2']:.4f}")
    print(f"\n  High-Risk Classification (>{config.HIGH_RISK_WSS_THRESHOLD} Pa):")
    print(f"    Accuracy: {metrics['High_Risk_Accuracy']:.4f}")
    print(f"    Precision: {metrics['High_Risk_Precision']:.4f}")
    print(f"    Recall: {metrics['High_Risk_Recall']:.4f}")
    print(f"    F1-Score: {metrics['High_Risk_F1']:.4f}")

    print(f"\nGenerated Files:")
    print(f"  Results:")
    print(f"    - {config.RESULTS_PATH / 'evaluation_metrics.json'}")
    print(f"    - {config.RESULTS_PATH / 'test_predictions.csv'}")
    if args.plot:
        print(f"\n  Visualizations:")
        print(f"    - {config.RESULTS_PATH / 'training_history.png'}")
        print(f"    - {config.RESULTS_PATH / 'prediction_results.png'}")
        print(f"    - {config.RESULTS_PATH / 'spatial_error_distribution.html'}")
        print(f"    - {config.RESULTS_PATH / 'comparison_3d.html'}")
        print(f"    - {config.RESULTS_PATH / 'metrics_summary.txt'}")
        print(f"    - {config.RESULTS_PATH / 'per_case_nrmse.txt'}")
        print(f"    - {config.FIGURES_PATH / 'Evaluation_XY_plane_comparison.png'}")
        print(f"    - {config.FIGURES_PATH / 'Evaluation_XZ_plane_comparison.png'}")
        print(f"    - {config.FIGURES_PATH / 'Evaluation_streamline_comparison.png'}")

    print("\n" + "="*80)

    return model, metrics, results_df


def eda_command(args):
    """Execute exploratory data analysis command"""
    print("="*80)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*80)

    # Import the EDA script
    import exploratory_data_analysis as eda_module

    # Run EDA
    df_summary, stats = eda_module.main()

    print("\nEDA complete!")
    return df_summary, stats


def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Physics-Informed Neural Networks for Wall Shear Stress Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default settings
  python main.py train

  # Train with custom settings
  python main.py train --epochs 300 --batch-size 4096 --learning-rate 5e-5

  # Evaluate trained model
  python main.py evaluate

  # Run exploratory data analysis
  python main.py eda

For more information, see CLAUDE.md
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    subparsers.required = True

    # =========================================================================
    # TRAIN command
    # =========================================================================
    train_parser = subparsers.add_parser('train', help='Train PINN model')

    train_parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=config.EPOCHS,
        help=f'Number of training epochs (default: {config.EPOCHS})'
    )

    train_parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=config.BATCH_SIZE,
        help=f'Batch size (default: {config.BATCH_SIZE})'
    )

    train_parser.add_argument(
        '--learning-rate', '-lr',
        type=float,
        default=config.LEARNING_RATE,
        help=f'Learning rate (default: {config.LEARNING_RATE})'
    )

    train_parser.add_argument(
        '--data-weight',
        type=float,
        default=config.DATA_WEIGHT,
        help=f'Weight for data fitting loss (default: {config.DATA_WEIGHT})'
    )

    train_parser.add_argument(
        '--nse-weight',
        type=float,
        default=config.PHYSICS_WEIGHT_NSE,
        help=f'Weight for Navier-Stokes loss (default: {config.PHYSICS_WEIGHT_NSE})'
    )

    train_parser.add_argument(
        '--cont-weight',
        type=float,
        default=config.PHYSICS_WEIGHT_CONT,
        help=f'Weight for continuity loss (default: {config.PHYSICS_WEIGHT_CONT})'
    )

    train_parser.add_argument(
        '--velocity-weight',
        type=float,
        default=config.VELOCITY_WEIGHT,
        help=f'Weight for velocity supervision loss (default: {config.VELOCITY_WEIGHT})'
    )

    train_parser.add_argument(
        '--use-streamlines',
        action='store_true',
        default=True,
        help='Use streamline (interior velocity) data in addition to wall data (default: True)'
    )

    train_parser.add_argument(
        '--no-streamlines',
        dest='use_streamlines',
        action='store_false',
        help='Only use wall data (disable streamline loading)'
    )

    train_parser.add_argument(
        '--use-collocation',
        action='store_true',
        default=config.USE_COLLOCATION_POINTS,
        help=f'Use collocation points for physics enforcement (default: {config.USE_COLLOCATION_POINTS})'
    )

    train_parser.add_argument(
        '--no-collocation',
        dest='use_collocation',
        action='store_false',
        help='Disable collocation point sampling'
    )

    train_parser.add_argument(
        '--n-collocation',
        type=int,
        default=config.COLLOCATION_POINTS_PER_BATCH,
        help=f'Number of collocation points per batch (default: {config.COLLOCATION_POINTS_PER_BATCH})'
    )

    train_parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu'],
        help='Device to use for training (default: auto-detect)'
    )

    train_parser.add_argument(
        '--seed',
        type=int,
        default=config.RANDOM_SEED,
        help=f'Random seed for reproducibility (default: {config.RANDOM_SEED})'
    )

    train_parser.add_argument(
        '--plot',
        action='store_true',
        default=True,
        help='Generate training history plot (default: True)'
    )

    train_parser.add_argument(
        '--no-plot',
        dest='plot',
        action='store_false',
        help='Skip training history plot'
    )

    train_parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        default=True,
        help='Verbose output (default: True)'
    )

    # Patient selection options (for custom train/val/test splits)
    train_parser.add_argument(
        '--train-patients',
        type=str,
        help='Comma-separated list of patient IDs for training (e.g., H-09_Healthy,H-12_Healthy)'
    )
    train_parser.add_argument(
        '--val-patients',
        type=str,
        help='Comma-separated list of patient IDs for validation'
    )
    train_parser.add_argument(
        '--test-patients',
        type=str,
        help='Comma-separated list of patient IDs for testing'
    )

    # Architecture options
    train_parser.add_argument(
        '--hidden',
        type=str,
        help='Comma-separated hidden layer sizes, e.g., 128,256,256,128'
    )
    train_parser.add_argument(
        '--depth',
        type=int,
        help='Number of hidden layers (used with --width)'
    )
    train_parser.add_argument(
        '--width',
        type=int,
        help='Hidden layer width (used with --depth)'
    )
    train_parser.add_argument(
        '--activation',
        type=str,
        default='tanh',
        choices=['tanh','relu','gelu','silu','swish','leaky_relu','swish_learnable','lswish'],
        help='Activation function (default: tanh)'
    )
    train_parser.add_argument(
        '--activation-beta',
        type=float,
        help='Initial beta for learnable Swish (used when activation=swish or lswish)'
    )
    train_parser.add_argument(
        '--init-method',
        type=str,
        default=None,
        choices=['xavier_normal', 'xavier_uniform', 'kaiming_normal', 'kaiming_uniform'],
        help='Weight initialization method (default: PyTorch defaults)'
    )
    train_parser.add_argument(
        '--arch',
        type=str,
        default='pinn',
        choices=['pinn','resnet'],
        help='Model architecture: standard pinn or resnet (default: pinn)'
    )
    train_parser.add_argument(
        '--res-blocks',
        type=int,
        default=6,
        help='Number of residual blocks (ResNetPINN only)'
    )
    train_parser.add_argument(
        '--res-width',
        type=int,
        default=256,
        help='Hidden width for residual blocks (ResNetPINN only)'
    )

    train_parser.set_defaults(func=train_command)

    # =========================================================================
    # EVALUATE command
    # =========================================================================
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate trained PINN model')

    eval_parser.add_argument(
        '--model-path', '-m',
        type=str,
        help=f'Path to trained model checkpoint (default: {config.MODEL_PATH / "best_pinn_model.pth"})'
    )

    eval_parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu'],
        help='Device to use for evaluation (default: auto-detect)'
    )

    eval_parser.add_argument(
        '--plot',
        action='store_true',
        default=True,
        help='Generate visualization plots (default: True)'
    )

    eval_parser.add_argument(
        '--no-plot',
        dest='plot',
        action='store_false',
        help='Skip visualization plots'
    )

    eval_parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        default=True,
        help='Verbose output (default: True)'
    )

    # Architecture options (must match the checkpoint used)
    eval_parser.add_argument(
        '--hidden',
        type=str,
        help='Comma-separated hidden layer sizes used in training (e.g., 128,256,256,128)'
    )
    eval_parser.add_argument(
        '--depth',
        type=int,
        help='Number of hidden layers (used with --width)'
    )
    eval_parser.add_argument(
        '--width',
        type=int,
        help='Hidden layer width (used with --depth)'
    )
    eval_parser.add_argument(
        '--activation',
        type=str,
        default='tanh',
        choices=['tanh','relu','gelu','silu','swish','leaky_relu','swish_learnable','lswish'],
        help='Activation function used in training (default: tanh)'
    )
    eval_parser.add_argument(
        '--activation-beta',
        type=float,
        help='Initial beta for learnable Swish (match training if used)'
    )
    eval_parser.add_argument(
        '--init-method',
        type=str,
        default=None,
        choices=['xavier_normal', 'xavier_uniform', 'kaiming_normal', 'kaiming_uniform'],
        help='Weight initialization method (must match training if used)'
    )
    eval_parser.add_argument(
        '--arch',
        type=str,
        default='pinn',
        choices=['pinn','resnet'],
        help='Model architecture used during training'
    )
    eval_parser.add_argument(
        '--res-blocks',
        type=int,
        default=6,
        help='Residual block count (ResNetPINN only)'
    )
    eval_parser.add_argument(
        '--res-width',
        type=int,
        default=256,
        help='Residual block width (ResNetPINN only)'
    )

    eval_parser.set_defaults(func=evaluate_command)

    # =========================================================================
    # EDA command
    # =========================================================================
    eda_parser = subparsers.add_parser('eda', help='Run exploratory data analysis')

    eda_parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        default=True,
        help='Verbose output (default: True)'
    )

    eda_parser.set_defaults(func=eda_command)

    # Parse arguments and execute command
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    main()
