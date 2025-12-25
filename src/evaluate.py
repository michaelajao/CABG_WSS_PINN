"""
Evaluation Module for PINN Model Assessment.

This module provides functions and classes to evaluate trained PINN models
against CFD ground truth data.

Components:
    - evaluate_model: Quick single-case evaluation.
    - PINNValidator: Comprehensive validation with batch processing and reporting.

Metrics Computed:
    - RMSE: Root Mean Squared Error (Pa)
    - MAE: Mean Absolute Error (Pa)
    - NRMSE: Normalized RMSE (RMSE / data range)
    - R²: Coefficient of Determination

Attributes:
    DEFAULT_HIDDEN_DIM (int): Default hidden dimension for model loading.
    DEFAULT_NUM_BLOCKS (int): Default number of blocks for model loading.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader

from src.config import (
    DATA_PATH,
    DEVICE,
    FIGURES_PATH,
    MODELS_PATH,
    PATIENT_DATA,
    RESULTS_PATH,
)
from src.dataset import PatientDataset, load_patient_data, load_vessel_data
from src.utils import compute_nrmse

# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_HIDDEN_DIM: int = 256
DEFAULT_NUM_BLOCKS: int = 4


# =============================================================================
# QUICK EVALUATION
# =============================================================================

def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    dataset: PatientDataset,
    coord_scale: torch.Tensor
) -> Dict:
    """
    Evaluate trained PINN model on WSS prediction.

    This function performs a quick evaluation of a trained model against
    ground truth CFD data, computing standard regression metrics.

    Args:
        model: Trained PINN model in evaluation mode.
        loader: DataLoader with evaluation data.
        dataset: PatientDataset instance (needed for inverse scaling).
        coord_scale: Scale factors for gradient computation with shape (1, 3).

    Returns:
        Dictionary containing evaluation metrics:
            - RMSE: Root mean squared error in Pa.
            - MAE: Mean absolute error in Pa.
            - NRMSE: Normalized RMSE (fraction of data range).
            - R2: Coefficient of determination.
            - n_points: Number of evaluation points.

    Example:
        >>> metrics = evaluate_model(model, test_loader, dataset, coord_scale)
        >>> print(f"R²: {metrics['R2']:.4f}")
    """
    model.eval()

    all_true: List[np.ndarray] = []
    all_pred: List[np.ndarray] = []
    all_coords: List[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            coords = batch['coords'].to(DEVICE)
            coords_raw = batch['coords_raw'].numpy()
            wss_raw = batch['wss_raw'].numpy().flatten()
            has_wss = batch['has_wss'].numpy().squeeze().astype(bool)

            # Get predicted WSS
            outputs = model(coords)
            wss_pred_scaled = outputs['wss'].cpu().numpy()
            wss_pred = dataset.scaler_y.inverse_transform(wss_pred_scaled).flatten()
            wss_pred = wss_pred[has_wss]

            if has_wss.any():
                all_true.append(wss_raw[has_wss])
                all_pred.append(wss_pred)
                all_coords.append(coords_raw[has_wss])

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)

    # Compute metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    nrmse = compute_nrmse(y_true, y_pred)

    metrics = {
        'RMSE': float(rmse),
        'MAE': float(mae),
        'R2': float(r2),
        'NRMSE': float(nrmse),
        'n_points': int(len(y_true))
    }

    print(f"  RMSE: {rmse:.4f} Pa | NRMSE: {nrmse:.4f} | R²: {r2:.4f}")

    return metrics


# =============================================================================
# PINN VALIDATOR CLASS
# =============================================================================

class PINNValidator:
    """
    Comprehensive PINN Validation and Visualization Engine.

    Handles the complete validation pipeline including:
    - Model evaluation on test data
    - Error metric computation (MAE, RMSE, NRMSE, R²)
    - Per-vessel and full-patient visualization
    - Batch validation across multiple patients
    - Results export to CSV for analysis

    Attributes:
        model (nn.Module): Trained PINN model.
        patient_id (str): Patient identifier.
        device (torch.device): Computation device.
        data (dict): Loaded patient data.
        per_vessel (dict): Per-vessel data dictionary.
        dataset (PatientDataset): Dataset instance for scaling.
        results (dict): Validation results storage.

    Example:
        >>> from src.model import VanillaPINN
        >>> model = VanillaPINN()
        >>> model.load_state_dict(torch.load('model.pth')['model_state_dict'])
        >>> validator = PINNValidator(model, patient_id='0073')
        >>> results = validator.validate()
        >>> validator.generate_full_patient_plots()
    """

    def __init__(
        self,
        model: nn.Module,
        patient_id: str,
        device: Optional[str] = None
    ) -> None:
        """
        Initialize the validator.

        Args:
            model: Trained PINN model (already loaded with weights).
            patient_id: Patient identifier (e.g., '0073', 'H-12').
            device: Device for computation ('cuda' or 'cpu').
                Auto-detected if None.

        Raises:
            KeyError: If patient_id is not found in PATIENT_DATA.
        """
        self.model = model
        self.patient_id = patient_id
        self.device = device if device else DEVICE
        self.model.to(self.device)
        self.model.eval()

        # Load patient data
        self.data, self.per_vessel = load_patient_data(patient_id)
        self.dataset = PatientDataset(self.data)

        # Results storage
        self.results: Dict = {
            'patient_id': patient_id,
            'category': PATIENT_DATA.get(patient_id, {}).get('category', 'Unknown'),
            'timestamp': datetime.now().isoformat(),
            'wss_metrics': {},
            'velocity_metrics': {},
            'per_vessel_metrics': {}
        }

        # Predictions cache
        self._predictions: Optional[Dict[str, np.ndarray]] = None

    def predict(self, coords: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Run model prediction on given coordinates.

        Args:
            coords: Raw coordinates with shape (N, 3) in meters.

        Returns:
            Dictionary with predictions (all denormalized):
                - wss: Wall shear stress in Pa.
                - u, v, w: Velocity components in m/s.
                - p: Pressure (normalized scale).
        """
        # Scale coordinates
        coords_scaled = self.dataset.scaler_X.transform(coords)
        coords_tensor = torch.FloatTensor(coords_scaled).to(self.device)

        with torch.no_grad():
            outputs = self.model(coords_tensor)

        # Denormalize outputs
        wss_pred = self.dataset.scaler_y.inverse_transform(
            outputs['wss'].cpu().numpy()
        ).flatten()

        vel_pred = np.column_stack([
            outputs['u'].cpu().numpy(),
            outputs['v'].cpu().numpy(),
            outputs['w'].cpu().numpy()
        ])
        vel_pred = self.dataset.scaler_vel.inverse_transform(vel_pred)

        return {
            'wss': wss_pred,
            'u': vel_pred[:, 0],
            'v': vel_pred[:, 1],
            'w': vel_pred[:, 2],
            'p': outputs['p'].cpu().numpy().flatten()
        }

    def validate(self, verbose: bool = True) -> Dict:
        """
        Run full validation on the patient data.

        Computes metrics for:
        - WSS prediction (wall points only)
        - Velocity magnitude (all points)
        - Per-vessel WSS metrics

        Args:
            verbose: If True, print results to console.

        Returns:
            Dictionary containing all validation metrics with keys:
                - patient_id: Patient identifier.
                - category: Patient category.
                - timestamp: Validation timestamp.
                - wss_metrics: WSS evaluation metrics.
                - velocity_metrics: Velocity evaluation metrics.
                - per_vessel_metrics: Per-vessel WSS metrics.
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Validating Patient: {self.patient_id}")
            print(f"{'='*60}")

        # Get predictions for all data
        predictions = self.predict(self.data['X'])
        self._predictions = predictions

        # WSS metrics (wall points only)
        has_wss = self.data['has_wss']
        wss_true = self.data['y'][has_wss]
        wss_pred = predictions['wss'][has_wss]

        self.results['wss_metrics'] = self._compute_metrics(
            wss_true, wss_pred, 'WSS'
        )

        # Velocity metrics (all points, compute magnitude)
        vel_true = self.data['velocity']
        vel_pred = np.column_stack([
            predictions['u'],
            predictions['v'],
            predictions['w']
        ])

        vel_mag_true = np.linalg.norm(vel_true, axis=1)
        vel_mag_pred = np.linalg.norm(vel_pred, axis=1)

        self.results['velocity_metrics'] = self._compute_metrics(
            vel_mag_true, vel_mag_pred, 'Velocity'
        )

        # Per-vessel metrics
        if verbose:
            print("\nPer-Vessel WSS Metrics:")

        for vessel_name, vessel_data in self.per_vessel.items():
            if vessel_name.lower() == 'aorta':
                continue  # Skip aorta (no specific model)

            vessel_has_wss = vessel_data['has_wss']
            if not vessel_has_wss.any():
                continue

            vessel_wss_true = vessel_data['y'][vessel_has_wss]

            # Get predictions for this vessel's coordinates
            vessel_preds = self.predict(vessel_data['X'])
            vessel_wss_pred = vessel_preds['wss'][vessel_has_wss]

            metrics = self._compute_metrics(
                vessel_wss_true, vessel_wss_pred, vessel_name, verbose=False
            )
            self.results['per_vessel_metrics'][vessel_name] = metrics

            if verbose:
                print(
                    f"  {vessel_name}: RMSE={metrics['RMSE']:.4f} Pa, "
                    f"NRMSE={metrics['NRMSE']:.4f}, R²={metrics['R2']:.4f}"
                )

        return self.results

    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        name: str,
        verbose: bool = True
    ) -> Dict:
        """
        Compute standard regression metrics.

        Args:
            y_true: Ground truth values.
            y_pred: Predicted values.
            name: Name of the quantity being evaluated.
            verbose: If True, print metrics to console.

        Returns:
            Dictionary with RMSE, MAE, R2, NRMSE, and n_points.
        """
        # Filter out NaN values
        valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        nrmse = compute_nrmse(y_true, y_pred)

        metrics = {
            'RMSE': float(rmse),
            'MAE': float(mae),
            'R2': float(r2),
            'NRMSE': float(nrmse),
            'n_points': int(len(y_true))
        }

        if verbose:
            print(f"\n{name} Metrics:")
            print(f"  RMSE:  {rmse:.4f}")
            print(f"  MAE:   {mae:.4f}")
            print(f"  NRMSE: {nrmse:.4f}")
            print(f"  R²:    {r2:.4f}")
            print(f"  Points: {len(y_true):,}")

        return metrics

    def generate_full_patient_plots(self, save_dir: Optional[Path] = None) -> None:
        """
        Generate full patient anatomy visualization with all vessels combined.

        Creates multi-view plots showing CFD ground truth, PINN predictions,
        and error distributions for the complete patient anatomy.

        Args:
            save_dir: Directory to save figures.
                Defaults to figures/{patient_id}/.
        """
        from src.plots import plot_full_patient_wss

        if save_dir is None:
            save_dir = FIGURES_PATH / self.patient_id
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nGenerating full patient plots for {self.patient_id}...")

        # Prepare vessel data for plotting
        vessel_data: List[Dict] = []

        for vessel_name, vdata in self.per_vessel.items():
            if vessel_name.lower() == 'aorta':
                continue

            has_wss = vdata['has_wss']
            if not has_wss.any():
                continue

            # Get predictions
            preds = self.predict(vdata['X'])

            vessel_entry = {
                'name': vessel_name,
                'coords': vdata['X'][has_wss],
                'wss_true': vdata['y'][has_wss],
                'wss_pred': preds['wss'][has_wss]
            }

            # Add streamline data if available
            interior_mask = ~has_wss
            if interior_mask.any():
                vel_true = vdata['velocity'][interior_mask]
                vel_mag_true = np.linalg.norm(vel_true, axis=1)

                vel_pred = np.column_stack([
                    preds['u'][interior_mask],
                    preds['v'][interior_mask],
                    preds['w'][interior_mask]
                ])
                vel_mag_pred = np.linalg.norm(vel_pred, axis=1)

                vessel_entry['stream_coords'] = vdata['X'][interior_mask]
                vessel_entry['vel_true'] = vel_mag_true
                vessel_entry['vel_pred'] = vel_mag_pred

            vessel_data.append(vessel_entry)

        # Get aorta coordinates for background
        aorta_data = self.per_vessel.get('Aorta', None)
        df_aorta = aorta_data['X'] if aorta_data is not None else None

        # Generate plots
        plot_full_patient_wss(self.patient_id, vessel_data, df_aorta, save_dir)

    def save_results(self, save_dir: Optional[Path] = None) -> None:
        """
        Save validation results to JSON and text files.

        Creates two files:
            - {patient_id}_validation.json: Full results in JSON format.
            - {patient_id}_validation_summary.txt: Human-readable summary.

        Args:
            save_dir: Directory for results.
                Defaults to results/{patient_id}/.
        """
        import json

        if save_dir is None:
            save_dir = RESULTS_PATH / self.patient_id
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save full results as JSON
        json_path = save_dir / f'{self.patient_id}_validation.json'
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        # Save summary as text
        txt_path = save_dir / f'{self.patient_id}_validation_summary.txt'
        with open(txt_path, 'w') as f:
            f.write(f"{'='*60}\n")
            f.write(f"PINN VALIDATION RESULTS - Patient {self.patient_id}\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"Category: {self.results['category']}\n")
            f.write(f"Timestamp: {self.results['timestamp']}\n\n")

            f.write("WSS Metrics:\n")
            for k, v in self.results['wss_metrics'].items():
                f.write(f"  {k}: {v}\n")

            f.write("\nVelocity Metrics:\n")
            for k, v in self.results['velocity_metrics'].items():
                f.write(f"  {k}: {v}\n")

            f.write("\nPer-Vessel Metrics:\n")
            for vessel, metrics in self.results['per_vessel_metrics'].items():
                f.write(f"  {vessel}:\n")
                for k, v in metrics.items():
                    f.write(f"    {k}: {v}\n")

        print(f"Results saved to {save_dir}")

    @staticmethod
    def batch_validate(
        patient_ids: Optional[List[str]] = None,
        save_csv: bool = True
    ) -> pd.DataFrame:
        """
        Validate multiple patients and compile results into a DataFrame.

        Iterates through all specified patients, loads their trained models,
        runs validation, and compiles results into a summary table.

        Args:
            patient_ids: List of patient IDs to validate.
                Defaults to all patients in PATIENT_DATA.
            save_csv: If True, save results to CSV file.

        Returns:
            DataFrame with validation metrics for all patients, containing:
                - patient_id, category
                - wss_rmse, wss_mae, wss_nrmse, wss_r2
                - vel_rmse, vel_r2
                - n_points
        """
        # Import here to avoid circular imports
        from src.model import VanillaPINN

        if patient_ids is None:
            patient_ids = list(PATIENT_DATA.keys())

        all_results: List[Dict] = []

        for pid in patient_ids:
            print(f"\n{'='*60}")
            print(f"Processing Patient: {pid}")

            # Check if model exists
            model_path = MODELS_PATH / pid / f'pinn_{pid}_best.pth'
            if not model_path.exists():
                print(f"  Model not found: {model_path}")
                continue

            try:
                # Load model checkpoint
                # Note: weights_only=False is required to load config dict.
                # Only use on trusted checkpoint files.
                checkpoint = torch.load(model_path, weights_only=False)
                config = checkpoint.get('config', {})

                model = VanillaPINN(
                    hidden_dim=config.get('hidden_dim', DEFAULT_HIDDEN_DIM),
                    num_blocks=config.get('num_blocks', DEFAULT_NUM_BLOCKS),
                    predict_wss=True
                )
                model.load_state_dict(checkpoint['model_state_dict'])

                # Validate
                validator = PINNValidator(model, pid)
                results = validator.validate(verbose=False)

                # Flatten results for DataFrame
                row = {
                    'patient_id': pid,
                    'category': results['category'],
                    'wss_rmse': results['wss_metrics']['RMSE'],
                    'wss_mae': results['wss_metrics']['MAE'],
                    'wss_nrmse': results['wss_metrics']['NRMSE'],
                    'wss_r2': results['wss_metrics']['R2'],
                    'vel_rmse': results['velocity_metrics']['RMSE'],
                    'vel_r2': results['velocity_metrics']['R2'],
                    'n_points': results['wss_metrics']['n_points']
                }
                all_results.append(row)

                print(
                    f"  WSS NRMSE: {row['wss_nrmse']:.4f}, "
                    f"R²: {row['wss_r2']:.4f}"
                )

            except FileNotFoundError as e:
                print(f"  Error: File not found - {e}")
                continue
            except KeyError as e:
                print(f"  Error: Missing key in checkpoint - {e}")
                continue
            except Exception as e:
                print(f"  Error validating {pid}: {e}")
                continue

        # Create DataFrame
        df = pd.DataFrame(all_results)

        if save_csv and len(df) > 0:
            csv_path = RESULTS_PATH / 'batch_validation_results.csv'
            df.to_csv(csv_path, index=False)
            print(f"\nBatch results saved to {csv_path}")

        return df
