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

import csv
import json
import random
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# `config` is imported as a module (not just symbols) because some routines
# mutate the rheology switch ``_cfg.RHEOLOGY`` at runtime, which only takes
# effect when callers reach through the module object.
from src import config as _cfg
from src.config import (
    CY_AVAILABLE_LABELS,
    DEVICE,
    FIGURES_PATH,
    MODELS_PATH,
    PATIENT_DATA,
    RESULTS_PATH,
)
from src.dataset import PatientData, load_patient_data, load_vessel_data
from src.model import FourierPINN
from src.utils import compute_normalised_rmse

# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_HIDDEN_DIM: int = 256
DEFAULT_NUM_BLOCKS: int = 4


# =============================================================================
# QUICK EVALUATION
# =============================================================================

def _pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson correlation coefficient r (numpy implementation, no scipy dependency)."""
    if y_true.size < 2:
        return float('nan')
    yt = y_true - y_true.mean()
    yp = y_pred - y_pred.mean()
    denom = np.sqrt((yt ** 2).sum()) * np.sqrt((yp ** 2).sum())
    if denom == 0:
        return float('nan')
    return float((yt * yp).sum() / denom)


def evaluate_model(
    model: nn.Module,
    dataset: PatientData,
    batch_size: int = 4096,
    split: str = "all",
) -> dict:
    """
    Evaluate trained PINN model on WSS prediction.

    Args:
        model: Trained PINN model in evaluation mode.
        dataset: PatientData instance with GPU tensors.
        batch_size: Batch size for evaluation.
        split: "all" (default; full dataset), "train" (training subset only),
            or "holdout" (held-out subset only — Physics of Fluids R1-5/R2-6).

    Returns:
        Dictionary with RMSE, MAE, NRMSE, R2, Pearson r, n_points, and split label.
    """
    if split not in ("all", "train", "holdout"):
        raise ValueError(f"split must be 'all', 'train', or 'holdout' (got {split!r})")

    model.eval()
    T_ref = dataset.T_ref

    # Pick the index set for this split.
    if split == "all":
        idx = torch.arange(dataset.num_samples, device=dataset.coords.device)
    elif split == "train":
        idx = dataset.train_indices
    else:
        idx = dataset.holdout_indices

    if idx.numel() == 0:
        return {
            'RMSE': float('nan'), 'MAE': float('nan'), 'R2': float('nan'),
            'NRMSE': float('nan'), 'pearson_r': float('nan'),
            'n_points': 0, 'split': split,
        }

    has_wss_split = dataset.has_wss[idx].cpu().numpy()
    y_true = dataset.y_raw[idx.cpu().numpy()][has_wss_split]

    all_pred: list[np.ndarray] = []
    coords_all = dataset.coords[idx]
    n = coords_all.shape[0]
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            outputs = model(coords_all[start:end])
            all_pred.append((outputs['wss'].cpu().numpy().flatten() * T_ref))
    y_pred_all = np.concatenate(all_pred)
    y_pred = y_pred_all[has_wss_split]

    if y_true.size == 0:
        return {
            'RMSE': float('nan'), 'MAE': float('nan'), 'R2': float('nan'),
            'NRMSE': float('nan'), 'pearson_r': float('nan'),
            'n_points': 0, 'split': split,
        }

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    nrmse = compute_normalised_rmse(y_true, y_pred)
    pearson = _pearson_r(y_true, y_pred)

    metrics = {
        'RMSE': float(rmse),
        'MAE': float(mae),
        'R2': float(r2),
        'NRMSE': float(nrmse),
        'pearson_r': pearson,
        'n_points': int(len(y_true)),
        'split': split,
    }

    label = split if split != 'all' else 'full'
    print(f"  [{label}] RMSE: {rmse:.4f} | NRMSE: {nrmse:.4f} | R²: {r2:.4f} | r: {pearson:.4f}")

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
        dataset (PatientData): Dataset instance with GPU tensors.
        results (dict): Validation results storage.

    Example:
        >>> from src.model import FourierPINN
        >>> model = FourierPINN()
        >>> model.load_state_dict(torch.load('model.pth')['model_state_dict'])
        >>> validator = PINNValidator(model, patient_id='H1', rheology='newtonian')
        >>> results = validator.validate()
        >>> validator.generate_full_patient_plots()
    """

    def __init__(
        self,
        model: nn.Module,
        patient_id: str,
        device: str | torch.device | None = None,
        rheology: str | None = None,
    ) -> None:
        """
        Initialize the validator.

        Args:
            model: Trained PINN model (already loaded with weights).
            patient_id: Public paper label (``'H1'``, ``'H4'``, ``'BG2'``, ...).
            device: Device for computation ('cuda' or 'cpu').
                Auto-detected if None.
            rheology: ``'newtonian'`` or ``'carreau_yasuda'``. Defaults to
                ``config.RHEOLOGY``. Determines which CFD ground truth to
                evaluate against AND the rheology subdir under
                ``RESULTS_PATH``/``FIGURES_PATH`` so outputs don't collide
                across rheologies.

        Raises:
            KeyError: If patient_id is not found in PATIENT_DATA.
        """
        self.model = model
        self.patient_id = patient_id
        self.rheology = rheology if rheology is not None else _cfg.RHEOLOGY
        self.device: torch.device = torch.device(device if device else DEVICE)
        self.model.to(self.device)
        self.model.eval()

        # Load patient data for the requested rheology.
        self.data, self.per_vessel = load_patient_data(patient_id, self.rheology)
        self.dataset = PatientData(self.data, device=self.device)

        # Results storage
        self.results: dict = {
            'patient_id': patient_id,
            'rheology': self.rheology,
            'category': PATIENT_DATA.get(patient_id, {}).get('category', 'Unknown'),
            'timestamp': datetime.now().isoformat(),
            'wss_metrics': {},
            'velocity_metrics': {},
            'per_vessel_metrics': {}
        }

        # Predictions cache
        self._predictions: dict[str, np.ndarray] | None = None

    def predict(self, coords: np.ndarray) -> dict[str, np.ndarray]:
        """
        Run model prediction on given coordinates.

        Args:
            coords: Raw coordinates with shape (N, 3) in meters.

        Returns:
            Dictionary with predictions (all denormalized to physical units):
                - wss: Wall shear stress in Pa.
                - u, v, w: Velocity components in m/s.
                - p: Pressure in Pa (using P_ref = rho * U_ref^2).
        """
        # Scale coordinates using uniform scaling
        coords_scaled = (coords - self.dataset.coord_offset) / self.dataset.L_ref
        coords_tensor = torch.FloatTensor(coords_scaled).to(self.device)

        with torch.no_grad():
            outputs = self.model(coords_tensor)

        # Denormalize outputs using reference scales
        # WSS: tau = tau* * T_ref where T_ref = mu * U_ref / L_ref
        wss_pred_nondim = outputs['wss'].cpu().numpy().flatten()
        wss_pred = wss_pred_nondim * self.dataset.T_ref

        # Velocity: u = u* * U_ref
        U_ref = self.dataset.U_ref
        vel_pred = np.column_stack([
            outputs['u'].cpu().numpy().flatten() * U_ref,
            outputs['v'].cpu().numpy().flatten() * U_ref,
            outputs['w'].cpu().numpy().flatten() * U_ref
        ])

        # Pressure: p = p* * P_ref where P_ref = rho * U_ref^2
        p_pred_nondim = outputs['p'].cpu().numpy().flatten()
        p_pred = p_pred_nondim * self.dataset.P_ref

        return {
            'wss': wss_pred,
            'u': vel_pred[:, 0],
            'v': vel_pred[:, 1],
            'w': vel_pred[:, 2],
            'p': p_pred
        }

    def validate(self, verbose: bool = True) -> dict:
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
    ) -> dict:
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
        nrmse = compute_normalised_rmse(y_true, y_pred)

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

    def generate_full_patient_plots(self, save_dir: Path | None = None) -> None:
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
            save_dir = FIGURES_PATH / self.rheology / self.patient_id
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nGenerating full patient plots for {self.patient_id} ({self.rheology})...")

        # Prepare vessel data for plotting
        vessel_data: list[dict] = []

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
        df_aorta: np.ndarray | None = aorta_data['X'] if aorta_data is not None else None

        # Generate plots
        plot_full_patient_wss(self.patient_id, vessel_data, df_aorta, save_dir)

    def save_results(self, save_dir: Path | None = None) -> None:
        """
        Save validation results to JSON and text files.

        Creates two files:
            - {patient_id}_validation.json: Full results in JSON format.
            - {patient_id}_validation_summary.txt: Human-readable summary.

        Args:
            save_dir: Directory for results.
                Defaults to results/{patient_id}/.
        """

        if save_dir is None:
            save_dir = RESULTS_PATH / self.rheology / self.patient_id
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
        patient_ids: list[str] | None = None,
        save_csv: bool = True,
        rheology: str | None = None,
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
        rheo = rheology if rheology is not None else _cfg.RHEOLOGY

        if patient_ids is None:
            patient_ids = list(PATIENT_DATA.keys())

        all_results: list[dict] = []

        for pid in patient_ids:
            print(f"\n{'='*60}")
            print(f"Processing Patient: {pid} ({rheo})")

            # Check if model exists; checkpoints are namespaced by rheology so
            # Newtonian and Carreau-Yasuda runs on the same patient don't collide.
            model_path = MODELS_PATH / rheo / pid / f'pinn_{pid}_best.pth'
            if not model_path.exists():
                print(f"  Model not found: {model_path}")
                continue

            # Load model checkpoint. weights_only=False is required to load
            # the embedded config dict; only use on trusted checkpoint files.
            checkpoint = torch.load(model_path, weights_only=False)
            cfg = checkpoint['config']
            hidden_dim = cfg.get('hidden_dim', DEFAULT_HIDDEN_DIM)
            num_blocks = cfg.get('num_blocks', DEFAULT_NUM_BLOCKS)
            model = FourierPINN(
                hidden_dim=hidden_dim,
                num_blocks=num_blocks,
                predict_wss=True,
                num_frequencies=cfg.get('num_frequencies', 64),
                fourier_scale=cfg.get('fourier_scale', 10.0),
            )

            model = model.to(DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])

            # Validate (pass the rheology so the validator loads matching CFD)
            validator = PINNValidator(model, pid, rheology=rheo)
            results = validator.validate(verbose=False)

            row = {
                'patient_id': pid,
                'rheology': rheo,
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

        # Create DataFrame
        df = pd.DataFrame(all_results)

        if save_csv and len(df) > 0:
            csv_path = RESULTS_PATH / 'batch_validation_results.csv'
            df.to_csv(csv_path, index=False)
            print(f"\nBatch results saved to {csv_path}")

        return df


# =============================================================================
# Cross-patient holdout sweep + per-patient sensitivity sweeps
# =============================================================================
# These two routines drive multiple ``train_patient`` runs and persist the
# aggregated metrics under ``reports/metrics/``. They are exposed via a
# ``__main__`` dispatcher at the bottom of this file:
#
#     python -m src.evaluate holdout     --rheology newtonian
#     python -m src.evaluate sensitivity --patient H4

def _flatten_holdout_result(patient_id: str, result: dict) -> dict:
    """Flatten a single train_patient result dict into a CSV-friendly row."""
    metrics = result.get('metrics', {})
    train = metrics.get('train', metrics)
    holdout = metrics.get('holdout', metrics)
    timing = result.get('timing', {})
    return {
        'patient_id': patient_id,
        'category': PATIENT_DATA[patient_id]['category'],
        'n_train': train.get('n_points', 0),
        'n_holdout': holdout.get('n_points', 0),
        'NRMSE_train': train.get('NRMSE', float('nan')),
        'NRMSE_holdout': holdout.get('NRMSE', float('nan')),
        'RMSE_train': train.get('RMSE', float('nan')),
        'RMSE_holdout': holdout.get('RMSE', float('nan')),
        'R2_train': train.get('R2', float('nan')),
        'R2_holdout': holdout.get('R2', float('nan')),
        'pearson_train': train.get('pearson_r', float('nan')),
        'pearson_holdout': holdout.get('pearson_r', float('nan')),
        'train_seconds': timing.get('train_seconds', float('nan')),
        'inference_seconds_full_field': timing.get('inference_seconds_full_field', float('nan')),
        'peak_gpu_mb': timing.get('peak_gpu_mb', float('nan')),
    }


def run_holdout_sweep(
    patients: list[str] | None = None,
    rheology: str = 'newtonian',
    epochs: int = 500,
    holdout_fraction: float = 0.20,
    holdout_seed: int = 0,
    metrics_dir: Path | None = None,
) -> list[dict]:
    """Train one PINN per patient under a spatial holdout and aggregate metrics.

    Writes ``reports/metrics/holdout_summary_<rheology>.csv`` (and matching
    JSON), flushed after every patient so a mid-sweep crash does not lose
    earlier work.

    Args:
        patients: Patient labels to run. If None, defaults to all patients
            eligible for the chosen rheology (``CY_AVAILABLE_LABELS`` for
            ``carreau_yasuda``, all of ``PATIENT_DATA`` for ``newtonian``).
        rheology: ``'newtonian'`` or ``'carreau_yasuda'``.
        epochs: Maximum training epochs per patient.
        holdout_fraction: Fraction of mesh points withheld from training.
        holdout_seed: Random seed for the holdout split.
        metrics_dir: Directory for the aggregated CSV/JSON. Defaults to
            ``reports/metrics/``.

    Returns:
        List of per-patient flattened metric dicts.
    """
    from src.train import train_patient

    _cfg.RHEOLOGY = rheology
    eligible = (
        list(CY_AVAILABLE_LABELS) if rheology == 'carreau_yasuda'
        else list(PATIENT_DATA.keys())
    )
    if patients:
        run_list = [p for p in patients if p in eligible]
        skipped = [p for p in patients if p not in eligible]
        if skipped:
            print(f'Skipping (no {rheology} CFD ground truth): {skipped}')
    else:
        run_list = eligible

    metrics_dir = Path(metrics_dir) if metrics_dir else (RESULTS_PATH.parent / 'metrics')
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_csv = metrics_dir / f'holdout_summary_{rheology}.csv'
    out_json = metrics_dir / f'holdout_summary_{rheology}.json'
    print(f'[holdout-sweep] rheology={rheology}; patients={run_list}')
    print(f'[holdout-sweep] writing results to {out_csv}')

    rows: list[dict] = []
    for pid in run_list:
        print(f'\n=== Holdout training: {pid} ===')
        _, result = train_patient(
            patient_id=pid,
            epochs=epochs,
            holdout_fraction=holdout_fraction,
            holdout_seed=holdout_seed,
            verbose=False,
        )
        rows.append(_flatten_holdout_result(pid, result))

        # Flush after each patient so a later crash doesn't lose earlier work.
        fieldnames = sorted({k for r in rows for k in r})
        with out_csv.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        out_json.write_text(json.dumps(rows, indent=2))

    print(f'\nWrote {len(rows)} rows to {out_csv}')
    print(f'Wrote {out_json}')
    return rows


# Sensitivity-sweep grids (Physics of Fluids revision: R2-8, R2-10, R2-11).
SENSITIVITY_LOSS_WEIGHT_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]   # multiplier on lambda_NS
SENSITIVITY_COLLOCATION_GRID = [256, 512, 1024, 2048, 4096]    # collocation points / iter
SENSITIVITY_SEED_GRID = [0, 1, 2, 3, 4]


def _sensitivity_train_once(
    patient_id: str, epochs: int, seed: int,
    num_collocation_points: int, lambda_ns_mult: float,
    holdout_seed: int = 0,
) -> dict:
    """Run one training pass and return held-out WSS metrics."""
    from src import train as _train_module

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Apply lambda_NS multiplier on the module-level LOSS_PRIORITY dict
    # (consumed by the per-run gradnorm balancer in train_patient), then
    # restore it after the call.
    saved = dict(_train_module.LOSS_PRIORITY)
    _train_module.LOSS_PRIORITY['navier_stokes'] = (
        saved['navier_stokes'] * lambda_ns_mult
    )
    try:
        _, result = _train_module.train_patient(
            patient_id=patient_id,
            epochs=epochs,
            num_collocation_points=num_collocation_points,
            holdout_fraction=0.20,
            holdout_seed=holdout_seed,
            verbose=False,
        )
    finally:
        _train_module.LOSS_PRIORITY.update(saved)

    metrics = result['metrics']
    holdout = metrics.get('holdout', metrics)
    return {
        'NRMSE_holdout': holdout['NRMSE'],
        'R2_holdout': holdout['R2'],
        'pearson_r_holdout': holdout['pearson_r'],
        'n_holdout': holdout['n_points'],
    }


def _sensitivity_write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for r in rows for k in r})
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f'Wrote {len(rows)} rows to {path}')


def run_sensitivity_sweeps(
    patient: str = 'H4',
    rheology: str = 'newtonian',
    epochs_short: int = 1000,
    epochs_full: int = 500,
    sweeps: str = 'all',
    metrics_dir: Path | None = None,
) -> None:
    """Run the three sensitivity sweeps (loss-weight, collocation, seeds).

    Output CSVs land at ``reports/metrics/sensitivity_<sweep>_<rheology>_<patient>.csv``
    so different rheologies and representative patients do not overwrite each
    other.

    Args:
        patient: Patient label to run all sweeps on.
        rheology: ``'newtonian'`` or ``'carreau_yasuda'``.
        epochs_short: Epoch budget for the loss-weight sweep.
        epochs_full: Epoch budget for collocation and seed sweeps.
        sweeps: ``'all'`` or one of ``'lossweight'``, ``'collocation'``, ``'seeds'``.
        metrics_dir: Directory for the output CSVs. Defaults to ``reports/metrics/``.
    """

    _cfg.RHEOLOGY = rheology
    if rheology == 'carreau_yasuda' and patient not in _cfg.CY_AVAILABLE_LABELS:
        raise SystemExit(
            f'ERROR: --rheology carreau_yasuda requested for patient {patient!r}, '
            f'but no Carreau-Yasuda CFD ground truth exists for it. '
            f'Eligible patients: {sorted(_cfg.CY_AVAILABLE_LABELS)}.'
        )

    metrics_dir = Path(metrics_dir) if metrics_dir else (RESULTS_PATH.parent / 'metrics')
    metrics_dir.mkdir(parents=True, exist_ok=True)
    suffix = f'_{rheology}_{patient}'

    if sweeps in ('all', 'lossweight'):
        rows: list[dict] = []
        for mult in SENSITIVITY_LOSS_WEIGHT_GRID:
            print(f'\n[lossweight] lambda_NS x{mult} on {patient} ({rheology})')
            m = _sensitivity_train_once(
                patient, epochs_short, seed=0,
                num_collocation_points=4096, lambda_ns_mult=mult,
            )
            rows.append({'sweep': 'lossweight', 'lambda_NS_mult': mult, **m})
        _sensitivity_write_csv(rows, metrics_dir / f'sensitivity_lossweight{suffix}.csv')

    if sweeps in ('all', 'collocation'):
        rows = []
        for nc in SENSITIVITY_COLLOCATION_GRID:
            print(f'\n[collocation] n_colloc={nc} on {patient} ({rheology})')
            m = _sensitivity_train_once(
                patient, epochs_full, seed=0,
                num_collocation_points=nc, lambda_ns_mult=1.0,
            )
            rows.append({'sweep': 'collocation', 'n_collocation': nc, **m})
        _sensitivity_write_csv(rows, metrics_dir / f'sensitivity_collocation{suffix}.csv')

    if sweeps in ('all', 'seeds'):
        rows = []
        for s in SENSITIVITY_SEED_GRID:
            print(f'\n[seeds] seed={s} on {patient} ({rheology})')
            m = _sensitivity_train_once(
                patient, epochs_full, seed=s,
                num_collocation_points=4096, lambda_ns_mult=1.0,
            )
            rows.append({'sweep': 'seed', 'seed': s, **m})
        _sensitivity_write_csv(rows, metrics_dir / f'sensitivity_seeds{suffix}.csv')


def _evaluate_main(argv: list[str] | None = None) -> None:
    """CLI entry point: ``python -m src.evaluate {holdout|sensitivity} ...``."""
    import argparse
    parser = argparse.ArgumentParser(
        description='Run a cross-patient holdout sweep or per-patient '
                    'sensitivity sweeps and write CSVs to reports/metrics/.'
    )
    sub = parser.add_subparsers(dest='cmd', required=True)

    h = sub.add_parser('holdout',
        help='Train every eligible patient under a 20%% spatial holdout.')
    h.add_argument('--patients', nargs='+', default=None,
                   help='Patients to run (default: all eligible for the chosen rheology).')
    h.add_argument('--epochs', type=int, default=500)
    h.add_argument('--holdout-fraction', type=float, default=0.20)
    h.add_argument('--holdout-seed', type=int, default=0)
    h.add_argument('--rheology', choices=['newtonian', 'carreau_yasuda'], default='newtonian')
    h.add_argument('--metrics-dir', default=None)

    s = sub.add_parser('sensitivity',
        help='Three sensitivity sweeps on one representative patient.')
    s.add_argument('--patient', default='H4',
                   help='Patient label used for all sweeps (default: H4).')
    s.add_argument('--rheology', choices=['newtonian', 'carreau_yasuda'], default='newtonian')
    s.add_argument('--epochs-short', type=int, default=1000,
                   help='Epoch budget for the loss-weight sweep (default: 1000).')
    s.add_argument('--epochs-full', type=int, default=500,
                   help='Epoch budget for the collocation and seed sweeps.')
    s.add_argument('--sweeps', choices=['all', 'lossweight', 'collocation', 'seeds'],
                   default='all', help='Which sweep to run (default: all).')
    s.add_argument('--metrics-dir', default=None)

    args = parser.parse_args(argv)
    if args.cmd == 'holdout':
        run_holdout_sweep(
            patients=args.patients, rheology=args.rheology, epochs=args.epochs,
            holdout_fraction=args.holdout_fraction, holdout_seed=args.holdout_seed,
            metrics_dir=Path(args.metrics_dir) if args.metrics_dir else None,
        )
    else:
        run_sensitivity_sweeps(
            patient=args.patient, rheology=args.rheology,
            epochs_short=args.epochs_short, epochs_full=args.epochs_full,
            sweeps=args.sweeps,
            metrics_dir=Path(args.metrics_dir) if args.metrics_dir else None,
        )


if __name__ == '__main__':
    _evaluate_main()
