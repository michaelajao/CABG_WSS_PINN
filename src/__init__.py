"""
Physics-Informed Neural Networks (PINNs) for Coronary Artery WSS Prediction.

This package provides a complete framework for training and evaluating PINNs
on patient-specific CFD simulation data from coronary arteries and saphenous
vein bypass grafts.

Modules:
    config: Configuration constants, file paths, and patient data registry
    dataset: Data loading, preprocessing, and GPU-resident data classes
    model: FourierPINN neural network architecture
    physics: Navier-Stokes and continuity equation residuals
    train: Training pipeline with early stopping and loss tracking
    evaluate: Model evaluation and metric computation
    plots: Publication-quality visualisation functions
    utils: Utility functions (EarlyStopping, compute_normalised_rmse)

For experimental architectures (VanillaPINN, PirateNetPINN, KANPINN) and
adaptive loss weighting (ReLoBRaLo), see the experimental/ folder.

Example:
    >>> from src.train import train_patient
    >>> model, results = train_patient(patient_id='0073', epochs=500)
    >>> print(f"WSS NRMSE: {results['metrics']['NRMSE']:.4f}")

Reference:
    Rehman et al. (2025). "Integrated CFD and Physics-Informed Neural Network
    Analysis of Hemodynamics in Healthy and Diseased Coronary Arteries and
    Saphenous Vein Grafts"
"""

__version__ = "1.0.0"
__author__ = "M. Abaid Ur Rehman, Ozgur Ekici, Sefik Evren Erdener, Michael Ajao-Olarinoye, Alex G. Kuchumov, Fei Jia"

from src.config import DEVICE, RHO, MU, PATIENT_DATA
from src.model import FourierPINN
from src.train import train_patient
from src.evaluate import evaluate_model
from src.dataset import (
    PatientData,
    CollocationSampler,
    CollocationSamplerGPU,
    load_patient_data
)
from src.physics import (
    compute_navier_stokes_residual,
    compute_continuity_residual,
    derive_wss_from_velocity_gradients,
    compute_wss_physics_residual,
)
from src.utils import compute_normalised_rmse, EarlyStopping

__all__ = [
    # Configuration
    'DEVICE', 'RHO', 'MU', 'PATIENT_DATA',
    # Models
    'FourierPINN',
    # Training
    'train_patient',
    # Evaluation
    'evaluate_model',
    # Dataset
    'PatientData', 'CollocationSampler', 'CollocationSamplerGPU', 'load_patient_data',
    # Physics
    'compute_navier_stokes_residual', 'compute_continuity_residual',
    'derive_wss_from_velocity_gradients', 'compute_wss_physics_residual',
    # Utilities
    'compute_normalised_rmse', 'EarlyStopping',
]
