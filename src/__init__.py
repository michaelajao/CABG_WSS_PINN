"""
Physics-Informed Neural Networks (PINNs) for Coronary Artery WSS Prediction

This package provides a complete framework for training and evaluating PINNs
on patient-specific CFD simulation data from coronary arteries and saphenous
vein bypass grafts.

Modules:
    config: Configuration constants, file paths, and patient data registry
    dataset: Data loading, preprocessing, and PyTorch Dataset classes
    model: Neural network architectures (VanillaPINN, FourierPINN, MultiResNetPINN, KANPINN)
    physics: Navier-Stokes and continuity equation residuals
    train: Training pipeline with early stopping and loss tracking
    evaluate: Model evaluation and metric computation
    plots: Publication-quality visualization functions
    utils: Utility functions (EarlyStopping, compute_nrmse)

Example:
    >>> from src.train import train_patient
    >>> model, results = train_patient(patient_id='0073', epochs=500)
    >>> print(f"WSS NRMSE: {results['metrics']['NRMSE']:.4f}")

Reference:
    Rehman et al. (2025). "Computational Investigation of Blood Flow in 
    Saphenous Vein Grafts and Coronary Arteries: CFD Analysis with 
    Physics-Informed Neural Network Surrogate Modelling"
"""

__version__ = "1.0.0"
__author__ = "M. Abaid Ur Rehman, Özgür Ekici, Şefik Evren Erdener, Michael Ajao-Olarinoye, Alex G. Kuchumov"

from src.config import DEVICE, RHO, MU, PATIENT_DATA
from src.model import VanillaPINN, FourierPINN, MultiResNetPINN, KANPINN, PirateNetPINN
from src.train import train_patient
from src.evaluate import evaluate_model
from src.physics import (
    navier_stokes_residual_nondim,
    continuity_residual_nondim,
    compute_wss_from_gradients,
    wss_physics_residual_nondim
)

__all__ = [
    'DEVICE', 'RHO', 'MU', 'PATIENT_DATA',
    'VanillaPINN', 'FourierPINN', 'MultiResNetPINN', 'KANPINN', 'PirateNetPINN',
    'train_patient',
    'evaluate_model',
    'navier_stokes_residual_nondim', 'continuity_residual_nondim',
    'compute_wss_from_gradients', 'wss_physics_residual_nondim'
]
