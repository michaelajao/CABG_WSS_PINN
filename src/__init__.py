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

from src.config import DEVICE, PATIENT_DATA, RHO, MU
from src.model import VanillaPINN, FourierPINN, MultiResNetPINN, KANPINN
from src.dataset import PatientDataset, load_patient_data, CollocationSampler
from src.train import train_patient
from src.evaluate import evaluate_model
from src.utils import compute_nrmse, EarlyStopping

__all__ = [
    # Configuration
    'DEVICE', 'PATIENT_DATA', 'RHO', 'MU',
    # Models
    'VanillaPINN', 'FourierPINN', 'MultiResNetPINN', 'KANPINN',
    # Data
    'PatientDataset', 'load_patient_data', 'CollocationSampler',
    # Training & Evaluation
    'train_patient', 'evaluate_model',
    # Utilities
    'compute_nrmse', 'EarlyStopping',
]
