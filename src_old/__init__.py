"""
Physics-Informed Neural Networks (PINNs) for Wall Shear Stress Prediction
in Coronary Artery Bypass Grafts

This package provides modular components for:
- Data loading and preprocessing
- PINN model architecture
- Physics-informed loss functions
- Training and evaluation
- Visualization and analysis

Author: Research Team
Date: November 8, 2025
"""

__version__ = "1.0.0"

from . import config
from . import utils
from . import dataset
from . import model
from . import physics
from . import train
from . import evaluate
from . import plots

__all__ = [
    "config",
    "utils",
    "dataset",
    "model",
    "physics",
    "train",
    "evaluate",
    "plots",
]
