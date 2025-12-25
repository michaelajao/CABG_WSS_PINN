"""
Configuration Module for PINN-based WSS Prediction.

This module contains all configuration constants, file paths, and patient
data definitions used throughout the project.

Attributes:
    DEVICE (torch.device): Computation device (CUDA if available, else CPU).
    PROJECT_ROOT (Path): Root directory of the project.
    DATA_PATH (Path): Path to CFD simulation data.
    MODELS_PATH (Path): Path to saved model checkpoints.
    FIGURES_PATH (Path): Path to generated figures.
    RESULTS_PATH (Path): Path to evaluation results.
    RHO (float): Blood density in kg/m³.
    MU (float): Blood dynamic viscosity in Pa·s.
    PATIENT_DATA (dict): Registry of patient vessels and data files.
    PRIMARY_VESSELS (dict): Primary vessels to analyze for each patient.

Physical Constants:
    - RHO: Blood density (1060 kg/m³)
    - MU: Blood dynamic viscosity (0.0035 Pa·s)

Patient Categories:
    - Healthy: Normal coronary arteries (H-09, H-12)
    - Diseased: Stenosed vessels (D-10)
    - SVG: Saphenous vein grafts (0149, 0156, 0148, 0150)
    - Mixed: Combination of vessel types (0073)
"""

from pathlib import Path
from typing import Dict, List, Optional

import torch

# =============================================================================
# DEVICE CONFIGURATION
# =============================================================================

DEVICE: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================================
# FILE PATHS
# =============================================================================

PROJECT_ROOT: Path = Path(__file__).parent.parent
DATA_PATH: Path = PROJECT_ROOT / "data" / "PINNS"
MODELS_PATH: Path = PROJECT_ROOT / "reports" / "models"
FIGURES_PATH: Path = PROJECT_ROOT / "reports" / "figures"
RESULTS_PATH: Path = PROJECT_ROOT / "reports" / "results"

# =============================================================================
# PHYSICAL CONSTANTS (Blood Properties)
# =============================================================================

RHO: float = 1060.0   # Blood density (kg/m³)
MU: float = 0.0035    # Blood dynamic viscosity (Pa·s)

# =============================================================================
# PRIMARY VESSELS CONFIGURATION
# =============================================================================

# Primary vessels to analyze for each patient (excludes Aorta)
PRIMARY_VESSELS: Dict[str, List[str]] = {
    'H-12': ['LCA'],
    'H-09': ['RCA'],
    'D-10': ['LCA', 'RCA'],
    '0149': ['G1', 'G2', 'G3'],
    '0073': ['LCA', 'RCA'],
    '0156': ['G2', 'G3'],
    '0148': ['G2'],
    '0150': ['G3'],
    'ND2': ['LCA'],
}

# =============================================================================
# PATIENT DATA CONFIGURATION
# =============================================================================

# Type alias for vessel data structure
VesselData = Dict[str, Dict[str, Optional[str]]]
PatientEntry = Dict[str, object]

PATIENT_DATA: Dict[str, PatientEntry] = {
    'H-12': {
        'category': 'Healthy',
        'vessels': {
            'Aorta': {'wall': 'H-12.csv', 'stream': None},
            'LCA': {'wall': 'H-12 LCA.csv', 'stream': 'H-12 LCA Streamlines.csv'}
        }
    },
    'H-09': {
        'category': 'Healthy',
        'vessels': {
            'Aorta': {'wall': 'H-09.csv', 'stream': None},
            'RCA': {'wall': 'H-09 RCA.csv', 'stream': 'H-09 Streamlines.csv'}
        }
    },
    'D-10': {
        'category': 'Diseased',
        'vessels': {
            'Aorta': {'wall': 'D-10.csv', 'stream': None},
            'LCA': {'wall': 'D-10 LCA.csv', 'stream': 'D-10 LCA Streamlines.csv'},
            'RCA': {'wall': 'D-10 RCA.csv', 'stream': 'D-10 RCA Streamlines.csv'}
        }
    },
    '0149': {
        'category': 'SVG',
        'vessels': {
            'Aorta': {'wall': '0149.csv', 'stream': None},
            'G1': {'wall': '0149 G1.csv', 'stream': '0149 G1 Streamlines.csv'},
            'G2': {'wall': '0149 G2.csv', 'stream': '0149 G2 Streamlines.csv'},
            'G3': {'wall': '0149 G3.csv', 'stream': '0149 G3 Streamlines.csv'},
        }
    },
    '0073': {
        'category': 'Mixed',
        'vessels': {
            'Aorta': {'wall': '0073.csv', 'stream': None},
            'LCA': {'wall': '0073 LCA.csv', 'stream': '0073 LCA Streamlines.csv'},
            'RCA': {'wall': '0073 RCA.csv', 'stream': '0073 RCA Streamlines.csv'},
        }
    },
    '0156': {
        'category': 'SVG',
        'vessels': {
            'Aorta': {'wall': '0156.csv', 'stream': None},
            'G2': {'wall': '0156 G2.csv', 'stream': '0156 G2 Streamlines.csv'},
            'G3': {'wall': '0156 G3.csv', 'stream': '0156 G3 Streamlines.csv'},
        }
    },
    '0148': {
        'category': 'SVG',
        'vessels': {
            'Aorta': {'wall': '0148.csv', 'stream': None},
            'G2': {'wall': '0148 G2.csv', 'stream': '0148 G2 Streamlines.csv'},
        }
    },
    '0150': {
        'category': 'SVG',
        'vessels': {
            'Aorta': {'wall': '0150.csv', 'stream': None},
            'G3': {'wall': '0150 G3.csv', 'stream': '0150 Streamlines.csv'},
        }
    },
    'ND2': {
        'category': 'Unknown',
        'vessels': {
            'Aorta': {'wall': 'ND2.csv', 'stream': None},
            'LCA': {'wall': 'ND2 LCA.csv', 'stream': 'ND2 lca Streamlines.csv'}
        }
    }
}
