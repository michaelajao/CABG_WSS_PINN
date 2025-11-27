"""
Configuration Module for PINN-based WSS Prediction

This module contains all configuration constants, file paths, and patient
data definitions used throughout the project.

Physical Constants:
    - RHO: Blood density (1060 kg/m³)
    - MU: Blood dynamic viscosity (0.0035 Pa·s)

Patient Categories:
    - Healthy: Normal coronary arteries (H-09, H-12)
    - Diseased: Stenosed vessels (D-10)
    - SVG: Saphenous vein grafts (0149, 0156, 0148, 0150)
    - Mixed: Combination of vessel types (0073)
"""

import torch
from pathlib import Path

# =============================================================================
# DEVICE CONFIGURATION
# =============================================================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================================
# FILE PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "PINNS"
MODELS_PATH = PROJECT_ROOT / "models"
FIGURES_PATH = PROJECT_ROOT / "figures"
RESULTS_PATH = PROJECT_ROOT / "results"

# =============================================================================
# PHYSICAL CONSTANTS (Blood Properties)
# =============================================================================

RHO = 1060.0   # Blood density (kg/m³)
MU = 0.0035    # Blood dynamic viscosity (Pa·s)

# =============================================================================
# PATIENT DATA CONFIGURATION
# =============================================================================

PATIENT_DATA = {
    'H-12': {
        'category': 'Healthy',
        'vessels': {
            'LCA': {'wall': 'H-12 LCA.csv', 'stream': 'H-12 LCA Streamlines.csv'}
        }
    },
    'H-09': {
        'category': 'Healthy', 
        'vessels': {
            'RCA': {'wall': 'H-09 RCA.csv', 'stream': 'H-09 Streamlines.csv'}
        }
    },
    'D-10': {
        'category': 'Diseased',
        'vessels': {
            'LCA': {'wall': 'D-10 LCA.csv', 'stream': 'D-10 LCA Streamlines.csv'},
            'RCA': {'wall': 'D-10 RCA.csv', 'stream': 'D-10 RCA STreamlines.csv'}
        }
    },
    '0149': {
        'category': 'SVG',
        'vessels': {
            'G1': {'wall': '0149 G1.csv', 'stream': '0149 G1 Streamlines.csv'},
            'G2': {'wall': '0149 G2.csv', 'stream': '0149 G2 Streamlines.csv'},
            'G3': {'wall': '0149 G3.csv', 'stream': '0149 G3 Streamlines.csv'},
        }
    },
    '0073': {
        'category': 'Mixed',
        'vessels': {
            'LCA': {'wall': '0073 LCA.csv', 'stream': '0073 LCA Streamlines.csv'},
            'RCA': {'wall': '0073 RCA.csv', 'stream': '0073 RCA Streamlines.csv'},
        }
    },
    '0156': {
        'category': 'SVG',
        'vessels': {
            'G2': {'wall': '0156 G2.csv', 'stream': '0156 G2 Streamlines.csv'},
            'G3': {'wall': '0156 G3.csv', 'stream': '0156 G3 Streamlines.csv'},
        }
    },
    '0148': {
        'category': 'SVG',
        'vessels': {
            'G2': {'wall': '0148 G2.csv', 'stream': '0148 G2 Streamlines.csv'},
        }
    },
    '0150': {
        'category': 'SVG',
        'vessels': {
            'G3': {'wall': '0150 G3.csv', 'stream': '0150 Streamlines.csv'},
        }
    },
    'ND2': {
        'category': 'Unknown',
        'vessels': {
            'LCA': {'wall': 'ND2 LCA.csv', 'stream': 'ND2 lca Streamlines.csv'}
        }
    }
}
