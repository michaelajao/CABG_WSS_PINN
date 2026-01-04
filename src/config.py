"""
Configuration Module for PINN-based WSS Prediction

This module defines all configuration constants, file paths, physical parameters,
and patient data registry used throughout the PINN training pipeline.
"""

import torch
from pathlib import Path

# =============================================================================
# DEVICE CONFIGURATION
# =============================================================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================================
# DIRECTORY PATHS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data' / 'PINNS'
FIGURES_DIR = BASE_DIR / 'reports' / 'figures'
MODELS_DIR = BASE_DIR / 'reports' / 'models'
RESULTS_DIR = BASE_DIR / 'reports' / 'results'

# Create output directories
for directory in [FIGURES_DIR, MODELS_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Path aliases (used by train.py and evaluate.py)
DATA_PATH = DATA_DIR
FIGURES_PATH = FIGURES_DIR
MODELS_PATH = MODELS_DIR
RESULTS_PATH = RESULTS_DIR

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

RHO = 1050.0    # Blood density [kg/m^3]
MU = 0.0035     # Blood dynamic viscosity [Pa.s]

# =============================================================================
# PRIMARY VESSELS FOR PLOTTING
# =============================================================================

PRIMARY_VESSELS = {
    '0073': ['LCA', 'RCA'],
    '0148': ['G2'],
    '0149': ['G1', 'G2', 'G3'],
    '0150': ['G3'],
    '0156': ['G2', 'G3'],
    'D-10': ['LCA', 'RCA'],
    'H-09': ['RCA'],
    'H-12': ['LCA'],
    'ND2': ['LCA']
}

# =============================================================================
# PATIENT DATA REGISTRY
# =============================================================================

PATIENT_DATA = {
    'H-12': {
        'category': 'healthy',
        'aorta_file': 'H-12.csv',
        'vessels': {
            'LCA': {'wall': 'H-12 LCA.csv', 'stream': 'H-12 LCA Streamlines.csv'}
        }
    },
    'H-09': {
        'category': 'healthy',
        'aorta_file': 'H-09.csv',
        'vessels': {
            'RCA': {'wall': 'H-09 RCA.csv', 'stream': 'H-09 Streamlines.csv'}
        }
    },
    'D-10': {
        'category': 'diseased',
        'aorta_file': 'D-10.csv',
        'vessels': {
            'LCA': {'wall': 'D-10 LCA.csv', 'stream': 'D-10 LCA Streamlines.csv'},
            'RCA': {'wall': 'D-10 RCA.csv', 'stream': 'D-10 RCA STreamlines.csv'}
        }
    },
    '0073': {
        'category': 'mixed',
        'aorta_file': '0073.csv',
        'vessels': {
            'LCA': {'wall': '0073 LCA.csv', 'stream': '0073 LCA Streamlines.csv'},
            'RCA': {'wall': '0073 RCA.csv', 'stream': '0073 RCA Streamlines.csv'}
        }
    },
    '0148': {
        'category': 'svg',
        'aorta_file': '0148.csv',
        'vessels': {
            'G2': {'wall': '0148 G2.csv', 'stream': '0148 G2 Streamlines.csv'}
        }
    },
    '0149': {
        'category': 'svg',
        'aorta_file': '0149.csv',
        'vessels': {
            'G1': {'wall': '0149 G1.csv', 'stream': '0149 G1 Streamlines.csv'},
            'G2': {'wall': '0149 G2.csv', 'stream': '0149 G2 Streamlines.csv'},
            'G3': {'wall': '0149 G3.csv', 'stream': '0149 G3 Streamlines.csv'}
        }
    },
    '0150': {
        'category': 'svg',
        'aorta_file': '0150.csv',
        'vessels': {
            'G3': {'wall': '0150 G3.csv', 'stream': '0150 Streamlines.csv'}
        }
    },
    '0156': {
        'category': 'svg',
        'aorta_file': '0156.csv',
        'vessels': {
            'G2': {'wall': '0156 G2.csv', 'stream': '0156 G2 Streamlines.csv'},
            'G3': {'wall': '0156 G3.csv', 'stream': '0156 G3 Streamlines.csv'}
        }
    },
    'ND2': {
        'category': 'unknown',
        'aorta_file': 'ND2.csv',
        'vessels': {
            'LCA': {'wall': 'ND2 LCA.csv', 'stream': 'ND2 lca Streamlines.csv'}
        }
    }
}
