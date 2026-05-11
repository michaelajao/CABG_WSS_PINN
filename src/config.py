"""
Configuration Module for PINN-based WSS Prediction

This module defines all configuration constants, file paths, physical parameters,
and patient data registry used throughout the PINN training pipeline.

The PATIENT_DATA registry is keyed by the **public paper label** (H1, H2, ...,
BG5, D1, D2, D3) and resolves rheology-specific file paths via
``patient_files(label, rheology)``. The on-disk dataset IDs (0073, 0066, H09,
H12, 0148, 0149, 0150, 0156, 0157, D1, D2, D10) live inside each entry under
``data_id`` for traceability.
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

# Per-rheology dataset roots, flat under data/. Constant names match the
# values of RHEOLOGY ("newtonian", "carreau_yasuda") for symmetry with the
# rest of the code.
NEWTONIAN_DATA_DIR = BASE_DIR / 'data' / 'Newtonian'
CARREAU_YASUDA_DATA_DIR = BASE_DIR / 'data' / 'Carreau'

# Map rheology string -> data directory. Use this rather than ad-hoc lookups.
DATA_DIR_BY_RHEOLOGY = {
    'newtonian': NEWTONIAN_DATA_DIR,
    'carreau_yasuda': CARREAU_YASUDA_DATA_DIR,
}

# Legacy aliases kept so older code that imports DATA_DIR / DATA_PATH still
# resolves; both point at the Newtonian root, which is the historical default.
DATA_DIR = NEWTONIAN_DATA_DIR
DATA_PATH = DATA_DIR

FIGURES_DIR = BASE_DIR / 'reports' / 'figures'
MODELS_DIR = BASE_DIR / 'reports' / 'models'
RESULTS_DIR = BASE_DIR / 'reports' / 'results'

for directory in [FIGURES_DIR, MODELS_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

FIGURES_PATH = FIGURES_DIR
MODELS_PATH = MODELS_DIR
RESULTS_PATH = RESULTS_DIR

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

RHO = 1050.0    # Blood density [kg/m^3]
MU = 0.0035     # Newtonian dynamic viscosity / mu_infinity for Carreau-Yasuda [Pa.s]

RHEOLOGY = "newtonian"  # default; overridden by --rheology CLI flag

# Carreau-Yasuda parameters (match the CFD simulations).
CY_MU_INF = 0.0035   # Pa.s
CY_MU_0 = 0.16       # Pa.s
CY_LAMBDA = 8.2      # s
CY_N = 0.2128
CY_A = 0.64

CY_PARAMS = {
    "mu_inf": CY_MU_INF,
    "mu_0": CY_MU_0,
    "lam": CY_LAMBDA,
    "n": CY_N,
    "a": CY_A,
}

# =============================================================================
# PATIENT DATA REGISTRY (12 patients, keyed by public paper label)
# =============================================================================
#
# Each entry has:
#   data_id   : on-disk dataset ID (used in CSV filenames)
#   category  : 'healthy' | 'svg' | 'diseased'
#   newtonian : {'aorta_file': str, 'vessels': {vessel_name: {'wall', 'stream'}}}
#   carreau_yasuda : same shape if CY data exists, else None
#
# All twelve patients have both Newtonian and Carreau-Yasuda CFD ground truth
# available; CY_AVAILABLE_LABELS (derived below) therefore contains all keys.

PATIENT_DATA = {
    'H1': {
        'data_id': '0073',
        'category': 'healthy',
        'newtonian': {
            'aorta_file': '0073.csv',
            'vessels': {
                'LCA': {'wall': '0073 LCA.csv', 'stream': '0073 LCA Streamlines.csv'},
                'RCA': {'wall': '0073 RCA.csv', 'stream': '0073 RCA Streamlines.csv'},
            },
        },
        'carreau_yasuda': {
            'aorta_file': '0073.csv',
            'vessels': {
                'LCA': {'wall': '0073 LCA.csv', 'stream': '0073 LCA Streamlines.csv'},
                'RCA': {'wall': '0073 RCA.csv', 'stream': '0073 RCA Streamlines.csv'},
            },
        },
    },
    'H2': {
        'data_id': '0066',
        'category': 'healthy',
        'newtonian': {
            'aorta_file': '0066 healthy.csv',
            'vessels': {
                'LCA': {'wall': '0066 healthy LCA.csv',
                        'stream': '0066 healthy LCA Streamlines.csv'},
                'RCA': {'wall': '0066 healthy RCA.csv',
                        'stream': '0066 healthy RCA Streamlines.csv'},
            },
        },
        'carreau_yasuda': {
            'aorta_file': '0066.csv',
            'vessels': {
                'LCA': {'wall': '0066 LCA.csv',
                        'stream': '0066 LCA Streamlines.csv'},
                'RCA': {'wall': '0066 RCA.csv',
                        'stream': '0066 RCA Streamlines.csv'},
            },
        },
    },
    'H3': {
        'data_id': 'H09',
        'category': 'healthy',
        'newtonian': {
            'aorta_file': 'H09.csv',
            'vessels': {
                'LCA': {'wall': 'H09 LCA.csv', 'stream': 'H09 LCA Streamlines.csv'},
                'RCA': {'wall': 'H09 RCA.csv', 'stream': 'H09 RCA Streamlines.csv'},
            },
        },
        'carreau_yasuda': {
            # Old CY run only covers RCA + a generic 'H09 Streamlines.csv'.
            'aorta_file': 'H09.csv',
            'vessels': {
                'RCA': {'wall': 'H09 RCA.csv', 'stream': 'H09 Streamlines.csv'},
            },
        },
    },
    'H4': {
        'data_id': 'H12',
        'category': 'healthy',
        'newtonian': {
            'aorta_file': 'H12.csv',
            'vessels': {
                'LCA': {'wall': 'H12 LCA.csv', 'stream': 'H12 LCA Streamlines.csv'},
                'RCA': {'wall': 'H12 RCA.csv', 'stream': 'H12 RCA Streamlines.csv'},
            },
        },
        'carreau_yasuda': {
            # Old CY run only covers LCA.
            'aorta_file': 'H12.csv',
            'vessels': {
                'LCA': {'wall': 'H12 LCA.csv', 'stream': 'H12 LCA Streamlines.csv'},
            },
        },
    },
    'BG1': {
        'data_id': '0148',
        'category': 'svg',
        'newtonian': {
            'aorta_file': '0148.csv',
            'vessels': {
                'G1': {'wall': '0148 G1.csv', 'stream': '0148 G1 Streamlines.csv'},
                'G2': {'wall': '0148 G2.csv', 'stream': '0148 G2 Streamlines.csv'},
                'G3': {'wall': '0148 G3.csv', 'stream': '0148 G3 Streamlines.csv'},
                'RCA': {'wall': '0148 RCA.csv', 'stream': '0148 RCA Streamlines.csv'},
            },
        },
        'carreau_yasuda': {
            # Old CY run only covers G2.
            'aorta_file': '0148.csv',
            'vessels': {
                'G2': {'wall': '0148 G2.csv', 'stream': '0148 G2 Streamlines.csv'},
            },
        },
    },
    'BG2': {
        'data_id': '0149',
        'category': 'svg',
        'newtonian': {
            'aorta_file': '0149.csv',
            'vessels': {
                'G1': {'wall': '0149 G1.csv', 'stream': '0149 G1 Streamlines.csv'},
                'G2': {'wall': '0149 G2.csv', 'stream': '0149 G2 Streamlines.csv'},
                'G3': {'wall': '0149 G3.csv', 'stream': '0149 G3 Streamlines.csv'},
                'LCA': {'wall': '0149 LCA.csv', 'stream': '0149 LCA Streamlines.csv'},
                'RCA': {'wall': '0149 RCA.csv', 'stream': '0149 RCA Streamlines.csv'},
            },
        },
        'carreau_yasuda': {
            # Old CY run covers G1/G2/G3 only (no LCA/RCA). The folder also
            # contains '0149 Streamlines.csv' (general) and '0149 WSS.csv'
            # which we don't reference -- they aren't per-vessel.
            'aorta_file': '0149.csv',
            'vessels': {
                'G1': {'wall': '0149 G1.csv', 'stream': '0149 G1 Streamlines.csv'},
                'G2': {'wall': '0149 G2.csv', 'stream': '0149 G2 Streamlines.csv'},
                'G3': {'wall': '0149 G3.csv', 'stream': '0149 G3 Streamlines.csv'},
            },
        },
    },
    'BG3': {
        'data_id': '0150',
        'category': 'svg',
        'newtonian': {
            'aorta_file': '0150.csv',
            'vessels': {
                'G1': {'wall': '0150 G1.csv', 'stream': '0150 G1 Streamlines.csv'},
                'G2': {'wall': '0150 G2.csv', 'stream': '0150 G2 Streamlines.csv'},
                'G3': {'wall': '0150 G3.csv', 'stream': '0150 G3 Streamlines.csv'},
                'LCA': {'wall': '0150 LCA.csv', 'stream': '0150 LCA Streamlines.csv'},
                'RCA': {'wall': '0150 RCA.csv', 'stream': '0150 RCA Streamlines.csv'},
            },
        },
        'carreau_yasuda': {
            # Old CY run only covers G3 + a generic '0150 Streamlines.csv'.
            'aorta_file': '0150.csv',
            'vessels': {
                'G3': {'wall': '0150 G3.csv', 'stream': '0150 Streamlines.csv'},
            },
        },
    },
    'BG4': {
        'data_id': '0156',
        'category': 'svg',
        'newtonian': {
            'aorta_file': '0156.csv',
            'vessels': {
                # Note: source file '0156 G1 .csv' has a trailing space.
                'G1': {'wall': '0156 G1 .csv', 'stream': '0156 G1 Streamlines.csv'},
                'G2': {'wall': '0156 G2.csv', 'stream': '0156 G2 Streamlines.csv'},
                'G3': {'wall': '0156 G3.csv', 'stream': '0156 G3 Streamlines.csv'},
                'LCA': {'wall': '0156 LCA.csv', 'stream': '0156 LCA Streamlines.csv'},
                'RCA': {'wall': '0156 RCA.csv', 'stream': '0156 RCA Streamlines.csv'},
            },
        },
        'carreau_yasuda': {
            # Old CY run covers G2, G3 (no G1, LCA, RCA).
            'aorta_file': '0156.csv',
            'vessels': {
                'G2': {'wall': '0156 G2.csv', 'stream': '0156 G2 Streamlines.csv'},
                'G3': {'wall': '0156 G3.csv', 'stream': '0156 G3 Streamlines.csv'},
            },
        },
    },
    'BG5': {
        'data_id': '0157',
        'category': 'svg',
        'newtonian': {
            'aorta_file': '0157.csv',
            'vessels': {
                'G1': {'wall': '0157 G1.csv', 'stream': '0157 G1 Streamlines.csv'},
                'G2': {'wall': '0157 G2.csv', 'stream': '0157 G2 Streamlines.csv'},
                'G3': {'wall': '0157 G3.csv', 'stream': '0157 G3 Streamlines.csv'},
                'LCA': {'wall': '0157 LCA.csv', 'stream': '0157 LCA Streamlines.csv'},
                'RCA': {'wall': '0157 RCA.csv', 'stream': '0157 RCA Streamlines.csv'},
            },
        },
        'carreau_yasuda': {
            'aorta_file': '0157.csv',
            'vessels': {
                'G1': {'wall': '0157 G1.csv', 'stream': '0157 G1 Streamlines.csv'},
                'G2': {'wall': '0157 G2.csv', 'stream': '0157 G2 Streamlines.csv'},
                'G3': {'wall': '0157 G3.csv', 'stream': '0157 G3 Streamlines.csv'},
            },
        },
    },
    'D1': {
        'data_id': 'D1',
        'category': 'diseased',
        'newtonian': {
            'aorta_file': 'D1.csv',
            'vessels': {
                'LCA': {'wall': 'D1 LCA.csv', 'stream': 'D1 LCA Streamlines.csv'},
                'RCA': {'wall': 'D1 RCA.csv', 'stream': 'D1 RCA Streamlines.csv'},
            },
        },
        'carreau_yasuda': {
            # CY directory uses 'D1 .csv' with trailing space for the aorta.
            'aorta_file': 'D1 .csv',
            'vessels': {
                'LCA': {'wall': 'D1 LCA.csv', 'stream': 'D1 LCA Streamlines.csv'},
                'RCA': {'wall': 'D1 RCA.csv', 'stream': 'D1 RCA Streamlines.csv'},
            },
        },
    },
    'D2': {
        'data_id': 'D2',
        'category': 'diseased',
        'newtonian': {
            'aorta_file': 'D2.csv',
            'vessels': {
                'LCA': {'wall': 'D2 LCA.csv', 'stream': 'D2 LCA Streamlines.csv'},
                'RCA': {'wall': 'D2 RCA.csv', 'stream': 'D2 RCA Streamlines.csv'},
            },
        },
        'carreau_yasuda': {
            # CY data was provided under the legacy alias 'ND2'; renamed to D2
            # on import. Old CY run only covers LCA (no RCA), like H4.
            'aorta_file': 'D2.csv',
            'vessels': {
                'LCA': {'wall': 'D2 LCA.csv', 'stream': 'D2 LCA Streamlines.csv'},
            },
        },
    },
    'D3': {
        'data_id': 'D10',
        'category': 'diseased',
        'newtonian': {
            'aorta_file': 'D10.csv',
            'vessels': {
                'LCA': {'wall': 'D10 LCA.csv', 'stream': 'D10 LCA Streamlines.csv'},
                'RCA': {'wall': 'D10 RCA.csv', 'stream': 'D10 RCA Streamlines.csv'},
            },
        },
        'carreau_yasuda': {
            # Old CY run covers both LCA and RCA (full vessel match).
            'aorta_file': 'D10.csv',
            'vessels': {
                'LCA': {'wall': 'D10 LCA.csv', 'stream': 'D10 LCA Streamlines.csv'},
                'RCA': {'wall': 'D10 RCA.csv', 'stream': 'D10 RCA Streamlines.csv'},
            },
        },
    },
}

# Convenience: which public labels have CY ground truth available.
CY_AVAILABLE_LABELS = tuple(
    label for label, info in PATIENT_DATA.items()
    if info.get('carreau_yasuda') is not None
)

# Primary vessels for plotting (by public label).
PRIMARY_VESSELS = {
    'H1':  ['LCA', 'RCA'],
    'H2':  ['LCA', 'RCA'],
    'H3':  ['LCA', 'RCA'],
    'H4':  ['LCA', 'RCA'],
    'BG1': ['G1', 'G2', 'G3'],
    'BG2': ['G1', 'G2', 'G3'],
    'BG3': ['G1', 'G2', 'G3'],
    'BG4': ['G1', 'G2', 'G3'],
    'BG5': ['G1', 'G2', 'G3'],
    'D1':  ['LCA', 'RCA'],
    'D2':  ['LCA', 'RCA'],
    'D3':  ['LCA', 'RCA'],
}


def patient_files(label: str, rheology: str = None) -> dict:
    """Resolve {aorta_file, vessels, data_root} for a patient + rheology.

    The returned dict carries:
        data_root   : Path to the rheology's CSV directory
        aorta_file  : str (relative to data_root)
        vessels     : {vessel_name: {'wall': str, 'stream': str}}

    Raises ValueError if the requested rheology is unavailable for this patient.
    """
    if rheology is None:
        rheology = RHEOLOGY
    if label not in PATIENT_DATA:
        raise KeyError(f"unknown patient label {label!r}; "
                       f"valid: {list(PATIENT_DATA.keys())}")
    info = PATIENT_DATA[label]
    files = info.get(rheology)
    if files is None:
        raise ValueError(
            f"no {rheology!r} CFD ground truth for patient {label} "
            f"(data_id {info['data_id']}); available rheologies for this "
            f"patient: "
            f"{[r for r in ('newtonian', 'carreau_yasuda') if info.get(r)]}"
        )
    data_root = DATA_DIR_BY_RHEOLOGY[rheology]
    return {
        'data_root': data_root,
        'aorta_file': files['aorta_file'],
        'vessels': files['vessels'],
        'category': info['category'],
        'data_id': info['data_id'],
    }
