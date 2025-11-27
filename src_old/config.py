"""
Configuration module for PINN training and evaluation

Contains all hyperparameters, physical constants, and path configurations.
"""

import torch
from pathlib import Path

# =============================================================================
# DEVICE CONFIGURATION
# =============================================================================

# Automatically detect CUDA availability
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Project root (relative to this file: src/../)
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
DATA_ROOT = PROJECT_ROOT / "data"
PINNS_PATH = DATA_ROOT / "PINNS"
STAT_PATH = DATA_ROOT / "statical_data"

# Output paths
OUTPUT_PATH = PROJECT_ROOT / "outputs"
MODEL_PATH = OUTPUT_PATH / "models"
RESULTS_PATH = OUTPUT_PATH / "results"
FIGURES_PATH = OUTPUT_PATH / "figures"
STATISTICS_PATH = OUTPUT_PATH / "statistics"
VISUALIZATIONS_3D_PATH = OUTPUT_PATH / "3d_visualizations"
PLANES_2D_PATH = OUTPUT_PATH / "2d_planes"

# Ensure output directories exist
for path in [MODEL_PATH, RESULTS_PATH, FIGURES_PATH, STATISTICS_PATH,
             VISUALIZATIONS_3D_PATH, PLANES_2D_PATH]:
    path.mkdir(parents=True, exist_ok=True)

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

# Blood properties (Newtonian approximation)
RHO = 1060.0  # Blood density (kg/m³)
MU = 0.0035   # Blood dynamic viscosity (Pa·s)

# Clinical thresholds
HIGH_RISK_WSS_THRESHOLD = 4.0  # Pa (equivalent to 40 dynes/cm²)

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

# PINN layer sizes: [input, hidden1, hidden2, ..., output]
# Output: [u, v, w, p, wss] = 5 neurons
# Increased capacity for 2.4× larger dataset (1.8M points)
PINN_LAYERS = [3, 256, 512, 768, 512, 256, 5]

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================

# Training configuration
BATCH_SIZE = 8192
EPOCHS = 500
LEARNING_RATE = 1e-4

# Data split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Physics loss weights
PHYSICS_WEIGHT_NSE = 1.0   # Navier-Stokes equation weight
PHYSICS_WEIGHT_CONT = 1.0  # Continuity equation weight

# Data supervision weights
DATA_WEIGHT = 1.0           # WSS data fitting weight
VELOCITY_WEIGHT = 1.0       # Velocity supervision weight (from streamlines)

# Collocation points (for pure physics enforcement)
USE_COLLOCATION_POINTS = True  # Whether to sample collocation points
COLLOCATION_POINTS_PER_BATCH = 2048  # Number of random interior points per batch

# Learning rate scheduler
LR_SCHEDULER_FACTOR = 0.5  # Reduce LR by this factor
LR_SCHEDULER_PATIENCE = 20  # Epochs to wait before reducing LR

# Early stopping
EARLY_STOPPING_PATIENCE = 50  # Epochs to wait before stopping

# Random seed for reproducibility
RANDOM_SEED = 42

# =============================================================================
# VISUALIZATION SETTINGS
# =============================================================================

# Plotting style
PLOT_STYLE = 'seaborn-v0_8-darkgrid'
PLOT_DPI = 300

# 3D visualization sampling (for large datasets)
MAX_3D_POINTS = 50000  # Maximum points to plot in 3D

# =============================================================================
# DATA LOADING CONFIGURATION
# =============================================================================

# Default patients to load for training (all 9 with full PINN data)
DEFAULT_TRAINING_PATIENTS = {
    'H-09_Healthy': PINNS_PATH / "H-09.csv",
    'H-12_Healthy': PINNS_PATH / "H-12.csv",
    'D-10_Diseased': PINNS_PATH / "D-10.csv",
    '0149_Graft': PINNS_PATH / "0149.csv",
    '0073_Mixed': PINNS_PATH / "0073.csv",
    '0156_Graft': PINNS_PATH / "0156.csv",
    '0148_Graft': PINNS_PATH / "0148.csv",
    '0150_Graft': PINNS_PATH / "0150.csv",
    'ND2': PINNS_PATH / "ND2.csv"
}

# Streamline file mappings (interior velocity data)
STREAMLINE_FILES = {
    'H-09_Healthy': PINNS_PATH / "H-09 Streamlines.csv",
    'H-12_Healthy': PINNS_PATH / "H-12 LCA Streamlines.csv",
    'D-10_Diseased': [PINNS_PATH / "D-10 LCA Streamlines.csv", PINNS_PATH / "D-10 RCA STreamlines.csv"],
    '0149_Graft': [PINNS_PATH / "0149 G1 Streamlines.csv", PINNS_PATH / "0149 G2 Streamlines.csv", PINNS_PATH / "0149 G3 Streamlines.csv"],
    '0073_Mixed': [PINNS_PATH / "0073 LCA Streamlines.csv", PINNS_PATH / "0073 RCA Streamlines.csv"],
    '0156_Graft': [PINNS_PATH / "0156 G2 Streamlines.csv", PINNS_PATH / "0156 G3 Streamlines.csv"],
    '0148_Graft': PINNS_PATH / "0148 G2 Streamlines.csv",
    '0150_Graft': PINNS_PATH / "0150 Streamlines.csv",
    'ND2': PINNS_PATH / "ND2 lca Streamlines.csv"
}

# Patient-level data split configuration
# Train: 6 patients (~70%), Val: 1 patient (~15%), Test: 2 patients (~15%)
DEFAULT_TRAIN_PATIENTS = ['H-09_Healthy', 'H-12_Healthy', 'D-10_Diseased', '0149_Graft', '0073_Mixed', '0156_Graft']
DEFAULT_VAL_PATIENTS = ['0148_Graft']
DEFAULT_TEST_PATIENTS = ['0150_Graft', 'ND2']

# Column name mappings (for different CSV formats)
COLUMN_MAPPINGS = {
    'X': ['X [ m ]', 'X', 'x'],
    'Y': ['Y [ m ]', 'Y', 'y'],
    'Z': ['Z [ m ]', 'Z', 'z'],
    'WSS': ['Wall Shear [ Pa ]', 'Wall Shear', 'WSS', 'wss'],
    'WSS_X': ['Wall Shear X [ Pa ]', 'WSS_X', 'wss_x'],
    'WSS_Y': ['Wall Shear Y [ Pa ]', 'WSS_Y', 'wss_y'],
    'WSS_Z': ['Wall Shear Z [ Pa ]', 'WSS_Z', 'wss_z'],
    'Velocity': ['Velocity [ m s^-1 ]', 'Velocity', 'vel'],
    'Velocity_u': ['Velocity u [ m s^-1 ]', 'Velocity[i]', 'u'],
    'Velocity_v': ['Velocity v [ m s^-1 ]', 'Velocity[j]', 'v'],
    'Velocity_w': ['Velocity w [ m s^-1 ]', 'Velocity[k]', 'w'],
    'Pressure': ['Pressure [ Pa ]', 'Pressure', 'p']
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_device_info():
    """Print device information"""
    print(f"  Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        import torch.version as tv
        cuda_ver = getattr(tv, 'cuda', 'unknown')
        print(f"  CUDA: {cuda_ver}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("  Device: CPU (CUDA not available)")

def print_config():
    """Print current configuration"""
    print("="*80)
    print("CONFIGURATION")
    print("="*80)
    print(f"\nDevice: {DEVICE}")
    print(f"\nPaths:")
    print(f"  Project Root: {PROJECT_ROOT}")
    print(f"  Data Root: {DATA_ROOT}")
    print(f"  Output Path: {OUTPUT_PATH}")
    print(f"\nPhysical Constants:")
    print(f"  Blood Density (rho): {RHO} kg/m^3")
    print(f"  Blood Viscosity (mu): {MU} Pa.s")
    print(f"  High-Risk WSS Threshold: {HIGH_RISK_WSS_THRESHOLD} Pa")
    print(f"\nTraining Configuration:")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Physics Weights: NSE={PHYSICS_WEIGHT_NSE}, Continuity={PHYSICS_WEIGHT_CONT}")
    print(f"\nModel Architecture:")
    print(f"  PINN Layers: {PINN_LAYERS}")
    print("="*80)
