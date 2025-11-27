#!/usr/bin/env python3
"""
Per-Patient PINN Training Script

This script trains a Physics-Informed Neural Network (PINN) for each patient 
individually, predicting Wall Shear Stress (WSS) from spatial coordinates.

The PINN incorporates:
- Navier-Stokes equations (momentum conservation)
- Continuity equation (mass conservation)  
- WSS physics constraint (wall shear from velocity gradients)

Usage:
    python train_per_patient.py --patient H-12 --epochs 2000
    python train_per_patient.py --patient all --epochs 1000

Author: PINN Hemodynamics Project
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from tqdm import tqdm

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
PROJECT_ROOT = Path(__file__).parent
DATA_PATH = PROJECT_ROOT / "data" / "PINNS"
OUTPUT_PATH = PROJECT_ROOT / "per_patient_outputs"

# Physical constants (blood properties)
RHO = 1060.0   # Blood density (kg/m³)
MU = 0.0035    # Blood dynamic viscosity (Pa·s)

# Patient data configuration with individual vessel files
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
            'Combined': {'wall': '0149.csv', 'stream': '0149 Streamlines.csv'}
        }
    },
    '0073': {
        'category': 'Mixed',
        'vessels': {
            'LCA': {'wall': '0073 LCA.csv', 'stream': '0073 LCA Streamlines.csv'},
            'RCA': {'wall': '0073 RCA.csv', 'stream': '0073 RCA Streamlines.csv'},
            'Combined': {'wall': '0073.csv', 'stream': None}
        }
    },
    '0156': {
        'category': 'SVG',
        'vessels': {
            'G2': {'wall': '0156 G2.csv', 'stream': '0156 G2 Streamlines.csv'},
            'G3': {'wall': '0156 G3.csv', 'stream': '0156 G3 Streamlines.csv'},
            'Combined': {'wall': '0156.csv', 'stream': None}
        }
    },
    '0148': {
        'category': 'SVG',
        'vessels': {
            'G2': {'wall': '0148 G2.csv', 'stream': '0148 G2 Streamlines.csv'},
            'Combined': {'wall': '0148.csv', 'stream': None}
        }
    },
    '0150': {
        'category': 'SVG',
        'vessels': {
            'G3': {'wall': '0150 G3.csv', 'stream': '0150 Streamlines.csv'},
            'Combined': {'wall': '0150.csv', 'stream': None}
        }
    },
    'ND2': {
        'category': 'Unknown',
        'vessels': {
            'LCA': {'wall': 'ND2 LCA.csv', 'stream': 'ND2 lca Streamlines.csv'}
        }
    }
}


# =============================================================================
# DATA LOADING
# =============================================================================

def parse_cfd_csv(filepath: Path) -> Optional[pd.DataFrame]:
    """
    Parse CFD simulation CSV file with [Name]/[Data] format.
    
    CFD files from ANSYS typically have:
    - Lines 1-4: Metadata ([Name] section)
    - Line 5: Column headers
    - Lines 6+: Data
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        DataFrame with parsed data, or None if parsing fails
    """
    if not filepath.exists():
        print(f"    Warning: File not found: {filepath}")
        return None
    
    # Skip first 5 rows (metadata), handle whitespace in column names
    df = pd.read_csv(filepath, skiprows=5, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    
    # Check for required columns
    required_cols = ['X [ m ]', 'Y [ m ]', 'Z [ m ]']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"    Warning: Missing columns in {filepath.name}: {missing}")
        return None
    
    # Convert all columns to numeric, coercing errors to NaN
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with any NaN values
    original_len = len(df)
    df = df.dropna()
    if len(df) < original_len * 0.5:
        print(f"    Warning: Dropped {original_len - len(df)} rows with NaN values")
    
    if len(df) == 0:
        print(f"    Warning: No valid data in {filepath.name}")
        return None
        
    return df


def load_vessel_data(wall_file: str, stream_file: str = None) -> Optional[Dict[str, np.ndarray]]:
    """
    Load data for a single vessel.
    
    Args:
        wall_file: Wall CSV filename
        stream_file: Streamline CSV filename (optional)
        
    Returns:
        Dictionary with vessel data or None if loading fails
    """
    all_X, all_y, all_vel, all_normals, all_has_wss = [], [], [], [], []
    
    # Load wall data
    wall_path = DATA_PATH / wall_file
    if not wall_path.exists():
        print(f"    Warning: Wall file not found: {wall_file}")
        return None
        
    df = parse_cfd_csv(wall_path)
    if df is None:
        return None
        
    # Extract required columns
    X = df[['X [ m ]', 'Y [ m ]', 'Z [ m ]']].values
    y = df['Wall Shear [ Pa ]'].values
    vel = df[['Velocity u [ m s^-1 ]', 'Velocity v [ m s^-1 ]', 
             'Velocity w [ m s^-1 ]']].values
    
    # Compute wall normals from WSS components
    wss_vec = df[['Wall Shear X [ Pa ]', 'Wall Shear Y [ Pa ]', 
                 'Wall Shear Z [ Pa ]']].values
    norms = np.linalg.norm(wss_vec, axis=1, keepdims=True) + 1e-10
    normals = wss_vec / norms
    
    all_X.append(X)
    all_y.append(y)
    all_vel.append(vel)
    all_normals.append(normals)
    all_has_wss.append(np.ones(len(X), dtype=bool))
    
    # Load streamline data if provided
    if stream_file:
        stream_path = DATA_PATH / stream_file
        if stream_path.exists():
            df = parse_cfd_csv(stream_path)
            if df is not None:
                X = df[['X [ m ]', 'Y [ m ]', 'Z [ m ]']].values
                vel = df[['Velocity u [ m s^-1 ]', 'Velocity v [ m s^-1 ]', 
                         'Velocity w [ m s^-1 ]']].values
                
                all_X.append(X)
                all_y.append(np.full(len(X), np.nan))
                all_vel.append(vel)
                all_normals.append(np.zeros((len(X), 3)))
                all_has_wss.append(np.zeros(len(X), dtype=bool))
    
    if len(all_X) == 0:
        return None
    
    return {
        'X': np.vstack(all_X).astype(np.float32),
        'y': np.concatenate(all_y).astype(np.float32),
        'velocity': np.vstack(all_vel).astype(np.float32),
        'normals': np.vstack(all_normals).astype(np.float32),
        'has_wss': np.concatenate(all_has_wss)
    }


def load_patient_data(patient_id: str) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, np.ndarray]]]:
    """
    Load wall and streamline data for a patient.
    
    Args:
        patient_id: Patient identifier (e.g., 'H-12')
        
    Returns:
        Tuple of:
            - Combined data dictionary with keys: 'X', 'y', 'velocity', 'normals', 'has_wss'
            - Per-vessel data dictionary: {vessel_name: data_dict}
    """
    config = PATIENT_DATA[patient_id]
    vessels = config['vessels']
    
    all_X, all_y, all_vel, all_normals, all_has_wss = [], [], [], [], []
    per_vessel_data = {}
    
    for vessel_name, files in vessels.items():
        if vessel_name == 'Combined':
            # Skip combined file for individual vessel analysis
            continue
            
        wall_file = files['wall']
        stream_file = files.get('stream')
        
        vessel_data = load_vessel_data(wall_file, stream_file)
        if vessel_data is not None:
            per_vessel_data[vessel_name] = vessel_data
            
            all_X.append(vessel_data['X'])
            all_y.append(vessel_data['y'])
            all_vel.append(vessel_data['velocity'])
            all_normals.append(vessel_data['normals'])
            all_has_wss.append(vessel_data['has_wss'])
            
            n_wall = vessel_data['has_wss'].sum()
            n_stream = len(vessel_data['X']) - n_wall
            print(f"  {vessel_name}: {n_wall:,} wall + {n_stream:,} streamline = {len(vessel_data['X']):,} points")
    
    # If no individual vessels loaded, try combined file
    if len(all_X) == 0 and 'Combined' in vessels:
        combined = vessels['Combined']
        vessel_data = load_vessel_data(combined['wall'], combined.get('stream'))
        if vessel_data is not None:
            all_X.append(vessel_data['X'])
            all_y.append(vessel_data['y'])
            all_vel.append(vessel_data['velocity'])
            all_normals.append(vessel_data['normals'])
            all_has_wss.append(vessel_data['has_wss'])
            per_vessel_data['Combined'] = vessel_data
            print(f"  Combined: {len(vessel_data['X']):,} points")
    
    if len(all_X) == 0:
        raise ValueError(f"No data loaded for patient {patient_id}")
    
    # Concatenate all data
    X_all = np.vstack(all_X)
    y_all = np.concatenate(all_y)
    vel_all = np.vstack(all_vel)
    normals_all = np.vstack(all_normals)
    has_wss_all = np.concatenate(all_has_wss)
    
    print(f"  Total: {len(X_all):,}")
    
    # Print WSS range for wall points
    valid_wss = y_all[~np.isnan(y_all)]
    if len(valid_wss) > 0:
        print(f"  WSS range: [{valid_wss.min():.2f}, {valid_wss.max():.2f}] Pa")
    
    combined_data = {
        'X': X_all.astype(np.float32),
        'y': y_all.astype(np.float32),
        'velocity': vel_all.astype(np.float32),
        'normals': normals_all.astype(np.float32),
        'has_wss': has_wss_all
    }
    
    return combined_data, per_vessel_data


# =============================================================================
# DATASET
# =============================================================================

class PatientDataset(Dataset):
    """PyTorch Dataset for per-patient PINN training."""
    
    def __init__(self, data: Dict[str, np.ndarray]):
        """
        Initialize dataset with data dictionary.
        
        Args:
            data: Dictionary with 'X', 'y', 'velocity', 'normals', 'has_wss'
        """
        self.X = data['X']
        self.y = data['y']
        self.velocity = data['velocity']
        self.normals = data['normals']
        self.has_wss = data['has_wss']
        
        # Fit scalers
        self.scaler_X = MinMaxScaler(feature_range=(0, 1))
        self.X_scaled = self.scaler_X.fit_transform(self.X)
        
        # WSS scaler (fit only on valid WSS values)
        self.scaler_y = MinMaxScaler(feature_range=(0, 1))
        valid_mask = ~np.isnan(self.y)
        if valid_mask.any():
            self.scaler_y.fit(self.y[valid_mask].reshape(-1, 1))
        
        self.y_scaled = np.zeros_like(self.y)
        if valid_mask.any():
            self.y_scaled[valid_mask] = self.scaler_y.transform(
                self.y[valid_mask].reshape(-1, 1)
            ).flatten()
        
        # Velocity scaler
        self.scaler_vel = MinMaxScaler(feature_range=(-1, 1))
        self.vel_scaled = self.scaler_vel.fit_transform(self.velocity)
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'coords': torch.FloatTensor(self.X_scaled[idx]),
            'coords_raw': torch.FloatTensor(self.X[idx]),
            'wss': torch.FloatTensor([self.y_scaled[idx]]),
            'wss_raw': torch.FloatTensor([self.y[idx]]),
            'velocity': torch.FloatTensor(self.vel_scaled[idx]),
            'velocity_raw': torch.FloatTensor(self.velocity[idx]),
            'normals': torch.FloatTensor(self.normals[idx]),
            'has_wss': torch.BoolTensor([self.has_wss[idx]])
        }


def create_dataloader(data: Dict[str, np.ndarray], batch_size: int = 4096
                     ) -> Tuple[DataLoader, PatientDataset]:
    """Create DataLoader from data dictionary."""
    dataset = PatientDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                       num_workers=0, pin_memory=True)
    return loader, dataset


# =============================================================================
# PINN MODEL - Multi-Model Architecture with Swish & Self-Adaptive Weights
# =============================================================================

class Swish(nn.Module):
    """
    Swish activation: x * sigmoid(β * x), with learnable β parameter.
    
    Swish often outperforms ReLU and Tanh for deep networks.
    Reference: Ramachandran et al., "Searching for Activation Functions" (2017)
    """
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(beta))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.beta * x)


def kaiming_init(m: nn.Module):
    """Apply Kaiming (He) initialization for better gradient flow."""
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class BasePINN(nn.Module):
    """
    Base PINN class with Swish activation and Kaiming initialization.
    
    Architecture: Fully-connected layers with optional batch normalization.
    """
    def __init__(self, in_dim: int = 3, hidden_dims: list = None, out_dim: int = 1,
                 use_batch_norm: bool = False):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 512, 512, 256]
        
        layers = []
        prev_dim = in_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(Swish())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, out_dim))
        
        self.net = nn.Sequential(*layers)
        self.apply(kaiming_init)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiPINN(nn.Module):
    """
    Multi-Model PINN Architecture for Hemodynamics.
    
    Separate neural networks for each output variable:
    - Velocity components: u, v, w
    - Pressure: p
    - Wall Shear Stress: wss
    
    Includes self-adaptive loss weighting (learnable log-λ parameters).
    """
    
    def __init__(self, hidden_dims: list = None, use_batch_norm: bool = False):
        """
        Initialize Multi-PINN.
        
        Args:
            hidden_dims: List of hidden layer sizes. Default: [256, 512, 512, 256]
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 512, 512, 256]
        
        # Separate networks for each output variable
        self.net_u = BasePINN(in_dim=3, hidden_dims=hidden_dims, out_dim=1, 
                              use_batch_norm=use_batch_norm)
        self.net_v = BasePINN(in_dim=3, hidden_dims=hidden_dims, out_dim=1,
                              use_batch_norm=use_batch_norm)
        self.net_w = BasePINN(in_dim=3, hidden_dims=hidden_dims, out_dim=1,
                              use_batch_norm=use_batch_norm)
        self.net_p = BasePINN(in_dim=3, hidden_dims=hidden_dims, out_dim=1,
                              use_batch_norm=use_batch_norm)
        self.net_wss = BasePINN(in_dim=3, hidden_dims=hidden_dims, out_dim=1,
                                use_batch_norm=use_batch_norm)
        
        self.hidden_dims = hidden_dims
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through all networks.
        
        Args:
            x: Input coordinates (batch, 3)
            
        Returns:
            Dictionary with 'u', 'v', 'w', 'p', 'wss' tensors
        """
        return {
            'u': self.net_u(x),
            'v': self.net_v(x),
            'w': self.net_w(x),
            'p': self.net_p(x),
            'wss': self.net_wss(x)
        }
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# RESNET PINN - Residual Connections for Better Gradient Flow
# =============================================================================

class ResidualBlock(nn.Module):
    """
    Residual block with skip connection: output = F(x) + x
    
    Helps gradient flow in deeper networks and enables training of very deep PINNs.
    Reference: He et al., "Deep Residual Learning for Image Recognition" (2015)
    """
    def __init__(self, dim: int, use_batch_norm: bool = False):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.activation = Swish()
        self.use_batch_norm = use_batch_norm
        
        if use_batch_norm:
            self.bn1 = nn.BatchNorm1d(dim)
            self.bn2 = nn.BatchNorm1d(dim)
        
        # Initialize for residual learning
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.fc1(x)
        if self.use_batch_norm:
            out = self.bn1(out)
        out = self.activation(out)
        
        out = self.fc2(out)
        if self.use_batch_norm:
            out = self.bn2(out)
        
        out = out + identity  # Skip connection
        out = self.activation(out)
        return out


class ResNetPINN(nn.Module):
    """
    ResNet-style PINN with residual blocks for better gradient flow.
    
    Architecture:
    - Input projection: 3 → hidden_dim
    - N residual blocks (each with skip connections)
    - Output projection: hidden_dim → 1
    """
    def __init__(self, hidden_dim: int = 256, num_blocks: int = 4, 
                 use_batch_norm: bool = False):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(3, hidden_dim)
        self.input_activation = Swish()
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, use_batch_norm) 
            for _ in range(num_blocks)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, 1)
        
        # Initialize
        nn.init.kaiming_normal_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.kaiming_normal_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_activation(self.input_proj(x))
        
        for block in self.res_blocks:
            x = block(x)
        
        return self.output_proj(x)


class MultiResNetPINN(nn.Module):
    """
    Multi-Model ResNet PINN Architecture for Hemodynamics.
    
    Uses ResNet-style networks with residual connections for each output.
    Better gradient flow enables training deeper networks.
    """
    
    def __init__(self, hidden_dim: int = 256, num_blocks: int = 4,
                 use_batch_norm: bool = False):
        """
        Initialize Multi-ResNet-PINN.
        
        Args:
            hidden_dim: Hidden layer dimension (default: 256)
            num_blocks: Number of residual blocks per network (default: 4)
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()
        
        # Separate ResNet networks for each output variable
        self.net_u = ResNetPINN(hidden_dim, num_blocks, use_batch_norm)
        self.net_v = ResNetPINN(hidden_dim, num_blocks, use_batch_norm)
        self.net_w = ResNetPINN(hidden_dim, num_blocks, use_batch_norm)
        self.net_p = ResNetPINN(hidden_dim, num_blocks, use_batch_norm)
        self.net_wss = ResNetPINN(hidden_dim, num_blocks, use_batch_norm)
        
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            'u': self.net_u(x),
            'v': self.net_v(x),
            'w': self.net_w(x),
            'p': self.net_p(x),
            'wss': self.net_wss(x)
        }
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Legacy single-model PINN (kept for compatibility)
class PINN(nn.Module):
    """
    Legacy single-network PINN (kept for backward compatibility).
    Now uses Swish activation and Kaiming initialization.
    """
    
    def __init__(self, layers: list = None):
        super().__init__()
        
        if layers is None:
            layers = [3, 256, 512, 512, 256, 5]
        
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
        
        self.activation = Swish()  # Changed from Tanh to Swish
        self.layer_sizes = layers
        
        # Kaiming initialization
        for layer in self.layers:
            nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='linear')
            nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        
        return {
            'u': x[:, 0:1],
            'v': x[:, 1:2],
            'w': x[:, 2:3],
            'p': x[:, 3:4],
            'wss': x[:, 4:5]
        }


# =============================================================================
# PHYSICS EQUATIONS
# =============================================================================

def compute_gradients(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """Compute gradients of outputs with respect to inputs."""
    return torch.autograd.grad(
        outputs, inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True, retain_graph=True
    )[0]


def navier_stokes_residual(model: nn.Module, coords: torch.Tensor, 
                           coord_scale: torch.Tensor
                          ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute Navier-Stokes momentum equation residuals.
    
    Steady incompressible NS: ρ(u·∇u) = -∇p + μ∇²u
    """
    coords = coords.requires_grad_(True)
    out = model(coords)
    u, v, w, p = out['u'], out['v'], out['w'], out['p']
    
    # First derivatives
    u_g = compute_gradients(u, coords) * coord_scale
    v_g = compute_gradients(v, coords) * coord_scale
    w_g = compute_gradients(w, coords) * coord_scale
    p_g = compute_gradients(p, coords) * coord_scale
    
    # Second derivatives
    u_xx = compute_gradients(u_g[:, 0:1], coords)[:, 0:1] * coord_scale[:, 0:1]
    u_yy = compute_gradients(u_g[:, 1:2], coords)[:, 1:2] * coord_scale[:, 1:2]
    u_zz = compute_gradients(u_g[:, 2:3], coords)[:, 2:3] * coord_scale[:, 2:3]
    
    v_xx = compute_gradients(v_g[:, 0:1], coords)[:, 0:1] * coord_scale[:, 0:1]
    v_yy = compute_gradients(v_g[:, 1:2], coords)[:, 1:2] * coord_scale[:, 1:2]
    v_zz = compute_gradients(v_g[:, 2:3], coords)[:, 2:3] * coord_scale[:, 2:3]
    
    w_xx = compute_gradients(w_g[:, 0:1], coords)[:, 0:1] * coord_scale[:, 0:1]
    w_yy = compute_gradients(w_g[:, 1:2], coords)[:, 1:2] * coord_scale[:, 1:2]
    w_zz = compute_gradients(w_g[:, 2:3], coords)[:, 2:3] * coord_scale[:, 2:3]
    
    # Residuals
    f_u = RHO * (u * u_g[:, 0:1] + v * u_g[:, 1:2] + w * u_g[:, 2:3]) + \
          p_g[:, 0:1] - MU * (u_xx + u_yy + u_zz)
    f_v = RHO * (u * v_g[:, 0:1] + v * v_g[:, 1:2] + w * v_g[:, 2:3]) + \
          p_g[:, 1:2] - MU * (v_xx + v_yy + v_zz)
    f_w = RHO * (u * w_g[:, 0:1] + v * w_g[:, 1:2] + w * w_g[:, 2:3]) + \
          p_g[:, 2:3] - MU * (w_xx + w_yy + w_zz)
    
    return f_u, f_v, f_w


def continuity_residual(model: nn.Module, coords: torch.Tensor, 
                        coord_scale: torch.Tensor) -> torch.Tensor:
    """Compute continuity equation residual: ∇·u = 0."""
    coords = coords.requires_grad_(True)
    out = model(coords)
    
    u_g = compute_gradients(out['u'], coords) * coord_scale
    v_g = compute_gradients(out['v'], coords) * coord_scale
    w_g = compute_gradients(out['w'], coords) * coord_scale
    
    return u_g[:, 0:1] + v_g[:, 1:2] + w_g[:, 2:3]


def wss_physics_residual(model: nn.Module, coords: torch.Tensor, 
                         normals: torch.Tensor, coord_scale: torch.Tensor
                        ) -> torch.Tensor:
    """Compute WSS physics residual: WSS = μ|∂u/∂n|."""
    coords = coords.requires_grad_(True)
    out = model(coords)
    
    u_g = compute_gradients(out['u'], coords) * coord_scale
    v_g = compute_gradients(out['v'], coords) * coord_scale
    w_g = compute_gradients(out['w'], coords) * coord_scale
    
    du_dn = (u_g * normals).sum(dim=1, keepdim=True)
    dv_dn = (v_g * normals).sum(dim=1, keepdim=True)
    dw_dn = (w_g * normals).sum(dim=1, keepdim=True)
    
    wss_physics = MU * torch.sqrt(du_dn**2 + dv_dn**2 + dw_dn**2 + 1e-8)
    return out['wss'] - wss_physics


# =============================================================================
# PLOTTING - Publication Quality Side-by-Side with Error
# =============================================================================

plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'axes.linewidth': 0.8,
    'axes.grid': False,
})


def plot_training_history(history: Dict, patient_id: str, save_path: Path):
    """Plot training convergence curves."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    epochs = range(1, len(history['train_loss']) + 1)
    
    axes[0].semilogy(epochs, history['train_loss'], 'b-', lw=1.5)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Total Loss')
    axes[0].set_title('Training Loss')
    
    axes[1].semilogy(epochs, history['wss_loss'], 'g-', lw=1.5)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('WSS Loss')
    axes[1].set_title('WSS Data Loss')
    
    axes[2].semilogy(epochs, history['physics_loss'], 'm-', lw=1.5)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Physics Loss')
    axes[2].set_title('Physics Residual')
    
    plt.tight_layout()
    plt.savefig(save_path / f'{patient_id}_training.png', dpi=300, bbox_inches='tight')
    plt.close()


def compute_nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Normalized RMSE: RMSE / (max - min)."""
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    data_range = np.max(y_true) - np.min(y_true) + 1e-10
    return rmse / data_range


def plot_wss_comparison(coords: np.ndarray, wss_true: np.ndarray, wss_pred: np.ndarray,
                        patient_id: str, save_path: Path, view: str,
                        x_idx: int, y_idx: int, xlabel: str, ylabel: str):
    """
    Create side-by-side WSS comparison: CFD vs PINN vs Error with NRMSE.
    
    Args:
        coords: Spatial coordinates (N, 3)
        wss_true: Ground truth WSS (N,)
        wss_pred: Predicted WSS (N,)
        patient_id: Patient identifier
        save_path: Directory to save figure
        view: View name (XY, XZ, YZ)
        x_idx, y_idx: Coordinate indices for the view
        xlabel, ylabel: Axis labels
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Compute metrics
    rmse = np.sqrt(np.mean((wss_pred - wss_true) ** 2))
    nrmse = compute_nrmse(wss_true, wss_pred)
    mae = np.mean(np.abs(wss_pred - wss_true))
    
    # Common colorbar limits for CFD and PINN
    vmax = min(5, max(wss_true.max(), wss_pred.max()))
    
    # CFD (Ground Truth)
    sc1 = axes[0].scatter(coords[:, x_idx] * 1000, coords[:, y_idx] * 1000,
                          c=wss_true, cmap='jet', s=0.3, vmin=0, vmax=vmax)
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    axes[0].set_title('CFD')
    axes[0].set_aspect('equal')
    plt.colorbar(sc1, ax=axes[0], shrink=0.7, label='WSS (Pa)')
    
    # PINN Prediction
    sc2 = axes[1].scatter(coords[:, x_idx] * 1000, coords[:, y_idx] * 1000,
                          c=wss_pred, cmap='jet', s=0.3, vmin=0, vmax=vmax)
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel(ylabel)
    axes[1].set_title('PINN')
    axes[1].set_aspect('equal')
    plt.colorbar(sc2, ax=axes[1], shrink=0.7, label='WSS (Pa)')
    
    # Absolute Error with metrics
    error = np.abs(wss_pred - wss_true)
    sc3 = axes[2].scatter(coords[:, x_idx] * 1000, coords[:, y_idx] * 1000,
                          c=error, cmap='Reds', s=0.3, vmin=0, vmax=2)
    axes[2].set_xlabel(xlabel)
    axes[2].set_ylabel(ylabel)
    axes[2].set_title(f'|Error|\nNRMSE={nrmse:.4f}, RMSE={rmse:.2f} Pa')
    axes[2].set_aspect('equal')
    plt.colorbar(sc3, ax=axes[2], shrink=0.7, label='|Error| (Pa)')
    
    plt.tight_layout()
    plt.savefig(save_path / f'{patient_id}_WSS_{view}.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_velocity_comparison(coords: np.ndarray, vel_true: np.ndarray, vel_pred: np.ndarray,
                             patient_id: str, save_path: Path, view: str,
                             x_idx: int, y_idx: int, xlabel: str, ylabel: str,
                             component: str, comp_idx: int):
    """Create side-by-side velocity comparison: CFD vs PINN vs Error with NRMSE."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    v_true = vel_true[:, comp_idx]
    v_pred = vel_pred[:, comp_idx]
    
    # Compute metrics
    rmse = np.sqrt(np.mean((v_pred - v_true) ** 2))
    nrmse = compute_nrmse(v_true, v_pred)
    
    # Symmetric colorbar
    vmax = max(abs(v_true.min()), abs(v_true.max()), abs(v_pred.min()), abs(v_pred.max()))
    vmin = -vmax
    
    # CFD
    sc1 = axes[0].scatter(coords[:, x_idx] * 1000, coords[:, y_idx] * 1000,
                          c=v_true, cmap='RdBu_r', s=0.3, vmin=vmin, vmax=vmax)
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    axes[0].set_title('CFD')
    axes[0].set_aspect('equal')
    plt.colorbar(sc1, ax=axes[0], shrink=0.7, label=f'{component} (m/s)')
    
    # PINN
    sc2 = axes[1].scatter(coords[:, x_idx] * 1000, coords[:, y_idx] * 1000,
                          c=v_pred, cmap='RdBu_r', s=0.3, vmin=vmin, vmax=vmax)
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel(ylabel)
    axes[1].set_title('PINN')
    axes[1].set_aspect('equal')
    plt.colorbar(sc2, ax=axes[1], shrink=0.7, label=f'{component} (m/s)')
    
    # Error with metrics
    error = np.abs(v_pred - v_true)
    sc3 = axes[2].scatter(coords[:, x_idx] * 1000, coords[:, y_idx] * 1000,
                          c=error, cmap='Reds', s=0.3, vmin=0, vmax=vmax * 0.3)
    axes[2].set_xlabel(xlabel)
    axes[2].set_ylabel(ylabel)
    axes[2].set_title(f'|Error|\nNRMSE={nrmse:.4f}, RMSE={rmse:.4f} m/s')
    axes[2].set_aspect('equal')
    plt.colorbar(sc3, ax=axes[2], shrink=0.7, label=f'|Error| (m/s)')
    
    plt.tight_layout()
    plt.savefig(save_path / f'{patient_id}_vel_{component}_{view}.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_all_plots(model: nn.Module, data_loader: DataLoader, dataset: PatientDataset,
                       patient_id: str, save_path: Path, metrics: Dict,
                       per_vessel_data: Dict[str, Dict[str, np.ndarray]] = None):
    """Generate all publication-quality comparison plots, including per-vessel."""
    model.eval()
    
    all_coords, all_wss_true, all_wss_pred = [], [], []
    all_velocity_true, all_velocity_pred = [], []
    all_coords_full = []
    
    with torch.no_grad():
        for batch in data_loader:
            coords = batch['coords'].to(DEVICE)
            coords_raw = batch['coords_raw'].numpy()
            wss_raw = batch['wss_raw'].numpy().flatten()
            vel_scaled = batch['velocity'].numpy()
            has_wss = batch['has_wss'].numpy().squeeze().astype(bool)
            
            outputs = model(coords)
            
            # WSS prediction
            wss_pred_scaled = outputs['wss'].cpu().numpy()
            wss_pred = dataset.scaler_y.inverse_transform(wss_pred_scaled).flatten()
            
            # Velocity prediction (inverse transform using MinMaxScaler attributes)
            vel_pred_scaled = torch.cat([outputs['u'], outputs['v'], outputs['w']], dim=1).cpu().numpy()
            vel_pred = dataset.scaler_vel.inverse_transform(vel_pred_scaled)
            vel_true = dataset.scaler_vel.inverse_transform(vel_scaled)
            
            if has_wss.any():
                all_coords.append(coords_raw[has_wss])
                all_wss_true.append(wss_raw[has_wss])
                all_wss_pred.append(wss_pred[has_wss])
            
            all_coords_full.append(coords_raw)
            all_velocity_true.append(vel_true)
            all_velocity_pred.append(vel_pred)
    
    # Concatenate
    coords_wall = np.concatenate(all_coords)
    wss_true = np.concatenate(all_wss_true)
    wss_pred = np.concatenate(all_wss_pred)
    coords_full = np.concatenate(all_coords_full)
    vel_true = np.concatenate(all_velocity_true)
    vel_pred = np.concatenate(all_velocity_pred)
    
    # View configurations
    views = [
        ('XY', 0, 1, 'X (mm)', 'Y (mm)'),
        ('XZ', 0, 2, 'X (mm)', 'Z (mm)'),
        ('YZ', 1, 2, 'Y (mm)', 'Z (mm)')
    ]
    
    print("  Generating WSS comparison plots...")
    for view, x_idx, y_idx, xlabel, ylabel in views:
        plot_wss_comparison(coords_wall, wss_true, wss_pred, patient_id, save_path,
                           view, x_idx, y_idx, xlabel, ylabel)
    
    print("  Generating velocity comparison plots...")
    for comp, comp_idx in [('u', 0), ('v', 1), ('w', 2)]:
        for view, x_idx, y_idx, xlabel, ylabel in views:
            plot_velocity_comparison(coords_full, vel_true, vel_pred, patient_id, save_path,
                                    view, x_idx, y_idx, xlabel, ylabel, comp, comp_idx)
    
    # Generate per-vessel WSS plots if vessel data available
    if per_vessel_data is not None and len(per_vessel_data) > 1:
        print(f"  Generating per-vessel comparison plots ({len(per_vessel_data)} vessels)...")
        generate_per_vessel_plots(model, per_vessel_data, dataset, patient_id, save_path)


def generate_per_vessel_plots(model: nn.Module, per_vessel_data: Dict[str, Dict[str, np.ndarray]],
                              dataset: PatientDataset, patient_id: str, save_path: Path):
    """
    Generate WSS comparison plots for each individual vessel.
    
    Args:
        model: Trained PINN model
        per_vessel_data: Dictionary {vessel_name: vessel_data_dict}
        dataset: PatientDataset for scalers
        patient_id: Patient identifier
        save_path: Output directory
    """
    model.eval()
    
    views = [
        ('XY', 0, 1, 'X (mm)', 'Y (mm)'),
        ('XZ', 0, 2, 'X (mm)', 'Z (mm)'),
        ('YZ', 1, 2, 'Y (mm)', 'Z (mm)')
    ]
    
    for vessel_name, vessel_data in per_vessel_data.items():
        if vessel_name == 'Combined':
            continue
        
        # Get wall points with WSS
        has_wss = vessel_data['has_wss']
        if not has_wss.any():
            continue
        
        coords_raw = vessel_data['X'][has_wss]
        wss_true = vessel_data['y'][has_wss]
        
        # Skip if too few points
        if len(coords_raw) < 100:
            print(f"    Skipping {vessel_name}: only {len(coords_raw)} points")
            continue
        
        # Scale coordinates for model input
        coords_scaled = dataset.scaler_X.transform(coords_raw)
        coords_tensor = torch.FloatTensor(coords_scaled).to(DEVICE)
        
        # Get predictions
        with torch.no_grad():
            outputs = model(coords_tensor)
            wss_pred_scaled = outputs['wss'].cpu().numpy()
            wss_pred = dataset.scaler_y.inverse_transform(wss_pred_scaled).flatten()
        
        # Compute metrics for this vessel
        rmse = np.sqrt(np.mean((wss_pred - wss_true) ** 2))
        nrmse = compute_nrmse(wss_true, wss_pred)
        r2 = 1 - np.sum((wss_pred - wss_true)**2) / (np.sum((wss_true - np.mean(wss_true))**2) + 1e-10)
        
        print(f"    {vessel_name}: {len(coords_raw):,} points | RMSE={rmse:.4f} Pa | NRMSE={nrmse:.4f} | R²={r2:.4f}")
        
        # Generate plots for each view
        for view, x_idx, y_idx, xlabel, ylabel in views:
            plot_vessel_wss_comparison(coords_raw, wss_true, wss_pred,
                                       patient_id, vessel_name, save_path,
                                       view, x_idx, y_idx, xlabel, ylabel)


def plot_vessel_wss_comparison(coords: np.ndarray, wss_true: np.ndarray, wss_pred: np.ndarray,
                               patient_id: str, vessel_name: str, save_path: Path, 
                               view: str, x_idx: int, y_idx: int, xlabel: str, ylabel: str):
    """
    Create side-by-side WSS comparison for a specific vessel: CFD vs PINN vs Error.
    
    Args:
        coords: Spatial coordinates (N, 3)
        wss_true: Ground truth WSS (N,)
        wss_pred: Predicted WSS (N,)
        patient_id: Patient identifier
        vessel_name: Vessel name (LCA, RCA, G1, G2, G3, etc.)
        save_path: Directory to save figure
        view: View name (XY, XZ, YZ)
        x_idx, y_idx: Coordinate indices for the view
        xlabel, ylabel: Axis labels
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Compute metrics
    rmse = np.sqrt(np.mean((wss_pred - wss_true) ** 2))
    nrmse = compute_nrmse(wss_true, wss_pred)
    r2 = 1 - np.sum((wss_pred - wss_true)**2) / (np.sum((wss_true - np.mean(wss_true))**2) + 1e-10)
    
    # Common colorbar limits for CFD and PINN
    vmax = min(5, max(wss_true.max(), wss_pred.max()))
    
    # CFD (Ground Truth)
    sc1 = axes[0].scatter(coords[:, x_idx] * 1000, coords[:, y_idx] * 1000,
                          c=wss_true, cmap='jet', s=0.5, vmin=0, vmax=vmax)
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    axes[0].set_title('CFD')
    axes[0].set_aspect('equal')
    plt.colorbar(sc1, ax=axes[0], shrink=0.7, label='WSS (Pa)')
    
    # PINN Prediction
    sc2 = axes[1].scatter(coords[:, x_idx] * 1000, coords[:, y_idx] * 1000,
                          c=wss_pred, cmap='jet', s=0.5, vmin=0, vmax=vmax)
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel(ylabel)
    axes[1].set_title('PINN')
    axes[1].set_aspect('equal')
    plt.colorbar(sc2, ax=axes[1], shrink=0.7, label='WSS (Pa)')
    
    # Absolute Error with metrics
    error = np.abs(wss_pred - wss_true)
    sc3 = axes[2].scatter(coords[:, x_idx] * 1000, coords[:, y_idx] * 1000,
                          c=error, cmap='Reds', s=0.5, vmin=0, vmax=2)
    axes[2].set_xlabel(xlabel)
    axes[2].set_ylabel(ylabel)
    axes[2].set_title(f'|Error|\nNRMSE={nrmse:.4f}, R²={r2:.4f}')
    axes[2].set_aspect('equal')
    plt.colorbar(sc3, ax=axes[2], shrink=0.7, label='|Error| (Pa)')
    
    plt.tight_layout()
    plt.savefig(save_path / f'{patient_id}_{vessel_name}_WSS_{view}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()


# =============================================================================
# TRAINING
# =============================================================================

def setup_paths(patient_id: str) -> Dict[str, Path]:
    """Create output directories for a patient."""
    base = OUTPUT_PATH / patient_id
    paths = {
        'base': base,
        'models': base / 'models',
        'figures': base / 'figures',
        'results': base / 'results'
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def train_patient(patient_id: str, epochs: int = 300, batch_size: int = 4096,
                  learning_rate: float = 1e-4, verbose: bool = True,
                  arch: str = 'multi', hidden_dim: int = 256, num_blocks: int = 4
                 ) -> Tuple[nn.Module, Dict]:
    """
    Train PINN for a single patient.
    
    Args:
        patient_id: Patient identifier
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Initial learning rate
        verbose: Print training progress
        arch: Architecture type ('multi', 'resnet', 'single')
        hidden_dim: Hidden layer dimension
        num_blocks: Number of ResNet blocks (for resnet arch)
        
    Returns:
        Tuple of (trained model, results dictionary)
    """
    paths = setup_paths(patient_id)
    
    print("\n" + "=" * 80)
    print(f"TRAINING PINN FOR PATIENT: {patient_id}")
    print(f"Category: {PATIENT_DATA[patient_id]['category']}")
    print("=" * 80)
    
    # Load data (returns combined + per-vessel data)
    print("\n[LOADING DATA]")
    data, per_vessel_data = load_patient_data(patient_id)
    
    # Create dataloader
    print("\n[CREATING DATALOADER]")
    data_loader, dataset = create_dataloader(data, batch_size=batch_size)
    print(f"  Total points: {len(dataset):,}")
    
    # Initialize model based on architecture
    if arch == 'resnet':
        model = MultiResNetPINN(hidden_dim=hidden_dim, num_blocks=num_blocks).to(DEVICE)
        print(f"\n[MODEL] MultiResNetPINN with Swish activation")
        print(f"  Architecture: 5 ResNet networks [3 → {hidden_dim}] × {num_blocks} residual blocks → 1")
        print(f"  Parameters: {model.count_parameters():,}")
    elif arch == 'multi':
        hidden_dims = [hidden_dim, hidden_dim * 2, hidden_dim * 2, hidden_dim]
        model = MultiPINN(hidden_dims=hidden_dims).to(DEVICE)
        print(f"\n[MODEL] MultiPINN with Swish activation")
        print(f"  Architecture: 5 separate networks [3 → {' → '.join(map(str, hidden_dims))} → 1]")
        print(f"  Parameters: {model.count_parameters():,}")
    else:  # single
        model = PINN().to(DEVICE)
        print(f"\n[MODEL] Single PINN with Swish activation")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    # Physics scaling
    coord_scale = torch.tensor(dataset.scaler_X.data_range_, dtype=torch.float32, 
                               device=DEVICE).view(1, 3)
    U_char, L_char = 0.1, 0.05
    nse_scale = (RHO * U_char**2) / L_char
    cont_scale = U_char / L_char
    
    # Collocation points setup
    n_collocation = 2048
    collocation_bounds = {
        'x_min': data['X'][:, 0].min(), 'x_max': data['X'][:, 0].max(),
        'y_min': data['X'][:, 1].min(), 'y_max': data['X'][:, 1].max(),
        'z_min': data['X'][:, 2].min(), 'z_max': data['X'][:, 2].max()
    }
    print("\n[COLLOCATION POINTS]")
    print(f"  Domain: X=[{collocation_bounds['x_min']:.4f}, {collocation_bounds['x_max']:.4f}]")
    print(f"  Sampling: {n_collocation} random interior points per batch")
    
    # Training history
    history = {'train_loss': [], 'wss_loss': [], 'physics_loss': [], 'lr': []}
    best_loss = float('inf')
    
    print(f"\n[TRAINING] {epochs} epochs")
    print("-" * 80)
    
    for epoch in range(epochs):
        model.train()
        epoch_loss, epoch_wss, epoch_physics = 0, 0, 0
        
        pbar = tqdm(data_loader, desc=f"Epoch {epoch + 1:3d}/{epochs}", 
                   disable=not verbose, leave=False)
        
        for batch in pbar:
            coords = batch['coords'].to(DEVICE)
            wss_true = batch['wss'].to(DEVICE)
            vel_true = batch['velocity'].to(DEVICE)
            normals = batch['normals'].to(DEVICE)
            has_wss = batch['has_wss'].to(DEVICE).squeeze()
            
            optimizer.zero_grad()
            
            outputs = model(coords)
            vel_pred = torch.cat([outputs['u'], outputs['v'], outputs['w']], dim=1)
            
            # Data losses
            loss_wss_data = nn.MSELoss()(outputs['wss'][has_wss], wss_true[has_wss]) \
                       if has_wss.any() else torch.tensor(0.0, device=DEVICE)
            loss_vel = nn.MSELoss()(vel_pred, vel_true)
            
            # Physics at data points
            f_u, f_v, f_w = navier_stokes_residual(model, coords, coord_scale)
            cont = continuity_residual(model, coords, coord_scale)
            loss_nse = (f_u**2 + f_v**2 + f_w**2).mean() / (nse_scale**2)
            loss_cont = (cont**2).mean() / (cont_scale**2)
            
            # Collocation points
            colloc_coords_np = np.random.uniform(
                low=[collocation_bounds['x_min'], collocation_bounds['y_min'], 
                     collocation_bounds['z_min']],
                high=[collocation_bounds['x_max'], collocation_bounds['y_max'], 
                      collocation_bounds['z_max']],
                size=(n_collocation, 3)
            ).astype(np.float32)
            colloc_coords_scaled = dataset.scaler_X.transform(colloc_coords_np)
            colloc_coords = torch.FloatTensor(colloc_coords_scaled).to(DEVICE)
            
            # Physics at collocation points
            f_u_c, f_v_c, f_w_c = navier_stokes_residual(model, colloc_coords, coord_scale)
            cont_c = continuity_residual(model, colloc_coords, coord_scale)
            loss_nse += (f_u_c**2 + f_v_c**2 + f_w_c**2).mean() / (nse_scale**2)
            loss_cont += (cont_c**2).mean() / (cont_scale**2)
            
            # WSS physics constraint
            if has_wss.any():
                wss_res = wss_physics_residual(model, coords[has_wss], 
                                               normals[has_wss], coord_scale)
                loss_wss_phys = (wss_res**2).mean()
            else:
                loss_wss_phys = torch.tensor(0.0, device=DEVICE)
            
            # Total loss with fixed weights
            # WSS data loss (primary), velocity loss, physics losses, WSS physics constraint
            loss = loss_wss_data + 0.1 * loss_vel + loss_nse + loss_cont + 0.1 * loss_wss_phys
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_wss += loss_wss_data.item()
            epoch_physics += (loss_nse.item() + loss_cont.item())
            
            pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'WSS': f'{loss_wss_data.item():.4f}'})
        
        scheduler.step()
        
        n_batches = len(data_loader)
        epoch_loss /= n_batches
        epoch_wss /= n_batches
        epoch_physics /= n_batches
        
        history['train_loss'].append(epoch_loss)
        history['wss_loss'].append(epoch_wss)
        history['physics_loss'].append(epoch_physics)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), paths['models'] / f"pinn_{patient_id}_best.pth")
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch + 1:3d}/{epochs} | Loss: {epoch_loss:.4f} | "
                  f"WSS: {epoch_wss:.4f} | Physics: {epoch_physics:.2e} | LR: {lr:.2e}")
    
    # Load best model
    model.load_state_dict(torch.load(paths['models'] / f"pinn_{patient_id}_best.pth", 
                                     weights_only=True))
    
    # Evaluation
    print("\n[EVALUATION]")
    metrics, y_true, y_pred, coords_all = evaluate_patient(model, data_loader, dataset)
    
    # Generate plots
    print("\n[GENERATING PLOTS]")
    plot_training_history(history, patient_id, paths['figures'])
    generate_all_plots(model, data_loader, dataset, patient_id, paths['figures'], 
                       metrics, per_vessel_data)
    print(f"  Saved to: {paths['figures']}")
    
    # Save results as readable text file
    results = {
        'history': history, 
        'metrics': metrics, 
        'patient_id': patient_id,
        'category': PATIENT_DATA[patient_id]['category']
    }
    
    results_file = paths['results'] / f'{patient_id}_results.txt'
    with open(results_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write(f"PINN TRAINING RESULTS - PATIENT {patient_id}\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Patient ID: {patient_id}\n")
        f.write(f"Category: {PATIENT_DATA[patient_id]['category']}\n\n")
        
        f.write("-" * 40 + "\n")
        f.write("EVALUATION METRICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"  RMSE:     {metrics['RMSE']:.4f} Pa\n")
        f.write(f"  MSE:      {metrics['MSE']:.6f} Pa²\n")
        f.write(f"  MAE:      {metrics['MAE']:.4f} Pa\n")
        f.write(f"  NRMSE:    {metrics['NRMSE']:.4f}\n")
        f.write(f"  R²:       {metrics['R2']:.4f}\n")
        f.write(f"  Pearson:  {metrics['Pearson']:.4f}\n\n")
        
        f.write("-" * 40 + "\n")
        f.write("TRAINING SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Final Loss:  {history['total'][-1]:.6f}\n")
        f.write(f"  Best Loss:   {min(history['total']):.6f}\n")
        f.write(f"  Epochs:      {len(history['total'])}\n")
    
    print(f"  Results saved to: {results_file}")
    
    return model, results


def evaluate_patient(model: nn.Module, test_loader: DataLoader, dataset: PatientDataset
                    ) -> Tuple[Dict, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate model on patient data."""
    model.eval()
    
    all_pred, all_true, all_coords = [], [], []
    
    with torch.no_grad():
        for batch in test_loader:
            coords = batch['coords'].to(DEVICE)
            coords_raw = batch['coords_raw'].cpu().numpy()
            wss_raw = batch['wss_raw'].cpu().numpy().flatten()
            has_wss = batch['has_wss'].cpu().numpy().squeeze().astype(bool)
            
            outputs = model(coords)
            wss_pred_scaled = outputs['wss'].cpu().numpy()
            wss_pred = dataset.scaler_y.inverse_transform(wss_pred_scaled).flatten()
            
            if has_wss.any():
                all_pred.append(wss_pred[has_wss])
                all_true.append(wss_raw[has_wss])
                all_coords.append(coords_raw[has_wss])
    
    y_pred = np.concatenate(all_pred)
    y_true = np.concatenate(all_true)
    coords_all = np.concatenate(all_coords)
    
    # Compute metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    corr, _ = pearsonr(y_true, y_pred)
    nrmse = rmse / (y_true.max() - y_true.min() + 1e-10)
    
    metrics = {
        'MSE': rmse**2, 'RMSE': rmse, 'NRMSE': nrmse, 'MAE': mae, 
        'R2': r2, 'Pearson': corr
    }
    
    print(f"  RMSE: {rmse:.4f} Pa | NRMSE: {nrmse:.4f} | MAE: {mae:.4f} Pa | R²: {r2:.4f}")
    
    return metrics, y_true, y_pred, coords_all


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Per-Patient PINN Training')
    parser.add_argument('--patient', type=str, default='H-12',
                        help=f"Patient ID or 'all'. Available: {list(PATIENT_DATA.keys())}")
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4096, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--arch', type=str, default='multi', 
                        choices=['multi', 'resnet', 'single'],
                        help='Architecture: multi (MultiPINN), resnet (MultiResNetPINN), single (legacy)')
    parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden layer dimension')
    parser.add_argument('--num-blocks', type=int, default=4, help='Number of ResNet blocks (for resnet arch)')
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("PER-PATIENT PINN TRAINING")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Architecture: {args.arch.upper()}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Select patients
    if args.patient.lower() == 'all':
        patients = list(PATIENT_DATA.keys())
    else:
        patients = [args.patient]
    
    # Train each patient
    all_results = {}
    for patient_id in patients:
        if patient_id not in PATIENT_DATA:
            print(f"\nWarning: Unknown patient {patient_id}. Skipping.")
            continue
        model, results = train_patient(
            patient_id, args.epochs, args.batch_size, args.lr,
            arch=args.arch, hidden_dim=args.hidden_dim, num_blocks=args.num_blocks
        )
        all_results[patient_id] = results['metrics']
    
    # Summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE - SUMMARY")
    print("=" * 80)
    for pid, m in all_results.items():
        print(f"{pid:8s} | RMSE: {m['RMSE']:.4f} Pa | NRMSE: {m['NRMSE']:.4f} | R²: {m['R2']:.4f}")
    
    # Save summary as readable text file
    summary_path = OUTPUT_PATH / "training_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("PINN TRAINING SUMMARY - ALL PATIENTS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{'Patient':<10} {'RMSE (Pa)':<12} {'NRMSE':<10} {'MAE (Pa)':<12} {'R²':<10} {'Pearson':<10}\n")
        f.write("-" * 70 + "\n")
        for pid, m in all_results.items():
            f.write(f"{pid:<10} {m['RMSE']:<12.4f} {m['NRMSE']:<10.4f} {m['MAE']:<12.4f} {m['R2']:<10.4f} {m['Pearson']:<10.4f}\n")
    print(f"\nSummary saved to: {summary_path}")


if __name__ == '__main__':
    main()
