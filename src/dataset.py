"""
Dataset Module for PINN-based WSS Prediction

This module handles all data loading, preprocessing, and PyTorch Dataset
creation for training physics-informed neural networks on coronary artery
CFD simulation data.

Key Components:
    - parse_cfd_csv: Parse ANSYS CFD-Post export format
    - load_vessel_data: Load wall and streamline data for a vessel
    - load_patient_data: Load all vessel data for a patient
    - CollocationSampler: Mesh-based collocation point sampling
    - PatientDataset: PyTorch Dataset with proper scaling

Data Format:
    CFD files contain columns: X [m], Y [m], Z [m], Velocity u/v/w [m/s],
    Wall Shear [Pa], Wall Shear X/Y/Z [Pa]
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, Tuple, Optional, List
from pathlib import Path

from src.config import DATA_PATH, PATIENT_DATA


def parse_cfd_csv(filepath: Path) -> Optional[pd.DataFrame]:
    """
    Parse CFD simulation CSV file with ANSYS CFD-Post [Name]/[Data] format.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame with parsed data, or None if parsing fails
    """
    if not filepath.exists():
        return None
    df = pd.read_csv(filepath, skiprows=5, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    required_cols = ['X [ m ]', 'Y [ m ]', 'Z [ m ]']
    if any(c not in df.columns for c in required_cols):
        return None
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()
    return df if len(df) > 0 else None


def load_vessel_data(wall_file: str, stream_file: str = None) -> Optional[Dict[str, np.ndarray]]:
    """
    Load CFD data for a single vessel (wall surface + optional streamlines).
    
    Args:
        wall_file: Filename of wall surface data (contains WSS)
        stream_file: Filename of streamline data (interior velocities)
        
    Returns:
        Dictionary containing:
            - X: Coordinates (N, 3)
            - y: WSS values (N,) - NaN for interior points
            - velocity: Velocity vectors (N, 3)
            - normals: Wall normal vectors (N, 3)
            - has_wss: Boolean mask for wall points
    """
    all_X, all_y, all_vel, all_normals, all_has_wss = [], [], [], [], []
    
    wall_path = DATA_PATH / wall_file
    df = parse_cfd_csv(wall_path)
    if df is None:
        return None
    
    X = df[['X [ m ]', 'Y [ m ]', 'Z [ m ]']].values
    y = df['Wall Shear [ Pa ]'].values
    vel = df[['Velocity u [ m s^-1 ]', 'Velocity v [ m s^-1 ]', 
             'Velocity w [ m s^-1 ]']].values
    
    # Wall normals from WSS vector direction
    wss_vec = df[['Wall Shear X [ Pa ]', 'Wall Shear Y [ Pa ]', 
                 'Wall Shear Z [ Pa ]']].values
    norms = np.linalg.norm(wss_vec, axis=1, keepdims=True) + 1e-10
    normals = wss_vec / norms
    
    all_X.append(X)
    all_y.append(y)
    all_vel.append(vel)
    all_normals.append(normals)
    all_has_wss.append(np.ones(len(X), dtype=bool))
    
    # Streamline data
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
    
    return {
        'X': np.vstack(all_X).astype(np.float32),
        'y': np.concatenate(all_y).astype(np.float32),
        'velocity': np.vstack(all_vel).astype(np.float32),
        'normals': np.vstack(all_normals).astype(np.float32),
        'has_wss': np.concatenate(all_has_wss)
    }


def load_patient_data(patient_id: str) -> Tuple[Dict[str, np.ndarray], Dict]:
    """
    Load all vessel data for a patient and combine into unified dataset.
    
    Args:
        patient_id: Patient identifier (e.g., 'H-12', '0156')
        
    Returns:
        Tuple of:
            - Combined data dictionary with all vessels merged
            - Per-vessel data dictionary for individual vessel analysis
    """
    config = PATIENT_DATA[patient_id]
    all_X, all_y, all_vel, all_normals, all_has_wss = [], [], [], [], []
    per_vessel = {}
    
    for vessel_name, files in config['vessels'].items():
        vessel_data = load_vessel_data(files['wall'], files.get('stream'))
        if vessel_data is not None:
            per_vessel[vessel_name] = vessel_data
            all_X.append(vessel_data['X'])
            all_y.append(vessel_data['y'])
            all_vel.append(vessel_data['velocity'])
            all_normals.append(vessel_data['normals'])
            all_has_wss.append(vessel_data['has_wss'])
            
            n_wall = vessel_data['has_wss'].sum()
            n_stream = len(vessel_data['X']) - n_wall
            print(f"  {vessel_name}: {n_wall:,} wall + {n_stream:,} streamline")
    
    if len(all_X) == 0:
        raise ValueError(f"No data loaded for patient {patient_id}")
    
    combined = {
        'X': np.vstack(all_X).astype(np.float32),
        'y': np.concatenate(all_y).astype(np.float32),
        'velocity': np.vstack(all_vel).astype(np.float32),
        'normals': np.vstack(all_normals).astype(np.float32),
        'has_wss': np.concatenate(all_has_wss)
    }
    
    print(f"  Total: {len(combined['X']):,} points")
    valid_wss = combined['y'][~np.isnan(combined['y'])]
    if len(valid_wss) > 0:
        print(f"  WSS range: [{valid_wss.min():.2f}, {valid_wss.max():.2f}] Pa")
    
    return combined, per_vessel


class CollocationSampler:
    """
    Mesh-based collocation point sampler for physics-informed training.
    
    CRITICAL: Coronary arteries are thin tubes (~3-5mm diameter) inside a 
    bounding box that could be 100mm+. Random uniform sampling in the bounding 
    box would create points OUTSIDE the vessel where Navier-Stokes doesn't apply.
    
    This sampler ensures collocation points are sampled FROM the actual mesh,
    with optional weighting to prefer interior points (where velocities are 
    non-zero and physics residuals are more informative).
    
    Attributes:
        coords: All mesh coordinates (N, 3)
        wall_indices: Indices of wall surface points
        interior_indices: Indices of interior (streamline) points
        weights: Sampling probability weights
    """
    
    def __init__(self, coords: np.ndarray, has_wss: np.ndarray, 
                 prefer_interior: bool = True):
        """
        Initialize the collocation sampler.
        
        Args:
            coords: All mesh coordinates (N, 3)
            has_wss: Boolean mask - True for wall points, False for interior
            prefer_interior: If True, weight interior points higher (recommended)
        """
        self.coords = coords
        self.has_wss = has_wss
        self.prefer_interior = prefer_interior
        
        # Separate wall and interior indices
        self.wall_indices = np.where(has_wss)[0]
        self.interior_indices = np.where(~has_wss)[0]
        
        # Compute sampling weights
        if prefer_interior and len(self.interior_indices) > 0:
            # Interior points are better for NS physics (non-zero velocity)
            # Wall points have u=0, so derivatives are less informative
            weights = np.ones(len(coords))
            weights[self.wall_indices] = 0.3  # Lower weight for wall
            weights[self.interior_indices] = 1.0  # Full weight for interior
        else:
            weights = np.ones(len(coords))
        
        self.weights = weights / weights.sum()
    
    def sample(self, n_points: int) -> np.ndarray:
        """
        Sample n collocation points from the mesh.
        
        Args:
            n_points: Number of points to sample
            
        Returns:
            Sampled coordinates (n_points, 3)
        """
        n_available = len(self.coords)
        n_points = min(n_points, n_available)
        
        indices = np.random.choice(
            n_available, 
            size=n_points, 
            replace=False,
            p=self.weights
        )
        return self.coords[indices]
    
    def sample_stratified(self, n_points: int, n_bins: int = 5) -> np.ndarray:
        """
        Stratified sampling: divide domain into voxels and sample from each.
        Ensures good coverage of the entire vessel.
        """
        coords = self.coords
        
        x_bins = np.linspace(coords[:, 0].min(), coords[:, 0].max(), n_bins + 1)
        y_bins = np.linspace(coords[:, 1].min(), coords[:, 1].max(), n_bins + 1)
        z_bins = np.linspace(coords[:, 2].min(), coords[:, 2].max(), n_bins + 1)
        
        sampled_indices = []
        points_per_voxel = max(1, n_points // (n_bins ** 3))
        
        for i in range(n_bins):
            for j in range(n_bins):
                for k in range(n_bins):
                    mask = (
                        (coords[:, 0] >= x_bins[i]) & (coords[:, 0] < x_bins[i+1]) &
                        (coords[:, 1] >= y_bins[j]) & (coords[:, 1] < y_bins[j+1]) &
                        (coords[:, 2] >= z_bins[k]) & (coords[:, 2] < z_bins[k+1])
                    )
                    voxel_indices = np.where(mask)[0]
                    if len(voxel_indices) > 0:
                        n_sample = min(points_per_voxel, len(voxel_indices))
                        sampled_indices.extend(
                            np.random.choice(voxel_indices, n_sample, replace=False)
                        )
        
        # Pad with random samples if needed
        if len(sampled_indices) < n_points:
            remaining = n_points - len(sampled_indices)
            available = list(set(range(len(coords))) - set(sampled_indices))
            if available:
                sampled_indices.extend(
                    np.random.choice(available, min(remaining, len(available)), replace=False)
                )
        
        return coords[sampled_indices[:n_points]]


class PatientDataset(Dataset):
    """
    PyTorch Dataset for patient-specific PINN training.
    
    Handles coordinate and target scaling using MinMaxScaler to normalize
    inputs to [0, 1] and velocities to [-1, 1] for stable training.
    
    Attributes:
        scaler_X: MinMaxScaler for coordinates
        scaler_y: MinMaxScaler for WSS values
        scaler_vel: MinMaxScaler for velocity components
    """
    
    def __init__(self, data: Dict[str, np.ndarray]):
        """
        Initialize dataset with scaling.
        
        Args:
            data: Dictionary from load_patient_data containing:
                - X: Coordinates (N, 3)
                - y: WSS values (N,)
                - velocity: Velocity vectors (N, 3)
                - normals: Wall normal vectors (N, 3)
                - has_wss: Boolean mask for wall points
        """
        self.X = data['X']
        self.y = data['y']
        self.velocity = data['velocity']
        self.normals = data['normals']
        self.has_wss = data['has_wss']
        
        # Fit scalers
        self.scaler_X = MinMaxScaler(feature_range=(0, 1))
        self.X_scaled = self.scaler_X.fit_transform(self.X)
        
        # WSS scaler (fit only on valid values)
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
