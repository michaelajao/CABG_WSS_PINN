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

# Try to import Open3D for accurate normal estimation
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("Warning: Open3D not installed. Using WSS-derived normals (less accurate).")
    print("Install with: pip install open3d")


def estimate_normals_open3d(points: np.ndarray, k_neighbors: int = 30,
                             orient_toward_center: bool = True) -> np.ndarray:
    """
    Estimate surface normals from point cloud using Open3D.

    Uses k-nearest neighbors to estimate local surface orientation.
    Normals are oriented consistently (either toward or away from vessel center).

    Args:
        points: (N, 3) array of point coordinates
        k_neighbors: Number of neighbors for normal estimation (default: 30)
        orient_toward_center: If True, orient normals toward vessel center (inward)
                              If False, orient normals away from center (outward)

    Returns:
        normals: (N, 3) array of unit normal vectors
    """
    if not OPEN3D_AVAILABLE:
        raise ImportError("Open3D is required for normal estimation")

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

    # Estimate normals using k-nearest neighbors
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k_neighbors)
    )

    # Orient normals consistently using the vessel centerline as reference
    # Approximate center as the mean of all points
    center = np.mean(points, axis=0)

    # Orient normals toward or away from center
    pcd.orient_normals_towards_camera_location(center)

    # Get normals as numpy array
    normals = np.asarray(pcd.normals).astype(np.float32)

    # If orient_toward_center=False, flip normals to point outward
    if not orient_toward_center:
        normals = -normals

    return normals


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

    # Estimate wall normals
    if OPEN3D_AVAILABLE:
        # Use Open3D for accurate geometric normal estimation
        # orient_toward_center=True gives inward-pointing normals (into vessel)
        normals = estimate_normals_open3d(X, k_neighbors=30,
                                           orient_toward_center=True)
    else:
        # Fallback: Use WSS vector direction (less accurate)
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


def parse_multi_section_csv(filepath: Path, section_name: str) -> Optional[pd.DataFrame]:
    """
    Parse a specific section from an ANSYS CFD-Post CSV with multiple [Name]/[Data] blocks.

    Args:
        filepath: Path to the CSV file
        section_name: Name of the section to extract (e.g., 'aorta', 'aorta_wall')

    Returns:
        DataFrame with the section's data, or None if not found
    """
    if not filepath.exists():
        return None

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Find the target section
    section_start = None
    data_start = None
    data_end = None

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line == '[Name]' and i + 1 < len(lines):
            name = lines[i + 1].strip().lower()
            # Match aorta sections (could be 'aorta', 'aorta_wall', etc.)
            if section_name.lower() in name or name in section_name.lower():
                section_start = i
                # Find [Data] marker
                j = i + 2
                while j < len(lines) and lines[j].strip() != '[Data]':
                    j += 1
                if j < len(lines):
                    header_line = j + 1
                    data_start = j + 2
                    # Find end (next [Name] or EOF)
                    k = data_start
                    while k < len(lines) and lines[k].strip() != '[Name]':
                        k += 1
                    data_end = k
                    break
        i += 1

    if data_start is None or data_end is None:
        return None

    # Parse the header and data
    header = lines[header_line].strip()
    columns = [c.strip() for c in header.split(',')]

    data_rows = []
    for line in lines[data_start:data_end]:
        stripped = line.strip()
        if not stripped or ',' not in stripped:
            continue

        parts = [p.strip() for p in stripped.split(',')]
        if len(parts) != len(columns):
            continue

        # Skip rows with empty values
        if not all(parts):
            continue

        # Convert to floats - skip row if any value is non-numeric
        # Note: try-except is necessary here as float() is the only reliable
        # way to validate numeric strings including scientific notation
        row = []
        skip_row = False
        for p in parts:
            try:
                row.append(float(p))
            except ValueError:
                skip_row = True
                break
        if not skip_row:
            data_rows.append(row)

    if not data_rows:
        return None

    df = pd.DataFrame(data_rows, columns=columns)
    return df


def load_aorta_data(patient_id: str) -> Optional[np.ndarray]:
    """
    Load aorta coordinates from the main patient file for visualization.

    The main patient files (e.g., H-12.csv) contain multiple anatomical regions.
    This function extracts only the aorta section for grey background display
    in full patient anatomy views.

    Args:
        patient_id: Patient identifier (e.g., 'H-12', '0156')

    Returns:
        Aorta coordinates (N, 3) in meters, or None if not available
    """
    config = PATIENT_DATA.get(patient_id, {})
    aorta_file = config.get('aorta_file')

    if not aorta_file:
        return None

    aorta_path = DATA_PATH / aorta_file

    # Try to find aorta section (could be named 'aorta', 'aorta_wall', etc.)
    df = parse_multi_section_csv(aorta_path, 'aorta')

    if df is None:
        return None

    # Extract coordinates - verify columns exist first
    required_cols = ['X [ m ]', 'Y [ m ]', 'Z [ m ]']
    if not all(col in df.columns for col in required_cols):
        return None

    coords = df[required_cols].values.astype(np.float32)
    return coords if len(coords) > 0 else None


def load_full_anatomy(patient_id: str) -> Optional[np.ndarray]:
    """
    Load complete anatomy coordinates from the main patient file for visualization.

    This function extracts ALL wall/surface sections from the patient's main CSV
    file to create a complete grey background in full patient plots. Unlike
    load_aorta_data() which only loads the aorta section, this loads all sections
    that represent the vessel walls.

    Args:
        patient_id: Patient identifier (e.g., 'H-12', '0156')

    Returns:
        All anatomy coordinates (N, 3) in meters, or None if not available
    """
    config = PATIENT_DATA.get(patient_id, {})
    aorta_file = config.get('aorta_file')

    if not aorta_file:
        return None

    filepath = DATA_PATH / aorta_file
    if not filepath.exists():
        return None

    # Read entire file to find all sections
    with open(filepath, 'r') as f:
        lines = f.readlines()

    all_coords = []

    # Parse all sections
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line == '[Name]' and i + 1 < len(lines):
            section_name = lines[i + 1].strip().lower()

            # Skip non-wall sections (inlets, outlets, streamlines)
            skip_keywords = ['inlet', 'outlet', 'stream', 'line']
            if any(kw in section_name for kw in skip_keywords):
                i += 1
                continue

            # Find [Data] marker
            j = i + 2
            while j < len(lines) and lines[j].strip() != '[Data]':
                j += 1

            if j < len(lines):
                header_line = j + 1
                data_start = j + 2

                # Find end of this section (next [Name] or EOF)
                k = data_start
                while k < len(lines) and lines[k].strip() != '[Name]':
                    k += 1
                data_end = k

                # Parse header to find X, Y, Z column indices
                if header_line < len(lines):
                    header = lines[header_line].strip()
                    columns = [c.strip() for c in header.split(',')]

                    try:
                        x_idx = columns.index('X [ m ]')
                        y_idx = columns.index('Y [ m ]')
                        z_idx = columns.index('Z [ m ]')
                    except ValueError:
                        i = k
                        continue

                    # Parse data rows
                    for line_num in range(data_start, data_end):
                        row_line = lines[line_num].strip()
                        if not row_line or ',' not in row_line:
                            continue

                        parts = [p.strip() for p in row_line.split(',')]
                        if len(parts) <= max(x_idx, y_idx, z_idx):
                            continue

                        # Skip rows with any non-numeric coordinate values
                        try:
                            x = float(parts[x_idx])
                            y = float(parts[y_idx])
                            z = float(parts[z_idx])
                            all_coords.append([x, y, z])
                        except (ValueError, IndexError):
                            continue

                i = k
                continue

        i += 1

    if not all_coords:
        return None

    return np.array(all_coords, dtype=np.float32)


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


class GPUDataCache:
    """
    GPU-resident data cache for maximum training throughput.

    Pre-loads ALL training data to GPU memory once, eliminating
    per-batch CPU→GPU transfers. For datasets that fit in GPU memory,
    this provides significant speedup (3-5x).

    Also provides efficient batching without DataLoader overhead.

    Attributes:
        coords: Scaled coordinates on GPU (N, 3)
        wss: WSS values on GPU (N, 1)
        velocity: Velocity values on GPU (N, 3)
        normals: Normal vectors on GPU (N, 3)
        has_wss: Boolean mask on GPU (N,)
        n_samples: Total number of samples
    """

    def __init__(self, dataset: 'PatientDataset', device: str = 'cuda'):
        """
        Initialize GPU cache from PatientDataset.

        Args:
            dataset: PatientDataset instance with scaled data
            device: PyTorch device
        """
        import torch

        self.device = device
        self.n_samples = len(dataset)

        # Pre-load ALL data to GPU (single transfer)
        print(f"  Pre-loading {self.n_samples:,} samples to GPU...")

        self.coords = torch.from_numpy(
            dataset.X_scaled.astype(np.float32)
        ).to(device)

        self.wss = torch.from_numpy(
            dataset.y_scaled.reshape(-1, 1).astype(np.float32)
        ).to(device)

        self.velocity = torch.from_numpy(
            dataset.vel_scaled.astype(np.float32)
        ).to(device)

        self.normals = torch.from_numpy(
            dataset.normals.astype(np.float32)
        ).to(device)

        self.has_wss = torch.from_numpy(
            dataset.has_wss
        ).to(device)

        # Store reference scales
        self.ref_scales = dataset.get_reference_scales()

        # Shuffled indices for epoch iteration
        self._indices = None
        self._current_epoch = -1

        # Memory usage estimate
        mem_bytes = (
            self.coords.numel() + self.wss.numel() +
            self.velocity.numel() + self.normals.numel()
        ) * 4  # float32 = 4 bytes
        print(f"  GPU memory used: {mem_bytes / 1e6:.1f} MB")

    def shuffle_for_epoch(self, epoch: int) -> None:
        """Shuffle indices for a new epoch."""
        import torch

        if epoch != self._current_epoch:
            self._current_epoch = epoch
            self._indices = torch.randperm(self.n_samples, device=self.device)

    def get_batch(self, batch_idx: int, batch_size: int) -> dict:
        """
        Get a batch of data (already on GPU - zero transfer overhead).

        Args:
            batch_idx: Batch index within epoch
            batch_size: Number of samples per batch

        Returns:
            Dictionary with coords, wss, velocity, normals, has_wss tensors
        """
        if self._indices is None:
            self.shuffle_for_epoch(0)

        start = batch_idx * batch_size
        end = min(start + batch_size, self.n_samples)

        # Handle epoch boundary
        if start >= self.n_samples:
            start = start % self.n_samples
            end = min(start + batch_size, self.n_samples)

        idx = self._indices[start:end]

        return {
            'coords': self.coords[idx],
            'wss': self.wss[idx],
            'velocity': self.velocity[idx],
            'normals': self.normals[idx],
            'has_wss': self.has_wss[idx]
        }

    def __len__(self) -> int:
        return self.n_samples

    @property
    def num_batches(self) -> int:
        """Number of batches per epoch (for progress bars)."""
        return self.n_samples  # Will be divided by batch_size in caller

    def get_reference_scales(self) -> dict:
        """Return reference scales for physics computations."""
        return self.ref_scales


class GPUCollocationSampler:
    """
    GPU-native collocation point sampler - eliminates CPU-GPU transfers.

    This sampler keeps ALL data on GPU and performs sampling operations
    using PyTorch tensors, avoiding the expensive CPU→GPU transfers that
    occur when using numpy/sklearn operations each batch.

    Performance improvement: 5-10x faster than CPU-based sampling.

    Attributes:
        coords_scaled: Pre-scaled coordinates on GPU (N, 3)
        weights: Sampling probability weights on GPU (N,)
        device: PyTorch device
    """

    def __init__(
        self,
        coords: np.ndarray,
        has_wss: np.ndarray,
        scaler_X,
        device: str = 'cuda',
        prefer_interior: bool = True
    ):
        """
        Initialize GPU-native collocation sampler.

        Args:
            coords: Raw coordinates (N, 3) in meters
            has_wss: Boolean mask - True for wall points
            scaler_X: Fitted sklearn MinMaxScaler for coordinate scaling
            device: PyTorch device ('cuda' or 'cpu')
            prefer_interior: Weight interior points higher
        """
        import torch

        self.device = device
        self.n_points = len(coords)

        # Pre-scale coordinates on CPU, then move to GPU ONCE
        coords_scaled = scaler_X.transform(coords).astype(np.float32)
        self.coords_scaled = torch.from_numpy(coords_scaled).to(device)

        # Store indices
        self.wall_indices = torch.from_numpy(np.where(has_wss)[0]).to(device)
        self.interior_indices = torch.from_numpy(np.where(~has_wss)[0]).to(device)

        # Compute sampling weights on GPU
        weights = torch.ones(self.n_points, device=device)
        if prefer_interior and len(self.interior_indices) > 0:
            weights[self.wall_indices] = 0.3
            weights[self.interior_indices] = 1.0
        self.weights = weights / weights.sum()

        # Pre-allocated buffer for sampling (avoid repeated allocations)
        self._sample_buffer = None

        # Epoch tracking for resampling
        self._current_epoch = -1
        self._epoch_indices = None

    def resample_for_epoch(self, epoch: int, n_points: int) -> None:
        """Pre-sample indices for an entire epoch (GPU-native)."""
        import torch

        if epoch != self._current_epoch:
            self._current_epoch = epoch

            # GPU-native weighted sampling using torch.multinomial
            # Note: multinomial samples WITH replacement, so we oversample
            # and take unique indices for better coverage
            if n_points <= self.n_points:
                # Sample more than needed, then take unique
                oversample = min(n_points * 2, self.n_points)
                sampled = torch.multinomial(
                    self.weights, oversample, replacement=False
                )
                self._epoch_indices = sampled[:n_points]
            else:
                # Need more points than available - sample with replacement
                self._epoch_indices = torch.multinomial(
                    self.weights, n_points, replacement=True
                )

            # Shuffle for batch randomization
            perm = torch.randperm(len(self._epoch_indices), device=self.device)
            self._epoch_indices = self._epoch_indices[perm]

    def sample(self, n_points: int) -> 'torch.Tensor':
        """Sample n collocation points (returns GPU tensor directly)."""
        import torch

        n_points = min(n_points, self.n_points)
        indices = torch.multinomial(self.weights, n_points, replacement=False)
        return self.coords_scaled[indices]

    def sample_batch(self, batch_idx: int, batch_size: int) -> 'torch.Tensor':
        """
        Get a batch of pre-sampled collocation points (GPU tensor).

        Call resample_for_epoch() at the start of each epoch.
        Returns coordinates already on GPU - NO CPU-GPU transfer!
        """
        if self._epoch_indices is None:
            raise RuntimeError("Call resample_for_epoch() before sample_batch()")

        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(self._epoch_indices))

        # Wrap around if needed
        if start_idx >= len(self._epoch_indices):
            start_idx = start_idx % len(self._epoch_indices)
            end_idx = min(start_idx + batch_size, len(self._epoch_indices))

        batch_indices = self._epoch_indices[start_idx:end_idx]
        return self.coords_scaled[batch_indices]


class CollocationSampler:
    """
    Mesh-based collocation point sampler for physics-informed training.

    CRITICAL: Coronary arteries are thin tubes (~3-5mm diameter) inside a
    bounding box that could be 100mm+. Random uniform sampling in the bounding
    box would create points OUTSIDE the vessel where Navier-Stokes doesn't apply.

    This sampler ensures collocation points are sampled FROM the actual mesh,
    with optional weighting to prefer interior points (where velocities are
    non-zero and physics residuals are more informative).

    Features:
        - Mesh-based sampling (always inside vessel geometry)
        - Weighted sampling preferring interior points
        - Per-epoch resampling with jitter for better coverage
        - Optional noise injection for regularization

    Attributes:
        coords: All mesh coordinates (N, 3)
        wall_indices: Indices of wall surface points
        interior_indices: Indices of interior (streamline) points
        weights: Sampling probability weights
        jitter_scale: Scale for coordinate jitter (fraction of min coord range)
    """

    def __init__(self, coords: np.ndarray, has_wss: np.ndarray,
                 prefer_interior: bool = True, jitter_scale: float = 0.0):
        """
        Initialize the collocation sampler.

        Args:
            coords: All mesh coordinates (N, 3)
            has_wss: Boolean mask - True for wall points, False for interior
            prefer_interior: If True, weight interior points higher (recommended)
            jitter_scale: Scale for coordinate jitter as fraction of min range.
                Set to 0 to disable jitter. Default 0.01 (1% of smallest dimension).
        """
        self.coords = coords
        self.has_wss = has_wss
        self.prefer_interior = prefer_interior
        self.jitter_scale = jitter_scale

        # Compute coordinate ranges for jitter scaling
        coord_range = coords.max(axis=0) - coords.min(axis=0)
        self.jitter_magnitude = jitter_scale * coord_range.min()

        # Separate wall and interior indices
        self.wall_indices = np.where(has_wss)[0]
        self.interior_indices = np.where(~has_wss)[0]

        # Compute sampling weights
        self._compute_weights()

        # Track epoch for resampling
        self._current_epoch = -1
        self._epoch_indices = None

    def _compute_weights(self) -> None:
        """Compute sampling probability weights."""
        if self.prefer_interior and len(self.interior_indices) > 0:
            # Interior points are better for NS physics (non-zero velocity)
            # Wall points have u=0, so derivatives are less informative
            weights = np.ones(len(self.coords))
            weights[self.wall_indices] = 0.3  # Lower weight for wall
            weights[self.interior_indices] = 1.0  # Full weight for interior
        else:
            weights = np.ones(len(self.coords))

        self.weights = weights / weights.sum()

    def resample_for_epoch(self, epoch: int, n_points: int) -> None:
        """
        Pre-sample collocation points for an entire epoch.

        This ensures consistent sampling within an epoch while varying
        across epochs for better domain coverage over training.

        Args:
            epoch: Current epoch number (used to detect epoch changes)
            n_points: Total number of collocation points for the epoch
        """
        if epoch != self._current_epoch:
            self._current_epoch = epoch
            n_available = len(self.coords)

            # For large n_points, we may need to sample with replacement
            # but shuffle the order for better batch diversity
            if n_points <= n_available:
                self._epoch_indices = np.random.choice(
                    n_available,
                    size=n_points,
                    replace=False,
                    p=self.weights
                )
            else:
                # Sample with replacement but ensure good coverage
                self._epoch_indices = np.random.choice(
                    n_available,
                    size=n_points,
                    replace=True,
                    p=self.weights
                )

            # Shuffle to randomize batch order
            np.random.shuffle(self._epoch_indices)

    def sample(self, n_points: int, add_jitter: bool = False) -> np.ndarray:
        """
        Sample n collocation points from the mesh.

        Args:
            n_points: Number of points to sample
            add_jitter: If True, add small random noise to coordinates.
                This provides implicit regularization and better coverage.

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

        sampled_coords = self.coords[indices].copy()

        # Add small jitter for regularization and better coverage
        if add_jitter and self.jitter_magnitude > 0:
            noise = np.random.randn(n_points, 3) * self.jitter_magnitude
            sampled_coords = sampled_coords + noise

        return sampled_coords

    def sample_batch(self, batch_idx: int, batch_size: int,
                     add_jitter: bool = True) -> np.ndarray:
        """
        Sample a batch of collocation points from pre-sampled epoch indices.

        Call resample_for_epoch() at the start of each epoch before using this.

        Args:
            batch_idx: Current batch index within the epoch
            batch_size: Number of points per batch
            add_jitter: If True, add small random noise to coordinates

        Returns:
            Sampled coordinates (batch_size, 3)
        """
        if self._epoch_indices is None:
            raise RuntimeError(
                "Call resample_for_epoch() before sample_batch()"
            )

        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(self._epoch_indices))

        # Wrap around if needed
        if start_idx >= len(self._epoch_indices):
            start_idx = start_idx % len(self._epoch_indices)
            end_idx = min(start_idx + batch_size, len(self._epoch_indices))

        batch_indices = self._epoch_indices[start_idx:end_idx]
        sampled_coords = self.coords[batch_indices].copy()

        # Add jitter for better coverage (only for interior points)
        if add_jitter and self.jitter_magnitude > 0:
            # Only jitter interior points, not wall points
            is_interior = ~self.has_wss[batch_indices]
            if is_interior.any():
                noise = np.random.randn(is_interior.sum(), 3) * self.jitter_magnitude
                sampled_coords[is_interior] += noise

        return sampled_coords


class PatientDataset(Dataset):
    """
    PyTorch Dataset for patient-specific PINN training.
    
    Implements PROPER NON-DIMENSIONALIZATION for physics-informed learning:
    - Coordinates: scaled to [0, 1] using MinMax (preserves relative geometry)
    - Velocities: scaled by a SINGLE reference velocity U_ref (preserves physics)
    - WSS: scaled by max WSS value for O(1) outputs (practical for training)
    - Pressure: model learns p* = p/(rho*U_ref^2)
    
    This ensures the non-dimensional Navier-Stokes equations are properly satisfied
    while keeping all network outputs in reasonable O(1) ranges.
    
    Attributes:
        scaler_X: MinMaxScaler for coordinates
        ref_scales: Dictionary of reference scales (U_ref, L_ref, T_ref, Re)
        U_ref: Reference velocity scale [m/s]
        L_ref: Reference length scale [m]
        T_ref: Reference WSS scale [Pa] (data-driven, for O(1) outputs)
    """
    
    def __init__(self, data: Dict[str, np.ndarray]):
        """
        Initialize dataset with proper non-dimensional scaling.
        
        Args:
            data: Dictionary from load_patient_data containing:
                - X: Coordinates (N, 3) in meters
                - y: WSS values (N,) in Pa
                - velocity: Velocity vectors (N, 3) in m/s
                - normals: Wall normal vectors (N, 3)
                - has_wss: Boolean mask for wall points
        """
        from src.config import RHO, MU
        
        self.X = data['X']
        self.y = data['y']
        self.velocity = data['velocity']
        self.normals = data['normals']
        self.has_wss = data['has_wss']
        
        # =====================================================================
        # COORDINATE SCALING: MinMax to [0, 1]
        # =====================================================================
        self.scaler_X = MinMaxScaler(feature_range=(0, 1))
        self.X_scaled = self.scaler_X.fit_transform(self.X)
        
        # Store coordinate ranges for gradient chain rule (in meters)
        self.coord_range = self.scaler_X.data_range_

        # Reference length: use minimum of coordinate ranges (vessel diameter)
        # For thin tubes (coronary arteries), the diameter is the characteristic length
        # Using mean would give meaningless values for anisotropic geometries
        self.L_ref = float(np.min(self.coord_range))
        
        # =====================================================================
        # VELOCITY SCALING: Single U_ref for all components (physics-preserving)
        # =====================================================================
        # Use 95th percentile of velocity magnitude for robustness
        vel_magnitude = np.linalg.norm(self.velocity, axis=1)
        self.U_ref = float(np.percentile(vel_magnitude[vel_magnitude > 0], 95))
        self.U_ref = max(self.U_ref, 1e-6)  # Avoid division by zero
        
        # Non-dimensional velocity: u* = u / U_ref
        self.vel_scaled = self.velocity / self.U_ref
        
        # =====================================================================
        # DERIVED QUANTITIES
        # =====================================================================
        # Pressure reference: P_ref = rho * U_ref^2
        self.P_ref = RHO * self.U_ref**2

        # Reynolds number for non-dimensional N-S
        self.Re = RHO * self.U_ref * self.L_ref / MU

        # =====================================================================
        # WSS SCALING: Physics-based for consistent non-dimensionalization
        # =====================================================================
        # Use physics-based scale: T_ref = mu * U_ref / L_ref
        self.T_ref_physics = MU * self.U_ref / self.L_ref
        
        # Decouple data scaling from physics scaling to avoid huge loss values
        # Use a fixed reference scale of 10.0 Pa (typical WSS magnitude)
        # This keeps network outputs in O(1) range (e.g. 0-5 instead of 0-2500)
        self.T_ref = 10.0

        valid_mask = ~np.isnan(self.y)
        if valid_mask.any():
            valid_wss = self.y[valid_mask]
            self.wss_range = (valid_wss.min(), valid_wss.max())
        else:
            self.wss_range = (0.0, 1.0)

        # Non-dimensional WSS: tau* = tau / T_ref
        self.y_scaled = np.zeros_like(self.y)
        if valid_mask.any():
            self.y_scaled[valid_mask] = self.y[valid_mask] / self.T_ref
        
        # Store all reference scales
        self.ref_scales = {
            'L_ref': self.L_ref,
            'U_ref': self.U_ref,
            'T_ref': self.T_ref,
            'T_ref_physics': self.T_ref_physics,
            'P_ref': self.P_ref,
            'Re': self.Re,
            'coord_ranges': self.coord_range.tolist()
        }
    
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
    
    def get_reference_scales(self) -> Dict[str, float]:
        """Return reference scales for physics computations."""
        return self.ref_scales
