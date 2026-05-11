"""
Dataset Module for PINN-based WSS Prediction.

Handles data loading, preprocessing, and GPU-resident tensor storage
for training physics-informed neural networks on coronary artery CFD data.

Key Components:
    - parse_cfd_csv: Parse ANSYS CFD-Post export format
    - load_vessel_data: Load wall and streamline data for a vessel
    - load_patient_data: Load all vessel data for a patient
    - CollocationSampler: Mesh-based collocation point sampling
    - PatientData: GPU-resident data with non-dimensionalization

Data Format:
    CFD files contain columns: X [m], Y [m], Z [m], Velocity u/v/w [m/s],
    Wall Shear [Pa], Wall Shear X/Y/Z [Pa]
"""

from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d
import pandas as pd
import torch

from src.config import DATA_PATH, PATIENT_DATA, patient_files


def _lines_to_numeric_df(lines: List[str], columns: List[str]) -> pd.DataFrame:
    """Parse a list of CSV data lines into a numeric DataFrame.

    Non-numeric cells are coerced to NaN by ``pd.to_numeric`` and dropped,
    matching the behaviour of :func:`parse_cfd_csv`. Returns an empty
    DataFrame if no rows survive coercion.
    """
    text = ''.join(lines).strip()
    if not text:
        return pd.DataFrame(columns=columns)
    df = pd.read_csv(StringIO(text), header=None, names=columns,
                     skipinitialspace=True)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.dropna()


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
    """Parse a single-section ANSYS CFD-Post CSV.

    Raises FileNotFoundError if the path doesn't exist (config error in
    PATIENT_DATA), and ValueError if the file lacks the expected coordinate
    columns. Returns None only when the file is well-formed but contains zero
    numeric rows after dropping NaN.
    """
    if not filepath.exists():
        raise FileNotFoundError(filepath)
    df = pd.read_csv(filepath, skiprows=5, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    required_cols = ['X [ m ]', 'Y [ m ]', 'Z [ m ]']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{filepath} is missing required columns: {missing}")
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()
    return df if len(df) > 0 else None


def load_vessel_data(wall_file: str, stream_file: str = None,
                     data_root: Path = None) -> Optional[Dict[str, np.ndarray]]:
    """
    Load CFD data for a single vessel (wall surface + optional streamlines).

    Args:
        wall_file: Filename of wall surface data (contains WSS)
        stream_file: Filename of streamline data (interior velocities)
        data_root: Directory holding the CSV files. Defaults to ``DATA_PATH``
            (the historical Newtonian root) if omitted; pass an explicit value
            when loading a non-default rheology, e.g.
            ``CARREAU_YASUDA_DATA_DIR``.

    Returns:
        Dictionary containing:
            - X: Coordinates (N, 3)
            - y: WSS values (N,) - NaN for interior points
            - velocity: Velocity vectors (N, 3)
            - normals: Wall normal vectors (N, 3)
            - has_wss: Boolean mask for wall points
    """
    root = Path(data_root) if data_root is not None else DATA_PATH
    all_X, all_y, all_vel, all_normals, all_has_wss = [], [], [], [], []

    wall_path = root / wall_file
    df = parse_cfd_csv(wall_path)
    if df is None:
        return None

    X = df[['X [ m ]', 'Y [ m ]', 'Z [ m ]']].values
    y = df['Wall Shear [ Pa ]'].values
    vel = df[['Velocity u [ m s^-1 ]', 'Velocity v [ m s^-1 ]',
             'Velocity w [ m s^-1 ]']].values

    # Estimate wall normals using Open3D
    # orient_toward_center=True gives inward-pointing normals (into vessel)
    normals = estimate_normals_open3d(X, k_neighbors=30, orient_toward_center=True)

    all_X.append(X)
    all_y.append(y)
    all_vel.append(vel)
    all_normals.append(normals)
    all_has_wss.append(np.ones(len(X), dtype=bool))

    # Streamline data
    if stream_file:
        stream_path = root / stream_file
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
    """Parse one section from an ANSYS CFD-Post CSV with multiple [Name]/[Data] blocks.

    Raises FileNotFoundError if the path doesn't exist. Returns None if the
    requested section is not present in the file or contains no numeric rows.
    """
    if not filepath.exists():
        raise FileNotFoundError(filepath)

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Locate the target section's [Name] -> [Data] -> next [Name] window.
    target = section_name.lower()
    header_line = data_start = data_end = None
    i = 0
    while i < len(lines):
        if lines[i].strip() == '[Name]' and i + 1 < len(lines):
            name = lines[i + 1].strip().lower()
            # Match e.g. 'aorta', 'aorta_wall', 'lca_wall' for section 'aorta'/'lca'.
            if target in name or name in target:
                j = i + 2
                while j < len(lines) and lines[j].strip() != '[Data]':
                    j += 1
                if j < len(lines):
                    header_line = j + 1
                    data_start = j + 2
                    k = data_start
                    while k < len(lines) and lines[k].strip() != '[Name]':
                        k += 1
                    data_end = k
                    break
        i += 1

    if data_start is None:
        return None

    columns = [c.strip() for c in lines[header_line].strip().split(',')]
    df = _lines_to_numeric_df(lines[data_start:data_end], columns)
    return df if len(df) > 0 else None


def load_aorta_data(patient_id: str) -> Optional[np.ndarray]:
    """
    Load aorta coordinates from the main patient file for visualization.

    The main patient files (e.g., H12.csv, 0156.csv) contain multiple
    anatomical regions. This function extracts only the aorta section for
    grey background display in full patient anatomy views.

    Args:
        patient_id: Public paper label (``'H4'``, ``'BG4'``, ...).

    Returns:
        Aorta coordinates (N, 3) in meters, or None if not available
    """
    if patient_id not in PATIENT_DATA:
        return None
    files_info = patient_files(patient_id)  # uses default rheology
    aorta_path = files_info['data_root'] / files_info['aorta_file']

    # Try to find aorta section (could be named 'aorta', 'aorta_wall', etc.)
    df = parse_multi_section_csv(aorta_path, 'aorta')
    if df is None:
        return None

    coords = df[['X [ m ]', 'Y [ m ]', 'Z [ m ]']].values.astype(np.float32)
    return coords if len(coords) > 0 else None


def load_full_anatomy(patient_id: str) -> Optional[np.ndarray]:
    """
    Load complete anatomy coordinates from the main patient file for visualization.

    This function extracts ALL wall/surface sections from the patient's main CSV
    file to create a complete grey background in full patient plots. Unlike
    load_aorta_data() which only loads the aorta section, this loads all sections
    that represent the vessel walls.

    Args:
        patient_id: Public paper label (``'H4'``, ``'BG4'``, ...).

    Returns:
        All anatomy coordinates (N, 3) in meters, or None if not available
    """
    if patient_id not in PATIENT_DATA:
        return None
    files_info = patient_files(patient_id)  # uses default rheology
    filepath = files_info['data_root'] / files_info['aorta_file']
    if not filepath.exists():
        raise FileNotFoundError(filepath)

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Sections we treat as flow boundaries (not vessel wall) and skip.
    skip_keywords = ('inlet', 'outlet', 'stream', 'line')
    coord_cols = ['X [ m ]', 'Y [ m ]', 'Z [ m ]']

    coord_chunks: List[np.ndarray] = []
    i = 0
    while i < len(lines):
        if lines[i].strip() != '[Name]' or i + 1 >= len(lines):
            i += 1
            continue
        section_name = lines[i + 1].strip().lower()
        if any(kw in section_name for kw in skip_keywords):
            i += 1
            continue

        # Locate [Data] marker, then header and data window for this section.
        j = i + 2
        while j < len(lines) and lines[j].strip() != '[Data]':
            j += 1
        if j >= len(lines):
            i += 1
            continue
        header_line = j + 1
        data_start = j + 2
        k = data_start
        while k < len(lines) and lines[k].strip() != '[Name]':
            k += 1
        data_end = k

        if header_line < len(lines):
            columns = [c.strip() for c in lines[header_line].strip().split(',')]
            if all(c in columns for c in coord_cols):
                df = _lines_to_numeric_df(lines[data_start:data_end], columns)
                if len(df) > 0:
                    coord_chunks.append(df[coord_cols].values.astype(np.float32))
        i = k

    if not coord_chunks:
        return None
    return np.vstack(coord_chunks)


def load_patient_data(patient_id: str,
                      rheology: str = None) -> Tuple[Dict[str, np.ndarray], Dict]:
    """
    Load all vessel data for a patient and combine into a unified dataset.

    Args:
        patient_id: Public paper label (``'H1'``, ``'H2'``, ..., ``'D3'``).
        rheology: ``'newtonian'`` or ``'carreau_yasuda'``. If omitted, the
            global ``config.RHEOLOGY`` is used. Raises ``ValueError`` if the
            requested rheology has no CFD ground truth for this patient.

    Returns:
        Tuple of (combined data dict, per-vessel data dict).
    """
    files_info = patient_files(patient_id, rheology)
    data_root = files_info['data_root']

    all_X, all_y, all_vel, all_normals, all_has_wss = [], [], [], [], []
    per_vessel = {}

    for vessel_name, files in files_info['vessels'].items():
        vessel_data = load_vessel_data(files['wall'], files.get('stream'),
                                       data_root=data_root)
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




class CollocationSamplerGPU:
    """
    GPU-resident mesh-based collocation point sampler for physics loss computation.

    Samples collocation points from the actual mesh geometry rather than
    a bounding box, ensuring physics constraints are enforced only within
    the vessel domain.

    All data is stored on GPU to avoid CPU-GPU transfers during training.

    Attributes:
        coords_scaled: Pre-scaled coordinates on GPU (N, 3)
        wall_indices: Indices of wall (boundary) points
        interior_indices: Indices of interior (streamline) points
        device: PyTorch device
    """

    def __init__(
        self,
        coords: np.ndarray,
        has_wss: np.ndarray,
        coord_offset: np.ndarray,
        L_ref: float,
        device: str = 'cuda',
        prefer_interior: bool = True
    ):
        """
        Initialize collocation sampler.

        Args:
            coords: Raw coordinates (N, 3) in meters
            has_wss: Boolean mask - True for wall points
            coord_offset: Translation offset for uniform scaling (shape (3,))
            L_ref: Reference length scale for uniform scaling
            device: PyTorch device ('cuda' or 'cpu')
            prefer_interior: Weight interior points higher for physics sampling
        """
        import torch

        self.device = device
        self.num_points = len(coords)

        # Apply uniform scaling: x* = (x - offset) / L_ref
        coords_scaled = ((coords - coord_offset) / L_ref).astype(np.float32)
        self.coords_scaled = torch.from_numpy(coords_scaled).to(device)

        # Store indices
        self.wall_indices = torch.from_numpy(np.where(has_wss)[0]).to(device)
        self.interior_indices = torch.from_numpy(np.where(~has_wss)[0]).to(device)

        # Compute sampling weights on GPU
        weights = torch.ones(self.num_points, device=device)
        if prefer_interior and len(self.interior_indices) > 0:
            weights[self.wall_indices] = 0.3
            weights[self.interior_indices] = 1.0
        self.weights = weights / weights.sum()

        # Pre-allocated buffer for sampling (avoid repeated allocations)
        self._sample_buffer = None

        # Epoch tracking for resampling
        self._current_epoch = -1
        self._epoch_indices = None

    def resample_for_epoch(self, epoch: int, num_points: int) -> None:
        """Pre-sample indices for an entire epoch (GPU-native)."""
        import torch

        if epoch != self._current_epoch:
            self._current_epoch = epoch

            # GPU-native weighted sampling using torch.multinomial
            # Note: multinomial samples WITH replacement, so we oversample
            # and take unique indices for better coverage
            if num_points <= self.num_points:
                # Sample more than needed, then take unique
                oversample = min(num_points * 2, self.num_points)
                sampled = torch.multinomial(
                    self.weights, oversample, replacement=False
                )
                self._epoch_indices = sampled[:num_points]
            else:
                # Need more points than available - sample with replacement
                self._epoch_indices = torch.multinomial(
                    self.weights, num_points, replacement=True
                )

            # Shuffle for batch randomization
            perm = torch.randperm(len(self._epoch_indices), device=self.device)
            self._epoch_indices = self._epoch_indices[perm]

    def sample(self, num_points: int) -> 'torch.Tensor':
        """Sample collocation points (returns GPU tensor directly)."""
        import torch

        num_points = min(num_points, self.num_points)
        indices = torch.multinomial(self.weights, num_points, replacement=False)
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

    def resample_for_epoch(self, epoch: int, num_points: int) -> None:
        """
        Pre-sample collocation points for an entire epoch.

        This ensures consistent sampling within an epoch while varying
        across epochs for better domain coverage over training.

        Args:
            epoch: Current epoch number (used to detect epoch changes)
            num_points: Total number of collocation points for the epoch
        """
        if epoch != self._current_epoch:
            self._current_epoch = epoch
            num_available = len(self.coords)

            # For large num_points, we may need to sample with replacement
            # but shuffle the order for better batch diversity
            if num_points <= num_available:
                self._epoch_indices = np.random.choice(
                    num_available,
                    size=num_points,
                    replace=False,
                    p=self.weights
                )
            else:
                # Sample with replacement but ensure good coverage
                self._epoch_indices = np.random.choice(
                    num_available,
                    size=num_points,
                    replace=True,
                    p=self.weights
                )

            # Shuffle to randomize batch order
            np.random.shuffle(self._epoch_indices)

    def sample(self, num_points: int, add_jitter: bool = False) -> np.ndarray:
        """
        Sample collocation points from the mesh.

        Args:
            num_points: Number of points to sample
            add_jitter: If True, add small random noise to coordinates.
                This provides implicit regularization and better coverage.

        Returns:
            Sampled coordinates (num_points, 3)
        """
        num_available = len(self.coords)
        num_points = min(num_points, num_available)

        indices = np.random.choice(
            num_available,
            size=num_points,
            replace=False,
            p=self.weights
        )

        sampled_coords = self.coords[indices].copy()

        # Add small jitter for regularization and better coverage
        if add_jitter and self.jitter_magnitude > 0:
            noise = np.random.randn(num_points, 3) * self.jitter_magnitude
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


class PatientData:
    """
    Patient-specific data for PINN training with GPU-resident tensors.

    Handles non-dimensionalization and stores all data as GPU tensors
    for efficient training without CPU-GPU transfer overhead.

    PHYSICS-BASED NON-DIMENSIONALIZATION:
    =====================================
    Uses UNIFORM SCALING to preserve geometry aspect ratios:
        - Coordinates: x* = (x - x_offset) / L_ref (same L_ref for all dimensions)
        - Velocities: u* = u / U_ref
        - Pressure: p* = p / (rho * U_ref^2)
        - WSS: tau* = tau / T_ref where T_ref = mu * U_ref / L_ref

    This yields the non-dimensional Navier-Stokes equations:
        (u*.grad*)u* = -grad*(p*) + (1/Re) * laplacian*(u*)
        div*(u*) = 0

    where Re = rho * U_ref * L_ref / mu is the Reynolds number.

    Attributes:
        coords: Scaled coordinates on GPU (N, 3)
        wss: Scaled WSS on GPU (N, 1)
        velocity: Scaled velocity on GPU (N, 3)
        normals: Normal vectors on GPU (N, 3)
        has_wss: Boolean mask on GPU (N,)
        X_raw: Raw coordinates in meters (numpy, for evaluation)
        y_raw: Raw WSS in Pa (numpy, for evaluation)
        coord_offset: Translation offset for centering (numpy, shape (3,))
        L_ref, U_ref, T_ref, Re: Reference scales for physics
    """

    def __init__(self, data: Dict[str, np.ndarray], device: str = 'cuda',
                 holdout_fraction: float = 0.0, holdout_seed: int = 0):
        """
        Initialize with data and transfer to GPU.

        Args:
            data: Dictionary from load_patient_data containing:
                - X: Coordinates (N, 3) in meters
                - y: WSS values (N,) in Pa
                - velocity: Velocity vectors (N, 3) in m/s
                - normals: Wall normal vectors (N, 3)
                - has_wss: Boolean mask for wall points
            device: PyTorch device ('cuda' or 'cpu')
            holdout_fraction: Fraction of points to withhold from training and
                evaluate as a per-patient spatial holdout (Physics of Fluids
                R1-5/R2-6: distinguishes interpolation from prediction).
                0.0 disables the split (back-compatible).
            holdout_seed: Seed for the held-out point selection so the split is
                reproducible across runs.
        """
        from src.config import RHO, MU

        self.device = device
        self.num_samples = len(data['X'])

        # Store raw data (numpy) for evaluation/denormalization
        self.X_raw = data['X']
        self.y_raw = data['y']
        self.velocity_raw = data['velocity']

        # =====================================================================
        # UNIFORM COORDINATE SCALING (preserves geometry aspect ratio)
        # =====================================================================
        # Step 1: Compute coordinate ranges
        X_min = data['X'].min(axis=0)
        X_max = data['X'].max(axis=0)
        coord_ranges = X_max - X_min

        # Step 2: L_ref is the MAXIMUM range (ensures coords fit in [0, 1])
        self.L_ref = float(np.max(coord_ranges))

        # Step 3: Center offset (shift to start from origin)
        self.coord_offset = X_min.copy()

        # Step 4: Apply UNIFORM scaling: x* = (x - offset) / L_ref
        X_scaled = (data['X'] - self.coord_offset) / self.L_ref

        # Store for inverse transform
        self.coord_ranges = coord_ranges

        # =====================================================================
        # VELOCITY SCALING: Single U_ref for all components
        # =====================================================================
        vel_magnitude = np.linalg.norm(data['velocity'], axis=1)
        self.U_ref = float(np.percentile(vel_magnitude[vel_magnitude > 0], 95))
        self.U_ref = max(self.U_ref, 1e-6)

        vel_scaled = data['velocity'] / self.U_ref

        # =====================================================================
        # PHYSICS-BASED DERIVED QUANTITIES
        # =====================================================================
        self.P_ref = RHO * self.U_ref**2  # Pressure scale
        self.Re = RHO * self.U_ref * self.L_ref / MU  # Reynolds number
        self.T_ref_physics = MU * self.U_ref / self.L_ref  # Physics-based WSS scale

        # =====================================================================
        # WSS SCALING (data-driven to keep outputs O(1))
        # =====================================================================
        # Use data-driven T_ref so that WSS* = WSS/T_ref is O(1)
        # This is critical for stable training - physics-based T_ref is too small
        # (typical arteries: T_ref_physics ~ 0.01 Pa, actual WSS ~ 1-50 Pa)
        valid_mask = ~np.isnan(data['y'])
        y_scaled = np.zeros_like(data['y'])
        if valid_mask.any():
            valid_wss = data['y'][valid_mask]
            self.wss_range = (valid_wss.min(), valid_wss.max())
            # T_ref from data: use 95th percentile to keep most values < 1
            self.T_ref = float(np.percentile(valid_wss[valid_wss > 0], 95))
            self.T_ref = max(self.T_ref, 1.0)  # At least 1 Pa
            y_scaled[valid_mask] = valid_wss / self.T_ref
        else:
            self.wss_range = (0.0, 1.0)
            self.T_ref = 10.0  # Fallback

        # =====================================================================
        # TRANSFER TO GPU
        # =====================================================================
        print(f"  Loading {self.num_samples:,} samples to {device}...")

        self.coords = torch.from_numpy(X_scaled.astype(np.float32)).to(device)
        self.wss = torch.from_numpy(y_scaled.reshape(-1, 1).astype(np.float32)).to(device)
        self.velocity = torch.from_numpy(vel_scaled.astype(np.float32)).to(device)
        self.normals = torch.from_numpy(data['normals'].astype(np.float32)).to(device)
        self.has_wss = torch.from_numpy(data['has_wss']).to(device)

        # =====================================================================
        # SPATIAL HOLDOUT SPLIT (Physics of Fluids R1-5 / R2-6)
        # =====================================================================
        # A reproducible random subset of points is withheld from training
        # and evaluated separately to distinguish predictive from interpolative
        # accuracy.
        self.holdout_fraction = float(holdout_fraction)
        self.holdout_seed = int(holdout_seed)
        if self.holdout_fraction > 0.0:
            g = torch.Generator(device='cpu').manual_seed(self.holdout_seed)
            perm = torch.randperm(self.num_samples, generator=g)
            n_hold = int(round(self.num_samples * self.holdout_fraction))
            self.holdout_indices = perm[:n_hold].to(device)
            self.train_indices = perm[n_hold:].to(device)
        else:
            self.holdout_indices = torch.empty(0, dtype=torch.long, device=device)
            self.train_indices = torch.arange(self.num_samples, device=device)
        self.num_train = int(self.train_indices.numel())
        self.num_holdout = int(self.holdout_indices.numel())

        # Shuffling state (over the training subset)
        self._indices = None
        self._current_epoch = -1

        # Memory usage
        mem_bytes = (self.coords.numel() + self.wss.numel() +
                     self.velocity.numel() + self.normals.numel()) * 4
        print(f"  Memory used: {mem_bytes / 1e6:.1f} MB")

        # Reference scales dictionary (for physics computations)
        self.ref_scales = {
            'L_ref': self.L_ref,
            'U_ref': self.U_ref,
            'T_ref': self.T_ref,  # Data-driven (for output scaling)
            'T_ref_physics': self.T_ref_physics,  # Physics-based (mu*U/L)
            'P_ref': self.P_ref,
            'Re': self.Re,
            'coord_offset': self.coord_offset.tolist(),
            'coord_ranges': self.coord_ranges.tolist()
        }

    def __len__(self) -> int:
        return self.num_samples

    def shuffle_for_epoch(self, epoch: int) -> None:
        """Shuffle the TRAINING subset for a new epoch (excludes holdout)."""
        if epoch != self._current_epoch:
            self._current_epoch = epoch
            perm = torch.randperm(self.num_train, device=self.device)
            self._indices = self.train_indices[perm]

    def get_batch(self, batch_idx: int, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Get a shuffled batch from the TRAINING subset (excludes holdout).

        Args:
            batch_idx: Batch index within epoch
            batch_size: Number of samples per batch

        Returns:
            Dictionary with coords, wss, velocity, normals, has_wss tensors
        """
        if self._indices is None:
            self.shuffle_for_epoch(0)

        n = self.num_train
        start = batch_idx * batch_size
        if start >= n:
            start = start % n
        end = min(start + batch_size, n)

        idx = self._indices[start:end]

        return {
            'coords': self.coords[idx],
            'wss': self.wss[idx],
            'velocity': self.velocity[idx],
            'normals': self.normals[idx],
            'has_wss': self.has_wss[idx]
        }

    def get_holdout(self) -> Dict[str, torch.Tensor]:
        """Return the entire held-out subset for evaluation.

        Returns an empty dict-of-tensors when holdout_fraction == 0.
        """
        idx = self.holdout_indices
        return {
            'coords': self.coords[idx],
            'wss': self.wss[idx],
            'velocity': self.velocity[idx],
            'normals': self.normals[idx],
            'has_wss': self.has_wss[idx],
            'coords_raw': self.X_raw[idx.cpu().numpy()] if idx.numel() else self.X_raw[:0],
            'wss_raw': self.y_raw[idx.cpu().numpy()] if idx.numel() else self.y_raw[:0],
        }

    def get_train_split(self) -> Dict[str, torch.Tensor]:
        """Return the full TRAINING subset (excluding holdout) for evaluation."""
        idx = self.train_indices
        return {
            'coords': self.coords[idx],
            'wss': self.wss[idx],
            'velocity': self.velocity[idx],
            'normals': self.normals[idx],
            'has_wss': self.has_wss[idx],
            'coords_raw': self.X_raw[idx.cpu().numpy()],
            'wss_raw': self.y_raw[idx.cpu().numpy()],
        }

    def get_batch_sequential(self, start: int, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Get a sequential batch for evaluation (no shuffling).

        Args:
            start: Starting index
            batch_size: Number of samples

        Returns:
            Dictionary with all data tensors for the batch
        """
        end = min(start + batch_size, self.num_samples)

        return {
            'coords': self.coords[start:end],
            'wss': self.wss[start:end],
            'velocity': self.velocity[start:end],
            'normals': self.normals[start:end],
            'has_wss': self.has_wss[start:end],
            'coords_raw': self.X_raw[start:end],
            'wss_raw': self.y_raw[start:end]
        }

    def get_reference_scales(self) -> Dict[str, float]:
        """Return reference scales for physics computations."""
        return self.ref_scales
