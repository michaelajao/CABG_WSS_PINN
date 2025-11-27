"""
Dataset module for PINN training

Contains PyTorch Dataset class and data loading utilities for hemodynamic data.
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Dict, Tuple, Optional, List
from pathlib import Path

from .config import (
    DEFAULT_TRAINING_PATIENTS, TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    DEFAULT_TRAIN_PATIENTS, DEFAULT_VAL_PATIENTS, DEFAULT_TEST_PATIENTS,
    BATCH_SIZE, RANDOM_SEED
)
from .utils import parse_cfd_file


class HemodynamicsDataset(Dataset):
    """
    PyTorch Dataset for hemodynamic data

    Handles loading and preprocessing of CFD simulation data including:
    - Spatial coordinates (x, y, z)
    - Velocity components (u, v, w)
    - Wall shear stress (WSS)
    """

    def __init__(
        self,
        data_dict: Dict[str, np.ndarray],
        scaler_X: Optional[MinMaxScaler] = None,
        scaler_y: Optional[MinMaxScaler] = None,
        scaler_vel: Optional[MinMaxScaler] = None
    ):
        """
        Initialize dataset

        Args:
            data_dict: Dictionary with 'X', 'y', 'velocity', 'has_wss' keys
                X: Spatial coordinates (N, 3) - x, y, z
                y: Wall shear stress (N,) - can contain NaN for interior points
                velocity: Velocity components (N, 3) - u, v, w
                has_wss: Boolean mask (N,) - True for wall points, False for interior
            scaler_X: MinMaxScaler for coordinates [0,1] (fit if None)
            scaler_y: MinMaxScaler for WSS [0,1] (fit if None)
            scaler_vel: MinMaxScaler for velocity [0,1] (fit if None)
        """
        self.X = data_dict['X']  # Spatial coordinates (x, y, z)
        self.y = data_dict['y']  # WSS magnitude (can have NaN)
        self.velocity = data_dict['velocity']  # Velocity components (u, v, w)
        self.has_wss = data_dict.get('has_wss', np.ones(len(data_dict['X']), dtype=bool))  # WSS availability flag

        # Fit or use provided scalers for coordinates (MinMaxScaler for physics)
        if scaler_X is None:
            self.scaler_X = MinMaxScaler(feature_range=(0, 1))
            self.X_scaled = self.scaler_X.fit_transform(self.X)
        else:
            self.scaler_X = scaler_X
            self.X_scaled = self.scaler_X.transform(self.X)

        # Fit or use provided scalers for WSS (MinMaxScaler for bounded outputs)
        # Handle NaN values by fitting only on valid WSS data
        if scaler_y is None:
            self.scaler_y = MinMaxScaler(feature_range=(0, 1))
            # Fit only on non-NaN values
            valid_mask = ~np.isnan(self.y)
            if valid_mask.any():
                self.scaler_y.fit(self.y[valid_mask].reshape(-1, 1))
            # Transform (NaN will remain NaN)
            self.y_scaled = np.full_like(self.y, np.nan)
            if valid_mask.any():
                self.y_scaled[valid_mask] = self.scaler_y.transform(self.y[valid_mask].reshape(-1, 1)).flatten()
        else:
            self.scaler_y = scaler_y
            valid_mask = ~np.isnan(self.y)
            self.y_scaled = np.full_like(self.y, np.nan)
            if valid_mask.any():
                self.y_scaled[valid_mask] = self.scaler_y.transform(self.y[valid_mask].reshape(-1, 1)).flatten()

        # Fit or use provided scalers for velocity (MinMaxScaler for bounded outputs)
        if scaler_vel is None:
            self.scaler_vel = MinMaxScaler(feature_range=(-1, 1))  # Velocity can be negative
            self.velocity_scaled = self.scaler_vel.fit_transform(self.velocity)
        else:
            self.scaler_vel = scaler_vel
            self.velocity_scaled = self.scaler_vel.transform(self.velocity)

    def __len__(self) -> int:
        """Return number of samples"""
        return len(self.X)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample

        Args:
            idx: Sample index

        Returns:
            Dictionary with tensors for coords, wss, velocity, has_wss (both scaled and raw)
        """
        return {
            'coords': torch.FloatTensor(self.X_scaled[idx]),
            'coords_raw': torch.FloatTensor(self.X[idx]),
            'wss': torch.FloatTensor([self.y_scaled[idx]]),
            'wss_raw': torch.FloatTensor([self.y[idx]]),
            'velocity': torch.FloatTensor(self.velocity_scaled[idx]),
            'velocity_raw': torch.FloatTensor(self.velocity[idx]),
            'has_wss': torch.BoolTensor([self.has_wss[idx]])
        }


def load_and_prepare_data(
    patient_files: Optional[Dict[str, Path]] = None,
    verbose: bool = True
) -> Dict[str, np.ndarray]:
    """
    Load and prepare training/validation/test datasets

    Args:
        patient_files: Dictionary mapping patient names to file paths
                      If None, uses DEFAULT_TRAINING_PATIENTS
        verbose: Whether to print loading progress

    Returns:
        Dictionary with 'X', 'y', 'velocity' keys containing concatenated data
    """
    if patient_files is None:
        patient_files = DEFAULT_TRAINING_PATIENTS

    if verbose:
        print("\n[DATA LOADING]")
        print("="*80)

    all_X, all_y, all_vel = [], [], []

    for name, path in patient_files.items():
        if not path.exists():
            if verbose:
                print(f"  Warning: File not found: {path}")
            continue

        if verbose:
            print(f"  Loading {name}...")

        df = parse_cfd_file(path)

        if df is None:
            if verbose:
                print(f"    Error: Failed to parse file")
            continue

        # Extract features - try different column name patterns
        try:
            # Coordinates
            X = df[['X [ m ]', 'Y [ m ]', 'Z [ m ]']].values

            # WSS
            y = df['Wall Shear [ Pa ]'].values

            # Velocity components
            vel = df[['Velocity u [ m s^-1 ]', 'Velocity v [ m s^-1 ]',
                     'Velocity w [ m s^-1 ]']].values

            if verbose:
                print(f"    Loaded {len(df):,} points")
                print(f"      WSS range: [{y.min():.3f}, {y.max():.3f}] Pa")

            all_X.append(X)
            all_y.append(y)
            all_vel.append(vel)

        except KeyError as e:
            if verbose:
                print(f"    Error: Missing columns: {e}")
            continue

    if len(all_X) == 0:
        raise ValueError("No data loaded! Check patient files and column names.")

    # Concatenate all data
    X_all = np.vstack(all_X)
    y_all = np.concatenate(all_y)
    vel_all = np.vstack(all_vel)

    if verbose:
        print(f"\n  Total samples: {len(X_all):,}")
        print("="*80)

    # Create dataset dictionary
    data_dict = {
        'X': X_all,
        'y': y_all,
        'velocity': vel_all
    }

    return data_dict


def load_patient_level_data(
    train_patients: Optional[List[str]] = None,
    val_patients: Optional[List[str]] = None,
    test_patients: Optional[List[str]] = None,
    patient_files: Optional[Dict[str, Path]] = None,
    verbose: bool = True
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Load data with patient-level splitting (prevents spatial data leakage)

    Args:
        train_patients: List of patient IDs for training
        val_patients: List of patient IDs for validation
        test_patients: List of patient IDs for testing
        patient_files: Dictionary mapping patient names to file paths
        verbose: Whether to print loading progress

    Returns:
        Tuple of (train_data_dict, val_data_dict, test_data_dict)
        Each dict has 'X', 'y', 'velocity' keys
    """
    if patient_files is None:
        patient_files = DEFAULT_TRAINING_PATIENTS

    if train_patients is None:
        train_patients = DEFAULT_TRAIN_PATIENTS
    if val_patients is None:
        val_patients = DEFAULT_VAL_PATIENTS
    if test_patients is None:
        test_patients = DEFAULT_TEST_PATIENTS

    if verbose:
        print("\n[PATIENT-LEVEL DATA LOADING]")
        print("="*80)
        print(f"  Train Patients: {', '.join(train_patients)}")
        print(f"  Val Patients: {', '.join(val_patients)}")
        print(f"  Test Patients: {', '.join(test_patients)}")
        print("="*80)

    def load_patient_group(patient_ids: List[str], group_name: str) -> Dict[str, np.ndarray]:
        """Load data for a group of patients"""
        all_X, all_y, all_vel = [], [], []

        if verbose:
            print(f"\n[Loading {group_name} set]")

        for patient_id in patient_ids:
            if patient_id not in patient_files:
                if verbose:
                    print(f"  Warning: Patient {patient_id} not in patient_files")
                continue

            path = patient_files[patient_id]
            if not path.exists():
                if verbose:
                    print(f"  Warning: File not found: {path}")
                continue

            if verbose:
                print(f"  Loading {patient_id}...")

            df = parse_cfd_file(path)

            if df is None:
                if verbose:
                    print(f"    Error: Failed to parse file")
                continue

            try:
                # Extract features
                X = df[['X [ m ]', 'Y [ m ]', 'Z [ m ]']].values
                y = df['Wall Shear [ Pa ]'].values
                vel = df[['Velocity u [ m s^-1 ]', 'Velocity v [ m s^-1 ]',
                         'Velocity w [ m s^-1 ]']].values

                if verbose:
                    print(f"    Loaded {len(df):,} points")
                    print(f"      WSS range: [{y.min():.3f}, {y.max():.3f}] Pa")

                all_X.append(X)
                all_y.append(y)
                all_vel.append(vel)

            except KeyError as e:
                if verbose:
                    print(f"    Error: Missing columns: {e}")
                continue

        if len(all_X) == 0:
            raise ValueError(f"No data loaded for {group_name} set!")

        # Concatenate all patient data for this group
        return {
            'X': np.vstack(all_X),
            'y': np.concatenate(all_y),
            'velocity': np.vstack(all_vel)
        }

    # Load each group separately
    train_data = load_patient_group(train_patients, "Training")
    val_data = load_patient_group(val_patients, "Validation")
    test_data = load_patient_group(test_patients, "Test")

    if verbose:
        print("\n[PATIENT-LEVEL SPLIT SUMMARY]")
        print("="*80)
        print(f"  Train: {len(train_data['X']):,} samples ({len(train_patients)} patients)")
        print(f"  Validation: {len(val_data['X']):,} samples ({len(val_patients)} patients)")
        print(f"  Test: {len(test_data['X']):,} samples ({len(test_patients)} patients)")
        print(f"  Total: {len(train_data['X']) + len(val_data['X']) + len(test_data['X']):,} samples")
        print("="*80)

    return train_data, val_data, test_data


def create_dataloaders(
    data_dict: Dict[str, np.ndarray],
    batch_size: int = BATCH_SIZE,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    num_workers: int = 0,
    pin_memory: bool = True,
    verbose: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader, HemodynamicsDataset]:
    """
    Create train, validation, and test data loaders

    Args:
        data_dict: Dictionary with 'X', 'y', 'velocity' arrays
        batch_size: Batch size for training
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for GPU transfer
        verbose: Whether to print dataset info

    Returns:
        Tuple of (train_loader, val_loader, test_loader, full_dataset)
    """
    # Create full dataset
    full_dataset = HemodynamicsDataset(data_dict)

    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )

    if verbose:
        print("\n[DATASET SPLIT]")
        print("="*80)
        print(f"  Train: {len(train_dataset):,} samples ({train_ratio*100:.0f}%)")
        print(f"  Validation: {len(val_dataset):,} samples ({val_ratio*100:.0f}%)")
        print(f"  Test: {len(test_dataset):,} samples ({test_ratio*100:.0f}%)")
        print("="*80)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader, full_dataset


def create_patient_level_dataloaders(
    train_data: Dict[str, np.ndarray],
    val_data: Dict[str, np.ndarray],
    test_data: Dict[str, np.ndarray],
    batch_size: int = BATCH_SIZE,
    num_workers: int = 0,
    pin_memory: bool = True,
    verbose: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader, HemodynamicsDataset]:
    """
    Create dataloaders from patient-level split data with proper scaler fitting

    This function prevents data leakage by:
    1. Fitting scalers ONLY on training data
    2. Applying same scalers to validation and test data
    3. Maintaining patient-level separation

    Args:
        train_data: Training data dictionary with 'X', 'y', 'velocity' keys
        val_data: Validation data dictionary with 'X', 'y', 'velocity' keys
        test_data: Test data dictionary with 'X', 'y', 'velocity' keys
        batch_size: Batch size for dataloaders
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for GPU transfer
        verbose: Whether to print dataset info

    Returns:
        Tuple of (train_loader, val_loader, test_loader, train_dataset)
        Note: Returns train_dataset (with fitted scalers) for later use
    """
    # Create training dataset (fits scalers)
    train_dataset = HemodynamicsDataset(train_data)

    # Create val/test datasets using training scalers (prevents data leakage)
    val_dataset = HemodynamicsDataset(
        val_data,
        scaler_X=train_dataset.scaler_X,
        scaler_y=train_dataset.scaler_y,
        scaler_vel=train_dataset.scaler_vel
    )

    test_dataset = HemodynamicsDataset(
        test_data,
        scaler_X=train_dataset.scaler_X,
        scaler_y=train_dataset.scaler_y,
        scaler_vel=train_dataset.scaler_vel
    )

    if verbose:
        print("\n[DATALOADER CREATION]")
        print("="*80)
        print(f"  Train: {len(train_dataset):,} samples")
        print(f"  Validation: {len(val_dataset):,} samples")
        print(f"  Test: {len(test_dataset):,} samples")
        print(f"  Batch size: {batch_size}")
        print("="*80)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader, train_dataset


def load_wall_and_streamline_data(
    patient_files: Dict[str, Path],
    streamline_files: Dict[str, Path],
    verbose: bool = True
) -> Dict[str, np.ndarray]:
    """
    Load both wall (WSS) and streamline (interior velocity) data

    Args:
        patient_files: Dictionary mapping patient names to wall CSV paths
        streamline_files: Dictionary mapping patient names to streamline CSV paths
        verbose: Whether to print loading progress

    Returns:
        Dictionary with 'X', 'y', 'velocity', 'has_wss' keys
        - X: Coordinates (N, 3)
        - y: WSS (N,) - NaN for interior points
        - velocity: Velocity components (N, 3)
        - has_wss: Boolean mask (N,) - True for wall, False for interior
    """
    if verbose:
        print("\n[HYBRID DATA LOADING: Wall + Streamline]")
        print("="*80)

    all_X, all_y, all_vel, all_has_wss = [], [], [], []

    for name, wall_path in patient_files.items():
        # Load wall data (WSS + u,v,w=0)
        if wall_path.exists():
            if verbose:
                print(f"  Loading {name} (Wall)...")

            df_wall = parse_cfd_file(wall_path)
            if df_wall is not None:
                try:
                    X_wall = df_wall[['X [ m ]', 'Y [ m ]', 'Z [ m ]']].values
                    y_wall = df_wall['Wall Shear [ Pa ]'].values
                    vel_wall = df_wall[['Velocity u [ m s^-1 ]', 'Velocity v [ m s^-1 ]',
                                       'Velocity w [ m s^-1 ]']].values

                    all_X.append(X_wall)
                    all_y.append(y_wall)
                    all_vel.append(vel_wall)
                    all_has_wss.append(np.ones(len(df_wall), dtype=bool))

                    if verbose:
                        print(f"    Wall: {len(df_wall):,} points, WSS: [{y_wall.min():.3f}, {y_wall.max():.3f}] Pa")

                except KeyError as e:
                    if verbose:
                        print(f"    Error: Missing columns: {e}")

        # Load streamline data (interior velocity + no WSS)
        if name in streamline_files:
            stream_paths = streamline_files[name]
            if not isinstance(stream_paths, list):
                stream_paths = [stream_paths]

            for stream_path in stream_paths:
                if stream_path.exists():
                    if verbose:
                        print(f"  Loading {name} (Streamlines: {stream_path.name})...")

                    df_stream = parse_cfd_file(stream_path)
                    if df_stream is not None:
                        try:
                            X_stream = df_stream[['X [ m ]', 'Y [ m ]', 'Z [ m ]']].values
                            vel_stream = df_stream[['Velocity u [ m s^-1 ]', 'Velocity v [ m s^-1 ]',
                                                    'Velocity w [ m s^-1 ]']].values

                            # No WSS for interior points
                            y_stream = np.full(len(df_stream), np.nan)

                            all_X.append(X_stream)
                            all_y.append(y_stream)
                            all_vel.append(vel_stream)
                            all_has_wss.append(np.zeros(len(df_stream), dtype=bool))

                            if verbose:
                                vel_mag = np.linalg.norm(vel_stream, axis=1)
                                print(f"    Streamlines: {len(df_stream):,} points, Vel: [{vel_mag.min():.3f}, {vel_mag.max():.3f}] m/s")

                        except KeyError as e:
                            if verbose:
                                print(f"    Error: Missing columns: {e}")

    if len(all_X) == 0:
        raise ValueError("No data loaded! Check patient files and column names.")

    # Concatenate all data
    X_all = np.vstack(all_X)
    y_all = np.concatenate(all_y)
    vel_all = np.vstack(all_vel)
    has_wss_all = np.concatenate(all_has_wss)

    if verbose:
        wall_count = has_wss_all.sum()
        interior_count = (~has_wss_all).sum()
        print(f"\n  Total samples: {len(X_all):,}")
        print(f"    Wall points (WSS): {wall_count:,} ({wall_count/len(X_all)*100:.1f}%)")
        print(f"    Interior points (velocity): {interior_count:,} ({interior_count/len(X_all)*100:.1f}%)")
        print("="*80)

    return {
        'X': X_all,
        'y': y_all,
        'velocity': vel_all,
        'has_wss': has_wss_all
    }


def load_patient_level_hybrid_data(
    train_patients: Optional[List[str]] = None,
    val_patients: Optional[List[str]] = None,
    test_patients: Optional[List[str]] = None,
    patient_files: Optional[Dict[str, Path]] = None,
    streamline_files: Optional[Dict[str, Path]] = None,
    verbose: bool = True
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Load hybrid wall+streamline data with patient-level splitting

    Args:
        train_patients: List of patient IDs for training
        val_patients: List of patient IDs for validation
        test_patients: List of patient IDs for testing
        patient_files: Dictionary mapping patient names to wall CSV paths
        streamline_files: Dictionary mapping patient names to streamline CSV paths
        verbose: Whether to print loading progress

    Returns:
        Tuple of (train_data_dict, val_data_dict, test_data_dict)
    """
    from .config import (DEFAULT_TRAINING_PATIENTS, STREAMLINE_FILES,
                         DEFAULT_TRAIN_PATIENTS, DEFAULT_VAL_PATIENTS, DEFAULT_TEST_PATIENTS)

    if patient_files is None:
        patient_files = DEFAULT_TRAINING_PATIENTS
    if streamline_files is None:
        streamline_files = STREAMLINE_FILES

    if train_patients is None:
        train_patients = DEFAULT_TRAIN_PATIENTS
    if val_patients is None:
        val_patients = DEFAULT_VAL_PATIENTS
    if test_patients is None:
        test_patients = DEFAULT_TEST_PATIENTS

    if verbose:
        print("\n[PATIENT-LEVEL HYBRID DATA LOADING]")
        print("="*80)
        print(f"  Train Patients: {', '.join(train_patients)}")
        print(f"  Val Patients: {', '.join(val_patients)}")
        print(f"  Test Patients: {', '.join(test_patients)}")
        print("="*80)

    def load_patient_group(patient_ids: List[str], group_name: str) -> Dict[str, np.ndarray]:
        """Load hybrid data for a group of patients"""
        group_patient_files = {pid: patient_files[pid] for pid in patient_ids if pid in patient_files}
        group_streamline_files = {pid: streamline_files[pid] for pid in patient_ids if pid in streamline_files}

        if verbose:
            print(f"\n[Loading {group_name} set]")

        return load_wall_and_streamline_data(group_patient_files, group_streamline_files, verbose)

    # Load each group
    train_data = load_patient_group(train_patients, "Training")
    val_data = load_patient_group(val_patients, "Validation")
    test_data = load_patient_group(test_patients, "Test")

    if verbose:
        print("\n[HYBRID SPLIT SUMMARY]")
        print("="*80)
        print(f"  Train: {len(train_data['X']):,} samples ({train_data['has_wss'].sum():,} wall + {(~train_data['has_wss']).sum():,} interior)")
        print(f"  Val: {len(val_data['X']):,} samples ({val_data['has_wss'].sum():,} wall + {(~val_data['has_wss']).sum():,} interior)")
        print(f"  Test: {len(test_data['X']):,} samples ({test_data['has_wss'].sum():,} wall + {(~test_data['has_wss']).sum():,} interior)")
        print("="*80)

    return train_data, val_data, test_data


def sample_collocation_points(
    data_dict: Dict[str, np.ndarray],
    n_points: int,
    method: str = 'mesh'
) -> np.ndarray:
    """
    Sample collocation points from existing mesh for physics enforcement

    CRITICAL: Samples from actual mesh points instead of random uniform sampling.
    This ensures collocation points are inside the vessel geometry, not in empty space.

    Args:
        data_dict: Dictionary with 'X' key containing coordinate data
        n_points: Number of collocation points to sample
        method: Sampling method
            - 'mesh': Sample from existing mesh points (RECOMMENDED)
            - 'uniform': Random uniform in bounding box (may sample outside vessel)
            - 'latin_hypercube': LHS in bounding box (may sample outside vessel)

    Returns:
        Array of collocation coordinates (n_points, 3)
    """
    X = data_dict['X']

    if method == 'mesh':
        # Sample from existing mesh points (ensures points are inside vessel)
        n_available = len(X)
        if n_points >= n_available:
            # If requesting more points than available, sample with replacement
            indices = np.random.choice(n_available, size=n_points, replace=True)
        else:
            # Sample without replacement
            indices = np.random.choice(n_available, size=n_points, replace=False)
        collocation_coords = X[indices]
    elif method == 'uniform':
        # Get bounding box
        x_min, y_min, z_min = X.min(axis=0)
        x_max, y_max, z_max = X.max(axis=0)
        # WARNING: May sample points outside vessel geometry
        collocation_coords = np.random.uniform(
            low=[x_min, y_min, z_min],
            high=[x_max, y_max, z_max],
            size=(n_points, 3)
        )
    elif method == 'latin_hypercube':
        # Get bounding box
        x_min, y_min, z_min = X.min(axis=0)
        x_max, y_max, z_max = X.max(axis=0)
        # Latin hypercube sampling (better coverage)
        from scipy.stats import qmc
        sampler = qmc.LatinHypercube(d=3)
        sample = sampler.random(n=n_points)
        # WARNING: May sample points outside vessel geometry
        collocation_coords = qmc.scale(sample, [x_min, y_min, z_min], [x_max, y_max, z_max])
    else:
        raise ValueError(f"Unknown sampling method: {method}. Use 'mesh', 'uniform', or 'latin_hypercube'")

    return collocation_coords
