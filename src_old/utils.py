"""
Utility functions for data loading and processing

Contains common utilities used across the PINN pipeline including:
- CSV file parsing
- Dataset discovery
- Data preprocessing helpers
"""

import pandas as pd
import numpy as np
from pathlib import Path
from io import StringIO
from typing import Optional, Dict, List
import warnings

from .config import PINNS_PATH


def parse_cfd_file(filepath: Path) -> Optional[pd.DataFrame]:
    """
    Parse CFD CSV file with [Name] and [Data] sections

    Args:
        filepath: Path to CSV file

    Returns:
        DataFrame with parsed data, or None if parsing fails
    """
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        # Find [Data] section
        data_start = None
        for i, line in enumerate(lines):
            if '[Data]' in line:
                data_start = i + 1
                break

        if data_start is None:
            # Try standard CSV format
            try:
                return pd.read_csv(filepath)
            except:
                return pd.read_csv(filepath, header=None)

        # Read data from [Data] section onwards
        data_lines = []
        header_line = None

        for i in range(data_start, len(lines)):
            line = lines[i].strip()
            if not line or line.startswith('['):
                continue

            if header_line is None:
                header_line = line
                continue

            # Try to parse as data
            try:
                values = [float(x.strip()) for x in line.split(',') if x.strip()]
                if len(values) > 0:
                    data_lines.append(line)
            except ValueError:
                continue

        if not data_lines or header_line is None:
            return None

        # Create dataframe
        csv_text = header_line + '\n' + '\n'.join(data_lines)
        df = pd.read_csv(StringIO(csv_text))

        # Strip whitespace from column names
        df.columns = df.columns.str.strip()

        return df

    except Exception as e:
        warnings.warn(f"Error parsing {filepath}: {str(e)}")
        return None


def get_all_datasets() -> Dict[str, Dict[str, Optional[Path]]]:
    """
    Get all available datasets from PINNS folder

    Returns:
        Dictionary mapping patient IDs to their main and streamline file paths
        Format: {patient_id: {'main': Path, 'streamlines': Path}}
    """
    cases = {}
    files = list(PINNS_PATH.glob("*.csv"))

    # Group files by patient/case
    for f in files:
        name = f.stem
        # Skip streamlines for now, we'll handle them separately
        if 'Streamlines' in name or 'streamlines' in name:
            continue

        base_name = name.split('.')[0]  # Get patient ID
        if base_name not in cases:
            cases[base_name] = {'main': None, 'streamlines': None}
        cases[base_name]['main'] = f

    # Add streamlines
    for f in files:
        name = f.stem
        if 'Streamlines' in name or 'streamlines' in name:
            # Try to match with main file
            for base_name in cases:
                if base_name in name:
                    cases[base_name]['streamlines'] = f
                    break

    return cases


def find_column(df: pd.DataFrame, possible_names: List[str]) -> Optional[str]:
    """
    Find column in DataFrame that matches one of the possible names

    Args:
        df: DataFrame to search
        possible_names: List of possible column names (in order of preference)

    Returns:
        Column name if found, None otherwise
    """
    for name in possible_names:
        if name in df.columns:
            return name
    return None


def get_column_safe(df: pd.DataFrame, column_key: str, column_mappings: Dict[str, List[str]]) -> Optional[str]:
    """
    Safely get column name from DataFrame using mapping

    Args:
        df: DataFrame to search
        column_key: Key in column_mappings (e.g., 'X', 'Y', 'WSS')
        column_mappings: Dictionary mapping keys to possible column names

    Returns:
        Actual column name in DataFrame, or None if not found
    """
    if column_key not in column_mappings:
        return None

    possible_names = column_mappings[column_key]
    return find_column(df, possible_names)


def extract_patient_id(filename: str) -> str:
    """
    Extract patient ID from filename

    Args:
        filename: Name of the file

    Returns:
        Patient ID string
    """
    # Remove extension
    name = Path(filename).stem

    # Extract patient ID (first part before space or dot)
    patient_id = name.split('.')[0].split()[0]

    return patient_id


def extract_vessel_type(filename: str) -> str:
    """
    Extract vessel type from filename

    Args:
        filename: Name of the file

    Returns:
        Vessel type (LCA, RCA, G1, G2, G3, Aorta, or Unknown)
    """
    filename_upper = filename.upper()

    if 'LCA' in filename_upper:
        return 'LCA'
    elif 'RCA' in filename_upper:
        return 'RCA'
    elif 'G1' in filename_upper or 'GRAFT1' in filename_upper:
        return 'G1'
    elif 'G2' in filename_upper or 'GRAFT2' in filename_upper:
        return 'G2'
    elif 'G3' in filename_upper or 'GRAFT3' in filename_upper:
        return 'G3'
    elif 'AORTA' in filename_upper:
        return 'Aorta'
    else:
        return 'Unknown'


def compute_velocity_magnitude(u: np.ndarray, v: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Compute velocity magnitude from components

    Args:
        u: x-component of velocity
        v: y-component of velocity
        w: z-component of velocity

    Returns:
        Velocity magnitude array
    """
    return np.sqrt(u**2 + v**2 + w**2)


def convert_units(value: float, from_unit: str, to_unit: str) -> float:
    """
    Convert between common units used in hemodynamics

    Args:
        value: Value to convert
        from_unit: Source unit (e.g., 'Pa', 'dynes/cm2', 'm', 'mm')
        to_unit: Target unit

    Returns:
        Converted value
    """
    # Pressure conversions
    if from_unit == 'Pa' and to_unit == 'dynes/cm2':
        return value * 10.0
    elif from_unit == 'dynes/cm2' and to_unit == 'Pa':
        return value / 10.0

    # Length conversions
    elif from_unit == 'm' and to_unit == 'mm':
        return value * 1000.0
    elif from_unit == 'mm' and to_unit == 'm':
        return value / 1000.0

    # No conversion needed
    elif from_unit == to_unit:
        return value

    else:
        raise ValueError(f"Unsupported unit conversion: {from_unit} to {to_unit}")


def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Validate that DataFrame contains required columns

    Args:
        df: DataFrame to validate
        required_columns: List of required column names

    Returns:
        True if all required columns present, False otherwise
    """
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        warnings.warn(f"Missing required columns: {missing}")
        return False

    return True


def sample_dataframe(df: pd.DataFrame, n_samples: int, random_state: int = 42) -> pd.DataFrame:
    """
    Sample DataFrame intelligently - sample if too large, return as-is if small enough

    Args:
        df: DataFrame to sample
        n_samples: Maximum number of samples
        random_state: Random seed

    Returns:
        Sampled DataFrame
    """
    if len(df) > n_samples:
        return df.sample(n=n_samples, random_state=random_state)
    else:
        return df


def print_dataset_info(df: pd.DataFrame, name: str = "Dataset"):
    """
    Print summary information about a dataset

    Args:
        df: DataFrame to summarize
        name: Name of the dataset
    """
    print(f"\n{name}:")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Memory: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")

    # Print statistics for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(f"  Numeric columns: {len(numeric_cols)}")
        print(f"\n  Summary statistics:")
        print(df[numeric_cols].describe())
