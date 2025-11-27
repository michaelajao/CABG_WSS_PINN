"""
Comprehensive Exploratory Data Analysis for CABG Hemodynamics Dataset
Physics-Informed Neural Networks (PINNs) for WSS Prediction

Features:
- Dataset structure analysis with comprehensive statistics
- WSS and velocity distribution analysis
- 2D plane visualizations (XY and XZ cuts) for all datasets
- 3D interactive visualizations
- Newtonian vs Non-Newtonian model comparisons
- Correlation analysis and statistical metrics

Author: Research Team
Date: November 8, 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import warnings
from pathlib import Path
from tqdm import tqdm
import json
from scipy.stats import skew, kurtosis

# Import from modular src package
from src_old import config
from src_old.utils import parse_cfd_file, get_all_datasets
from src_old.plots import plot_wss_distributions, create_2d_plane_cuts

warnings.filterwarnings('ignore')

# Plotting style
plt.style.use(config.PLOT_STYLE)
sns.set_palette("husl")

print("="*80)
print("COMPREHENSIVE EXPLORATORY DATA ANALYSIS")
print("Physics-Informed Neural Networks for WSS Prediction")
print("="*80)

# =============================================================================
# PART 1: DATA LOADING AND STRUCTURE ANALYSIS
# =============================================================================

def load_dataset_summary():
    """Load and summarize all datasets"""
    print("\n" + "="*80)
    print("PART 1: DATASET STRUCTURE ANALYSIS")
    print("="*80)

    summary_data = []

    # Load PINNS folder files
    print("\n[1.1] Scanning PINNS folder (Full CFD Data)...")
    pinns_files = list(config.PINNS_PATH.glob("*.csv"))

    for file in tqdm(pinns_files, desc="Processing PINNS files"):
        try:
            df = parse_cfd_file(file)
            if df is None:
                continue

            # Determine file type
            if 'Streamlines' in file.name or 'streamlines' in file.name.lower():
                file_type = 'Streamline'
            elif 'WSS' in file.name:
                file_type = 'Wall'
            else:
                # Check if it has WSS columns
                if any('Wall Shear' in col for col in df.columns):
                    file_type = 'Wall'
                else:
                    file_type = 'Other'

            # Extract patient ID
            patient_id = file.name.split('.')[0].split()[0]

            # Extract vessel info
            vessel = 'Unknown'
            if 'LCA' in file.name:
                vessel = 'LCA'
            elif 'RCA' in file.name:
                vessel = 'RCA'
            elif 'G1' in file.name:
                vessel = 'G1'
            elif 'G2' in file.name:
                vessel = 'G2'
            elif 'G3' in file.name:
                vessel = 'G3'
            elif 'aorta' in str(df.columns).lower():
                vessel = 'Aorta'

            summary_data.append({
                'Source': 'PINNS',
                'Patient': patient_id,
                'Vessel': vessel,
                'Type': file_type,
                'Filename': file.name,
                'Rows': len(df),
                'Columns': len(df.columns),
                'File_Size_MB': file.stat().st_size / (1024**2),
                'Has_Velocity': any('Velocity' in col for col in df.columns),
                'Has_WSS': any('Wall Shear' in col for col in df.columns),
                'Has_Coordinates': all(c in ' '.join(df.columns) for c in ['X', 'Y', 'Z'])
            })
        except Exception as e:
            print(f"Error processing {file.name}: {e}")

    # Load statistical_data folder
    print("\n[1.2] Scanning statical_data folder (Newtonian vs Non-Newtonian)...")
    for patient_dir in config.STAT_PATH.iterdir():
        if patient_dir.is_dir():
            # Newtonian files
            for file in patient_dir.glob("*.csv"):
                try:
                    df = pd.read_csv(file, header=None)
                    summary_data.append({
                        'Source': 'statical_data',
                        'Patient': patient_dir.name,
                        'Vessel': file.stem,
                        'Type': 'Newtonian',
                        'Filename': f"{patient_dir.name}/{file.name}",
                        'Rows': len(df),
                        'Columns': len(df.columns),
                        'File_Size_MB': file.stat().st_size / (1024**2),
                        'Has_Velocity': False,
                        'Has_WSS': True,
                        'Has_Coordinates': True
                    })
                except:
                    pass

            # Non-Newtonian files
            for nn_dir in patient_dir.glob("*ewtonian"):
                if nn_dir.is_dir():
                    for file in nn_dir.glob("*.csv"):
                        try:
                            df = pd.read_csv(file, header=None)
                            summary_data.append({
                                'Source': 'statical_data',
                                'Patient': patient_dir.name,
                                'Vessel': file.stem,
                                'Type': 'Non-Newtonian',
                                'Filename': f"{patient_dir.name}/{nn_dir.name}/{file.name}",
                                'Rows': len(df),
                                'Columns': len(df.columns),
                                'File_Size_MB': file.stat().st_size / (1024**2),
                                'Has_Velocity': False,
                                'Has_WSS': True,
                                'Has_Coordinates': True
                            })
                        except:
                            pass

    df_summary = pd.DataFrame(summary_data)

    # Save summary
    df_summary.to_csv(config.STATISTICS_PATH / "dataset_summary.csv", index=False)

    print(f"\nTotal files processed: {len(df_summary)}")
    print(f"PINNS files: {len(df_summary[df_summary['Source']=='PINNS'])}")
    print(f"Statistical files: {len(df_summary[df_summary['Source']=='statical_data'])}")
    print(f"Total data points: {df_summary['Rows'].sum():,}")
    print(f"Total data size: {df_summary['File_Size_MB'].sum():.2f} MB")

    return df_summary

# =============================================================================
# PART 2: DETAILED STATISTICAL ANALYSIS
# =============================================================================

def compute_statistics():
    """Compute comprehensive statistics for all datasets"""
    print("\n" + "="*80)
    print("PART 2: STATISTICAL ANALYSIS")
    print("="*80)

    stats_results = {}

    # Select representative files for detailed analysis
    analysis_files = {
        'H-12_Healthy': config.PINNS_PATH / "H-12.csv",
        'D-10_Diseased': config.PINNS_PATH / "D-10.csv",
        '0149_Graft': config.PINNS_PATH / "0149.csv",
        '0073': config.PINNS_PATH / "0073.csv"
    }

    for name, filepath in analysis_files.items():
        if not filepath.exists():
            print(f"[WARNING] File not found: {filepath}")
            continue

        print(f"\n[2.{list(analysis_files.keys()).index(name)+1}] Analyzing {name}...")

        try:
            df = parse_cfd_file(filepath)
            if df is None:
                continue

            # Basic statistics
            stats = {
                'n_points': len(df),
                'spatial_extent': {},
                'velocity_stats': {},
                'wss_stats': {}
            }

            # Spatial extent
            for coord in ['X [ m ]', 'Y [ m ]', 'Z [ m ]']:
                if coord in df.columns:
                    stats['spatial_extent'][coord] = {
                        'min': float(df[coord].min()),
                        'max': float(df[coord].max()),
                        'mean': float(df[coord].mean()),
                        'std': float(df[coord].std()),
                        'range': float(df[coord].max() - df[coord].min())
                    }

            # Velocity statistics
            vel_col = 'Velocity [ m s^-1 ]'
            if vel_col in df.columns:
                stats['velocity_stats'] = {
                    'min': float(df[vel_col].min()),
                    'max': float(df[vel_col].max()),
                    'mean': float(df[vel_col].mean()),
                    'std': float(df[vel_col].std()),
                    'median': float(df[vel_col].median()),
                    'q25': float(df[vel_col].quantile(0.25)),
                    'q75': float(df[vel_col].quantile(0.75))
                }

            # WSS statistics
            wss_col = 'Wall Shear [ Pa ]'
            if wss_col in df.columns:
                wss_data = df[wss_col]
                stats['wss_stats'] = {
                    'min': float(wss_data.min()),
                    'max': float(wss_data.max()),
                    'mean': float(wss_data.mean()),
                    'std': float(wss_data.std()),
                    'median': float(wss_data.median()),
                    'q25': float(wss_data.quantile(0.25)),
                    'q75': float(wss_data.quantile(0.75)),
                    'percent_high_risk': float((wss_data > config.HIGH_RISK_WSS_THRESHOLD).sum() / len(wss_data) * 100),
                    'skewness': float(skew(wss_data)),
                    'kurtosis': float(kurtosis(wss_data))
                }

            stats_results[name] = stats

            # Print summary
            print(f"  - Points: {stats['n_points']:,}")
            if stats['wss_stats']:
                print(f"  - WSS range: [{stats['wss_stats']['min']:.3f}, {stats['wss_stats']['max']:.3f}] Pa")
                print(f"  - WSS mean: {stats['wss_stats']['mean']:.3f} +/- {stats['wss_stats']['std']:.3f} Pa")
                print(f"  - High-risk (>{config.HIGH_RISK_WSS_THRESHOLD} Pa): {stats['wss_stats']['percent_high_risk']:.2f}%")

        except Exception as e:
            print(f"  [ERROR] Error: {e}")

    # Save statistics
    with open(config.STATISTICS_PATH / "detailed_statistics.json", 'w') as f:
        json.dump(stats_results, f, indent=2)

    return stats_results

# =============================================================================
# PART 3: DATA VISUALIZATION - DISTRIBUTIONS
# =============================================================================

def create_distribution_plots(stats_results):
    """Create comprehensive distribution plots"""
    print("\n" + "="*80)
    print("PART 3: DISTRIBUTION ANALYSIS")
    print("="*80)

    # Load sample datasets
    datasets = {}
    files_to_load = {
        'H-12 (Healthy)': config.PINNS_PATH / "H-12.csv",
        'D-10 (Diseased)': config.PINNS_PATH / "D-10.csv",
        '0149 (Graft)': config.PINNS_PATH / "0149.csv"
    }

    for name, path in files_to_load.items():
        if path.exists():
            df = parse_cfd_file(path)
            if df is not None:
                wss_col = next((c for c in df.columns if 'Wall Shear' in c), None)
                if wss_col:
                    datasets[name] = pd.DataFrame({'WSS': df[wss_col]})

    if len(datasets) == 0:
        print("  [WARNING] No datasets loaded for distribution analysis")
        return

    # Create distribution plots using src/plots.py
    print("\n[3.1] Creating WSS distribution plots...")
    plot_wss_distributions(datasets, save_path=config.FIGURES_PATH)


# =============================================================================
# PART 4: 2D PLANE VISUALIZATIONS
# =============================================================================

def create_all_2d_planes():
    """Generate 2D plane visualizations for all datasets"""
    print("\n" + "="*80)
    print("PART 4: 2D PLANE VISUALIZATIONS")
    print("="*80)

    # Get all datasets
    cases = get_all_datasets()
    print(f"\nFound {len(cases)} cases to process\n")

    total_figures = 0

    for i, (case_name, files) in enumerate(cases.items(), 1):
        print(f"\n[4.{i}] Processing: {case_name}")

        # Process main file
        if files['main']:
            print(f"  Loading main data...")
            df_main = parse_cfd_file(files['main'])
            if df_main is not None:
                print(f"    Loaded {len(df_main):,} points")

                # Prepare data with standardized columns
                coords_cols = [c for c in df_main.columns if any(x in c for x in ['X [', 'Y [', 'Z ['])]
                wss_col = next((c for c in df_main.columns if 'Wall Shear' in c), None)

                if len(coords_cols) >= 3 and wss_col:
                    viz_df = pd.DataFrame({
                        'X': df_main[coords_cols[0]],
                        'Y': df_main[coords_cols[1]],
                        'Z': df_main[coords_cols[2]],
                        'WSS': df_main[wss_col]
                    })
                    create_2d_plane_cuts(viz_df, case_name, save_path=config.PLANES_2D_PATH)
                    total_figures += 1
            else:
                print(f"    Failed to load main data")

    print(f"\nTotal 2D plane figures generated: {total_figures}")


# =============================================================================
# PART 5: 3D VISUALIZATIONS
# =============================================================================

def create_3d_visualizations():
    """Create 3D visualizations of vessel geometry and WSS distribution"""
    print("\n" + "="*80)
    print("PART 5: 3D SPATIAL VISUALIZATIONS")
    print("="*80)

    # Load representative dataset
    print("\n[5.1] Creating 3D scatter plots...")

    datasets_3d = {
        'H-12_Healthy': config.PINNS_PATH / "H-12.csv",
        'D-10_Diseased': config.PINNS_PATH / "D-10.csv",
        '0149_Graft': config.PINNS_PATH / "0149.csv"
    }

    for name, path in datasets_3d.items():
        if not path.exists():
            continue

        print(f"\n  Processing {name}...")
        df = parse_cfd_file(path)

        if df is None:
            continue

        # Sample data for visualization (too many points otherwise)
        if len(df) > config.MAX_3D_POINTS:
            df_viz = df.sample(n=config.MAX_3D_POINTS, random_state=42)
        else:
            df_viz = df

        # 3D scatter with WSS coloring
        wss_col = 'Wall Shear [ Pa ]'
        if wss_col in df_viz.columns:
            fig = go.Figure(data=[go.Scatter3d(
                x=df_viz['X [ m ]'] * 1000,  # Convert to mm
                y=df_viz['Y [ m ]'] * 1000,
                z=df_viz['Z [ m ]'] * 1000,
                mode='markers',
                marker=dict(
                    size=2,
                    color=df_viz[wss_col],
                    colorscale='Jet',
                    colorbar=dict(title="WSS (Pa)"),
                    cmin=0,
                    cmax=df_viz[wss_col].quantile(0.95)
                ),
                text=[f'WSS: {w:.3f} Pa' for w in df_viz[wss_col]],
                hoverinfo='text'
            )])

            fig.update_layout(
                title=f'3D Vessel Geometry with WSS Distribution<br>{name}',
                scene=dict(
                    xaxis_title='X (mm)',
                    yaxis_title='Y (mm)',
                    zaxis_title='Z (mm)',
                    aspectmode='data'
                ),
                width=1000,
                height=800
            )

            fig.write_html(config.VISUALIZATIONS_3D_PATH / f"3d_wss_{name}.html")
            print(f"  Saved: 3d_wss_{name}.html")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""
    print("\n>> Starting Comprehensive EDA and Visualization...\n")

    # Part 1: Dataset structure
    df_summary = load_dataset_summary()

    # Part 2: Statistical analysis
    stats_results = compute_statistics()

    # Part 3: Distribution plots
    create_distribution_plots(stats_results)

    # Part 4: 2D plane visualizations
    create_all_2d_planes()

    # Part 5: 3D visualizations
    create_3d_visualizations()

    print("\n" + "="*80)
    print(">> COMPREHENSIVE EDA AND VISUALIZATION COMPLETE")
    print("="*80)
    print(f"\nAll outputs saved to: {config.OUTPUT_PATH}")
    print("\nGenerated files:")
    print(f"  - Statistics: {config.STATISTICS_PATH}/")
    print(f"  - Figures: {config.FIGURES_PATH}/")
    print(f"  - 3D Visualizations: {config.VISUALIZATIONS_3D_PATH}/")
    print(f"  - 2D Planes: {config.PLANES_2D_PATH}/")
    print("\n" + "="*80)

    return df_summary, stats_results


if __name__ == "__main__":
    main()
