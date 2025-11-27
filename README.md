# Physics-Informed Neural Networks for Coronary Artery WSS Prediction

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)

## Overview

This repository implements **Physics-Informed Neural Networks (PINNs)** for predicting Wall Shear Stress (WSS) in coronary arteries and bypass grafts. The model learns from CFD simulation data while enforcing the incompressible Navier-Stokes equations as physics constraints.

### Key Features

- **Physics-Informed Learning**: Incorporates Navier-Stokes momentum and continuity equations
- **Multi-Output Architecture**: Predicts velocity (u, v, w), pressure (p), and WSS simultaneously
- **Patient-Specific Models**: Per-patient training with organized output structure
- **Two Architecture Options**: SharedTrunk (efficient) or Multi-network (separate encoders)
- **Publication-Quality Plots**: Automatic generation of CFD vs PINN comparison figures

---

## Installation

### Prerequisites

- NVIDIA GPU with 8GB+ VRAM (tested on RTX 5060 Ti)
- CUDA 12.x
- Python 3.10+
- Conda (recommended)

### Setup

```bash
# Create conda environment
conda create -n dl_env python=3.10
conda activate dl_env

# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install dependencies
pip install numpy pandas matplotlib scikit-learn scipy tqdm

# Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## Quick Start

### Train a Single Patient

```bash
python main.py --patient 0156 --epochs 500
```

### Train All Patients

```bash
python main.py --patient all --epochs 500
```

### View All Options

```bash
python main.py --help
```

---

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--patient` | str | `0073` | Patient ID or `all` (trains sequentially) |
| `--epochs` | int | `500` | Maximum training epochs |
| `--batch-size` | int | `4096` | Training batch size |
| `--lr` | float | `1e-4` | Initial learning rate |
| `--patience` | int | `50` | Early stopping patience |
| `--grad-clip` | float | `1.0` | Gradient clipping (0 to disable) |
| `--arch` | str | `shared` | Architecture: `shared`, `multi`, or `kan` |
| `--hidden-dim` | int | `256` | Hidden layer dimension |
| `--num-blocks` | int | `4` | Number of ResNet blocks (or KAN layers) |
| `--collocation` | int | `2048` | Physics collocation points/batch |
| `--compute-wss` | flag | False | Compute WSS from gradients (instead of predicting) |
| `--kan-grid-size` | int | `5` | KAN: B-spline grid size (only for `--arch kan`) |
| `--kan-spline-order` | int | `3` | KAN: B-spline order (only for `--arch kan`) |

### Example Commands

```bash
# Train single patient with SharedTrunk (recommended)
python main.py --patient H-12 --epochs 1000 --arch shared

# Train ALL patients sequentially with all settings specified
python main.py --patient all --epochs 500 --batch-size 4096 --lr 1e-4 --patience 50 --grad-clip 1.0 --arch shared --hidden-dim 256 --num-blocks 4 --collocation 2048

# Higher capacity Multi-network architecture
python main.py --patient 0149 --epochs 500 --arch multi --hidden-dim 512

# KAN architecture (experimental - better accuracy with fewer parameters)
python main.py --patient 0156 --epochs 500 --arch kan --hidden-dim 64 --num-blocks 3 --kan-grid-size 5

# Compute WSS from velocity gradients (physics-based)
python main.py --patient D-10 --epochs 500 --compute-wss

# Disable gradient clipping
python main.py --patient 0156 --epochs 500 --grad-clip 0

# Large model with more capacity
python main.py --patient 0156 --epochs 1000 --hidden-dim 512 --num-blocks 6 --collocation 4096
```

---

## Dataset

### Available Patients

| Patient ID | Category | Vessels | Description |
|------------|----------|---------|-------------|
| H-09 | Healthy | RCA | Normal coronary artery |
| H-12 | Healthy | LCA | Normal coronary artery |
| D-10 | Diseased | LCA, RCA | Stenosed vessels |
| 0073 | Mixed | LCA, RCA | Native coronary |
| 0148 | SVG | G2 | Saphenous vein graft |
| 0149 | SVG | G1, G2, G3 | Saphenous vein grafts |
| 0150 | SVG | G3 | Saphenous vein graft |
| 0156 | SVG | G2, G3 | Saphenous vein grafts |
| ND2 | Unknown | LCA | Unclassified |

### Data Format

CFD data from ANSYS CFD-Post exports with columns:
- Coordinates: `X [m]`, `Y [m]`, `Z [m]`
- Velocity: `Velocity u/v/w [m s^-1]`
- Wall Shear: `Wall Shear [Pa]`, `Wall Shear X/Y/Z [Pa]`

Each patient has:
- **Wall files**: Surface points with WSS and zero velocity
- **Streamline files**: Interior points with non-zero velocity

---

## Architecture

### 1. SharedTrunkPINN (Recommended for Most Cases)

Single shared encoder with multiple output heads. More parameter-efficient (~858K params).

```
Input(x,y,z) → [Shared Trunk: ResNet Blocks] → Features
                                                   ↓
                    ┌────────┬────────┬────────┬────────┬────────┐
                    ↓        ↓        ↓        ↓        ↓
                  Head_u   Head_v   Head_w   Head_p   Head_wss
```

**Use when:** You want fast training with good accuracy and shared features across outputs.

### 2. MultiResNetPINN

Separate networks for each output. More parameters (~2.6M) but independent feature learning.

```
Input(x,y,z) → Net_u → u
            → Net_v → v
            → Net_w → w
            → Net_p → p
            → Net_wss → wss
```

**Use when:** Outputs have very different spatial patterns and memory is not constrained.

### 3. KANPINN (Experimental - Kolmogorov-Arnold Networks)

Replaces fixed activations with learnable B-spline functions on each edge. Better accuracy with fewer parameters.

```
Input(x,y,z) → [KAN Layer: Learnable B-spline φ(x) per edge] → ... → Outputs
```

**Key advantages:**
- Better accuracy with 50-70% fewer parameters
- Naturally smooth derivatives (crucial for PINNs)
- Interpretable learned activation functions
- 10-100x improvement on some scientific computing tasks

**Recommended settings:**
- `--hidden-dim 32-64` (much smaller than MLP!)
- `--num-blocks 2-4` (fewer layers needed)
- `--kan-grid-size 3-8` (controls expressivity)

**Use when:** You want maximum accuracy with minimal parameters, and can tolerate slower training per epoch.

**Reference:** Liu, Z., et al. (2024). KAN: Kolmogorov-Arnold Networks. arXiv:2404.19756

---

## Physics Constraints

### Navier-Stokes Momentum (Steady-State)

$$\rho(\mathbf{u} \cdot \nabla)\mathbf{u} = -\nabla p + \mu \nabla^2 \mathbf{u}$$

### Continuity (Incompressible)

$$\nabla \cdot \mathbf{u} = \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} + \frac{\partial w}{\partial z} = 0$$

### Wall Shear Stress

$$\tau_w = \mu \left| \frac{\partial u_{tangent}}{\partial n} \right|$$

### Coordinate Scaling

Since inputs are normalized, gradients require chain rule correction:

$$\frac{\partial u}{\partial x_{physical}} = \frac{\partial u}{\partial x_{scaled}} \cdot \frac{1}{x_{max} - x_{min}}$$

---

## Loss Function

The total loss combines data fitting and physics constraints:

$$\mathcal{L}_{total} = \lambda_{wss} \mathcal{L}_{wss} + \lambda_{vel} \mathcal{L}_{vel} + \lambda_{NS} \mathcal{L}_{NS} + \lambda_{cont} \mathcal{L}_{cont}$$

| Component | Weight | Description |
|-----------|--------|-------------|
| WSS Loss | 1.0 | MSE between predicted and CFD WSS |
| Velocity Loss | 0.1 | MSE for velocity components |
| Navier-Stokes Loss | 1.0 | Momentum equation residuals |
| Continuity Loss | 1.0 | Mass conservation residuals |

---

## Evaluation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| RMSE | Root Mean Squared Error (Pa) | < 0.5 Pa |
| MAE | Mean Absolute Error (Pa) | < 0.3 Pa |
| NRMSE | Normalized RMSE (unitless) | < 0.1 |
| R² | Coefficient of Determination | > 0.9 |
| Pearson | Correlation coefficient | > 0.95 |

---

## Project Structure

```
PINNS/
├── main.py                 # CLI entry point
├── README.md               # Documentation
├── .gitignore              # Git ignore rules
│
├── src/                    # Source code modules
│   ├── __init__.py
│   ├── config.py           # Paths, constants, patient definitions
│   ├── dataset.py          # Data loading and PyTorch Dataset
│   ├── model.py            # Neural network architectures
│   ├── physics.py          # Navier-Stokes and continuity equations
│   ├── train.py            # Training loop with early stopping
│   ├── evaluate.py         # Metrics computation
│   ├── plots.py            # Visualization functions
│   └── utils.py            # Helper functions
│
├── data/                   # CFD simulation data (not tracked)
│   └── PINNS/
│       ├── {patient} {vessel}.csv           # Wall surface data
│       └── {patient} {vessel} Streamlines.csv  # Interior velocity
│
├── models/                 # Trained model checkpoints
│   └── {patient_id}/
│       └── pinn_{patient_id}_best.pth
│
├── figures/                # Generated comparison plots
│   └── {patient_id}/
│       ├── {patient_id}_WSS_XY.png
│       ├── {patient_id}_WSS_XZ.png
│       ├── {patient_id}_WSS_YZ.png
│       ├── {patient_id}_vel_u_XY.png
│       ├── {patient_id}_vel_v_XY.png
│       ├── {patient_id}_vel_w_XY.png
│       └── {patient_id}_{vessel}_WSS_*.png  # Per-vessel plots
│
└── results/                # Evaluation metrics
    └── {patient_id}/
        ├── {patient_id}_results.txt         # Human-readable summary
        └── {patient_id}_history.json        # Training loss history
```

---

## Physical Constants

| Parameter | Value | Description |
|-----------|-------|-------------|
| ρ (rho) | 1060 kg/m³ | Blood density |
| μ (mu) | 0.0035 Pa·s | Blood dynamic viscosity |

---

## Troubleshooting

### CUDA Out of Memory

Reduce batch size:
```bash
python main.py --patient 0156 --batch-size 2048
```

### Physics Residuals Too Large

Ensure coordinate scaling is being applied. Check that `coord_scale` is passed to physics functions.

### Poor WSS Predictions

- Try more epochs: `--epochs 1000`
- Increase model capacity: `--hidden-dim 512 --num-blocks 6`
- Use physics-based WSS: `--compute-wss`

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{pinn_wss_2025,
  title={Physics-Informed Neural Networks for Wall Shear Stress Prediction in Coronary Artery Bypass Grafts},
  author={[Authors]},
  journal={[Journal]},
  year={2025}
}
```

---

## License

Research code for academic and non-commercial use.

---

**Last Updated**: November 2025
