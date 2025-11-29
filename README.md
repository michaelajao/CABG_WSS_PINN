# Physics-Informed Neural Networks for Coronary Artery WSS Prediction

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)

## Overview

This repository implements **Physics-Informed Neural Networks (PINNs)** for predicting Wall Shear Stress (WSS) in coronary arteries and saphenous vein bypass grafts. The model learns from CFD simulation data while enforcing the incompressible Navier-Stokes equations as physics constraints.

### Research Context

Wall shear stress plays a crucial role in graft success and disease progression, particularly in regions prone to atherosclerosis and stenosis. This work extends CFD-based WSS analysis of coronary artery bypass grafts by introducing a data-driven surrogate model that:

- Learns WSS patterns from CFD simulations on patient-specific geometries
- Enforces physical constraints (Navier-Stokes, continuity) during training
- Enables rapid WSS prediction on new geometries without full CFD re-simulation
- Supports analysis of both Newtonian and non-Newtonian blood flow models

The dataset includes models from the **Vascular Model Repository** and **ASOCA** open-source datasets, covering healthy coronary arteries, diseased (stenosed) vessels, and saphenous vein grafts from CABG surgery.

### Key Features

- **Physics-Informed Learning**: Incorporates Navier-Stokes momentum and continuity equations
- **Multi-Output Architecture**: Predicts velocity (u, v, w), pressure (p), and WSS simultaneously
- **Patient-Specific Models**: Per-patient training with organized output structure
- **Multiple Architectures**: VanillaPINN, FourierPINN, MultiResNetPINN, KANPINN
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
| `--arch` | str | `vanilla` | Architecture: `vanilla`, `fourier`, `multi`, or `kan` |
| `--hidden-dim` | int | `256` | Hidden layer dimension |
| `--num-blocks` | int | `4` | Number of ResNet blocks (or KAN layers) |
| `--collocation` | int | `2048` | Physics collocation points/batch |
| `--compute-wss` | flag | False | Compute WSS from gradients (instead of predicting) |
| `--kan-grid-size` | int | `5` | KAN: B-spline grid size (only for `--arch kan`) |
| `--kan-spline-order` | int | `3` | KAN: B-spline order (only for `--arch kan`) |

### Example Commands

```bash
# Train single patient with Vanilla PINN (baseline)
python main.py --patient H-12 --epochs 1000 --arch vanilla

# Train with FourierPINN (recommended for best accuracy)
python main.py --patient 0073 --epochs 1000 --arch fourier --hidden-dim 128 --num-blocks 6

# Train ALL patients sequentially
python main.py --patient all --epochs 500 --batch-size 4096 --lr 1e-4 --patience 20 --arch fourier

# Higher capacity Multi-network architecture
python main.py --patient 0149 --epochs 500 --arch multi --hidden-dim 512

# KAN architecture (experimental)
python main.py --patient 0156 --epochs 500 --arch kan --hidden-dim 64 --num-blocks 3 --kan-grid-size 5

# Large model with more capacity
python main.py --patient 0156 --epochs 1000 --hidden-dim 512 --num-blocks 6 --collocation 4096
```

---

## Dataset

### Patient Categories

The dataset comprises 9 patient models across three clinical categories:

| Category | Patients | Description |
|----------|----------|-------------|
| **Healthy** | H-09, H-12 | Normal coronary arteries without disease |
| **Diseased** | D-10 | Stenosed coronary arteries |
| **SVG (CABG)** | 0073, 0148, 0149, 0150, 0156 | Saphenous vein grafts from bypass surgery |
| **Other** | ND2 | Additional coronary model |

### Available Patients

| Patient ID | Category | Vessels | Wall Points | WSS Range (Pa) |
|------------|----------|---------|-------------|----------------|
| H-09 | Healthy | Aorta, RCA | 145,432 | 0.00 - 98.77 |
| H-12 | Healthy | Aorta, LCA | 254,492 | 0.00 - 94.22 |
| D-10 | Diseased | Aorta, LCA, RCA | 267,767 | 0.01 - 139.93 |
| 0073 | SVG | Aorta, LCA, RCA | 321,448 | 0.00 - 13.44 |
| 0148 | SVG | Aorta, G2 | 272,684 | 0.00 - 31.44 |
| 0149 | SVG | Aorta, G1, G2, G3 | 312,911 | 0.00 - 66.77 |
| 0150 | SVG | Aorta, G3 | 213,961 | 0.00 - 167.56 |
| 0156 | SVG | Aorta, G2, G3 | 387,696 | 0.00 - 34.67 |
| ND2 | Other | Aorta, LCA | 149,972 | 0.01 - 1015.09 |

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

### 1. VanillaPINN (Baseline)

Standard feedforward neural network with SiLU (Swish) activation functions and residual connections.

```
Input(x,y,z) → [ResBlock: Linear → SiLU → Linear + Skip] × num_blocks → [Output Heads] → u, v, w, p, wss
```

**Parameters:** ~83K (default settings)

### 2. FourierPINN (Recommended)

Extends VanillaPINN with Fourier feature embedding for better high-frequency pattern capture.

```
Input(x,y,z) → FourierFeatures(64 frequencies) → [ResBlocks] → [Output Heads] → u, v, w, p, wss
```

**Parameters:** ~100K (default settings)

**Use when:** WSS has sharp spatial variations (stenoses, bifurcations).

### 3. MultiResNetPINN

Separate encoder networks for each output. More parameters but independent feature learning.

```
Input(x,y,z) → Net_u → u
            → Net_v → v
            → Net_w → w
            → Net_p → p
            → Net_wss → wss
```

**Parameters:** ~334K

**Use when:** Outputs have very different spatial patterns and memory is not constrained.

### 4. KANPINN (Experimental)

Kolmogorov-Arnold Networks with learnable B-spline activation functions.

```
Input(x,y,z) → [KAN Layer: Learnable B-spline φ(x) per edge] → ... → Outputs
```

**Parameters:** ~140K

**Key advantages:**
- Better accuracy with fewer parameters on some problems
- Naturally smooth derivatives (beneficial for physics gradients)
- Interpretable learned activation functions

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
| NRMSE | Normalized RMSE (%) | < 5% |
| R² | Coefficient of Determination | > 0.9 |
| MAE | Mean Absolute Error (Pa) | < 0.5 Pa |


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
