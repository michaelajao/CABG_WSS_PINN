# Physics-Informed Neural Networks for Coronary Artery Wall Shear Stress Prediction

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository implements **Physics-Informed Neural Networks (PINNs)** for prediction of Wall Shear Stress (WSS) and velocity fields in coronary arteries and saphenous vein bypass grafts. The models learn from Computational Fluid Dynamics (CFD) simulation data while enforcing the incompressible Navier-Stokes equations as physics constraints.

---

## Features

- **Two Training Paradigms:**
  - **Method 1 (Data-Centric):** Dense CFD data with physics regularisation — achieves R² > 0.99
  - **Method 2 (TRUE PINN):** Sparse data with strong physics constraints — for limited measurements
- **Multiple Architectures:** Vanilla MLP, Fourier Features (recommended), PirateNet, Multi-ResNet, and KAN
- **Physics-Informed:** Enforces Navier-Stokes momentum and continuity equations
- **Adaptive Training:** Optional ReLoBRaLo algorithm for dynamic loss balancing
- **Patient-Specific:** Tailored training for different patient geometries (Healthy, Diseased, SVG)
- **GPU Optimised:** Pre-loads data to GPU memory for 5-10× faster training

---

## Installation

### Requirements
- NVIDIA GPU with 8GB+ VRAM (recommended)
- Python 3.10+
- CUDA 12.x

### Setup

```bash
# Clone the repository
git clone https://github.com/username/pinn-coronary-wss.git
cd pinn-coronary-wss

# Create virtual environment
conda create -n pinn python=3.10
conda activate pinn

# Install PyTorch (see pytorch.org for your CUDA version)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install dependencies
pip install -r requirements.txt
```

> **Note:** `open3d` is recommended for accurate surface normal estimation.

---

## Quick Start

### Method 1: Data-Centric PINN (Recommended for dense CFD data)

```bash
python main.py train --patient H-12 --epochs 500 --arch fourier --verbose
```

### Method 2: TRUE PINN (For sparse measurements)

```bash
python main.py train --patient H-12 --true-pinn --epochs 5000 --sample-every-n 100 --verbose
```

### Train All Patients

```bash
python main.py train --patient all --epochs 1000 --verbose
```

---

## Training Methods

### Method 1: Data-Centric PINN

Uses dense CFD data at all mesh points with physics as regularisation:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--arch` | `fourier` | FourierPINN architecture |
| `--epochs` | `500` | Training epochs |
| `--batch-size` | `4096` | Batch size |
| `--lr` | `1e-4` | Learning rate |
| Loss weights | WSS=1.0, Physics=0.1 | Data-dominated |

**Expected performance:** R² > 0.99, NRMSE < 5%

### Method 2: TRUE PINN

Uses sparse measurements with strong physics constraints:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--true-pinn` | flag | Enable TRUE PINN mode |
| `--sample-every-n` | `100-200` | Use every Nth point as data |
| `--epochs` | `5000` | More epochs needed |
| `--derive-wss` | flag | Derive WSS from velocity gradients |
| Loss weights | Physics=1.0, Data=20.0 | Physics-dominated |

**Use when:** Only sparse measurements available or testing physics extrapolation.

---

## Command-Line Arguments

### Patient Selection
| Argument | Default | Description |
|----------|---------|-------------|
| `--patient` | `H-12` | Patient ID(s) or `all` |
| `--seed` | `42` | Random seed |

### Training Hyperparameters
| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | `500` | Maximum training epochs |
| `--batch-size` | `4096` | Training batch size |
| `--lr` | `1e-4` | Initial learning rate |
| `--patience` | `50` | Early stopping patience |
| `--n-collocation` | `2048` | Physics collocation points per batch |
| `--grad-clip` | `1.0` | Gradient clipping (0 to disable) |

### Model Architecture
| Argument | Default | Description |
|----------|---------|-------------|
| `--arch` | `fourier` | Architecture: `vanilla`, `fourier`, `pirate`, `multi`, `kan` |
| `--hidden-dim` | `256` | Hidden layer dimension |
| `--num-blocks` | `4` | Number of residual blocks |
| `--num-frequencies` | `64` | Fourier frequencies |
| `--fourier-scale` | `10.0` | Frequency scale |

### TRUE PINN Mode
| Argument | Default | Description |
|----------|---------|-------------|
| `--true-pinn` | `False` | Enable TRUE PINN mode |
| `--sample-every-n` | `200` | Sample every Nth point for sparse data |
| `--lr-step-size` | `800` | LR decay interval (epochs) |
| `--lr-decay` | `0.5` | LR decay factor |
| `--derive-wss` | `False` | Derive WSS from velocity gradients |

### Other Options
| Argument | Default | Description |
|----------|---------|-------------|
| `--adaptive-weights` | `False` | Enable ReLoBRaLo adaptive loss balancing |
| `--verbose` | `False` | Show progress bars |

---

## Dataset

CFD simulation data exported from ANSYS CFD-Post in CSV format.

### Directory Structure
```
data/PINNS/
├── H-12 LCA.csv              # Wall surface data (WSS)
├── H-12 LCA Streamlines.csv  # Interior velocity field
├── H-12.csv                  # Full anatomy (optional)
└── ...
```

### Patient Registry
| Patient | Category | Vessels | Description |
|---------|----------|---------|-------------|
| H-12 | Healthy | LCA | Normal left coronary artery |
| H-09 | Healthy | RCA | Normal right coronary artery |
| D-10 | Diseased | LCA, RCA | Stenosed coronary arteries |
| 0073 | Mixed | LCA, RCA | Mixed condition |
| 0148 | SVG | G2 | Single saphenous vein graft |
| 0149 | SVG | G1, G2, G3 | Multiple grafts |
| 0150 | SVG | G3 | Single graft |
| 0156 | SVG | G2, G3 | Multiple grafts |
| ND2 | Unknown | LCA | Additional case |

---

## Model Architectures

| Architecture | Description | Recommended Use |
|--------------|-------------|-----------------|
| **FourierPINN** | Fourier feature encoding to overcome spectral bias | Default choice |
| VanillaPINN | Standard MLP with SiLU activations | Baseline comparison |
| PirateNetPINN | Adaptive residual connections | Deep networks |
| MultiResNetPINN | Separate encoders per output | Experimental |
| KANPINN | Learnable B-spline activations | High accuracy |

---

## Physics Constraints

The model minimises a composite loss:

**Method 1:** `L = L_wss + 0.1*L_vel + 0.1*L_NS + 0.1*L_cont`

**Method 2:** `L = L_NS + L_cont + 20*L_BC + 20*L_data`

Where:
- **L_wss / L_vel:** MSE against CFD data
- **L_NS:** Navier-Stokes momentum residual
- **L_cont:** Continuity equation residual
- **L_BC:** No-slip boundary condition (TRUE PINN only)

**Physical Constants:**
- Blood Density (ρ): 1060 kg/m³
- Dynamic Viscosity (μ): 0.0035 Pa·s

---

## Project Structure

```
PINNS/
├── main.py                  # CLI entry point
├── requirements.txt         # Python dependencies
├── LICENSE                  # MIT License
├── README.md
│
├── src/                     # Source code
│   ├── config.py            # Configuration, paths, patient registry
│   ├── dataset.py           # Data loading, GPU caching, collocation sampling
│   ├── model.py             # Neural network architectures
│   ├── physics.py           # Navier-Stokes and continuity constraints
│   ├── train.py             # Training loop (both methods)
│   ├── evaluate.py          # Evaluation metrics (RMSE, MAE, R²)
│   ├── plots.py             # Visualisation utilities
│   └── utils.py             # EarlyStopping, ReLoBRaLo
│
├── data/PINNS/              # CFD simulation data (CSV)
│
├── doc/pinn_methodology/    # LaTeX methodology documentation
│   └── pinn_methodology.tex
│
└── reports/                 # Generated outputs
    ├── models/              # Saved checkpoints (.pth)
    ├── figures/             # WSS and velocity plots
    └── results/             # Training metrics (JSON)
```

---

## Output

After training, results are saved to:

```
reports/
├── models/{patient_id}/pinn_{patient_id}_best.pth    # Best model checkpoint
├── figures/{patient_id}/                              # PNG visualisations
│   ├── wss_comparison_{vessel}.png
│   ├── loss_curves.png
│   └── velocity_field.png
└── results/{patient_id}/metrics.json                  # Evaluation metrics
```

---

## Documentation

Detailed methodology documentation is available in LaTeX format:

```
doc/pinn_methodology/pinn_methodology.tex
```

This includes full mathematical derivations, algorithm pseudocode, and architecture diagrams.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
