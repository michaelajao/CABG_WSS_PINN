# Physics-Informed Neural Networks for Coronary Artery Wall Shear Stress Prediction

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository implements **Physics-Informed Neural Networks (PINNs)** for real-time prediction of Wall Shear Stress (WSS) and velocity fields in coronary arteries and saphenous vein bypass grafts. The models learn from Computational Fluid Dynamics (CFD) simulation data while enforcing the incompressible Navier-Stokes equations as physics constraints.

**Paper:** *Computational Investigation of Blood Flow in Saphenous Vein Grafts and Coronary Arteries: CFD Analysis with Physics-Informed Neural Network Surrogate Modelling*

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Architectures](#model-architectures)
- [Physics Constraints](#physics-constraints)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [License](#license)

---

## Features

- **Multiple Architectures:** Support for Vanilla MLP, Fourier Features, Multi-ResNet, PirateNet, and Kolmogorov-Arnold Networks (KAN).
- **Physics-Informed:** Enforces Navier-Stokes momentum and continuity equations.
- **Adaptive Training:** Optional ReLoBRaLo algorithm for dynamic loss balancing.
- **Patient-Specific:** tailored training for different patient geometries (Healthy, Diseased, SVG).
- **Visualization:** Automated generation of WSS and velocity comparison plots.

---

## Installation

### Requirements
- NVIDIA GPU with 8GB+ VRAM (Recommended)
- Python 3.10+
- CUDA 12.x

### Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/username/pinn-coronary-wss.git
    cd pinn-coronary-wss
    ```

2.  **Create a virtual environment (Conda recommended):**
    ```bash
    conda create -n pinn python=3.10
    conda activate pinn
    ```

3.  **Install PyTorch:**
    Follow the instructions at [pytorch.org](https://pytorch.org/get-started/locally/) to install the version compatible with your CUDA setup.
    ```bash
    # Example for CUDA 12.1
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
    ```

4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: `open3d` is recommended for accurate surface normal estimation.*

---

## Quick Start

### Train a Single Patient
Train a model for patient `H-12` using the default Fourier architecture:

```bash
python main.py train --patient H-12 --epochs 500 --verbose
```

### Train Multiple Patients
Train on a specific list of patients:

```bash
python main.py train --patient 0073 0148 0149 --epochs 1000 --verbose
```

---

## Usage

The `main.py` script is the entry point for training.

### Command Syntax

```bash
python main.py train [OPTIONS]
```

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--patient` | `H-12` | Patient ID (e.g., `H-12`, `0073`) or `all`. Can accept multiple IDs. |
| `--seed` | `42` | Random seed for reproducibility. |
| `--epochs` | `500` | Maximum number of training epochs. |
| `--batch-size` | `4096` | Batch size for training. |
| `--lr` | `1e-4` | Initial learning rate. |
| `--patience` | `50` | Early stopping patience. |
| `--n-collocation` | `2048` | Number of physics collocation points per batch. |
| `--grad-clip` | `1.0` | Gradient clipping value (0 to disable). |
| `--arch` | `fourier` | Model architecture: `vanilla`, `fourier`, `pirate`, `multi`, `kan`. |
| `--hidden-dim` | `256` | Dimension of hidden layers. |
| `--num-blocks` | `4` | Number of residual blocks/layers. |
| `--num-frequencies` | `64` | Number of Fourier frequencies (for `fourier` arch). |
| `--fourier-scale` | `10.0` | Fourier frequency scale (for `fourier` arch). |
| `--kan-grid-size` | `5` | KAN grid size (for `kan` arch). |
| `--kan-spline-order` | `3` | KAN spline order (for `kan` arch). |
| `--adaptive-weights` | `False` | Enable ReLoBRaLo adaptive loss balancing. |
| `--verbose` | `False` | Show progress bars during training. |

### Advanced Example
Train on all patients *except* H-12 with specific hyperparameters and the PirateNet architecture:

```bash
python main.py train --patient 0073 0148 0149 0150 0156 D-10 H-09 ND2 --epochs 5000 --batch-size 4096 --n-collocation 4096 --patience 100 --lr 1e-4 --verbose --arch pirate --num-blocks 4 --hidden-dim 128 --num-frequencies 64
```

---

## Dataset

The project uses CFD simulation data exported from ANSYS CFD-Post.

### Directory Structure
Data should be placed in `data/PINNS/` with the following naming convention:
- **Wall Surface:** `{patient} {vessel}.csv` (contains WSS data)
- **Streamlines:** `{patient} {vessel} Streamlines.csv` (contains velocity field)
- **Aorta (Optional):** `{patient}.csv`

### Patient Registry
| Patient | Category | Vessels | Description |
|---------|----------|---------|-------------|
| **H-12** | Healthy | LCA | Normal left coronary artery |
| **H-09** | Healthy | RCA | Normal right coronary artery |
| **D-10** | Diseased | LCA, RCA | Stenosed coronary arteries |
| **0073** | Mixed | LCA, RCA | Mixed condition |
| **0148** | SVG | G2 | Single saphenous vein graft |
| **0149** | SVG | G1, G2, G3 | Multiple grafts |
| **0150** | SVG | G3 | Single graft |
| **0156** | SVG | G2, G3 | Multiple grafts |
| **ND2** | Unknown | LCA | Additional case |

---

## Model Architectures

The code (`src/model.py`) implements several architectures:

1.  **VanillaPINN:** Standard Multi-Layer Perceptron (MLP) with SiLU activations. Simple baseline.
2.  **FourierPINN (Recommended):** Uses Random Fourier Features mapping to overcome spectral bias, allowing the network to learn high-frequency WSS patterns effectively.
3.  **PirateNetPINN:** A modified architecture designed for better gradient flow and training stability.
4.  **MultiResNetPINN:** Uses separate ResNet encoders for each output variable (u, v, w, p, wss) with shared or independent trunks.
5.  **KANPINN (Experimental):** Kolmogorov-Arnold Networks using learnable B-spline activation functions on edges instead of fixed activation functions on nodes.

---

## Physics Constraints

The model minimizes a composite loss function:
$$ \mathcal{L} = w_{wss}\mathcal{L}_{wss} + w_{vel}\mathcal{L}_{vel} + w_{NS}\mathcal{L}_{NS} + w_{cont}\mathcal{L}_{cont} + w_{phys}\mathcal{L}_{phys} $$

Where:
- **$\mathcal{L}_{wss}$:** MSE loss against CFD Wall Shear Stress.
- **$\mathcal{L}_{vel}$:** MSE loss against CFD Velocity field.
- **$\mathcal{L}_{NS}$:** Residual of the non-dimensional Navier-Stokes momentum equations.
- **$\mathcal{L}_{cont}$:** Residual of the continuity equation (mass conservation).
- **$\mathcal{L}_{phys}$:** Consistency loss between predicted WSS and velocity gradients at the wall.

**Physical Constants:**
- Blood Density ($\rho$): $1060 \text{ kg/m}^3$
- Dynamic Viscosity ($\mu$): $0.0035 \text{ Pa}\cdot\text{s}$

---

## Project Structure

```
PINNS/
├── main.py              # CLI entry point
├── requirements.txt     # Python dependencies
├── LICENSE              # MIT License
├── README.md            # Project documentation
│
├── src/                 # Source code
│   ├── __init__.py
│   ├── config.py        # Configuration, paths, patient registry
│   ├── dataset.py       # Data loading, preprocessing, collocation sampling
│   ├── model.py         # Neural network architectures (Vanilla, Fourier, KAN, etc.)
│   ├── physics.py       # Navier-Stokes and continuity equation constraints
│   ├── train.py         # Training loop and loss calculation
│   ├── evaluate.py      # Evaluation metrics (RMSE, MAE, R2)
│   ├── plots.py         # Visualization utilities
│   └── utils.py         # Helper functions, ReLoBRaLo implementation
│
├── data/                # Input data directory
│   └── PINNS/           # CSV files from CFD
│
└── reports/             # Generated outputs
    ├── models/          # Saved model checkpoints (.pth)
    ├── figures/         # WSS and velocity plots
    └── results/         # Training history and metrics
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{rehman2025pinn_wss,
  title={Computational Investigation of Blood Flow in Saphenous Vein Grafts and Coronary Arteries: CFD Analysis with Physics-Informed Neural Network Surrogate Modelling},
  author={Rehman, M. Abaid Ur and Ekici, Özgür and Erdener, Şefik Evren and Ajao-Olarinoye, Michael and Kuchumov, Alex G.},
  journal={},
  year={2025}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
