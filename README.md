# Physics-Informed Neural Networks for Coronary Artery Wall Shear Stress Prediction

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository implements **Physics-Informed Neural Networks (PINNs)** for real-time prediction of Wall Shear Stress (WSS) in coronary arteries and saphenous vein bypass grafts. The models learn from CFD simulation data while enforcing the incompressible Navier-Stokes equations as physics constraints.

**Paper:** *Computational Investigation of Blood Flow in Saphenous Vein Grafts and Coronary Arteries: CFD Analysis with Physics-Informed Neural Network Surrogate Modelling*

---

## Key Results

| Metric | Value |
|--------|-------|
| **NRMSE** | 0.61% |
| **R²** | 0.967 |
| **Speedup** | ~10,000× vs CFD |

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Dataset](#dataset)
- [Model Architectures](#model-architectures)
- [Training](#training)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [License](#license)

---

## Installation

### Requirements
- NVIDIA GPU with 8GB+ VRAM
- CUDA 12.x
- Python 3.10+

### Setup

```bash
# Clone repository
git clone https://github.com/username/pinn-coronary-wss.git
cd pinn-coronary-wss

# Create conda environment
conda create -n pinn python=3.10
conda activate pinn

# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

### Train a Single Patient

```bash
python main.py --patient H-12 --epochs 500 --arch fourier
```

### Train All Patients

```bash
python main.py --patient all --epochs 500 --batch-size 4096 --lr 1e-4
```

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--patient` | `0073` | Patient ID or `all` |
| `--epochs` | `500` | Maximum training epochs |
| `--batch-size` | `4096` | Training batch size |
| `--lr` | `1e-4` | Initial learning rate |
| `--patience` | `50` | Early stopping patience |
| `--arch` | `vanilla` | Architecture: `vanilla`, `fourier`, `multi`, `kan` |
| `--hidden-dim` | `256` | Hidden layer dimension |
| `--num-blocks` | `4` | Number of residual blocks |
| `--collocation` | `2048` | Physics collocation points per batch |

---

## Dataset

CFD simulation data from the [Vascular Model Repository](http://www.vascularmodel.org) and [ASOCA Dataset](https://asoca.grand-challenge.org/), exported from ANSYS CFD-Post.

### Patient Categories

| Patient | Category | Vessels | Description |
|---------|----------|---------|-------------|
| H-12 | Healthy | LCA | Normal left coronary artery |
| H-09 | Healthy | RCA | Normal right coronary artery |
| D-10 | Diseased | LCA, RCA | Stenosed coronary arteries |
| 0073 | SVG | LCA, RCA, Aorta | Saphenous vein graft |
| 0148 | SVG | G2 | Single graft |
| 0149 | SVG | G1, G2, G3 | Multiple grafts |
| 0150 | SVG | G3 | Single graft |
| 0156 | SVG | G2, G3 | Multiple grafts |
| ND2 | Other | LCA | Additional case |

### Data Format

```
data/PINNS/
├── {patient}.csv                       # Aorta wall surface
├── {patient} {vessel}.csv              # Vessel wall surface (with WSS)
└── {patient} {vessel} Streamlines.csv  # Interior velocity field
```

**CSV Columns:** `X [m]`, `Y [m]`, `Z [m]`, `Velocity u/v/w [m s^-1]`, `Wall Shear [Pa]`

---

## Model Architectures

All architectures predict velocity (u, v, w), pressure (p), and WSS from spatial coordinates (x, y, z).

### FourierPINN (Recommended)

Random Fourier feature encoding to overcome spectral bias for high-frequency WSS gradients.

```
Input (x, y, z) → Fourier Features → [ResBlock × N] → Output Heads
```

### VanillaPINN

Standard MLP baseline with SiLU activations.

```
Input (x, y, z) → [Linear → SiLU] × N → Output Heads
```

### MultiResNetPINN

Separate encoder networks for each output variable with skip connections.

### KANPINN (Experimental)

Kolmogorov-Arnold Networks with learnable B-spline activation functions.

---

## Training

### Physics Constraints

The network enforces incompressible Navier-Stokes equations via automatic differentiation:

**Momentum:** ρ(u·∇)u = -∇p + μ∇²u

**Continuity:** ∇·u = 0

### Loss Function

```
L = L_wss + 0.1·L_vel + L_NS + L_cont
```

| Component | Description |
|-----------|-------------|
| `L_wss` | MSE vs CFD wall shear stress |
| `L_vel` | MSE for velocity components |
| `L_NS` | Navier-Stokes momentum residuals |
| `L_cont` | Mass conservation (continuity) |

### Physical Constants

| Parameter | Value | Description |
|-----------|-------|-------------|
| ρ | 1060 kg/m³ | Blood density |
| μ | 0.0035 Pa·s | Blood dynamic viscosity |

---

## Project Structure

```
PINNS/
├── main.py              # CLI entry point
├── requirements.txt     # Python dependencies
├── LICENSE              # MIT License
├── README.md            # This file
│
├── src/                 # Source code
│   ├── __init__.py      # Package exports
│   ├── config.py        # Paths, constants, patient registry
│   ├── dataset.py       # Data loading, collocation sampling
│   ├── model.py         # Neural network architectures
│   ├── physics.py       # Navier-Stokes equations
│   ├── train.py         # Training pipeline
│   ├── evaluate.py      # Metrics computation
│   ├── plots.py         # Visualization functions
│   └── utils.py         # Helper functions
│
├── data/PINNS/          # CFD simulation data
├── models/              # Trained model checkpoints
├── figures/             # Generated plots
└── results/             # Evaluation metrics
```

### Output Files

```
models/{patient}/pinn_{patient}_best.pth    # Model checkpoint
figures/{patient}/{patient}_WSS_*.png       # WSS comparison plots
figures/{patient}/{patient}_vel_*.png       # Velocity comparison plots
results/{patient}/{patient}_results.txt     # Evaluation metrics
results/{patient}/{patient}_history.json    # Training history
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{rehman2025pinn_wss,
  title={Computational Investigation of Blood Flow in Saphenous Vein 
         Grafts and Coronary Arteries: CFD Analysis with Physics-Informed 
         Neural Network Surrogate Modelling},
  author={Rehman, M. Abaid Ur and Ekici, Özgür and Erdener, Şefik Evren 
          and Ajao-Olarinoye, Michael and Kuchumov, Alex G.},
  journal={},
  year={2025}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- CFD data from [Vascular Model Repository](http://www.vascularmodel.org)
- Coronary artery models from [ASOCA Grand Challenge](https://asoca.grand-challenge.org/)
