# Physics-Informed Neural Networks for Coronary Artery WSS Prediction

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)

This repository implements **Physics-Informed Neural Networks (PINNs)** for predicting Wall Shear Stress (WSS) in coronary arteries and saphenous vein bypass grafts. The models learn from CFD simulation data while enforcing the incompressible Navier-Stokes equations as physics constraints.

**Paper:** *Computational Investigation of Blood Flow in Saphenous Vein Grafts and Coronary Arteries: CFD Analysis with Physics-Informed Neural Network Surrogate Modelling*

**Results:** NRMSE **0.61%** | R² **0.967**

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Dataset](#dataset)
4. [Architectures](#architectures)
5. [Training](#training)
6. [Output](#output)
7. [Project Structure](#project-structure)
8. [Citation](#citation)

---

## Installation

**Requirements:** NVIDIA GPU (8GB+ VRAM), CUDA 12.x, Python 3.10+

```bash
conda create -n dl_env python=3.10
conda activate dl_env
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install numpy pandas matplotlib scikit-learn scipy tqdm
```

---

## Quick Start

```bash
# Train single patient
python main.py --patient H-12 --epochs 5000 --arch fourier

# Train all patients
python main.py --patient all --epochs 5000 --batch-size 4096 --lr 2e-4 --arch fourier --hidden-dim 128 --num-blocks 8 --collocation 2048
```

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--patient` | `0073` | Patient ID or `all` |
| `--epochs` | `500` | Maximum training epochs |
| `--batch-size` | `4096` | Training batch size |
| `--lr` | `1e-4` | Initial learning rate |
| `--patience` | `50` | Early stopping patience |
| `--grad-clip` | `1.0` | Gradient clipping value |
| `--arch` | `vanilla` | `vanilla`, `fourier`, `multi`, `kan` |
| `--hidden-dim` | `256` | Hidden layer dimension |
| `--num-blocks` | `4` | Number of residual blocks |
| `--collocation` | `2048` | Physics collocation points per batch |

---

## Dataset

CFD simulation data from [Vascular Model Repository](http://www.vascularmodel.org) and [ASOCA Dataset](https://asoca.grand-challenge.org/), exported from ANSYS CFD-Post.

### Patients

| Patient | Category | Vessels | Points |
|---------|----------|---------|--------|
| H-12 | Healthy | LCA | ~171K |
| H-09 | Healthy | RCA | ~134K |
| D-10 | Diseased | LCA, RCA | ~153K |
| 0073 | SVG | LCA, RCA, Aorta | ~173K |
| 0148 | SVG | G2 | ~263K |
| 0149 | SVG | G1, G2, G3 | ~253K |
| 0150 | SVG | G3 | ~194K |
| 0156 | SVG | G2, G3 | ~340K |
| ND2 | Other | LCA | ~112K |

### File Structure

```
data/PINNS/
├── {patient}.csv                       # Aorta wall surface
├── {patient} {vessel}.csv              # Vessel wall surface
└── {patient} {vessel} Streamlines.csv  # Interior velocity
```

**Columns:** `X [m]`, `Y [m]`, `Z [m]`, `Velocity u/v/w [m s^-1]`, `Wall Shear [Pa]`

---

## Architectures

All architectures predict velocity (u, v, w), pressure (p), and WSS from spatial coordinates (x, y, z).

### FourierPINN (Recommended)

Random Fourier feature encoding overcomes spectral bias to capture high-frequency WSS variations.

```
Input (x, y, z)
    ↓
Fourier Features: γ(x) = [x, cos(2πBx), sin(2πBx)]
    ↓
┌─────────────────────────────────────┐
│  Residual Block × 8                 │
│  Linear → LayerNorm → SiLU → Linear │
│  + skip connection                  │
└─────────────────────────────────────┘
    ↓
Output Heads → u, v, w, p, WSS
```

| Setting | Value |
|---------|-------|
| Parameters | ~133K |
| Frequencies | 64 (scale=10.0) |
| Blocks | 8 |
| Hidden dim | 128 |
| Activation | SiLU |

---

### VanillaPINN

Standard MLP baseline with SiLU activations.

```
Input (x, y, z)
    ↓
[Linear → SiLU] × num_blocks
    ↓
Output Heads → u, v, w, p, WSS
```

| Setting | Value |
|---------|-------|
| Parameters | ~83K |
| Blocks | 4 |
| Hidden dim | 256 |
| Activation | SiLU |
| Init | Xavier normal |

---

### MultiResNetPINN

Separate encoder networks for each output variable.

```
Input (x, y, z)
    ↓
┌─────────────────────────────────────────────┐
│  5 Independent Networks                     │
│  Net_u: [Linear → Swish → ResBlock×4] → u  │
│  Net_v: [Linear → Swish → ResBlock×4] → v  │
│  Net_w: [Linear → Swish → ResBlock×4] → w  │
│  Net_p: [Linear → Swish → ResBlock×4] → p  │
│  Net_wss: [Linear → Swish → ResBlock×4] → WSS │
└─────────────────────────────────────────────┘

ResBlock: Linear → Swish → Linear → (+x)
```

| Setting | Value |
|---------|-------|
| Parameters | ~334K |
| Blocks | 4 per network |
| Hidden dim | 256 |
| Activation | Swish (learnable β) |
| Init | Kaiming normal |

---

### KANPINN (Experimental)

Kolmogorov-Arnold Networks with learnable B-spline activations.

```
Input (x, y, z)
    ↓
┌─────────────────────────────────────────┐
│  KAN Layer × num_layers                 │
│  φ_ij(x) = Σ_k c_ijk · B_k(x)          │
│  y_j = Σ_i φ_ij(x_i) + base_activation │
└─────────────────────────────────────────┘
    ↓
u, v, w, p, WSS
```

| Setting | Value |
|---------|-------|
| Parameters | ~140K |
| Layers | 3 |
| Hidden dim | 64 |
| Grid size | 5 |
| B-spline order | 3 (cubic) |

**Reference:** Liu et al. (2024). *KAN: Kolmogorov-Arnold Networks.* arXiv:2404.19756

---

## Training

### Physics Constraints

The network enforces incompressible Navier-Stokes via automatic differentiation:

**Momentum:** $\rho(\mathbf{u} \cdot \nabla)\mathbf{u} = -\nabla p + \mu \nabla^2 \mathbf{u}$

**Continuity:** $\nabla \cdot \mathbf{u} = 0$

### Loss Function

$$\mathcal{L} = \mathcal{L}_{wss} + 0.1 \cdot \mathcal{L}_{vel} + \mathcal{L}_{NS} + \mathcal{L}_{cont}$$

| Component | Description |
|-----------|-------------|
| WSS Loss | MSE vs CFD wall shear stress |
| Velocity Loss | MSE for u, v, w components |
| NS Loss | Momentum residuals at collocation points |
| Continuity Loss | Mass conservation residuals |

### Collocation Sampling

Physics constraints are enforced at points sampled from the mesh:

- **Interior points** (weight 1.0): Streamline data with velocity
- **Wall points** (weight 0.3): Surface data with WSS

### Configuration

| Setting | Value |
|---------|-------|
| Optimiser | AdamW (weight decay 1e-5) |
| Scheduler | Cosine annealing (2e-4 → 1e-6) |
| Gradient clip | 1.0 |
| Early stopping | 20 epochs patience |
| Blood density (ρ) | 1060 kg/m³ |
| Blood viscosity (μ) | 0.0035 Pa·s |

---

## Output

```
models/{patient}/pinn_{patient}_best.pth     # Checkpoint
figures/{patient}/{patient}_WSS_*.png        # WSS plots (XY, XZ, YZ)
figures/{patient}/{patient}_vel_*.png        # Velocity plots
results/{patient}/{patient}_results.txt      # Metrics
results/{patient}/{patient}_history.json     # Training history
```

---

## Project Structure

```
PINNS/
├── main.py           # CLI entry point
├── src/
│   ├── config.py     # Paths, patient definitions
│   ├── dataset.py    # Data loading, collocation sampling
│   ├── model.py      # Network architectures
│   ├── physics.py    # Navier-Stokes equations
│   ├── train.py      # Training loop
│   ├── evaluate.py   # Metrics
│   └── plots.py      # Visualisation
├── data/PINNS/       # CFD data
├── models/           # Checkpoints
├── figures/          # Plots
└── results/          # Metrics
```

---

## Citation

```bibtex
@article{rehman2025cabg_pinn,
  title={Computational Investigation of Blood Flow in Saphenous Vein 
         Grafts and Coronary Arteries: CFD Analysis with Physics-Informed 
         Neural Network Surrogate Modelling},
  author={Rehman, M. Abaid Ur and Ekici, Özgür and Erdener, Şefik Evren 
          and Ajao-Olarinoye, Michael and Kuchumov, Alex G.},
  year={2025}
}
```
<!-- 
## Authors

- **M. Abaid Ur Rehman** — NUST, Pakistan & Hacettepe University, Turkey
- **Özgür Ekici** — Hacettepe University, Turkey
- **Şefik Evren Erdener** — Hacettepe University, Turkey
- **Michael Ajao-Olarinoye** — Coventry University, UK
- **Alex G. Kuchumov** — Sirius University & Perm Polytechnic University, Russia -->

---

**Last Updated:** November 2025
