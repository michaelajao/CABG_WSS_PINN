# Physics-Informed Neural Networks for Coronary Artery Wall Shear Stress Prediction

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.0-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository implements **Physics-Informed Neural Networks (PINNs)** for prediction of Wall Shear Stress (WSS) and velocity fields in coronary arteries and saphenous vein bypass grafts. The models learn from Computational Fluid Dynamics (CFD) simulation data while enforcing the incompressible Navier-Stokes equations as physics constraints.

## Overview

This codebase accompanies the paper:

> **Integrated CFD and Physics-Informed Neural Network Analysis of Hemodynamics in Healthy and Diseased Coronary Arteries and Saphenous Vein Grafts**

The PINN surrogate models achieve:
- **Mean NRMSE: 1.08%** compared to CFD ground truth
- **Mean R²: 0.973** across nine patient cases
- **Inference time: < 1 second** vs hours for traditional CFD

---

## Quick Start

### Train a Single Patient

```bash
python main.py train --patient H-12 --epochs 500 --verbose
```

### Train All Patients

```bash
python main.py train --patient all --epochs 500 --verbose
```

---

## Model Architecture

The **FourierPINN** architecture used in the paper:

| Component | Specification |
|-----------|---------------|
| Input | 3D coordinates (x, y, z) |
| Fourier Encoding | 64 frequencies, σ = 10.0 |
| Residual Blocks | 6 blocks, 48 hidden units each |
| Activation | SiLU (Sigmoid Linear Unit) |
| Output Heads | 5 outputs: u, v, w, p, τ_w |
| Total Parameters | ~34,000 |

---

## Training Configuration

The following hyperparameters were used for the published results:

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Initial Learning Rate | 2 × 10⁻⁴ |
| Weight Decay | 10⁻⁵ |
| Batch Size (Data) | 8,192 |
| Batch Size (Collocation) | 4,096 |
| LR Schedule | Cosine Annealing |
| Early Stopping Patience | 100 epochs |
| Gradient Clipping | 1.0 |

### Loss Function Weights

| Loss Term | Weight |
|-----------|--------|
| WSS Data (λ_wss) | 10.0 |
| Velocity Data (λ_vel) | 10.0 |
| Navier-Stokes (λ_NS) | 1.0 |
| Continuity (λ_cont) | 1.0 |
| WSS Physics (λ_τ) | 1.0 |

---

## Dataset

CFD simulation data exported from ANSYS CFD-Post in CSV format.

### Directory Structure
```
data/PINNS/
├── H-12 LCA.csv              # Wall surface data (WSS)
├── H-12 LCA Streamlines.csv  # Interior velocity field
├── H-12.csv                  # Aorta/full anatomy
└── ...
```

### Patient Registry

| Patient | Category | Vessels | Description |
|---------|----------|---------|-------------|
| H-12 (H4) | Healthy | LCA | Normal left coronary artery |
| H-09 | Healthy | RCA | Normal right coronary artery |
| 0073 (H1) | Healthy | LCA, RCA | Healthy from Vascular Model Repository |
| D-10 (D3) | Diseased | LCA, RCA | Stenosed coronary arteries |
| ND2 | Diseased | LCA | Diseased coronary artery |
| 0148 | SVG | G2 | Single saphenous vein graft |
| 0149 (BG2) | SVG | G1, G2, G3 | Multiple grafts |
| 0150 | SVG | G3 | Single graft |
| 0156 | SVG | G2, G3 | Multiple grafts |

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
| `--batch-size` | `8192` | Training batch size |
| `--lr` | `2e-4` | Initial learning rate |
| `--patience` | `100` | Early stopping patience |
| `--num-collocation-points` | `4096` | Physics collocation points per batch |
| `--grad-clip` | `1.0` | Gradient clipping norm |

### Model Architecture
| Argument | Default | Description |
|----------|---------|-------------|
| `--hidden-dim` | `48` | Hidden layer dimension |
| `--num-blocks` | `6` | Number of residual blocks |
| `--num-frequencies` | `64` | Fourier encoding frequencies |
| `--fourier-scale` | `10.0` | Fourier frequency scale (σ) |

### Other Options
| Argument | Default | Description |
|----------|---------|-------------|
| `--verbose` | `False` | Show progress bars |

---

## Physics Constraints

The model minimises a composite loss function:

```
L = λ_wss·L_wss + λ_vel·L_vel + λ_NS·L_NS + λ_cont·L_cont + λ_τ·L_τ
```

Where:
- **L_wss:** MSE between predicted and CFD wall shear stress
- **L_vel:** MSE between predicted and CFD velocity fields
- **L_NS:** Navier-Stokes momentum equation residual
- **L_cont:** Continuity equation residual (∇·u = 0)
- **L_τ:** WSS physics consistency (τ = μ·∂u_t/∂n)

### Physical Constants
| Property | Value |
|----------|-------|
| Blood Density (ρ) | 1050 kg/m³ |
| Dynamic Viscosity (μ) | 0.0035 Pa·s |

---

## Project Structure

```
PINNS/
├── main.py                  # CLI entry point
├── LICENSE                  # MIT License
├── README.md
│
├── src/                     # Source code
│   ├── config.py            # Configuration and patient registry
│   ├── dataset.py           # Data loading and preprocessing
│   ├── model.py             # FourierPINN architecture
│   ├── physics.py           # Navier-Stokes physics constraints
│   ├── train.py             # Training loop
│   ├── evaluate.py          # Evaluation metrics
│   ├── plots.py             # Visualisation utilities
│   └── utils.py             # Helper functions (EarlyStopping)
├── data/PINNS/              # CFD simulation data (CSV)
│
│
└── reports/                 # New training outputs
    ├── models/
    ├── figures/
    └── results/
```

---

## Hardware Requirements

The published results were generated on:

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA GeForce RTX 5060 Ti (16 GB VRAM) |
| CPU | AMD Ryzen 7 7700 (8 cores, 16 threads) |
| RAM | 32 GB |
| Framework | PyTorch 2.7.0 |
| CUDA | 12.8 |

### Training Time
- **Per patient:** 1.5–2.5 hours (350–500 epochs)
- **Inference:** < 1 second for full 3D WSS field

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{Rehman2025,
  title={Integrated CFD and Physics-Informed Neural Network Analysis of Hemodynamics 
         in Healthy and Diseased Coronary Arteries and Saphenous Vein Grafts},
  author={Rehman, M. Abaid Ur and Ekici, Özgür and Erdener, Şefik Evren and 
          Ajao-Olarinoye, Michael and Kuchumov, Alex G. and Jia, Fei},
  journal={[Journal Name]},
  year={2025},
  doi={[DOI]}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


<!-- ## Acknowledgements

The CFD simulation data were obtained from the [Vascular Model Repository](http://www.vascularmodel.org) and the [ASOCA dataset](https://www.kaggle.com/datasets/). We thank the creators of these open-source resources for making patient-specific vascular geometries available for research. -->
