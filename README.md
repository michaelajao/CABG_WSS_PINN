# Physics-Informed Neural Networks for Coronary Artery Wall Shear Stress Prediction

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.0-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository implements **Physics-Informed Neural Networks (PINNs)** for prediction of Wall Shear Stress (WSS) and velocity fields in coronary arteries and saphenous vein bypass grafts. The models learn from Computational Fluid Dynamics (CFD) simulation data while enforcing the incompressible Navier-Stokes equations as physics constraints.

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
