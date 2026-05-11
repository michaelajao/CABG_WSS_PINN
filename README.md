# Physics-Informed Neural Networks for Coronary Artery Wall Shear Stress Prediction

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.0-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository implements **Physics-Informed Neural Networks (PINNs)** for prediction of Wall Shear Stress (WSS) and velocity fields in coronary arteries and saphenous vein bypass grafts. The models learn from Computational Fluid Dynamics (CFD) simulation data while enforcing the incompressible Navier-Stokes equations as physics constraints.

## Quick Start

### Train a Single Patient (Newtonian, default)

```bash
python main.py train --patient H4 --epochs 500 --verbose
```

### Train a Single Patient (Carreau-Yasuda)

```bash
python main.py train --patient H2 --rheology carreau_yasuda --epochs 500 --verbose
```

Carreau-Yasuda data is available for all twelve patients; `src.config.CY_AVAILABLE_LABELS`
enumerates the eligible labels and the runtime guard refuses to pair the CY flag with
any patient missing CY-CFD ground truth.

### Train All Patients

```bash
python main.py train --patient all --epochs 500 --verbose
```
---

## Dataset

CFD simulation data exported from ANSYS CFD-Post in CSV format, split by
rheology under `data/`.

### Patient Labels

Patients use the published paper labels: **H1..H4** (healthy), **BG1..BG5**
(saphenous vein grafts), **D1..D3** (diseased coronary arteries). Twelve in
total. The mapping from the on-disk CSV IDs (e.g. `0073`, `H12`, `D-10`) to
public labels lives in `src/config.py:PATIENT_DATA`.

### Directory Structure
```
data/
├── Newtonian/                 # 12 patients (H1-H4, BG1-BG5, D1-D3)
│   ├── H12 LCA.csv            # Wall surface (WSS field)
│   ├── H12 LCA Streamlines.csv  # Interior velocity field
│   ├── H12.csv                # Full-patient mesh
│   └── ...
└── Carreau/                   # Same 12 patients with CY-CFD ground truth
    ├── 0066 LCA.csv           # e.g. H2 wall surface (Carreau-Yasuda)
    ├── 0157 G1.csv            # e.g. BG5 graft 1 (Carreau-Yasuda)
    └── ...
```

Outputs are namespaced by rheology so Newtonian and Carreau-Yasuda runs on the
same patient never collide:

```
reports/
├── models/<rheology>/<patient>/pinn_<patient>_best.pth
├── results/<rheology>/<patient>/...
└── figures/<rheology>/<patient>/...
```

---

## Command-Line Arguments

### Patient Selection
| Argument | Default | Description |
|----------|---------|-------------|
| `--patient` | `H4` | Patient label(s) (H1..D3) or `all` |
| `--rheology` | `newtonian` | `newtonian` or `carreau_yasuda` (CY only valid for H2, BG5, D1) |
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

## Reproducing the Paper Results

All paper outputs are generated via two CLI modules under `src/`:

| Entry point | Purpose |
|-------------|---------|
| `python -m src.evaluate holdout` | Sweep all eligible patients under a 20% spatial holdout; writes `reports/metrics/holdout_summary_<rheology>.csv`. |
| `python -m src.evaluate sensitivity` | Sensitivity sweeps (loss weights, collocation density, seeds) on a representative patient. |
| `python -m src.plots` | Render the per-patient holdout summary figure **and** patch the corresponding LaTeX table (`tab:pinn_holdout` for Newtonian, `tab:pinn_holdout_cy` for Carreau–Yasuda) in `doc/CABG_Paper/main.tex`. |

```bash
python -m src.evaluate holdout --rheology newtonian --epochs 500
python -m src.evaluate holdout --rheology carreau_yasuda --epochs 500
python -m src.evaluate sensitivity --patient H4 --epochs-short 200 --epochs-full 200
python -m src.plots --rheology newtonian
python -m src.plots --rheology carreau_yasuda
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

The CFD simulation data were obtained from the [Vascular Model Repository](http://www.vascularmodel.org) and the ASOCA dataset. We thank the creators of these open-source resources for making patient-specific vascular geometries available for research.
