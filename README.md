# Physics-Informed Neural Networks for Coronary Artery Wall Shear Stress Prediction

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository implements Physics-Informed Neural Networks (PINNs) that predict
wall shear stress (WSS) and velocity fields in coronary arteries and saphenous
vein bypass grafts. Each model learns from patient-specific Computational Fluid
Dynamics (CFD) data while enforcing the incompressible Navier-Stokes equations
as soft physics constraints. It accompanies the manuscript listed under
[Citation](#citation).

## Requirements

- Python 3.10 or newer
- PyTorch 2.x (a CUDA build is strongly recommended for training)
- NumPy, pandas, matplotlib, scikit-learn, tqdm, Open3D

Install everything with:

```bash
pip install -r requirements.txt
```

For GPU training, install a CUDA-matched PyTorch build first using the selector
at [pytorch.org](https://pytorch.org), then run the command above for the rest.

## Quick Start

Every training run is launched through the `train` subcommand of `main.py`.

Train a single patient (Newtonian rheology, the default):

```bash
python main.py train --patient H4 --epochs 500 --verbose
```

Train a single patient with the Carreau-Yasuda rheology:

```bash
python main.py train --patient H2 --rheology carreau_yasuda --epochs 500 --verbose
```

Train every patient in one pass:

```bash
python main.py train --patient all --epochs 500 --verbose
```

The Carreau-Yasuda flag is only valid for patients that have Carreau-Yasuda CFD
ground truth. A runtime guard rejects the flag for any patient outside
`src.config.CY_AVAILABLE_LABELS`. See the [coverage note](#carreau-yasuda-coverage)
for the important caveat about which vessels that ground truth covers.

## Repository Layout

```
CABG_WSS_PINN/
├── main.py                 # CLI entry point: train one patient, several, or all
├── requirements.txt
├── src/
│   ├── config.py           # patient registry, constants, rheology + vessel-subset config
│   ├── dataset.py          # CFD CSV loading, surface normals, holdout split, collocation sampler
│   ├── model.py            # FourierPINN architecture
│   ├── physics.py          # Navier-Stokes, continuity and WSS residuals
│   ├── train.py            # training loop and early stopping
│   ├── evaluate.py         # holdout sweep, sensitivity sweeps, contour re-plotting
│   ├── plots.py            # per-patient figures and the holdout summary figure
│   └── utils.py            # metrics and the early-stopping helper
├── data/
│   ├── Newtonian/          # 12 patients, Newtonian CFD ground truth
│   └── Carreau/            # Carreau-Yasuda CFD ground truth (see coverage note)
└── reports/
    ├── metrics/            # authoritative holdout and sensitivity result CSV/JSON
    └── common_subset/
        ├── metrics/        # like-for-like (common-vessel) holdout summaries
        └── results/        # per-patient training histories and timing
```

Rendered figures (anything under `reports/figures/` or `reports/**/figures/`) are
regenerable from trained checkpoints and are deliberately not tracked. The
authoritative numbers behind the paper live in the small CSV/JSON files under
`reports/metrics/` and `reports/common_subset/`.

## Dataset

The CFD ground truth was exported from ANSYS CFD-Post as CSV, split by rheology
under `data/`.

### Patient Labels

Patients use the published paper labels: **H1..H4** (healthy), **BG1..BG5**
(saphenous vein grafts) and **D1..D3** (diseased coronary arteries), twelve in
total. The on-disk CSV IDs differ from the paper labels. The full mapping lives
in `src/config.py` (`PATIENT_DATA`) and, for convenience, in
`data/Model Names Detail.pdf`:

| On-disk ID | Paper label | On-disk ID | Paper label |
| --- | --- | --- | --- |
| 0073 | H1 | 0150 | BG3 |
| 0066 | H2 | 0156 | BG4 |
| H9 | H3 | 0157 | BG5 |
| H12 | H4 | D1 | D1 |
| 0148 | BG1 | D2 | D2 |
| 0149 | BG2 | D10 | D3 |

### Directory Structure

```
data/
├── Newtonian/                    # 12 patients (H1-H4, BG1-BG5, D1-D3)
│   ├── H12 LCA.csv               # wall surface (WSS field), e.g. H4 left coronary
│   ├── H12 LCA Streamlines.csv   # interior velocity field
│   ├── H12.csv                   # full-patient mesh
│   └── ...
└── Carreau/                      # Carreau-Yasuda CFD ground truth
    ├── 0066 LCA.csv              # e.g. H2 wall surface
    ├── 0157 G1.csv               # e.g. BG5 graft 1
    └── ...
```

The two `data/` files above 50 MB (`0156.csv`, `0157.csv`) still sit under
GitHub's 100 MB per-file limit, so a plain `git clone` brings everything needed
to retrain. If the clone is too large for your remote, untrack `data/`
(`git rm -r --cached data`) and transfer it separately.

### Carreau-Yasuda coverage

Carreau-Yasuda CFD ground truth exists for all twelve patients, so
`CY_AVAILABLE_LABELS` contains every label. The export is **complete**, covering
every vessel that the Newtonian set covers, only for **H1, H2, D1 and D3**. For
the other eight patients (**H3, H4, BG1-BG5, D2**) only a vessel subset is
available. `src.config.COMMON_SUBSET_NEWTONIAN_VESSELS` lists exactly which
vessels overlap. For a strictly like-for-like Newtonian vs Carreau-Yasuda
comparison, set `CABG_VESSEL_SUBSET=common` (see
[Environment variables](#environment-variables)) so the Newtonian training set
is trimmed to the same vessels the Carreau-Yasuda data covers.

## Command-Line Arguments

All flags below belong to the `train` subcommand
(`python main.py train ...`).

### Patient Selection

| Argument | Default | Description |
|----------|---------|-------------|
| `--patient` | `H4` | One or more patient labels (H1..D3), or `all` |
| `--rheology` | *(config)* | `newtonian` or `carreau_yasuda`. When omitted it falls back to `src.config.RHEOLOGY`, which ships as `newtonian` |
| `--seed` | `42` | Random seed for the global RNGs |

### Training Hyperparameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | `500` | Maximum training epochs |
| `--batch-size` | `8192` | Training batch size |
| `--lr` | `2e-4` | Initial learning rate |
| `--patience` | `100` | Early-stopping patience |
| `--num-collocation-points` | `4096` | Physics collocation points per batch |
| `--grad-clip` | `1.0` | Gradient-clipping norm (0 disables) |

### Model Architecture

| Argument | Default | Description |
|----------|---------|-------------|
| `--hidden-dim` | `48` | Hidden layer dimension |
| `--num-blocks` | `6` | Number of residual blocks |
| `--num-frequencies` | `64` | Fourier encoding frequencies |
| `--fourier-scale` | `10.0` | Fourier frequency scale (sigma) |

### Evaluation and Reproducibility

| Argument | Default | Description |
|----------|---------|-------------|
| `--holdout-fraction` | `0.20` | Fraction of mesh points withheld per patient for evaluation (0 disables the split) |
| `--holdout-seed` | `0` | Seed for the per-patient spatial holdout split |
| `--verbose` | off | Show progress bars |

### Environment variables

| Variable | Effect |
|----------|--------|
| `CABG_REPORTS_SUBDIR=<name>` | Route all run outputs (figures, models, results and the default holdout metrics) under `reports/<name>/` so an experiment never overwrites the authoritative artifacts |
| `CABG_VESSEL_SUBSET=common` | Trim each Newtonian patient to the vessels its Carreau-Yasuda data also covers, for a like-for-like comparison. When no explicit subdir is set, outputs auto-route to `reports/common_subset/` |

## Physics Constraints

The model minimises a composite loss built from five terms:

```
L = w_wss·L_wss + w_vel·L_vel + w_NS·L_NS + w_cont·L_cont + w_tau·L_tau
```

- **L_wss** MSE between predicted and CFD wall shear stress
- **L_vel** MSE between predicted and CFD velocity fields
- **L_NS** Navier-Stokes momentum residual
- **L_cont** continuity residual (∇·u = 0)
- **L_tau** WSS physics consistency (τ = μ·∂u_t/∂n)

The term weights are balanced adaptively during training (the WSS term is given
double priority), so they are not fixed hyperparameters. Each run records the
final weights it settled on in its `timing.json`.

### Physical Constants

| Property | Value |
|----------|-------|
| Blood density (ρ) | 1050 kg/m³ |
| Dynamic viscosity (μ) | 0.0035 Pa·s (Newtonian μ, and μ_∞ for Carreau-Yasuda) |

## Reproducing the Paper Results

All paper outputs are produced through two CLI modules under `src/`.

| Entry point | Purpose |
|-------------|---------|
| `python -m src.evaluate holdout` | Train every eligible patient under a per-patient spatial holdout and write `holdout_summary_<rheology>.csv/json` to the metrics dir |
| `python -m src.evaluate sensitivity` | Loss-weight, collocation-density and random-seed sweeps on the representative patient (H4), writing `sensitivity_*_H4.csv` to `reports/metrics/` |
| `python -m src.evaluate replot` | Re-render the per-patient WSS contour figures from saved checkpoints (no training) |
| `python -m src.plots` | Render the holdout summary figure. It also patches the LaTeX table in `doc/CABG_Paper/main.tex` when that file is present, and skips that step otherwise |

### Full-coverage per-patient holdout

Writes `reports/metrics/holdout_summary_<rheology>.csv/json`:

```bash
python -m src.evaluate holdout --rheology newtonian      --epochs 3000
python -m src.evaluate holdout --rheology carreau_yasuda --epochs 3000
```

### Like-for-like common-vessel-subset holdout

This is the strictly matched Newtonian vs Carreau-Yasuda comparison. The shipped
`reports/common_subset/` numbers were produced with 5000 epochs and global seed
1000. Setting `CABG_VESSEL_SUBSET=common` both trims the Newtonian vessels and
auto-routes the outputs to `reports/common_subset/`:

```bash
# Linux / macOS
CABG_VESSEL_SUBSET=common python -m src.evaluate holdout --rheology newtonian      --epochs 5000 --global-seed 1000
CABG_VESSEL_SUBSET=common python -m src.evaluate holdout --rheology carreau_yasuda --epochs 5000 --global-seed 1000
```

```powershell
# Windows PowerShell
$env:CABG_VESSEL_SUBSET = 'common'
python -m src.evaluate holdout --rheology newtonian      --epochs 5000 --global-seed 1000
python -m src.evaluate holdout --rheology carreau_yasuda --epochs 5000 --global-seed 1000
```

### Sensitivity sweeps

Writes `reports/metrics/sensitivity_{lossweight,collocation,seeds}_newtonian_H4.csv`:

```bash
python -m src.evaluate sensitivity --patient H4 --rheology newtonian --sweeps all --epochs-short 1000 --epochs-full 1000
```

### Figures

The per-patient contour figures and the holdout summary figure are regenerated
locally (they are not tracked). They land under `reports/figures/`:

```bash
python -m src.evaluate replot --rheology newtonian
python -m src.plots --rheology newtonian --no-update-table
```

`--no-update-table` renders the summary figure only. Without it, `src.plots`
additionally patches the paper's LaTeX table when `doc/CABG_Paper/main.tex`
exists, which it does not in this public repository.

## Citation

If you use this code or the CFD training data, please cite the associated
manuscript (under revision at *Physics of Fluids*, 2026; bibliographic details
will be finalised on acceptance):

```bibtex
@article{AbaidUrRehman2026_CABG_WSS_PINN,
  author  = {Abaid Ur Rehman, M. and Ekici, {\"O}. and Erdener, {\c{S}}. E.
             and Ajao-Olarinoye, M. and Kuchumov, A. G. and Jia, F.},
  title   = {Wall Shear Stress in Healthy and Diseased Coronary Arteries and
             Saphenous Vein Grafts via Physics-Informed Neural Network
             Surrogates},
  journal = {Physics of Fluids},
  year    = {2026},
  note    = {In revision}
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file
for details.

## Acknowledgements

The CFD simulation data were obtained from the
[Vascular Model Repository](http://www.vascularmodel.org) and the ASOCA dataset.
We thank the creators of these open-source resources for making patient-specific
vascular geometries available for research.
