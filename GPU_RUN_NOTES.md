# GPU Run Notes — CABG WSS PINN on the HPC box

> Operational notes for re-running the PINN holdout / sensitivity / plots pipeline
> after moving off the RTX 5060 Ti (16 GB). Written 2026-05-16.

## 1. Hardware

`nvidia-smi` on this box:

| GPU | Model | VRAM | State at check |
|-----|-------|------|----------------|
| 0 | Quadro RTX 8000 | 48 GB | idle, ~19 MiB used, 0 % util |
| 1 | Quadro RTX 8000 | 48 GB | idle, ~127 MiB used, 0 % util |

Both cards are free. Each has **3× the VRAM** of the old 5060 Ti (16 GB) and there are
**two** of them, so independent jobs can run in parallel.

Architecture note: Quadro RTX 8000 is **Turing (sm_75)**. The
`torch.set_float32_matmul_precision('high')` / TF32 path in `src/train.py` is an
Ampere+ feature and is a **no-op** here — it neither speeds up nor changes results.
Automatic mixed precision (AMP) is **not** used and must not be added: the physics
loss needs double backward (2nd derivatives) and fp16 there is numerically unsafe.

## 2. Environment

Only one conda env has a CUDA-capable PyTorch:

| Env | torch | CUDA | Use |
|-----|-------|------|-----|
| `deep_tf` | 2.6.0+cu124 | ✅ True | **use this** |
| `tf` | 2.0.1 | ✅ True | fallback only |
| `pytorch` | 1.12.1 | ❌ CPU-only | do not use |
| `base` | — | no torch | do not use |

Canonical interpreter:

```bash
DT=/home/olarinoyem/miniconda3/envs/deep_tf/bin/python
```

GPU selection: the code uses the first visible CUDA device
(`torch.device('cuda' if torch.cuda.is_available() else 'cpu')`), so pin a card with
`CUDA_VISIBLE_DEVICES=0` or `=1` in the environment of each run.

## 3. Do the parameters need to change for the bigger GPU?

**Short answer: memory-wise, no. We bump batch size only for speed, gated by an A/B.**

- The model is tiny (~34,806 params). On the old 16 GB card the recorded
  `peak_gpu_mb ≈ 1931` (see `reports/results/newtonian/H1/timing.json`). The 48 GB
  cards have ~25× headroom the current recipe never touches — **nothing is forced to
  change for memory**.
- The real cost is wall-clock: H1 took **18,669 s ≈ 5.2 h** (4779 epochs @ ~3.9
  s/epoch). 12 patients × 2 rheologies + sensitivity is days.
- The strong H1 result (R² ≈ 0.98, NRMSE < 1 %) was produced by the **current recipe**.
  To make the other 11 patients reproduce that, the numerics that shape convergence are
  **kept fixed**:

  | Param | Value | Why kept |
  |-------|-------|----------|
  | collocation points | 4096 | paper-cited default; production setting |
  | learning rate | 2e-4 | tuned with the recipe that gave H1≈0.98 |
  | early-stop patience | 100 | governs convergence/stop |
  | hidden dim / blocks | 48 / 6 | architecture behind the H1 result |
  | LR schedule | cosine, T_max=epochs | unchanged |

- **Only `batch_size` is raised**, because it is the one knob that can cut epoch wall
  time on a big card without changing the loss math. `run_holdout_sweep` /
  `run_sensitivity_sweeps` do **not** expose `batch_size`, so the change is the
  **default in `src/train.py` (`def train_patient(... batch_size: int = ...)`)** — this
  propagates consistently to both holdout and sensitivity. Larger batches mean fewer
  optimizer steps per epoch under the same cosine-LR/patience, which *can* shift the
  converged metrics, so the change is gated by the A/B below.

### Batch-size A/B (H4, Newtonian, equal seed/epochs, scratch metrics dir)

Accept the **largest** batch where, vs the 8192 control:
`|ΔR2_holdout| ≤ 0.01` **and** `|ΔNRMSE_holdout| ≤ 0.002` (0.2 pp)
**and** `train_seconds` is lower. Otherwise fall back to 8192.

Run on H4 Newtonian, `--epochs 1500`, seed 0, 20 % holdout (2026-05-16):

| batch | train_seconds | s/epoch | NRMSE_holdout | R2_holdout | clean? | decision |
|-------|---------------|---------|---------------|------------|--------|----------|
| 8192 (control) | 3452.9 | 2.30 | **0.45 %** | **0.9740** | ✅ clean | **baseline — keep** |
| 16384 | 1827.8 | 1.22 | 0.81 % | 0.9161 | ✅ clean | **REJECTED** |
| 16384 (concurrent, discarded) | 1796.8 | 1.20 | 0.53 % | 0.9645 | ⚠️ checkpoint race | invalid — see note |

**Chosen `batch_size` = 8192** (unchanged; `src/train.py` reverted to the default).

**Why 16384 was rejected:** ~1.9× faster per epoch, but at the *same* epoch budget /
cosine-LR / patience it materially underfits — clean-vs-clean
ΔR²_holdout = **0.058** (tolerance 0.01) and ΔNRMSE = **0.36 pp** (tolerance 0.20 pp),
i.e. H4 holdout R² drops 0.974 → 0.916. Larger batch = fewer optimizer steps per epoch;
recovering quality would need more epochs or an LR rescale that would eat the speed gain.
Keeping 8192 preserves the recipe behind the H1 ≈ 0.98 result.

**Process caveat learned:** `--metrics-dir` only redirects the summary CSV. The
per-patient checkpoint `reports/models/<rheology>/<pid>/pinn_<pid>_best.pth` (reloaded
by `train_patient` right before eval) is **shared**, so two concurrent runs of the *same
patient* corrupt each other's eval. The first (concurrent) 16384 arm read the
better-converged 8192 checkpoint and reported a falsely good R² 0.965; the clean
sequential rerun gave the true 0.916. **Rule: never run the same patient on both GPUs at
once.** Different patients / different rheologies are fine (separate subtrees).

**Remaining speedup options (since the batch lever is out):** the only safe accelerator
is GPU-parallel *scheduling across different patients/rheologies* (Stage 2 already does
this). A bigger batch with a linearly-scaled LR and a larger epoch budget could be
revisited later as a dedicated experiment, but it is out of scope for this run to avoid
risking the H1-match requirement.

## 4. Make sure the results are saved

Artifacts written by the pipeline (all under `reports/`, which is git-tracked except
`*.pth`/`models/`):

| Artifact | Path | Producer |
|----------|------|----------|
| Holdout summary (12 rows) | `reports/metrics/holdout_summary_<rheology>.csv` + `.json` | `src.evaluate holdout` (flushed per patient — crash-safe) |
| Sensitivity metrics | `reports/metrics/sensitivity_{lossweight,collocation,seeds}_<rheology>_<patient>.csv` | `src.evaluate sensitivity` |
| Holdout per-patient timing/history | `reports/results/<rheology>/<pid>/{timing.json,<pid>_history.json}` | `train_patient` (holdout run) |
| Holdout WSS contour figures (paper) | `reports/figures/<rheology>/<pid>/full_patient/<pid>_full_patient_wss_{XY,XZ}.png` | `train_patient` → `plot_full_patient_wss` |
| **Sensitivity per-patient artifacts (isolated)** | `reports/{figures,results,models}/<rheology>/_sensitivity/<pid>/…` | sensitivity `train_patient` (`output_tag='_sensitivity'`) |
| Holdout summary figure | `doc/CABG_Paper/figures/pinn_holdout_summary[_<rheology>].png` (+ `.pdf`) | `src.plots` |
| Best model | `reports/models/<rheology>/<pid>/pinn_<pid>_best.pth` | `train_patient` (**gitignored**, regenerable) |

**Sensitivity no longer overwrites holdout figures.** `train_patient` gained an
`output_tag` arg ([src/train.py](src/train.py) signature + path block); the sensitivity
sweep passes `output_tag='_sensitivity'` ([src/evaluate.py](src/evaluate.py)
`_sensitivity_train_once`), so its ~15 short H4 re-trainings write to
`reports/figures/<rheology>/_sensitivity/H4/…` (and matching `results`/`models`) and the
Stage-1 holdout's `reports/figures/<rheology>/H4/…` + `timing.json` are **never touched**.
The aggregated sensitivity numbers still land in the separate
`reports/metrics/sensitivity_*.csv`. `output_tag=None` (holdout default) keeps the
original `<base>/<rheology>/<pid>/` layout byte-for-byte. Edit applied while the holdout
sweeps were running — safe, because a live Python process does not reload edited source.

`.pth`/`models/` are gitignored on purpose (large, regenerable). Per-rheology
regenerated `reports/metrics` + `reports/figures` are committed **directly to `main`**
(no branch) at the gate and when each sweep finishes.

**Plots gotcha:** `python -m src.plots --rheology X` with no flags renders the figure
then **hard-exits** trying to patch `doc/CABG_Paper/main.tex`, which does not exist in
this repo. Always pass `--no-update-table`. The wanted output is the **`.png`**
(`pinn_holdout_summary.png` @180 dpi); the `.pdf` is a harmless byproduct. The combined
`pinn_holdout_comparison.png` the paper cites has no entrypoint and is out of scope
unless explicitly requested.

## 5. Run order (gated)

```bash
cd /home/olarinoyem/Project/CABG_WSS_PINN
DT=/home/olarinoyem/miniconda3/envs/deep_tf/bin/python
mkdir -p logs

# --- Stage 1: Newtonian holdout, all 12, GPU 0 (GATE — verify before continuing) ---
nohup env CUDA_VISIBLE_DEVICES=0 $DT -m src.evaluate holdout \
  --rheology newtonian --epochs 3000 > logs/holdout_newtonian.log 2>&1 &

# Gate: reports/metrics/holdout_summary_newtonian.csv has 12 rows AND
#       R2_holdout ≈ 0.98 / NRMSE_holdout < 0.01 like H1. THEN continue.

# --- Stage 2: parallel across both GPUs (after the gate passes) ---
nohup env CUDA_VISIBLE_DEVICES=0 $DT -m src.evaluate holdout \
  --rheology carreau_yasuda --epochs 3000 > logs/holdout_cy.log 2>&1 &
nohup env CUDA_VISIBLE_DEVICES=1 $DT -m src.evaluate sensitivity \
  --patient H4 --rheology newtonian --sweeps all \
  --epochs-short 1000 --epochs-full 1000 > logs/sensitivity_h4.log 2>&1 &

# --- Plots (after the matching holdout CSV exists) ---
$DT -m src.plots --rheology newtonian      --no-update-table
$DT -m src.plots --rheology carreau_yasuda --no-update-table
```

Steps 1–3 are independent (order among them is preference); plots must come after the
matching holdout CSV exists.
