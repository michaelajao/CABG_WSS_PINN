# Physics-contribution ablation pilot (D3, Newtonian)

Source of the four numbers in the manuscript's `tab:pinn_physics_ablation`.
Produced by [`scripts/ablation_pilot.py`](../../scripts/ablation_pilot.py), which
runs a 2x2 design (physics on/off x random/clustered holdout) on one patient.

## Command

```bash
CABG_REPORTS_SUBDIR=ablation_pilot CUDA_VISIBLE_DEVICES=1 \
    python -m scripts.ablation_pilot --patient D3 --rheology newtonian \
    --epochs 5000 --holdout-seed 0
```

`--patience` keeps its default of 100000, so all four configurations run the
full 5000-epoch budget rather than early-stopping. Hardware was a Quadro
RTX 8000; wall time was ~3.3 h per configuration, ~13.1 h total.

## Results

Holdout WSS NRMSE, from `results/ablation/ablation_summary_D3_newtonian.json`:

| holdout mode | physics | NRMSE | R2 | n points |
|---|---|---|---|---|
| random | on | 1.48% | 0.929 | 25,233 |
| random | off | 0.41% | 0.995 | 25,233 |
| clustered | on | 15.79% | -1.058 | 22,703 |
| clustered | off | 21.20% | -2.708 | 22,703 |

Under the dense random holdout the data-only configuration attains an NRMSE
3.6 times lower than the physics-informed configuration. Under the clustered
holdout the ordering reverses, but both configurations return negative R2,
so neither reconstructs the withheld region.

## Difference between 1.48% here and 0.78% for D3 in `reports/common_subset/`

These are the same geometry and the same split — both evaluate 25,233 wall
points against 100,986 training points, and both use the production loss
priorities (`wss` 2.0, everything else 1.0). The budget is not the difference
either: this run completed 5000 epochs, the common-subset run early-stopped at
4320, and both restore the best-loss checkpoint before evaluating.

What differs is the adaptive loss weighting. `train_patient` sets each weight
inversely to that term's gradient norm on the first batch, so the weights
depend on the initialization. Comparing the two `timing.json` files:

| term | common_subset | ablation pilot |
|---|---|---|
| wss | 27,808 | 7,279 |
| velocity | 15,756 | 10,755 |
| wss_physics | 14,024 | 22,216 |

The production run weighted the WSS data term at 1.77 times its velocity term;
this run weighted it at 0.68 times, with approximately 1.6 times greater
weight on `wss_physics`. The resulting WSS error is 1.9 times higher. This is
consistent with the 1.67 +/- 0.46% spread reported for the seed sweep and
indicates that the gradient-norm weight initialization is sensitive to the
random seed.

## Contents

Committed: `results/` (summary, per-config `timing.json` and `D3_history.json`),
`run.log.gz`. The per-config results sit under `_ablation/` and are force-added
past the `reports/**/_*/` ignore rule.

Not committed: `figures/` (~40 MB, reproduced by re-running the driver) and
`models/` (this repository tracks no `.pth` checkpoints anywhere). The
20-epoch `reports/ablation_smoke/` tree, which verifies the driver end to
end, is ignored.
