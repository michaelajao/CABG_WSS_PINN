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

The command above has no `--seed`, because the flag did not exist when this run
was made. The driver went straight to `train_patient` and so never reached the
global seeding in `main.py`, which means this run started from an unseeded
initialization and is **not bit-reproducible**. `scripts/ablation_pilot.py` now
takes an optional `--seed`, reapplied before each of the four configurations so
they share one initialization and the physics on/off contrast is paired. Re-run
with `--seed 1000` to match the production surrogates. Doing so will not
reproduce the numbers below, for the reason set out in the next section.

## Results

Holdout WSS NRMSE, from `results/ablation/ablation_summary_D3_newtonian.json`:

| holdout mode | physics | NRMSE | R2 | n points |
|---|---|---|---|---|
| random | on | 1.48% | 0.929 | 25,233 |
| random | off | 0.41% | 0.995 | 25,233 |
| clustered | on | 15.79% | -1.058 | 22,703 |
| clustered | off | 21.20% | -2.708 | 22,703 |

Physics-off wins the dense random holdout by 3.6x; physics-on wins the
clustered (spatial extrapolation) holdout, but both are far past the point
where R2 goes negative, so neither is usable there.

## Why 1.48% here and 0.78% for D3 in `reports/common_subset/`

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

The production run happened to weight the WSS data term 1.77x its velocity
term; this run weighted it 0.68x, and put ~1.6x more pull on `wss_physics`.
A WSS error 1.9x higher is the expected consequence. This is the same
run-to-run spread as the 1.67 +/- 0.46% seed sweep, and it means the
gradient-norm weight initialization is seed-sensitive enough to matter — worth
stating in the manuscript rather than reconciling away.

That is now done. The sensitivity subsection of the manuscript states that
run-to-run variability persists at the full budget, gives this D3 pair as the
instance, and tells the reader that the per-patient values in the holdout
tables are single draws under a fixed seed rather than converged optima. The
ablation discussion reconciles 1.48% against 0.78% in the same terms, and notes
that the variability does not explain the physics-off result, since 0.41% falls
outside the range the two physics-on runs span.

## What is and isn't committed here

Committed: `results/` (summary, per-config `timing.json` and `D3_history.json`),
`run.log.gz`. The per-config results sit under `_ablation/` and are force-added
past the `reports/**/_*/` ignore rule.

Not committed: `figures/` (~40 MB, reproduced by re-running) and `models/`
(this repository tracks no `.pth` checkpoints anywhere). The 20-epoch
`reports/ablation_smoke/` tree was only a driver sanity check and is ignored.
