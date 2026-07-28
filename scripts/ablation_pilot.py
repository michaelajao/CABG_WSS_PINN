"""
Physics-contribution ablation pilot (Referee #5, Point 3-C).

Runs a 2x2 design on a single patient to test whether the physics loss
helps predictive accuracy, and whether that depends on the holdout regime:

    physics = {ON  (default LOSS_PRIORITY),
               OFF (navier_stokes = continuity = wss_physics = 0 -> data only)}
    holdout = {random    (uniform 20%%, dense within-geometry interpolation),
               clustered (one contiguous withheld ball -> spatial extrapolation)}

Expectation under Ref #5's hypothesis: physics ~irrelevant for the random
(interpolation) holdout, but ON < OFF (lower NRMSE) for the clustered
(extrapolation) holdout, which would substantiate the "physics-informed"
value proposition. A null result there instead supports softening the claim.

Isolated from authoritative artifacts via CABG_REPORTS_SUBDIR (set by the
caller) so reports/models/figures land under reports/<subdir>/.

Usage:
    CABG_REPORTS_SUBDIR=ablation_pilot CUDA_VISIBLE_DEVICES=1 \
        python -m scripts.ablation_pilot --patient D3 --rheology newtonian \
        --epochs 5000 --holdout-seed 0
"""
import argparse
import json
import time
from pathlib import Path

import src.config as cfg


def _extract(metrics: dict, key: str) -> dict:
    """Pull the flat metric fields from a train/holdout sub-dict."""
    m = metrics.get(key, {}) if isinstance(metrics, dict) else {}
    return {
        'NRMSE': m.get('NRMSE'),
        'RMSE': m.get('RMSE'),
        'R2': m.get('R2'),
        'pearson_r': m.get('pearson_r'),
        'n_points': m.get('n_points'),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--patient', default='D3')
    ap.add_argument('--rheology', default='newtonian',
                    choices=['newtonian', 'carreau_yasuda'])
    ap.add_argument('--epochs', type=int, default=5000)
    ap.add_argument('--holdout-fraction', type=float, default=0.20)
    ap.add_argument('--holdout-seed', type=int, default=0)
    ap.add_argument('--patience', type=int, default=100000,
                    help='high default => run the full epoch budget (no early stop)')
    args = ap.parse_args()

    # Select rheology before any training call (train.py reads cfg.RHEOLOGY).
    cfg.RHEOLOGY = args.rheology

    import src.train as train_mod
    from src.train import train_patient

    # Physics-OFF = zero the three physics priorities; physics-ON = defaults.
    default_priority = dict(train_mod.LOSS_PRIORITY)
    physics_off = dict(default_priority)
    physics_off['navier_stokes'] = 0.0
    physics_off['continuity'] = 0.0
    physics_off['wss_physics'] = 0.0

    configs = [
        ('random', 'on', default_priority),
        ('random', 'off', physics_off),
        ('clustered', 'on', default_priority),
        ('clustered', 'off', physics_off),
    ]

    out_dir = cfg.RESULTS_DIR / 'ablation'
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f'ablation_summary_{args.patient}_{args.rheology}.json'
    rows = []

    for mode, phys, priority in configs:
        tag = f'{mode}_{phys}'
        print('\n' + '#' * 80)
        print(f'# ABLATION {args.patient} [{args.rheology}] holdout={mode} physics={phys}')
        print('#' * 80)

        # Apply the priority override in place (train.py reads the module dict).
        train_mod.LOSS_PRIORITY.clear()
        train_mod.LOSS_PRIORITY.update(priority)

        t0 = time.time()
        _, results = train_patient(
            args.patient,
            epochs=args.epochs,
            patience=args.patience,
            holdout_fraction=args.holdout_fraction,
            holdout_seed=args.holdout_seed,
            holdout_mode=mode,
            output_tag=f'_ablation/{tag}',
            verbose=True,
        )
        wall = time.time() - t0

        metrics = results.get('metrics', {})
        row = {
            'patient': args.patient,
            'rheology': args.rheology,
            'holdout_mode': mode,
            'physics': phys,
            'epochs': args.epochs,
            'holdout_fraction': args.holdout_fraction,
            'holdout_seed': args.holdout_seed,
            'priorities': priority,
            'wall_seconds': wall,
            'holdout': _extract(metrics, 'holdout'),
            'train': _extract(metrics, 'train'),
        }
        rows.append(row)
        # Persist after each config so partial progress survives interruptions.
        with open(summary_path, 'w') as f:
            json.dump(rows, f, indent=2)
        h = row['holdout']
        print(f"\n[ABLATION RESULT] {tag}: holdout NRMSE={h['NRMSE']}, "
              f"R2={h['R2']}, r={h['pearson_r']}, n={h['n_points']}, "
              f"{wall/60:.1f} min")

    # Restore defaults (tidy; process exits anyway).
    train_mod.LOSS_PRIORITY.clear()
    train_mod.LOSS_PRIORITY.update(default_priority)

    print('\n' + '=' * 80)
    print(f'ABLATION COMPLETE -> {summary_path}')
    for r in rows:
        h = r['holdout']
        print(f"  {r['holdout_mode']:>9} / physics-{r['physics']:<3}  "
              f"NRMSE={h['NRMSE']:.4f}  R2={h['R2']:.4f}  r={h['pearson_r']:.4f}")
    print('=' * 80)


if __name__ == '__main__':
    main()
