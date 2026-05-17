#!/usr/bin/env python3
"""Stitch the preserved H1 row back into a resumed holdout sweep.

Context: the full 12-patient holdout sweep was interrupted (relaunched
disconnect-proof in tmux). H1 was already finished and its summary row was
snapshotted to ``reports/metrics/_h1_keep/``. The resumed run trains the
remaining 11 patients into ``reports/metrics/_resume/``. This script merges
H1 + the resumed 11 into the canonical ``reports/metrics/holdout_summary_<rheology>.csv``
(+ ``.json``) in the registry patient order, so ``python -m src.plots`` and the
paper see a single coherent 12-row table.

Usage:  python tools/stitch_holdout.py <newtonian|carreau_yasuda>
"""
import csv
import json
import sys
from pathlib import Path

# Running as `python tools/stitch_holdout.py` puts tools/ on sys.path[0], not
# the repo root, so `import src` fails. Add the repo root explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from src.config import PATIENT_DATA
    _ORDER = list(PATIENT_DATA.keys())
except Exception:
    # Fallback: canonical paper order, so a stitch never fails on import.
    _ORDER = ['H1', 'H2', 'H3', 'H4', 'BG1', 'BG2', 'BG3', 'BG4', 'BG5',
              'D1', 'D2', 'D3']

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "reports" / "metrics"


def _load_json(p: Path) -> list[dict]:
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text())
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in ("newtonian", "carreau_yasuda"):
        print("usage: stitch_holdout.py <newtonian|carreau_yasuda>")
        return 2
    rheo = sys.argv[1]

    # Sources: the preserved H1 snapshot + every resume part (the sweep can be
    # split across GPUs/restarts, each writing its own _resume* dir). Glob is
    # rheology-scoped via the filename so newtonian/CY never cross-contaminate.
    sources = [METRICS / "_h1_keep" / f"holdout_summary_{rheo}.json"]
    sources += sorted(METRICS.glob(f"_resume*/holdout_summary_{rheo}.json"))

    by_pid: dict[str, dict] = {}
    for src in sources:
        for r in _load_json(src):          # later parts win on any dup
            pid = r.get("patient_id")
            if pid:
                by_pid[pid] = r
    print(f"[stitch] {rheo}: read {len(by_pid)} unique patients from "
          f"{len(sources)} source(s)")

    order = _ORDER
    merged = [by_pid[p] for p in order if p in by_pid]
    merged += [r for pid, r in by_pid.items() if pid not in order]

    if not merged:
        print(f"[stitch] nothing to write for {rheo} (no rows found)")
        return 1

    out_csv = METRICS / f"holdout_summary_{rheo}.csv"
    out_json = METRICS / f"holdout_summary_{rheo}.json"
    fieldnames = sorted({k for r in merged for k in r})
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(merged)
    out_json.write_text(json.dumps(merged, indent=2))
    pids = ", ".join(r.get("patient_id", "?") for r in merged)
    print(f"[stitch] wrote {len(merged)} rows -> {out_csv} ({pids})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
