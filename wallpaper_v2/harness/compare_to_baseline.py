"""
Re-run the pipeline and diff against outputs/baseline/<sample>.json.

Exits non-zero if any hard gate is violated.

GATE (revised after the first T0 run -- see planning/phase2-implementation-plan.md,
"Measured noise floor"). sklearn's KMeans is not bit-reproducible across
processes even with random_state fixed: floating-point summation order in the
threaded Lloyd iteration varies, which flips a handful of pixels sitting
exactly on a cluster boundary. That moves reported plane-normal direction by a
tiny, measured amount (<=0.027 deg on the 12-sample set) with zero effect on
d, gravity, wall_coverage, or the rendered bytes. The gate below is exact
everywhere except plane normal/offset, which get a tolerance sized to that
measured noise floor rather than zero.

    final.png bytes         exact
    wall_coverage            exact
    gravity vector           exact
    plane offset d           <= 1 mm
    plane normal direction   <= 0.1 deg
    planes lost               0

Usage:
    python3 harness/compare_to_baseline.py
"""

import json
import sys

import numpy as np

from _common import BASELINE_DIR, all_samples, render_one
from resolution_sensitivity import match

D_TOL_M = 0.001
NORMAL_TOL_DEG = 0.1

FMT = "%-24s%10s%10s%10s%12s%10s%8s"


def _as_match_planes(baseline):
    return dict(planes=[dict(n=np.array(p["normal"]), d=p["d"]) for p in baseline["planes"]])


def compare_one(name, baseline):
    current, err = render_one(name)
    if err:
        return None, err

    pairs = match(_as_match_planes(baseline), _as_match_planes(current))
    normal_deltas = [ang for _, pb, ang in pairs if pb is not None]
    d_deltas = [abs(pa["d"] - pb["d"]) for pa, pb, _ in pairs if pb is not None]
    planes_lost = sum(1 for _, pb, _ in pairs if pb is None)

    g_a = np.array(baseline["gravity"])
    g_b = np.array(current["gravity"])
    gravity_delta_deg = float(np.degrees(np.arccos(np.clip(abs(g_a @ g_b), -1, 1))))

    row = dict(
        sample=name,
        max_normal_delta_deg=max(normal_deltas) if normal_deltas else 0.0,
        max_d_delta_m=max(d_deltas) if d_deltas else 0.0,
        planes_lost=planes_lost,
        gravity_delta_deg=gravity_delta_deg,
        wall_coverage_delta=abs(baseline["wall_coverage"] - current["wall_coverage"]),
        render_identical=(baseline["render_sha256"] == current["render_sha256"]),
    )
    return row, None


def _gate_ok(r):
    return (
        r["max_normal_delta_deg"] <= NORMAL_TOL_DEG
        and r["max_d_delta_m"] <= D_TOL_M
        and r["planes_lost"] == 0
        and r["gravity_delta_deg"] == 0.0
        and r["wall_coverage_delta"] == 0.0
        and r["render_identical"]
    )


def main():
    samples = all_samples()
    rows, missing, failed = [], [], []

    print(FMT % ("sample", "dNormal", "dD(m)", "lost", "dGravity", "dCover", "render"))
    for name in samples:
        bp = BASELINE_DIR / f"{name}.json"
        if not bp.exists():
            missing.append(name)
            continue
        baseline = json.loads(bp.read_text())
        row, err = compare_one(name, baseline)
        if err:
            failed.append((name, err))
            continue
        rows.append(row)
        print(FMT % (
            name,
            "%.3f" % row["max_normal_delta_deg"],
            "%.4f" % row["max_d_delta_m"],
            row["planes_lost"],
            "%.3f" % row["gravity_delta_deg"],
            "%.4f" % row["wall_coverage_delta"],
            "yes" if row["render_identical"] else "NO",
        ))

    print(f"\n{'=' * 70}")
    if missing:
        print(f"no baseline captured for: {', '.join(missing)}")
    for name, err in failed:
        print(f"FAILED to rerun {name}: {err}")

    noise_floor = max((r["max_normal_delta_deg"] for r in rows), default=0.0)
    print(f"noise floor (max plane-normal delta across set): {noise_floor:.4f} deg "
          f"[tolerance {NORMAL_TOL_DEG} deg]"
          + ("  ** approaching tolerance, worth investigating **"
             if noise_floor > 0.5 * NORMAL_TOL_DEG else ""))

    gate_fail = [r["sample"] for r in rows if not _gate_ok(r)]
    if gate_fail:
        print(f"GATE FAILED for: {', '.join(gate_fail)}")
    else:
        print(f"gate PASSED across {len(rows)} samples "
              f"(exact except plane normal <= {NORMAL_TOL_DEG} deg, "
              f"plane offset <= {D_TOL_M * 1000:.0f} mm)")
    print("=" * 70)

    return 1 if (gate_fail or failed) else 0


if __name__ == "__main__":
    sys.exit(main())
