"""
T0 self-test: capture -> compare with no code change -> gate passes.

Not a pytest suite (the repo has none) -- a plain assert-based script in the
same style as verify.py. Exits non-zero on any failure.

The gate is not exact float equality -- see compare_to_baseline.py's module
docstring for why (sklearn KMeans is not bit-reproducible across processes;
measured noise floor is <=0.027 deg of plane-normal direction with zero
effect on rendered pixels).

Usage:
    python3 harness/test_baseline.py
"""

import json
import sys

from _common import BASELINE_DIR, REQUIRED_KEYS, all_samples
from compare_to_baseline import D_TOL_M, NORMAL_TOL_DEG, _gate_ok, compare_one


def test_baseline_files_parse_and_complete():
    samples = all_samples()
    missing = [n for n in samples if not (BASELINE_DIR / f"{n}.json").exists()]
    assert not missing, f"no baseline captured for: {missing}"

    for name in samples:
        b = json.loads((BASELINE_DIR / f"{name}.json").read_text())
        missing_keys = REQUIRED_KEYS - b.keys()
        assert not missing_keys, f"{name}: baseline missing keys {missing_keys}"
        assert 0.0 <= b["wall_coverage"] <= 1.0, \
            f"{name}: wall_coverage {b['wall_coverage']} out of [0, 1]"
    print(f"  ok: {len(samples)} baseline files parse and have required keys")


def test_zero_drift_on_rerun():
    samples = all_samples()
    for name in samples:
        baseline = json.loads((BASELINE_DIR / f"{name}.json").read_text())
        row, err = compare_one(name, baseline)
        assert err is None, f"{name}: rerun failed: {err}"
        assert row["max_normal_delta_deg"] <= NORMAL_TOL_DEG, \
            f"{name}: normal drift {row['max_normal_delta_deg']} exceeds {NORMAL_TOL_DEG} deg"
        assert row["max_d_delta_m"] <= D_TOL_M, \
            f"{name}: distance drift {row['max_d_delta_m']} exceeds {D_TOL_M} m"
        assert row["planes_lost"] == 0, f"{name}: lost planes on rerun {row}"
        assert row["gravity_delta_deg"] == 0.0, f"{name}: gravity drift {row}"
        assert row["wall_coverage_delta"] == 0.0, f"{name}: wall_coverage drift {row}"
        assert row["render_identical"], f"{name}: render bytes differ on rerun"
        assert _gate_ok(row), f"{name}: gate failed {row}"
        print(f"  ok: {name} within gate (normal {row['max_normal_delta_deg']:.4f} deg)")


def main():
    test_baseline_files_parse_and_complete()
    test_zero_drift_on_rerun()
    print("\nall T0 harness tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
