"""
T1 self-test: plane IDs are content-derived and stable.

Not a pytest suite (the repo has none) -- a plain assert-based script in the
same style as verify.py. Exits non-zero on any failure.

Usage:
    python3 harness/test_determinism.py
"""

import json
import sys

import numpy as np

from _common import BASELINE_DIR, all_samples, load_geometry_inputs
from compare_to_baseline import compare_one
from geometry import cluster_planes, fit_planes, points_for

SEEDS = [42, 7, 1234]
ANGLE_TOL_DEG = 1.0
D_TOL_M = 0.05


def test_seed_stability():
    stable_count, total, unstable = 0, 0, []
    for name in all_samples():
        gi = load_geometry_inputs(name)
        if gi is None:
            continue
        total += 1

        results = {}
        for seed in SEEDS:
            labels = cluster_planes(gi["normals"], gi["mask"], depth=gi["depth"],
                                    fx=gi["f"], cx=gi["cx"], cy=gi["cy"],
                                    n_clusters=8, scale=gi["scale"], random_state=seed)
            results[seed] = fit_planes(labels, gi["normals"], depth=gi["depth"],
                                       fx=gi["f"], cx=gi["cx"], cy=gi["cy"])

        stable = len({len(results[s]) for s in SEEDS}) == 1
        if stable:
            base = results[SEEDS[0]]
            for seed in SEEDS[1:]:
                other = results[seed]
                for pid, p in base.items():
                    if pid not in other:
                        stable = False
                        break
                    ang = np.degrees(np.arccos(
                        np.clip(abs(p["n"] @ other[pid]["n"]), -1, 1)))
                    dd = abs(p["d"] - other[pid]["d"])
                    if ang > ANGLE_TOL_DEG or dd > D_TOL_M:
                        stable = False
                if not stable:
                    break

        print(f"  {name}: seed-stable across {SEEDS} = {stable} "
              f"({len(results[SEEDS[0]])} planes)")
        if stable:
            stable_count += 1
        else:
            unstable.append(name)

    print(f"seed-stable samples: {stable_count}/{total}")
    assert stable_count >= 11, \
        f"seed stability {stable_count}/{total} below gate (>=11); unstable: {unstable}"


def test_near_tie_simulation():
    for name in all_samples():
        gi = load_geometry_inputs(name)
        if gi is None:
            continue
        labels = cluster_planes(gi["normals"], gi["mask"], depth=gi["depth"],
                                fx=gi["f"], cx=gi["cx"], cy=gi["cy"],
                                n_clusters=8, scale=gi["scale"])
        ids = sorted(int(l) for l in np.unique(labels) if l >= 0)
        wall_px = int(gi["mask"].sum())

        keys = {}
        for l in ids:
            sel = labels == l
            npix = int(sel.sum())
            n = gi["normals"][sel].mean(0)
            n /= np.linalg.norm(n) + 1e-12
            d = float(np.median(points_for(sel, gi["depth"], gi["f"], gi["cx"], gi["cy"]) @ n))
            ys, xs = np.nonzero(sel)
            keys[l] = dict(npix=npix, qd=round(d / 0.01),
                           cx=float(xs.mean()), cy=float(ys.mean()))

        def qcount(npix):
            return round(npix / max(wall_px, 1) * 200)

        for pct in (0.004, -0.004):
            order = sorted(ids, key=lambda l: (
                -qcount(keys[l]["npix"] * (1 + pct)), keys[l]["qd"], keys[l]["cx"], keys[l]["cy"]))
            assert order == ids, \
                f"{name}: ID reordering under {pct:+.1%} pixel-count jitter: {ids} -> {order}"
        print(f"  ok: {name} order stable under +/-0.4% pixel-count jitter ({len(ids)} planes)")


def test_render_identity():
    for name in all_samples():
        bp = BASELINE_DIR / f"{name}.json"
        assert bp.exists(), f"{name}: no baseline captured, run capture_baseline.py first"
        baseline = json.loads(bp.read_text())
        row, err = compare_one(name, baseline)
        assert err is None, f"{name}: rerun failed: {err}"
        assert row["render_identical"], f"{name}: render bytes differ after T1 ID reordering"
        print(f"  ok: {name} render_sha256 identical")


def main():
    test_seed_stability()
    test_near_tie_simulation()
    test_render_identity()
    print("\nall T1 harness tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
