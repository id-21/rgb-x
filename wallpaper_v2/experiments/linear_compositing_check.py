"""
Characterise the T2 linear-vs-sRGB compositing change: per-sample luminance
shift within the wall mask, old (sRGB-space multiply) vs new (linear-space).

Not a gate -- T2's gate is the round-trip/known-value/invariant unit tests in
harness/test_color.py plus "no geometry change" (checked separately via
harness/compare_to_baseline.py). This script only characterises the expected
direction: shadowed regions should get lighter, highlights roughly unchanged.

Runs geometry + texture sampling ONCE per sample (unaffected by T2) and calls
composite() twice -- linear=False, linear=True -- on the same shaded/sampled
data, so this costs one pipeline pass per sample, not two.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from run_all import find_inputs, fit_scale, BATCH  # noqa: E402
from geometry import (load_normals, estimate_gravity, cluster_planes,   # noqa: E402
                      fit_planes, compute_uv, WALL_CONF_T)
from render import sample_texture, shading_from_irradiance, composite   # noqa: E402

PATTERN_W_M = 0.53


def _lum(rgb):
    return (rgb.astype(np.float32) @ np.array([0.2126, 0.7152, 0.0722], np.float32))


def check_one(name, tex_bgr):
    g = find_inputs(name)
    if g is None or g["depth"] is None or g["focal"] is None or not g["irr"].exists():
        return None

    rgb = cv2.cvtColor(cv2.imread(str(g["rgb"])), cv2.COLOR_BGR2RGB)
    normals, _ = load_normals(g["normal"], verbose=False)
    normals = normals.astype(np.float32)
    conf = np.load(g["conf"])
    depth = np.load(g["depth"]).astype(np.float32)
    f_full = float(np.load(g["focal"]))
    irr = np.load(g["irr"])

    s = fit_scale(rgb.shape)
    if s < 1.0:
        H2, W2 = int(round(rgb.shape[0] * s)), int(round(rgb.shape[1] * s))
        rgb = cv2.resize(rgb, (W2, H2), interpolation=cv2.INTER_AREA)
        normals = cv2.resize(normals, (W2, H2), interpolation=cv2.INTER_NEAREST)
        normals /= (np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-12)
        conf = cv2.resize(conf, (W2, H2), interpolation=cv2.INTER_AREA)
        depth = cv2.resize(depth, (W2, H2), interpolation=cv2.INTER_NEAREST)
        irr = cv2.resize(irr, (W2, H2), interpolation=cv2.INTER_AREA)
    H, W = depth.shape
    f = f_full * s
    cx, cy = W / 2.0, H / 2.0
    mask = conf > WALL_CONF_T
    if mask.sum() < 0.01 * mask.size:
        return None

    grav, _, _, _ = estimate_gravity(normals, mask)
    labels = cluster_planes(normals, mask, depth=depth, fx=f, cx=cx, cy=cy,
                            n_clusters=8, top_n=4)
    planes = fit_planes(labels, normals, depth=depth, fx=f, cx=cx, cy=cy)
    if not planes:
        return None
    uv, info = compute_uv(labels, planes, grav, f, cx, cy)

    tex_rgb = cv2.cvtColor(tex_bgr, cv2.COLOR_BGR2RGB)
    sampled, wall = sample_texture(uv, labels, tex_rgb, PATTERN_W_M)
    if not wall.any():
        return None
    shading = shading_from_irradiance(irr, wall)

    old = composite(rgb, sampled, wall, shading, linear=False)
    new = composite(rgb, sampled, wall, shading, linear=True)

    lum_old, lum_new = _lum(old)[wall], _lum(new)[wall]
    pct = [10, 50, 90]
    return dict(
        sample=name,
        mean_old=float(lum_old.mean()), mean_new=float(lum_new.mean()),
        p_old=[float(np.percentile(lum_old, p)) for p in pct],
        p_new=[float(np.percentile(lum_new, p)) for p in pct],
    )


def main():
    tex_bgr = cv2.imread(str(BATCH / "30106.jpg"))
    rows = []
    for d in sorted(BATCH.iterdir()):
        if not d.is_dir():
            continue
        r = check_one(d.name, tex_bgr)
        if r is None:
            continue
        rows.append(r)

    print(f"{'sample':<24}{'mean old':>10}{'mean new':>10}"
          f"{'p10 old':>9}{'p10 new':>9}{'p50 old':>9}{'p50 new':>9}{'p90 old':>9}{'p90 new':>9}")
    for r in rows:
        print(f"{r['sample']:<24}{r['mean_old']:>10.1f}{r['mean_new']:>10.1f}"
              f"{r['p_old'][0]:>9.1f}{r['p_new'][0]:>9.1f}"
              f"{r['p_old'][1]:>9.1f}{r['p_new'][1]:>9.1f}"
              f"{r['p_old'][2]:>9.1f}{r['p_new'][2]:>9.1f}")

    p10_deltas = [r["p_new"][0] - r["p_old"][0] for r in rows]
    p90_deltas = [r["p_new"][2] - r["p_old"][2] for r in rows]
    print(f"\np10 (shadow) delta: mean {np.mean(p10_deltas):+.1f}, "
          f"all positive (lighter)={all(d > 0 for d in p10_deltas)}")
    print(f"p90 (highlight) delta: mean {np.mean(p90_deltas):+.1f}")

    Path(__file__).with_suffix(".json").write_text(json.dumps(rows, indent=2))
    return rows


if __name__ == "__main__":
    main()
