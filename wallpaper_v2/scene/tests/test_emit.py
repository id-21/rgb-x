"""
P3-T2 self-test: emitted bundles validate, resolve their own regions, are
resolution-invariant in the camera intrinsics, and obey the plane-in-front
sign convention.

Not a pytest suite (the repo has none, see harness/test_regions.py) -- a
plain assert-based script. Exits non-zero on any failure.

Usage:
    python3 scene/tests/test_emit.py
"""
import json
import sys
import tempfile
from pathlib import Path

import cv2
import jsonschema
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from run_all import BATCH  # noqa: E402
from scene.emit import emit_scene, SHADING_SCALE  # noqa: E402

SCENE_DIR = Path(__file__).resolve().parent.parent
SCHEMA = json.loads((SCENE_DIR / "schema" / "scene-v1.schema.json").read_text())

F_PX_TOL = 1e-6
SIGN_SAMPLE_PX = 5000


def all_samples():
    return sorted(p.name for p in BATCH.iterdir() if p.is_dir())


def _try_emit(name, out_dir, max_width=None):
    try:
        emit_scene(name, out_dir, max_width=max_width)
        return True, None
    except ValueError as e:
        return False, str(e)


def test_all_samples_emit_valid_bundles():
    emitted, skipped = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for name in all_samples():
            out = Path(tmp) / name
            ok, err = _try_emit(name, out)
            if not ok:
                skipped.append(f"{name}: {err}")
                continue
            scene = json.loads((out / "scene.json").read_text())
            jsonschema.validate(scene, SCHEMA)

            plane_ids = {p["id"] for p in scene["planes"]}
            region_planes = {r["plane"] for r in scene["regions"]}
            assert region_planes <= plane_ids, \
                f"{name}: region references a plane id not in planes[]"
            assert plane_ids <= region_planes, \
                f"{name}: plane with no region -- {plane_ids - region_planes}"

            maps = cv2.imread(str(out / "maps.png"))
            assert maps is not None, f"{name}: maps.png failed to load"
            r_ids = set(np.unique(maps[..., 2]).tolist())
            expected = {0} | {rid + 1 for rid in region_planes | {r["id"] for r in scene["regions"]}}
            assert r_ids == expected, \
                f"{name}: maps.png R channel {sorted(r_ids)} != expected {sorted(expected)}"

            shading16 = cv2.imread(str(out / "shading.png"), cv2.IMREAD_UNCHANGED)
            assert shading16 is not None and shading16.dtype == np.uint16, \
                f"{name}: shading.png missing or not 16-bit"

            emitted.append(name)
            print(f"  ok: {name} -- {len(plane_ids)} planes, {len(scene['regions'])} regions, "
                  f"schema-valid")

    print(f"\nemitted {len(emitted)}/{len(emitted) + len(skipped)} samples")
    for s in skipped:
        print("  SKIPPED:", s)
    assert not skipped, f"{len(skipped)} sample(s) failed to emit: {skipped}"
    return emitted


def test_camera_scale_invariance(samples):
    checked = 0
    with tempfile.TemporaryDirectory() as tmp:
        for name in samples:
            out_small = Path(tmp) / f"{name}_small"
            out_native = Path(tmp) / f"{name}_native"
            ok1, _ = _try_emit(name, out_small, max_width=1024)
            ok2, _ = _try_emit(name, out_native, max_width=None)
            if not (ok1 and ok2):
                continue
            s_small = json.loads((out_small / "scene.json").read_text())
            s_native = json.loads((out_native / "scene.json").read_text())

            ratio_small = s_small["camera"]["f_px"] / s_small["image"]["width"]
            ratio_native = s_native["camera"]["f_px"] / s_native["image"]["width"]
            err = abs(ratio_small - ratio_native)
            assert err < F_PX_TOL, \
                f"{name}: f_px/width differs across resolutions by {err:.2e} (>= {F_PX_TOL})"
            checked += 1
            print(f"  ok: {name} f_px/width invariant "
                  f"({ratio_small:.6f} @ {s_small['image']['width']}px vs "
                  f"{ratio_native:.6f} @ {s_native['image']['width']}px, err={err:.2e})")
    assert checked > 0, "no sample was checked for scale invariance"


def test_sign_convention(samples):
    """
    For sampled pixels of every plane, t = d / (n . K^-1 p) must be > 0 --
    the ray/plane intersection lies in front of the camera, not behind it.
    """
    rng = np.random.default_rng(0)
    checked_planes = 0
    with tempfile.TemporaryDirectory() as tmp:
        for name in samples:
            out = Path(tmp) / name
            ok, _ = _try_emit(name, out)
            if not ok:
                continue
            scene = json.loads((out / "scene.json").read_text())
            maps = cv2.imread(str(out / "maps.png"))
            f_px, cx, cy = (scene["camera"]["f_px"], scene["camera"]["cx"],
                            scene["camera"]["cy"])
            region_id_map = maps[..., 2].astype(np.int32) - 1   # -1 = no region
            plane_of_region = {r["id"]: r["plane"] for r in scene["regions"]}

            for plane in scene["planes"]:
                pid = plane["id"]
                own_regions = {rid for rid, pl in plane_of_region.items() if pl == pid}
                sel = np.isin(region_id_map, list(own_regions))
                ys, xs = np.nonzero(sel)
                if len(ys) == 0:
                    continue
                if len(ys) > SIGN_SAMPLE_PX:
                    idx = rng.choice(len(ys), SIGN_SAMPLE_PX, replace=False)
                    ys, xs = ys[idx], xs[idx]

                n = np.array(plane["normal"])
                d = plane["d"]
                rx = (xs - cx) / f_px
                ry = (ys - cy) / f_px
                denom = rx * n[0] + ry * n[1] + n[2]
                nondegenerate = np.abs(denom) > 1e-6
                t = d / denom[nondegenerate]
                assert np.all(t > 0), \
                    f"{name} plane {pid}: {int((t <= 0).sum())} sampled pixels have t <= 0 " \
                    f"(hit point behind the camera)"
                checked_planes += 1
    assert checked_planes > 0, "no plane was sign-checked"
    print(f"  ok: sign convention (t > 0) holds for {checked_planes} plane instances")


def main():
    samples = test_all_samples_emit_valid_bundles()
    test_camera_scale_invariance(samples)
    test_sign_convention(samples)
    print("\nall P3-T2 emitter tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
