# Regenerating the golden bundles

These are the P3-T4 reference bundles: one `scene.json` + `maps.png` +
`shading.png` + `photo.jpg` per eval sample, committed so that
`scene/tests/test_golden.py` can catch any silent change in `emit.py` or the
geometry/render pipeline it depends on.

## How

```bash
cd wallpaper_v2
python3 -c "
from scene.emit import emit_scene
from run_all import BATCH
from pathlib import Path
for p in sorted(BATCH.iterdir()):
    if p.is_dir():
        emit_scene(p.name, Path('scene/golden')/p.name)
"
```

Or regenerate a single sample:

```bash
python3 scene/emit.py <sample> --out scene/golden/<sample>
```

## When

Only when a change to `geometry.py`, `render.py`, or `scene/emit.py` is
*expected* to change the emitted bundles — e.g. a new default, a bug fix
that shifts a plane fit, a change to the `maps.png`/`shading.png` packing.
Never regenerate to make a failing test pass without first understanding
*why* it failed.

## Review the diff — do not blindly accept it

`scene.json` is text; diff it normally. For `maps.png`, `shading.png` and
`photo.jpg`, at minimum re-render the comparison sheet
(`run_all.py --comparisons`) or eyeball the new bundle's `photo.jpg` next to
the old one, and sanity-check `scene.json`'s `quality` block hasn't
regressed (`wall_coverage`, `gravity_tilt_deg`, `non_manhattan_major_pairs`).
A silent geometry regression that still "looks plausible" in a diff is
exactly what these goldens exist to catch.

## The noise floor

Phase 2 T0 established that k-means shifts plane normals by up to 0.027°
between processes (`experiments/` in that phase's harness). `scene.json`'s
floats are therefore compared at **1e-6** absolute tolerance, not exact
equality. `maps.png`, `shading.png` and `photo.jpg` ARE compared byte-exact:
region ids and quantised shading/alpha are insensitive to that magnitude of
noise, so if any of the three ever differ, something real changed.
