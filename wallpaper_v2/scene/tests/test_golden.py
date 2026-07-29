"""
P3-T4 self-test: re-emitting reproduces every committed golden bundle.

Not a pytest suite (the repo has none, see harness/test_regions.py) -- a
plain assert-based script. Exits non-zero on any failure.

Usage:
    python3 scene/tests/test_golden.py
"""
import filecmp
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scene.emit import emit_scene  # noqa: E402

GOLDEN_DIR = Path(__file__).resolve().parent.parent / "golden"
FLOAT_TOL = 1e-6
BYTE_EXACT_FILES = ["maps.png", "shading.png", "photo.jpg"]


def _diff_floats(a, b, path=""):
    """Yield (path, a, b) for any pair of floats differing by > FLOAT_TOL."""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in a.keys() | b.keys():
            yield from _diff_floats(a.get(k), b.get(k), f"{path}.{k}")
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            yield (path, a, b)
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                yield from _diff_floats(x, y, f"{path}[{i}]")
    elif isinstance(a, float) or isinstance(b, float):
        if a is None or b is None or abs(a - b) > FLOAT_TOL:
            yield (path, a, b)
    else:
        if a != b:
            yield (path, a, b)


def golden_samples():
    return sorted(p.name for p in GOLDEN_DIR.iterdir()
                  if p.is_dir() and (p / "scene.json").exists())


def test_reemission_matches_golden():
    samples = golden_samples()
    assert samples, "no golden bundles found -- has P3-T4 committed any?"

    mismatches = []
    with tempfile.TemporaryDirectory() as tmp:
        for name in samples:
            golden = GOLDEN_DIR / name
            fresh = Path(tmp) / name
            emit_scene(name, fresh)

            g_scene = json.loads((golden / "scene.json").read_text())
            f_scene = json.loads((fresh / "scene.json").read_text())
            diffs = list(_diff_floats(g_scene, f_scene))
            if diffs:
                mismatches.append(f"{name}: scene.json differs at "
                                   f"{[d[0] for d in diffs]} (tol {FLOAT_TOL})")
                for path, a, b in diffs[:5]:
                    print(f"    {name}{path}: golden={a} fresh={b}")
                continue

            byte_ok = True
            for fname in BYTE_EXACT_FILES:
                if not filecmp.cmp(golden / fname, fresh / fname, shallow=False):
                    mismatches.append(f"{name}: {fname} is not byte-identical")
                    byte_ok = False

            if byte_ok:
                print(f"  ok: {name} reproduces golden "
                      f"(scene.json within {FLOAT_TOL}, {len(BYTE_EXACT_FILES)} files byte-exact)")

    if mismatches:
        for m in mismatches:
            print("  MISMATCH:", m)
    n_ok = len(samples) - len({m.split(":")[0] for m in mismatches})
    print(f"\n{n_ok}/{len(samples)} golden bundles reproduced")
    assert not mismatches, f"{len(mismatches)} golden mismatch(es): {mismatches}"


def main():
    test_reemission_matches_golden()
    print("\nall P3-T4 golden tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
