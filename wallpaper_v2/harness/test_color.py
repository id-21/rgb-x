"""
T2 self-test: sRGB <-> linear round trip, known values, compositing invariant.

Not a pytest suite (the repo has none) -- a plain assert-based script in the
same style as verify.py. Exits non-zero on any failure.

Usage:
    python3 harness/test_color.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from render import srgb_to_linear, linear_to_srgb, composite  # noqa: E402


def test_round_trip():
    x = np.arange(256, dtype=np.float32) / 255.0
    back = linear_to_srgb(srgb_to_linear(x))
    err = np.abs(back - x)
    assert err.max() < 1.0 / 255, f"round-trip max error {err.max():.6f} >= 1/255"
    print(f"  ok: round trip, max error {err.max():.6f} (< {1/255:.6f})")


def test_known_values():
    cases = [(0.0, 0.0), (1.0, 1.0), (128 / 255.0, 0.2159)]
    for srgb, expected_linear in cases:
        got = float(srgb_to_linear(np.array(srgb, dtype=np.float32)))
        assert abs(got - expected_linear) < 1e-3, \
            f"srgb_to_linear({srgb}) = {got}, expected ~{expected_linear}"
        print(f"  ok: srgb_to_linear({srgb:.4f}) = {got:.4f} (~{expected_linear})")


def _invariant_case(linear):
    rng = np.random.default_rng(0)
    H, W = 64, 64
    photo = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
    texture = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
    mask = np.ones((H, W), dtype=bool)          # feather_mask(all-True) == 1 everywhere
    shading = np.ones((H, W), dtype=np.float32)

    out = composite(photo, texture, mask, shading, feather=2, linear=linear)
    err = np.abs(out.astype(np.int16) - texture.astype(np.int16))
    assert err.max() <= 1, \
        f"linear={linear}: composite(shading=1, alpha=1) != texture, max err {err.max()}"
    print(f"  ok: invariant holds for linear={linear} (max px error {err.max()})")


def test_invariant():
    _invariant_case(linear=False)
    _invariant_case(linear=True)


def main():
    test_round_trip()
    test_known_values()
    test_invariant()
    print("\nall T2 harness tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
