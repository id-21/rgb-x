"""
Confidence-threshold sensitivity, across the full 12-sample set.

Promotes the single-room analysis in
"Notebook for Output visualisation/confidence_contour_explorer.ipynb"
(SAMPLE_INDEX = 0, Otsu = 0.482) to something reproducible over every sample,
per T0 in planning/phase2-implementation-plan.md. That notebook justified
WALL_CONF_T = 0.5 in geometry.py from one room; this script re-checks it and
should be re-run whenever the eval set grows.

Reports, per sample:
  - Otsu threshold on the wall-confidence map
  - fraction of pixels in the "ambiguous band" (0.02, 0.98) -- neither
    confidently wall nor confidently not-wall
  - mask-area swing between threshold 0.3 and 0.7, relative to the area at 0.5

The ambiguous-band and area-swing numbers are the evidence for T6: a large
fraction of boundary pixels are not resolved by any single threshold, which is
the argument for the guided-filter approach over hard thresholding.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from geometry import WALL_CONF_T  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
BATCH = ROOT / "rgb-x_output" / "batch_output"


def otsu_threshold(conf):
    u8 = np.clip(conf * 255.0, 0, 255).astype(np.uint8)
    t, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return t / 255.0


def sweep_one(conf):
    otsu = otsu_threshold(conf)
    ambiguous_frac = float(((conf > 0.02) & (conf < 0.98)).mean())
    area_05 = int((conf > WALL_CONF_T).sum())
    area_03 = int((conf > 0.3).sum())
    area_07 = int((conf > 0.7).sum())
    area_swing_pct = 100.0 * (area_03 - area_07) / max(area_05, 1)
    return dict(otsu=float(otsu), ambiguous_band_frac=ambiguous_frac,
                area_swing_pct=float(area_swing_pct))


def main():
    rows = []
    for d in sorted(BATCH.iterdir()):
        if not d.is_dir():
            continue
        cf = d / f"{d.name}_wall_confidence.npy"
        if not cf.exists():
            continue
        conf = np.load(cf).astype(np.float32)
        r = sweep_one(conf)
        r["sample"] = d.name
        rows.append(r)

    otsu = np.array([r["otsu"] for r in rows])
    amb = np.array([r["ambiguous_band_frac"] for r in rows])
    swing = np.array([r["area_swing_pct"] for r in rows])

    print(f"{'sample':<24}{'otsu':>8}{'ambig%':>9}{'swing%':>9}")
    for r in rows:
        print(f"{r['sample']:<24}{r['otsu']:>8.3f}{100*r['ambiguous_band_frac']:>8.1f}%"
              f"{r['area_swing_pct']:>8.1f}%")

    print(f"\n{'=' * 55}")
    print(f"Otsu threshold  : min={otsu.min():.3f} median={np.median(otsu):.3f} "
          f"max={otsu.max():.3f} std={otsu.std():.3f}")
    print(f"Ambiguous band  : min={100*amb.min():.1f}% median={100*np.median(amb):.1f}% "
          f"max={100*amb.max():.1f}%")
    print(f"Area swing 0.3->0.7: min={swing.min():.1f}% median={np.median(swing):.1f}% "
          f"max={swing.max():.1f}%")
    print(f"WALL_CONF_T = {WALL_CONF_T} is "
          f"{abs(WALL_CONF_T - np.median(otsu)) / max(otsu.std(), 1e-9):.2f} sigma "
          f"from the median per-sample Otsu optimum")
    print("=" * 55)

    Path(__file__).with_suffix(".json").write_text(json.dumps(rows, indent=2))
    return rows


if __name__ == "__main__":
    main()
