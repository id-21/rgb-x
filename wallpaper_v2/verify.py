"""
Independent checks on the rendered results. Run after run_all.py.

These do not feed back into the pipeline; they exist to catch silent breakage.
"""
import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent / "outputs"


def main():
    s = json.loads((OUT / "summary.json").read_text())
    fails, warns = [], []

    print(f"{'sample':<24}{'ceil m':>8}{'tilt':>7}{'planes':>8}  checks")
    for d in s["samples"]:
        name = d["sample"]
        msgs = []

        # 1. normals must have been decoded
        if not d["normals_were_encoded"]:
            warns.append(f"{name}: normals were NOT [0,1]-encoded (unexpected)")

        # 2. plausible room height
        h = d["reconstructed_vertical_extent_m"]
        if not (2.0 <= h <= 4.5):
            fails.append(f"{name}: implausible vertical extent {h:.2f} m")
            msgs.append("BAD HEIGHT")

        # 3. camera not absurdly tilted
        t = d["gravity_tilt_from_image_vertical_deg"]
        if t > 25 and not d.get("gravity_ambiguous_single_wall"):
            fails.append(f"{name}: gravity tilt {t:.0f} deg")
            msgs.append("BAD GRAVITY")

        # 4. wall pairs should be perpendicular or parallel
        odd = [a for a in d["wall_pair_angles_deg"] if not (a < 20 or abs(a - 90) < 20)]
        if odd:
            warns.append(f"{name}: {len(odd)} non-Manhattan wall pair(s) "
                         f"{['%.0f' % a for a in odd]}")
            msgs.append(f"{len(odd)} odd angle(s)")

        # 5. planes must be in front of the camera at a sane distance
        for k, p in d["planes"].items():
            if not (0.4 <= p["distance_m"] <= 15.0):
                fails.append(f"{name}/plane{k}: distance {p['distance_m']:.2f} m")
                msgs.append("BAD DIST")
            if p["pattern_repeats_across"] > 40:
                warns.append(f"{name}/plane{k}: {p['pattern_repeats_across']:.0f} "
                             f"pattern repeats - texture may alias")

        # 6. rendered files exist and are not blank
        import cv2
        for tag in ("final.png",):
            f = OUT / "new" / f"{name}__{d['texture']}" / tag
            if not f.exists():
                fails.append(f"{name}: missing {tag}")
                continue
            im = cv2.imread(str(f))
            if im is None or im.std() < 5:
                fails.append(f"{name}: {tag} is blank")

        print(f"{name:<24}{h:>8.2f}{t:>7.1f}{d['n_planes']:>8}  "
              f"{'OK' if not msgs else ', '.join(msgs)}")

    print()
    for w in warns:
        print("  WARN ", w)
    for f in fails:
        print("  FAIL ", f)
    if s.get("failures"):
        print()
        for f in sorted(set(s["failures"])):
            print("  SKIPPED ", f)

    print(f"\n{len(s['samples'])} samples | {len(fails)} failures | {len(warns)} warnings")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
