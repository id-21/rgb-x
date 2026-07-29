"""
Gather evidence for classifying non-Manhattan plane pairs (T5 in
planning/phase2-implementation-plan.md). No pipeline change -- this script
only measures and renders overlays; planning/plane-diagnosis-findings.md is
written by hand from what it reports.

Scope: the three samples the plan's own table names (complex_living,
fluted_walls_living, obstacle_course_living). T3 already established (see
its commit) that after removing the top_n cap, most of the *new* raw
non-Manhattan pair volume (61 of 69 across the full 12-sample set) lives
among sub-10%-of-wall slivers and was explicitly NOT re-gated per-pair for
exactly that reason -- classifying each one individually would be noise, not
signal. This script instead classifies every PLANE (not pair) that
participates in a non-Manhattan pair AND covers at least PX_SHARE_MIN of the
wall, in the three named samples, which is where the plan's original table
came from.

For each flagged plane, reports:
  - normal, model_vs_depth_normal_deg, px_share
  - region decomposition: region count and the largest region's share of the
    plane's own pixels (close to 1 = one compact blob, i.e. NOT scattered;
    low with many regions = interleaved/scattered, suggestive of a
    clustering artifact per T4's docstring)
  - whether the plane overlaps a high-model_vs_depth area (glass/mirror
    suspicion), using the existing 20 deg threshold from verify.py's checks
  - an overlay PNG colour-coding the flagged planes on the photo
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "harness"))
from geometry import cluster_planes, fit_planes, extract_regions, estimate_gravity  # noqa: E402
from _common import load_geometry_inputs, BATCH  # noqa: E402

SAMPLES = ["complex_living", "fluted_walls_living", "obstacle_course_living"]
PX_SHARE_MIN = 0.03           # below this, T3 already showed it's slivers, not signal
MODEL_VS_DEPTH_BAD = 20.0     # deg; matches verify()'s existing threshold
OUT_DIR = Path(__file__).parent / "plane_diagnosis"


def non_manhattan_pairs(planes):
    ids = sorted(planes)
    out = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = planes[ids[i]], planes[ids[j]]
            ang = float(np.degrees(np.arccos(np.clip(abs(a["n"] @ b["n"]), -1, 1))))
            if not (ang < 20 or abs(ang - 90) < 20):
                out.append((ids[i], ids[j], ang))
    return out


def overlay(rgb, labels, flagged_ids):
    pal = np.array([[238, 82, 83], [46, 134, 222], [255, 159, 67],
                    [16, 172, 132], [162, 155, 254]], np.uint8)
    out = rgb.copy()
    for i, pid in enumerate(flagged_ids):
        m = labels == pid
        color = pal[i % len(pal)]
        out[m] = (0.5 * out[m].astype(np.float32) + 0.5 * color).astype(np.uint8)
    return out


def diagnose(name):
    gi = load_geometry_inputs(name)
    if gi is None:
        return None
    labels = cluster_planes(gi["normals"], gi["mask"], depth=gi["depth"],
                            fx=gi["f"], cx=gi["cx"], cy=gi["cy"],
                            n_clusters=8, scale=gi["scale"])
    planes = fit_planes(labels, gi["normals"], depth=gi["depth"],
                        fx=gi["f"], cx=gi["cx"], cy=gi["cy"],
                        wall_px=int(gi["mask"].sum()))
    pairs = non_manhattan_pairs(planes)
    wall_px = int(gi["mask"].sum())

    flagged_ids = sorted({int(pid) for a, b, _ in pairs for pid in (a, b)
                          if planes[pid]["npix"] / wall_px >= PX_SHARE_MIN})
    if not flagged_ids:
        return dict(sample=name, pairs=pairs, planes={}, flagged=[])

    region_labels, regions = extract_regions(labels)
    regions_by_plane = {}
    for r in regions.values():
        regions_by_plane.setdefault(r["plane"], []).append(r["npix"])

    info = {}
    for pid in flagged_ids:
        p = planes[pid]
        sizes = sorted(regions_by_plane.get(pid, [p["npix"]]), reverse=True)
        info[pid] = dict(
            normal=[float(x) for x in p["n"]],
            px_share=float(p["npix"] / wall_px),
            model_vs_depth_normal_deg=(None if p["disagree_deg"] is None
                                       else float(p["disagree_deg"])),
            model_vs_depth_suspect=(p["disagree_deg"] is not None
                                    and p["disagree_deg"] > MODEL_VS_DEPTH_BAD),
            n_regions=len(sizes),
            largest_region_frac=float(sizes[0] / sum(sizes)),
            confidence=float(p["confidence"]),
        )

    d = Path(gi and BATCH / name / f"{name}.jpg")
    rgb = cv2.cvtColor(cv2.imread(str(next((BATCH / name).glob(f"{name}.jp*g")))),
                       cv2.COLOR_BGR2RGB)
    if rgb.shape[:2] != labels.shape:
        rgb = cv2.resize(rgb, (labels.shape[1], labels.shape[0]), interpolation=cv2.INTER_AREA)
    OUT_DIR.mkdir(exist_ok=True)
    ov = overlay(rgb, labels, flagged_ids)
    cv2.imwrite(str(OUT_DIR / f"{name}_overlay.png"), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))

    return dict(sample=name,
               pairs=[(int(a), int(b), ang) for a, b, ang in pairs],
               n_pairs_total=len(pairs),
               n_pairs_among_flagged=sum(1 for a, b, _ in pairs
                                         if a in flagged_ids and b in flagged_ids),
               planes=info, flagged=flagged_ids)


def main():
    results = []
    for name in SAMPLES:
        r = diagnose(name)
        if r is None:
            print(f"{name}: SKIPPED (missing inputs)")
            continue
        results.append(r)
        print(f"\n=== {name} ===")
        print(f"  {r.get('n_pairs_total', len(r['pairs']))} non-Manhattan pairs total, "
              f"{r.get('n_pairs_among_flagged', 0)} among flagged (>= {PX_SHARE_MIN:.0%} share) planes")
        for pid, info in r["planes"].items():
            print(f"  plane {pid}: share={info['px_share']:.3f} "
                 f"model_vs_depth={info['model_vs_depth_normal_deg']} "
                 f"({'SUSPECT' if info['model_vs_depth_suspect'] else 'ok'}) "
                 f"regions={info['n_regions']} largest_frac={info['largest_region_frac']:.2f} "
                 f"confidence={info['confidence']:.2f}")

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "findings_raw.json").write_text(json.dumps(results, indent=2, default=float))
    return results


if __name__ == "__main__":
    main()
