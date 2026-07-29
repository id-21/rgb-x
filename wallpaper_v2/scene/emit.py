"""
P3-T2: emit scene.json + maps.png + photo.jpg bundles.

The bundle is the sole boundary between the Python pipeline and everything
downstream (see SCHEMA.md). This module runs the same geometry pipeline as
run_all.process(), but stops before texture sampling/compositing -- a
consumer does that itself, from the bundle alone (see scene/consume.py).
"""

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from geometry import (load_normals, estimate_gravity, cluster_planes,
                      fit_planes, compute_uv, verify, WALL_CONF_T,
                      extract_regions, suggested_label)
from render import shading_from_irradiance, feather_mask, irradiance_polarity
from run_all import find_inputs, fit_scale, PATTERN_W_M

SHADING_SCALE = 1.6           # matches shading_from_irradiance's clip ceiling
JPEG_QUALITY = 92             # P3-T3 measures and may revise this
MANHATTAN_TOL_DEG = 15.0      # matches geometry.verify()'s tolerance
MAJOR_PLANE_SHARE = 0.10      # "major" plane threshold for non_manhattan_major_pairs


def _pipeline_sha(repo_root):
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _region_area_m2(uv, region_labels, rid):
    """
    Bounding-box area of a region's on-plane UV footprint, in square metres.

    An approximation: the true footprint may be L-shaped or otherwise
    non-rectangular (e.g. a wall split by a doorway), so this over-estimates
    for irregular shapes. Adequate for the field's stated purpose --
    hit-testing and label placement -- not for anything render-critical.
    """
    sel = region_labels == rid
    u, v = uv[sel, 0], uv[sel, 1]
    return float((u.max() - u.min()) * (v.max() - v.min()))


def _load_geometry(name, max_width):
    """Run the pipeline through region extraction + shading. No compositing."""
    g = find_inputs(name)
    if g is None:
        raise ValueError(f"{name}: missing normal/mask inputs")
    if g["depth"] is None:
        raise ValueError(f"{name}: no Depth Pro depth map")
    if g["focal"] is None:
        raise ValueError(f"{name}: no Depth Pro focal length")
    if not g["irr"].exists():
        raise ValueError(f"{name}: no irradiance map, cannot emit shading")

    rgb = cv2.cvtColor(cv2.imread(str(g["rgb"])), cv2.COLOR_BGR2RGB)
    normals, _ = load_normals(g["normal"], verbose=False)
    normals = normals.astype(np.float32)
    conf = np.load(g["conf"])
    depth = np.load(g["depth"]).astype(np.float32)
    f_full = float(np.load(g["focal"]))
    irr = np.load(g["irr"])

    W_native = rgb.shape[1]
    s = fit_scale(rgb.shape, maxdim=max_width)
    if s < 1.0:
        H2, W2 = int(round(rgb.shape[0] * s)), int(round(rgb.shape[1] * s))
        rgb = cv2.resize(rgb, (W2, H2), interpolation=cv2.INTER_AREA)
        normals = cv2.resize(normals, (W2, H2), interpolation=cv2.INTER_NEAREST)
        normals /= (np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-12)
        conf = cv2.resize(conf, (W2, H2), interpolation=cv2.INTER_AREA)
        depth = cv2.resize(depth, (W2, H2), interpolation=cv2.INTER_NEAREST)
        irr = cv2.resize(irr, (W2, H2), interpolation=cv2.INTER_AREA)
    H, W = depth.shape
    # f must be scaled by the REALIZED width ratio (W/W_native), not the
    # nominal continuous `s` from fit_scale: W was rounded to the nearest
    # integer pixel, and cx is anchored to that rounded W. Scaling f by `s`
    # instead lets sub-pixel rounding noise leak into f_px/width -- a smaller
    # instance of the fx_multiplier class of bug (see SCHEMA.md), just several
    # orders of magnitude subtler. Deriving f from the same ratio as cx keeps
    # f_px/image.width exactly invariant regardless of how H rounded.
    f = f_full * (W / W_native)
    cx, cy = W / 2.0, H / 2.0

    mask = conf > WALL_CONF_T
    if mask.sum() < 0.01 * mask.size:
        raise ValueError(f"{name}: wall mask nearly empty ({100*mask.mean():.1f}%)")

    grav, resid, tilt, grav_ambiguous = estimate_gravity(normals, mask)
    labels = cluster_planes(normals, mask, depth=depth, fx=f, cx=cx, cy=cy,
                            n_clusters=8, scale=s)
    planes = fit_planes(labels, normals, depth=depth, fx=f, cx=cx, cy=cy,
                        wall_px=int(mask.sum()))
    if not planes:
        raise ValueError(f"{name}: no planes fitted")
    uv, info = compute_uv(labels, planes, grav, f, cx, cy)
    region_labels, regions = extract_regions(labels)

    checks = verify(planes, grav, info)
    angs = [v for k, v in [(c[0], c[1]) for c in checks] if k.startswith("angle")]
    wall_coverage = float((labels >= 0).sum() / max(mask.sum(), 1))
    wall = labels >= 0

    step = max(1, int(np.sqrt(H * W / 400_000)))
    ys, xs = np.mgrid[0:H:step, 0:W:step]
    Zs = depth[::step, ::step]
    along_g = (((xs - cx) / f) * grav[0] + ((ys - cy) / f) * grav[1] + grav[2]) * Zs
    ceiling_h = float(np.percentile(along_g, 99.5) - np.percentile(along_g, 0.5))

    pol = irradiance_polarity(irr, rgb, wall)
    shading = shading_from_irradiance(irr, wall)
    alpha = feather_mask(wall, radius=2)

    return dict(
        rgb=rgb, f=f, cx=cx, cy=cy, W=W, H=H,
        gravity=grav, gravity_tilt=tilt, gravity_ambiguous=grav_ambiguous,
        labels=labels, planes=planes, uv=uv, info=info,
        region_labels=region_labels, regions=regions,
        wall_pair_angles=angs, wall_coverage=wall_coverage,
        vertical_extent_m=ceiling_h, irradiance_polarity_corr=pol,
        shading=shading, alpha=alpha, mask_npix=int(mask.sum()),
    )


def _build_scene_dict(name, geo, repo_root):
    plane_ids = sorted(geo["planes"].keys())
    planes_json = []
    for pid in plane_ids:
        p, inf = geo["planes"][pid], geo["info"][pid]
        planes_json.append(dict(
            id=int(pid),
            normal=[float(x) for x in p["n"]],
            d=float(p["d"]),
            u_axis=[float(x) for x in inf["u_axis"]],
            v_axis=[float(x) for x in inf["v_axis"]],
            origin=[float(x) for x in inf["origin"]],
            px_share=float(p["npix"] / max(geo["mask_npix"], 1)),
            extent_m=[float(inf["width_m"]), float(inf["height_m"])],
            confidence=float(p["confidence"]),
            model_vs_depth_deg=(None if p["disagree_deg"] is None
                                 else float(p["disagree_deg"])),
            suggested_label=suggested_label(p["n"]),
        ))

    region_ids = sorted(geo["regions"].keys())
    regions_json = []
    for rid in region_ids:
        r = geo["regions"][rid]
        regions_json.append(dict(
            id=int(rid),
            plane=int(r["plane"]),
            npix=int(r["npix"]),
            area_m2=_region_area_m2(geo["uv"], geo["region_labels"], rid),
            centroid_px=[float(x) for x in r["centroid_px"]],
            bbox_px=[float(x) for x in r["bbox_px"]],
        ))

    major = [pid for pid in plane_ids
             if geo["planes"][pid]["npix"] / max(geo["mask_npix"], 1) >= MAJOR_PLANE_SHARE]
    non_manhattan = 0
    for i in range(len(major)):
        for j in range(i + 1, len(major)):
            ni, nj = geo["planes"][major[i]]["n"], geo["planes"][major[j]]["n"]
            ang = float(np.degrees(np.arccos(np.clip(abs(ni @ nj), -1, 1))))
            if not (ang < MANHATTAN_TOL_DEG or abs(ang - 90) < MANHATTAN_TOL_DEG):
                non_manhattan += 1

    disagree = [p["disagree_deg"] for p in geo["planes"].values()
                if p["disagree_deg"] is not None]

    return dict(
        version=1,
        pipeline_sha=_pipeline_sha(repo_root),
        image=dict(width=int(geo["W"]), height=int(geo["H"])),
        camera=dict(f_px=float(geo["f"]), cx=float(geo["cx"]), cy=float(geo["cy"])),
        gravity=[float(x) for x in geo["gravity"]],
        planes=planes_json,
        regions=regions_json,
        quality=dict(
            vertical_extent_m=geo["vertical_extent_m"],
            wall_pair_angles_deg=[float(a) for a in geo["wall_pair_angles"]],
            non_manhattan_major_pairs=int(non_manhattan),
            gravity_tilt_deg=float(geo["gravity_tilt"]),
            gravity_ambiguous=bool(geo["gravity_ambiguous"]),
            wall_coverage=float(geo["wall_coverage"]),
            irradiance_polarity_corr=float(geo["irradiance_polarity_corr"]),
            max_model_vs_depth_deg=float(max(disagree)) if disagree else 0.0,
        ),
        shading=dict(scale=SHADING_SCALE),
        pattern=dict(default_width_m=PATTERN_W_M),
    )


def _build_maps_png(geo):
    """
    Pack region id (R) and alpha (B), 8-bit -- see SCHEMA.md.

    Shading is NOT packed here. P3-T3 measured 8-bit shading quantisation
    causing a max composited-render delta of up to 3/255 in the smooth wall
    interior (encoding_budget.json), over the plan's 2/255 banding gate --
    so shading ships as its own 16-bit PNG (shading.png) instead of this
    file's G channel. G is left at 0, reserved.
    """
    H, W = geo["H"], geo["W"]
    n_regions = len(geo["regions"])
    if n_regions > 254:
        raise ValueError(f"{n_regions} regions exceeds the 254 the R channel can address")
    out = np.zeros((H, W, 3), np.uint8)
    region_ids = geo["region_labels"]
    out[..., 2] = np.where(region_ids >= 0, region_ids + 1, 0).astype(np.uint8)  # R (BGR: idx2)
    out[..., 0] = np.round(np.clip(geo["alpha"], 0, 1) * 255).astype(np.uint8)   # B (BGR: idx0)
    return out


def _build_shading_png(geo):
    """16-bit single-channel shading map. Decode: shading = (v/65535)*scale."""
    sh = np.clip(geo["shading"], 0, SHADING_SCALE) / SHADING_SCALE
    return np.round(sh * 65535).astype(np.uint16)


def emit_scene(sample_name, out_dir, max_width=None):
    """Run the pipeline and write scene.json + maps.png + shading.png + photo.jpg."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    geo = _load_geometry(sample_name, max_width)
    scene = _build_scene_dict(sample_name, geo, repo_root)
    maps_bgr = _build_maps_png(geo)
    shading16 = _build_shading_png(geo)

    (out_dir / "scene.json").write_text(json.dumps(scene, indent=2) + "\n")
    cv2.imwrite(str(out_dir / "maps.png"), maps_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    cv2.imwrite(str(out_dir / "shading.png"), shading16, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    cv2.imwrite(str(out_dir / "photo.jpg"),
                cv2.cvtColor(geo["rgb"], cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return out_dir


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("sample")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-width", type=int, default=None)
    a = ap.parse_args()
    out = Path(a.out) if a.out else Path(__file__).parent / "outputs" / a.sample
    emit_scene(a.sample, out, max_width=a.max_width)
    print(f"wrote {out}/scene.json + maps.png + photo.jpg")
