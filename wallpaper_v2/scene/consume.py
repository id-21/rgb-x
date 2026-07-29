"""
P3-T5: reference consumer -- render using ONLY the bundle.

Hard rule: this module may not import geometry.py, may not read any .npy,
and may not touch the depth map, normal map or irradiance. It reimplements
ray/plane intersection from the emitted plane parameters alone, proving the
bundle (scene.json + maps.png + shading.png + photo.jpg) is sufficient on
its own. See scene/tests/test_consumer.py for the isolation test that
enforces this, and SCHEMA.md for the bundle's field conventions.

This is also the oracle Phase 4's WebGL renderer will be checked against --
because this consumer, like the browser, sees only the bundle.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from render import sample_texture, srgb_to_linear, linear_to_srgb  # noqa: E402

GRAZING_TOL = 0.08     # matches geometry.compute_uv's grazing_tol


def load_bundle(bundle_dir):
    bundle_dir = Path(bundle_dir)
    scene = json.loads((bundle_dir / "scene.json").read_text())
    maps = cv2.imread(str(bundle_dir / "maps.png"))
    shading16 = cv2.imread(str(bundle_dir / "shading.png"), cv2.IMREAD_UNCHANGED)
    photo_bgr = cv2.imread(str(bundle_dir / "photo.jpg"))
    if scene is None or maps is None or shading16 is None or photo_bgr is None:
        raise ValueError(f"{bundle_dir}: incomplete bundle")
    return scene, maps, shading16, photo_bgr


def _reconstruct_plane_labels(scene, maps):
    """
    labels[y,x] = plane id, or -1. Rebuilt from maps.png's region-id channel
    (R) plus scene.json's region->plane mapping -- production always renders
    by PLANE (see geometry.extract_regions' docstring: a plane's UV is
    shared across all its regions), so this reconstructs the same grouping
    a consumer with no other source of plane membership can recover.
    """
    H, W = maps.shape[:2]
    region_id_map = maps[..., 2].astype(np.int32) - 1     # R channel, BGR idx 2; -1 = none
    plane_of_region = {r["id"]: r["plane"] for r in scene["regions"]}
    labels = np.full((H, W), -1, np.int32)
    for rid, pid in plane_of_region.items():
        labels[region_id_map == rid] = pid
    return labels


def _ray_plane_uv(scene, labels):
    """
    ray(p) = K^-1 p ,  hit = ray * d / (n . ray) ,  (u,v) = (hit.ua, hit.va)

    Reimplements geometry.compute_uv's homography using only the plane
    parameters emitted in scene.json -- no depth, no normal map.
    """
    H, W = labels.shape
    cam = scene["camera"]
    f, cx, cy = cam["f_px"], cam["cx"], cam["cy"]
    uv = np.zeros((H, W, 2), np.float32)

    for plane in scene["planes"]:
        pid = plane["id"]
        sel = labels == pid
        ys, xs = np.nonzero(sel)
        if len(ys) == 0:
            continue
        n = np.array(plane["normal"], np.float64)
        d = float(plane["d"])
        ua = np.array(plane["u_axis"], np.float64)
        va = np.array(plane["v_axis"], np.float64)
        origin = np.array(plane["origin"], np.float64)

        r = np.empty((len(ys), 3), np.float64)
        r[:, 0] = (xs - cx) / f
        r[:, 1] = (ys - cy) / f
        r[:, 2] = 1.0
        denom = r @ n
        bad = np.abs(denom) < GRAZING_TOL
        t = np.where(bad, np.nan, d / np.where(bad, 1.0, denom))
        hit = r * t[:, None]
        rel = hit - origin

        uv[ys, xs, 0] = np.nan_to_num(rel @ ua)
        uv[ys, xs, 1] = np.nan_to_num(rel @ va)
        if bad.any():
            labels[ys[bad], xs[bad]] = -1

    return uv


def render_from_bundle(bundle_dir, texture_path, pattern_width_m=None):
    """
    Render using ONLY scene.json + maps.png + shading.png + photo.jpg.

    Returns an (H, W, 3) uint8 RGB array.
    """
    scene, maps, shading16, photo_bgr = load_bundle(bundle_dir)
    labels = _reconstruct_plane_labels(scene, maps)
    uv = _ray_plane_uv(scene, labels)

    alpha = maps[..., 0].astype(np.float32) / 255.0         # B channel, BGR idx 0
    scale = scene["shading"]["scale"]
    shading = (shading16.astype(np.float32) / 65535.0) * scale

    if pattern_width_m is None:
        pattern_width_m = scene["pattern"]["default_width_m"]
    tex_bgr = cv2.imread(str(texture_path))
    tex_rgb = cv2.cvtColor(tex_bgr, cv2.COLOR_BGR2RGB)
    sampled_rgb, _wall = sample_texture(uv, labels, tex_rgb, pattern_width_m)

    # Linear-space composite (render.composite's math, without re-feathering
    # the mask -- `alpha` above IS the pipeline's own feathered mask, already
    # baked into the bundle, so re-deriving it from a hard boolean would not
    # reproduce the same edge blend).
    photo_rgb = cv2.cvtColor(photo_bgr, cv2.COLOR_BGR2RGB)
    photo_lin = srgb_to_linear(photo_rgb.astype(np.float32) / 255.0)
    tex_lin = srgb_to_linear(sampled_rgb.astype(np.float32) / 255.0)
    shaded_lin = np.clip(tex_lin * shading[..., None], 0.0, None)
    a = alpha[..., None]
    out_lin = photo_lin * (1 - a) + shaded_lin * a
    out = linear_to_srgb(np.clip(out_lin, 0.0, 1.0)) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle_dir")
    ap.add_argument("texture")
    ap.add_argument("--out", default="consumed.png")
    ap.add_argument("--pattern-width-m", type=float, default=None)
    a = ap.parse_args()
    rgb = render_from_bundle(a.bundle_dir, a.texture, a.pattern_width_m)
    cv2.imwrite(a.out, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print(f"wrote {a.out}")
