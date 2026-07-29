"""Shared helpers for capture_baseline.py and compare_to_baseline.py."""

import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

from run_all import BATCH, OUT, run_isolated, find_inputs, fit_scale  # noqa: E402
from geometry import load_normals, WALL_CONF_T  # noqa: E402

BASELINE_DIR = OUT / "baseline"
TEXTURE = BATCH / "30106.jpg"
TAG = "30106"

REQUIRED_KEYS = {
    "sample", "focal_px", "working_size", "gravity", "gravity_tilt_deg",
    "gravity_ambiguous", "planes", "wall_pair_angles_deg", "vertical_extent_m",
    "wall_coverage", "render_sha256",
}


def all_samples():
    return sorted(p.name for p in BATCH.iterdir() if p.is_dir())


def load_geometry_inputs(name):
    """
    Load and resample one sample's normals/mask/depth exactly as
    run_all.process() does, without running the rest of the pipeline. Used by
    tests that need direct access to intermediate arrays. Returns None if the
    sample is missing depth/focal, or dict(normals, mask, depth, f, cx, cy, scale).
    """
    g = find_inputs(name)
    if g is None or g["depth"] is None or g["focal"] is None:
        return None
    normals, _ = load_normals(g["normal"], verbose=False)
    normals = normals.astype(np.float32)
    conf = np.load(g["conf"])
    depth = np.load(g["depth"]).astype(np.float32)
    f_full = float(np.load(g["focal"]))

    s = fit_scale(depth.shape)
    if s < 1.0:
        H2, W2 = int(round(depth.shape[0] * s)), int(round(depth.shape[1] * s))
        normals = cv2.resize(normals, (W2, H2), interpolation=cv2.INTER_NEAREST)
        normals /= (np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-12)
        conf = cv2.resize(conf, (W2, H2), interpolation=cv2.INTER_AREA)
        depth = cv2.resize(depth, (W2, H2), interpolation=cv2.INTER_NEAREST)
    H, W = depth.shape
    f = f_full * s
    cx, cy = W / 2.0, H / 2.0
    mask = conf > WALL_CONF_T
    return dict(normals=normals, mask=mask, depth=depth, f=f, cx=cx, cy=cy, scale=s)


def render_one(name):
    """
    Run one sample in an isolated subprocess and shape its diagnostics into
    the baseline schema. Returns (baseline_dict, error_string_or_None).
    """
    r, err = run_isolated(name, TEXTURE, TAG, save_steps=False)
    if err:
        return None, err
    diag = r["diag"]
    final_png = Path(r["dir"]) / "final.png"
    render_sha256 = hashlib.sha256(final_png.read_bytes()).hexdigest()

    planes = [
        dict(
            plane_id=pid,
            normal=p["normal"],
            d=p["distance_m"],
            px_share=p["px_share"],
            extent_m=p["extent_m"],
            model_vs_depth_deg=p["model_vs_depth_normal_deg"],
        )
        for pid, p in sorted(diag["planes"].items(), key=lambda kv: int(kv[0]))
    ]

    baseline = dict(
        sample=name,
        focal_px=diag["focal_px_used"],
        working_size=diag["working_size"],
        gravity=diag["gravity"],
        gravity_tilt_deg=diag["gravity_tilt_from_image_vertical_deg"],
        gravity_ambiguous=diag["gravity_ambiguous_single_wall"],
        planes=planes,
        wall_pair_angles_deg=diag["wall_pair_angles_deg"],
        vertical_extent_m=diag["reconstructed_vertical_extent_m"],
        irradiance_polarity_corr=diag.get("irradiance_polarity_corr"),
        wall_coverage=diag["wall_coverage"],
        render_sha256=render_sha256,
    )
    return baseline, None
