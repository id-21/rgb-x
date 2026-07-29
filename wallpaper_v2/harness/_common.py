"""Shared helpers for capture_baseline.py and compare_to_baseline.py."""

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

from run_all import BATCH, OUT, run_isolated  # noqa: E402

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
