"""
P3-T3: measure the bundle's precision budget before choosing formats.

Three encoding choices are being made silently in emit.py and are measured
here rather than assumed:

  1. Shading at 8 bits (maps.png's G channel). Step size is scale/255 --
     smooth wall gradients are exactly where 8-bit banding would show.
  2. Photo as JPEG. Recompression loses detail the user will compare against.
  3. Bundle size at whatever resolution emit.py happens to produce.

No gate here -- this is a measurement task. It produces a decision, which is
written into SCHEMA.md's "Encoding budget" section, not enforced by a test.

Usage:
    python3 scene/experiments/encoding_budget.py
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from run_all import BATCH, PATTERN_W_M  # noqa: E402
from render import sample_texture, composite  # noqa: E402
from scene.emit import _load_geometry, SHADING_SCALE  # noqa: E402

OUT = Path(__file__).parent / "encoding_budget.json"
JPEG_QUALITIES = [100, 95, 92, 85]
PSNR_GATE_DB = 40.0


def all_samples():
    return sorted(p.name for p in BATCH.iterdir() if p.is_dir())


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return 10.0 * np.log10(255.0 ** 2 / mse)


def quantize_shading(shading, scale, bits):
    levels = 2 ** bits - 1
    q = np.round(np.clip(shading, 0, scale) / scale * levels)
    return (q / levels) * scale


def measure_shading_precision(shading, wall):
    """8-bit vs 16-bit round-trip error, and banding-level count, within the wall."""
    out = {}
    for bits in (8, 16):
        decoded = quantize_shading(shading, SHADING_SCALE, bits)
        err = np.abs(decoded - shading)[wall]
        levels = 2 ** bits - 1
        q_int = np.round(np.clip(shading, 0, SHADING_SCALE) / SHADING_SCALE * levels)
        n_levels_used = int(len(np.unique(q_int[wall])))
        out[f"{bits}bit"] = dict(
            max_err=float(err.max()) if err.size else 0.0,
            rms_err=float(np.sqrt(np.mean(err ** 2))) if err.size else 0.0,
            distinct_levels_used=n_levels_used,
            levels_available=levels + 1,
        )
    return out


def measure_render_delta(uv, labels, shading, wall, alpha, rgb, tex_rgb):
    """
    PSNR / max channel delta between compositing with float shading vs 8-bit,
    restricted to the smooth wall INTERIOR (alpha > 0.99). Feathered edge
    pixels blend photo and texture and show unrelated compositing deltas at
    the seam -- not banding -- so including them would measure the wrong
    thing against the plan's ">2/255 in smooth regions" gate.
    """
    sampled, _ = sample_texture(uv, labels, tex_rgb, PATTERN_W_M)
    ref = composite(rgb, sampled, wall, shading)
    sh8 = quantize_shading(shading, SHADING_SCALE, 8)
    test = composite(rgb, sampled, wall, sh8)
    interior = wall & (alpha > 0.99)
    diff = np.abs(ref[interior].astype(int) - test[interior].astype(int))
    return dict(
        psnr_db=psnr(ref[interior], test[interior]),
        max_channel_delta=int(diff.max()) if diff.size else 0,
    )


def measure_jpeg(rgb_bgr):
    native_size = rgb_bgr.nbytes
    out = {}
    for q in JPEG_QUALITIES:
        ok, buf = cv2.imencode(".jpg", rgb_bgr, [cv2.IMWRITE_JPEG_QUALITY, q])
        decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        out[str(q)] = dict(
            size_kb=len(buf) / 1024,
            psnr_db=psnr(rgb_bgr, decoded),
        )
    return out, native_size


def measure_sample(name, tex_rgb):
    geo = _load_geometry(name, max_width=None)
    shading_prec = measure_shading_precision(geo["shading"], geo["labels"] >= 0)
    render_delta = measure_render_delta(
        geo["uv"], geo["labels"], geo["shading"], geo["labels"] >= 0,
        geo["alpha"], geo["rgb"], tex_rgb,
    )
    rgb_bgr = cv2.cvtColor(geo["rgb"], cv2.COLOR_RGB2BGR)
    jpeg, native_size = measure_jpeg(rgb_bgr)

    maps_png_8bit_size = len(cv2.imencode(
        ".png",
        np.stack([
            np.round(np.clip(geo["alpha"], 0, 1) * 255).astype(np.uint8),
            np.round(np.clip(geo["shading"], 0, SHADING_SCALE)
                      / SHADING_SCALE * 255).astype(np.uint8),
            np.where(geo["region_labels"] >= 0, geo["region_labels"] + 1, 0).astype(np.uint8),
        ], axis=-1),
        [cv2.IMWRITE_PNG_COMPRESSION, 6],
    )[1])

    return dict(
        sample=name,
        image_size=[int(geo["W"]), int(geo["H"])],
        shading_precision=shading_prec,
        render_delta_8bit_vs_float=render_delta,
        jpeg=jpeg,
        maps_png_kb=maps_png_8bit_size / 1024,
    )


def main():
    tex_bgr = cv2.imread(str(BATCH / "30106.jpg"))
    tex_rgb = cv2.cvtColor(tex_bgr, cv2.COLOR_BGR2RGB)

    results = []
    for name in all_samples():
        try:
            r = measure_sample(name, tex_rgb)
        except ValueError as e:
            print(f"  SKIP {name}: {e}")
            continue
        results.append(r)
        sp = r["shading_precision"]["8bit"]
        rd = r["render_delta_8bit_vs_float"]
        print(f"  {name:24s} 8-bit shading max_err={sp['max_err']:.4f} "
              f"({sp['distinct_levels_used']}/{sp['levels_available']} levels used) | "
              f"render delta {rd['max_channel_delta']}/255, {rd['psnr_db']:.1f} dB")

    max_render_delta = max(r["render_delta_8bit_vs_float"]["max_channel_delta"] for r in results)
    chosen_shading_bits = 16 if max_render_delta > 2 else 8

    # lowest JPEG quality with PSNR >= 40 dB, worst-case (min) across samples
    chosen_q, chosen_psnr = None, None
    for q in JPEG_QUALITIES:
        worst_psnr = min(r["jpeg"][str(q)]["psnr_db"] for r in results)
        if worst_psnr >= PSNR_GATE_DB:
            chosen_q, chosen_psnr = q, worst_psnr
    if chosen_q is None:
        chosen_q, chosen_psnr = JPEG_QUALITIES[0], min(
            r["jpeg"][str(JPEG_QUALITIES[0])]["psnr_db"] for r in results)

    median_bundle_kb = float(np.median([
        r["maps_png_kb"] + r["jpeg"][str(chosen_q)]["size_kb"] + 3.0  # + ~scene.json
        for r in results
    ]))

    decision = dict(
        chosen_shading_bits=chosen_shading_bits,
        max_render_delta_8bit=max_render_delta,
        chosen_jpeg_quality=chosen_q,
        chosen_jpeg_psnr_db=chosen_psnr,
        median_bundle_kb_at_chosen_settings=median_bundle_kb,
    )
    print("\nDECISION:", json.dumps(decision, indent=2))

    OUT.write_text(json.dumps(dict(samples=results, decision=decision), indent=2))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
