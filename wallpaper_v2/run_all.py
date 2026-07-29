"""
Driver: run the corrected pipeline over every sample.

Outputs
  outputs/new/<sample>/           - new approach only, plus step diagnostics
  outputs/comparisons/<run>.png   - original | previous output | new output

Previous outputs are READ from the existing rgb-x_output tree, never recomputed.
"""

import gc
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from geometry import (load_normals, estimate_gravity, cluster_planes,
                      fit_planes, compute_uv, verify, WALL_CONF_T,
                      extract_regions, suggested_label)
from render import (sample_texture, shading_from_irradiance,
                    shading_from_luminance, composite, feather_mask,
                    irradiance_polarity)

ROOT = Path(__file__).resolve().parent.parent   # the rgb-x repo root
BATCH = ROOT / "rgb-x_output" / "batch_output"
OUT = Path(__file__).parent / "outputs"
MAXDIM = None          # None = no cap on the long edge
# Cap on total pixels. Native resolution is used whenever the image fits.
# Only room1 (13.5 MP) and no_walls_living (12.8 MP) exceed 10 MP; everything
# else renders at full native size. Raise this on a machine with more RAM --
# the pipeline itself has no resolution limit, this is purely a memory budget.
MAXPIXELS = 10_000_000
PATTERN_W_M = 0.53     # real-world width of one texture tile, in metres

# Previous runs, and which texture each used (recovered by matching the
# wallpaper panel in their saved 00_inputs.png).
PREVIOUS = [
    ("living_simple2",   "wallpaper_predicted_focal_length",           "30106.jpg"),
    ("bed_back_bedroom", "wallpaper_with_precomputed_normals_energie", "30106.jpg"),
    ("bed_back_bedroom", "wallpaper_with_precomputed_normals_botany",  "15513.jpg"),
    ("bed_back_bedroom", "wallpaper_with_precomputed_normals",         "__recover__"),
    ("bed_back_bedroom", "wallpaper_with_precomputed_normals_1",       "__recover__"),
]


# ---------------------------------------------------------------------------

def find_inputs(name):
    d = BATCH / name
    img = next((p for p in d.glob(f"{name}.jp*g")), None)
    if img is None:
        return None
    g = dict(name=name, dir=d, rgb=img,
             normal=d / f"{name}_normal.npy",
             conf=d / f"{name}_wall_confidence.npy",
             irr=d / f"{name}_irradiance.npy")
    # Prefer the exact <name>_depth.npy. bed_back_bedroom/ contains an older
    # bed_back_depth.npy under a different stem, and a bare glob picked between
    # them by alphabetical accident.
    dep = ([d / f"{name}_depth.npy"] if (d / f"{name}_depth.npy").exists()
           else sorted(d.glob("*depth*.npy")))
    foc = ([d / f"{name}_focal_length.npy"] if (d / f"{name}_focal_length.npy").exists()
           else sorted(d.glob("*focal*.npy")))
    # living_simple2's Depth Pro output lives in normal_irradiance/example
    alt = ROOT / "normal_irradiance" / "example"
    if not dep and (alt / f"{name}_depth.npy").exists():
        dep = [alt / f"{name}_depth.npy"]
    if not foc and (alt / f"{name}_focal_length.npy").exists():
        foc = [alt / f"{name}_focal_length.npy"]
    g['depth'] = dep[0] if dep else None
    g['focal'] = foc[0] if foc else None
    if not g['normal'].exists() or not g['conf'].exists():
        return None
    return g


def recover_texture(prev_dir):
    """Pull the wallpaper swatch out of a previous run's 00_inputs.png."""
    p = prev_dir / "00_inputs.png"
    if not p.exists():
        return None
    im = cv2.imread(str(p))
    H, W = im.shape[:2]
    panel = im[int(H * 0.06):int(H * 0.97), int(W * 0.795):int(W * 0.985)]
    g = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(g < 250)
    if len(ys) < 100:
        return panel
    return panel[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def fit_scale(shape, maxdim=MAXDIM, maxpixels=MAXPIXELS):
    H, W = shape[:2]
    s = 1.0
    if maxdim is not None:
        s = min(s, maxdim / max(H, W))
    if maxpixels is not None and H * W > maxpixels:
        s = min(s, (maxpixels / (H * W)) ** 0.5)
    return s


def process(name, texture_bgr, tag, save_steps=True):
    g = find_inputs(name)
    if g is None:
        return None, f"{name}: missing normal/mask"
    if g['depth'] is None:
        return None, f"{name}: no Depth Pro depth map"
    if g['focal'] is None:
        return None, f"{name}: no Depth Pro focal length (_focal_length.npy)"

    print(f"\n=== {name}  [{tag}] ===")
    rgb = cv2.cvtColor(cv2.imread(str(g['rgb'])), cv2.COLOR_BGR2RGB)
    normals, was_enc = load_normals(g['normal'])
    normals = normals.astype(np.float32)      # float64 costs 300 MB at 4k
    conf = np.load(g['conf'])
    depth = np.load(g['depth']).astype(np.float32)
    f_full = float(np.load(g['focal']))
    irr = np.load(g['irr']) if g['irr'].exists() else None

    # ---- resample everything consistently, rescaling f with it ----
    s = fit_scale(rgb.shape)
    if s < 1.0:
        H2, W2 = int(round(rgb.shape[0] * s)), int(round(rgb.shape[1] * s))
        rgb = cv2.resize(rgb, (W2, H2), interpolation=cv2.INTER_AREA)
        normals = cv2.resize(normals, (W2, H2), interpolation=cv2.INTER_NEAREST)
        normals /= (np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-12)
        conf = cv2.resize(conf, (W2, H2), interpolation=cv2.INTER_AREA)
        depth = cv2.resize(depth, (W2, H2), interpolation=cv2.INTER_NEAREST)
        if irr is not None:
            irr = cv2.resize(irr, (W2, H2), interpolation=cv2.INTER_AREA)
    H, W = depth.shape
    f = f_full * s          # focal length scales with the image, as do cx, cy
    cx, cy = W / 2.0, H / 2.0
    print(f"      f = {f_full:.1f} px at native, {f:.1f} px at {W}x{H} "
          f"({f/W:.2f} x width)" + ("" if s == 1.0 else f"  [scaled {s:.2f}x]"))

    mask = conf > WALL_CONF_T
    if mask.sum() < 0.01 * mask.size:
        return None, f"{name}: wall mask nearly empty ({100*mask.mean():.1f}%)"

    # ---- geometry ----
    grav, resid, tilt, grav_ambiguous = estimate_gravity(normals, mask)
    labels = cluster_planes(normals, mask, depth=depth, fx=f, cx=cx, cy=cy,
                            n_clusters=8, scale=s)
    planes = fit_planes(labels, normals, depth=depth, fx=f, cx=cx, cy=cy,
                        wall_px=int(mask.sum()))
    if not planes:
        return None, f"{name}: no planes fitted"
    uv, info = compute_uv(labels, planes, grav, f, cx, cy)

    # ---- diagnostics the user asked to see ----
    # extent along gravity, computed on a subsample without building a full
    # (H, W, 3) point cloud
    step = max(1, int(np.sqrt(H * W / 400_000)))
    ys, xs = np.mgrid[0:H:step, 0:W:step]
    Zs = depth[::step, ::step]
    along_g = (((xs - cx) / f) * grav[0] + ((ys - cy) / f) * grav[1] + grav[2]) * Zs
    ceiling_h = float(np.percentile(along_g, 99.5) - np.percentile(along_g, 0.5))
    del along_g, ys, xs, Zs
    checks = verify(planes, grav, info)
    angs = [v for k, v in [(c[0], c[1]) for c in checks] if k.startswith("angle")]
    wall_coverage = float((labels >= 0).sum() / max(mask.sum(), 1))

    region_labels, regions = extract_regions(labels)

    diag = dict(
        sample=name, texture=tag,
        normals_were_encoded=bool(was_enc),
        focal_px_native=f_full, focal_px_used=f, working_size=[int(W), int(H)],
        gravity=[float(x) for x in grav],
        gravity_tilt_from_image_vertical_deg=float(tilt),
        gravity_residual_rms=float(resid),
        gravity_ambiguous_single_wall=bool(grav_ambiguous),
        n_planes=len(planes),
        wall_pair_angles_deg=[float(a) for a in angs],
        reconstructed_vertical_extent_m=ceiling_h,
        wall_coverage=wall_coverage,
        planes={int(l): dict(
            normal=[float(x) for x in p['n']],
            distance_m=float(p['d']),
            model_vs_depth_normal_deg=(None if p['disagree_deg'] is None
                                       else float(p['disagree_deg'])),
            px_share=float(p['npix'] / max(mask.sum(), 1)),
            extent_m=[info[l]['width_m'], info[l]['height_m']],
            pattern_repeats_across=float(info[l]['width_m'] / PATTERN_W_M),
            grazing_px_dropped=int(info[l]['degenerate']),
            confidence=float(p['confidence']),
        ) for l, p in planes.items()},
        regions={int(r): dict(
            plane=reg['plane'],
            npix=reg['npix'],
            centroid_px=[float(x) for x in reg['centroid_px']],
            bbox_px=[int(x) for x in reg['bbox_px']],
            suggested_label=suggested_label(planes[reg['plane']]['n']),
        ) for r, reg in regions.items()},
    )
    print(f"      gravity tilt {tilt:.1f} deg"
          f"{' (AMBIGUOUS: one wall only)' if grav_ambiguous else ''} | {len(planes)} planes | "
          f"wall angles {['%.0f' % a for a in angs]} | "
          f"vertical extent {ceiling_h:.2f} m")

    # ---- render ----
    tex_rgb = cv2.cvtColor(texture_bgr, cv2.COLOR_BGR2RGB)
    sampled, wall = sample_texture(uv, labels, tex_rgb, PATTERN_W_M)

    if irr is not None:
        pol = irradiance_polarity(irr, rgb, wall)
        diag['irradiance_polarity_corr'] = float(pol)
        if pol > 0:
            print(f"      !! irradiance correlates POSITIVELY ({pol:+.3f}) with "
                  f"wall brightness - convention may have changed, check invert=")
        else:
            print(f"      irradiance polarity {pol:+.3f} (negative = stored "
                  f"inverted, as expected) -> inverting")
        sh_irr = shading_from_irradiance(irr, wall)
        res_irr = composite(rgb, sampled, wall, sh_irr)
    else:
        sh_irr, res_irr = None, None
    sh_lab = shading_from_luminance(rgb, wall)
    res_lab = composite(rgb, sampled, wall, sh_lab)

    primary = res_irr if res_irr is not None else res_lab
    diag['shading'] = 'irradiance' if res_irr is not None else 'lab_fallback'

    # ---- save ----
    od = OUT / "new" / f"{name}__{tag}"
    od.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(od / "final.png"), cv2.cvtColor(primary, cv2.COLOR_RGB2BGR))
    if res_lab is not None:
        cv2.imwrite(str(od / "final_lab_shading.png"),
                    cv2.cvtColor(res_lab, cv2.COLOR_RGB2BGR))
    if save_steps:
        save_diagnostics(od, rgb, normals, labels, uv, sampled,
                         sh_irr if sh_irr is not None else sh_lab, primary, mask,
                         region_labels=region_labels)
    (od / "diagnostics.json").write_text(json.dumps(diag, indent=2))
    del normals, depth, conf, irr, uv, labels, region_labels, sampled, sh_lab, sh_irr, primary
    gc.collect()
    return dict(diag=diag, dir=str(od)), None


# ---------------------------------------------------------------------------

def _colorise(labels):
    out = np.zeros(labels.shape + (3,), np.uint8)
    pal = np.array([[46, 134, 222], [255, 159, 67], [16, 172, 132],
                    [238, 82, 83], [162, 155, 254], [255, 206, 84],
                    [72, 219, 251], [255, 107, 129], [10, 189, 227],
                    [95, 39, 205], [201, 24, 74], [0, 148, 50]], np.uint8)
    for i, l in enumerate([x for x in np.unique(labels) if x >= 0]):
        out[labels == l] = pal[i % len(pal)]
    return out


def _norm8(a):
    a = np.nan_to_num(a.astype(np.float32))
    lo, hi = np.percentile(a, [1, 99])
    return np.clip((a - lo) / (hi - lo + 1e-8) * 255, 0, 255).astype(np.uint8)


def _shrink(im, maxw=1100):
    if im.shape[1] <= maxw:
        return im
    s = maxw / im.shape[1]
    return cv2.resize(im, (maxw, max(1, int(im.shape[0] * s))),
                      interpolation=cv2.INTER_AREA)


def save_diagnostics(od, rgb, normals, labels, uv, sampled, shading, final, mask,
                     region_labels=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nvis = ((normals + 1) * 127.5).astype(np.uint8).copy()
    nvis[~mask] = 30
    uvis = _norm8(np.where(labels >= 0, uv[..., 0], np.nan))
    vvis = _norm8(np.where(labels >= 0, uv[..., 1], np.nan))

    # These are diagnostics, not deliverables -- render them small. At native
    # 4k resolution nine imshow panels are enough to exhaust memory on their own.
    panels = [
        (rgb, "Original"),
        (nvis, "Normals (decoded, masked)"),
        (_colorise(labels), "Planes (merged)"),
        (_colorise(region_labels) if region_labels is not None else _colorise(labels),
         "Regions (selection)"),
        (uvis, "U (metres on plane)"),
        (vvis, "V (metres on plane)"),
        (sampled, "Texture via homography"),
        (_norm8(shading), "Shading field"),
        (final, "Final"),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(24, 15))
    for ax, (im, t) in zip(axes.ravel(), panels):
        im = _shrink(im)
        ax.imshow(im, cmap=None if im.ndim == 3 else "magma")
        ax.set_title(t, fontsize=13, fontweight="bold")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(od / "steps.png", dpi=110, bbox_inches="tight")
    plt.close()


def _label_strip(width, text, h=64):
    strip = np.full((h, width, 3), 245, np.uint8)
    cv2.putText(strip, text, (14, int(h * 0.66)), cv2.FONT_HERSHEY_SIMPLEX,
                0.95, (25, 25, 25), 2, cv2.LINE_AA)
    return strip


def make_comparison(name, prev_dir, new_png, out_path, label):
    """
    Original | previous | new, concatenated at NATIVE pixels.

    Built with cv2 rather than matplotlib: a matplotlib figure resamples every
    panel down to the figure's dpi, which was making the results look far softer
    than the actual renders.
    """
    orig = cv2.imread(str(next(BATCH.joinpath(name).glob(f"{name}.jp*g"))))
    prev = prev_dir / "final_result.png"
    if not prev.exists():
        return False
    prev_img = cv2.imread(str(prev))
    new_bgr = cv2.imread(str(new_png))

    h = max(orig.shape[0], prev_img.shape[0], new_bgr.shape[0])
    panels = []
    for im, txt in [(orig, "Original photo"),
                    (prev_img, f"Previous pipeline - {prev_dir.name}"),
                    (new_bgr, "Corrected pipeline")]:
        if im.shape[0] != h:
            im = cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h),
                            interpolation=cv2.INTER_LANCZOS4)
        panels.append(np.vstack([_label_strip(im.shape[1], txt), im]))

    gap = np.full((panels[0].shape[0], 16, 3), 245, np.uint8)
    sheet = panels[0]
    for p in panels[1:]:
        sheet = np.hstack([sheet, gap, p])
    header = _label_strip(sheet.shape[1], f"{name}   |   texture: {label}", 70)
    cv2.imwrite(str(out_path), np.vstack([header, sheet]),
                [cv2.IMWRITE_PNG_COMPRESSION, 3])
    return True


def make_overview(paths, out_path, panel_w=760):
    """Contact sheet: original vs new, one row per sample."""
    rows = []
    for name, final_png in paths:
        o = cv2.imread(str(next(BATCH.joinpath(name).glob(f"{name}.jp*g"))))
        f = cv2.imread(str(final_png))
        row = []
        for im in (o, f):
            sc = panel_w / im.shape[1]
            row.append(cv2.resize(im, (panel_w, int(im.shape[0] * sc)),
                                  interpolation=cv2.INTER_AREA))
        h = max(r.shape[0] for r in row)
        row = [np.vstack([r, np.full((h - r.shape[0], panel_w, 3), 245, np.uint8)])
               for r in row]
        band = np.hstack([row[0], np.full((h, 12, 3), 245, np.uint8), row[1]])
        rows.append(np.vstack([_label_strip(band.shape[1], name, 46), band,
                               np.full((18, band.shape[1], 3), 245, np.uint8)]))
    cv2.imwrite(str(out_path), np.vstack(rows),
                [cv2.IMWRITE_PNG_COMPRESSION, 3])


# ---------------------------------------------------------------------------

def _worker(args):
    """Runs in a fresh subprocess; see run_isolated."""
    name, tex_path, tag, save_steps, recover_from = args
    tex = (recover_texture(Path(recover_from)) if recover_from
           else cv2.imread(str(tex_path)))
    try:
        return process(name, tex, tag, save_steps=save_steps)
    except Exception as e:
        return None, f"{name}/{tag}: {type(e).__name__}: {e}"


def run_isolated(name, tex_path, tag, save_steps=True, recover_from=None):
    """
    Render one sample in its own process.

    Peak RSS for a 4k sample is ~2.3 GB. Run back to back in one process, the
    allocator does not return freed arenas to the OS and the third or fourth
    large sample gets OOM-killed. A fresh process per sample makes the peak
    per-sample rather than cumulative.
    """
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    with ctx.Pool(1, maxtasksperchild=1) as pool:
        return pool.apply(_worker, ((name, str(tex_path) if tex_path else None,
                                     tag, save_steps, recover_from),))


def main_one(name):
    """Render a single sample. Used by run_batch.sh, one OS process each."""
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "_diag").mkdir(exist_ok=True)
    r, err = _worker((name, str(BATCH / "30106.jpg"), "30106", True, None))
    if err:
        print(f"  !! SKIP {err}")
        (OUT / "_diag" / f"{name}.err").write_text(err)
        return 1
    (OUT / "_diag" / f"{name}.json").write_text(json.dumps(r['diag'], indent=2))
    return 0


def main_comparisons():
    """Build the three-way sheets, after all samples have been rendered."""
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "comparisons").mkdir(exist_ok=True)
    (OUT / "_diag").mkdir(exist_ok=True)
    made = []
    for name, prev_name, tex_name in PREVIOUS:
        prev_dir = BATCH / name / prev_name
        if not prev_dir.exists():
            continue
        if tex_name == "__recover__":
            if not (prev_dir / "00_inputs.png").exists():
                continue
            tex_path, recover_from = None, str(prev_dir)
            label = "recovered from previous run"
        else:
            tex_path, recover_from = BATCH / tex_name, None
            label = tex_name
        r, err = _worker((name, str(tex_path) if tex_path else None,
                          prev_name, False, recover_from))
        if err:
            print(f"  !! SKIP comparison {err}")
            (OUT / "_diag" / f"cmp_{name}_{prev_name}.err").write_text(err)
            continue
        outp = OUT / "comparisons" / f"{name}__{prev_name}.png"
        if make_comparison(name, prev_dir, Path(r['dir']) / "final.png",
                           outp, label):
            made.append(outp.name)
            print(f"  + {outp.name}")
    return made


def collect():
    """Gather per-sample diagnostics into summary.json and the overview sheet."""
    dg = OUT / "_diag"
    samples = [json.loads(p.read_text()) for p in sorted(dg.glob("*.json"))]
    failures = sorted({p.read_text() for p in dg.glob("*.err")})
    rendered = [(d['sample'], OUT / "new" / f"{d['sample']}__{d['texture']}" / "final.png")
                for d in samples]
    rendered = [(n, p) for n, p in rendered if p.exists()]
    if rendered:
        make_overview(rendered, OUT / "_overview_all_samples.png")
    made = sorted(p.name for p in (OUT / "comparisons").glob("*.png"))
    (OUT / "summary.json").write_text(json.dumps(
        dict(samples=samples, comparisons=made, failures=failures), indent=2))
    print(f"\n{'='*70}\nrendered {len(samples)} samples, {len(made)} comparisons")
    for f in failures:
        print("  SKIPPED:", f)
    print("=" * 70)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "comparisons").mkdir(exist_ok=True)

    samples = sorted(p.name for p in BATCH.iterdir() if p.is_dir())
    summary, failures, rendered = [], [], []

    # --- pass 1: every sample with the default texture ---
    for name in samples:
        r, err = run_isolated(name, BATCH / "30106.jpg", "30106")
        if err:
            print(f"  !! SKIP {err}")
            failures.append(err)
        else:
            summary.append(r['diag'])
            rendered.append((name, Path(r['dir']) / "final.png"))
        r = None
        gc.collect()

    if rendered:
        make_overview(rendered, OUT / "_overview_all_samples.png")

    # --- pass 2: comparisons against the saved previous runs ---
    made = []
    for name, prev_name, tex_name in PREVIOUS:
        prev_dir = BATCH / name / prev_name
        if not prev_dir.exists():
            continue
        if tex_name == "__recover__":
            if not (prev_dir / "00_inputs.png").exists():
                continue
            tex_path, recover_from = None, str(prev_dir)
            label = "recovered from previous run"
        else:
            tex_path, recover_from = BATCH / tex_name, None
            label = tex_name
        r, err = run_isolated(name, tex_path, prev_name, save_steps=False,
                              recover_from=recover_from)
        if err:
            print(f"  !! SKIP comparison {err}")
            failures.append(err)
            continue
        outp = OUT / "comparisons" / f"{name}__{prev_name}.png"
        if make_comparison(name, prev_dir, Path(r['dir']) / 'final.png', outp, label):
            made.append(str(outp.name))
        r = None
        gc.collect()

    (OUT / "summary.json").write_text(json.dumps(
        dict(samples=summary, comparisons=made, failures=failures), indent=2))

    print("\n" + "=" * 70)
    print(f"rendered {len(summary)} samples, {len(made)} comparisons")
    for f in failures:
        print("  SKIPPED:", f)
    print("=" * 70)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "--one":
        sys.exit(main_one(a[1]))
    elif a and a[0] == "--comparisons":
        main_comparisons()
    elif a and a[0] == "--collect":
        collect()
    else:
        main()
