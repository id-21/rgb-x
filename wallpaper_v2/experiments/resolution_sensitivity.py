"""
Does running the pipeline at reduced resolution change the answer?

Tests the claim in planning/web-service-strategy.md that depth, normals and
irradiance are low-frequency enough to compute at ~768 px.

SCOPE -- read this before trusting the output.

This downsamples the MODEL OUTPUTS we already have and re-runs the geometry.
That isolates one of the two questions:

  (1) SAMPLING DENSITY  -- is the geometry stable when fed fewer samples?
                           <-- this script answers that
  (2) MODEL BEHAVIOUR   -- does a network fed a 768 px image produce the same
                           depth/normals/focal as one fed a 3000 px image?
                           <-- this script CANNOT answer that; needs a GPU re-run

They are different questions and (2) is the riskier one, especially for focal
length, which is a single scalar prediction rather than an average over pixels.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from geometry import (load_normals, estimate_gravity, cluster_planes,
                      fit_planes, compute_uv, WALL_CONF_T)

ROOT = Path(__file__).resolve().parents[2]
BATCH = ROOT / "rgb-x_output" / "batch_output"
ALT = ROOT / "normal_irradiance" / "example"


def inputs(name):
    d = BATCH / name
    img = next((p for p in d.glob(f"{name}.jp*g")), None)
    nm, cf = d / f"{name}_normal.npy", d / f"{name}_wall_confidence.npy"
    ir = d / f"{name}_irradiance.npy"
    dep = sorted(d.glob("*depth*.npy")) or ([ALT / f"{name}_depth.npy"]
                                            if (ALT / f"{name}_depth.npy").exists() else [])
    foc = sorted(d.glob("*focal*.npy")) or ([ALT / f"{name}_focal_length.npy"]
                                            if (ALT / f"{name}_focal_length.npy").exists() else [])
    if not (img and nm.exists() and cf.exists() and dep and foc):
        return None
    return dict(img=img, nm=nm, cf=cf, ir=ir, dep=dep[0], foc=foc[0])


def geometry_at(normals, conf, depth, f, scale):
    """Run the geometry stack at a given scale factor; return plane params."""
    if scale != 1.0:
        H, W = depth.shape
        H2, W2 = int(round(H * scale)), int(round(W * scale))
        normals = cv2.resize(normals, (W2, H2), interpolation=cv2.INTER_NEAREST)
        normals /= (np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-12)
        conf = cv2.resize(conf, (W2, H2), interpolation=cv2.INTER_AREA)
        depth = cv2.resize(depth, (W2, H2), interpolation=cv2.INTER_NEAREST)
        f = f * scale
    H, W = depth.shape
    cx, cy = W / 2.0, H / 2.0
    mask = conf > WALL_CONF_T
    if mask.sum() < 0.01 * mask.size:
        return None
    grav, _, tilt, amb = estimate_gravity(normals, mask)
    labels = cluster_planes(normals, mask, depth=depth, fx=f, cx=cx, cy=cy,
                            n_clusters=8, top_n=4)
    planes = fit_planes(labels, normals, depth=depth, fx=f, cx=cx, cy=cy)
    _, info = compute_uv(labels, planes, grav, f, cx, cy)
    return dict(
        gravity=grav, tilt=tilt,
        planes=[dict(n=p['n'], d=p['d'],
                     share=p['npix'] / max(int(mask.sum()), 1),
                     width_m=info[l]['width_m'])
                for l, p in planes.items()],
        size=(W, H),
    )


def match(a, b):
    """Match planes between two runs by normal direction."""
    out, used = [], set()
    for i, pa in enumerate(a['planes']):
        best, bj = 181.0, None
        for j, pb in enumerate(b['planes']):
            if j in used:
                continue
            ang = np.degrees(np.arccos(np.clip(abs(pa['n'] @ pb['n']), -1, 1)))
            if ang < best:
                best, bj = ang, j
        if bj is not None and best < 25:
            used.add(bj)
            out.append((pa, b['planes'][bj], best))
        else:
            out.append((pa, None, None))
    return out


def irradiance_error(ir, mask, scale):
    """Downsample -> bilinear upsample, and measure where the error lands."""
    E = ir.astype(np.float32)
    if E.ndim == 3:
        E = E @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    H, W = E.shape
    small = cv2.resize(E, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA)
    back = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
    err = np.abs(E - back)
    gx, gy = np.gradient(E)
    g = np.hypot(gx, gy)
    m = mask & np.isfinite(err)
    if m.sum() < 100:
        return None
    hi = m & (g >= np.percentile(g[m], 90))     # sharp shadow edges
    lo = m & (g <= np.percentile(g[m], 50))     # smooth falloff
    return dict(rms_all=float(err[m].std()),
                rms_smooth=float(err[lo].std()),
                rms_edges=float(err[hi].std()),
                dynamic_range=float(np.percentile(E[m], 99) - np.percentile(E[m], 1)))


def main(target_long_edge=768):
    rows = []
    for d in sorted(BATCH.iterdir()):
        if not d.is_dir():
            continue
        g = inputs(d.name)
        if g is None:
            continue
        normals, _ = load_normals(g['nm'], verbose=False)
        normals = normals.astype(np.float32)
        conf = np.load(g['cf'])
        depth = np.load(g['dep']).astype(np.float32)
        f = float(np.load(g['foc']))
        H, W = depth.shape
        scale = target_long_edge / max(H, W)
        if scale >= 1.0:
            continue

        a = geometry_at(normals, conf, depth, f, 1.0)
        b = geometry_at(normals, conf, depth, f, scale)
        if a is None or b is None:
            continue

        pairs = match(a, b)
        ang = [p[2] for p in pairs if p[2] is not None]
        dd = [abs(p[0]['d'] - p[1]['d']) for p in pairs if p[1] is not None]
        ddp = [100 * abs(p[0]['d'] - p[1]['d']) / max(p[0]['d'], 1e-6)
               for p in pairs if p[1] is not None]
        lost = sum(1 for p in pairs if p[1] is None)
        smallest = min((p['share'] for p in a['planes']), default=0)

        ir = np.load(g['ir']) if g['ir'].exists() else None
        ierr = irradiance_error(ir, conf > 0.5, scale) if ir is not None else None

        rows.append(dict(
            sample=d.name, native=f"{W}x{H}", small=f"{b['size'][0]}x{b['size'][1]}",
            scale=scale,
            n_native=len(a['planes']), n_small=len(b['planes']), lost=lost,
            max_angle=max(ang) if ang else None,
            max_dd=max(dd) if dd else None, max_ddp=max(ddp) if ddp else None,
            gravity_shift=float(np.degrees(np.arccos(np.clip(abs(a['gravity'] @ b['gravity']), -1, 1)))),
            smallest_share=smallest,
            irr=ierr,
        ))

    # ---- report ----
    print(f"GEOMETRY AT NATIVE vs {target_long_edge}px LONG EDGE "
          f"(model outputs downsampled; models NOT re-run)\n")
    hdr = ("sample", "native", "small", "planes", "lost", "dNormal", "dD", "dD%", "dGrav")
    print("%-24s%12s%12s%9s%6s%9s%9s%7s%8s" % hdr)
    for r in rows:
        ang = "%.2f" % r["max_angle"] if r["max_angle"] is not None else "-"
        dd = "%.3f" % r["max_dd"] if r["max_dd"] is not None else "-"
        ddp = "%.1f%%" % r["max_ddp"] if r["max_ddp"] is not None else "-"
        print("%-24s%12s%12s%9s%6d%9s%9s%7s%7.2f" % (
            r["sample"], r["native"], r["small"],
            "%d/%d" % (r["n_native"], r["n_small"]), r["lost"],
            ang, dd, ddp, r["gravity_shift"]))

    print()
    print("%-24s%16s%13s%9s%9s%13s" % (
        "sample", "smallest plane", "irrRMS all", "smooth", "edges", "edges/range"))
    for r in rows:
        i = r["irr"]
        if i is None:
            continue
        print("%-24s%15.1f%%%13.4f%9.4f%9.4f%12.1f%%" % (
            r["sample"], 100 * r["smallest_share"], i["rms_all"],
            i["rms_smooth"], i["rms_edges"],
            100 * i["rms_edges"] / max(i["dynamic_range"], 1e-6)))

    Path(__file__).with_suffix(".json").write_text(
        json.dumps(rows, indent=2, default=float))
    return rows


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 768)
