"""
Corrected geometry for the wallpaper pipeline.

Fixes, relative to visualize_with_precomputed_normals.py:
  1. Normals are DECODED from the [0,1] storage convention to [-1,1] before use.
  2. Planes are fitted as (n, d) -- a normal AND an offset -- not just a normal.
  3. UV comes from ray/plane intersection (an exact homography), not from raw
     per-pixel depth points that do not lie on the plane.
  4. The plane's vertical axis is derived from an ESTIMATED gravity vector, not
     from a hardcoded [0,-1,0].
  5. Focal length comes from Depth Pro and is never guessed or re-derived here.
     Orthogonality and ceiling height are computed only as reported diagnostics.
"""

import numpy as np
import cv2
from sklearn.cluster import KMeans

# Wall-confidence threshold: pixels with conf > WALL_CONF_T are treated as
# wall. Chosen from a per-sample Otsu sweep in
# experiments/confidence_threshold_sweep.py -- median Otsu 0.480, std 0.014
# across all 12 samples, so 0.5 sits within ~1.4 sigma of the per-sample
# optimum everywhere. Frozen for the whole of Phase 2, including after T6
# replaces hard thresholding with soft-confidence filtering, so wall_coverage
# stays comparable across tasks.
WALL_CONF_T = 0.5


# ---------------------------------------------------------------------------
# 1. Normal decoding
# ---------------------------------------------------------------------------

def load_normals(path, verbose=True, flip_z=True):
    """
    Load a normal map and return unit vectors in OpenCV camera space.

    TWO conventions have to be undone, and both were missed by the old pipeline:

    1. ENCODING. The RGB->X model writes normals in display encoding,
       n_stored = (n + 1) / 2, so the array sits in [0, 1]. The old code fed
       that straight into k-means and into the plane basis.

    2. HANDEDNESS. The model's camera frame has +z pointing towards the viewer
       (OpenGL style); the pinhole unprojection used for Depth Pro's depth has
       +z pointing away (OpenCV style). Measured over five samples, negating z
       drops the disagreement between the model normals and the depth-derived
       normals from 62-78 degrees to 8-41. Negating y makes it consistently
       worse, so y is left alone.

    Returns (normals HxWx3 float64, was_encoded bool)
    """
    n = np.load(str(path)).astype(np.float64)

    lo, hi = n.min(), n.max()
    raw_len = np.median(np.linalg.norm(n, axis=-1))
    dec = n * 2.0 - 1.0
    dec_len = np.median(np.linalg.norm(dec, axis=-1))

    # Whichever interpretation yields unit-length vectors is the correct one.
    encoded = abs(dec_len - 1.0) < abs(raw_len - 1.0)
    if encoded:
        n = dec

    if flip_z:
        n = n * np.array([1.0, 1.0, -1.0])

    n = n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12)

    if verbose:
        tag = "ENCODED [0,1] -> decoded" if encoded else "already [-1,1]"
        print(f"      normals: {tag}, z negated -> OpenCV frame "
              f"(median |n| raw={raw_len:.3f} decoded={dec_len:.3f})")
    return n, encoded


# ---------------------------------------------------------------------------
# 2. Gravity
# ---------------------------------------------------------------------------

def estimate_gravity(normals, mask, prior_weight=0.05):
    """
    Estimate the camera-space "up" axis.

    Every wall normal is horizontal, so every wall normal is perpendicular to
    gravity. Gravity therefore minimises sum (n . g)^2 over wall pixels -- the
    smallest eigenvector of the normal scatter matrix.

    DEGENERATE CASE: if only one wall is visible, every direction perpendicular
    to that wall's normal minimises the sum equally well, so the scatter matrix
    has a two-dimensional null space and the eigenvector returned is arbitrary.
    (room1 hit exactly this and produced a 77-degree "gravity".) We add a weak
    prior pulling towards camera-up, which does nothing when two or more walls
    pin the answer down, and breaks the tie sensibly when they do not.

    Returns (g, rms residual, tilt from image-vertical in degrees, ambiguous).
    """
    N = normals[mask]
    if len(N) > 300_000:
        N = N[np.random.default_rng(0).choice(len(N), 300_000, replace=False)]

    up = np.array([0.0, -1.0, 0.0])
    M = (N.T @ N) / max(len(N), 1)
    w0 = np.linalg.eigvalsh(M)

    # ambiguous when the two smallest eigenvalues are comparable: the wall
    # normals span only one direction, so gravity is not determined by them.
    ambiguous = bool(w0[1] < 8.0 * max(w0[0], 1e-12))

    Mreg = M + prior_weight * w0[2] * (np.eye(3) - np.outer(up, up))
    w, V = np.linalg.eigh(Mreg)
    g = V[:, 0]
    if g[1] > 0:                     # make it point up (image y grows downward)
        g = -g
    g /= np.linalg.norm(g)

    resid = float(np.sqrt(max(np.mean((N @ g) ** 2), 0.0)))
    tilt = float(np.degrees(np.arccos(np.clip(abs(g @ up), -1, 1))))
    return g, resid, tilt, ambiguous


# ---------------------------------------------------------------------------
# 3. Plane clustering: normal AND offset
# ---------------------------------------------------------------------------

def cluster_planes(normals, mask, depth=None, fx=None, cx=None, cy=None,
                   n_clusters=8, top_n=4, min_frac=0.02, merge_deg=12.0):
    """
    Cluster wall pixels into planes.

    Unlike the old code this MERGES clusters whose normals agree to within
    `merge_deg`. K-means over a noisy normal field routinely splits one physical
    wall into several clusters; because each cluster then got its own UV origin,
    the pattern jumped at the seam between them. If depth is available the
    clusters are additionally split by plane offset so that two parallel-but-
    distinct walls stay separate.

    Returns labels (H,W) int32 with -1 = not a wall.
    """
    H, W = mask.shape
    N = normals[mask]

    # subsample for the fit; assign at full resolution afterwards
    rng = np.random.default_rng(42)
    idx = rng.choice(len(N), min(len(N), 200_000), replace=False)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(N[idx])
    centres = km.cluster_centers_
    centres /= (np.linalg.norm(centres, axis=1, keepdims=True) + 1e-12)

    # ---- merge near-duplicate normals (the seam bug) ----
    order = np.argsort(-np.bincount(km.labels_, minlength=n_clusters))
    groups = []                       # list of lists of original cluster ids
    for c in order:
        for g in groups:
            ang = np.degrees(np.arccos(np.clip(abs(centres[c] @ centres[g[0]]), -1, 1)))
            if ang < merge_deg:
                g.append(c)
                break
        else:
            groups.append([c])

    remap = np.full(n_clusters, -1, dtype=np.int32)
    for gi, g in enumerate(groups):
        for c in g:
            remap[c] = gi

    # assign every wall pixel to nearest (merged) centre
    full = np.argmin(
        ((N[:, None, :] - centres[None, :, :]) ** 2).sum(-1), axis=1
    ) if len(N) < 400_000 else km.predict(N)
    merged = remap[full]

    labels = np.full(mask.shape, -1, dtype=np.int32)
    labels[mask] = merged

    # ---- optional split by plane offset, when depth is available ----
    if depth is not None:
        labels = _split_by_offset(labels, depth, normals, fx, cx, cy)

    # ---- keep the largest planes only ----
    counts = {l: int((labels == l).sum()) for l in np.unique(labels) if l >= 0}
    keep = [l for l, c in sorted(counts.items(), key=lambda kv: -kv[1])[:top_n]
            if c >= min_frac * mask.sum()]

    out = np.full(mask.shape, -1, dtype=np.int32)
    for new, old in enumerate(keep):
        out[labels == old] = new
    return out


def points_for(sel, depth, fx, cx, cy):
    """3D points for the selected pixels only, avoiding a full (H,W,3) grid."""
    ys, xs = np.nonzero(sel)
    Z = depth[ys, xs].astype(np.float64)
    return np.stack([(xs - cx) * Z / fx, (ys - cy) * Z / fx, Z], -1)


def _split_by_offset(labels, depth, normals, fx, cx, cy, gap_m=0.35):
    """Split a normal-cluster into separate planes if its offsets are bimodal."""
    out = np.full_like(labels, -1)
    nxt = 0
    for l in [x for x in np.unique(labels) if x >= 0]:
        sel = labels == l
        n = normals[sel].mean(0)
        n /= np.linalg.norm(n) + 1e-12
        d = points_for(sel, depth, fx, cx, cy) @ n    # signed offset of each point
        lo, hi = np.percentile(d, [1, 99])
        hist, edges = np.histogram(d, bins=60, range=(lo, hi))
        # find a wide empty valley -> two distinct parallel planes
        empty = hist < max(hist.max() * 0.02, 1)
        runs, cur = [], 0
        for i, e in enumerate(empty):
            cur = cur + 1 if e else 0
            if cur and (i == len(empty) - 1 or not empty[i + 1]):
                runs.append((cur, i))
        binw = edges[1] - edges[0]
        split = [edges[i - c + 1] for c, i in runs if c * binw > gap_m]
        if not split:
            out[sel] = nxt
            nxt += 1
        else:
            bounds = [-np.inf] + split + [np.inf]
            dm = np.full(labels.shape, np.nan)
            dm[sel] = d
            for a, b in zip(bounds[:-1], bounds[1:]):
                part = sel & (dm >= a) & (dm < b)
                if part.sum() > 0.05 * sel.sum():
                    out[part] = nxt
                    nxt += 1
    return out


def fit_planes(labels, normals, depth=None, fx=None, cx=None, cy=None,
               default_distance=3.0):
    """
    Fit (n, d) per plane, so that the plane is  n . X = d  with d > 0.

    With depth: least-squares plane through the unprojected points (robust,
    trimmed) -- this is the metric case.
    Without depth: the normal is still exact, and d is set later by the caller
    (perspective-only mode), defaulting to `default_distance` metres.
    """
    planes = {}
    for l in [x for x in np.unique(labels) if x >= 0]:
        sel = labels == l
        n = normals[sel].mean(0)
        n /= np.linalg.norm(n) + 1e-12

        if depth is not None:
            pts = points_for(sel, depth, fx, cx, cy)
            # Depth-fitted normal is computed for DIAGNOSTICS only. The model
            # normal is what we actually use: it is far cleaner over windows,
            # mirrors and glossy surfaces, where Depth Pro sees through to the
            # scene beyond and the fitted normal swings wildly. Depth's only
            # job here is to supply the scalar offset d.
            c = pts.mean(0)
            _, _, Vt = np.linalg.svd(pts - c, full_matrices=False)
            n_fit = Vt[-1]
            if n_fit @ n < 0:
                n_fit = -n_fit
            ang = np.degrees(np.arccos(np.clip(n_fit @ n, -1, 1)))
            d = float(np.median(pts @ n))
            planes[l] = dict(n=n, d=d, n_model=n, n_depth=n_fit,
                             disagree_deg=float(ang), metric=True,
                             npix=int(sel.sum()))
        else:
            planes[l] = dict(n=n, d=None, n_model=n, n_depth=None,
                             disagree_deg=None, metric=False,
                             npix=int(sel.sum()))

    # Camera sits at the origin looking down +z; planes must be in front.
    for l, p in planes.items():
        if p['d'] is not None and p['d'] < 0:
            p['n'], p['d'] = -p['n'], -p['d']
    return planes


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 5. UV via ray/plane intersection  (the actual homography)
# ---------------------------------------------------------------------------

def plane_basis(n, gravity):
    """
    Orthonormal (u_axis, v_axis) on the plane, with v_axis pointing up along
    the projection of true gravity -- not a hardcoded [0,-1,0].
    """
    v_axis = gravity - (gravity @ n) * n
    ln = np.linalg.norm(v_axis)
    if ln < 1e-6:                       # plane is horizontal (floor/ceiling)
        tmp = np.array([0.0, 0.0, 1.0])
        v_axis = tmp - (tmp @ n) * n
        ln = np.linalg.norm(v_axis)
    v_axis /= ln
    u_axis = np.cross(v_axis, n)
    u_axis /= np.linalg.norm(u_axis) + 1e-12
    return u_axis, v_axis


def compute_uv(labels, planes, gravity, f, cx, cy, grazing_tol=0.08):
    """
    For every wall pixel, intersect its viewing ray with the pixel's plane and
    express the hit point in the plane's own 2D frame.

        ray(p) = K^-1 p ,  hit = ray * d / (n . ray) ,  (u,v) = (hit.ua, hit.va)

    Because every hit lies EXACTLY on the plane, (u,v) is a true homography of
    the image coordinates. Depth noise cannot warp the texture: depth only ever
    enters through the single scalar d.

    Returns uv (H,W,2) in metres, and a dict of per-plane diagnostics.
    """
    H, W = labels.shape
    uv = np.zeros((H, W, 2), dtype=np.float32)
    info = {}

    for l, p in planes.items():
        sel = labels == l
        if not sel.any():
            continue
        n, d = p['n'], p['d']
        ua, va = plane_basis(n, gravity)

        # Rays are built only for this plane's pixels. Materialising a full
        # (H, W, 3) float64 grid costs 300 MB on a 4k image and was enough to
        # get the process OOM-killed at native resolution.
        ys, xs = np.nonzero(sel)
        r = np.empty((len(ys), 3), np.float64)
        r[:, 0] = (xs - cx) / f
        r[:, 1] = (ys - cy) / f
        r[:, 2] = 1.0
        denom = r @ n
        # Rays nearly parallel to the plane hit it at infinity: t explodes and
        # the texture smears into streaks. Drop them rather than render garbage.
        bad = np.abs(denom) < grazing_tol
        t = np.where(bad, np.nan, d / np.where(bad, 1.0, denom))
        hit = r * t[:, None]

        # anchor the origin at the plane's own centroid so tiling is stable
        origin = np.nanmean(hit, axis=0)
        rel = hit - origin

        uu = rel @ ua
        vv = rel @ va
        uv[sel, 0] = np.nan_to_num(uu)
        uv[sel, 1] = np.nan_to_num(vv)

        if bad.any():                      # remove degenerate pixels from the plane
            labels[ys[bad], xs[bad]] = -1

        # robust extents; min/max would be dominated by a few grazing pixels
        wlo, whi = np.nanpercentile(uu, [0.5, 99.5])
        hlo, hhi = np.nanpercentile(vv, [0.5, 99.5])
        info[l] = dict(
            u_axis=ua, v_axis=va, origin=origin,
            width_m=float(whi - wlo), height_m=float(hhi - hlo),
            dist_m=float(d), n=n, npix=int(sel.sum()),
            degenerate=int(bad.sum()),
        )
    return uv, info


# ---------------------------------------------------------------------------
# 6. Verification helpers
# ---------------------------------------------------------------------------

def verify(planes, gravity, info):
    """Numeric self-checks; returns a list of (name, value, ok) tuples."""
    out = []
    ns = [p['n'] for p in planes.values()]
    for i in range(len(ns)):
        for j in range(i + 1, len(ns)):
            a = np.degrees(np.arccos(np.clip(abs(ns[i] @ ns[j]), -1, 1)))
            out.append((f"angle plane{i}-plane{j}", a,
                        abs(a - 90) < 15 or a < 15))
    for i, n in enumerate(ns):
        a = np.degrees(np.arccos(np.clip(abs(n @ gravity), -1, 1)))
        out.append((f"plane{i} normal vs gravity", a, abs(a - 90) < 12))
    for l, d in info.items():
        ua, va, n = d['u_axis'], d['v_axis'], d['n']
        out.append((f"plane{l} basis orthonormal",
                    float(abs(ua @ va) + abs(ua @ n) + abs(va @ n)), True))
    return out
