"""
Texture sampling and compositing.

Fixes, relative to the old pipeline:
  1. Trilinear mip-mapped sampling instead of nearest-neighbour integer indexing.
     At grazing angles a wall pattern compresses below one pixel; point sampling
     turns that into moire and speckle, which reads as "cheap" even when the
     geometry is perfect.
  2. Shading is taken from the irradiance model, not from the original image's
     LAB luminance. The old approach re-used the photo's own luminance, which
     baked the previous wall's sockets, artwork and shadows into the new texture.
  3. The wall mask is feathered, so edges are not staircased.
"""

import numpy as np
import cv2


# ---------------------------------------------------------------------------
# Mip pyramid
# ---------------------------------------------------------------------------

def build_mips(tex, levels=9):
    mips = [tex.astype(np.float32)]
    for _ in range(levels - 1):
        h, w = mips[-1].shape[:2]
        if min(h, w) <= 2:
            break
        mips.append(cv2.pyrDown(mips[-1]))
    return mips


def _bilinear(tex, x, y):
    """Wrapped bilinear fetch. x,y in texel units, tiling."""
    h, w = tex.shape[:2]
    xf = np.floor(x)
    yf = np.floor(y)
    fx = (x - xf).astype(np.float32)[:, None]
    fy = (y - yf).astype(np.float32)[:, None]
    x0 = xf.astype(np.int32)                 # int32: half the memory of int64
    y0 = yf.astype(np.int32)
    x0m, x1m = x0 % w, (x0 + 1) % w
    y0m, y1m = y0 % h, (y0 + 1) % h
    return ((tex[y0m, x0m] * (1 - fx) + tex[y0m, x1m] * fx) * (1 - fy) +
            (tex[y1m, x0m] * (1 - fx) + tex[y1m, x1m] * fx) * fy)


def _trilinear(mips, x, y, lod):
    """Trilinear fetch for a flat array of coords with per-sample LOD."""
    out = np.zeros((len(x), 3), np.float32)
    l0 = np.floor(lod).astype(np.int32)
    frac = (lod - l0)[:, None]
    for lev in np.unique(l0):
        m = l0 == lev
        s = 2.0 ** lev
        a = _bilinear(mips[lev], x[m] / s, y[m] / s)
        lev2 = min(lev + 1, len(mips) - 1)
        s2 = 2.0 ** lev2
        b = _bilinear(mips[lev2], x[m] / s2, y[m] / s2)
        out[m] = a * (1 - frac[m]) + b * frac[m]
    return out


def sample_texture(uv, labels, tex, pattern_width_m, pattern_height_m=None,
                   rotate_deg=0.0, max_aniso=16, chunk=600_000, lod_bias=-0.5):
    """
    Sample a tiling texture using UV in metres, with ANISOTROPIC filtering.

    A wall receding from the camera is compressed far more along one screen axis
    than the other. Isotropic mip-mapping has to pick a single blur radius, so it
    blurs the un-compressed axis just as hard as the compressed one and the
    pattern turns to mush. Here the texel footprint is treated as an ellipse: the
    mip level is chosen from the MINOR axis, and up to `max_aniso` taps are
    averaged along the MAJOR axis. Detail along the sharp axis survives, and the
    compressed axis is still correctly averaged so nothing aliases.

    uv must be the CONTINUOUS (un-wrapped) plane coordinate, so that its screen-
    space derivative gives the true texel footprint per pixel.
    """
    H, W = labels.shape
    Ph, Pw = tex.shape[:2]
    if pattern_height_m is None:
        pattern_height_m = pattern_width_m * (Ph / Pw)

    mips = build_mips(tex)
    maxlod = len(mips) - 1.001
    out = np.zeros((H, W, 3), np.float32)
    wall = labels >= 0
    ct, st = np.cos(np.radians(rotate_deg)), np.sin(np.radians(rotate_deg))

    for l in [x for x in np.unique(labels) if x >= 0]:
        sel = labels == l
        ys, xs = np.where(sel)
        if len(ys) == 0:
            continue
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        sub = sel[y0:y1, x0:x1]

        # Texture coordinates are built per plane, inside its bounding box.
        # Two full-image float arrays cost ~200 MB on a 4k image and were enough
        # to push the largest samples past the memory ceiling.
        U = uv[y0:y1, x0:x1, 0].astype(np.float32)
        V = uv[y0:y1, x0:x1, 1].astype(np.float32)
        if rotate_deg:
            U, V = U * ct - V * st, U * st + V * ct
        # metres -> texels. v is negated: plane-v points up, texture rows go down.
        TX = U / pattern_width_m * Pw
        TY = -V / pattern_height_m * Ph
        del U, V

        # Jacobian of texture coords w.r.t. screen coords
        dTXdy, dTXdx = np.gradient(TX)
        dTYdy, dTYdx = np.gradient(TY)

        lx = np.hypot(dTXdx, dTYdx)          # footprint along screen x
        ly = np.hypot(dTXdy, dTYdy)          # footprint along screen y
        major = np.maximum(lx, ly)
        minor = np.minimum(lx, ly)
        major = np.clip(major, 1e-4, None)
        minor = np.clip(minor, 1e-4, None)

        # direction of the major axis, in texture space, unit length
        use_x = lx >= ly
        mvx = np.where(use_x, dTXdx, dTXdy)
        mvy = np.where(use_x, dTYdx, dTYdy)
        mlen = np.hypot(mvx, mvy)
        mvx, mvy = mvx / (mlen + 1e-9), mvy / (mlen + 1e-9)

        ratio = np.clip(major / minor, 1.0, max_aniso)
        ntaps = np.clip(np.ceil(ratio).astype(np.int32), 1, max_aniso)
        # LOD from the minor axis; widening it by the tap count keeps energy right.
        # lod_bias sharpens by half a level: trilinear blending between mips is
        # inherently soft, and the anisotropic taps give enough headroom to pull
        # detail back without aliasing.
        lod = np.clip(np.log2(np.clip(major / ratio, 1e-4, None)) + lod_bias,
                      0, maxlod)

        acc = np.zeros(sub.shape + (3,), np.float32)
        for N in np.unique(ntaps[sub]):
            m = sub & (ntaps == N)
            k = int(m.sum())
            if not k:
                continue
            X, Y = TX[m], TY[m]
            L, MJ = lod[m], major[m]
            ux, uy = mvx[m], mvy[m]
            offs = (np.arange(N) + 0.5) / N - 0.5        # symmetric, centred

            # Chunked: a single plane on a 4k image can be several million
            # pixels, and each tap allocates a handful of same-length temporaries.
            # Chunking bounds peak memory independently of image size.
            res = np.empty((k, 3), np.float32)
            for a in range(0, k, chunk):
                b = min(a + chunk, k)
                tot = np.zeros((b - a, 3), np.float32)
                for o in offs:
                    d = o * MJ[a:b]
                    tot += _trilinear(mips, X[a:b] + ux[a:b] * d,
                                      Y[a:b] + uy[a:b] * d, L[a:b])
                res[a:b] = tot / N
            acc[m] = res

        region = out[y0:y1, x0:x1]
        region[sub] = acc[sub]
        out[y0:y1, x0:x1] = region
        del acc, TX, TY, dTXdx, dTXdy, dTYdx, dTYdy, lx, ly, major, minor

    return np.clip(out, 0, 255).astype(np.uint8), wall


# ---------------------------------------------------------------------------
# Shading
# ---------------------------------------------------------------------------

def _lum(x):
    x = x.astype(np.float32)
    if x.ndim == 3:
        x = x @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    return x


def irradiance_polarity(irradiance, rgb, mask, samples=200_000):
    """
    Correlation between the raw irradiance channel and observed wall brightness.

    Walls are close to uniform albedo, so observed luminance should track the
    light landing on them. NEGATIVE means the map is stored inverted: dark = more
    light. Measured over all 12 samples this is -0.47 to -0.97, i.e. always
    inverted -- hence `invert=True` below. This is reported per sample so the
    convention can never silently flip again.
    """
    if not mask.any():
        return 0.0
    e, l = _lum(irradiance)[mask], _lum(rgb.astype(np.float32) / 255.0)[mask]
    if len(e) > samples:
        i = np.random.default_rng(0).choice(len(e), samples, replace=False)
        e, l = e[i], l[i]
    if e.std() < 1e-8 or l.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(e, l)[0, 1])


def shading_from_irradiance(irradiance, mask, percentile=85, gamma=1.0,
                            invert=True):
    """
    Build a multiplicative shading field from the irradiance model.

    POLARITY: the model stores irradiance inverted -- DARK means MORE light, not
    less. Using it as-is lit the wallpaper backwards: bright near the window got
    darkened and shadowed corners got brightened. `invert` undoes that.

    A plain linear inversion is used deliberately. Fitting exp(-kE) or (1-E)^g
    against observed wall luminance improved R^2 by under 0.01 on average, and
    the best-fit exponent ranged from 0.4 to 2.6 across samples -- that is noise,
    not a systematic relationship, so the extra parameter is not worth it.

    Normalised so a well-lit part of the wall sits at 1.0: the wallpaper keeps
    its own albedo there and only darkens where the room's light falls off.
    """
    E = _lum(irradiance)
    if invert:
        E = 1.0 - E
    ref = np.percentile(E[mask], percentile) if mask.any() else 1.0
    s = E / (ref + 1e-6)
    if gamma != 1.0:
        s = np.power(np.clip(s, 0, None), gamma)
    return np.clip(s, 0.0, 1.6)


def shading_from_luminance(rgb, mask):
    """Fallback: the old LAB-luminance ratio, kept for comparison."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[..., 0]
    return np.clip(L / (L[mask].mean() + 1e-6), 0.0, 1.6)


def feather_mask(mask, radius=2):
    m = mask.astype(np.float32)
    k = 2 * radius + 1
    return cv2.GaussianBlur(m, (k, k), 0)


def srgb_to_linear(x):
    """sRGB-encoded [0,1] -> linear light [0,1]. Proper piecewise transfer
    function (IEC 61966-2-1), not a 2.2 gamma approximation."""
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    """Linear light [0,1] -> sRGB-encoded [0,1]. Inverse of srgb_to_linear."""
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)


def composite(original_rgb, texture_rgb, mask, shading, feather=2, linear=True):
    """
    Apply shading to the texture and blend it over the photo.

    linear=True (default): decode photo and texture from sRGB to linear
    light, multiply the texture by shading and alpha-blend there, then
    re-encode to sRGB. Light attenuation is a linear-space operation --
    multiplying sRGB-encoded values directly (linear=False, kept for A/B
    comparison) is not a physical model of attenuation and crushes midtones,
    making shaded regions read as muddy rather than dimmer.
    """
    a = feather_mask(mask, feather)[..., None]
    if not linear:
        tex = texture_rgb.astype(np.float32)
        shaded = np.clip(tex * shading[..., None], 0, 255)
        out = original_rgb.astype(np.float32) * (1 - a) + shaded * a
        return np.clip(out, 0, 255).astype(np.uint8)

    photo_lin = srgb_to_linear(original_rgb.astype(np.float32) / 255.0)
    tex_lin = srgb_to_linear(texture_rgb.astype(np.float32) / 255.0)
    shaded_lin = np.clip(tex_lin * shading[..., None], 0.0, None)
    out_lin = photo_lin * (1 - a) + shaded_lin * a
    out = linear_to_srgb(np.clip(out_lin, 0.0, 1.0)) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)
