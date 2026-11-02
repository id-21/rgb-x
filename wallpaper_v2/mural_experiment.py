"""
Standalone experiment: place a mural as ONE continuous image across all 5
panels (living_room_foyer_panels sample), instead of the usual tiling
texture. Does not modify run_all.py / geometry.py / render.py.

Usage: python3 mural_experiment.py --texture <file in patterns/> --tag <output subdir suffix>
"""
import argparse
import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from geometry import load_normals, estimate_gravity, fit_planes, compute_uv, WALL_CONF_T
from render import sample_texture, shading_from_irradiance, composite, irradiance_polarity

ap = argparse.ArgumentParser()
ap.add_argument("--texture", required=True, help="filename inside the sample's patterns/ dir")
ap.add_argument("--tag", required=True, help="output subdir suffix: outputs/new/<sample>__<tag>/")
args = ap.parse_args()

ROOT = Path(__file__).resolve().parent.parent
BATCH = ROOT / "rgb-x_output" / "batch_output"
SAMPLE = "living_room_foyer_panels"
D = BATCH / SAMPLE
OUT = Path(__file__).parent / "outputs" / "new" / f"{SAMPLE}__{args.tag}"
OUT.mkdir(parents=True, exist_ok=True)
MURAL_PATH = D / "patterns" / args.texture

rgb = cv2.cvtColor(cv2.imread(str(D / f"{SAMPLE}.jpeg")), cv2.COLOR_BGR2RGB)
normals, was_enc = load_normals(D / f"{SAMPLE}_normal.npy")
normals = normals.astype(np.float32)
conf = np.load(D / f"{SAMPLE}_wall_confidence.npy")
depth = np.load(D / f"{SAMPLE}_depth.npy").astype(np.float32)
f = float(np.load(D / f"{SAMPLE}_focal_length.npy"))
irr = np.load(D / f"{SAMPLE}_irradiance.npy")

H, W = depth.shape
cx, cy = W / 2.0, H / 2.0
mask = conf > WALL_CONF_T
print(f"mask covers {mask.mean()*100:.1f}% of image, {int(mask.sum())} px")

# Force every panel pixel onto ONE plane. SAM already isolated exactly the
# right pixels (all 5 panels, same coplanar wall surface) -- letting k-means
# re-cluster by normal only fragments this into noise slivers, which is fine
# for a repeating pattern but breaks a single continuous mural.
labels = np.full(mask.shape, -1, dtype=np.int32)
labels[mask] = 0

grav, resid, tilt, grav_ambiguous = estimate_gravity(normals, mask)
planes = fit_planes(labels, normals, depth=depth, fx=f, cx=cx, cy=cy, wall_px=int(mask.sum()))
p = planes[0]
print(f"plane 0: n={p['n']}, d={p['d']:.3f}, confidence={p['confidence']:.3f}, "
      f"model_vs_depth_deg={p['disagree_deg']:.1f}")

uv, info = compute_uv(labels, planes, grav, f, cx, cy)
p0 = info[0]
print(f"plane extent: {p0['width_m']:.3f} x {p0['height_m']:.3f} m")

mural_bgr = cv2.imread(str(MURAL_PATH))
assert mural_bgr is not None, f"couldn't read {MURAL_PATH}"
mural_rgb = cv2.cvtColor(mural_bgr, cv2.COLOR_BGR2RGB)
Ph, Pw = mural_rgb.shape[:2]

W_target, H_target = p0['width_m'], p0['height_m']
# COVER fit: preserve the mural's own aspect ratio, scale up until it fully
# covers the panels' bounding extent, crop whichever axis has slack -- same
# as how a real photo-mural gets trimmed to a wall.
pattern_width_m = max(W_target, H_target * Pw / Ph)
pattern_height_m = pattern_width_m * Ph / Pw
print(f"mural px {Pw}x{Ph} -> pattern_width_m={pattern_width_m:.3f} "
      f"pattern_height_m={pattern_height_m:.3f} (covers {W_target:.3f}x{H_target:.3f} m target)")

# Centre on the panels' true bounding-box midpoint (not compute_uv's
# pixel-count-weighted centroid -- the big centre panel would pull that off
# the true geometric middle), then shift so that midpoint lands exactly on
# the mural's own centre. Keeps the whole panel span within one texture
# tile so sample_texture's wrap-around is never actually triggered.
U, V = uv[..., 0], uv[..., 1]
u_lo, u_hi = np.percentile(U[mask], [0.5, 99.5])
v_lo, v_hi = np.percentile(V[mask], [0.5, 99.5])
u_mid, v_mid = (u_lo + u_hi) / 2.0, (v_lo + v_hi) / 2.0

U2 = U - u_mid + pattern_width_m / 2.0
V2 = V - v_mid - pattern_height_m / 2.0
uv2 = np.stack([U2, V2], axis=-1).astype(np.float32)

sampled, wall = sample_texture(uv2, labels, mural_rgb, pattern_width_m, pattern_height_m)

pol = irradiance_polarity(irr, rgb, wall)
print(f"irradiance polarity: {pol:+.3f} -> using invert=False (established this session)")
sh_irr = shading_from_irradiance(irr, wall, invert=False)
final = composite(rgb, sampled, wall, sh_irr)

cv2.imwrite(str(OUT / "final.png"), cv2.cvtColor(final, cv2.COLOR_RGB2BGR))
cv2.imwrite(str(OUT / "sampled_texture_only.png"), cv2.cvtColor(sampled, cv2.COLOR_RGB2BGR))
print("wrote", OUT / "final.png")
