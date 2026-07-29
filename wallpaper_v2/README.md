# wallpaper_v2 — what changed

Corrected rewrite of `visualize_with_precomputed_normals.py`. Same inputs
(RGB + SegFormer mask + RGB→X normals + RGB→X irradiance + Depth Pro depth &
focal length), same idea, five bugs fixed.

```
geometry.py   normals, gravity, plane fitting, ray/plane UV
render.py     mip-mapped texture sampling, shading, compositing
run_all.py    driver -> outputs/
```

Run: `python3 run_all.py`

---

## The five differences

### 1. Normals were never decoded

`*_normal.npy` is stored **display-encoded in [0,1]**: `n_stored = (n+1)/2`.
The old pipeline used it raw.

Proof: `median(‖2·n−1‖) = 1.000` exactly, vs `median(‖n‖) = 1.08`.

K-means was unaffected (a uniform scale+shift doesn't move Euclidean cluster
boundaries), so the cluster map looked fine — which is why this hid for so long.
But `fit_plane_normals` averaged encoded vectors, and the result points somewhere
completely different:

| wall | old (encoded) | decoded |
|---|---|---|
| TV wall | `[0.056, 0.559, 0.827]` | `[-0.879, 0.006, 0.477]` |
| window wall | `[0.508, 0.383, 0.771]` | `[0.287, -0.026, 0.957]` |
| left return | `[0.847, 0.422, 0.325]` | `[0.968, -0.024, -0.249]` |

Angles between the three walls: **28°/56°/33°** before → **78°/88°/89°** after.
The decoded normals are a clean Manhattan room. Since `build_plane_basis` derived
the texture axes from that normal, every wall's texture frame was rotated and
sheared by 30–60°.

### 2. Normals and depth were in different coordinate frames

RGB→X emits normals with **+z toward the viewer** (OpenGL-style); the pinhole
unprojection used for Depth Pro's depth has **+z away** (OpenCV). Measured over
five samples, negating z drops model-vs-depth normal disagreement from **62–78°
to 8–41°**; negating y makes it consistently worse. So `load_normals` negates z
and leaves y alone.

This bug and #1 stack: both had to be fixed before anything else could be judged.

### 3. There was no homography — UV came from raw per-pixel depth

The old `compute_uv_coordinates` dotted the **unprojected depth point cloud**
against the plane axes. Those points don't lie on the plane, so every depth
wobble became texture warping. Clustering by normal made it *look* per-plane, but
the UVs were still per-pixel.

Now each plane is fitted as `(n, d)` and every pixel's viewing ray is intersected
with it:

```
X = K⁻¹p · d / (n · K⁻¹p)
```

Every sampled point lies on the plane by construction, so the image→plane map is
exactly a homography. **Depth now enters only through the single scalar `d` per
plane** — depth noise can shift a wall nearer or further but can no longer warp
the pattern. `f` and `n` do all the perspective work.

Rays within `|n·r| < 0.08` of parallel are dropped rather than rendered as streaks.

### 4. "Up" was hardcoded; one wall became several planes

- `build_plane_basis` assumed `up = [0,-1,0]`, i.e. zero camera roll and pitch.
  Gravity is now estimated as the axis perpendicular to all wall normals
  (smallest eigenvector of the normal scatter matrix). Measured tilts across the
  set: 1.4°–5.0°, so this was a real if modest error everywhere.
  *Degenerate case:* with only one visible wall, gravity is underdetermined and
  `eigh` returns an arbitrary null-space vector — `room1` produced a 77° "gravity"
  this way. A weak prior toward camera-up breaks the tie; it does nothing when
  two or more walls pin the answer down.
- K-means with `n_clusters=10, top_n=5` split single physical walls into several
  clusters, each of which then got **its own UV origin** → the pattern jumped at
  the seam. Clusters within 12° are now merged, and split by plane offset `d`
  instead, so genuinely parallel-but-separate walls still stay apart.

### 5. Sampling and shading

- **Nearest-neighbour → anisotropic mip-mapped.** The old code did
  `(u_norm * (Pw-1)).astype(np.int32)`. At grazing angles the pattern compresses
  below one pixel, and point sampling turns that into moiré — which reads as
  "cheap" even when the geometry is right. See *Resolution* below.
- **Shading now comes from the irradiance model**, not the photo's LAB luminance.
  The old approach re-used the original image's own luminance, which baked the
  previous wall's sockets, artwork and shadows into the new texture.
  `final_lab_shading.png` keeps the old behaviour for comparison.
- **The irradiance map is stored inverted: dark means MORE light.** Measured as
  the correlation between raw irradiance and observed wall brightness, on walls,
  which are near-uniform albedo and so should track the light landing on them:

  | | corr |
  |---|---|
  | room1, fluted_walls, simple_living_3, perspective_living, living_simple2 | −0.96 to −0.99 |
  | simple_living_2, no_walls, complex_living, living_small_walls | −0.85 to −0.87 |
  | simple_living | −0.75 |
  | obstacle_course_living | −0.45 |

  Negative in all 12. Used as-is it lit the room backwards — brightening the
  ceiling/wall creases, which are physically the *darkest* points from ambient
  occlusion, and darkening the open wall near the window. `shading_from_irradiance`
  now inverts, and the polarity is measured and reported per sample so the
  convention cannot silently flip again. See `outputs/_shading_polarity.png`.

  The inversion is plain linear (`1 − E`) by design: fitting `exp(-kE)` or
  `(1-E)^g` against observed luminance improved R² by under 0.01 on average, and
  the best-fit exponent scattered from 0.4 to 2.6 across samples — noise, not a
  systematic relationship.
- Wall mask is feathered instead of hard-thresholded.
- The final mask is no longer `labels >= 0` after a top-N filter, which used to
  leave holes showing the original wall.

Also: `pattern_height_m` was accepted as a parameter and then silently
overwritten by `pattern_width_m * (Ph/Pw)`.

---

## On scale

Scale was never actually broken. Depth Pro's depth is metric and its focal length
is accurate, so `pattern_width_m = 0.53` applies directly. What broke it was
`fx_multiplier = 0.8` — every `bed_back_bedroom` run used `f = 0.8 × W`, and only
the last `living_simple2` run used the prediction.

On `living_simple2`, Depth Pro says **473 px**; `0.8 × W` says **819 px**. At 819
the reconstruction gives a floor-to-ceiling height of **1.89 m** — impossible —
versus **3.55 m** at 473. The room was compressed 1.73× and the pattern rendered
1.73× too large.

`f` is always read from `*_focal_length.npy` and never guessed. It is rescaled
alongside `cx, cy` when the working image is downscaled (`MAXDIM = 2000`).

---

## Resolution

Three separate things were making the pattern look low-res. All three are fixed.

**1. Anisotropic filtering.** A receding wall is compressed far more along one
screen axis than the other — measured across the set, 2–25 texels per screen
pixel. Isotropic mip-mapping picks a single blur radius, so it blurs the sharp
axis exactly as hard as the compressed one. The footprint is now treated as an
ellipse: mip level comes from the **minor** axis, and up to 16 taps are averaged
along the **major** axis. Detail survives along the sharp axis; the compressed
axis is still properly averaged, so nothing aliases. A `lod_bias` of −0.5
compensates for the inherent softness of blending between mip levels.

**2. Native resolution.** The working resolution was capped at 2000 px on the
long edge, costing 1.5–2.2× on seven samples. Now 9 of 11 render at full native
size. `room1` and `no_walls_living` render at 0.86× — a memory budget
(`MAXPIXELS = 10_000_000`), not a pipeline limit. Raise it on a machine with more
RAM and they go to native too.

**3. The comparison sheets themselves.** They were matplotlib figures at 52–105
dpi, which resampled every panel far below the actual render. They are now
concatenated with cv2 at native pixels.

`outputs/_sharpness_check.png` is a 2× crop against the previous pipeline. The old
render is sharp but speckled — that grain *is* the aliasing from point sampling.

### Memory

Peak is ~1.8 GB for a 4k sample. Rendering many large samples back to back in one
process exhausted memory, so `run_batch.sh` runs one OS process per sample. To
get there, the pipeline also stopped materialising full-image `(H, W, 3)` grids:
rays, point clouds and texture coordinates are built per plane, and the sampler
works in chunks. That cut peak RSS from 2.3 GB to 1.8 GB on `room1`.

```bash
./run_batch.sh                       # all samples, one process each
python3 run_all.py --one <sample>    # a single sample
python3 run_all.py --comparisons
python3 run_all.py --collect         # summary.json + overview sheet
python3 verify.py                    # numeric self-checks
```

## Outputs

```
outputs/_overview_all_samples.png   all 11, original vs new
outputs/new/<sample>__<texture>/
    final.png                 irradiance shading
    final_lab_shading.png     old LAB shading, for comparison
    steps.png                 8-panel: normals, planes, U, V, texture, shading
    diagnostics.json
outputs/comparisons/              original | previous output | new output
outputs/summary.json
```

Previous outputs are read from `rgb-x_output/`, never recomputed.

## Diagnostics

Reported per sample, **never used to alter `f`** — they exist so the models can be
sanity-checked:

- `reconstructed_vertical_extent_m` — floor-to-ceiling, expect 2.4–3m
- `wall_pair_angles_deg` — expect ~90° (perpendicular) or ~0° (parallel walls)
- `model_vs_depth_normal_deg` — per plane; large values flag windows and mirrors,
  where Depth Pro sees through to the scene beyond. The **model normal is always
  what gets used**, precisely because it survives glass; depth only supplies `d`.
- `gravity_ambiguous_single_wall` — true when only one wall is visible
- `irradiance_polarity_corr` — expect negative (map is stored inverted). A
  positive value means the model's convention changed and `invert=` needs
  revisiting; the run prints a warning if that happens.

Across the 11 samples: ceiling heights 2.8–3.7 m, gravity tilt 1.4–5.0°, wall
angles clustering at 82–89°.

## Known gaps

- **`bed_back_bedroom` is not rendered** — it has depth but no
  `_focal_length.npy`. It's the only sample with previous outputs to compare
  against besides `living_simple2`, so 4 of the 5 three-way comparisons are
  missing until Depth Pro is re-run on it.
- `complex_living` and `obstacle_course_living` produce some plane pairs at
  29–58°, which are neither perpendicular nor parallel — likely spurious splits
  worth a look.
- Wall masks include ceiling soffits and window reveals in places; that's
  SegFormer's output, untouched here.
- Working resolution capped at 2000 px on the long edge.
