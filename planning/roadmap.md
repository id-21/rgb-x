# Plan of action

Companion to `web-service-strategy.md`. That doc is *what to build*; this is *what
order, and how you know each step is done*.

Governing constraint: **realism over latency.** Scene prep may be slow and async.

---

## The uncomfortable thing to say first

Every number in this project comes from **12 images**, and several are the same kind
of room — bright, modern, Manhattan-geometry apartments with large windows. That
set has already produced two known-marginal samples out of twelve.

Everything from here — which model to keep, whether the mask fix works, what
`min_frac` should be — is a decision being made on a sample too small to support it.
Growing the eval set is not a chore to do later; it is the thing that makes every
subsequent decision trustworthy. It is step 1 for that reason.

**Specifically missing from the current 12:**

| gap | why it matters |
|---|---|
| phone photos | lens distortion — see risks below |
| small / narrow rooms | grazing angles, where UV degeneracy bites |
| dim and mixed-lighting rooms | irradiance is only validated on bright daylight |
| mirrors, glass partitions, TVs off | depth sees through them |
| already-patterned walls | shading fix was validated on plain walls |
| non-Manhattan geometry | angled walls, bays, sloped ceilings |
| portrait phone framing | only `room1` is portrait, and it has one wall |

---

## Phase 0 — close out (days)

| task | done when |
|---|---|
| Run Depth Pro on `bed_back_bedroom` | 4 remaining three-way comparisons build |
| Fix EXIF orientation handling | a rotated phone photo produces upright output |

`bed_back_bedroom` is the only sample with previous outputs besides `living_simple2`,
so it's carrying 4 of the 5 comparisons. Cheap to unblock.

## Phase 1 — grow the eval set and lock the scorecard (1–2 weeks)

| task | done when |
|---|---|
| Collect ~100–200 room photos spanning the gaps above | set exists, checked in or in object storage |
| Extend `verify.py` into the scorecard | one command, one table, pass/fail per sample |
| Establish the baseline | current pipeline scored across the whole set |

**The scorecard metrics already exist** — this is assembling them, not inventing them:

- `reconstructed_vertical_extent_m` — expect 2.4–3.0 m
- `wall_pair_angles_deg` — expect ~90° or ~0°
- `gravity_tilt_deg`, `gravity_ambiguous_single_wall`
- `model_vs_depth_normal_deg` — flags glass and mirrors
- `irradiance_polarity_corr` — expect negative
- `corr(HF_shading, HF_photo)` — expect ≪ 0.9; catches shading regressions
- plane count, smallest-plane share — catches thin-sliver loss

Add a **human rating pass** (1–5 realism, on maybe 40 images) alongside the numeric
gates. The numeric gates catch geometry breakage; they will not catch "correct but
looks fake". Given realism is the priority, you need the subjective number too, and
you need it *before* you start changing things, so you have a baseline to compare to.

## Phase 2 — finish the geometry work in Python (2–4 weeks)

Do this **before** any web work. It changes what goes into `scene.json`, and it is
where the remaining realism lives.

| task | done when |
|---|---|
| **Connected components per plane** | two coplanar but physically separate walls get distinct IDs |
| **Small-plane handling** | lower `min_frac`, add a guard; `obstacle_course_living` keeps its 2.6% plane |
| **Mask edge refinement** | guided filter using `\|∇D\|` for occlusions + normal discontinuity for creases; soffits and reveals no longer wallpapered |
| **Investigate spurious planes** | `complex_living` / `obstacle_course_living` 29–58° pairs either explained as real geometry or fixed |
| **Stable plane IDs** | derived from area/centroid, not k-means label order; survive a re-run |
| **Linear-space compositing** | shading multiply in linear, not sRGB |

Connected components is the keystone: it's required for per-wall selection *and* it
likely fixes some of the spurious-plane cases, since a spurious "plane" spanning two
disconnected regions is exactly what it splits.

## Phase 3 — freeze the contract (days)

| task | done when |
|---|---|
| Emit `scene.json` v1 from Python | schema documented, `"version": 1` |
| Commit golden files | every eval image has a committed scene.json |

Design it with per-wall selection already in mind — many planes, stable IDs,
`area_m2`, `centroid_px`, `confidence`. Do not ship a v1 that per-wall selection
immediately forces to v2.

## Phase 4 — client and server, in parallel

These are independent once the contract is frozen.

**4a. WebGL2 renderer** — build against static golden files, no server.

| done when |
|---|
| renders all eval scenes from `scene.json` + maps + photo |
| anisotropic filtering enabled (`EXT_texture_filter_anisotropic`, max 16) |
| linear-space compositing matches the Python reference |
| **golden-frame CI**: headless render vs numpy reference, perceptual diff under threshold |

The numpy renderer is the oracle. That's what stops the two implementations drifting.

**4b. Texture pipeline**

| done when |
|---|
| catalogue ingest → seamless 1024² tile → KTX2/Basis with precomputed mips |
| `pattern_width_m` per SKU, not hardcoded 0.53 |
| automated seam check rejects non-tiling assets |
| progressive mip streaming: low level first, then refine |

**4c. Server**

| done when |
|---|
| async job: upload → job id → poll/websocket → bundle URL |
| scene bundles cached by image hash |
| scorecard runs server-side as a quality gate; bad reconstructions are refused, not shipped |
| GPU worker scales to zero; cold start measured |

## Phase 5 — per-wall selection UI

| done when |
|---|
| click a wall to select it (hit-test the ID channel on CPU) |
| different wallpaper per wall, single render pass via `sampler2DArray` |
| hover highlight |
| **manual merge/split** of plane IDs, client-side only |

Manual override matters more than it looks: it's the escape hatch for every case
where segmentation is wrong, and it's cheap because it's a client-side remap.

## Deferred until there's a measured reason

- **Rust core** — only once `scene.json` and the geometry have stopped moving.
- **Model swaps** (feed-forward intrinsic, RANSAC normals) — these are latency
  optimisations, and latency is not the priority. Revisit if wait time hurts
  conversion.
- **Resolution reduction** — measured cost, latency-only benefit. If revisited,
  re-run `experiments/resolution_sensitivity.py` with *real re-inference*, not
  downsampled outputs.
- **Batching RGB→X's two channels** — do this one whenever convenient; it's free.

---

## Risks not yet addressed

**Lens distortion — the biggest unhandled one.** The entire geometry assumes a
pinhole camera: `ray = K⁻¹p`, no distortion terms. Real users shoot on phones, often
ultrawide, which have substantial barrel distortion. Straight walls bow, and the
plane-fit and UV mapping both degrade toward the frame edges — exactly where walls
usually are. Depth Pro gives you `f` but no distortion coefficients. Options: detect
and reject heavily distorted shots, undistort using a per-device model from EXIF, or
estimate distortion from line straightness. **Test this early** — get ultrawide phone
photos into the Phase 1 eval set, because it may change the geometry model.

**EXIF orientation.** Phone photos carry rotation flags. Ignore them and normals,
gravity and depth all disagree with the pixels. Trivial to fix, embarrassing to ship.

**Overfitting to 12 images.** Covered above; it is the reason Phase 1 is Phase 1.

**Wallpaper colour under room light.** Swatches are photographed under studio
lighting; rooms are not. Expect "the colour isn't right" complaints. A mild
white-balance transfer from the room's irradiance is the likely fix.

**Occluders.** Thin objects — plant stems, cables, chair legs — will be missed by the
mask and get wallpapered over. Matting helps; it will not fully solve it.

**Privacy.** Photos of people's homes. Decide retention before launch, not after. The
scene bundle is ~200 KB and contains no recognisable interior — consider keeping only
that and discarding the source photo.

---

## If you only do three things

1. **Grow the eval set to ~100 images** with phone photos and hard cases. Everything
   else is guesswork until this exists.
2. **Connected components + mask edge refinement.** The two biggest remaining
   realism gaps, and both are prerequisites for per-wall selection.
3. **Freeze `scene.json` and build the WebGL renderer against golden files.** Proves
   the architecture end to end without a server, and locks in the reference-oracle
   discipline that keeps the two implementations honest.
