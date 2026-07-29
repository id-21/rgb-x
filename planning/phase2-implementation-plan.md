# Phase 2 — geometry & quality work: implementation plan

**This document specifies changes. It does not make them.**

Intended to be executed by Claude Code, one task at a time, with a commit per task.
Phase 2 is the last body of work in Python before `scene.json` is frozen and the web
client starts. Everything here changes what ends up in that contract, which is why it
comes first.

Governing constraint: **realism over latency.** No task here may be justified on speed.

---

## Working agreement

**Branch:** `phase2-geometry`

**Do not** change these without an explicit task saying so:
- the normal decode (`[0,1]` → `[-1,1]`, z negated) — verified over 5 samples
- irradiance inversion — verified over 12 samples
- Depth Pro's focal length is authoritative; never re-derive or guess `f`
- anisotropic filtering and the `-0.5` LOD bias in `sample_texture`

**Task order is a dependency order, not a preference.** T0 first. T6 last.

**Every task ends in exactly one commit** containing the code change *and* its test.
Commit message format:

```
<type>(<area>): <summary>

<what changed and why, 2-4 lines>

Metrics:
  <metric>: <before> -> <after>   [PASS/FAIL vs gate]
  ...

Samples affected: <list, or "all 12">
```

`<type>` is `feat`, `fix`, `test`, `perf`, or `chore`. `<area>` is `geometry`,
`render`, `mask`, `harness`, or `docs`.

**If a gate fails, stop and report.** Do not proceed to the next task, and do not
loosen a gate to make it pass. A failing gate is information.

**Never delete or overwrite `outputs/baseline/`** (see T0).

---

## T0 — Baseline harness

*No behaviour change. Everything downstream is measured against this.*

### Why

There is currently no way to tell whether a change improved things or just moved
them. Every later task reports a before/after, and this is what "before" means.

### Changes

Create `wallpaper_v2/harness/` with:

**`capture_baseline.py`** — runs the current pipeline over every sample and writes
`outputs/baseline/<sample>.json` containing:

```jsonc
{
  "sample": "...",
  "focal_px": 473.4,
  "working_size": [1024, 682],
  "gravity": [...],
  "gravity_tilt_deg": 1.9,
  "gravity_ambiguous": false,
  "planes": [
    {"normal": [...], "d": 2.97, "px_share": 0.668,
     "extent_m": [4.97, 3.69], "model_vs_depth_deg": 5.0}
  ],
  "wall_pair_angles_deg": [78, 14, 88],
  "vertical_extent_m": 3.55,
  "irradiance_polarity_corr": -0.965,
  "wall_coverage": 0.94,          // NEW, see below
  "render_sha256": "..."          // of final.png bytes
}
```

**`wall_coverage` is a new metric and the most important one T0 adds.** Defined as:

```
wall_coverage = (pixels with labels >= 0) / (pixels with conf > 0.5)
```

i.e. what fraction of the segmenter's wall pixels actually received a plane. The
current `top_n=4` filter silently discards the remainder, leaving holes where the
original wall shows through. Nobody has measured this yet.

**`compare_to_baseline.py`** — reruns the pipeline, matches planes to the baseline by
normal direction (reuse the `match()` logic in
`experiments/resolution_sensitivity.py`), prints a per-sample delta table, and exits
non-zero if any hard gate is violated.

### Tests

`harness/test_baseline.py`:
- capture → compare with no code change → all deltas exactly zero
- `wall_coverage` ∈ [0, 1] for every sample
- every baseline JSON parses and has all required keys

### Gate

Comparing the pipeline to its own freshly-captured baseline must produce zero drift.
If it doesn't, there is nondeterminism to find before anything else proceeds — most
likely the k-means seed.

### Commit

```
chore(harness): add baseline capture and comparison harness

Adds outputs/baseline/*.json and a comparison tool so every later change
reports a measured delta rather than an assertion. Introduces wall_coverage,
the fraction of segmented wall pixels that actually receive a plane.

Metrics:
  self-comparison drift: n/a -> 0.000   [PASS]
  wall_coverage (min across samples): n/a -> <value>

Samples affected: all 12
```

---

## T1 — Deterministic, stable plane IDs

*Do this early: it makes every later diff readable.*

### Why

Plane IDs currently come from k-means label order, which is arbitrary. Two runs can
assign different IDs to the same wall. That makes baseline comparison unreliable, and
it will break saved user designs the moment the pipeline is updated.

### Changes

In `geometry.py`, at the end of `cluster_planes`, order planes by a **content-derived
key** rather than cluster size alone:

1. primary: descending pixel count (stable, already computed)
2. tie-break: ascending plane distance `d`
3. final tie-break: ascending centroid x, then y in image space

Emit the ordering as the plane ID. Document that IDs are positional and only stable
for a *given* input — they are not global identifiers across different photos.

Also expose `random_state` as a parameter of `cluster_planes` rather than the
hardcoded `42`, so determinism can actually be tested.

### Tests

`harness/test_determinism.py`:
- run `cluster_planes` + `fit_planes` with `random_state` ∈ {42, 7, 1234} on every
  sample; assert the resulting **sorted set** of `(normal, d)` matches within
  1° / 0.05 m
- assert plane ID *k* refers to the same physical plane across all three seeds
- run the full pipeline twice with identical settings; assert identical `render_sha256`

### Gate

- plane identity stable across ≥3 seeds on ≥11 of 12 samples
- byte-identical renders across repeated runs

If a sample fails seed-stability, that is a finding worth reporting — it means
k-means is landing in different local minima, which is itself a quality signal.

### Commit

```
fix(geometry): derive plane IDs from content, not k-means label order

Plane IDs were positional on arbitrary k-means output, so repeat runs could
renumber the same walls. IDs now sort by pixel count, then distance, then
image centroid. random_state is now a parameter so determinism is testable.

Metrics:
  seed-stable samples: <n>/12   [gate >= 11]
  repeat-run render identical: yes/no   [gate yes]

Samples affected: all 12
```

---

## T2 — Linear-space compositing

*Small, self-contained, a real correctness fix.*

### Why

`composite()` multiplies shading against **sRGB-encoded** values. sRGB is a
perceptual encoding, not a linear measure of light, so multiplying it does not model
attenuation — it crushes midtones and makes shaded regions read as muddy rather than
dimmer. Light attenuation is linear; the multiply belongs in linear space.

### Changes

In `render.py`:

- add `srgb_to_linear(x)` / `linear_to_srgb(x)` (proper piecewise transfer function,
  not a 2.2 gamma approximation)
- rewrite `composite()` to: decode photo and texture to linear → multiply texture by
  shading → alpha-blend in linear → encode result to sRGB
- keep the old path behind `linear=False` so the two can be A/B'd

### Tests

`harness/test_color.py`:
- **round trip**: `linear_to_srgb(srgb_to_linear(x)) ≈ x` for all 256 uint8 values,
  max error < 1/255
- **known values**: sRGB 0 → 0.0, sRGB 255 → 1.0, sRGB 128 → ~0.2159 linear
- **invariant**: with `shading == 1.0` everywhere and `alpha == 1`, composite output
  inside the mask must equal the texture exactly (±1/255). This must hold in both
  the old and new paths — it's what proves the change is a pure re-parameterisation
  of the shading response, not a brightness shift.

### Metrics to report (characterise, don't gate)

For each sample, within the wall mask:
- mean and p10/p50/p90 luminance, old vs new
- expected direction: **shadowed regions get lighter**, highlights roughly unchanged

### Gate

- round-trip and invariant tests pass
- no change to `wall_coverage` or any plane parameter (this task must not touch
  geometry — if it does, something is wired wrong)

### Commit

```
fix(render): composite in linear light rather than sRGB

Shading was multiplied against sRGB-encoded values, which is not a physical
model of attenuation and crushes midtones. Photo and texture are now decoded
to linear, shaded, blended, and re-encoded.

Metrics:
  wall-region p10 luminance: <before> -> <after>
  wall-region p50 luminance: <before> -> <after>
  plane params changed: none   [gate none]

Samples affected: all 12
```

---

## T3 — Small-plane retention and wall coverage

### Why

`cluster_planes(top_n=4, min_frac=0.02)` discards everything else. Two measured
consequences:

- `obstacle_course_living` loses a 2.6%-of-wall plane entirely when run at 768 px
  (see `experiments/resolution_sensitivity.py`)
- discarded pixels get no plane, so they render as original wall — holes

Per-wall selection makes both visible to users. A thin return wall or the strip
between two windows is exactly the kind of surface someone will try to click.

### Changes

In `cluster_planes`:

- raise `top_n` default 4 → **8**
- lower `min_frac` default 0.02 → **0.005**
- add an **absolute** floor alongside the relative one: `min_px = 1500` at native
  resolution, scaled by the working-resolution factor so it means the same thing at
  any size
- attach a `confidence` per plane, so the UI can rank rather than the pipeline
  silently dropping. Suggested components (document the formula in the docstring):
  - fraction of wall pixels
  - k-means inertia within the cluster (tight normals = high confidence)
  - `model_vs_depth_normal_deg` (already computed; large = glass/mirror suspicion)

Do **not** add a fallback that assigns leftover pixels to the nearest plane. If a
pixel doesn't belong to a fitted plane, that should stay visible in `wall_coverage`
rather than be papered over.

### Tests

`harness/test_small_planes.py`:
- `wall_coverage` ≥ **0.98** on every sample
- re-run `experiments/resolution_sensitivity.py`: the `lost` column must be **0** for
  all samples, including `obstacle_course_living`
- every returned plane has `confidence` ∈ [0, 1]
- no plane below `min_px` at the working resolution

### Gate

| metric | gate |
|---|---|
| `wall_coverage`, min across samples | ≥ 0.98 |
| planes lost native → 768 px | 0 |
| non-Manhattan plane pairs | must not increase by more than 2 across the set |

That last one is the counterweight: retaining more planes will surface more spurious
ones. If it blows past +2, the confidence score needs to do more work before this
lands.

### Commit

```
feat(geometry): retain small planes and measure wall coverage

top_n 4->8, min_frac 0.02->0.005, plus an absolute pixel floor and a
per-plane confidence score. Previously-discarded planes left holes rendering
as original wall, and obstacle_course_living lost a real 2.6% plane at 768px.

Metrics:
  wall_coverage (min): <before> -> <after>   [gate >= 0.98]
  planes lost at 768px: 1 -> <after>         [gate 0]
  non-Manhattan pairs (total): 8 -> <after>  [gate <= 10]

Samples affected: all 12
```

---

## T4 — Regions: separate geometry from selection

*The keystone task. Read the rationale before implementing — the obvious design is
the wrong one.*

### Why the obvious approach is wrong

The instinct is "split planes by connected component." Don't. **Two disconnected
components of the same plane already render correctly**, because UV comes from
ray/plane intersection against a shared `(n, d, origin)` — the mapping is continuous
across the gap whether or not the pixels are. Splitting them into separate planes
with separate origins would actively *break* pattern continuity across a doorway.

What connected components are actually needed for is **selection**, which is a
different concern. So model it as two levels:

| concept | what it is | used for |
|---|---|---|
| **plane** | `(n, d, u_axis, v_axis, origin)` | UV mapping, perspective |
| **region** | one connected component of wall pixels, referencing a plane | user selection, per-wall texture assignment |

Several regions may share one plane. A wall interrupted by a doorway is two regions
on one plane — and because they share the plane, the pattern lines up across the
doorway, which is what you want. Two coplanar walls in different parts of the room
are also two regions on one plane, and can still take different wallpapers, because
texture is assigned per region.

This also means the label raster the client consumes should encode **region ID**, not
plane ID, with a region → plane table in `scene.json`.

### Changes

In `geometry.py`, add:

```python
def extract_regions(labels, min_region_px=1500, connectivity=8):
    """
    Decompose each plane's pixel set into connected components.

    Returns (region_labels, regions) where regions[i] = {
        'plane': int, 'npix': int, 'centroid_px': (x, y),
        'bbox_px': (x0, y0, x1, y1), 'area_m2': float,
    }
    Regions below min_region_px are merged into the nearest adjacent region on
    the SAME plane if one exists, otherwise dropped.
    """
```

Use `cv2.connectedComponentsWithStats`. Compute `area_m2` by summing per-pixel
world-space area on the plane — or, simpler and adequate, from the region's UV extent.

Add a `suggested_label` derived from the plane normal relative to camera and gravity:
`"back wall"` (normal ≈ −z), `"left wall"`, `"right wall"`, `"angled wall"`.

`run_all.process` should record regions in `diagnostics.json` and add a region
visualisation panel to `steps.png`.

### Tests

`harness/test_regions.py`:
- every region is genuinely 8-connected (flood fill from centroid reaches all its
  pixels)
- every region references a valid plane
- `sum(region.npix) == (labels >= 0).sum()` — no pixels lost or double-counted
- no region smaller than `min_region_px`
- **rendering invariance**: rendering by plane and rendering by region must produce a
  byte-identical image. This is the test that proves regions are a selection concept
  and carry no geometric meaning. If it fails, regions have leaked into the UV path.

### Metrics to report

Per sample: plane count, region count, regions-per-plane, smallest region as % of
wall. Specifically report whether `complex_living` and `obstacle_course_living`
spurious planes decompose into multiple disconnected regions — that would be evidence
they are clustering artifacts rather than real geometry, and feeds directly into T5.

### Gate

- all structural tests pass
- render-invariance test passes (byte-identical)
- region count ≥ plane count on every sample

### Commit

```
feat(geometry): add regions as a selection layer over planes

Connected components decompose each plane into selectable regions. Planes stay
the unit of geometry (shared UV, so patterns run continuously across a doorway);
regions become the unit of selection and texture assignment.

Metrics:
  planes -> regions: <per-sample table>
  render invariance (plane vs region path): byte-identical   [gate yes]
  pixel accounting: exact   [gate exact]

Samples affected: all 12
```

---

## T5 — Investigate spurious non-Manhattan planes

*A diagnosis task. Do not fix anything until the cause is written down.*

### Why

`verify.py` reports plane pairs that are neither perpendicular nor parallel:

| sample | angles |
|---|---|
| `complex_living` | 54°, 33°, 35° |
| `fluted_walls_living` | 69° |
| `obstacle_course_living` | 46°, 29°, 50°, 58° |

Three hypotheses, and they need different fixes:

1. **Genuine geometry** — bay windows, angled alcoves, sloped ceilings. `fluted_walls_living`
   is a plausible candidate: fluted walls have ribbed normals by construction.
2. **Clustering artifact** — k-means splitting one curved or noisy surface into
   several arbitrary slices. T4's region decomposition would show these as
   spatially interleaved rather than compact.
3. **Bad normals** — glass, mirrors, TV screens. `model_vs_depth_normal_deg` is
   already computed and should flag these.

### Changes

Add `experiments/plane_diagnosis.py` that, for every non-Manhattan pair, emits:

- an overlay of both planes on the photo, colour-coded
- the two normals, their `model_vs_depth_normal_deg`, pixel shares
- region decomposition for each (compact or interleaved?)
- whether either plane overlaps a high-`model_vs_depth` area

Write findings into `planning/plane-diagnosis-findings.md` — one paragraph per case,
classified into the three hypotheses above.

**No pipeline change in this task.** The fix, if any, is a separate task informed by
the findings.

### Gate

Every non-Manhattan pair in the set is classified. No unexplained cases.

### Commit

```
test(geometry): diagnose non-Manhattan plane pairs

Adds experiments/plane_diagnosis.py and records findings. No pipeline change.

Metrics:
  non-Manhattan pairs classified: <n>/<n>   [gate all]
  genuine / clustering artifact / bad normals: <a>/<b>/<c>

Samples affected: complex_living, fluted_walls_living, obstacle_course_living
```

---

## T6 — Mask edge refinement

*Last, because it perturbs every metric above, and because it needs annotation work.*

### Why

The mask is thresholded at `conf > 0.5` and feathered with a fixed Gaussian. It
includes ceiling soffits and window reveals, and its edges are staircased. After
geometry, this is the most visible defect.

### Why the photo alone is the wrong guide

Colour edges and surface boundaries are not the same thing. A picture frame on a wall
has a strong colour edge and no surface boundary; a photo-guided filter will snap the
mask to it. The two real boundary types have different signatures:

| boundary | example | signal |
|---|---|---|
| **occlusion** | sofa in front of wall, doorway | large `\|∇D\|` — depth jumps |
| **crease** | wall meets ceiling, wall meets wall | depth **continuous**; normal flips |

This distinction matters and is easy to get wrong: at a wall/ceiling junction the
surfaces physically meet, so `|∇D|` shows *no* spike. A depth-gradient guide alone
misses exactly the soffit case that is currently wrong. Creases live in the normal
field.

### Changes — staged, measure between stages

**T6a — guided filter, photo guide only.** Implement a self-contained guided filter
in `mask.py` (`cv2.ximgproc` is not available in this environment — do not add the
dependency). Apply it to the **soft** confidence map, not the thresholded mask, using
photo luminance as guide. Measure.

**T6b — add the geometric guide.** Build an edge map:

```
E_occlusion = normalize(|grad(depth)| / depth)      # scale-invariant
E_crease    = angle between neighbouring normals
E_geom      = max(E_occlusion, E_crease)
```

Use `E_geom` to gate the photo guidance: where the photo has a strong edge but
`E_geom` is near zero, suppress the guide's influence. That is the picture-frame case.
Measure the delta against T6a — if it is not clearly better, keep T6a and say so.

**T6c — feathering from confidence.** Replace the fixed-radius Gaussian in
`feather_mask` with a width driven by local confidence gradient: sharp where the
model is confident, soft where it isn't.

### Tests and metrics — this needs ground truth

Proxy metrics work across the whole set but cannot settle correctness. **Annotate
ground-truth wall masks for 20 images** spanning the hard cases (soffits, window
reveals, furniture occlusion, curtains). Then:

| metric | what it measures | gate |
|---|---|---|
| **IoU** vs annotation | overall mask correctness | ≥ 0.95, and must not regress |
| **Boundary F-score** @ 2 px | edge accuracy specifically | primary metric; report @1/2/5 px |
| **edge-alignment proxy** (no GT) | mean distance from mask boundary to nearest `E_geom` peak | should decrease |
| **staircase ratio** (no GT) | boundary length ÷ smoothed boundary length | should decrease toward 1.0 |

Boundary F-score is the one that matters. IoU barely moves when only the boundary
changes — it is there to catch a refinement that improves edges while eating the
interior.

Also re-run the full harness: geometry metrics must not regress. Mask changes shift
which pixels are wall, which shifts plane fits.

### Gate

- IoU ≥ 0.95 on annotated set, no regression vs baseline
- boundary F-score @2 px improves by ≥ 0.05 absolute
- `wall_coverage` stays ≥ 0.98
- no new non-Manhattan plane pairs
- `verify.py` still reports 0 failures across the set

### Commits

Three, one per stage:

```
feat(mask): guided-filter refinement of wall confidence (photo guide)
feat(mask): gate photo guidance by depth and normal discontinuity
feat(mask): confidence-driven adaptive feathering
```

Each with its IoU / boundary-F before→after in the body.

---

## Metrics reference

Everything the harness should report, in one place.

| metric | where | healthy range | notes |
|---|---|---|---|
| `wall_coverage` | T0 | ≥ 0.98 | new; fraction of wall pixels given a plane |
| plane count | T0 | 1–8 | |
| region count | T4 | ≥ plane count | |
| `vertical_extent_m` | existing | 2.4–3.0 | ceiling height sanity |
| `wall_pair_angles_deg` | existing | ~0° or ~90° | non-Manhattan pairs flagged |
| `gravity_tilt_deg` | existing | < 8° typical | > 25° fails unless ambiguous |
| `model_vs_depth_normal_deg` | existing | < 20° | high = glass/mirror |
| `irradiance_polarity_corr` | existing | negative | positive = convention changed |
| `corr(HF_shading, HF_photo)` | existing | ≪ 0.9 | catches shading regressions |
| planes lost native → 768 px | T3 | 0 | |
| seed stability | T1 | ≥ 11/12 | |
| render invariance | T4 | byte-identical | |
| IoU vs annotation | T6 | ≥ 0.95 | 20 annotated images |
| boundary F-score @2 px | T6 | +0.05 vs baseline | primary mask metric |

---

## Out of scope for Phase 2

Do not start these, even if they seem adjacent:

- `scene.json` emission — Phase 3, after this work settles the schema's contents
- any WebGL, shader or client code
- Rust
- swapping or shrinking any model
- resolution reduction of model **inputs** — measured cost, latency-only benefit
- texture pipeline / KTX2
- per-wall selection **UI** (T4 provides the data model, not the interface)

## Definition of done for Phase 2

- [ ] all six tasks committed on `phase2-geometry`, each with passing gates
- [ ] `verify.py` reports 0 failures across the full eval set
- [ ] `wall_coverage` ≥ 0.98 everywhere
- [ ] every non-Manhattan plane pair classified in writing
- [ ] boundary F-score improved by ≥ 0.05 on the annotated set
- [ ] plane IDs stable across seeds; renders byte-reproducible
- [ ] `planning/plane-diagnosis-findings.md` written
- [ ] a short note listing every field the client will need, as input to Phase 3
