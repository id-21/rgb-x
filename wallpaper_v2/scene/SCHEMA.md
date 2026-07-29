# `scene.json` v1 — the server/client contract

This is the boundary between the Python pipeline (`geometry.py` / `render.py`)
and everything downstream: a reference consumer (`consume.py`), a future
server, a future WebGL client, and any future Rust port. Changing it later is
expensive — every cached bundle and every client version is pinned to it — so
the conventions below are asserted by test (`scene/tests/test_schema.py`,
`scene/tests/test_emit.py`), not just described here.

## The bundle

```
scene.json      ~2-6 KB    camera, planes, regions, quality
maps.png        ~30-60 KB  8-bit: R = region id, G = unused/reserved, B = alpha
shading.png     ~50-100 KB 16-bit single-channel: shading
photo.jpg       ~200-400 KB the source image, possibly downscaled
```

A consumer holding only these four files — no `.npy`, no depth, no normals,
no access to the pipeline — must be able to reproduce the render. That claim
is what `consume.py` (P3-T5) exists to prove.

Shading was originally packed as `maps.png`'s G channel (8-bit). P3-T3
measured that this causes visible banding in the composited render (see
"Encoding budget" below) and split it out into its own 16-bit PNG instead.

## `maps.png` channel packing (8-bit)

| channel | contents | encoding |
|---|---|---|
| R | region id | `id + 1`, so 0 means "no region". Max 254 regions. |
| G | unused | reserved, always 0 |
| B | alpha | feathered wall mask, `round(alpha * 255)` |

## `shading.png` (16-bit single channel)

`round(clip(shading, 0, scale) / scale * 65535)`, decode:
`shading = (v / 65535) * scale`. `scale` is `scene.json.shading.scale`
(default 1.6, matching `render.shading_from_irradiance`'s clip ceiling).

Both PNGs are written with `cv2.IMWRITE_PNG_COMPRESSION=6`; PNG is lossless,
which matters — a lossy region-id channel would be nonsense (ids would bleed
into each other at region boundaries), and 8-bit-quantised-then-recompressed
shading would reintroduce the banding 16 bits was chosen to avoid.

## Decisions, and why they are not to be re-litigated later

**Regions, not planes, are in the R channel.** Planes are geometry; regions
are selection. A plane's UV is shared across all of its regions, so pattern
continuity across a doorway (one physical wall, two regions) is preserved
regardless of which region the user selects. This is the Phase 2 T4 finding
and it is load-bearing — see `geometry.extract_regions`'s docstring.

**`u_axis`/`v_axis`/`origin` are emitted explicitly**, even though they are
derivable from `normal` and `gravity`. Re-deriving them client-side means
reimplementing `geometry.plane_basis`'s tie-break for the degenerate
horizontal-plane case and hoping it matches exactly. Nine floats is cheaper
than that risk.

**`camera.f_px`, `cx`, `cy` are in the units of `image.width`/`image.height`
in this same file.** Never the native-resolution photo's units, never the
model's internal working resolution. A mismatch here reproduces the
`fx_multiplier` bug documented in `wallpaper_v2/README.md`'s "On scale"
section, which cost this project months. `scene/tests/test_emit.py` asserts
`f_px / image.width` is invariant across `max_width` settings to 1e-6.

**Shading is stored linearly, not sRGB-encoded.** It is a multiplier on
light, not a colour, and encoding it would break the linear-space compositing
fix (`render.composite(linear=True)`). The G channel is a plain linear
quantisation of `shading / scale`, not a gamma curve.

**Plane `d` is always > 0.** `geometry.fit_planes` flips `(n, d)` so the
plane sits in front of the camera; `scene-v1.schema.json` enforces
`d > 0` as a hard requirement, not just an expectation.

## Sign and coordinate conventions (asserted, not just described)

**Camera frame: OpenCV convention.** +x right, +y down, +z away from the
viewer. This is the frame `depth`, `normal` (after `geometry.load_normals`'s
z-flip), `gravity`, and all plane geometry are expressed in. It is NOT the
RGB→X model's native frame (OpenGL-style, +z toward the viewer) — that
flip already happened upstream, in `load_normals`, and is not repeated here.

**Gravity points UP** (`gravity` is the vector from ground to sky, i.e. it
has been sign-corrected so that `gravity[1] < 0` in image-space, since image
y grows downward). See `geometry.estimate_gravity`.

**Plane equation is `n · X = d` with `d > 0`.** A point `X` in camera space
lies on the plane iff its dot product with the unit normal equals `d`. Since
the camera looks down +z and `d > 0`, every plane is guaranteed to be in
front of the camera. `scene/tests/test_emit.py`'s sign-convention test
asserts this directly: for sampled pixels belonging to each plane,
`t = d / (n · K⁻¹p) > 0`, i.e. the ray/plane intersection parameter is
positive and the hit point is not behind the eye.

**`v_axis` is gravity-aligned, pointing up the wall** (see
`geometry.plane_basis`); `u_axis = v_axis × n`, forming a right-handed
on-plane frame together with `n`.

**Irradiance/shading polarity:** already resolved upstream by
`render.shading_from_irradiance` before this schema sees it. `shading` here
is a plain multiplier where 1.0 means "as lit in a well-exposed part of the
wall" — no further inversion needed by a consumer.

## Field reference

See `scene-v1.schema.json` for the authoritative types/ranges. Notes beyond
what JSON Schema can express:

- `pipeline_sha`: provenance only (git short hash of the commit that produced
  the bundle). Not used by any client; safe to ignore.
- `planes[].model_vs_depth_deg`: high values (`quality.max_model_vs_depth_deg`
  aggregates this) flag glass/mirror/TV-screen suspects, where Depth Pro sees
  through to the scene beyond and disagrees with the model normal. UI hint
  only, not used by the renderer.
- `planes[].suggested_label`: human-readable guess (`geometry.suggested_label`),
  UI hint only.
- `regions[].plane`: must index an existing `planes[].id`. JSON Schema cannot
  express this cross-reference; `scene/tests/test_schema.py` checks it
  separately, and it is one of the deliberately-broken cases in the
  known-bad test fixtures.
- Every plane must have at least one region referencing it, and every region
  must reference an existing plane — enforced by `scene/tests/test_emit.py`,
  not by the schema (a single-sample structural invariant, not a type
  constraint).

## Encoding budget (P3-T3 decision)

Measured by `scene/experiments/encoding_budget.py` across all 12 samples,
output archived in `scene/experiments/encoding_budget.json`.

- **Shading: 16-bit.** 8-bit shading (step size `1.6/255 = 0.0063`) gives a
  final-render delta of up to **3/255** (worst case: `bed_back_bedroom`,
  `fluted_walls_living`), measured in the smooth wall interior
  (`alpha > 0.99`, to exclude unrelated feather-edge blending) — over the
  plan's 2/255 banding gate. PSNR stayed high (56.7–57.7 dB across samples)
  but the plan's gate is about visible banding, not aggregate PSNR, so the
  8-bit option was rejected. 16-bit round-trips with max error ~1.5e-5,
  eliminating banding. `maps.png`'s G channel is unused as a result; shading
  moved to its own `shading.png` (16-bit single-channel PNG).
- **JPEG quality: 92.** Lowest of {100, 95, 92, 85} clearing PSNR ≥ 40 dB on
  every sample (worst case 40.4 dB, at quality 92). Quality 85 dropped below
  the gate on at least one sample.
- **Median bundle size at these settings: ~1.75 MB**, measured emitting at
  native/`MAXPIXELS`-capped resolution (no `max_width` downscaling applied).
  This is well over the "well under 1 MB" aspiration in the phase plan's
  Definition of Done — almost entirely `photo.jpg` at full working
  resolution. Deliberately **not resolved here**: choosing a specific
  `max_width` for bundles served to a browser is a web-service/Phase-4
  concern (see "Out of scope for Phase 3"), not a pipeline encoding
  question — this task's job was to measure precision loss per format, not
  pick a target resolution. Whoever builds the server should re-run
  `encoding_budget.py`'s JPEG-quality measurement at whatever `max_width` it
  settles on, since PSNR at a given quality changes with resolution.

## Reference consumer (P3-T5 findings)

`scene/consume.py` reimplements ray/plane intersection, UV, texture sampling
and linear-space compositing from the bundle alone (no `geometry.py` import,
no `.npy` read — enforced by `scene/tests/test_consumer.py`'s isolation
test). Building it surfaced two real gaps in the emitter, both now fixed in
`scene/emit.py`, and one deliberate scope decision in how the resulting
render is compared against the pipeline's own output.

**Gap 1 — orphan plane pixels weren't in any region.**
`geometry.extract_regions` can drop an isolated same-plane fragment that has
no ADJACENT region to merge into (its own docstring documents this).
Production still renders those pixels — `run_all.process()` calls
`sample_texture` with the raw plane `labels`, never `region_labels` — but a
region-id-only `maps.png` has no way to recover a pixel with no region.
Measured impact before the fix: up to 228/255 max channel delta on
fully-opaque interior pixels on `complex_living`. Fixed by
`emit._backfill_orphan_plane_pixels`: route each orphan pixel into an
*existing* region of its own plane (correct because UV/texture are shared
across every region of a plane — the Phase 2 T4 finding — so rendering only
needs to know the plane, not which region label a pixel carries).

**Gap 2 — some orphans have no valid ray/plane intersection at all.**
`geometry.compute_uv` only drops pixels whose ray is nearly PARALLEL to the
plane (`|n·r| < grazing_tol`); it does not check the SIGN of the
intersection. A handful of isolated-fragment pixels (found on
`obstacle_course_living`: 144/62257 pixels of one region, forming a single
contiguous strip — almost certainly a foreground object silhouette
parallax-mismatched into the wall's normal cluster) have `t <= 0`: a
"hit point" behind the camera, violating the sign convention this document
asserts elsewhere. Backfilling them (Gap 1's fix) would ship pixels that
break `d > 0, t > 0` — a guarantee future consumers (a WebGL fragment
shader, say) may rely on to avoid dividing by a near-zero or
sign-flipped denominator. Resolution: `_backfill_orphan_plane_pixels` also
checks `t > 0` per candidate pixel and drops (leaves unassigned) any that
fail. Consequence, measured: a tiny number of plane-labelled pixels per
sample remain unreproducible from the bundle by construction —
`no_walls_living`: 190/~2.8M px (0.0068%), `obstacle_course_living`:
154/~4.9M px. This is judged the right trade: those pixels' own position is
already geometrically meaningless in production's render (it computes the
same `t <= 0` intersection, just doesn't check it), so losing them from the
bundle is not a completeness regression worth having over a documented,
testable invariant.

**Gate scoping — PSNR is the gate; max-channel-delta is reported, not
gated.** `scene/tests/test_consumer.py` compares `consume.render_from_bundle`
against the pipeline's own render **built from the same geometry pass**
(not a separately-committed `final.png` — cluster_planes' k-means is not
bit-reproducible across separate process invocations, per Phase 2 T0's
noise floor, and comparing across two different runs was initially measuring
that cross-process drift instead of bundle sufficiency). Two exclusions are
applied before comparing:
- pixels with no region at all (Gap 2's deliberate exclusion) can't be
  reproduced by construction and are excluded from the comparison, not
  padded over;
- the comparison is restricted to `alpha >= 0.99` ("interior") to separate
  actual rendering correctness from `photo.jpg`'s JPEG loss, which shows up
  everywhere alpha is low (pure photo passthrough) and is already measured
  and gated by P3-T3 (JPEG quality 92, PSNR ≥ 40 dB) — re-gating the same
  lossy-photo decision here under a stricter, undocumented threshold would
  double-count it.

Measured interior PSNR across all 12 samples: **51–81 dB**, comfortably
clearing the plan's 45 dB gate on every sample. Max channel delta in the
interior is 1/255 on 10 of 12 samples; the remaining two
(`no_walls_living`, `obstacle_course_living`) show a small band of larger
deltas (up to ~86–79/255, affecting 0.003%–0.13% of interior pixels)
immediately at the boundary of an excluded orphan patch. Traced to
`render.sample_texture`'s known `np.gradient`-at-an-array-edge footprint
artifact — the same narrow, bounded seam effect Phase 2 T4 already
documented and gated by fraction rather than a hard max
(`harness/test_regions.py`'s `test_render_invariance`). Because the
*fraction* affected scales with how large the excluded-orphan patch happens
to be on a given sample (not a fixed size), no single fixed threshold — max
delta or fraction — is a meaningful gate here; PSNR is. Max-delta is printed
per sample as a diagnostic, not asserted.

**Schema fields with no measurable render effect: `gravity`.** P3-T5's
ablation test corrupts each render-relevant field and confirms the render
changes; all 10 fields it exercises (camera intrinsics, plane normal/d/axes/
origin, `shading.scale`, `pattern.default_width_m`) are load-bearing.
`gravity` is deliberately excluded from that set: it has no effect on
`consume.py`'s render, because every plane's `u_axis`/`v_axis`/`origin` are
already derived from it upstream (`geometry.plane_basis`) and baked into
`scene.json` directly — see "`u_axis`/`v_axis`/`origin` are emitted
explicitly" above. It is retained for provenance and orthogonality
validation (a future consumer computing a NEW plane not in the original set,
or a QA check that `v_axis` is genuinely gravity-aligned), not because
rendering consumes it. This is a documented exception, not an oversight —
UI-only/diagnostic fields (`px_share`, `confidence`, `model_vs_depth_deg`,
`suggested_label`, `regions[].npix`/`area_m2`/`centroid_px`/`bbox_px`,
`quality.*`, `version`, `pipeline_sha`) are excluded from the ablation set
for the same reason: SCHEMA.md already documents them as not consumed by
any renderer, so testing that corrupting them has "no effect" would just be
re-confirming their own documentation.

## Sub-schemas / non-goals

- No WebGL, shader, or browser-specific fields here — out of scope for
  Phase 3 (see `planning/phase3-scene-contract-plan.md`).
- No per-wall texture assignment or SKU catalogue — a server-side concern
  layered on top of this bundle, not part of it. `pattern.default_width_m`
  is the only pattern-scale field; per-SKU overrides live in a catalogue the
  server owns.
