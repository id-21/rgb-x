# Wallpaper Visualiser — strategy for a live web service

Written against `wallpaper_v2/`. Every number here traces back to something measured
on the 12-sample set, or is flagged as an estimate.

---

## 0. The one architectural insight that shapes everything

**The renderer does not need a UV map.** It doesn't need depth, normals, or
irradiance either.

From `compute_uv`, the entire mapping from screen pixel to wallpaper coordinate is:

```
ray  = K⁻¹ · p                     # 3 multiply-adds
hit  = ray · d / (n · ray)          # 1 dot, 1 divide
uv   = ((hit − origin)·u_axis, (hit − origin)·v_axis)
```

Per plane that is **13 floats**: `n` (3), `d` (1), `u_axis` (3), `v_axis` (3),
`origin` (3). Per image, a further 4: `f, cx, cy` and the image size. Everything
else in the pipeline exists only to *derive* those numbers.

So the server's job is to turn a photo into ~60 floats plus a label map, and the
client's job is a single fragment shader. This is the difference between shipping
tens of megabytes of per-pixel buffers and shipping ~200 KB.

It also means the second thing worth internalising:

> **"2 seconds per 4K image" is the wrong target.**

There are two latencies, and they have nothing to do with each other:

| | what happens | budget | why |
|---|---|---|---|
| **Scene prep** | models run, scene bundle produced | once per photo, cacheable forever | user sees a progress state |
| **Wallpaper change** | swap a texture, redraw | must be < 16 ms | this is the actual product |

Scene prep at 5 s behind a good progress state is a fine product. Wallpaper change
at 200 ms is a bad one. Optimise accordingly — and note that with this
architecture, wallpaper change is already a texture bind and a redraw.

**Stated project priority: realism over latency.** Scene prep may be async and slow
if that buys accuracy. This document is written to that constraint — where a
speed/quality trade-off appears below, it resolves toward quality, and the
resolution section in particular reaches the opposite conclusion from an earlier
draft because of it.

---

## 1. Models

### What the pipeline actually needs

| model | gives us | what we do with it |
|---|---|---|
| **SegFormer-B2** (ADE20K) | wall mask | which pixels are wall |
| **Depth Pro** | metric depth + focal length | `f` (all the perspective) and `d` (one scalar per plane) |
| **RGB→X** | normals + irradiance | plane orientation `n`, and the shading field |

- SegFormer-B2: [`nvidia/segformer-b2-finetuned-ade-512-512`](https://huggingface.co/nvidia/segformer-b2-finetuned-ade-512-512) — ~28 M params. Trivial: well under 1 GB VRAM, ~20 ms.
- Depth Pro: [`apple/ml-depth-pro`](https://github.com/apple/ml-depth-pro) — ViT-based, ~1.9 GB checkpoint. Apple's claim is [sharp metric depth "in less than a second"](https://github.com/apple/ml-depth-pro) (0.3 s on a V100). Community reports [~5 GB VRAM at fp16](https://learnopencv.com/depth-pro-monocular-metric-depth/) unoptimised; a [TensorRT port](https://github.com/yuvraj108c/ml-depth-pro-tensorrt) exists.
- RGB→X: [project page](https://zheng95z.github.io/publications/rgbx24) / [arXiv 2405.00666](https://arxiv.org/abs/2405.00666), SIGGRAPH 2024 (Adobe).

### The problem: RGB→X is a diffusion model

This is the thing to confront before you build anything. RGB→X is a **Stable
Diffusion fine-tune** — a ~1 B-parameter UNet run for N denoising steps. It is one
to two orders of magnitude slower than the other two models combined, and it needs
a pass per intrinsic channel. You need two (normals, irradiance).

Rough budget on one A100, models at 768²:

| stage | time |
|---|---|
| SegFormer | ~0.02 s |
| Depth Pro | ~0.3 s |
| RGB→X ×2 channels @ ~10 steps | **~3–6 s** |
| geometry + encode | ~0.2 s |

**Sub-2 s is not reachable with RGB→X in the critical path.** Given realism outranks
latency, the right response is to accept that and go async — not to trade quality
for speed. The material below is therefore about *understanding* the cost, and about
the one change (batching) that is free.

**Do not treat this as one binary decision.** RGB→X produces two things — normals
and irradiance — and they have completely different replacement stories. Split them.

#### Normals: worth an experiment, but a narrower one than it first appears

You need *four plane normals*, not a per-pixel normal field, and Depth Pro already
gives you a metric point cloud. Sequential RANSAC on it (fit best plane → remove
inliers → repeat) would plausibly give the same four normals.

Two caveats that make this less of a slam-dunk than it looks:

- **Normals are used for clustering, not just fitting.** `cluster_planes` runs
  k-means on the *per-pixel* normal field to decide which pixels belong to the same
  wall. Remove RGB→X and you lose the grouping signal, not just the fitting signal.
  The substitute is a depth-gradient normal (cross product of finite differences),
  which inherits exactly the weakness that broke the earlier SVD fit — over glass,
  Depth Pro sees the buildings outside, so the gradient normal there is garbage.
  The failure just moves one stage earlier, from fitting to clustering.
- **Plain single-shot RANSAC fits one plane.** Multi-plane needs sequential RANSAC
  or region-growing segmentation. That's a real algorithm, not a substitution.

So the experiment must be evaluated **specifically on the hard cases** — window-heavy
and cluttered rooms — not averaged over the set. `obstacle_course_living` and
`complex_living` already produce spurious 29–58° plane pairs *with* learned normals;
there is no reason to expect depth-only does better on exactly the scenes already
failing. A method that ties on clean rooms and loses on cluttered ones is not a win,
because cluttered rooms are the realistic ones.

#### Irradiance: do not try to get this from geometry

The `shading_from_luminance` fallback is **not** an acceptable substitute, and an
earlier draft of this document was wrong to suggest it. It re-introduces the exact
defect that `wallpaper_v2` fix #5 was written to remove. Measured over all 12
samples — correlation between the high-frequency content of the shading field and
the high-frequency content of the original photo, wall pixels only:

| | irradiance | LAB luminance |
|---|---|---|
| median corr(HF_shading, HF_photo) | **0.42** | **0.998** |
| HF energy in wall region | 1.0× | 1.4–3.7× |

At 0.998 the LAB shading field does not merely resemble the old wall, it *is* the
old wall. Sockets, picture-frame edges and TV outlines get multiplied straight into
the new texture. Irradiance's 0.42 is expected and legitimate — it is derived from
the photo, and creases and light falloff genuinely do align with image structure.

Irradiance needs its own model. The good news is it doesn't need a diffusion one:
feed-forward intrinsic decomposition is a mature area. [compphoto/Intrinsic](https://github.com/compphoto/Intrinsic)
(Aksoy's group) is the obvious first thing to benchmark — [Ordinal Shading](https://dl.acm.org/doi/10.1145/3630750)
(TOG 2023) and [Colorful Diffuse Intrinsic Decomposition in the Wild](https://yaksoy.github.io/ColorfulShading/)
(TOG 2024), both single-pass CNNs producing albedo and diffuse shading. Newer
transformer approaches like [IDT](https://arxiv.org/html/2512.23667) are also
feed-forward.

#### If you do want it faster: shrink RGB→X rather than drop it

Lower priority now that realism is the stated goal, but worth knowing. Diffusion
cost scales with **latent area** and **step count**:

- **Batching — do this regardless.** Normals and irradiance as a batch of 2 in one
  forward pass, not two sequential calls. Costs no quality at all.
- **Resolution.** Fewer latent tokens is cheaper, but note the resolution findings
  below before assuming this is free: measure the quality plateau and stop there,
  don't chase a latency number past it.
- **Steps.** Measure quality against step count on your own samples. Intrinsic
  estimation is a far more constrained problem than open-ended generation and often
  converges in very few steps. LCM/turbo-style distillation is a further option.
These could plausibly take 3–6 s to around a second, and it is a better risk profile
than replacing a model you have already validated. But every one of them except
batching is a quality gamble, so each needs a measured plateau before adoption.

#### The recommended default

**Keep RGB→X at full quality, accept 3–8 s, hide it behind an async job.** Scene prep
is once-per-photo and cacheable, and the interactive loop — changing wallpaper — is
unaffected because it happens client-side. Given realism is the priority, this is
not a compromise; it is the correct answer, and the other options above are only
worth revisiting if the wait turns out to hurt conversion.

### Resolution: what is actually safe to shrink

An earlier draft asserted "run the models at 768², plane fits will not measurably
change" with no number attached — the one claim in this document held to a lower
evidentiary standard than everything else. `experiments/resolution_sensitivity.py`
now measures it. **The result does not support the original recommendation.**

Two questions were being conflated, and they have different answers:

#### (1) Sampling density — is the geometry stable with fewer samples? Mostly yes.

Model outputs downsampled to 768 px long edge, geometry re-run, 11 samples:

| quantity | worst case across the set |
|---|---|
| plane normal direction | **0.72°** |
| plane offset `d` | **0.031 m (2.1%)** |
| gravity vector | **0.20°** |

Negligible, and for the reason you'd expect: `d` is a median over 10⁴–10⁶ points and
`n` is a mean over the same. Extra samples buy redundant agreement, not information.

**But there is one real failure, and it is where you'd predict.**
`obstacle_course_living` **lost a plane** — its smallest was 2.6% of wall pixels, and
at 768 px it fell below `cluster_planes`'s `min_frac = 0.02` and vanished. Thin
slivers (a strip between two windows, a small return wall) are the failure mode, and
it landed on one of the two samples already known to be marginal. `simple_living`
(smallest plane 2.3%) also produced the worst normal error, 0.72°.

So: sampling density is safe for *large* planes, and unsafe for exactly the small
ones that per-wall selection is going to make visible to users.

#### (2) Model behaviour — does the network give the same answer? Untested, and one answer inverts the advice.

The script above downsamples model *outputs*. It says nothing about what a network
does when fed a smaller *input*. That's the riskier question, and focal length is the
sharpest case: it is not an average over the depth map but a **separate prediction
from a small convolutional head** on ViT features. Nothing about the averaging
argument protects it. Given `fx_multiplier = 0.8` already distorted reconstructed
room height by 1.73× once, this is not a number to take on faith.

**Depth Pro resizes every input to a fixed 1536×1536 internally** ([issue #18](https://github.com/apple/ml-depth-pro/issues/18),
[arXiv 2410.02073](https://arxiv.org/pdf/2410.02073)) — chosen as a multiple of the
384² ViT patch size, then further downsampled internally to 768² and 384² for the
multi-scale encoders. Which means:

- Feeding it **more than ~1536 px** on the long edge buys nothing; it gets discarded.
- Feeding it **768 px is actively harmful** — it gets upsampled back to 1536² with no
  detail to recover. The original recommendation was wrong in this specific way.
- **Feed Depth Pro ~1536 px.** That is its canonical resolution.

This is model-specific and must be checked per model, not assumed. RGB→X, being a
latent diffusion model, has its own preferred operating resolution and its own
degradation curve — measure it separately rather than generalising from Depth Pro.

#### Irradiance: smooth almost everywhere, and the error goes exactly where you'd fear

Downsample to 768, bilinear upsample back, measure error against native, split by
local gradient:

| region | RMS error |
|---|---|
| smooth falloff (bottom 50% gradient) | **0.0003 – 0.0011** |
| sharp edges (top 10% gradient) | **0.0050 – 0.0142** |
| edge error as % of dynamic range | **1.3 – 3.8%** |

The error is **~20× concentrated at high-gradient regions** — hard shadow boundaries,
a window mullion's cast line, a doorframe edge. The physical claim holds (ambient
light on a flat wall genuinely is smooth), and so does the caveat: the cost is not
distributed, it lands entirely on the shadow edges, which is where the eye goes.

Worth being precise about the failure mode: a low-res shading field bilinearly
upsampled and composited under a full-res texture reads as **soft**, not blocky.
Pixelation comes from nearest-neighbour upsampling specifically. And most of what
reads as "sharp" in these renders is the wallpaper's own texture detail, which is
rendered client-side at full resolution regardless. So this is a real but bounded
cost — softer shadow boundaries, not a visibly degraded image.

#### Recommendation

**Given realism outranks latency: do not downsample the models.** The measured
upside is latency only; the measured cost is a lost plane on the hardest sample and
20×-concentrated error at shadow edges. Run Depth Pro at its native 1536², run
RGB→X at whatever resolution you measure as its quality plateau, and put the time
into an async job with good progress UX instead.

The one thing that genuinely *is* safe: the **geometry** stage can subsample the
model outputs for k-means and plane fitting (it already does — `cluster_planes` caps
at 200k samples). That's free. It's the model inputs that shouldn't shrink.

If you later need latency, revisit this with `min_frac` lowered and a small-plane
guard, and re-run the experiment rather than trusting these numbers — they were
measured on downsampled outputs, not re-inferred models.

### The mask is the exception

Only the **mask** needs full resolution, because its edges are directly visible. But
don't tile the image to get it: semantic segmentation depends on wide-context
attention, and a 500×500 crop of a flat beige surface with no visible corner, floor
line or furniture is genuinely ambiguous between wall, door, curtain and cabinet.
You would also get seams at tile boundaries and an apparent object scale the model
was never trained on.

The right pattern is **global context + local refinement**: run SegFormer once on
the whole image at its native input size for the semantics, then sharpen the
boundary against the full-res photo with a guided filter or matting — which uses
local pixel affinity, not a second segmentation pass. If you do want tiled inference
for edge sharpness, the standard technique is *overlapping* sliding-window tiles with
blended overlaps, still combined with a global low-res pass — never tiling alone.

### Serving

- One GPU worker holding all models resident (~8 GB VRAM, comfortable on an A10/L4).
- fp16, `torch.compile`, or TensorRT for Depth Pro.
- **Scale to zero.** Traffic will be spiky. Modal / RunPod / Replicate-style
  serverless GPU, or your own autoscaler. Cold start is the thing to measure —
  loading 2 GB of weights dominates.
- **Cache scene bundles by image hash.** Model output is deterministic. A user
  trying 30 wallpapers should hit the GPU exactly once.

---

## 2. The wire format

This is the contract between server and client, and it is the most expensive thing
to change later. Get it right before you optimise anything else.

```
scene.json          ~2 KB     camera, planes, metadata
photo.jpg           ~300 KB   the original, possibly downscaled
maps.png            ~50 KB    R = plane id, G = shading, B = alpha
```

```jsonc
{
  "version": 1,
  "image": { "width": 3840, "height": 2560 },
  "camera": {                      // Depth Pro, rescaled to image size
    "f_px": 1774.3, "cx": 1920.0, "cy": 1280.0
  },
  "gravity": [-0.024, -0.999, -0.031],
  "planes": [
    {
      "id": 0,
      "normal": [0.880, -0.004, 0.474],
      "d": 2.97,                   // metres; the ONLY thing depth contributes
      "u_axis": [0.474, 0.0, -0.880],
      "v_axis": [-0.004, -0.999, -0.007],
      "origin": [0.31, 0.02, 3.11],
      "area_m2": 16.3,
      "extent_m": [4.97, 3.69],
      "centroid_px": [2790, 900],   // for UI hit-testing and label placement
      "suggested_label": "right wall",
      "confidence": 0.94
    }
  ],
  "diagnostics": {                 // from verify.py — see quality gates below
    "vertical_extent_m": 3.55,
    "wall_pair_angles_deg": [78, 88],
    "gravity_tilt_deg": 1.9,
    "gravity_ambiguous": false
  }
}
```

**`maps.png`** packs three single-channel rasters into one RGB PNG:

- **R — plane id.** Small integers, so PNG compresses it to almost nothing.
- **G — shading.** Already inverted and normalised (see the polarity fix). Smooth,
  so it survives half resolution fine.
- **B — alpha.** The feathered wall mask.

One fetch, one decode, one GPU upload. Ship it at half resolution and let bilinear
sampling handle the rest — except consider keeping alpha at full resolution if edge
quality suffers, since that's the channel with real high-frequency content.

Emit `u_axis`/`v_axis`/`origin` explicitly rather than deriving them client-side.
They're 9 floats, and it guarantees the client's basis matches the reference
implementation exactly rather than re-deriving it from `gravity` and hoping the
tie-breaking matches.

---

## 3. The renderer

### It's a fragment shader

```glsl
// per fragment
int   id  = int(texture(maps, uv0).r * 255.0 + 0.5);
if (id == 255 || !enabled[id]) { out = texture(photo, uv0); return; }

vec3  ray = vec3((gl_FragCoord.xy - c) / f, 1.0);
vec3  hit = ray * (planeD[id] / dot(planeN[id], ray));
vec3  rel = hit - planeOrigin[id];
vec2  uv  = vec2(dot(rel, planeU[id]), -dot(rel, planeV[id])) / patternSize[id];

vec3  tex   = texture(patterns, vec3(uv, layer[id])).rgb;   // GL_REPEAT + aniso
float shade = texture(maps, uv0).g * shadeScale;
float a     = texture(maps, uv0).b;

// composite in LINEAR space, then re-encode
vec3 lin = srgbToLinear(tex) * shade;
out = vec4(linearToSrgb(mix(srgbToLinear(texture(photo, uv0).rgb), lin, a)), 1.0);
```

Notes that matter:

- **Pass UV unwrapped and let `GL_REPEAT` tile.** Then hardware derivatives are
  correct and you get correct mip selection for free. If you `fract()` the UV
  yourself, derivatives explode at every tile boundary and you get a visible seam
  grid. If you need to be exact at plane boundaries, use `textureGrad` with
  analytically-derived derivatives.
- **`EXT_texture_filter_anisotropic` gives you, in hardware, exactly the
  anisotropic filtering we implemented in numpy.** Set `TEXTURE_MAX_ANISOTROPY` to
  16. This is the single most important quality setting in the renderer.
- **Composite in linear space.** Our Python does the shading multiply on sRGB uint8
  values, which is subtly wrong — it crushes midtones. The shader should decode to
  linear, multiply, then re-encode. This will visibly improve realism and costs
  nothing.

### Should you write it in Rust?

Partly. The useful split is:

**No** for the pixel work. The per-fragment loop belongs on the GPU. Rust compiled
to WASM runs on the CPU and will lose to a fragment shader by a very large margin
for this workload. "Rust for speed" is the wrong axis here — the win comes from
being on the GPU at all.

**Yes** for a shared geometry core, and this is genuinely valuable. `geometry.py`
is ~300 lines of linear algebra with no I/O. As a Rust crate it compiles to:

- a **native** binary or Python extension (PyO3) for the server, and
- **wasm32** for the browser,

from one source. That eliminates the entire class of bug where the server and
client disagree about what a plane basis is — which, given this project has already
been bitten by three separate convention mismatches, is not a hypothetical risk.

If you want one codebase for web *and* a future native desktop app, **wgpu** (Rust,
targets WebGPU + Vulkan/Metal/DX12) is the right choice and WGSL replaces GLSL.

**But don't start there.** Sequence it:

1. **Phase 1** — Python server, TypeScript + WebGL2 client. Prove the format.
2. **Phase 2** — port `geometry.rs` once `scene.json` has stopped changing.
3. **Phase 3** — wgpu only if you actually need native.

### How to avoid a rewrite: keep the Python as the reference implementation

You already have the thing that solves this problem — `wallpaper_v2` is a working,
verified renderer. Make it the oracle rather than throwing it away:

- **Golden scene tests.** Python emits `scene.json` for all 12 samples and commits
  them. Any reimplementation must reproduce every float within tolerance.
- **Golden frame tests.** The numpy renderer produces reference PNGs. The WebGL
  renderer runs headless (Playwright + `readPixels`) on the same inputs, and CI
  fails if the perceptual diff exceeds a threshold. Expect small differences from
  hardware aniso vs our 16-tap approximation — set the threshold empirically, don't
  demand exact equality.
- **Keep `verify.py` in the serving path** as a quality gate, not just a dev tool.

That's what stops drift: not writing the same language twice, but having an
executable definition of correct.

---

## 4. Texture storage

Your instinct — small thumbnails load fast but look pixelated — is the mip-mapping
problem, and the answer is that you need *both* resolutions, which is what a mip
chain is.

**First, the size you actually need is much smaller than you'd think.**

A 3840 px-wide render, wall spanning ~55% of frame (~2100 px) covering ~5 m of real
wall, pattern repeat 0.53 m:

```
screen px per pattern repeat = 2100 × (0.53 / 5) ≈ 223 px
```

Even with 2× headroom for a closer wall, **512² per repeat is enough and 1024² is
generous.** Your current `30106.jpg` is 1502² — already past the point of
diminishing returns. You do not need 4K textures.

**Recommended pipeline:**

1. **Store one seamless repeat unit**, not a sheet of repeats. `GL_REPEAT` tiles for
   free; storing 3×3 repeats wastes 9× the memory for nothing.
2. **KTX2 + Basis Universal.** Transcodes on-device to BC7 (desktop) / ASTC
   (mobile) / ETC2. Stays compressed in VRAM — roughly 4–8× less than RGBA8 — and
   uploads without a CPU decode step. JPEG/PNG must be decoded to RGBA8 and cost
   full VRAM.
3. **Ship the full mip chain in the file.** Do not let the GPU generate mips at
   runtime; precompute them with a good filter. This is what kills the pixelation
   *and* the aliasing.
4. **Stream mips progressively.** KTX2 supports level-by-level loading: fetch the
   64² level first (~2 KB, renders instantly, slightly soft), then the rest.
   Perceptually this reads as "instant, then sharpens" rather than "blank, then
   pops".
5. **Verify seamlessness at authoring time.** A tile that doesn't wrap produces a
   visible grid that no amount of filtering hides. Worth an automated check in the
   catalogue ingest.

Budget: a 1024² Basis-compressed tile with mips is roughly **200–700 KB**. You can
prefetch a dozen while the user browses.

**Store `pattern_width_m` per SKU in the catalogue.** It's currently hardcoded at
0.53. Real wallpaper varies (rolls are commonly 0.53 m in Europe, 0.686 m in the
US, and panel/mural products are entirely different). Getting this wrong scales the
pattern wrongly — which was one of the original bugs, just relocated.

---

## 5. Per-wall selection

The architecture already supports this; the changes are modest but real.

**What already works:** planes have IDs, the shader reads an ID per pixel. Add a
per-plane `enabled` flag and `layer` index into a `sampler2DArray`, and you can
render different wallpapers on different walls **in a single pass**.

**What needs to change:**

1. **Split planes into connected components.** This is the important one. Plane
   clustering groups by *orientation*, so a single plane id can currently cover two
   physically separate walls that happen to be parallel — and `_split_by_offset`
   only catches them when their distances differ. Conversely one wall interrupted
   by a doorway should probably still be one selectable region. Run connected
   components in image space on each plane, then let coplanar-and-adjacent
   components re-merge. A user clicking the left wall must not wallpaper the right
   one.

2. **Stop discarding small planes.** `top_n=4` exists to keep the automatic result
   clean. For manual selection, return everything above a small area threshold with
   a `confidence` score, and let the UI decide what to show.

3. **Stable IDs across re-runs**, so saved designs survive a pipeline update. Derive
   them from something geometric — sorted by area, or by centroid position — not
   from k-means label order, which is arbitrary.

4. **Hit-testing.** Read the R channel of `maps.png` at the click point. Keep a CPU
   copy of that channel (it's small) rather than doing a GPU readback.

5. **Let the user correct the segmentation.** Automatic wall splitting will
   sometimes be wrong. Merge/split of plane IDs is a pure client-side remap of the
   ID → layer table, no server round trip. Cheap to build, and it rescues the cases
   where segmentation fails — which, on `obstacle_course_living` and
   `complex_living`, it currently does.

6. **UI affordances:** hover-highlight per plane (tint by ID in the shader), and use
   `centroid_px` + `area_m2` to place labels sensibly.

---

## 6. Things you didn't ask about but will hit

**Quality gates.** `verify.py` already computes the right signals — ceiling height,
wall orthogonality, gravity tilt, gravity ambiguity. Run it server-side and refuse
to serve a bad reconstruction. Better to say "we couldn't read this room, try a
wider shot" than to render a room with 1.9 m ceilings. The failure modes are known:
one visible wall makes gravity ambiguous; `obstacle_course_living` and
`complex_living` produce plane pairs at 29–58°.

**Mask edge quality.** The current mask includes ceiling soffits and window reveals,
and its edges are thresholded rather than matted. On a real product this is the most
visible defect after geometry.

The fix is a guided/joint-bilateral filter — but *what you guide it with* matters,
and the photo alone is the wrong choice. Colour edges and surface boundaries are not
the same thing: a picture frame on a wall has a strong colour edge and no surface
boundary at all, so a photo-guided filter will happily snap the mask to it.

Use a composite guide, because the two boundary types have different signatures:

| boundary | example | signal |
|---|---|---|
| **occlusion** | sofa in front of wall, doorway | large `\|∇D\|` — depth jumps |
| **crease** | wall meets ceiling, wall meets wall | depth is *continuous*; the **normal** flips |

This distinction is easy to get wrong. At a wall/ceiling junction the two surfaces
physically meet, so depth is C0-continuous and `|∇D|` shows no spike — a
depth-gradient guide alone will miss exactly the soffit case that's currently
wrong. Creases show up as a discontinuity in the *normal* field, not the depth field.

So: `|∇D|` for occlusion edges, normal discontinuity for creases, and the photo only
for sub-pixel localisation once the filter knows which side is which. Depth Pro's
depth is also a fairly smooth field, so it will not give pixel-accurate edges on its
own regardless.

Worth noting this is a third consumer of the normal field, alongside clustering and
plane fitting — another reason not to assume normals can be cheaply removed.

**Occluders.** Furniture in front of a wall is excluded by the mask, which is right.
But thin objects — plant stems, lamp cables, chair legs — will be missed and get
wallpapered over. Worth a known-limitation note and eventually a matting pass.

**Linear-space compositing.** Mentioned above; it's a real correctness bug we
carried over from the Python.

**Colour and white balance.** The wallpaper swatch is photographed under studio
light; the room is not. Some users will complain the colour "isn't right". Consider
a mild white-balance transfer from the room's irradiance to the texture.

**Async job pattern.** Upload → 202 + job id → poll or websocket → bundle URL.
Don't hold an HTTP connection open for 5 s; you'll hit proxy timeouts and mobile
network drops.

**Privacy.** These are photos of people's homes. Decide retention up front, publish
it, and make deletion real. Consider processing-then-discard with only the scene
bundle retained — it's ~200 KB and contains no recognisable interior.

**Format versioning.** `"version": 1` is in the schema above for a reason. Clients
will be cached in browsers long after you ship v2.

**Mobile GPU limits.** Check `MAX_ARRAY_TEXTURE_LAYERS` and confirm anisotropic
filtering is available — it's an extension, not core, and the fallback path needs to
look acceptable.

---

## Suggested sequence

1. **Measure RGB→X's quality-vs-resolution and quality-vs-steps curves** on your 12
   samples, and batch the two channels into one forward pass. Batching is free
   quality-wise. Resolution and step count are only worth reducing to the point
   where measured quality is flat — given realism outranks latency, stop there
   rather than pushing for a latency target.
2. **Freeze `scene.json` v1.** Emit it from the existing Python. Commit golden files
   for all 12 samples. Do this in parallel — it does not depend on the model work.
3. **Build the WebGL2 renderer** against those static files — no server yet. This is
   where you'll find out whether the format is missing anything.
4. **Golden-frame CI** comparing WebGL output to the numpy reference.
5. **Wrap the Python pipeline in an async job API** with caching by image hash.
6. **Texture pipeline**: catalogue ingest → seamless 1024² tile → KTX2 with mips.
7. **Per-wall selection**: connected components, then the UI.
8. **Benchmark a feed-forward intrinsic model** for irradiance, scored against
   RGB→X using the HF-correlation metric above — not by eye. Only adopt it if
   quality holds; this is a latency optimisation and latency is not the priority.
9. **Sequential-RANSAC normals experiment**, scored on the hard cases only. Lowest
   priority of the three model questions, for the same reason.
10. **Re-run `experiments/resolution_sensitivity.py` with real re-inference** at each
   candidate resolution, rather than downsampled outputs, before changing any model
   input size. Add a small-plane guard first — the current `min_frac = 0.02` already
   drops a real plane on `obstacle_course_living` at 768 px.
11. **Rust core** only once steps 2–5 have stopped moving.

---

## Sources

- [Depth Pro — apple/ml-depth-pro](https://github.com/apple/ml-depth-pro)
- [Depth Pro explained (LearnOpenCV)](https://learnopencv.com/depth-pro-monocular-metric-depth/)
- [ml-depth-pro TensorRT port](https://github.com/yuvraj108c/ml-depth-pro-tensorrt/)
- [RGB↔X project page](https://zheng95z.github.io/publications/rgbx24)
- [RGB↔X, arXiv 2405.00666](https://arxiv.org/abs/2405.00666)
- [RGB↔X, ACM SIGGRAPH 2024](https://dl.acm.org/doi/10.1145/3641519.3657445)
- [Depth Pro fixed 1536x1536 resolution (issue #18)](https://github.com/apple/ml-depth-pro/issues/18)
- [Depth Pro paper, arXiv 2410.02073](https://arxiv.org/pdf/2410.02073)
- [SegFormer-B2 ADE20K](https://huggingface.co/nvidia/segformer-b2-finetuned-ade-512-512)
- [compphoto/Intrinsic — feed-forward intrinsic decomposition](https://github.com/compphoto/Intrinsic)
- [Intrinsic Image Decomposition via Ordinal Shading, TOG 2023](https://dl.acm.org/doi/10.1145/3630750)
- [Colorful Diffuse Intrinsic Image Decomposition in the Wild, TOG 2024](https://yaksoy.github.io/ColorfulShading/)
- [IDT: feed-forward multi-view intrinsic decomposition](https://arxiv.org/html/2512.23667)
