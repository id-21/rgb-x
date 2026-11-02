# How the renderer applies a texture, and where scale comes from

Reference for diagnosing scale problems. Every formula below is the actual code in
`wallpaper_v2/webapp/index.html`; the Python equivalent is `scene/consume.py`.

---

## The chain, end to end

```
screen pixel
   │  needs: f_px, cx, cy                    ← scene.json camera
   ▼
ray direction (unitless, camera frame)
   │  needs: plane normal n, offset d        ← scene.json planes[]
   ▼
3D hit point, IN METRES
   │  needs: origin, u_axis, v_axis          ← scene.json planes[]
   ▼
position on the wall, IN METRES  (u_m, v_m)
   │  needs: pattern width in metres         ← textureAssets.json  "scale"   ★
   ▼
position in TILE UNITS  (1.0 = one full repeat)
   │  fract() → wraps into a single tile
   ▼
texel → colour → × shading → composite
```

★ is the only place the physical size of the wallpaper enters. Everything above it is
geometry; everything below it is sampling.

---

## Step by step, with the real numbers

Using `living_simple2`, plane 0 ("right wall"), a 0.53 m Botany House SKU.

### 1. Pixel to ray

```glsl
vec3 ray = vec3((x - cx)/f, (y - cy)/f, 1.0);
```

`f_px = 473.37`, `cx = 512`, `cy = 341`. A ray is a *direction*, not a position — it
has no scale yet. Doubling `f` narrows the field of view and changes how the wall
projects, which is why a wrong `f` distorts perspective.

### 2. Ray to wall

```glsl
float t = d / dot(n, ray);
vec3 hit = ray * t;              // metres
```

`d = 2.97 m` — the perpendicular distance from camera to that wall, and the **only
number in the whole pipeline that carries physical scale**. It comes from Depth Pro's
metric depth. Everything metric downstream is proportional to it.

If `d` were 30% too large, every wall would be 30% too big, and a fixed-size
wallpaper would tile 30% more times across it.

### 3. Wall position in metres

```glsl
vec3 rel = hit - origin;
float u_m = dot(rel, u_axis);    // metres along the wall, horizontal
float v_m = dot(rel, v_axis);    // metres along the wall, vertical (gravity-aligned)
```

`u_axis` and `v_axis` are unit vectors lying in the plane, so these dot products are
distances in metres. `origin` just fixes where the pattern starts — it shifts the
tiling, it does not scale it.

For this wall the values span **4.16 m × 3.22 m** (`extent_m` in `scene.json`).

### 4. Metres to tile units — the scale step

```glsl
float pw = uRegW[rid];                       // = asset.scale, metres
vec2 uv = vec2(u_m, -v_m) / pw;
```

**Both axes are divided by the same number.** `uv = (1.0, 1.0)` means "one full
repeat right and one full repeat down".

So across this wall: `4.16 / 0.53 = 7.8` repeats horizontally, and
`3.22 / 0.53 = 6.1` repeats vertically.

### 5. Tile units to texel

```glsl
vec2 ddx = dFdx(uv), ddy = dFdy(uv);         // gradients BEFORE any half-drop
uv.y += off * floor(uv.x) * 0.5;             // offset/half-drop repeats only
vec3 tex = textureGrad(uTex, vec3(fract(uv), layer), ddx, ddy).rgb;
```

`fract()` wraps into `[0,1)`, which the sampler maps onto the whole texture image.
So **the texture image is assumed to cover exactly one repeat**, corner to corner.

### 6. Shade and composite

```glsl
float sh = texture(uShade, uv0).r * shading.scale;
vec3 lit = srgbToLinear(tex) * sh;
out      = linearToSrgb(mix(srgbToLinear(photo), lit, alpha));
```

Shading and alpha are per-screen-pixel lookups into `shading.png` and `maps.png`.
Neither affects scale.

---

## The assumption that decides whether scale is right

Step 4 divides **u and v by the same `pw`**. That is correct **only if the texture
image represents a square of wall** — `scale` metres wide *and* `scale` metres tall.

The Python reference does not make this assumption:

```python
pattern_height_m = pattern_width_m * (Ph / Pw)      # from image aspect ratio
tx = u_m / pattern_width_m  * Pw
ty = -v_m / pattern_height_m * Ph
```

The two agree **only when the source image is square in pixels**. Your GCS images are
1000×1000, so they currently agree — but that is a coincidence of the asset library,
not a property of the contract. A non-square source would render stretched in the
browser and correct in Python.

**What is not verified: whether a 1000×1000 image that is 0.53 m wide is also
0.53 m tall.** Nothing in the data says so. If the image is one roll-width across but
shows a different vertical rapport, the pattern is vertically compressed or stretched.

Suggestive detail from your catalogue: **0.53 m is the standard European roll width,
but 0.64 m is not a standard roll width** — it *is* a common vertical pattern repeat.
So `scale` may not mean the same thing for all 685 entries.

---

## Two independent ways scale can be wrong

They produce different symptoms, which is how to tell them apart.

| | cause | symptom | where to look |
|---|---|---|---|
| **A** | `scale` semantics — image is not a square metre-for-metre | pattern correct on one axis, stretched or squashed on the other | supplier spec: is `scale` roll width or vertical rapport? |
| **B** | metric scale — Depth Pro's `d` is off | pattern proportionally correct, but uniformly too small or too large | measure a known object in the photo |

**A leaves aspect ratio wrong. B leaves aspect ratio right and size wrong.**

If the pattern looks correctly proportioned but simply too fine, suspect B. If motifs
look subtly elongated or squat compared to the swatch, suspect A.

### Test for B — one measurement settles it

Pick something in a photo whose real size you know: a TV, a door (~2.0 m tall), a
standard sideboard, the ceiling height. Then:

```
reported_size / true_size = the scale error factor
```

Current reported values for `living_simple2`:

| quantity | reported |
|---|---|
| room vertical extent | 3.55 m |
| camera → right wall (`d`) | 2.97 m |
| right wall extent | 4.16 × 3.22 m |
| back wall extent | 3.63 × 3.20 m |

A 3.55 m ceiling is tall but not impossible for that building. If the true ceiling is
2.7 m, everything is inflated by **1.31×** and the wallpaper renders 1.31× too small.

### Test for A — compare axes

Render a SKU with a strongly directional motif and compare a single repeat against
the flat swatch. Equal proportions means the square assumption holds for that asset.

---

## What to fix once the answer is known

**If A (semantics):** add explicit `repeat_width_m` and `repeat_height_m` per SKU
rather than one `scale`, and change step 4 to divide each axis by its own value.
Also fixes the **58 assets with `scale: 0`** currently defaulting to 0.53 m.

**If B (metric):** the renderer is correct and the fault is upstream in Depth Pro.
Options are a global calibration factor, or a user-facing "set ceiling height" control
that rescales `d` per scene.

**Either way**, the definition belongs in `scene/SCHEMA.md`. Right now the rule "one
texture image = one repeat, square in metres" exists only as an unwritten assumption
inside one shader, and a second renderer would have no way to know it.
