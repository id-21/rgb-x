# Texture repeat conventions

How a wallpaper SKU maps onto a wall. Belongs in `scene/SCHEMA.md` — currently the
only definition of `repeatType` lives inside the webapp's fragment shader, which is
not a contract.

## Catalogue fields

From `textureAssets.json` (685 assets):

| field | meaning | distribution |
|---|---|---|
| `scale` | width of one repeat, **in metres** | 0.53 ×540, 0.64 ×75, 0.60 ×7, 0.59 ×5, **0 ×58** |
| `repeatType` | `"straight"` or `"offset"` | straight ×523, offset ×162 |

**58 assets have `scale: 0`** and currently fall back to 0.53 m. A wrong pattern
width is the exact failure this project spent months eliminating; fix it in the
catalogue rather than in the renderer.

## Straight repeat

Tiles align on a regular grid. `uv = (u, v)` in tile units, sample `fract(uv)`.

```
┌───┬───┬───┐
│ A │ A │ A │
├───┼───┼───┤
│ A │ A │ A │
└───┴───┴───┘
```

## Offset repeat (half-drop)

Tiles stack normally **within a column**. Each successive **column** starts half a
tile further **down**.

```
    ┌───┐   ┌───┐        column 0 at v = 0
┌───┤ A ├───┤ A │        column 1 at v = 0.5
│ A │   │ A │   │        column 2 at v = 0   (0.5 · 2 = 1.0 ≡ 0)
├───┤   ├───┤   │
│   │ A │   │ A │
└───┴───┴───┴───┘
```

```glsl
uv.y += 0.5 * floor(uv.x);   // offset only
```

The shift is **vertical, indexed by column**. Shifting horizontally by row is a
different pattern (a brick bond) and is wrong for these SKUs.

## The mip-level trap

A half-drop makes `uv.y` **discontinuous at every column boundary**. Automatic
texture derivatives sample a 2×2 pixel quad, so at a boundary they see a jump of a
whole tile, conclude the texture is enormously minified, and select the coarsest mip
level. The result is a blurred vertical line down every column edge — which looks
like a broken renderer rather than a filtering artefact.

Fix: take the gradients **before** applying the drop, and pass them explicitly.

```glsl
vec2 ddx = dFdx(uv), ddy = dFdy(uv);      // before the shift
uv.y += off * floor(uv.x) * 0.5;
vec3 tex = textureGrad(sampler, vec3(fract(uv), layer), ddx, ddy).rgb;
```

The drop is piecewise constant in `u`, so the true derivative is unchanged
everywhere except exactly at the seam. Supplying the pre-shift gradients is
therefore correct, not an approximation.

## Non-square tiles

`scale` gives the repeat **width**. Height must be derived from the source image's
aspect ratio:

```
pattern_height_m = scale * (tile_height_px / tile_width_px)
```

The webapp currently resamples every tile to a square 1024×1024, which stretches
non-square repeats. Acceptable for evaluation, wrong for production.

## What belongs in the contract

`repeatType` and `scale` are per-SKU catalogue data, not per-scene, so they stay out
of `scene.json`. But the **definitions above** must live in `scene/SCHEMA.md`,
because any second renderer needs them to agree pixel-for-pixel — and a wrong drop
axis is silent: the render looks plausible, just not like the product.

Suggested addition to `SCHEMA.md`:

- `pattern.default_width_m` — fallback when a SKU has no `scale`
- a `pattern_conventions` section restating straight vs offset, the drop axis, and
  the derivative requirement
- a note that `scale` is repeat **width**, and height derives from tile aspect ratio

## Related: resolution invariance

The client may render at any resolution by scaling `f_px`, `cx`, `cy` by the same
factor as the image. This is the invariant P3-T2 tests to 1e-16, and the webapp uses
it to super-sample above bundle resolution on high-DPI displays. Worth stating in
`SCHEMA.md` as an intended capability rather than leaving it as an accident of the
maths.
