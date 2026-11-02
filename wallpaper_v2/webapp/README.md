# webapp — WebGL2 renderer against real bundles

A single-file client that consumes the Phase 3 bundle contract. No build step, no
framework, no dependencies.

## Run

From the **repo root** (the server must be able to see both `wallpaper_v2/scene/golden/`
and `textureAssets.json`):

```bash
cd /Users/ishan-aiworkspace/Documents/apps/rgb-x
python3 -m http.server 8000
```

Then open <http://localhost:8000/wallpaper_v2/webapp/>

## What it reads

| file | used for |
|---|---|
| `scene/golden/<sample>/scene.json` | camera, planes, regions, quality |
| `scene/golden/<sample>/maps.png` | R = region id, B = alpha (G unused) |
| `scene/golden/<sample>/shading.png` | 16-bit single-channel shading |
| `scene/golden/<sample>/photo.jpg` | the base photograph |
| `textureAssets.json` | 685 wallpapers: `scale` (metres), `repeatType`, thumbnails |

**Note the four-file bundle.** P3-T3 measured 8-bit shading at up to 3/255 render
error — over the 2/255 gate — and moved shading into its own 16-bit PNG. `maps.png`'s
green channel is now unused. Browsers decode 16-bit PNG to 8 bits on upload, so the
client sees the precision the original packing would have given; that is acceptable
for preview and worth revisiting if banding shows up on large smooth walls.

It reads **nothing else** — no `.npy`, no depth, no normals, no Python. Same
constraint as `scene/consume.py`, which is the point: if this renders correctly, the
contract is doing its job in a second independent implementation.

## What it demonstrates

- **The bundle is sufficient.** A browser with no access to the pipeline reproduces
  the render from ~40 floats per plane plus two images.
- **Instant wallpaper changes.** Clicking a swatch is a texture bind and a redraw.
  Watch the frame-time readout — it doesn't move.
- **Real catalogue metadata.** `scale` drives pattern width in metres, so a 0.64 m
  Aura repeat renders visibly larger than a 0.53 m one. This is the scale problem
  that took the whole project to get right, now driven by the actual SKU data.
- **Offset repeats.** 162 of your 685 assets are `repeatType: "offset"` (half-drop).
  Tiles stack normally within a column; each successive **column** starts half a
  tile further **down**. The shader adds `0.5 · floor(u)` to `v`, and supplies
  explicit texture gradients taken before that shift — otherwise the automatic
  derivative sees the discontinuity at each column boundary, drops to the coarsest
  mip level, and draws a blurred vertical line down every column edge.
- **Per-wall assignment.** Pick a wall from the list or the dropdown, then a
  wallpaper — different regions can hold different wallpapers simultaneously via a
  `sampler2DArray`. This is the Phase 5 feature arriving early, because T4's region
  layer already made the data available.
- **Hardware anisotropic filtering** via `EXT_texture_filter_anisotropic` at 16×,
  the same thing implemented by hand in numpy in `render.py`.
- **Linear-space compositing**, matching the Phase 2 T2 fix — sRGB decode, multiply
  by shading, blend, re-encode.

## Known limitations

- **Swatch thumbnails are 200×121.** The app tries `localPath`, then `cloudUrl`, then
  falls back to the embedded base64 `placeholder`. If neither of the first two
  resolves, textures will look soft — that is the thumbnail, not the renderer. Point
  `localPath` at real files for a fair quality judgement.
- **58 assets have `scale: 0`** (missing data). Those fall back to 0.53 m. Worth
  fixing in the catalogue — a wrong pattern width is exactly the class of error this
  project spent months eliminating.
- **8 simultaneous textures max** (`MAXLAYERS`). Raise if needed; texture array
  layers are cheap.
- **Textures are resampled to 512×512** on upload, which distorts non-square repeats.
  Fine for evaluation, wrong for production — the real pipeline should keep the tile's
  aspect ratio and use `scale` for width with height derived from it.
- **No mask edge refinement.** Phase 2 T6 is postponed, so soffits and window reveals
  are still wallpapered and edges are still thresholded. Most visible on
  `no_walls_living` and `living_small_walls`.
