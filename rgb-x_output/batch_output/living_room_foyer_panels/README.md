# living_room_foyer_panels — panel-mask + mural experiment

Quick reference for how the 5 wall-molding panels in `living_room_foyer.jpeg`
got their own segmentation mask, and how that mask was used to render both a
repeating pattern and a continuous mural across them. Standalone experiment —
nothing in `run_all.py` / `geometry.py` / `render.py` was modified.

## Why

`run_all.py`'s normal SegFormer wall mask (`_wall_confidence.npy`) doesn't
distinguish the wall from its 5 recessed/raised molding panels — ADE20K (what
SegFormer is trained on) has no "wall panel" class. The result was jagged,
torn cutout shapes instead of the panels' true clean rectangular edges. Goal
here: get a precise mask of just the 5 panels, then wallpaper *them*
specifically.

## 1. Segmenting the 5 panels with SAM

Used **MobileSAM** (ViT-T, ~40MB — much faster to get running than full SAM's
~375MB ViT-B checkpoint) rather than SegFormer, since SAM is *promptable*
(point-in, mask-out) rather than fixed-class.

```bash
pip install git+https://github.com/ChaoningZhang/MobileSAM.git
# pinned deps needed alongside it (newer transformers/timm break otherwise):
pip install torch==2.4.1 torchvision==0.19.1 timm==0.4.12
```

- **Point prompts** (one per panel, positive foreground): tried automatic
  blob detection first (Otsu threshold on LAB-L + connected components), but
  the panels partly merged with each other / the ceiling line in that pass,
  so it fell back to 5 manually-eyeballed points. Image is 1160×522:
  `(110,256), (261,256), (551,287), (841,256), (980,256)` — left→right,
  matching: big, thin, big-centre, thin, big.
- For each point, ran MobileSAM once, kept its highest-scoring mask.
- Unioned the 5 masks into one boolean `(H, W)` array.

**Outputs** (in `living_room_foyer/`, the *original* sample dir):
- `living_room_foyer_panel_mask.npy` — boolean mask, same shape as the photo
- `living_room_foyer_panel_mask_viz.png` — green outline overlay, for QA

Quality: tight, clean boundaries right at the trim edges on all 5 panels. The
switch-plate (panel 5) and a wire (panel 4) ended up *inside* their panel's
mask rather than excluded — left as-is, harmless for wallpapering.

## 2. Wiring the mask into wallpaper_v2

`run_all.py` hardcodes its wall-mask input path as `<sample>_wall_confidence.npy`
and does `mask = conf > 0.5` — a bool array satisfies that comparison
directly, so no code changes were needed, just a new sample pointing at the
panel mask instead of the SegFormer one:

```
rgb-x_output/batch_output/living_room_foyer_panels/    <- new sample dir
    living_room_foyer_panels.jpeg              -> symlink to original photo
    living_room_foyer_panels_normal.npy        -> symlink (RGB→X)
    living_room_foyer_panels_irradiance.npy    -> symlink (RGB→X)
    living_room_foyer_panels_depth.npy         -> symlink (Depth Pro)
    living_room_foyer_panels_focal_length.npy  -> symlink (Depth Pro)
    living_room_foyer_panels_wall_confidence.npy -> COPY of the SAM panel mask
```

First render (`python3 run_all.py --one living_room_foyer_panels`) worked
end to end unmodified — **8 planes dropped to 5**, matching the 5 real
panels far more cleanly than the original whole-wall mask (which fragmented
into 8, mostly noise from jagged boundary pixels).

### Shading bug found along the way

`shading_from_irradiance()` defaults to `invert=True` — correct for all 12
original samples (irradiance always measured *negatively* correlated with
observed brightness there). This photo measured **positively** correlated
(+0.75 on the full wall, +0.99 on the panels-only mask) — the opposite
convention. Applying the default invert on top of an already-normal map
double-flipped it, producing a black blob where the ceiling spotlight should
have been bright. Fix used everywhere below: `invert=False` for this sample
specifically (not a global pipeline change — the other 12 samples still need
`invert=True`; making this automatic per-photo, off the correlation the code
already measures, is still open).

## 3. Mural placement — `wallpaper_v2/mural_experiment.py`

The repeating-tile texture path (`sample_texture(..., pattern_width_m)`,
which wraps/tiles by design) doesn't suit a **mural** (one continuous,
non-repeating image) — so `mural_experiment.py` is a separate standalone
script, reusing `geometry.py`/`render.py` functions unmodified:

```bash
cd wallpaper_v2
python3 mural_experiment.py --texture <file in .../patterns/> --tag <output name>
# writes outputs/new/living_room_foyer_panels__<tag>/final.png
```

Key steps it does differently from `run_all.py`:

1. **Forces all 5 panels onto one plane** (`labels[mask] = 0`, skipping
   `cluster_planes`'s k-means entirely) before calling `fit_planes`. This is
   safe *because* SAM already gives exactly the right pixel membership — the
   panels are genuinely one coplanar wall surface, and letting k-means
   re-cluster by normal only reintroduces edge-noise fragmentation. Because
   `compute_uv` computes UV **per plane label**, not per disjoint blob, all
   5 panels end up sharing one continuous, real-world-metre UV space —
   which is what makes a seamless cross-panel mural possible at all.
2. **Cover-fit scaling**: `pattern_width_m = max(W_target, H_target * Pw/Ph)`
   (mural's own aspect ratio preserved, scaled up until it fully covers the
   panels' bounding extent in metres, cropping whichever axis has slack) —
   same idea as CSS `background-size: cover`.
3. **Centring**: shifts UV so the panels' true bounding-box midpoint (by
   percentile, not `compute_uv`'s pixel-count-weighted centroid, which the
   big centre panel would skew) lands exactly on the mural's own centre —
   keeps the whole visible span inside one texture tile so `sample_texture`'s
   wrap-around never triggers (no accidental seam through the middle).
4. Shading: `shading_from_irradiance(irr, wall, invert=False)`, per §2.

### Iteration history (patterns live in `living_room_foyer_panels/patterns/`)

| texture | tag | result |
|---|---|---|
| `donana-silver-murals-coordonne.jpg` (2500×1813) | `donana_mural` | Cover-fit crops top/bottom hard (panel span is much wider than tall vs. the mural's aspect) — clipped the flamingos' wingtips. |
| `donana_x3_plain.jpg` (3× hstack, 7500×1813) | `donana_x3_plain` | Tripling the width flips the cover-fit to be height-constrained instead — full wingspan preserved, no vertical crop. Trade-off: two faint seams (source image isn't quite seamless — edge color/texture diff was small but visible under close inspection) fall within the visible crop. |
| `donana_x3_mirrored.jpg` (flip+orig+flip hstack) | `donana_x3_mirrored` | Same fix, but mirroring makes each copy boundary pixel-identical to its neighbour — **zero seam, guaranteed**. Side-panel birds appear flipped (facing inward) rather than repeated normally; reads as intentional, not broken. |

All `final.png` (+ `sampled_texture_only.png` pre-shading) copied into
`renders/` in this directory, alongside the two `run_all.py` pattern-texture
renders (`30106_*`) and the invert-bug comparison.

## File map

```
rgb-x_output/batch_output/
  living_room_foyer/                          <- original SAM outputs live here
    living_room_foyer_panel_mask.npy
    living_room_foyer_panel_mask_viz.png
  living_room_foyer_panels/                    <- this dir; symlinked sample for the mask swap
    patterns/                                  <- mural source images (added by user + generated tiles)
    renders/                                   <- every final.png produced, consolidated
    README.md                                  <- this file
wallpaper_v2/
  mural_experiment.py                          <- standalone mural renderer (new file, no core edits)
```

## Open items / caveats

- Switch-plate (panel 5) renders as a dark blotch in mural renders — likely
  an anomalous low irradiance value right at that real 3D object; not
  investigated further (left as-is per request).
- `invert=False` is hardcoded in `mural_experiment.py` for this one sample,
  not derived automatically from `irradiance_polarity()`'s sign.
- The "force one plane" step in `mural_experiment.py` is specific to this
  sample's panels; not a general-purpose flag on the real pipeline.
