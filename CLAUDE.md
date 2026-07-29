# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repo orientation

This repo has several generations of the same idea — compositing realistic wallpaper into room photos — plus the upstream model code that feeds it. Know which parts are current before touching anything:

**Current / active:**
- `wallpaper_v2/` — the working, bug-fixed downstream pipeline (geometry + rendering). This is where active development happens.
- `normal_irradiance/` — upstream model inference (RGB→X variant) producing the `*_normal.npy` / `*_irradiance.npy` files `wallpaper_v2` consumes. Has its own detailed `normal_irradiance/CLAUDE.md`.

**Legacy / superseded — don't extend these, only reference for history:**
- `visualize_with_precomputed_normals.py` (root) — the script `wallpaper_v2` was written to fix. Has 5 known bugs (see `wallpaper_v2/README.md`).
- `wallpaper_pipeline.ipynb` (root) — even older: derives normals from depth gradients rather than from the RGB→X model. Most superseded artifact in the repo.
- `RGB_X_Colab.ipynb`, `RGB_X_Colab_Notebook.ipynb`, `Normal_Irradiance_Colab.ipynb` — Colab experimentation notebooks.
- `rgbx/` — original vendored upstream RGB↔X (SIGGRAPH 2024) repo, producing 5 AOVs (albedo/normal/roughness/metallic/irradiance). Superseded for this project's needs by the trimmed 2-AOV `normal_irradiance/`, but still the reference implementation, documented via its own `rgbx/rgb2x/CLAUDE.md`.

**Planning docs** (`planning/`): `restructuring-plan.md` predates `wallpaper_v2` and is partially stale (much of the "extract into modules" work it proposes is already done). `roadmap.md` and `web-service-strategy.md` are the live forward-looking docs.

## Architecture / data flow

```
room photo
  → normal_irradiance/ (RGB→X model)   → *_normal.npy, *_irradiance.npy
  → Depth Pro (external, sibling project) → *_depth.npy, *_focal_length.npy
  → SegFormer (external)                  → wall mask
  → rgb-x_output/batch_output/<sample>/     (precomputed inputs, read-only, one dir per sample)
  → wallpaper_v2/run_all.py
      geometry.py: load_normals → estimate_gravity → cluster_planes → fit_planes → compute_uv
      render.py:   sample_texture → shading_from_irradiance → composite
  → wallpaper_v2/outputs/new/<sample>__<texture>/final.png (+ diagnostics.json)
  → run_all.py --collect → outputs/summary.json → verify.py (sanity checks)
```

Three load-bearing conventions that are easy to silently break (all documented in depth in `wallpaper_v2/README.md`):
- **Normals are stored encoded in `[0,1]`** (`n_stored = (n+1)/2`) and must be decoded back to `[-1,1]`, then **z-flipped** — RGB→X emits OpenGL-style normals (+z toward viewer) but the depth unprojection is OpenCV-style (+z away).
- **Irradiance is stored inverted** — dark means *more* light. Shading must invert it (`shading_from_irradiance`), not use it raw.
- **Depth only ever supplies the per-plane scalar offset `d`.** The plane normal always comes from the model, never from depth — this matters near glass/mirrors, where Depth Pro sees through to the scene beyond but the model normal doesn't.

## Commands

```bash
cd wallpaper_v2
./run_batch.sh                       # all samples, one OS process each (bounds peak memory)
python3 run_all.py --one <sample>    # single sample
python3 run_all.py --comparisons     # build original|previous|new comparison sheets
python3 run_all.py --collect         # summary.json + overview sheet
python3 verify.py                    # numeric sanity checks (plane angles, gravity tilt, etc.)
```

There is no unit test suite. `verify.py` (plus the inline `geometry.verify()` self-checks called from `run_all.py`) is the closest equivalent — numeric assertions that wall angles are ~90°/0°, gravity is ~90° from wall normals, and renders aren't blank.

There is no repo-wide dependency manifest for `wallpaper_v2`. Its runtime deps (inferred from imports) are `numpy`, `opencv-python`, `scikit-learn`, `matplotlib`. `rgbx/environment.yml` (conda, CUDA-only, **UTF-16 encoded** — garbles under plain `cat`) is only for running the upstream models in `rgbx/`/`normal_irradiance/`, not for `wallpaper_v2` itself, which just consumes precomputed `.npy` files.

## Key files in `wallpaper_v2/`

- `geometry.py` — normals, gravity estimation, plane clustering/fitting, ray/plane UV homography
- `render.py` — anisotropic mip-mapped texture sampling, irradiance-based shading, compositing
- `run_all.py` — driver; CLI flags `--one <sample>`, `--comparisons`, `--collect`
- `verify.py` — standalone post-hoc numeric QA over `outputs/summary.json`
- `experiments/resolution_sensitivity.py` — standalone experiment validating a resolution claim in `planning/web-service-strategy.md`

## Sub-project docs (don't duplicate, read these directly)

- `normal_irradiance/CLAUDE.md` — the RGB→X inference variant (normals + irradiance only), gamma/linear-space conventions
- `rgbx/rgb2x/CLAUDE.md` — original upstream 5-AOV pipeline
- `wallpaper_v2/README.md` — full bug-by-bug writeup of what `wallpaper_v2` fixed vs. the legacy script, plus output layout and known gaps

## Known gaps / active constraints

- Tuning is based on only ~12 sample images; growing the eval set is the top priority in `planning/roadmap.md`.
- `bed_back_bedroom` has depth but no `_focal_length.npy`, so it can't be rendered yet.
- Camera model is pinhole-only — no lens distortion correction.
- `planning/web-service-strategy.md` defines a `scene.json` wire-format contract for a future web service: the server computes ~13 floats per plane plus a packed `maps.png`, and the client does a single WebGL2 fragment-shader pass. Relevant if asked to build a service around this pipeline later.
