# Implementation Plan: Normal + Irradiance Pipeline

**Planning Date:** 2025-10-11
**Research Reference:** `research/pipeline-data-flow-research.md`, `research/understanding-shadow.md`
**Target:** Create streamlined RGB→X variant outputting only normal + irradiance (shadow) for wallpaper visualization

---

## Required Reading

**CLAUDE.md Files:**
- `rgbx/rgb2x/CLAUDE.md` - Sections: Workflow, Patterns, Critical Notes

**Reference Implementations:**
- `rgbx/rgb2x/gradio_demo_rgb2x.py:63-143` - AOV generation callback
- `rgbx/rgb2x/pipeline_rgb2x.py:798-821` - AOV post-processing logic
- `rgbx/rgb2x/gradio_demo_rgb2x.py:32-60` - Shadow export for ST-RoomNet V3

**Dependencies:**
- Model: `zheng95z/rgb-to-x` from HuggingFace (single model, all AOVs)
- Module: `diffusers.DDIMScheduler` with `rescale_betas_zero_snr=True`
- Module: `StableDiffusionAOVMatEstPipeline` from `pipeline_rgb2x.py`
- Functions: `load_exr_image()`, `load_ldr_image()` from `load_image.py`

---

## Files to Create/Modify

### New: `normal_irradiance/` (root level, sibling to `rgbx/`)
**Purpose:** Isolated experiment directory for 2-AOV pipeline
**Pattern source:** `rgbx/rgb2x/` directory structure

### New: `normal_irradiance/load_image.py`
**Purpose:** Image loading utilities (copy as-is)
**Pattern source:** `rgbx/rgb2x/load_image.py:1-120`

### New: `normal_irradiance/pipeline_normal_irradiance.py`
**Purpose:** Simplified pipeline (rename, keep full logic)
**Pattern source:** `rgbx/rgb2x/pipeline_rgb2x.py:1-822`
**Changes:** Lines 801-808 (simplify post-processing to handle only normal/irradiance)

### New: `normal_irradiance/gradio_demo_normal_irradiance.py`
**Purpose:** Gradio API with 2 AOVs
**Pattern source:** `rgbx/rgb2x/gradio_demo_rgb2x.py:1-210`
**Changes:**
- Line 100: `required_aovs = ["normal", "irradiance"]`
- Lines 101-107: Remove albedo/roughness/metallic from prompts
- Line 23: Change cache dir to `model_cache_normal_irradiance/`
- Lines 190-192: Update UI text to "Normal + Shadow Output"

---

## Implementation Steps

### Step 1: Create Directory Structure
**Files:** `normal_irradiance/` (at root level, sibling to `rgbx/`)
**Actions:**
1. Create `normal_irradiance/` directory
2. Test: `ls normal_irradiance/` shows empty directory

**Success criteria:** Directory exists at root level

### Step 2: Copy Image Loading Utilities
**Files:** `normal_irradiance/load_image.py`
**Actions:**
1. Copy `rgbx/rgb2x/load_image.py` to `normal_irradiance/load_image.py`
2. Test: File exists, no modifications needed

**Success criteria:** File copied, 120 lines, no changes

### Step 3: Adapt Pipeline Class
**Files:** `normal_irradiance/pipeline_normal_irradiance.py`
**Actions:**
1. Copy `rgbx/rgb2x/pipeline_rgb2x.py` to `normal_irradiance/pipeline_normal_irradiance.py`
2. Simplify lines 801-808: Keep normal (no gamma) + irradiance (gamma) logic only
3. Remove albedo/roughness/metallic branches
4. Test: Import `StableDiffusionAOVMatEstPipeline` succeeds

**Success criteria:** Pipeline imports without errors, post-processing simplified

### Step 4: Create Streamlined Gradio Demo
**Files:** `normal_irradiance/gradio_demo_normal_irradiance.py`
**Actions:**
1. Copy `rgbx/rgb2x/gradio_demo_rgb2x.py` to new location
2. Line 100: Change to `required_aovs = ["normal", "irradiance"]`
3. Lines 101-107: Update prompts dict to only `{"normal": "Camera-space Normal", "irradiance": "Irradiance (diffuse lighting)"}`
4. Line 23: Change cache_dir to `model_cache_normal_irradiance/`
5. Lines 190-192: Update UI markdown to "Normal + Shadow Output"
6. Keep shadow export logic (lines 32-60, 136-143) unchanged
7. Test: `python gradio_demo_normal_irradiance.py` launches without import errors

**Success criteria:** Gradio demo launches, shows 2 AOVs, exports shadow EXR

### Step 5: Validate Imports and Dependencies
**Files:** All new files
**Actions:**
1. Run `python -c "from pipeline_normal_irradiance import StableDiffusionAOVMatEstPipeline"`
2. Run `python -c "from load_image import load_exr_image, load_ldr_image"`
3. Run `python gradio_demo_normal_irradiance.py` and check UI loads
4. Test: No import errors, model downloads to correct cache dir

**Success criteria:** All imports resolve, UI accessible

---

## Data Flow

```
Input[.png/.jpg/.exr] → load_ldr/exr_image() → Tensor[C,H,W] →
  StableDiffusionAOVMatEstPipeline (×2: normal, irradiance) →
  Post-process[gamma for irradiance only] → PIL Images →
  Gradio Gallery + Shadow EXR Export[linear RGB]
```

---

## Testing Plan

**Per-step tests:**
- Step 1: Directory exists at root level (sibling to `rgbx/`)
- Step 2: `load_image.py` has 120 lines, imports work
- Step 3: Pipeline class imports, simplified post-processing code visible
- Step 4: Gradio UI shows "Normal + Shadow Output", 2 AOVs only
- Step 5: Full inference run with test image from `rgbx/rgb2x/example/`

**Integration test:**
- Run: `python normal_irradiance/gradio_demo_normal_irradiance.py` (from root directory)
- Upload: `rgbx/rgb2x/example/Castlereagh_corridor_photo.png`
- Expected: 2 outputs (normal map + irradiance), shadow EXR downloadable
- Compare: Outputs match `rgbx/rgb2x/` quality, ~3x faster (2 AOVs vs 5)

---

## Rollback Plan

If implementation fails:
1. Delete `normal_irradiance/` directory (at root level)
2. Original `rgbx/rgb2x/` remains untouched as fallback

---

## Success Criteria

- [ ] Directory `normal_irradiance/` created at root level (sibling to `rgbx/`)
- [ ] 3 files copied and adapted (load_image.py, pipeline, gradio_demo)
- [ ] Gradio demo launches on localhost
- [ ] Inference generates 2 AOVs (normal + irradiance) only
- [ ] Shadow EXR exports in linear RGB format
- [ ] ~3x faster than 5-AOV pipeline (2 inference calls vs 5)
- [ ] Model cache isolated to `model_cache_normal_irradiance/`
- [ ] No modifications to original `rgbx/rgb2x/` code
