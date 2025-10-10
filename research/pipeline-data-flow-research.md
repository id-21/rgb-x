# RGB-X Pipeline Data Flow Research

**Research Type:** Integration Analysis
**Date:** 2025-10-10
**Objective:** Understand the data flow between RGB→X and X→RGB pipelines, including output formats, data transformations, and how outputs from one pipeline map to inputs of the other.

---

## Goal Statement

Investigate the Gradio demo scripts to:
1. Identify where pipeline outputs are stored/returned
2. Understand the data format and transformations applied to outputs
3. Map how RGB→X outputs correspond to X→RGB inputs
4. Document the complete data flow for potential pipeline chaining

---

## Current State Analysis

### RGB→X Pipeline (Image Decomposition)

**Entry Point:** `rgbx/rgb2x/gradio_demo_rgb2x.py`

**Input Processing:**
- Accepts: `.exr`, `.png`, `.jpg`, `.jpeg` files
- Input loading: `rgbx/rgb2x/gradio_demo_rgb2x.py:37-44`
  - EXR files: `load_exr_image()` with tonemapping and clamping enabled
  - LDR files (PNG/JPG): `load_ldr_image()` with sRGB to linear conversion (`from_srgb=True`)
- Image preprocessing: `rgbx/rgb2x/gradio_demo_rgb2x.py:46-64`
  - Resizes to max 1000px on longest side
  - Ensures dimensions are multiples of 8 (VAE requirement)
  - Returns torch tensor in shape `(C, H, W)`

**AOV Channels Generated:**
Defined at `rgbx/rgb2x/gradio_demo_rgb2x.py:66-73`
```
required_aovs = ["albedo", "normal", "roughness", "metallic", "irradiance"]
```

**Pipeline Execution:**
Location: `rgbx/rgb2x/gradio_demo_rgb2x.py:79-87`
- Iterates through each AOV channel independently
- Uses `StableDiffusionAOVMatEstPipeline` from `rgbx/rgb2x/pipeline_rgb2x.py`
- Pipeline call signature:
  - `prompt`: Text description of the channel (e.g., "Albedo (diffuse basecolor)")
  - `photo`: Preprocessed input image tensor
  - `required_aovs`: Single-element list specifying which AOV to generate
  - Returns: `.images[0][0]` - the generated AOV image

**Output Processing:**
Location: `rgbx/rgb2x/gradio_demo_rgb2x.py:89-94`
- Resizes output back to original image dimensions
- Creates tuple: `(generated_image, f"Generated {aov_name} {i}")`
- Accumulates all AOVs in `return_list`

**Output Format:**
- Returns: List of tuples `[(image, label), ...]`
- Image type: Torch tensor or PIL Image (depending on pipeline output_type)
- Default output_type: "pil" (from `rgbx/rgb2x/pipeline_rgb2x.py:559`)

**Pipeline Internal Output Processing:**
Key location: `rgbx/rgb2x/pipeline_rgb2x.py:798-816`
- Decodes latents through VAE
- Applies gamma correction for albedo and irradiance: `image^(1/2.2)` (line 98)
- For roughness/metallic: Takes only first channel, repeats 3x to create RGB (line 808)
- Normalizes output to [0, 1] range

---

### X→RGB Pipeline (Image Synthesis)

**Entry Point:** `rgbx/x2rgb/gradio_demo_x2rgb.py`

**Input Processing:**
Location: `rgbx/x2rgb/gradio_demo_x2rgb.py:42-99`

**Per-Channel Loading:**
All five AOV channels are optional (can be `None`):

1. **Albedo** (lines 42-51):
   - EXR: `load_exr_image()` with `clamp=True`
   - LDR: `load_ldr_image()` with `from_srgb=True`

2. **Normal** (lines 53-62):
   - EXR: `load_exr_image()` with `normalize=True`
   - LDR: `load_ldr_image()` with `normalize=True`

3. **Roughness** (lines 64-73):
   - EXR: `load_exr_image()` with `clamp=True`
   - LDR: `load_ldr_image()` with `clamp=True`

4. **Metallic** (lines 75-84):
   - EXR: `load_exr_image()` with `clamp=True`
   - LDR: `load_ldr_image()` with `clamp=True`

5. **Irradiance** (lines 86-99):
   - EXR: `load_exr_image()` with `tonemaping=True, clamp=True`
   - LDR: `load_ldr_image()` with `from_srgb=True, clamp=True`

**Image Dimension Handling:**
Location: `rgbx/x2rgb/gradio_demo_x2rgb.py:103-120`
- Default: 768x768
- Uses first non-None input image's dimensions
- All inputs must have same dimensions

**Pipeline Execution:**
Location: `rgbx/x2rgb/gradio_demo_x2rgb.py:125-141`
- Uses `StableDiffusionAOVDropoutPipeline` from `rgbx/x2rgb/pipeline_x2rgb.py`
- Accepts all 5 AOV channels as separate parameters
- Supports dropout: Any channel can be `None`, model will hallucinate missing data
- Returns: `.images[0]` - synthesized RGB image
- Output type: "np" (numpy array) as specified at line 140

**Output Visualization:**
Location: `rgbx/x2rgb/gradio_demo_x2rgb.py:146-185`
- Converts input AOVs to display format for gallery
- **Albedo**: Applies inverse gamma `image^(1/2.2)` (line 147)
- **Normal**: Remaps from [-1, 1] to [0, 1] via `normal * 0.5 + 0.5` (line 153)
- **Roughness/Metallic**: Direct conversion (lines 158-166)
- **Irradiance**: Applies inverse gamma `image^(1/2.2)` (line 169)
- Creates gallery with synthesized image + all input AOVs

---

## Data Format Mapping: RGB→X Output to X→RGB Input

### Critical Data Transformations

**Gamma Correction Mismatch:**
- **RGB→X output** (for albedo/irradiance): Applies gamma `^(1/2.2)` - converts to sRGB-like space
  - Location: `rgbx/rgb2x/pipeline_rgb2x.py:98`
- **X→RGB input** (for albedo/irradiance): Expects linear RGB if loaded as LDR with `from_srgb=True` (applies `^2.2`)
  - Location: `rgbx/x2rgb/gradio_demo_x2rgb.py:51, 98` + `load_image.py:88`

**Implication:** RGB→X outputs are gamma-corrected, but X→RGB expects linear. If saving RGB→X outputs as PNG and reloading, they get double gamma correction.

**Normal Vector Format:**
- **RGB→X output**: Normalized camera-space normals, range unknown from demo (likely [0, 1] or [-1, 1])
  - No explicit denormalization found in RGB→X pipeline output
- **X→RGB input**: Expects normalized normals in [-1, 1] range
  - LDR normal loading: Converts [0, 255] → [0, 1] → [-1, 1] → normalizes (line 93-94 in load_image.py)
  - EXR normal loading: Normalizes directly (line 118 in load_image.py)

**X→RGB visualization** shows normals need `* 0.5 + 0.5` to display, suggesting internal format is [-1, 1].

**Roughness/Metallic Format:**
- **RGB→X output**: Single-channel values repeated 3x to create RGB
  - Location: `rgbx/rgb2x/pipeline_rgb2x.py:808`
- **X→RGB input**: Accepts 3-channel images, likely uses first channel only
  - No explicit channel extraction visible in demo

**Irradiance Format:**
- **RGB→X output**: Tone-mapped and gamma-corrected HDR data
  - Tone mapping applied during input loading (demo line 38)
  - Gamma correction on output (pipeline line 98, 804)
- **X→RGB input**: Accepts HDR (EXR with tone mapping) or LDR (PNG/JPG with sRGB conversion)
  - Location: `rgbx/x2rgb/gradio_demo_x2rgb.py:88-99`

---

## Integration Points

### Direct Pipeline Chaining (In-Memory)

**Current Limitation:** RGB→X demo returns Gradio gallery tuples, not raw tensors suitable for X→RGB input.

**Potential Integration Path:**
1. Modify RGB→X callback to return raw torch tensors before Gradio conversion
2. Save tensors with proper format:
   - **Albedo/Irradiance**: Save in linear space (reverse gamma correction before saving)
   - **Normal**: Ensure [-1, 1] range, consider saving as EXR to preserve precision
   - **Roughness/Metallic**: Extract single channel if needed
3. Load into X→RGB with correct flags matching the saved format

**File Format Recommendation for Intermediate Storage:**
- **EXR format** for all channels avoids gamma correction issues
- **Normal vectors**: EXR preserves full range without [0, 1] clamping
- **Albedo/Irradiance**: Linear space preserved in EXR

### Current User Workflow (Manual Transfer)

**Steps:**
1. Run RGB→X demo, download generated AOV images from gallery
2. Upload same images to X→RGB demo
3. Potential issue: Gamma correction mismatch if using PNG/JPG

**User Must:**
- Save AOVs in appropriate format (preferably EXR)
- Understand which channels need special treatment
- Ensure all images have matching dimensions

---

## Dependencies

### Shared Utilities
Both pipelines use identical `load_image.py` implementations:
- `rgbx/rgb2x/load_image.py`
- `rgbx/x2rgb/load_image.py`
- Functions: `load_ldr_image()`, `load_exr_image()`, color space conversions

### Pipeline Classes
- **RGB→X**: `StableDiffusionAOVMatEstPipeline` in `rgbx/rgb2x/pipeline_rgb2x.py:138-821`
- **X→RGB**: `StableDiffusionAOVDropoutPipeline` in `rgbx/x2rgb/pipeline_x2rgb.py:132+`

Both inherit from `diffusers.DiffusionPipeline` and use similar architecture with custom `VaeImageProcrssorAOV`.

---

## Key Findings

1. **RGB→X Output Location**:
   - Generated images returned at `rgbx/rgb2x/gradio_demo_rgb2x.py:87`
   - Stored in `return_list` as tuples with labels (line 94)
   - Pipeline internal output: `rgbx/rgb2x/pipeline_rgb2x.py:816` returns `StableDiffusionAOVPipelineOutput(images=aovs)`

2. **Data Format Incompatibility**:
   - Gamma correction mismatch for albedo/irradiance when using PNG/JPG intermediate format
   - Normal vectors may need explicit range verification
   - Roughness/metallic channel redundancy (3-channel vs 1-channel)

3. **X→RGB Input Flexibility**:
   - Any AOV channel can be omitted (dropout mechanism)
   - Supports both EXR and LDR formats with automatic preprocessing
   - Requires all provided channels to have matching dimensions

4. **Recommended Integration Approach**:
   - Use EXR format for all intermediate AOV storage to avoid gamma issues
   - Verify normal vector range compatibility (may need explicit [-1, 1] mapping)
   - Consider extracting single channel for roughness/metallic before saving

5. **Missing Documentation**:
   - No explicit guidance on data format compatibility between pipelines
   - Gamma correction behavior not documented in demo code
   - Normal vector range conventions unclear

---

## References

### Code Files
- `rgbx/rgb2x/gradio_demo_rgb2x.py` - RGB→X Gradio interface
- `rgbx/x2rgb/gradio_demo_x2rgb.py` - X→RGB Gradio interface
- `rgbx/rgb2x/pipeline_rgb2x.py:138-821` - RGB→X pipeline class
- `rgbx/x2rgb/pipeline_x2rgb.py:132+` - X→RGB pipeline class
- `rgbx/rgb2x/load_image.py` - Image loading utilities
- `rgbx/x2rgb/load_image.py` - Image loading utilities (identical)

### Key Line References
- RGB→X output generation: `pipeline_rgb2x.py:798-816`
- RGB→X gamma correction: `pipeline_rgb2x.py:98`
- X→RGB input loading: `gradio_demo_x2rgb.py:42-99`
- X→RGB normal visualization: `gradio_demo_x2rgb.py:153`
- Channel definitions: `gradio_demo_rgb2x.py:66`

---

## Notes

- No CLAUDE.md files exist in this repository (original research code)
- Both pipelines use Hugging Face Diffusers library as base
- Models downloaded from: `zheng95z/rgb-to-x` and `zheng95z/x-to-rgb`
- CUDA required for both pipelines (`.to("cuda")` calls throughout)
