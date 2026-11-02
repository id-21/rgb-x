<system_context>
Streamlined RGB→X variant that outputs only normal maps and irradiance (shadow) layers for wallpaper visualization. Optimized for ST-RoomNet V3 integration: extracts shadow patterns from room photos to apply realistic lighting to warped wallpaper patterns.
</system_context>

<file_map>
## FILE MAP
- `pipeline_normal_irradiance.py` - Simplified AOV pipeline (2 outputs vs 5)
- `gradio_demo_normal_irradiance.py` - Gradio interface with shadow EXR export
- `load_image.py` - Image loading utilities (unchanged from rgb2x)
- `model_cache_normal_irradiance/` - Isolated HuggingFace model cache
</file_map>

<workflow>
## DATA WORKFLOW

### Pipeline Output Format (pipeline_normal_irradiance.py)
**The pipeline supports 4 output types via `output_type` parameter:**
1. **`"np"` (NumPy)** - Used by Gradio demo (lines 62-106):
   - VAE decodes latents to tensor (line 799)
   - Denormalize to [0, 1] range (lines 89-94)
   - Apply gamma correction if needed (lines 97-98): `image^(1/2.2)`
   - Convert to numpy via `pt_to_numpy()` (line 103)
   - Return shape: `(H, W, C)` with values in [0, 1]

2. **`"pil"` (PIL Image)** - Default, converts numpy→PIL (line 109)
3. **`"pt"` (PyTorch Tensor)** - Returns tensor after gamma (line 101)
4. **`"latent"` (Latent Space)** - Returns raw latents (line 84)

**AOV-specific gamma correction (lines 803-816):**
- **Normal maps** (line 809): `do_gamma_correction = False`
  - Remain in linear space
  - RGB values represent XYZ surface directions
- **Irradiance** (line 807): `do_gamma_correction = True`
  - Gamma-corrected for display: `linear^(1/2.2)`
  - Visually accurate shadow representation

### Shadow + Normal Extraction Pipeline
1. **Input Processing** (load_image.py)
   - LDR images (.png/.jpg): sRGB→linear conversion via `load_ldr_image()`
   - HDR images (.exr): tone mapping via `load_exr_image()`

2. **Inference** (gradio_demo_normal_irradiance.py:127-152)
   - Call pipeline with `output_type="np"` to get numpy arrays directly
   - Returns shape: `(H, W, C)` float32 in [0, 1] range
   - **Normal**: Linear space, no gamma correction
   - **Irradiance**: Gamma-corrected for display
   - Resize using cv2 (line 140-144)
   - Store numpy arrays for download, convert to tensors for Gradio display

3. **Normal Map Export** (gradio_demo_normal_irradiance.py:45-55)
   - `.npy` file: Raw float32 numpy array in linear space (line 48)
   - `.png` file: 8-bit RGB preview for visualization (line 52-54)

4. **Shadow Export** (gradio_demo_normal_irradiance.py:58-72)
   - Invert irradiance: `1.0 - img_np` (model outputs bright=shadow) (line 60)
   - Reverse gamma correction: `linear = gamma^2.2` (line 63)
   - Save as EXR in linear color space for ST-RoomNet V3 (line 65)
   - PNG preview saved for visual verification (line 69-71)

### Wallpaper Visualization Use Case
**Goal**: Apply realistic room shadows to warped wallpaper patterns

**Integration with ST-RoomNet V3**:
1. ST-RoomNet detects wall normals (perspective warp data)
2. This module extracts irradiance (shadow pattern) as linear EXR
3. Normalize irradiance relative to wall's brightest points
4. Multiply warped wallpaper by shadow mask
5. Result: wallpaper appears naturally lit in the room's lighting

**Shadow Processing Strategy** (see `research/understanding-shadow.md`):
- Identify wall region using ST-RoomNet normals
- Find baseline brightness (98th percentile of wall pixels)
- Create relative shadow mask: `irradiance / baseline`
- Clamp minimum to 0.4-0.5 (preserve ambient light realism)
- Apply multiplicatively to wallpaper RGB
</workflow>

<patterns>
## PATTERNS

**NumPy-first workflow (NEW APPROACH)**:
The Gradio demo uses `output_type="np"` to work with numpy arrays throughout:
```python
# Get numpy array directly from pipeline (gradio_demo_normal_irradiance.py:128-137)
generated_image_np = pipe(
    prompt=prompt,
    photo=photo,
    output_type="np",  # Returns (H, W, C) float32 in [0, 1]
).images[0]

# Resize in numpy space using cv2 (line 140-144)
generated_image_np = cv2.resize(generated_image_np, (width, height))

# Store numpy for downloads, convert to tensor only for Gradio display (line 151)
display_tensor = torch.from_numpy(generated_image_np.transpose(2, 0, 1))
```

**Why this is correct:**
- Pipeline's `postprocess()` method (pipeline_normal_irradiance.py:60-109) already handles:
  - Denormalization to [0, 1] (lines 89-94)
  - AOV-specific gamma correction (lines 97-98)
  - NumPy conversion (line 103)
- No need for manual tensor→numpy conversion
- Cleaner code, fewer transformations

**Linear vs sRGB Workflow**:
The pipeline intelligently handles color spaces per AOV type:
- Display outputs (irradiance): gamma-corrected PNG for humans
- Processing outputs (irradiance EXR): linear RGB for math operations
Example: `gradio_demo_normal_irradiance.py:63` (gamma reversal)

**Shadow mask multiplication**:
Always work in linear space when multiplying wallpaper × shadow:
```python
wallpaper_linear = wallpaper_srgb ** 2.2  # Convert to linear
shadowed = wallpaper_linear * shadow_mask  # Multiply
result_srgb = shadowed ** (1/2.2)  # Convert back
```
</patterns>

<critical_notes>
## CRITICAL NOTES

**Data Format:**
- **Pipeline output with `output_type="np"`** - Returns numpy arrays shaped `(H, W, C)` with float32 values in [0, 1]
  - Already denormalized (pipeline_normal_irradiance.py:89-94)
  - Gamma correction applied per AOV type (lines 97-98, 806-809)
  - No further conversion needed for numpy operations
- **Normal maps are linear** - RGB represents XYZ surface directions, no gamma correction applied (line 809)
- **Irradiance is gamma-corrected** - Display-ready with gamma 2.2→sRGB (line 807, 98)

**Color Space Requirements:**
- **Linear RGB requirement** - Shadow multiplication MUST happen in linear space to avoid color shifts in saturated wallpapers (see `research/understanding-shadow.md:95-98`)
- **EXR format** - ST-RoomNet V3 expects LINEAR RGB in EXR. The demo reverses gamma (gradio_demo_normal_irradiance.py:63) before saving
- **Gamma reversal formula** - `linear = gamma^2.2` undoes the `gamma = linear^(1/2.2)` applied during inference (pipeline_normal_irradiance.py:98)

**Model Configuration:**
- **Model sharing** - Uses same `zheng95z/rgb-to-x` weights as rgb2x, isolated cache prevents conflicts
- **Dimension constraints** - Input images MUST have width/height divisible by 8 (VAE downsampling factor)
- **Prompts are frozen** - "Camera-space Normal" and "Irradiance (diffuse lighting)" prompts are model-specific, do not modify

**Deployment:**
- **Public Gradio links** - Demo launches with `share=True` (gradio_demo_normal_irradiance.py:236) for Colab access
</critical_notes>

<paved_path>
## PAVED PATH

**Running the demo**:
1. Set up environment: `conda env create -f ../rgbx/environment.yml`
2. Launch demo: `python gradio_demo_normal_irradiance.py`
3. Access public URL (printed to console)
4. Upload room photo → downloads:
   - `normal_map.npy` - Raw numpy array for programmatic use
   - `normal_map.png` - Visual preview of surface normals
   - `irradiance_shadow.exr` - Linear RGB shadow layer for ST-RoomNet V3
   - `irradiance_preview.png` - Visual preview of shadows

**Using normal maps programmatically**:
```python
import numpy as np

# Load raw numpy array (H, W, 3) float32 in [0, 1]
normals = np.load('normal_map.npy')

# RGB channels represent XYZ surface directions in camera space
# Already in linear space, no gamma correction needed
nx, ny, nz = normals[:, :, 0], normals[:, :, 1], normals[:, :, 2]
```

**Using shadow EXR with ST-RoomNet V3**:
```python
import cv2
import numpy as np

# 1. Load the shadow layer (linear RGB)
irradiance = cv2.imread('irradiance_shadow.exr', cv2.IMREAD_UNCHANGED)
irradiance = cv2.cvtColor(irradiance, cv2.COLOR_BGR2RGB)

# 2. Mask wall pixels (from ST-RoomNet normals)
wall_mask = (normals == target_wall_id)

# 3. Compute baseline brightness (98th percentile)
baseline = np.percentile(irradiance[wall_mask], 98)

# 4. Create relative shadow mask
shadow_mask = irradiance / baseline
shadow_mask = np.clip(shadow_mask, 0.4, 1.0)  # Preserve ambient light

# 5. Apply to wallpaper (in linear space!)
wallpaper_linear = wallpaper_srgb ** 2.2  # Convert to linear
shadowed = wallpaper_linear * shadow_mask  # Multiply
result_srgb = shadowed ** (1/2.2)  # Convert back to sRGB
```

**For fastest inference**:
- Resize input to max 1000px (demo does this automatically at gradio_demo_normal_irradiance.py:88-111)
- Use GPU with at least 8GB VRAM
- Expected speed: 2 AOVs in ~5-8 seconds (T4 GPU)
</paved_path>


## Setup / rebuild (local Mac, non-CUDA)
`venv/` is disposable; `requirements-local.txt` lists the deps for `run_normal_irradiance_local.py`. For the Gradio demo or GPU use the conda route in PAVED PATH instead.
```bash
python3 -m venv venv
./venv/bin/pip install -r requirements-local.txt
```
