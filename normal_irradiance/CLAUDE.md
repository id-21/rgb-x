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

### Shadow Extraction Pipeline
1. **Input Processing** (load_image.py)
   - LDR images (.png/.jpg): sRGB→linear conversion via `load_ldr_image()`
   - HDR images (.exr): tone mapping via `load_exr_image()`

2. **Inference** (pipeline_normal_irradiance.py)
   - Model: `zheng95z/rgb-to-x` (same as rgb2x, shared weights)
   - Generates 2 AOVs only: normal + irradiance (~3x faster than 5-AOV pipeline)
   - Post-processing (lines 803-809):
     - Normal: linear space (no gamma correction)
     - Irradiance: gamma-corrected for display (linear→sRGB)

3. **Shadow Export** (gradio_demo_normal_irradiance.py:32-60)
   - Irradiance → Linear RGB: reverse gamma via `img^2.2` (line 50)
   - Save as EXR in linear color space for ST-RoomNet V3
   - PNG preview saved for visual verification

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

**Linear vs sRGB Workflow**:
The pipeline intelligently handles color spaces per AOV type:
- Display outputs (irradiance): gamma-corrected PNG for humans
- Processing outputs (irradiance EXR): linear RGB for math operations
Example: `gradio_demo_normal_irradiance.py:50` (gamma reversal)

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
- **Linear RGB requirement** - Shadow multiplication MUST happen in linear space to avoid color shifts in saturated wallpapers (see `research/understanding-shadow.md:95-98`)
- **EXR format** - ST-RoomNet V3 expects LINEAR RGB in EXR. The demo reverses gamma (line 50) before saving
- **Gamma reversal formula** - `linear = gamma^2.2` undoes the `gamma = linear^(1/2.2)` applied during inference (pipeline_normal_irradiance.py:807)
- **Model sharing** - Uses same `zheng95z/rgb-to-x` weights as rgb2x, isolated cache prevents conflicts
- **Dimension constraints** - Input images MUST have width/height divisible by 8 (VAE downsampling factor)
- **Prompts are frozen** - "Camera-space Normal" and "Irradiance (diffuse lighting)" prompts are model-specific, do not modify
- **Public Gradio links** - Demo launches with `share=True` (line 206) for Colab access
</critical_notes>

<paved_path>
## PAVED PATH

**Running the demo**:
1. Set up environment: `conda env create -f ../rgbx/environment.yml`
2. Launch demo: `python gradio_demo_normal_irradiance.py`
3. Access public URL (printed to console)
4. Upload room photo → downloads irradiance_shadow.exr + preview PNG

**Using shadow EXR with ST-RoomNet V3**:
1. Load EXR: `irradiance = cv2.imread('irradiance_shadow.exr', cv2.IMREAD_UNCHANGED)`
2. Mask wall pixels: `wall_mask = (normals == target_wall_id)`
3. Compute baseline: `baseline = np.percentile(irradiance[wall_mask], 98)`
4. Create shadow mask: `shadow = np.clip(irradiance / baseline, 0.4, 1.0)`
5. Apply to wallpaper (linear space): `result = wallpaper_linear * shadow`

**For fastest inference**:
- Resize input to max 1000px (demo does this automatically)
- Use GPU with at least 8GB VRAM
- Expected speed: 2-3 AOVs in ~5-8 seconds (T4 GPU)
</paved_path>
