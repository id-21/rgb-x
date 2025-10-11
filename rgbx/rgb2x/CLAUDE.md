<system_context>
This directory implements RGB-to-X: a diffusion-based pipeline that converts realistic photographs into intrinsic image decomposition channels (AOVs: Arbitrary Output Variables). It extracts albedo, surface normals, roughness, metallic properties, and irradiance from single RGB images using a fine-tuned Stable Diffusion model.
</system_context>

<file_map>
## FILE MAP
- `pipeline_rgb2x.py` - Custom Stable Diffusion pipeline for AOV material estimation
- `gradio_demo_rgb2x.py` - Gradio web interface for RGB→X inference with shadow export
- `load_image.py` - Image loading utilities with EXR/LDR support and color space conversions
- `example/` - Sample input images for testing (Castlereagh corridor photo)
</file_map>

<workflow>
## DATA WORKFLOW

### Input Processing (load_image.py)
1. **LDR images** (.png/.jpg): `load_ldr_image()` reads, converts sRGB→linear (gamma 2.2), normalizes to [0,1] (lines 81-95)
2. **HDR images** (.exr): `load_exr_image()` reads, applies optional tone mapping via Yxy color space, clamps (lines 98-119)
3. Color space utilities: RGB↔XYZ↔Yxy transformations using standard matrices (lines 10-78)

### Pipeline Architecture (pipeline_rgb2x.py)
**Custom components:**
- `VaeImageProcrssorAOV` (lines 32-119): Extends diffusers' VAE processor with gamma correction toggle (line 98) for linear→sRGB conversion
- `StableDiffusionAOVMatEstPipeline` (lines 138-821): Modified InstantPix2Pix-style pipeline

**Inference flow in `__call__()` (lines 533-821):**
1. Encode text prompt via CLIP (line 678: `_encode_prompt()`)
2. Preprocess photo to [-1,1] range (line 690)
3. Encode photo to latent space via VAE (lines 697-706)
4. Initialize noise latents (lines 714-723)
5. Denoising loop with dual guidance (lines 742-796):
   - Concatenate latents with image latents (lines 754-756)
   - UNet predicts noise with text+image conditioning (lines 759-764)
   - Apply classifier-free guidance: `guidance_scale` (text) + `image_guidance_scale` (photo adherence) (lines 773-777)
6. Decode latents to AOV (line 799)
7. Post-process per AOV type:
   - Albedo/irradiance: apply gamma correction (lines 802-805)
   - Roughness/metallic: extract single channel, repeat to RGB (lines 807-808)

### Gradio Demo (gradio_demo_rgb2x.py)
**Inference callback (lines 63-143):**
- Loads model from HuggingFace `zheng95z/rgb-to-x` with DDIM scheduler (lines 20-29)
- Resizes input to max 1000px, ensures dimensions divisible by 8 (lines 80-98)
- Iterates over 5 AOV types with specific prompts (lines 100-123):
  - "Albedo (diffuse basecolor)", "Camera-space Normal", "Roughness", "Metallicness", "Irradiance (diffuse lighting)"
- Generates each AOV via pipeline call (lines 115-123)

**Shadow export for ST-RoomNet V3 (lines 32-60):**
- `save_aov_images()` extracts irradiance channel
- Reverses gamma correction: `linear = gamma^2.2` (line 50)
- Saves as EXR in linear color space (line 51) for downstream shadow-aware scene understanding
- Also saves PNG preview for visual verification (lines 54-58)
</workflow>

<patterns>
## PATTERNS
**AOV-specific processing pattern:**
Each AOV type has dedicated post-processing logic based on physical properties. Check `pipeline_rgb2x.py`, search: `aov_name ==` (lines 802-808)

**Color space handling:**
All HDR operations use Yxy for perceptually-uniform tone mapping. See `load_image.py`, search: `convert_rgb_2_Yxy` (lines 39-42, 104-114)

**Dual guidance:**
Pipeline uses both text and image guidance for better photo-realism. See `pipeline_rgb2x.py`, search: `guidance_scale` and `image_guidance_scale` (lines 773-777)
</patterns>

<critical_notes>
## CRITICAL NOTES
- **Gamma correction toggle** - Albedo/irradiance require gamma 2.2→sRGB for display, but roughness/metallic are linear quantities (pipeline_rgb2x.py:802-805)
- **Shadow layer format** - ST-RoomNet V3 requires LINEAR RGB in EXR format. The demo reverses gamma (gradio_demo_rgb2x.py:50) before saving
- **Dimension constraints** - Input images MUST have width/height divisible by 8 due to VAE downsampling factor (gradio_demo_rgb2x.py:94-96)
- **Model weights** - Pipeline downloads from HuggingFace `zheng95z/rgb-to-x` to `model_cache/` subdirectory (gradio_demo_rgb2x.py:21-23)
- **Prompt engineering** - AOV quality depends on exact prompt phrasing. Do not modify prompts dictionary without testing (gradio_demo_rgb2x.py:101-107)
</critical_notes>

<paved_path>
## PAVED PATH
To use RGB→X in another project:
1. Import pipeline: `from pipeline_rgb2x import StableDiffusionAOVMatEstPipeline`
2. Load model: `pipe = StableDiffusionAOVMatEstPipeline.from_pretrained("zheng95z/rgb-to-x")`
3. Configure scheduler: Use DDIM with `rescale_betas_zero_snr=True` (see gradio_demo_rgb2x.py:25-27)
4. Load image via `load_ldr_image()` or `load_exr_image()` with appropriate flags
5. Call pipeline with AOV-specific prompt and `required_aovs=["aov_name"]`
6. For irradiance→shadow export: reverse gamma correction before saving EXR (see gradio_demo_rgb2x.py:48-51)
</paved_path>
