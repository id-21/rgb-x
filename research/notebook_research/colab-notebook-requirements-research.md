# Colab Notebook Requirements - Research Analysis

**Research Date:** 2025-10-14
**Research Type:** Workflow Design
**Researcher:** Human/Claude

---

## Executive Summary

Create a Jupyter notebook for Google Colab that imports RGB→X model as modules but defines the Gradio interface inline for easy debugging. User can modify and re-run Gradio cells without reloading the model. Extracts normal maps and irradiance data for wallpaper visualization.

---

## CLAUDE.md Analysis

**Files Read:**
- `implementation.md` - Python 3.9 venv setup, MPLBACKEND clearing
- `rgbx/rgb2x/CLAUDE.md` - Pipeline initialization, DDIM scheduler config
- `normal_irradiance/CLAUDE.md` - NumPy output format, gamma correction per AOV type

**Key Patterns:**
- Python 3.9 venv required (JAX compatibility with diffusers 0.20.0)
- Import model as modules, define Gradio callback in notebook cells
- Use `output_type="np"` for direct numpy array access
- PIL fallback for JPEG loading already implemented in `load_image.py:83-94`

---

## Goal Statement

**Objective:**
Import RGB→X pipeline as modules, but define Gradio interface in editable notebook cells for rapid iteration and debugging.

**Current Problem:**
Standalone script `normal_irradiance/gradio_demo_normal_irradiance.py` requires full restart to test changes. Cannot add print statements or inspect intermediate results easily.

**Solution:**
Notebook with separate cells for:
1. Model import (run once)
2. Model initialization (run once, slow)
3. Gradio callback definition (edit and re-run quickly)
4. Gradio interface creation (edit and re-run)
5. Launch (restart without reloading model)

---

## Notebook Structure

### Cell Block 1: Environment Setup
- Install Python 3.9 in Colab: `apt-get install python3.9 python3.9-venv`
- Create venv: `python3.9 -m venv /content/py39env`
- Install dependencies: PyTorch 2.3.0+cu121, diffusers 0.20.0, gradio 3.48.0
- Clear `MPLBACKEND` environment variable

### Cell Block 2: Clone & Import Model (Run Once)
- Clone/install model repo
- Import as modules:
  ```python
  from pipeline_normal_irradiance import StableDiffusionAOVMatEstPipeline
  from load_image import load_ldr_image, load_exr_image
  ```

### Cell Block 3: Initialize Pipeline (Run Once, Slow)
- Load from HuggingFace: `StableDiffusionAOVMatEstPipeline.from_pretrained("zheng95z/rgb-to-x")`
- Configure DDIM scheduler with `rescale_betas_zero_snr=True`
- Move to GPU: `.to("cuda")`

### Cell Block 4: Define Gradio Callback (Editable, Re-runnable)
- Define `callback(photo, seed, inference_step, num_samples)` function
- Add print statements for debugging (image shapes, data types, processing time)
- Call pipeline with `output_type="np"` for numpy arrays
- **Return 5 downloadable files:**
  1. **Normal - NumPy**: `normal_map.npy` - Raw float32 array in linear space
  2. **Normal - Visualization**: `normal_map.png` or `.jpg` - RGB image for visual inspection
  3. **Irradiance - NumPy**: `irradiance.npy` - Raw float32 array (gamma-corrected)
  4. **Irradiance - EXR**: `irradiance_shadow.exr` - Linear RGB (gamma reversed) for ST-RoomNet V3
  5. **Irradiance - Visualization**: `irradiance_preview.jpg` - RGB image for visual inspection

### Cell Block 5: Create Gradio Interface (Editable, Re-runnable)
- Define `gr.Blocks()` interface
- Connect callback to button clicks
- Configure file uploads (.png, .jpg, .exr)

### Cell Block 6: Launch (Restartable)
- `demo.launch(share=True)` for public Colab URL
- Can close and relaunch without reloading model

---

## Critical Implementation Notes

**Environment (from `implementation.md`):**
- Python 3.9 required (Colab defaults to 3.12, causes JAX AttributeError)
- Must clear `MPLBACKEND` before running

**Image Processing (follow original gradio_demo_rgb2x.py:80-98):**
- Resize to max 1000px while maintaining aspect ratio
- Round dimensions to nearest multiple of 8 (VAE downsampling constraint)
- Use `torchvision.transforms.Resize()` for tensor resizing
- `load_image.py` already has PIL fallback for problematic JPEGs (lines 83-94)

**Output Format (from `normal_irradiance/CLAUDE.md:16-59`):**
- Pipeline with `output_type="np"` returns numpy arrays shaped `(H, W, C)` in [0, 1] range
- Access with `.images[0][0]` when `required_aovs` has one item
- **Normal outputs:**
  - `.npy`: Raw numpy array (H, W, 3) float32, linear space (no gamma correction)
  - Visualization: Convert to 8-bit PNG/JPG for visual inspection
- **Irradiance outputs:**
  - `.npy`: Raw numpy array (H, W, 3) float32, gamma-corrected (display-ready)
  - `.exr`: Linear RGB (reverse gamma: `linear = gamma^2.2`) for ST-RoomNet V3
  - Visualization: 8-bit JPG (gamma-corrected) for visual inspection

**Prompts (frozen, do not modify):**
- Normal: "Camera-space Normal"
- Irradiance: "Irradiance (diffuse lighting)"

---

## Benefits of This Approach

**vs. Standalone Script:**
- Model stays loaded between Gradio interface changes
- Add/modify print statements without full restart
- Test callback function pieces in separate cells
- See intermediate results directly in notebook

**vs. Copying Code Inline:**
- Model code stays as importable modules (no code duplication)
- Easier to update if model repo changes
- Only Gradio interface definition lives in notebook

---

## References

**CLAUDE.md Files:**
- `implementation.md` - Python 3.9 venv setup
- `rgbx/rgb2x/CLAUDE.md` - Pipeline initialization pattern
- `normal_irradiance/CLAUDE.md` - Output format, gamma correction rules

**Model Files (import as modules):**
- `pipeline_normal_irradiance.py` - Custom Stable Diffusion pipeline
- `load_image.py` - Image loading with PIL fallback

---

## Conclusion

Notebook imports model as modules but defines Gradio interface in editable cells. This enables rapid iteration (change callback → re-run cell) without slow model reloading, while maintaining model code as reusable modules.
