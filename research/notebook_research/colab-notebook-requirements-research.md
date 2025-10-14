# Colab Notebook Requirements - Research Analysis

**Research Date:** 2025-10-14
**Research Type:** Workflow Design
**Researcher:** Human/Claude

---

## Executive Summary

User requires a Jupyter notebook for Google Colab that loads the RGB→X model inline (not as modules) and runs a custom Gradio interface for debugging. The notebook must extract normal maps and irradiance data from room photos for wallpaper visualization, with full visibility into the pipeline execution for debugging. This replaces the current approach of running standalone Python scripts which are difficult to debug remotely.

---

## CLAUDE.md Analysis

**Files Read:**
- `implementation.md` - Environment constraints: Python 3.9 venv required, MPLBACKEND must be cleared
- `rgbx/rgb2x/CLAUDE.md` - Model architecture: pipeline initialization, image loading, dual guidance patterns
- `normal_irradiance/CLAUDE.md` - Output format requirements: numpy arrays, gamma correction rules, EXR export format

**Relevant Patterns Identified:**
- **Python 3.9 venv pattern** from `implementation.md` - Required to avoid JAX compatibility issues
- **Pipeline initialization pattern** from `rgbx/rgb2x/CLAUDE.md:79-84` - Load from HuggingFace with DDIM scheduler
- **NumPy-first workflow** from `normal_irradiance/CLAUDE.md:81-104` - Use `output_type="np"` for direct array access
- **Linear vs sRGB handling** from `normal_irradiance/CLAUDE.md:106-118` - Gamma correction applied per AOV type

**Existing Code to Reference:**
- Model initialization: Documented in `rgbx/rgb2x/CLAUDE.md` (section: PAVED PATH)
- Image loading utilities: Documented in `rgbx/rgb2x/CLAUDE.md` (section: Input Processing)
- Output format handling: Documented in `normal_irradiance/CLAUDE.md` (section: Pipeline Output Format)

---

## Goal Statement

**Objective:**
Create a Jupyter notebook that runs on Google Colab, loads the RGB→X model components inline, and provides a debuggable Gradio interface for extracting normal maps and irradiance data from room photos.

**Current State:**
- Standalone Gradio script exists at `normal_irradiance/gradio_demo_normal_irradiance.py`
- Script runs successfully in Python 3.9 venv on Colab but debugging is difficult
- Cannot see intermediate outputs or add print statements easily
- Model requires NVIDIA GPU (can't run locally on Mac M4)

**Integration Points:**
- **Environment setup** (documented in `implementation.md:76-130`)
  - Python 3.9 virtual environment
  - Clear `MPLBACKEND` environment variable
  - Install dependencies matching `rgbx/environment.yml`

- **Model components** (documented in `rgbx/rgb2x/CLAUDE.md`)
  - StableDiffusionAOVMatEstPipeline for inference
  - DDIMScheduler with specific configuration
  - Image loading utilities for preprocessing

- **Output requirements** (documented in `normal_irradiance/CLAUDE.md:16-59`)
  - Normal maps: numpy arrays in linear space
  - Irradiance: EXR files in linear RGB for ST-RoomNet V3 integration

**Dependencies:**
- **Core model files** (from `rgbx/rgb2x/CLAUDE.md:5-10`)
  - `pipeline_rgb2x.py` or `pipeline_normal_irradiance.py` - Custom Stable Diffusion pipeline
  - `load_image.py` - Image preprocessing with color space conversions

- **HuggingFace model weights** (from `rgbx/rgb2x/CLAUDE.md:72`)
  - Model ID: `zheng95z/rgb-to-x`
  - Cached to isolated directory to prevent conflicts

---

## Current Workflow Issues

**Problem:**
Running standalone Python scripts on Colab prevents effective debugging because:
1. Can't see intermediate outputs between steps
2. Can't add print/console.log statements easily
3. Errors provide limited context about where they originate
4. Must re-run entire script to test small changes

**Why This Approach Was Used:**
- Original repo by model authors uses standalone Gradio scripts
- Works well for local development with direct terminal access
- Assumption: Users have NVIDIA GPU locally (not true for Mac users)

**Why The Assumption Is Wrong:**
- Mac M4 has MPS (Metal Performance Shaders), not CUDA
- Google Colab is only option for GPU access
- Colab environment is designed for notebooks, not standalone scripts

---

## Notebook Requirements

### Section 1: Environment Setup
**Purpose:** Create isolated Python 3.9 environment and clear conflicting variables

**Requirements from `implementation.md:76-105`:**
- Install Python 3.9 in Colab
- Create venv at `/content/py39env`
- Install PyTorch 2.3.0 with CUDA 12.1
- Install diffusers 0.20.0, gradio 3.48.0, and supporting libraries
- Clear `MPLBACKEND` environment variable

### Section 2: Model Component Loading
**Purpose:** Load pipeline, scheduler, and helper functions inline for visibility

**Requirements from `rgbx/rgb2x/CLAUDE.md:76-84`:**
- Import StableDiffusionAOVMatEstPipeline class definition
- Import image loading utilities (load_ldr_image, load_exr_image)
- Import color space conversion functions
- Load model from `zheng95z/rgb-to-x` with torch.float16
- Configure DDIMScheduler with `rescale_betas_zero_snr=True`

### Section 3: Custom Gradio Interface
**Purpose:** Create debuggable interface for normal + irradiance extraction

**Requirements from `normal_irradiance/CLAUDE.md:42-76`:**
- Accept image uploads (.png, .jpg, .exr)
- Resize to max 1000px, ensure dimensions divisible by 8
- Run inference with prompts: "Camera-space Normal", "Irradiance (diffuse lighting)"
- Use `output_type="np"` for direct numpy array access
- Export normal maps as .npy and .png
- Export irradiance as linear RGB .exr for ST-RoomNet V3
- Invert irradiance (model outputs bright=shadow, need bright=light)
- Reverse gamma correction before saving EXR: `linear = gamma^2.2`

### Section 4: Debugging & Testing
**Purpose:** Add extensive logging and intermediate output visibility

**Requirements:**
- Print statements showing:
  - Image dimensions at each step
  - Data types and value ranges
  - Processing time per AOV
- Display intermediate results in notebook cells
- Ability to inspect numpy arrays directly
- Error messages with full stack traces

---

## Critical Notes

**From `implementation.md:1-17`:**
- MUST use Python 3.9 (Colab defaults to 3.12, causes JAX errors)
- MUST clear `MPLBACKEND` environment variable (Colab-specific conflict)
- Cannot run locally on Mac M4 (requires NVIDIA GPU with CUDA)

**From `rgbx/rgb2x/CLAUDE.md:68-74`:**
- Gamma correction toggle: albedo/irradiance require gamma 2.2→sRGB, but normal maps are linear
- Shadow layer format: ST-RoomNet V3 requires LINEAR RGB in EXR format
- Dimension constraints: images MUST have width/height divisible by 8 (VAE downsampling)
- Prompt engineering: AOV quality depends on exact prompt phrasing, do not modify

**Image Loading Robustness (CRITICAL):**
- OpenCV's `cv2.imread()` fails on some JPEG files (too strict error handling)
- Fixed implementation uses PIL fallback: Try OpenCV first, if returns None, use `PIL.Image.open().convert('RGB')`
- This fix is ALREADY implemented in both `rgbx/rgb2x/load_image.py:83-94` and `normal_irradiance/load_image.py:83-94`
- Notebook should use the existing `load_image.py` files as-is (no modification needed)

**From `normal_irradiance/CLAUDE.md:122-143`:**
- Normal maps stay in linear space (RGB represents XYZ surface directions)
- Irradiance is gamma-corrected for display, must reverse before EXR export
- Linear RGB requirement: shadow multiplication must happen in linear space to avoid color shifts
- Model uses same weights as full rgb2x, isolated cache prevents conflicts

---

## Next Steps

**Notebook Creation** (create `colab-notebook-debug.ipynb`):
- Implement 4 sections: setup, model loading, Gradio interface, testing
- Keep all code inline (no external script imports)
- Add extensive print statements for debugging
- Test with sample images from `normal_irradiance/sample_images/`

---

## Open Questions

- [ ] Should we include the white wall × shadow visualization from existing Gradio script?
- [ ] What sample images are available for testing?
- [ ] Should notebook auto-download sample images or require upload?

---

## References

**CLAUDE.md Files:**
- `implementation.md` - Sections: Platform Incompatibility, Final Working Solution (Python 3.9 venv)
- `rgbx/rgb2x/CLAUDE.md` - Sections: Pipeline Architecture, Gradio Demo, PAVED PATH
- `normal_irradiance/CLAUDE.md` - Sections: Pipeline Output Format, Shadow Export, CRITICAL NOTES

**Documentation Files:**
- `make_gradio_ipynb.txt` - User requirements and problem statement
- `rgbx/environment.yml` - Dependency specifications

**Related Research:**
- `normal_irradiance/CLAUDE.md` references `research/understanding-shadow.md` for shadow processing strategy

---

## Conclusion

The notebook must provide full visibility into the RGB→X pipeline execution by loading model components inline and adding extensive debugging outputs. This requires Python 3.9 venv setup, clearing Colab-specific environment conflicts, and implementing a custom Gradio interface that exposes numpy arrays at each step. Success depends on maintaining exact dependency versions and respecting the model's gamma correction requirements for different AOV types.
