# Implementation Plan: Colab Notebook with Gradio UI for RGB→X Model

**Planning Date:** 2025-10-14
**Research Reference:** `research/notebook_research/colab-notebook-requirements-research.md`
**Target:** Create Jupyter notebook with modular Gradio UI supporting custom uploads and dynamic sample image processing with hierarchical ZIP downloads

---

## Required Reading

**CLAUDE.md Files:**
- `implementation.md` - Section: Final Working Solution (Python 3.9 venv setup)
- `rgbx/rgb2x/CLAUDE.md` - Section: PAVED PATH (pipeline initialization)
- `normal_irradiance/CLAUDE.md` - Section: Pipeline Output Format, Shadow Export

**Reference Implementations:**
- `normal_irradiance/gradio_demo_normal_irradiance.py:18-29` - Pipeline loading pattern
- `normal_irradiance/gradio_demo_normal_irradiance.py:32-74` - AOV save function with gamma handling
- `normal_irradiance/gradio_demo_normal_irradiance.py:77-183` - Callback structure with numpy output
- `normal_irradiance/gradio_demo_normal_irradiance.py:185-247` - Gradio Blocks interface layout
- `rgbx/rgb2x/gradio_demo_rgb2x.py:80-98` - Image resize logic (max 1000px, divisible by 8)
- `normal_irradiance/sample_images/` - Sample room images (dynamic count)

**Dependencies:**
- Module: `pipeline_normal_irradiance` - Class: `StableDiffusionAOVMatEstPipeline`
- Module: `load_image` - Functions: `load_ldr_image(path, from_srgb=True)`, `load_exr_image(path, tonemaping=True, clamp=True)`
- Module: `diffusers` - Class: `DDIMScheduler`
- Python: `zipfile` for hierarchical ZIP creation
- Environment: Python 3.9 venv with PyTorch 2.3.0+cu121, diffusers 0.20.0, gradio 3.48.0

---

## Files to Create/Modify

### New: `RGB_X_Colab_Notebook.ipynb`
**Purpose:** Jupyter notebook for Colab with modular Gradio UI, custom uploads, dynamic sample processing, and hierarchical ZIP downloads
**Pattern source:** `normal_irradiance/gradio_demo_normal_irradiance.py` (entire structure)
**Cell blocks to implement:**
- Cell Block 1: Environment setup (Python 3.9 venv, dependencies, MPLBACKEND clearing)
- Cell Block 2: Clone repo and navigate to working directory
- Cell Block 3: Import modules
- Cell Block 4: Initialize pipeline (run once, stays loaded)
- Cell Block 5: Define ZIP creation helper function
- Cell Block 6: Define AOV save helper function (from gradio_demo_normal_irradiance.py:32-74)
- Cell Block 7: Define processing callback with session tracking (editable)
- Cell Block 8: Create Gradio interface with dynamic sample loading (editable)
- Cell Block 9: Launch demo

---

## Implementation Steps

### Step 1: Create Environment Setup Cell
**Files:** `RGB_X_Colab_Notebook.ipynb`
**Actions:**
1. Follow pattern from `implementation.md:84-98`
2. Install Python 3.9: `apt-get install python3.9 python3.9-venv`
3. Create venv: `python3.9 -m venv /content/py39env`
4. Activate venv: `source /content/py39env/bin/activate`
5. Install dependencies: `pip install torch==2.3.0 torchvision diffusers==0.20.0 gradio==3.48.0 opencv-python`
6. Clear MPLBACKEND: `os.environ.pop('MPLBACKEND', None)`
7. Verify GPU: `torch.cuda.is_available()`
8. Test: Run cell, verify Python 3.9 active, GPU detected, packages import successfully

**Success criteria:** Python 3.9 venv created, GPU available, dependencies installed, MPLBACKEND cleared

### Step 2: Create Repo Clone Cell
**Files:** `RGB_X_Colab_Notebook.ipynb`
**Actions:**
1. Clone repo: `git clone <repo_url>` (or use GitHub authentication if private)
2. Change directory: `cd rgb-x/normal_irradiance`
3. Checkout branch: `git checkout normal-irradiance-pipeline`
4. Print current directory for verification
5. Test: Verify in correct directory, pipeline files present

**Success criteria:** Repo cloned, in `normal_irradiance/` directory, correct branch

### Step 3: Create Module Import Cell
**Files:** `RGB_X_Colab_Notebook.ipynb`
**Actions:**
1. Set OpenEXR flag: `os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"`
2. Import: `from pipeline_normal_irradiance import StableDiffusionAOVMatEstPipeline`
3. Import: `from load_image import load_ldr_image, load_exr_image`
4. Import: `import gradio as gr, torch, torchvision, cv2, numpy as np, os, zipfile, glob`
5. Import: `from diffusers import DDIMScheduler`
6. Test: Run cell, verify all imports succeed

**Success criteria:** All modules import without errors, OpenEXR enabled

### Step 4: Create Pipeline Initialization Cell
**Files:** `RGB_X_Colab_Notebook.ipynb`
**Actions:**
1. Follow pattern from `gradio_demo_normal_irradiance.py:20-29`
2. Load pipeline: `pipe = StableDiffusionAOVMatEstPipeline.from_pretrained("zheng95z/rgb-to-x", torch_dtype=torch.float16, cache_dir="model_cache_colab")`
3. Configure scheduler: `pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, rescale_betas_zero_snr=True, timestep_spacing="trailing")`
4. Disable progress bar: `pipe.set_progress_bar_config(disable=True)`
5. Move to GPU: `pipe.to("cuda")`
6. Print: "✓ Pipeline loaded. Model stays in memory - edit Gradio cells below without reloading."
7. Test: Verify model loads, ~5-10 sec load time, no CUDA errors

**Success criteria:** Pipeline loaded to GPU, scheduler configured, ready for inference

### Step 5: Create ZIP Helper Function Cell
**Files:** `RGB_X_Colab_Notebook.ipynb`
**Actions:**
1. Define `create_hierarchical_zip(input_path, aov_files, output_zip_name)`
2. Create ZIP structure:
   - `input_images/{original_filename}`
   - `normal_model/{filename}_normal_npy.npy`
   - `normal_model/{filename}_normal_preview.png`
   - `irradiance_model/{filename}_irradiance_npy.npy`
   - `irradiance_model/{filename}_irradiance.exr`
   - `irradiance_model/{filename}_irradiance_preview.png`
3. Use `zipfile.ZipFile()` with `arcname` parameter for folder structure
4. Return path to created ZIP file
5. Test: Call with dummy files, verify ZIP structure correct

**Success criteria:** Function creates hierarchical ZIP with correct folder structure and renamed files

### Step 6: Create AOV Save Helper Function Cell
**Files:** `RGB_X_Colab_Notebook.ipynb`
**Actions:**
1. Follow pattern from `gradio_demo_normal_irradiance.py:32-74`
2. Define `save_aov_images(aov_dict, input_filename, output_dir="/tmp/rgb2x_output")`
3. Add `input_filename` parameter for file naming
4. For normal: Save `{input_filename}_normal_npy.npy` (raw float32), save `{input_filename}_normal_preview.png` (8-bit)
5. For irradiance: Invert (`1.0 - img_np`), save `{input_filename}_irradiance_npy.npy` (gamma), save `{input_filename}_irradiance.exr` (linear: `img^2.2`), save `{input_filename}_irradiance_preview.png` (gamma)
6. Return dictionary: `{aov_name: {file_type: path}}`
7. Test: Call with dummy arrays, verify 5 files created with correct naming

**Success criteria:** Function creates 5 files with input filename prefix and correct gamma handling

### Step 7: Create Processing Callback Cell with Session Tracking
**Files:** `RGB_X_Colab_Notebook.ipynb`
**Actions:**
1. Follow pattern from `gradio_demo_normal_irradiance.py:77-183`
2. Initialize session state: `processed_samples = {}` (global dictionary tracking processed files)
3. Define `process_image_callback(photo, seed, inference_step, num_samples, image_identifier)`
4. Add `image_identifier` parameter (filename for tracking)
5. Load image using `load_ldr_image()` or `load_exr_image()` based on extension
6. Implement resize logic from `gradio_demo_rgb2x.py:80-98`: max 1000px, round to multiple of 8
7. Loop over `required_aovs = ["normal", "irradiance"]` with frozen prompts
8. Call pipeline with `output_type="np"`, access `.images[0][0]`
9. Resize output back to original dimensions using `cv2.resize()`
10. Call `save_aov_images()` with input filename
11. Create individual ZIP: `create_hierarchical_zip(photo.name, aov_files, f"{input_filename}_output.zip")`
12. Update session state: `processed_samples[image_identifier] = {"files": aov_files, "zip": zip_path, "success": True}`
13. On error: Update session state with `{"success": False, "error": error_message}`
14. Return: `(gallery, individual_zip_path, error_message)`
15. Test: Run with sample image, verify session state updated, ZIP created

**Success criteria:** Callback processes image, updates session tracking, creates individual hierarchical ZIP, handles errors gracefully

### Step 8: Create Gradio Interface Cell with Dynamic Sample Loading
**Files:** `RGB_X_Colab_Notebook.ipynb`
**Actions:**
1. Follow pattern from `gradio_demo_normal_irradiance.py:185-247`
2. Initialize session state: `processed_samples = {}` (if not already initialized)
3. Create `gr.Blocks()` context
4. **Section 1: Custom Upload (Row 1)**
   - Left column: File input, parameters (seed, inference_step, num_samples), Run button
   - Right column: Output gallery, individual download button
5. **Section 2: Sample Images (Dynamic Rows)**
   - Use `glob.glob("sample_images/*.jpg") + glob.glob("sample_images/*.jpeg") + glob.glob("sample_images/*.png")` to get all samples
   - For each sample image:
     - Create new row with left/right columns
     - Left: Display preloaded image (use `gr.Image(value=sample_path, type="filepath")`), parameters, Run button
     - Right: Dedicated output gallery, individual download button, error display
   - Each row has unique component IDs and callbacks
6. **Section 3: Batch Processing and Download All**
   - Add "Process All Samples" button
   - Define batch callback: loops through all samples, calls `process_image_callback()` for each
   - Add "Download All" button with status text: "Download All ({successful}/{total} processed)"
   - Define download_all callback: creates `all_samples_output.zip` from `processed_samples` dictionary (only successful runs)
7. Connect all callbacks to respective buttons
8. Test: Run cell, verify UI renders with dynamic sample count, components visible

**Success criteria:** Gradio interface displays with 1 custom row + N dynamic sample rows + batch controls, all components functional

### Step 9: Define Batch Processing Callback Cell
**Files:** `RGB_X_Colab_Notebook.ipynb`
**Actions:**
1. Define `process_all_samples_callback()`
2. Get list of sample images from `sample_images/` directory
3. Loop through each sample:
   - Call `process_image_callback()` with default parameters
   - Update UI progress indicator (if possible with Gradio)
   - Collect results for each sample
4. Return list of results for each sample's output gallery
5. Test: Click "Process All Samples", verify all samples processed sequentially

**Success criteria:** Batch callback processes all samples sequentially, updates session state for each

### Step 10: Define Download All Callback Cell
**Files:** `RGB_X_Colab_Notebook.ipynb`
**Actions:**
1. Define `download_all_callback()`
2. Filter `processed_samples` dictionary for successful runs only
3. Create master ZIP: `all_samples_output.zip`
4. For each successful sample:
   - Add to hierarchical structure with same folder pattern
   - Rename files: `{sample_name}_normal_npy.npy`, etc.
5. Calculate success rate: `successful_count / total_count`
6. Return ZIP path and status message: "Downloaded {successful} of {total} samples"
7. Test: Process multiple samples, click "Download All", verify ZIP contains only successful runs

**Success criteria:** Download All creates master ZIP with hierarchical structure, excludes failed runs, shows accurate count

### Step 11: Create Launch Cell
**Files:** `RGB_X_Colab_Notebook.ipynb`
**Actions:**
1. Get demo from previous cell: `demo = [interface from Step 8]`
2. Configure queue: `demo.queue(max_size=10)`
3. Launch with share: `demo.launch(share=True, debug=True)`
4. Print: "✓ Gradio live. Modify callback cells (Steps 7-10) and re-run without reloading model."
5. Test: Run cell, verify public gradio.live URL appears, interface accessible

**Success criteria:** Public Gradio URL generated, interface accessible, can close and relaunch without model reload

---

## Data Flow

```
Custom Upload:
  User Image [.png/.jpg/.exr]
    → load_image() → Tensor[C,H,W]
    → Resize (max 1000px, ÷8) → Tensor[C,H',W']
    → Pipeline (2 AOVs) → output_type="np" → numpy[H,W,C]
    → save_aov_images() → 5 files with {filename} prefix
    → create_hierarchical_zip() → {filename}_output.zip
    → Update processed_samples[custom_upload]
    → Return gallery + individual ZIP

Sample Images (Dynamic):
  sample_images/*.jpg (N files discovered via glob)
    → For each: Preload as gr.Image(value=path)
    → On Run click: Same flow as custom upload
    → Update processed_samples[sample_name]
    → Display in dedicated output gallery

Process All Samples:
  → Loop through all N sample images
    → Call process_image_callback() for each
    → Update processed_samples for each
    → Display results in respective galleries

Download All:
  → Filter processed_samples for success=True
    → Create all_samples_output.zip
      → input_images/{sample1}, {sample2}, ...
      → normal_model/{sample1}_normal_npy.npy, {sample2}_normal_npy.npy, ...
      → irradiance_model/{sample1}_irradiance.exr, {sample2}_irradiance.exr, ...
    → Return master ZIP + status message
```

---

## Testing Plan

**Per-step tests:**
- Step 1: Verify Python 3.9 active, GPU detected, packages import
- Step 2: Verify repo cloned, correct directory, branch checked out
- Step 3: All modules import without errors
- Step 4: Pipeline loads to GPU, no CUDA errors
- Step 5: ZIP helper creates correct hierarchical structure
- Step 6: AOV save creates 5 files with correct naming
- Step 7: Callback updates session state, creates individual ZIP
- Step 8: UI renders with dynamic sample count (N rows)
- Step 9: Batch callback processes all samples sequentially
- Step 10: Download All creates master ZIP, excludes failures
- Step 11: Public URL generated, interface accessible

**Integration test:**
1. Upload custom image via UI
2. Click "Run", verify output gallery displays normal + irradiance
3. Download individual ZIP, verify hierarchical structure and file naming
4. Click on first sample image, click "Run"
5. Verify sample output displays in dedicated gallery
6. Click "Process All Samples", verify all N samples processed
7. Introduce failure (invalid file), verify error message displayed
8. Click "Download All", verify:
   - Master ZIP excludes failed sample
   - Status shows correct count (e.g., "11/12 processed")
   - Hierarchical structure maintained across all samples
9. Modify callback cell (add print statement)
10. Re-run callback + interface cells without reloading model
11. Verify changes reflected in new runs

**Editability test:**
1. Modify callback function in Step 7 (add debug print)
2. Re-run Step 7 cell
3. Re-run Step 8 cell (interface)
4. Re-run Step 11 cell (launch)
5. Verify changes reflected without reloading model (Step 4 untouched)

---

## Rollback Plan

If implementation fails:
1. Verify Python 3.9 venv active before running cells
2. If JAX errors: Confirm diffusers==0.20.0 and Python 3.9
3. If CUDA errors: Check GPU runtime enabled in Colab
4. If module import fails: Verify in `normal_irradiance/` directory
5. If dynamic sample loading fails: Check `sample_images/` directory exists and contains images
6. If ZIP creation fails: Verify `zipfile` module imported, test with dummy files
7. If session state tracking fails: Initialize `processed_samples = {}` in Step 3
8. Alternative: Use original standalone script from `gradio_demo_normal_irradiance.py` if notebook approach blocked

---

## Success Criteria

- [ ] All 11 steps completed
- [ ] Notebook runs end-to-end on Google Colab
- [ ] Model loads once, stays loaded between Gradio restarts
- [ ] Custom upload section functional with left input / right output layout
- [ ] Dynamic sample discovery: N rows created based on `sample_images/` directory contents
- [ ] Each sample has dedicated output gallery and individual download
- [ ] Individual ZIP downloads with hierarchical structure:
  - [ ] `input_images/{filename}`
  - [ ] `normal_model/{filename}_normal_npy.npy`
  - [ ] `normal_model/{filename}_normal_preview.png`
  - [ ] `irradiance_model/{filename}_irradiance_npy.npy`
  - [ ] `irradiance_model/{filename}_irradiance.exr`
  - [ ] `irradiance_model/{filename}_irradiance_preview.png`
- [ ] "Process All Samples" button processes all N samples sequentially
- [ ] Session tracking: `processed_samples` dictionary tracks success/failure
- [ ] "Download All" button creates `all_samples_output.zip` with:
  - [ ] Only successful runs included
  - [ ] Same hierarchical structure maintained
  - [ ] Status message shows count: "{successful}/{total} processed"
- [ ] Error handling: Failed runs show error message, excluded from downloads
- [ ] Callback cells editable and re-runnable without model reload
- [ ] Public Gradio URL accessible from browser
- [ ] Follows patterns from `normal_irradiance/gradio_demo_normal_irradiance.py`
- [ ] Image resize logic matches original (max 1000px, divisible by 8)
