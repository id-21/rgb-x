# Plan: Restructure RGB-X Wallpaper Pipeline Repository

## Context

**Current State:**
- 3 ML models in separate locations:
  1. **RGB→X** (normals) - `/Users/ishan-aiworkspace/Documents/apps/rgb-x/normal_irradiance/` (runs on Colab)
  2. **SegFormer** (wall masks) - `/Users/ishan-aiworkspace/Documents/apps/ST-RoomNet-for-visualisation/workflow-v1-separate-processing/1-model-inference/run_segformer.py` (runs locally, model: `nvidia/segformer-b2-finetuned-ade-512-512`)
  3. **ML Depth Pro** (depth + focal length) - `/Users/ishan-aiworkspace/Documents/apps/ml-depth-pro/`
- Main pipeline: `visualize_with_precomputed_normals.py` - maps wallpaper using 5 inputs
- Current rgb-x directory is messy with old testing files
- Goal: Deploy full pipeline on Google Cloud (not prioritizing local execution)

**What to Keep:**
- `visualize_with_precomputed_normals.py` - main wallpaper pipeline ✓
- `wallpaper_pipeline.ipynb` - notebook version ✓
- `normal_irradiance/` - RGB→X model code ✓
- `normal_irradiance/sample_images/` - test images ✓
- `rgb-x_output/batch_output/` - precomputed model outputs ✓
- `planning/` and `research/` - documentation ✓

**What to Discard:**
- Old notebooks: `RGB_X_Colab.ipynb`, `RGB_X_Colab_Notebook.ipynb`, `Normal_Irradiance_Colab.ipynb`
- Debug files: `debugging_normal_pipeline.txt`, `make_gradio_ipynb.txt`
- Old implementation: `rgbx/` directory
- Misc: `implementation.md`, `outputs_of_models.md`, `private-key.txt`, `image.png`

---

## Proposed New Structure

```
wallpaper-pipeline/              # New clean repo
├── README.md                    # Overview, setup, usage
├── requirements.txt             # Python dependencies
├── .gitignore
│
├── config/                      # Configuration files
│   ├── pipeline_config.yaml     # Pipeline parameters (pattern size, clustering, etc.)
│   └── model_paths.yaml         # Model locations (local/cloud)
│
├── models/                      # Model inference scripts
│   ├── __init__.py
│   ├── depth_pro.py            # Depth + focal length (from ml-depth-pro)
│   ├── rgb2x.py                # Normal map inference (wrapper for normal_irradiance/)
│   ├── segformer.py            # Wall mask segmentation
│   └── base.py                 # Base model interface
│
├── pipeline/                    # Core wallpaper pipeline
│   ├── __init__.py
│   ├── geometry.py             # 3D unprojection, UV mapping, plane fitting
│   ├── clustering.py           # KMeans wall plane clustering
│   ├── texture.py              # Wallpaper sampling and tiling
│   ├── lighting.py             # LAB color space lighting preservation
│   └── visualize.py            # Visualization and output generation
│
├── scripts/                     # Standalone scripts
│   ├── run_model_inference.py  # Run individual models (depth/normal/mask)
│   ├── run_pipeline.py         # Main pipeline orchestrator
│   └── compare_focal_lengths.py # Compare predicted vs default focal length
│
├── data/                        # Sample data
│   ├── sample_images/          # Test images (from normal_irradiance/sample_images/)
│   └── batch_output/           # Precomputed model outputs (from rgb-x_output/batch_output/)
│
├── outputs/                     # Generated outputs (gitignored)
│   └── .gitkeep
│
├── notebooks/                   # Jupyter notebooks
│   └── wallpaper_pipeline.ipynb # Interactive pipeline (current notebook)
│
├── docs/                        # Documentation
│   ├── planning/               # From current planning/
│   ├── research/               # From current research/
│   ├── FOCAL_LENGTH_COMPARISON.md
│   └── model_details.md        # Model specs and usage
│
├── normal_irradiance/          # RGB→X model (copied as-is)
│   ├── pipeline_normal_irradiance.py
│   ├── gradio_demo_normal_irradiance.py
│   └── load_image.py
│
└── deployment/                 # Cloud deployment (future)
    ├── Dockerfile
    ├── cloud_run.yaml
    └── api_server.py           # FastAPI server (future)
```

---

## Implementation Steps

### Phase 1: Create New Repository Structure

**Actions:**
1. Create new directory: `wallpaper-pipeline/` (sibling to `rgb-x/`)
2. Create all subdirectories: `config/`, `models/`, `pipeline/`, `scripts/`, `data/`, `outputs/`, `notebooks/`, `docs/`, `deployment/`
3. Create placeholder `__init__.py` files in Python package directories
4. Create basic README.md with project overview

**Success Criteria:** Clean directory structure exists

### Phase 2: Refactor Core Pipeline

**Extract from `visualize_with_precomputed_normals.py` into modules:**

1. **pipeline/geometry.py** - Extract functions:
   - `unproject_to_3d()` - 3D unprojection from depth
   - `build_plane_basis()` - UV coordinate system construction
   - `compute_uv_coordinates()` - Project 3D to plane UV

2. **pipeline/clustering.py** - Extract functions:
   - `cluster_wall_planes_kmeans()` - Plane clustering
   - `fit_plane_normals()` - Average normal per cluster

3. **pipeline/texture.py** - Extract functions:
   - `sample_wallpaper()` - Wallpaper sampling with tiling

4. **pipeline/lighting.py** - Extract functions:
   - `composite_with_lighting()` - LAB lighting preservation

5. **pipeline/visualize.py** - Extract functions:
   - `save_visualization()` - Multi-image visualization saving
   - `normalize_for_display()` - Image normalization

6. **pipeline/__init__.py** - Export main pipeline function:
   - `run_wallpaper_pipeline()` - Orchestrator (simplified from `run_pipeline_with_precomputed_normals()`)

**Success Criteria:** Modular pipeline with clean imports

### Phase 3: Create Model Inference Scripts

**Create wrapper scripts for each model:**

1. **models/depth_pro.py** - Wrapper for ML Depth Pro
   - Copy logic from current `run_depth_pro.py`
   - Add class interface: `DepthProModel`
   - Methods: `load()`, `infer(image_path)`, `save_outputs()`

2. **models/rgb2x.py** - Wrapper for RGB→X normal inference
   - Interface to `normal_irradiance/` Gradio demo
   - Class: `RGB2XModel`
   - Methods: `load()`, `infer(image_path)`, `save_outputs()`

3. **models/segformer.py** - Wall segmentation
   - Wrapper for `/Users/.../ST-RoomNet-for-visualisation/.../run_segformer.py`
   - Uses HuggingFace `nvidia/segformer-b2-finetuned-ade-512-512`
   - Class: `SegFormerModel`
   - Methods: `load()`, `infer(image_path)`, `save_wall_mask()`
   - Output: Wall confidence map (H, W) with values 0-1

4. **models/base.py** - Abstract base class
   - Interface all models inherit from
   - Ensures consistent API

**Success Criteria:** Unified model interface, can run each model independently

### Phase 4: Create Configuration System

**Create YAML config files:**

1. **config/pipeline_config.yaml:**
   ```yaml
   wallpaper:
     pattern_width_m: 0.53
     pattern_height_m: 1.0  # Auto-computed if not set

   camera:
     fx_multiplier: 0.8  # Default focal length multiplier
     use_predicted_focal_length: true  # Use depth-pro prediction

   clustering:
     n_clusters: 10
     top_n_clusters: 5

   output:
     save_intermediates: true  # Save step-by-step visualizations
     output_format: "png"
   ```

2. **config/model_paths.yaml:**
   ```yaml
   depth_pro:
     location: "/Users/ishan-aiworkspace/Documents/apps/ml-depth-pro/"
     device: "cpu"  # or "cuda"

   rgb2x:
     location: "./normal_irradiance/"
     run_mode: "colab"  # or "local" when supported

   segformer:
     model: "nvidia/segformer-b2-finetuned-ade-512-512"
     source_script: "/Users/ishan-aiworkspace/Documents/apps/ST-RoomNet-for-visualisation/workflow-v1-separate-processing/1-model-inference/run_segformer.py"
     device: "cpu"  # or "cuda" if available
   ```

**Success Criteria:** Configuration loaded from YAML, no hardcoded paths

### Phase 5: Create Orchestration Scripts

**Create main entry points:**

1. **scripts/run_model_inference.py:**
   - CLI to run individual models
   - Usage: `python run_model_inference.py --model depth_pro --image path/to/img.jpg`
   - Outputs saved to `data/batch_output/{image_name}/`

2. **scripts/run_pipeline.py:**
   - Main pipeline orchestrator
   - Loads config from YAML
   - Accepts image path or uses precomputed inputs
   - Usage: `python run_pipeline.py --image path/to/img.jpg --wallpaper path/to/wallpaper.jpg`
   - Or: `python run_pipeline.py --precomputed data/batch_output/living_simple2/ --wallpaper wallpaper.jpg`

3. **scripts/compare_focal_lengths.py:**
   - Run pipeline twice (predicted vs default focal length)
   - Side-by-side comparison
   - Usage: `python compare_focal_lengths.py --image path/to/img.jpg`

**Success Criteria:** Can run full pipeline end-to-end from command line

### Phase 6: Copy Essential Files

**Copy files from old repo:**

1. Copy `normal_irradiance/` directory (entire RGB→X model code)
2. Copy `normal_irradiance/sample_images/` → `data/sample_images/`
3. Copy `rgb-x_output/batch_output/` → `data/batch_output/`
4. Copy `wallpaper_pipeline.ipynb` → `notebooks/wallpaper_pipeline.ipynb`
5. Copy `planning/` → `docs/planning/`
6. Copy `research/` → `docs/research/`
7. Copy `FOCAL_LENGTH_COMPARISON.md` → `docs/`

**Success Criteria:** All essential data and docs migrated

### Phase 7: Create Documentation

**Create docs:**

1. **README.md:**
   - Project overview
   - Quick start guide
   - Installation instructions
   - Example usage

2. **docs/model_details.md:**
   - Details on each model (RGB→X, SegFormer, Depth Pro)
   - Where they run (local/colab)
   - Model outputs format

3. **requirements.txt:**
   - All Python dependencies
   - Separate sections for local vs cloud

**Success Criteria:** Clear documentation for setup and usage

---

## Verification Plan

**Test the restructured pipeline:**

1. **Model inference test:**
   ```bash
   python scripts/run_model_inference.py --model depth_pro --image data/sample_images/living_simple2.jpg
   # Verify: depth + focal length saved to data/batch_output/living_simple2/
   ```

2. **Pipeline test with precomputed inputs:**
   ```bash
   python scripts/run_pipeline.py --precomputed data/batch_output/living_simple2/ --wallpaper wallpaper.jpg
   # Verify: Final wallpaper result generated in outputs/
   ```

3. **Config modification test:**
   - Edit `config/pipeline_config.yaml` (change pattern_width_m to 1.0)
   - Re-run pipeline
   - Verify: Pattern scale changes

4. **Focal length comparison:**
   ```bash
   python scripts/compare_focal_lengths.py --precomputed data/batch_output/living_simple2/
   # Verify: Two outputs (predicted vs default) generated
   ```

---

## Migration Strategy

**Two-phase approach:**

**Phase A: Build new structure** (this plan)
- Create clean repo structure
- Refactor code into modules
- Test with precomputed inputs

**Phase B: Cloud deployment** (future plan)
- Add Docker containerization
- Create FastAPI server
- Deploy to Google Cloud Run
- Add GCS bucket integration

---

## Success Criteria

- [ ] New repository structure created
- [ ] Core pipeline refactored into 5 modules (geometry, clustering, texture, lighting, visualize)
- [ ] 3 model wrappers created (depth_pro, rgb2x, segformer)
- [ ] YAML configuration system working
- [ ] 3 CLI scripts functional (run_model_inference, run_pipeline, compare_focal_lengths)
- [ ] All essential files migrated (normal_irradiance/, sample data, docs)
- [ ] README and documentation complete
- [ ] Pipeline runs end-to-end with precomputed inputs
- [ ] Can compare predicted vs default focal length
- [ ] Old `rgb-x/` directory can be archived/deleted


Here are all the absolute paths needed to understand and implement the restructuring plan:

Core Implementation Files

/Users/ishan-aiworkspace/Documents/apps/rgb-x/visualize_with_precomputed_normals.py
/Users/ishan-aiworkspace/Documents/apps/rgb-x/run_depth_pro.py
/Users/ishan-aiworkspace/Documents/apps/rgb-x/wallpaper_pipeline.ipynb
Model Implementation Files
Depth Pro:


/Users/ishan-aiworkspace/Documents/apps/ml-depth-pro/README.md
/Users/ishan-aiworkspace/Documents/apps/ml-depth-pro/run_depth_pro.py
RGB→X (Normals):


/Users/ishan-aiworkspace/Documents/apps/rgb-x/normal_irradiance/pipeline_normal_irradiance.py
/Users/ishan-aiworkspace/Documents/apps/rgb-x/normal_irradiance/gradio_demo_normal_irradiance.py
/Users/ishan-aiworkspace/Documents/apps/rgb-x/normal_irradiance/load_image.py
SegFormer (Wall Masks):


/Users/ishan-aiworkspace/Documents/apps/ST-RoomNet-for-visualisation/workflow-v1-separate-processing/1-model-inference/run_segformer.py
/Users/ishan-aiworkspace/Documents/apps/ST-RoomNet-for-visualisation/workflow-v1-separate-processing/1-model-inference/CLAUDE.md
Planning & Documentation

/Users/ishan-aiworkspace/.claude/plans/giggly-honking-starfish.md
/Users/ishan-aiworkspace/Documents/apps/rgb-x/planning/simple-segformer-plan.md
/Users/ishan-aiworkspace/Documents/apps/rgb-x/planning/normal-irradiance-pipeline-plan.md
/Users/ishan-aiworkspace/Documents/apps/rgb-x/planning/restructuring-plan.md
/Users/ishan-aiworkspace/Documents/apps/rgb-x/FOCAL_LENGTH_COMPARISON.md
Sample Data & Outputs

/Users/ishan-aiworkspace/Documents/apps/rgb-x/normal_irradiance/sample_images/
/Users/ishan-aiworkspace/Documents/apps/rgb-x/rgb-x_output/batch_output/
Additional Context (Optional)

/Users/ishan-aiworkspace/Documents/apps/rgb-x/planning/
/Users/ishan-aiworkspace/Documents/apps/rgb-x/research/
Total: 14 critical files + 3 directories that contain everything needed to understand the current implementation and build the restructured version.