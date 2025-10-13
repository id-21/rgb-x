# RGB2X Model Transition - Research Document

**Research Date:** Oct 13, 2025
**Research Type:** Model Architecture Investigation
**Researcher:** Human/Claude
**Status:** Planning Phase

---

## Executive Summary

This research document explores transitioning from ST-RoomNet to RGB↔X (RGB-to-X) diffusion model for geometric understanding in the wallpaper visualization pipeline. RGB2X offers per-pixel continuous normal vectors instead of discrete plane segmentation, enabling more accurate perspective-correct wallpaper application for complex room geometries.

---

## Problem Statement with ST-RoomNet

### Current Architecture Limitations

**ST-RoomNet Approach:**
- Outputs 5-class segmentation (0=ceiling, 1-3=walls, 4=floor)
- Provides single homography matrix (8 parameters → 3×3 theta)
- Assumes **cuboid room geometry** (orthogonal walls, Manhattan world)
- Normal vectors extracted via homography decomposition

**Critical Issues:**
1. **Discrete plane assumption**: Only detects 3 wall planes, fails on:
   - L-shaped rooms
   - Angled walls (non-orthogonal)
   - Alcoves, bay windows
   - Irregular architectural features

2. **Single homography limitation**: One transformation for entire scene
   - Cannot handle multiple vanishing points
   - Breaks on complex perspectives

3. **Normal vector accuracy**: Normals derived from homography are approximate
   - Depends on reference image alignment
   - No per-pixel geometric detail
   - Subject to aspect ratio distortion (v3 known issue)

---

## RGB↔X Model Overview

### Architecture

**Paper:** "RGB↔X: Image Decomposition and Synthesis Using Material- and Lighting-aware Diffusion Models" (SIGGRAPH 2024)
**Model:** Diffusion-based (Stable Diffusion backbone)
**Training:** Interior scenes dataset with ground truth intrinsics

### Model Outputs (RGB→X Direction)

RGB2X produces **5 intrinsic channels** from a single RGB image:

| Channel | Description | Format | Relevance |
|---------|-------------|--------|-----------|
| **Normal** | Camera-space surface normals | RGB-encoded XYZ vectors | **PRIMARY - needed for walls** |
| Albedo | Diffuse base color | sRGB | Not needed |
| Roughness | Surface roughness | Grayscale | Not needed |
| Metallic | Metallicity | Grayscale | Not needed |
| Irradiance | Diffuse lighting (shadows) | Linear RGB | **SECONDARY - for shadow overlay** |

### Normal Map Encoding

```python
# RGB → XYZ Normal Decoding
normal_rgb = model_output  # Range [0, 1] from model
normal_xyz = (normal_rgb * 2.0) - 1.0  # Convert to [-1, 1]
# Normalize to unit vectors
normal_xyz = normal_xyz / np.linalg.norm(normal_xyz, axis=-1, keepdims=True)
```

**Camera-space convention:**
- X: Right (+) / Left (-)
- Y: Down (+) / Up (-)
- Z: Away from camera (+) / Toward camera (-)

---

## Proposed Approaches

### **Approach 1: Pure Normal Clustering**

**Pipeline:**
1. RGB2X → extract normal map (H×W×3)
2. Decode normals to XYZ vectors
3. **K-means clustering** on normal vectors (k=3-5 clusters)
4. Filter for wall-like orientations: `|normal.y| < 0.3` (reject floor/ceiling)
5. Extract connected components per cluster → Wall objects
6. Set `Wall.normal_vector` = cluster centroid, `Wall.pixel_mask` = membership

**Pros:**
- Simple, interpretable
- No dependency on semantic models
- Works for any planar surface

**Cons:**
- No semantic understanding (may detect furniture, art on walls)
- Sensitive to noise (diffusion model variance)
- Requires manual k selection or cluster validation
- No distinction between wall/non-wall planes

---

### **Approach 2: Hybrid RGB2X + SegFormer (RECOMMENDED)**

**Pipeline:**
1. **SegFormer** → semantic wall mask (ADE20K label 0="wall")
2. **RGB2X** → per-pixel normal vectors
3. **Masked clustering**: Cluster normals **only within SegFormer wall regions**
4. **Plane fitting**: RANSAC or least-squares fit per cluster
5. Generate Wall objects with:
   - `normal_vector`: Fitted plane normal (high accuracy)
   - `pixel_mask`: Cluster mask ∩ SegFormer mask
   - `corners_2d`: Boundary tracing of mask

**Pros:**
- **Best of both worlds**: Semantic filtering + geometric precision
- Robust to furniture (SegFormer filters it out)
- Handles non-orthogonal walls (clustering adapts to actual geometry)
- Can detect 5+ walls in complex rooms
- Maintains existing SegFormer integration (minimal pipeline disruption)

**Cons:**
- Two model inference passes (slower)
- Dependency on SegFormer accuracy
- Clustering hyperparameters need tuning

---

### **Approach 3: Normal Edge Detection**

**Pipeline:**
1. RGB2X → normal map
2. Compute **normal discontinuity map**: `∇normal = ||∂normal/∂x|| + ||∂normal/∂y||`
3. Threshold discontinuities → wall boundaries
4. Flood-fill regions with consistent normals
5. Validate planarity (low normal variance) → reject non-planar regions

**Pros:**
- Directly detects geometric boundaries
- No clustering hyperparameters
- Good for sharp architectural edges

**Cons:**
- Sensitive to normal map noise
- May over-segment (detects fine details)
- Requires post-processing to merge similar regions
- No semantic filtering (furniture problem)

---

## Technical Comparison: ST-RoomNet vs RGB2X

| Aspect | ST-RoomNet | RGB2X |
|--------|------------|-------|
| **Geometric output** | 5 discrete planes | Per-pixel continuous normals |
| **Room assumption** | Cuboid (3 walls max) | No assumption (any geometry) |
| **Normal accuracy** | Homography-derived (approximate) | Per-pixel diffusion output (high detail) |
| **Inference time** | ~50ms (TF model) | ~5-10s (diffusion, 50 steps) |
| **Model size** | 111MB (.h5 file) | ~5GB (Stable Diffusion backbone) |
| **GPU memory** | ~2GB | ~12GB (FP16) |
| **Resolution handling** | Fixed 400×400 | Flexible (downsampled to multiples of 8) |
| **Wall detection** | Direct (class labels 1-3) | Indirect (requires clustering) |

---

## Data Structure Compatibility

### Current Wall Class (unchanged)

```python
class Wall:
    wall_id: str
    corners_2d: List[Tuple[float, float]]  # Boundary points
    normal_vector: Optional[np.ndarray]    # [x, y, z] unit vector
    pixel_mask: Optional[np.ndarray]       # Binary mask
    surface_type: str = "wall"
    is_visible: bool = True
    confidence: float = 1.0
```

### RGB2X Pipeline Output (new)

```python
def extract_walls_from_rgb2x(
    image: np.ndarray,
    segformer_mask: np.ndarray,
    normal_map: np.ndarray,
    num_walls: int = 3
) -> List[Wall]:
    """
    Extract Wall objects from RGB2X normal map + SegFormer semantics

    Args:
        image: Original room image (H×W×3)
        segformer_mask: Binary wall mask from SegFormer (H×W)
        normal_map: RGB2X normal output (H×W×3), RGB-encoded
        num_walls: Expected number of walls (for k-means)

    Returns:
        List of Wall objects with accurate normals and masks
    """
    # Decode normals
    normals_xyz = (normal_map * 2.0) - 1.0
    normals_xyz /= np.linalg.norm(normals_xyz, axis=-1, keepdims=True)

    # Filter wall pixels (from SegFormer)
    wall_pixels = segformer_mask > 0.5
    wall_normals = normals_xyz[wall_pixels]

    # Cluster normals (K-means on unit sphere)
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=num_walls, random_state=42)
    labels = kmeans.fit_predict(wall_normals)

    # Generate Wall objects
    walls = []
    for i in range(num_walls):
        wall = Wall(wall_id=f"wall_{i}")

        # Cluster mask
        cluster_mask = np.zeros_like(segformer_mask)
        cluster_mask[wall_pixels] = (labels == i)

        # Extract connected components (handle disjoint regions)
        from scipy.ndimage import label as connected_components
        labeled, num_components = connected_components(cluster_mask)

        # Take largest component
        if num_components > 0:
            largest_component = np.argmax([
                np.sum(labeled == j) for j in range(1, num_components + 1)
            ]) + 1
            wall.pixel_mask = (labeled == largest_component).astype(np.uint8)

            # Refined normal (average of cluster)
            wall.normal_vector = kmeans.cluster_centers_[i]

            # Extract corners (boundary tracing)
            wall.corners_2d = extract_boundary_points(wall.pixel_mask)

            # Confidence (planarity metric)
            cluster_normals = normals_xyz[wall.pixel_mask > 0]
            normal_variance = np.var(cluster_normals, axis=0).sum()
            wall.confidence = np.exp(-normal_variance * 10)  # High planarity → high confidence

            walls.append(wall)

    return walls
```

---

## Experiment Plan

### Phase 1: Proof of Concept (Approach 2)

**Test Setup:**
- Dataset: 5 room images (simple cuboid, L-shaped, angled walls, alcove, irregular)
- Compare: ST-RoomNet vs RGB2X wall detection

**Metrics:**
1. **Wall count accuracy**: Does RGB2X detect all visible walls?
2. **Normal vector accuracy**: Compare to manual ground truth (protractor method)
3. **Mask quality**: IoU with manual annotations
4. **Edge precision**: SSAE (sum of squared angular errors) at boundaries

**Implementation Steps:**
1. Integrate RGB2X inference (load model, run normal extraction)
2. Implement hybrid clustering pipeline (SegFormer + normals)
3. Generate Wall objects from clusters
4. Visualize: overlay normals as color, show cluster boundaries
5. Quantitative comparison with ST-RoomNet outputs

---

### Phase 2: Performance Optimization

**Challenges:**
- RGB2X inference time: 5-10s (vs 50ms for ST-RoomNet)
- GPU memory: 12GB requirement
- Inference steps: 50 steps default (tunable 10-100)

**Optimization Strategies:**
1. **Reduce inference steps**: Test 20, 30, 40 steps → quality vs speed trade-off
2. **Model quantization**: FP16 → INT8 (may degrade normals)
3. **Resolution downsampling**: Process at 512×512 → upsample normals
4. **Caching**: Pre-compute normals for test images (offline processing acceptable?)
5. **Batching**: Process multiple images in parallel (if applicable)

---

### Phase 3: Integration with V3 Pipeline

**Modified Pipeline:**
```
Input Image
    ↓
[SegFormer] → Semantic wall mask (ADE20K)
    ↓
[RGB2X] → Normal map + Irradiance (shadow overlay)
    ↓
[Hybrid Clustering] → Wall objects with normals
    ↓
[Wallpaper Application] → Perspective-correct transforms
    ↓
[Shadow Overlay] → Apply irradiance modulation (existing v3 work)
    ↓
Output Visualization
```

**Integration Points:**
- Replace: `workflow-v3/modules/st_roomnet_model.py`
- New module: `workflow-v3/modules/rgb2x_model.py`
- New module: `workflow-v3/modules/normal_clustering.py`
- Modify: `workflow-v3/modules/end_to_end_pipeline.py`

---

## Benefits of RGB2X Transition

### Geometric Advantages
1. **Arbitrary wall detection**: Not limited to 3 walls, handles complex rooms
2. **Non-orthogonal support**: Works with angled walls, curved surfaces (approximated as planes)
3. **Per-pixel accuracy**: High-resolution normals for fine details (molding, trim)
4. **No Manhattan assumption**: Adapts to actual room geometry

### Pipeline Improvements
1. **Better perspective correction**: Accurate normals → better homography estimation
2. **Irradiance bonus**: Shadow overlay comes "free" from same model
3. **Material awareness**: Albedo/roughness available for future PBR rendering
4. **Unified model**: Single diffusion model vs dual ST-RoomNet+SegFormer

### Research Value
1. **State-of-art**: SIGGRAPH 2024 (vs ST-RoomNet 2019)
2. **Extensibility**: X→RGB direction enables texture synthesis, relighting
3. **Active development**: Hugging Face models, ongoing improvements

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Inference time (5-10s)** | User experience degradation | Reduce steps to 20-30, optimize with TensorRT |
| **GPU memory (12GB)** | Hardware requirement increase | Quantization, cloud deployment option |
| **Clustering instability** | Inconsistent wall detection | RANSAC outlier rejection, min cluster size threshold |
| **Normal map noise** | Poor plane fitting | Gaussian smoothing, median filtering pre-clustering |
| **Complex room over-segmentation** | Too many walls detected | Merge similar normals (angular threshold <15°) |
| **Model hallucination** | Incorrect normals on textures | SegFormer pre-filtering, confidence thresholding |

---

## Next Steps

### Immediate Actions (Research Phase)
1. ✅ **DONE**: Understand RGB2X model architecture and outputs
2. ✅ **DONE**: Identify 3 candidate approaches
3. ✅ **DONE**: Document in research file

### Short-term (Planning Phase)
4. **Test RGB2X inference**: Run on 5 test images, validate normal maps visually
5. **Implement Approach 2 prototype**: Hybrid SegFormer + RGB2X clustering
6. **Quantitative validation**: Measure normal accuracy vs ST-RoomNet
7. **Create planning document**: `planning/05-rgb2x-integration-plan.md`

### Long-term (Implementation Phase)
8. **Module development**: Create `rgb2x_model.py` and `normal_clustering.py`
9. **Pipeline integration**: Modify `end_to_end_pipeline.py`
10. **Performance benchmarking**: Optimize inference time
11. **A/B testing**: Compare v3-ST-RoomNet vs v3-RGB2X outputs

---

## References

### Papers
- RGB↔X: Zeng et al., "Image Decomposition and Synthesis Using Material- and Lighting-aware Diffusion Models", SIGGRAPH 2024
- ST-RoomNet: (Original ST-RoomNet paper reference needed)

### Code Repositories
- RGB2X Hugging Face: `zheng95z/rgb-to-x`
- Local implementation: `/Users/ishan-aiworkspace/Documents/apps/rgb-x/rgbx/rgb2x/`

### Related Research Documents
- `04-shadow-overlay-integration-research.md`: Irradiance usage
- `03-normal-detection-and-shading.md`: Normal vector validation
- `02-resolution-aspect-ratio-analysis.md`: Resolution constraints

---

## Appendix A: RGB2X Model Details

**Model Architecture:**
- Base: Stable Diffusion 1.5
- Custom UNet: 8-channel input (4 latent + 4 image conditioning)
- Custom VAE processor: AOV-specific postprocessing
- Scheduler: DDIM with zero-SNR beta rescaling

**Inference Parameters:**
- Default steps: 50
- Guidance scale: 0.0 (unconditional, image-conditioned only)
- Image guidance scale: 0.0
- Resolution: Multiple of 8 (model constraint)
- Max recommended: 1000px (per demo code)

**Output Channels:**
- Albedo: Gamma-corrected sRGB (γ=2.2)
- Normal: Camera-space, RGB-encoded, gamma-corrected for visualization
- Roughness: Grayscale, gamma-corrected
- Metallic: Grayscale, gamma-corrected
- Irradiance: Linear RGB (saved as EXR), gamma-corrected PNG preview

---

## Appendix B: Clustering Algorithm Considerations

### K-means on Unit Sphere
**Pros:** Fast, simple, fixed k
**Cons:** Requires pre-specified k, sensitive to initialization

### DBSCAN (Density-based)
**Pros:** Automatic k, handles noise
**Cons:** Angular distance metric needed, epsilon parameter tuning

### Mean Shift
**Pros:** Automatic k, robust to outliers
**Cons:** Slow on high-res images, bandwidth parameter tuning

### Hierarchical Clustering
**Pros:** Dendrogram for optimal k selection
**Cons:** O(n²) complexity, slow

**Recommendation:** Start with K-means (k=3-5), validate with silhouette score, fallback to DBSCAN if k selection is poor.

---

## Appendix C: Code Snippets

### Loading RGB2X Model

```python
import torch
from diffusers import DDIMScheduler
from pipeline_rgb2x import StableDiffusionAOVMatEstPipeline

def load_rgb2x_model(cache_dir="/path/to/cache"):
    pipe = StableDiffusionAOVMatEstPipeline.from_pretrained(
        "zheng95z/rgb-to-x",
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
    ).to("cuda")

    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config,
        rescale_betas_zero_snr=True,
        timestep_spacing="trailing"
    )
    pipe.set_progress_bar_config(disable=True)
    return pipe
```

### Extracting Normals

```python
def extract_normals(pipe, image, num_steps=50):
    """
    Args:
        pipe: RGB2X pipeline
        image: RGB image tensor (3×H×W), range [0,1]
        num_steps: Inference steps (20-50 recommended)

    Returns:
        normal_map: (H×W×3) numpy array, range [0,1] RGB-encoded
    """
    generator = torch.Generator(device="cuda").manual_seed(42)

    # Ensure multiple of 8
    H, W = image.shape[1:]
    H_new = (H // 8) * 8
    W_new = (W // 8) * 8
    image_resized = torchvision.transforms.Resize((H_new, W_new))(image)

    normal_output = pipe(
        prompt="Camera-space Normal",
        photo=image_resized,
        num_inference_steps=num_steps,
        height=H_new,
        width=W_new,
        generator=generator,
        required_aovs=["normal"],
    ).images[0][0]  # First sample, first (only) AOV

    # Convert to numpy
    normal_np = np.array(normal_output).astype(np.float32) / 255.0
    return normal_np
```

---

**End of Research Document**
