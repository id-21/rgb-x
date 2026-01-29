# Wallpaper Visualization: Implementation Plan

## Starting Point
- ✅ Depth map (numpy array, shape: H×W)
- ✅ Wall segmentation mask (numpy array, shape: H×W, binary)

---

## Pipeline Overview

```
Depth + Mask → Plane Fitting → UV Mapping → Texture Sampling → Composite
```

---

## Step 1: Compute Surface Normals

**Goal**: Get per-pixel normal vectors from depth gradients.

```python
import numpy as np
from scipy.ndimage import sobel

def compute_normals(depth, mask):
    """
    Compute surface normals from depth map.
    Returns: (H, W, 3) normal vectors
    """
    # Smooth depth to reduce noise (critical!)
    from scipy.ndimage import gaussian_filter
    depth_smooth = gaussian_filter(depth, sigma=2)
    
    # Compute gradients
    dz_dx = sobel(depth_smooth, axis=1, mode='constant')  # horizontal
    dz_dy = sobel(depth_smooth, axis=0, mode='constant')  # vertical
    
    # Normal = (-dz/dx, -dz/dy, 1), then normalize
    normals = np.stack([-dz_dx, -dz_dy, np.ones_like(depth)], axis=-1)
    norm = np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-8
    normals = normals / norm
    
    # Mask to wall region only
    normals[~mask] = 0
    
    return normals
```

---

## Step 2: Cluster Wall Pixels into Planes

**Goal**: Group wall pixels by similar normal → identify distinct wall planes.

```python
from sklearn.cluster import DBSCAN
# or use KMeans for speed

def cluster_wall_planes(normals, mask, min_cluster_size=1000):
    """
    Cluster wall pixels by normal direction.
    Returns: (H, W) array with plane labels (-1 = no plane)
    """
    # Extract normals for wall pixels only
    wall_coords = np.argwhere(mask)  # (N, 2) - row, col positions
    wall_normals = normals[mask]     # (N, 3)
    
    # Cluster by normal direction
    # DBSCAN: eps controls how "similar" normals must be
    clustering = DBSCAN(eps=0.1, min_samples=100).fit(wall_normals)
    
    # Build label map
    labels = np.full(mask.shape, -1, dtype=np.int32)
    labels[mask] = clustering.labels_
    
    # Filter tiny clusters
    for label in np.unique(labels):
        if label == -1:
            continue
        if (labels == label).sum() < min_cluster_size:
            labels[labels == label] = -1
    
    return labels
```

---

## Step 3: Fit Single Plane Normal per Cluster

**Goal**: Robust single normal vector for each wall plane.

```python
def fit_plane_normals(normals, labels):
    """
    Compute one representative normal per plane cluster.
    Returns: dict {label: normal_vector}
    """
    plane_normals = {}
    
    for label in np.unique(labels):
        if label == -1:
            continue
        
        cluster_normals = normals[labels == label]
        
        # Robust: median or SVD-based fitting
        # Simple: mean + normalize
        avg_normal = cluster_normals.mean(axis=0)
        avg_normal /= np.linalg.norm(avg_normal)
        
        plane_normals[label] = avg_normal
    
    return plane_normals
```

---

## Step 4: Build UV Coordinate System per Plane

**Goal**: For each plane, define 2D texture axes (u = horizontal, v = vertical on wall).

```python
def build_plane_basis(normal):
    """
    Given a plane normal, return orthonormal (u_axis, v_axis) on the plane.
    u_axis: horizontal on wall
    v_axis: vertical on wall (aligned with gravity)
    """
    # Assume gravity is -Y in camera space (common convention)
    up = np.array([0, -1, 0], dtype=np.float64)
    
    # v_axis: project "up" onto plane, normalize
    v_axis = up - np.dot(up, normal) * normal
    v_axis /= np.linalg.norm(v_axis) + 1e-8
    
    # u_axis: perpendicular to both normal and v
    u_axis = np.cross(normal, v_axis)
    u_axis /= np.linalg.norm(u_axis) + 1e-8
    
    return u_axis, v_axis
```

---

## Step 5: Unproject Pixels to 3D

**Goal**: Convert (x, y, depth) → (X, Y, Z) world coordinates.

```python
def unproject_to_3d(depth, fx, fy, cx, cy):
    """
    Unproject depth map to 3D point cloud.
    
    Args:
        fx, fy: focal lengths (pixels)
        cx, cy: principal point (usually image center)
    
    Returns: (H, W, 3) XYZ coordinates
    """
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    
    Z = depth
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    
    return np.stack([X, Y, Z], axis=-1)
```

**Camera intrinsics estimation** (if unknown):
```python
# Rough estimate for phone cameras
H, W = depth.shape
fx = fy = W * 0.8  # ~50mm equivalent on full-frame
cx, cy = W / 2, H / 2
```

---

## Step 6: Compute UV Coordinates per Pixel

**Goal**: For each wall pixel, get texture coordinates (u, v) in meters.

```python
def compute_uv_coordinates(points_3d, labels, plane_normals):
    """
    Compute (u, v) texture coordinates for each wall pixel.
    Returns: (H, W, 2) UV map
    """
    H, W = labels.shape
    uv_map = np.zeros((H, W, 2), dtype=np.float64)
    
    for label, normal in plane_normals.items():
        plane_mask = (labels == label)
        plane_points = points_3d[plane_mask]  # (N, 3)
        
        # Get plane basis
        u_axis, v_axis = build_plane_basis(normal)
        
        # Pick origin: mean of plane points (or corner)
        origin = plane_points.mean(axis=0)
        
        # Project onto plane axes
        relative = plane_points - origin
        u_coords = relative @ u_axis
        v_coords = relative @ v_axis
        
        uv_map[plane_mask, 0] = u_coords
        uv_map[plane_mask, 1] = v_coords
    
    return uv_map
```

---

## Step 7: Sample Wallpaper Texture

**Goal**: Map (u, v) in meters → wallpaper pixel, with tiling.

```python
def sample_wallpaper(uv_map, labels, wallpaper, pattern_width_m, pattern_height_m):
    """
    Sample wallpaper color for each wall pixel.
    
    Args:
        uv_map: (H, W, 2) in meters
        wallpaper: (Ph, Pw, 3) pattern image
        pattern_width_m, pattern_height_m: physical size of one tile
    
    Returns: (H, W, 3) sampled colors
    """
    H, W = labels.shape
    Ph, Pw = wallpaper.shape[:2]
    
    result = np.zeros((H, W, 3), dtype=np.uint8)
    wall_mask = labels >= 0
    
    # Convert meters → pattern pixels, with tiling
    u_meters = uv_map[wall_mask, 0]
    v_meters = uv_map[wall_mask, 1]
    
    # Normalize to [0, 1] within pattern, then tile
    u_norm = (u_meters / pattern_width_m) % 1.0
    v_norm = (v_meters / pattern_height_m) % 1.0
    
    # Convert to pixel indices
    px = (u_norm * (Pw - 1)).astype(np.int32)
    py = (v_norm * (Ph - 1)).astype(np.int32)
    
    # Sample
    result[wall_mask] = wallpaper[py, px]
    
    return result
```

---

## Step 8: Composite with Lighting Preservation

**Goal**: Blend wallpaper onto original image, keeping shadows/highlights.

```python
import cv2

def composite_with_lighting(original, wallpaper_sampled, mask):
    """
    Apply wallpaper while preserving original lighting.
    """
    # Extract luminance from original wall region
    original_lab = cv2.cvtColor(original, cv2.COLOR_RGB2LAB).astype(np.float32)
    wallpaper_lab = cv2.cvtColor(wallpaper_sampled, cv2.COLOR_RGB2LAB).astype(np.float32)
    
    # Get original luminance
    L_original = original_lab[..., 0]
    L_wallpaper = wallpaper_lab[..., 0]
    
    # Compute ratio (avoid division by zero)
    L_wallpaper_safe = np.clip(L_wallpaper, 1, 255)
    ratio = L_original / L_wallpaper_safe
    
    # Apply ratio to wallpaper luminance
    wallpaper_lab[..., 0] = np.clip(L_wallpaper * ratio, 0, 255)
    
    # Convert back
    result_rgb = cv2.cvtColor(wallpaper_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
    
    # Composite
    output = original.copy()
    output[mask] = result_rgb[mask]
    
    return output
```

---

## Full Pipeline

```python
def apply_wallpaper(image, depth, wall_mask, wallpaper, 
                    pattern_width_m=0.53, pattern_height_m=1.0,
                    fx=None, fy=None):
    """
    Main pipeline.
    """
    H, W = depth.shape
    
    # Default intrinsics
    if fx is None:
        fx = fy = W * 0.8
    cx, cy = W / 2, H / 2
    
    # Step 1: Normals
    normals = compute_normals(depth, wall_mask)
    
    # Step 2: Cluster planes
    labels = cluster_wall_planes(normals, wall_mask)
    
    # Step 3: Fit plane normals
    plane_normals = fit_plane_normals(normals, labels)
    
    # Step 4-5: Unproject to 3D
    points_3d = unproject_to_3d(depth, fx, fy, cx, cy)
    
    # Step 6: UV coordinates
    uv_map = compute_uv_coordinates(points_3d, labels, plane_normals)
    
    # Step 7: Sample wallpaper
    wallpaper_sampled = sample_wallpaper(uv_map, labels, wallpaper,
                                          pattern_width_m, pattern_height_m)
    
    # Step 8: Composite
    result = composite_with_lighting(image, wallpaper_sampled, labels >= 0)
    
    return result
```

---

## Testing Strategy

1. **Start with synthetic data**: Render a simple 3D room, extract ground-truth depth → verify math
2. **Single flat wall**: Test on head-on wall photo first
3. **Corner case**: Two walls meeting at 90°
4. **Real photos**: Test with actual phone camera images

---

## Key Parameters to Tune

| Parameter | Starting Value | Notes |
|-----------|----------------|-------|
| `gaussian_filter sigma` | 2 | Increase if normals too noisy |
| `DBSCAN eps` | 0.1 | Smaller = stricter plane matching |
| `pattern_width_m` | 0.53m | Standard European wallpaper roll width |
| `fx, fy` | W × 0.8 | Adjust based on phone camera |

---

## Potential Issues & Fixes

| Issue | Solution |
|-------|----------|
| Noisy plane boundaries | Morphological cleanup on label map |
| Wrong pattern scale | Add user calibration (mark known distance) |
| Texture seams visible | Blend at cluster boundaries |
| Depth holes | Inpaint depth before processing |