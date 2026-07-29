"""
Wallpaper Pipeline Visualization with Pre-computed Normals

This script runs the wallpaper visualization pipeline using a pre-computed
normal map instead of computing it from depth. It saves visualization previews
of each major step.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
import cv2
from scipy.ndimage import gaussian_filter, sobel
from sklearn.cluster import KMeans
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib for file output
plt.rcParams['figure.figsize'] = (15, 5)
plt.rcParams['figure.dpi'] = 150  # Higher DPI for better quality saved images


# ============================================================================
# Helper Functions
# ============================================================================

def normalize_for_display(img):
    """Normalize image to 0-255 range for display."""
    img = img.copy()
    img = img - img.min()
    img = img / (img.max() + 1e-8)
    return (img * 255).astype(np.uint8)


def save_visualization(images, titles, filename, output_dir, figsize=(20, 5), cmap=None):
    """
    Save multiple images side by side to a file.

    Args:
        images: List of numpy arrays
        titles: List of titles for each image
        filename: Output filename
        output_dir: Output directory path
        figsize: Figure size tuple
        cmap: Colormap (can be single value or list)
    """
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    for i, (img, title) in enumerate(zip(images, titles)):
        if isinstance(cmap, list):
            c = cmap[i]
        else:
            c = cmap

        if len(img.shape) == 2:
            axes[i].imshow(img, cmap=c if c else 'viridis')
        else:
            axes[i].imshow(img)
        axes[i].set_title(title, fontsize=12, fontweight='bold')
        axes[i].axis('off')

    plt.tight_layout()
    output_path = output_dir / filename
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  ✓ Saved: {filename}")


# ============================================================================
# Pipeline Functions (extracted from notebook)
# ============================================================================

def cluster_wall_planes_kmeans(normals, mask, n_clusters=10, top_n=5):
    """
    Cluster wall pixels by normal direction using KMeans.

    Args:
        normals: (H, W, 3) normal vectors
        mask: (H, W) boolean mask
        n_clusters: Number of KMeans clusters
        top_n: Number of largest clusters to keep

    Returns:
        labels: (H, W) array with plane labels (-1 = no plane, 0-4 = top clusters)
        cluster_sizes: List of (label, size) tuples for all clusters
    """
    # Extract normals for wall pixels only
    wall_coords = np.argwhere(mask)  # (N, 2) - row, col positions
    wall_normals = normals[mask]     # (N, 3)

    print(f"    Clustering {len(wall_normals)} wall pixels with KMeans...")

    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(wall_normals)

    # Build full label map
    labels = np.full(mask.shape, -1, dtype=np.int32)
    labels[mask] = cluster_labels

    # Count cluster sizes
    cluster_sizes = [(label, (labels == label).sum()) for label in range(n_clusters)]
    cluster_sizes.sort(key=lambda x: x[1], reverse=True)

    # Keep only top N largest clusters
    top_labels = {label for label, _ in cluster_sizes[:top_n]}

    # Create a mapping from old labels to new labels (0-4 for top 5)
    label_mapping = {}
    for new_label, (old_label, _) in enumerate(cluster_sizes[:top_n]):
        label_mapping[old_label] = new_label

    # Apply mapping: keep top clusters, discard rest
    new_labels = np.full(mask.shape, -1, dtype=np.int32)
    for old_label, new_label in label_mapping.items():
        new_labels[labels == old_label] = new_label

    return new_labels, cluster_sizes


def fit_plane_normals(normals, labels):
    """
    Compute one representative normal per plane cluster.

    Args:
        normals: (H, W, 3) normal vectors
        labels: (H, W) plane labels

    Returns:
        plane_normals: dict {label: normal_vector}
    """
    plane_normals = {}

    for label in np.unique(labels):
        if label == -1:
            continue

        cluster_normals = normals[labels == label]

        # Average normals and re-normalize
        avg_normal = cluster_normals.mean(axis=0)
        avg_normal = avg_normal / (np.linalg.norm(avg_normal) + 1e-8)

        plane_normals[label] = avg_normal

    return plane_normals


def build_plane_basis(normal):
    """
    Given a plane normal, return orthonormal (u_axis, v_axis) on the plane.

    Args:
        normal: (3,) unit normal vector

    Returns:
        u_axis: (3,) horizontal axis on plane
        v_axis: (3,) vertical axis on plane (aligned with gravity)
    """
    # Assume gravity is -Y in camera space
    up = np.array([0, -1, 0], dtype=np.float64)

    # v_axis: project "up" onto plane, normalize
    v_axis = up - np.dot(up, normal) * normal
    v_axis = v_axis / (np.linalg.norm(v_axis) + 1e-8)

    # u_axis: perpendicular to both normal and v
    u_axis = np.cross(normal, v_axis)
    u_axis = u_axis / (np.linalg.norm(u_axis) + 1e-8)

    return u_axis, v_axis


def unproject_to_3d(depth, fx, fy, cx, cy):
    """
    Unproject depth map to 3D point cloud.

    Args:
        depth: (H, W) depth map
        fx, fy: Focal lengths in pixels
        cx, cy: Principal point coordinates

    Returns:
        points_3d: (H, W, 3) XYZ coordinates
    """
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    Z = depth
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    points_3d = np.stack([X, Y, Z], axis=-1)

    return points_3d


def compute_uv_coordinates(points_3d, labels, plane_normals):
    """
    Compute (u, v) texture coordinates for each wall pixel.

    Args:
        points_3d: (H, W, 3) 3D point cloud
        labels: (H, W) plane labels
        plane_normals: dict {label: normal_vector}

    Returns:
        uv_map: (H, W, 2) UV coordinates in meters
    """
    H, W = labels.shape
    uv_map = np.zeros((H, W, 2), dtype=np.float64)

    for label, normal in plane_normals.items():
        plane_mask = (labels == label)
        plane_points = points_3d[plane_mask]  # (N, 3)

        if len(plane_points) == 0:
            continue

        # Get plane basis
        u_axis, v_axis = build_plane_basis(normal)

        # Pick origin: mean of plane points
        origin = plane_points.mean(axis=0)

        # Project onto plane axes
        relative = plane_points - origin
        u_coords = relative @ u_axis
        v_coords = relative @ v_axis

        uv_map[plane_mask, 0] = u_coords
        uv_map[plane_mask, 1] = v_coords

    return uv_map


def sample_wallpaper(uv_map, labels, wallpaper, pattern_width_m, pattern_height_m):
    """
    Sample wallpaper color for each wall pixel.

    Args:
        uv_map: (H, W, 2) UV coordinates in meters
        labels: (H, W) plane labels
        wallpaper: (Ph, Pw, 3) pattern image
        pattern_width_m: Physical width of pattern in meters
        pattern_height_m: Physical height of pattern in meters

    Returns:
        sampled: (H, W, 3) sampled wallpaper colors
    """
    H, W = labels.shape
    Ph, Pw = wallpaper.shape[:2]

    # Auto-compute height from width
    pattern_height_m = pattern_width_m * (Ph / Pw)

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

    # Clamp to valid range (safety)
    px = np.clip(px, 0, Pw - 1)
    py = np.clip(py, 0, Ph - 1)

    # Sample
    result[wall_mask] = wallpaper[py, px]

    return result


def composite_with_lighting(original, wallpaper_sampled, mask):
    """
    Apply wallpaper while preserving original lighting.

    Args:
        original: (H, W, 3) original RGB image
        wallpaper_sampled: (H, W, 3) sampled wallpaper colors
        mask: (H, W) boolean mask for wall regions

    Returns:
        result: (H, W, 3) composited image
        L_original: Original luminance channel
        L_wallpaper: Wallpaper luminance channel
        lighting_factor: Lighting factor applied
    """
    # Convert to LAB color space
    original_lab = cv2.cvtColor(original, cv2.COLOR_RGB2LAB).astype(np.float32)
    wallpaper_lab = cv2.cvtColor(wallpaper_sampled, cv2.COLOR_RGB2LAB).astype(np.float32)

    # Get luminance channels
    L_original = original_lab[..., 0]
    L_wallpaper = wallpaper_lab[..., 0]

    # Compute lighting factor relative to wall average (preserves pattern)
    L_original_mean = L_original[mask].mean()
    lighting_factor = L_original / (L_original_mean + 1e-8)

    # Apply lighting to wallpaper (scales luminance, preserving pattern variation)
    wallpaper_lab[..., 0] = np.clip(L_wallpaper * lighting_factor, 0, 255)

    # Convert back to RGB
    result_rgb = cv2.cvtColor(wallpaper_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)

    # Composite: replace wall regions only
    output = original.copy()
    output[mask] = result_rgb[mask]

    return output, L_original, L_wallpaper, lighting_factor


# ============================================================================
# Main Pipeline with Pre-computed Normals
# ============================================================================

def run_pipeline_with_precomputed_normals(
    rgb_image_path,
    depth_map_path,
    wall_mask_path,
    normal_map_path,  # NEW: Pre-computed normal file
    wallpaper_path,
    output_dir,
    pattern_width_m=0.53,
    pattern_height_m=1.0,
    fx_multiplier=0.8,
    focal_length_path=None,  # NEW: Optional predicted focal length file
    n_clusters=10,
    top_n_clusters=5
):
    """
    Run the complete wallpaper pipeline with pre-computed normals.
    Saves visualization previews at each step.
    """

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 80)
    print("WALLPAPER PIPELINE WITH PRE-COMPUTED NORMALS")
    print("=" * 80)

    # ========================================================================
    # Step 0: Load Input Data
    # ========================================================================
    print("\n[Step 0] Loading input data...")

    # Load RGB image
    rgb_image = cv2.imread(str(rgb_image_path))
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    print(f"  ✓ RGB image: {rgb_image.shape}")

    # Load depth map
    depth_map = np.load(str(depth_map_path))
    print(f"  ✓ Depth map: {depth_map.shape} (range: {depth_map.min():.3f} to {depth_map.max():.3f})")

    # Load wall mask
    wall_mask = np.load(str(wall_mask_path))
    if wall_mask.dtype != bool:
        wall_mask = wall_mask > 0.5
    print(f"  ✓ Wall mask: {wall_mask.shape} ({100*wall_mask.sum()/wall_mask.size:.1f}% walls)")

    # Load PRE-COMPUTED NORMALS (instead of computing from depth)
    normals = np.load(str(normal_map_path))
    print(f"  ✓ Pre-computed normals: {normals.shape}")
    print(f"    Normal range X: [{normals[..., 0].min():.3f}, {normals[..., 0].max():.3f}]")
    print(f"    Normal range Y: [{normals[..., 1].min():.3f}, {normals[..., 1].max():.3f}]")
    print(f"    Normal range Z: [{normals[..., 2].min():.3f}, {normals[..., 2].max():.3f}]")

    # Load wallpaper
    wallpaper = cv2.imread(str(wallpaper_path))
    wallpaper = cv2.cvtColor(wallpaper, cv2.COLOR_BGR2RGB)
    print(f"  ✓ Wallpaper: {wallpaper.shape}")

    # Save input visualization
    save_visualization(
        [rgb_image, normalize_for_display(depth_map), wall_mask.astype(np.uint8)*255, wallpaper],
        ['Original Image', 'Depth Map', 'Wall Mask', 'Wallpaper Texture'],
        '00_inputs.png',
        output_dir,
        figsize=(20, 5),
        cmap=[None, 'viridis', 'gray', None]
    )

    # ========================================================================
    # Step 1: Visualize Pre-computed Normals
    # ========================================================================
    print("\n[Step 1] Visualizing pre-computed normals...")

    # Convert normals from [-1, 1] to [0, 255] for visualization
    normals_vis = ((normals + 1) * 127.5).astype(np.uint8)
    normals_masked = normals_vis.copy()
    normals_masked[~wall_mask] = 0

    save_visualization(
        [normals_vis, normals_masked],
        ['Normal Map (Full)', 'Normal Map (Masked to Walls)'],
        '01_precomputed_normals.png',
        output_dir,
        figsize=(15, 7)
    )

    # ========================================================================
    # Step 2: Cluster Wall Planes
    # ========================================================================
    print("\n[Step 2] Clustering wall planes...")

    labels, cluster_sizes = cluster_wall_planes_kmeans(
        normals, wall_mask,
        n_clusters=n_clusters,
        top_n=top_n_clusters
    )

    unique_labels = [l for l in np.unique(labels) if l >= 0]
    print(f"    ✓ Kept {len(unique_labels)} plane clusters")

    # Visualize clusters
    labels_colored = np.zeros((*labels.shape, 3), dtype=np.uint8)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))[:, :3] * 255

    for i, label in enumerate(unique_labels):
        color = colors[i % len(colors)]
        labels_colored[labels == label] = color

    # Overlay on original
    overlay = rgb_image.copy().astype(np.float32)
    alpha = 0.6
    mask_3d = (labels >= 0)[..., None]
    overlay = overlay * (1 - alpha * mask_3d) + labels_colored * alpha * mask_3d
    overlay = overlay.astype(np.uint8)

    save_visualization(
        [labels_colored, overlay],
        [f'Plane Clusters ({len(unique_labels)} planes)', 'Clusters Overlaid on Image'],
        '02_plane_clusters.png',
        output_dir,
        figsize=(15, 7)
    )

    # ========================================================================
    # Step 3: Fit Plane Normals
    # ========================================================================
    print("\n[Step 3] Fitting plane normals...")

    plane_normals = fit_plane_normals(normals, labels)
    print(f"    ✓ Fitted {len(plane_normals)} plane normals")

    # ========================================================================
    # Step 4: Unproject to 3D
    # ========================================================================
    print("\n[Step 4] Unprojecting to 3D point cloud...")

    H, W = depth_map.shape

    # Use predicted focal length if provided, otherwise use default
    if focal_length_path is not None:
        focal_length_px = np.load(str(focal_length_path)).item()
        fx = fy = focal_length_px
        print(f"    Using PREDICTED focal length: {focal_length_px:.2f} pixels")
        print(f"    (Default would be: {W * fx_multiplier:.2f} pixels)")
    else:
        fx = fy = W * fx_multiplier
        print(f"    Using DEFAULT focal length: {fx:.2f} pixels (fx_multiplier={fx_multiplier})")

    cx, cy = W / 2, H / 2

    points_3d = unproject_to_3d(depth_map, fx, fy, cx, cy)
    print(f"    ✓ Point cloud: {points_3d.shape}")

    # ========================================================================
    # Step 5: Compute UV Coordinates
    # ========================================================================
    print("\n[Step 5] Computing UV coordinates...")

    uv_map = compute_uv_coordinates(points_3d, labels, plane_normals)
    print(f"    ✓ UV coordinates: {uv_map.shape}")

    # Visualize UV
    u_display = normalize_for_display(uv_map[..., 0])
    v_display = normalize_for_display(uv_map[..., 1])

    wall_mask_3d = (labels >= 0)
    u_display[~wall_mask_3d] = 0
    v_display[~wall_mask_3d] = 0

    save_visualization(
        [u_display, v_display],
        ['U Coordinates (Horizontal)', 'V Coordinates (Vertical)'],
        '03_uv_coordinates.png',
        output_dir,
        figsize=(15, 7),
        cmap='coolwarm'
    )

    # ========================================================================
    # Step 6: Sample Wallpaper
    # ========================================================================
    print("\n[Step 6] Sampling wallpaper texture...")

    wallpaper_sampled = sample_wallpaper(
        uv_map, labels, wallpaper,
        pattern_width_m, pattern_height_m
    )
    print(f"    ✓ Wallpaper sampled: {wallpaper_sampled.shape}")

    save_visualization(
        [rgb_image, wallpaper_sampled, wallpaper],
        ['Original Image', 'Sampled Wallpaper (No Lighting)', 'Wallpaper Pattern'],
        '04_wallpaper_sampled.png',
        output_dir,
        figsize=(20, 7)
    )

    # ========================================================================
    # Step 7: Apply Lighting Preservation
    # ========================================================================
    print("\n[Step 7] Applying lighting preservation...")

    wall_mask_final = (labels >= 0)
    result, L_original, L_wallpaper, lighting_factor = composite_with_lighting(
        rgb_image, wallpaper_sampled, wall_mask_final
    )
    print(f"    ✓ Lighting applied")

    # Visualize lighting transfer
    save_visualization(
        [L_original, L_wallpaper, normalize_for_display(lighting_factor)],
        ['Original Luminance', 'Wallpaper Luminance', 'Lighting Factor'],
        '05_lighting_transfer.png',
        output_dir,
        figsize=(20, 5),
        cmap='gray'
    )

    # ========================================================================
    # Step 8: Final Result
    # ========================================================================
    print("\n[Step 8] Saving final result...")

    save_visualization(
        [rgb_image, wallpaper_sampled, result],
        ['Original', 'Wallpaper (No Lighting)', 'Final Result (With Lighting)'],
        '06_final_result.png',
        output_dir,
        figsize=(20, 7)
    )

    # Save side-by-side comparison
    save_visualization(
        [rgb_image, result],
        ['Original Image', 'Final Result'],
        '07_comparison.png',
        output_dir,
        figsize=(16, 8)
    )

    # Save final result as standalone image
    cv2.imwrite(
        str(output_dir / 'final_result.png'),
        cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    )
    print(f"  ✓ Saved: final_result.png")

    # ========================================================================
    # Save Parameters and Metadata
    # ========================================================================
    print("\n[Step 9] Saving metadata...")

    params = {
        'pattern_width_m': pattern_width_m,
        'pattern_height_m': pattern_height_m,
        'fx_multiplier': fx_multiplier,
        'n_clusters': n_clusters,
        'top_n_clusters': top_n_clusters,
        'image_size': f"{W} x {H}",
        'focal_length_px': f"{fx:.2f}",
        'focal_length_source': 'predicted' if focal_length_path else 'default',
        'focal_length_file': str(focal_length_path) if focal_length_path else None,
        'num_planes_detected': len(plane_normals),
        'used_precomputed_normals': True,
        'normal_file': str(normal_map_path)
    }

    with open(output_dir / 'parameters.json', 'w') as f:
        json.dump(params, f, indent=2)
    print(f"  ✓ Saved: parameters.json")

    print("\n" + "=" * 80)
    print(f"✓ PIPELINE COMPLETED!")
    print(f"  All outputs saved to: {output_dir}")
    print("=" * 80)

    return result, {
        'normals': normals,
        'labels': labels,
        'plane_normals': plane_normals,
        'uv_map': uv_map,
        'wallpaper_sampled': wallpaper_sampled,
        'result': result
    }


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    # # Configure paths
    # DATA_DIR = Path("/Users/ishan-aiworkspace/Documents/apps/rgb-x/rgb-x_output/batch_output/bed_back_bedroom/")

    # # Input files
    # rgb_image_path = DATA_DIR / "bed_back_bedroom.jpg"
    # depth_map_path = DATA_DIR / "bed_back_depth.npy"
    # wall_mask_path = DATA_DIR / "bed_back_bedroom_wall_confidence.npy"
    # normal_map_path = DATA_DIR / "bed_back_bedroom_normal.npy"  # PRE-COMPUTED NORMALS
    # wallpaper_path = Path("/Users/ishan-aiworkspace/Documents/apps/rgb-x/rgb-x_output/batch_output/30106.jpg")

    # # Optional: Use predicted focal length (from run_depth_pro.py)
    # # focal_length_path = DATA_DIR / "bed_back_bedroom_focal_length.npy"
    # focal_length_path = None  # Set to None to use default fx_multiplier

    # # Output directory
    # output_dir = DATA_DIR / "wallpaper_with_precomputed_normals_energie"

    # # Pipeline parameters
    # params = {
    #     'pattern_width_m': 0.53,
    #     'pattern_height_m': 1.0,
    #     'fx_multiplier': 0.8,
    #     'n_clusters': 10,
    #     'top_n_clusters': 5
    # }

    # Configure paths
    DATA_DIR = Path("/Users/ishan-aiworkspace/Documents/apps/rgb-x/rgb-x_output/batch_output/living_simple2/")

    # Input files
    rgb_image_path = DATA_DIR / "living_simple2.jpg"
    depth_map_path = Path("/Users/ishan-aiworkspace/Documents/apps/rgb-x/normal_irradiance/example/living_simple2_depth.npy")
    wall_mask_path = DATA_DIR / "living_simple2_wall_confidence.npy"
    normal_map_path = DATA_DIR / "living_simple2_normal.npy"
    wallpaper_path = Path("/Users/ishan-aiworkspace/Documents/apps/rgb-x/rgb-x_output/batch_output/30106.jpg")  # Or your preferred wallpaper

    # USE PREDICTED FOCAL LENGTH
    focal_length_path = Path("/Users/ishan-aiworkspace/Documents/apps/rgb-x/normal_irradiance/example/living_simple2_focal_length.npy")

    # Output directory
    output_dir = DATA_DIR / "wallpaper_predicted_focal_length"

    # Pipeline parameters
    params = {
        'pattern_width_m': 0.53,
        'pattern_height_m': 1.0,
        'fx_multiplier': 0.8,  # This will be ignored when focal_length_path is set
        'n_clusters': 10,
        'top_n_clusters': 5
    }


    # Run pipeline
    result, intermediates = run_pipeline_with_precomputed_normals(
        rgb_image_path=rgb_image_path,
        depth_map_path=depth_map_path,
        wall_mask_path=wall_mask_path,
        normal_map_path=normal_map_path,
        wallpaper_path=wallpaper_path,
        output_dir=output_dir,
        focal_length_path=focal_length_path,
        **params
    )
