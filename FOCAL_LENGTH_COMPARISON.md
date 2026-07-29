# Comparing Predicted vs Default Focal Length

This guide shows how to use the Depth Pro model's predicted focal length and compare the wallpaper mapping results with the default focal length.

## Step 1: Run Depth Pro to Get Predictions

Use `run_depth_pro.py` to generate depth map and focal length predictions:

```bash
python run_depth_pro.py --image path/to/your/image.jpg --output output_directory/
```

This will create:
- `image_depth.npy` - Depth map in meters
- `image_focal_length.npy` - Predicted focal length in pixels
- `image_focal_length.txt` - Human-readable focal length info with comparison

## Step 2: Use Predicted Focal Length in Pipeline

Edit [visualize_with_precomputed_normals.py](visualize_with_precomputed_normals.py) and set:

```python
# Enable predicted focal length
focal_length_path = DATA_DIR / "image_focal_length.npy"
```

Then run the pipeline:

```bash
python visualize_with_precomputed_normals.py
```

## Step 3: Compare with Default Focal Length

To compare, run the pipeline twice:

### Run 1: With Predicted Focal Length
```python
focal_length_path = DATA_DIR / "image_focal_length.npy"
output_dir = DATA_DIR / "results_predicted_focal_length"
```

### Run 2: With Default Focal Length
```python
focal_length_path = None  # Use default
output_dir = DATA_DIR / "results_default_focal_length"
```

## Understanding the Difference

The focal length affects how 3D coordinates are unprojected from the depth map:

```python
X = (u - cx) * Z / fx  # fx is the focal length
Y = (v - cy) * Z / fy
```

**Higher focal length** = narrower field of view = points spread further apart in 3D
**Lower focal length** = wider field of view = points closer together in 3D

This affects:
- UV coordinate spacing on walls
- Wallpaper pattern scale and alignment
- How "stretched" or "compressed" the wallpaper appears

## Expected Differences

If predicted focal length differs significantly from default (0.8 × width):
- **Wallpaper scale may change** - patterns could appear larger or smaller
- **UV coordinate distribution changes** - affects tiling alignment
- **Perspective correction accuracy** - predicted focal length should be more accurate

Check the `parameters.json` in each output directory to see which focal length was used:
```json
{
  "focal_length_px": "960.00",
  "focal_length_source": "predicted",
  "focal_length_file": "path/to/focal_length.npy"
}
```

## Example Workflow

```bash
# 1. Generate depth and focal length from image
python run_depth_pro.py -i bedroom.jpg -o output/

# 2. Run pipeline with predicted focal length
# (Edit visualize_with_precomputed_normals.py to set focal_length_path)
python visualize_with_precomputed_normals.py

# 3. Compare side-by-side
# Look at the final_result.png in each output directory
```

## Tips

- The `run_depth_pro.py` script prints the ratio between predicted and default focal length
- A ratio close to 1.0 means they're similar and results should look nearly identical
- A ratio > 1.2 or < 0.8 means significant difference - results may vary noticeably
- Predicted focal length is generally more accurate for metric depth reconstruction
