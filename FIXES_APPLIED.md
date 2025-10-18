# Fixes Applied to gradio_demo_normal_irradiance.py

## Problem Summary

The Gradio demo was failing with error:
```
ValueError: Cannot process type as image: <class 'torch.Tensor'>
```

**Root Cause**: Gradio 3.48.0 Gallery component only accepts **PIL Images**, not torch.Tensor objects.

## Solution Overview

Changed the display format from torch.Tensor to PIL.Image while **maintaining** numpy array exports for debugging.

---

## Changes Made

### 1. Fixed Main Gallery Display (Lines 149-154)

**Before** (❌ FAILED):
```python
# Convert to tensor for Gradio gallery display (C, H, W)
display_tensor = torch.from_numpy(generated_image_np.transpose(2, 0, 1))
return_list.append((display_tensor, f"Generated {aov_name} {i}"))
```

**After** (✅ WORKS):
```python
# Convert numpy → PIL for Gradio gallery display (Gradio 3.48.0 requires PIL)
# numpy is in (H, W, C) format, range [0, 1]
display_image = Image.fromarray(
    (np.clip(generated_image_np, 0, 1) * 255).astype(np.uint8)
)
return_list.append((display_image, f"Generated {aov_name} {i}"))
```

**Why it works**:
- Pipeline returns numpy array in (H, W, C) format, range [0, 1]
- `Image.fromarray()` expects uint8 values [0, 255]
- We multiply by 255 and convert to uint8
- Gradio 3.48.0 accepts PIL Images ✓

---

### 2. Fixed White Wall Visualization (Lines 170-176)

**Before** (❌ FAILED):
```python
# Convert to tensor for display (C, H, W)
shaded_white_tensor = torch.from_numpy(shaded_white.transpose(2, 0, 1))
return_list.append((shaded_white_tensor, "White Wall × Shadow (Debug Visualization)"))
```

**After** (✅ WORKS):
```python
# Convert numpy → PIL for Gradio display (Gradio 3.48.0 requires PIL)
shaded_white_image = Image.fromarray(
    (np.clip(shaded_white, 0, 1) * 255).astype(np.uint8)
)
return_list.append((shaded_white_image, "White Wall × Shadow (Debug Visualization)"))
```

---

### 3. Added Irradiance NPY Export (Lines 62-65)

**Added** (✅ NEW FEATURE):
```python
# Save NPY (gamma-corrected, inverted)
npy_path = os.path.join(output_dir, "irradiance_gamma.npy")
np.save(npy_path, img_inverted.astype(np.float32))
saved_files["irradiance_npy"] = npy_path
```

**Why it's needed**:
- Allows debugging of irradiance values
- Matches the export pattern for normal maps
- Provides gamma-corrected values for analysis

---

## Output File Structure

### For Normal Maps:
1. **`normal_map.npy`** - Raw numpy array (H, W, C), float32, linear space, range [0, 1]
2. **`normal_map.png`** - Visual preview, 8-bit PNG

### For Irradiance:
1. **`irradiance_gamma.npy`** - Gamma-corrected numpy array (inverted), float32, range [0, 1]
2. **`irradiance_shadow.exr`** - Linear RGB for ST-RoomNet V3 (gamma reversed: value^2.2)
3. **`irradiance_preview.png`** - Visual preview, 8-bit PNG

### Gradio Gallery Display:
- All images displayed as **PIL.Image** objects
- Compatible with Gradio 3.48.0

---

## Pipeline Output Flow

```
Pipeline (output_type="np")
    ↓
numpy array (H, W, C), float32, range [0, 1]
    ↓
├─→ Save to .npy file (for debugging)
├─→ Save to .exr file (linear space, for ST-RoomNet)
├─→ Save to .png file (8-bit preview)
└─→ Convert to PIL.Image for Gradio display
        ↓
    Image.fromarray((arr * 255).astype(np.uint8))
        ↓
    Gradio Gallery ✓
```

---

## Testing Checklist

When testing in Colab:

- [ ] Gradio interface launches without errors
- [ ] Images display correctly in gallery
- [ ] Can upload and process images
- [ ] Download links work for all file types:
  - [ ] normal_map.npy
  - [ ] normal_map.png
  - [ ] irradiance_gamma.npy
  - [ ] irradiance_shadow.exr
  - [ ] irradiance_preview.png
- [ ] White wall visualization displays correctly
- [ ] No torch.Tensor errors in console

---

## Key Insights

### 1. Pipeline Returns One Format at a Time
The pipeline's `output_type` parameter returns **either** numpy **or** PIL, not both:
- `output_type="np"` → numpy array (H, W, C), range [0, 1]
- `output_type="pil"` → PIL Image (H, W, C), range [0, 255]

### 2. Dual Output Strategy
To get both numpy arrays AND Gradio display:
1. Use `output_type="np"` to get numpy arrays
2. Save numpy arrays to files
3. Convert numpy → PIL for Gradio using `Image.fromarray()`

### 3. Gamma Correction Handling
- **Normal maps**: No gamma correction (linear space)
- **Irradiance**: Gamma correction applied by pipeline (value^(1/2.2))
- **For EXR export**: Reverse gamma (value^2.2) to get linear RGB

### 4. Gradio 3.48.0 Compatibility
- Gradio 3.48.0 Gallery only accepts: **PIL.Image** or **file paths**
- Does NOT accept: torch.Tensor, numpy arrays (directly)
- Newer Gradio (5.x) supports more formats, but we're using 3.48.0 for dependency compatibility

---

## Next Steps

1. **Test in Colab**: Upload the fixed script and run Normal_Irradiance_Colab.ipynb
2. **Process sample images**: Verify all outputs are generated correctly
3. **Check numpy values**: Load .npy files and verify value ranges
4. **Validate EXR files**: Ensure linear RGB is correct for ST-RoomNet integration
5. **Debug if needed**: Add print statements to track data flow

---

## Related Files

- **Fixed script**: [normal_irradiance/gradio_demo_normal_irradiance.py](normal_irradiance/gradio_demo_normal_irradiance.py)
- **Pipeline**: [normal_irradiance/pipeline_normal_irradiance.py](normal_irradiance/pipeline_normal_irradiance.py)
- **Colab notebook**: [Normal_Irradiance_Colab.ipynb](Normal_Irradiance_Colab.ipynb)
- **Working reference**: [rgbx/rgb2x/gradio_demo_rgb2x.py](rgbx/rgb2x/gradio_demo_rgb2x.py)

---

**Date**: 2025-10-18
**Status**: ✅ Ready for testing in Colab
