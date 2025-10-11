# Understanding Shadow Application for Wallpaper Visualization

## Project Context: RGB→X Model

The RGB→X diffusion model takes a single photograph and decomposes it into physically meaningful channels called **AOVs (Arbitrary Output Variables)**. These are the fundamental properties that determine how surfaces appear.

### Model Outputs

**Albedo (Diffuse Basecolor)**
- The "true" color of a surface without any lighting effects
- What the material would look like under perfectly uniform white light
- Example: A red wall's albedo is pure red, even if it appears dark red in shadow

**Surface Normals**
- The orientation of each surface point in 3D space
- Critical for understanding wall geometry and perspective
- ST-RoomNet provides these for your wallpaper warping needs

**Roughness**
- How smooth or rough a surface is (0 = mirror, 1 = matte)
- Determines how sharply light reflects
- A scalar value, not a color

**Metallic**
- Whether a surface behaves like metal (1) or dielectric/non-metal (0)
- Affects how light interacts with the material
- Also a scalar value, not a color

**Irradiance (Diffuse Lighting)**
- The total incoming light arriving at each surface point
- Captures both direct illumination and shadows
- This is what you need for shadow extraction

## Understanding Gamma Correction

### The Human Vision Problem

Human eyes perceive brightness non-linearly. We're incredibly sensitive to changes in dark values (can distinguish many shades of dark gray) but less sensitive to changes in bright values. If we stored images the way physics works (linearly), we'd waste storage on bright distinctions we can't see and lack precision in shadows where we need it.

### Two Color Spaces

**Linear RGB (Physical Space)**
- How light actually behaves in the real world
- Value 0.5 means literally half the light energy of 1.0
- Math works correctly: averaging two light sources, multiplying reflections, etc.
- Used internally by rendering engines and physics simulations

**sRGB (Perceptual/Display Space)**
- Gamma-encoded to match human perception
- Approximately: sRGB = Linear^(1/2.2)
- More bits dedicated to dark values where we notice differences
- What JPEGs, PNGs, and monitors expect
- A midtone gray (linear 0.5) is stored as ~0.73 in sRGB to look correct on screen

### The Conversion

**Linear to sRGB**: Raise to power 1/2.2 (compress brights, expand darks)  
**sRGB to Linear**: Raise to power 2.2 (expand brights, compress darks)

### Why It Matters for AOVs

The RGB→X model makes intelligent decisions about which outputs need gamma correction:

**Gamma-Corrected Outputs (Albedo, Irradiance)**
- These represent colors meant for human viewing
- Saved with gamma encoding so they display correctly on screens
- If you save linear values as PNG, they look washed out and pale

**Linear Outputs (Roughness, Metallic)**
- These are numerical parameters for rendering equations
- A roughness of 0.5 means "50% rough" in shader math
- Gamma encoding would corrupt the value: rendering engines would interpret 0.5 as 0.21
- Must stay in linear space for physical accuracy

## When Linear Space Is Critical

### Use Linear Space For:

**Physically-Based Rendering (PBR)**
- When creating 3D visualizations or architectural renders
- Light transport calculations require linear math
- Color mixing, reflections, and refractions only work correctly in linear space
- All professional rendering engines (Unreal, Unity, Blender) work in linear space internally

**Accurate Color Blending**
- Averaging colors: (red + blue) / 2 gives correct purple only in linear space
- Multiplying reflections: surface_color × light_color requires linear values
- Any mathematical operation on light values needs linear space

**Multi-Layer Compositing**
- When stacking multiple shadow layers, transparencies, or lighting effects
- Accumulated errors from gamma-space math become visible
- VFX industry standard is linear workflow

**Vibrant/Saturated Colors**
- Saturated reds, blues, greens show color shifts if processed in sRGB
- The gamma curve affects color channels differently, causing hue shifts

### You Can Use sRGB For:

**Simple Image Compositing**
- Single-layer blends or overlays
- When matching perceptual appearance matters more than physical accuracy
- Many photo editing tools (Photoshop layer modes) work in sRGB
- Fast previews where absolute accuracy isn't critical

**Your Wallpaper Visualization**
- Since you're doing a relatively simple shadow overlay (not full rendering)
- The perceptual error is small and may be acceptable
- Significantly faster processing
- However, test both approaches to compare

## Shadow Extraction Strategy

### Understanding Irradiance

The model's irradiance output represents incoming light, which naturally contains shadow information:
- **Bright areas**: Directly lit by windows, lamps, or ambient light
- **Dark areas**: Shadowed by objects (furniture, decorations) or self-shadowing (corners)
- **Gradient transitions**: Soft shadow edges where light partially reaches

### The Extraction Process

**Step 1: Isolate Wall Region**  
Use ST-RoomNet's normal detection to identify which pixels belong to your target wall. This ensures you're only analyzing shadow patterns on the wall, not on furniture or other surfaces.

**Step 2: Establish Baseline Brightness**  
Find the brightest areas of the wall (typically the 98th percentile to avoid outliers like specular highlights). This represents the "fully lit" reference that shadows will be relative to.

**Step 3: Create Relative Shadow Mask**  
Divide the irradiance by the baseline. Now you have values where:
- 1.0 = fully lit areas
- 0.5 = half as bright (moderate shadow)
- 0.2 = heavily shadowed

**Step 4: Clamp to Preserve Ambient Light**  
Real rooms always have ambient light from bounce lighting, so even the darkest shadows aren't pure black. Clamp the minimum to 0.4-0.5 to maintain realism.

**Step 5: Apply to Wallpaper**  
Multiply your warped wallpaper pattern by this shadow mask. Areas with value 1.0 remain unchanged; areas with value 0.5 darken by 50%.

## The Wallpaper Visualization Workflow

**Input**: Room photograph with existing walls  
**Goal**: Show how new wallpaper would look in that exact lighting

1. **Extract Scene Properties**
   - ST-RoomNet detects wall normals (surface orientation)
   - RGB→X model predicts irradiance (lighting and shadows)

2. **Prepare Wallpaper**
   - Start with your flat wallpaper pattern (no perspective, no shadows)
   - Apply perspective transformation based on wall normals
   - This warps the pattern to match the wall's viewing angle

3. **Extract Shadow Pattern**
   - Normalize irradiance relative to the wall's brightest points
   - Create a multiplicative mask representing shadow intensity
   - Optionally reduce shadow strength for commercial visualization aesthetics

4. **Composite Final Result**
   - Multiply warped wallpaper by shadow mask
   - Shadows from vases, furniture, window frames now appear on wallpaper
   - Result looks like the wallpaper was actually in the room when photographed

## Key Considerations for Maximum Realism

### Shadow Strength Control

Full-strength shadows (directly multiplying by irradiance) may look too dramatic for commercial visualization. Most interior design tools use 70-80% shadow strength to maintain "aspirational" aesthetics while preserving realism.

### Ambient Occlusion

The darkest shadows should preserve some ambient light. Real rooms have light bouncing from multiple surfaces. A completely black shadow suggests no ambient lighting, which looks unnatural. Clamping to 40-50% minimum brightness prevents this.

### Color Fidelity

If your wallpaper has saturated colors (deep blues, vibrant reds), working in linear space prevents hue shifts that occur when multiplying in sRGB space. Test with your most saturated patterns to decide.

### Edge Handling

Objects that you're digitally removing may have cast shadows. Consider whether you need to inpaint those shadow regions before applying the shadow mask to the wallpaper, or whether the shadow adds realism (e.g., a new wallpaper behind where a painting used to be might naturally show the painting's shadow).

### Texture Preservation

If your wallpaper pattern has its own texture or embossing that creates micro-shadows, be careful not to double-darken. The pattern's intrinsic variation should be preserved while room shadows are added separately.

## Quality Validation Checklist

**Physical Plausibility**
- Shadows align with visible objects in the scene
- Shadow direction matches light source positions (windows, lamps)
- Soft shadows have gradual transitions, not hard edges

**Color Accuracy**
- Wallpaper hue remains consistent between lit and shadowed areas
- No unexpected color shifts or desaturation in shadows
- Saturated colors don't look muddy or contaminated

**Perceptual Realism**
- No pure black regions (unless the room is genuinely unlit)
- Shadow intensity feels natural, not overdramatic
- The composite looks like a single photograph, not a layered fake

**Technical Correctness**
- No visible seams at perspective warp boundaries
- Shadow mask doesn't extend beyond wall boundaries
- Resolution and detail match throughout the image

## Choosing Your Approach

**Start with sRGB** if:
- You need fast processing for interactive previews
- Your wallpapers have moderate saturation
- The visual result looks good in testing

**Switch to Linear** if you notice:
- Hue shifts in saturated colors (reds becoming orange, blues becoming purple)
- Muddy or overly dark shadows that look artificial
- You're comparing against professional architectural renderings
- You plan to do additional compositing or lighting adjustments

The diffusion model's strength is capturing realistic shadow patterns that traditional computer vision struggles with. By carefully applying these shadows to your wallpaper, you bridge the gap between flat pattern and lived-in realism.