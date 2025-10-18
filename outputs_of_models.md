Answers to Your Questions
1. What is the output of the pipeline?
The pipeline defined in pipeline_normal_irradiance.py has one output format at a time, controlled by the output_type parameter: Pipeline Flow (lines 798-817):
VAE decode → torch.Tensor (C, H, W) 
→ image_processor.postprocess(output_type=...)
→ Returns ONE of: 'latent', 'pt', 'np', or 'pil'
Postprocess method (lines 59-109):
Input: torch.Tensor in (C, H, W) format
Processing:
Denormalize (if enabled)
Gamma correction: image = torch.pow(image, 1.0 / 2.2) (if enabled)
Convert based on output_type:
output_type	Return Format	Range	Shape
'latent'	torch.Tensor	raw	(C, H, W)
'pt'	torch.Tensor	[0, 1]	(C, H, W)
'np'	numpy.ndarray	[0, 1]	(H, W, C) ← Note transpose!
'pil'	PIL.Image	[0, 255]	(H, W, C)
Key Point: The pipeline returns EITHER numpy OR PIL, not both. The conversion happens inside postprocess():
Line 103: image = self.pt_to_numpy(image) converts (C,H,W) → (H,W,C)
Line 109: return self.numpy_to_pil(image) converts numpy [0,1] → PIL [0,255]
2. How to accept numpy and convert to PIL in gradio_demo_normal_irradiance.py?
Current Code (lines 127-151):
# Gets numpy array (H, W, C), range [0, 1]
generated_image_np = pipe(
    ...,
    output_type="np",  # Returns numpy array
).images[0][0]

# Resize the numpy array
generated_image_np = cv2.resize(generated_image_np, ...)

# Store for file export
if i == 0:
    aov_images[aov_name] = generated_image_np  # Store numpy

# Convert to tensor for Gradio ❌ THIS FAILS
display_tensor = torch.from_numpy(generated_image_np.transpose(2, 0, 1))
return_list.append((display_tensor, f"Generated {aov_name} {i}"))
Proposed Fix (using PIL.Image):
# Gets numpy array (H, W, C), range [0, 1]
generated_image_np = pipe(
    ...,
    output_type="np",
).images[0][0]

# Resize the numpy array
generated_image_np = cv2.resize(generated_image_np, ...)

# Store numpy for file export
if i == 0:
    aov_images[aov_name] = generated_image_np

# Convert numpy → PIL for Gradio display ✅ THIS WORKS
# numpy is in [0, 1], convert to [0, 255] uint8 for PIL
display_image = Image.fromarray(
    (np.clip(generated_image_np, 0, 1) * 255).astype(np.uint8)
)
return_list.append((display_image, f"Generated {aov_name} {i}"))
Why this works:
output_type="np" gives us numpy in (H, W, C) format, range [0, 1]
We keep the numpy array for saving to .npy files
We convert numpy → PIL for Gradio display using Image.fromarray()
Gradio 3.48.0 accepts PIL Images but rejects torch.Tensor
Same fix for white wall visualization (lines 153-171):
# Create white wall × shadow visualization
shadow_np = aov_images["irradiance"]
shadow_inverted = 1.0 - shadow_np
white_surface = np.ones_like(shadow_inverted)
shaded_white = white_surface * shadow_inverted

# Convert to PIL instead of tensor ✅
shaded_white_image = Image.fromarray(
    (np.clip(shaded_white, 0, 1) * 255).astype(np.uint8)
)
return_list.append((shaded_white_image, "White Wall × Shadow (Debug)"))
Summary
Answer 1: The pipeline returns one format at a time - you choose output_type when calling the pipeline. It's either numpy OR PIL, not both. Answer 2: Use output_type="np" to get numpy arrays, then manually convert numpy → PIL using Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)) for Gradio display. This lets you:
✅ Save raw numpy arrays for debugging
✅ Display PIL images in Gradio (compatible with v3.48.0)
✅ Export EXR files with proper gamma handling
Would you like me to proceed with implementing these fixes?