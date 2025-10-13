import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import gradio as gr
import torch
import torchvision
import cv2
import numpy as np
from diffusers import DDIMScheduler
from load_image import load_exr_image, load_ldr_image
from pipeline_normal_irradiance import StableDiffusionAOVMatEstPipeline
from PIL import Image

current_directory = os.path.dirname(os.path.abspath(__file__))


def get_rgb2x_demo():
    # Load pipeline
    pipe = StableDiffusionAOVMatEstPipeline.from_pretrained(
        "zheng95z/rgb-to-x",
        torch_dtype=torch.float16,
        cache_dir=os.path.join(current_directory, "model_cache_normal_irradiance"),
    ).to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config, rescale_betas_zero_snr=True, timestep_spacing="trailing"
    )
    pipe.set_progress_bar_config(disable=True)
    pipe.to("cuda")

    # Helper function to save AOV images
    def save_aov_images(aov_dict, output_dir="/tmp/rgb2x_output"):
        """Save AOV images as EXR and PNG for download"""
        os.makedirs(output_dir, exist_ok=True)
        saved_files = {}

        for aov_name, image_tensor in aov_dict.items():
            # Convert tensor to numpy
            if isinstance(image_tensor, torch.Tensor):
                img_np = image_tensor.cpu().numpy()
                if img_np.ndim == 3 and img_np.shape[0] in [1, 3]:
                    img_np = img_np.transpose(1, 2, 0)
            else:
                img_np = np.array(image_tensor)

            # Save irradiance as EXR (linear color space for ST-RoomNet V3)
            if aov_name == "irradiance":
                exr_path = os.path.join(output_dir, "irradiance_shadow.exr")
                # INVERT: Model outputs bright=shadow, we need bright=light
                img_np = 1.0 - img_np
                # Reverse gamma correction to get linear RGB: linear = gamma^2.2
                img_linear = img_np ** 2.2
                cv2.imwrite(exr_path, cv2.cvtColor(img_linear.astype(np.float32), cv2.COLOR_RGB2BGR))
                saved_files["shadow_exr"] = exr_path

                # Also save PNG preview (gamma-corrected for visual verification)
                png_path = os.path.join(output_dir, "irradiance_preview.png")
                img_png = (img_np * 255).astype(np.uint8)
                cv2.imwrite(png_path, cv2.cvtColor(img_png, cv2.COLOR_RGB2BGR))
                saved_files["shadow_png"] = png_path

        return saved_files

    # Augmentation
    def callback(
        photo,
        seed,
        inference_step,
        num_samples,
    ):
        generator = torch.Generator(device="cuda").manual_seed(seed)

        if photo.name.endswith(".exr"):
            photo = load_exr_image(photo.name, tonemaping=True, clamp=True).to("cuda")
        elif (
            photo.name.endswith(".png")
            or photo.name.endswith(".jpg")
            or photo.name.endswith(".jpeg")
        ):
            photo = load_ldr_image(photo.name, from_srgb=True).to("cuda")

        # Check if the width and height are multiples of 8. If not, crop it using torchvision.transforms.CenterCrop
        old_height = photo.shape[1]
        old_width = photo.shape[2]
        new_height = old_height
        new_width = old_width
        radio = old_height / old_width
        max_side = 1000
        if old_height > old_width:
            new_height = max_side
            new_width = int(new_height / radio)
        else:
            new_width = max_side
            new_height = int(new_width * radio)

        if new_width % 8 != 0 or new_height % 8 != 0:
            new_width = new_width // 8 * 8
            new_height = new_height // 8 * 8

        photo = torchvision.transforms.Resize((new_height, new_width))(photo)

        required_aovs = ["normal", "irradiance"]
        prompts = {
            "normal": "Camera-space Normal",
            "irradiance": "Irradiance (diffuse lighting)",
        }

        return_list = []
        aov_images = {}  # Store AOV images for download

        for i in range(num_samples):
            for aov_name in required_aovs:
                prompt = prompts[aov_name]
                generated_image = pipe(
                    prompt=prompt,
                    photo=photo,
                    num_inference_steps=inference_step,
                    height=new_height,
                    width=new_width,
                    generator=generator,
                    required_aovs=[aov_name],
                ).images[0][0]

                generated_image = torchvision.transforms.Resize(
                    (old_height, old_width)
                )(generated_image)

                # Store first sample for download
                if i == 0:
                    aov_images[aov_name] = generated_image

                return_list.append((generated_image, f"Generated {aov_name} {i}"))

        # Create white wall × shadow visualization
        if "irradiance" in aov_images:
            # Get irradiance and invert it (model outputs bright=shadow, we need bright=light)
            irradiance = aov_images["irradiance"]

            # Convert to numpy for processing
            if isinstance(irradiance, torch.Tensor):
                shadow_np = irradiance.cpu().numpy()
                if shadow_np.ndim == 3 and shadow_np.shape[0] in [1, 3]:
                    shadow_np = shadow_np.transpose(1, 2, 0)
            else:
                shadow_np = np.array(irradiance)

            # Invert: bright areas become lit, dark areas become shadowed
            shadow_inverted = 1.0 - shadow_np

            # Create white surface (same dimensions as shadow)
            white_surface = np.ones_like(shadow_inverted)

            # Multiply: white × shadow = shaded white surface (in gamma space for quick preview)
            shaded_white = white_surface * shadow_inverted

            # Convert back to tensor for display
            shaded_white_tensor = torch.from_numpy(shaded_white.transpose(2, 0, 1))

            # Add to gallery with descriptive label
            return_list.append((shaded_white_tensor, "White Wall × Shadow (Debug Visualization)"))

        # Save images and get file paths
        saved_files = save_aov_images(aov_images)

        # Return: gallery, irradiance PNG preview, irradiance EXR for ST-RoomNet
        return (
            return_list,
            saved_files.get("shadow_png"),
            saved_files.get("shadow_exr"),
        )

    block = gr.Blocks()
    with block:
        with gr.Row():
            gr.Markdown("## Model RGB -> X (Realistic image -> Intrinsic channels)")
        with gr.Row():
            # Input side
            with gr.Column():
                gr.Markdown("### Given Image")
                photo = gr.File(label="Photo", file_types=[".exr", ".png", ".jpg"])

                gr.Markdown("### Parameters")
                run_button = gr.Button(label="Run")
                with gr.Accordion("Advanced options", open=False):
                    seed = gr.Slider(
                        label="Seed",
                        minimum=-1,
                        maximum=2147483647,
                        step=1,
                        randomize=True,
                    )
                    inference_step = gr.Slider(
                        label="Inference Step",
                        minimum=1,
                        maximum=100,
                        step=1,
                        value=50,
                    )
                    num_samples = gr.Slider(
                        label="Samples",
                        minimum=1,
                        maximum=100,
                        step=1,
                        value=1,
                    )

            # Output side
            with gr.Column():
                gr.Markdown("### Output Gallery")
                result_gallery = gr.Gallery(
                    label="Output",
                    show_label=False,
                    elem_id="gallery",
                    columns=3,
                )

                gr.Markdown("### 📥 Normal + Shadow Output")
                shadow_preview = gr.File(label="Irradiance Preview (PNG)")
                shadow_download = gr.File(label="Shadow Layer (EXR - Use this in ST-RoomNet)")

        inputs = [
            photo,
            seed,
            inference_step,
            num_samples,
        ]
        outputs = [result_gallery, shadow_preview, shadow_download]
        run_button.click(fn=callback, inputs=inputs, outputs=outputs, queue=True)

    return block


if __name__ == "__main__":
    demo = get_rgb2x_demo()
    demo.queue(max_size=1)
    demo.launch(share=True)
