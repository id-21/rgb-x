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
        """Save AOV images as NPY/EXR and PNG for download

        Args:
            aov_dict: Dictionary of {aov_name: numpy_array} where numpy_array is (H, W, C) float32
        """
        os.makedirs(output_dir, exist_ok=True)
        saved_files = {}

        for aov_name, img_np in aov_dict.items():
            # img_np is already a numpy array in (H, W, C) format from pipeline output_type="np"

            # Save normal maps as NPY and PNG
            if aov_name == "normal":
                # Save as .npy file (raw numpy array in linear space)
                npy_path = os.path.join(output_dir, "normal_map.npy")
                np.save(npy_path, img_np.astype(np.float32))
                saved_files["normal_npy"] = npy_path

                # Save PNG preview (already in display-ready format)
                png_path = os.path.join(output_dir, "normal_map.png")
                img_png = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
                cv2.imwrite(png_path, cv2.cvtColor(img_png, cv2.COLOR_RGB2BGR))
                saved_files["normal_png"] = png_path

            # Save irradiance as EXR (linear color space for ST-RoomNet V3)
            elif aov_name == "irradiance":
                # INVERT: Model outputs bright=shadow, we need bright=light
                img_inverted = 1.0 - img_np

                # Save NPY (gamma-corrected, inverted)
                npy_path = os.path.join(output_dir, "irradiance_gamma.npy")
                np.save(npy_path, img_inverted.astype(np.float32))
                saved_files["irradiance_npy"] = npy_path

                # Reverse gamma correction to get linear RGB: linear = gamma^2.2
                img_linear = img_inverted ** 2.2
                exr_path = os.path.join(output_dir, "irradiance_shadow.exr")
                cv2.imwrite(exr_path, cv2.cvtColor(img_linear.astype(np.float32), cv2.COLOR_RGB2BGR))
                saved_files["shadow_exr"] = exr_path

                # Also save PNG preview (gamma-corrected for visual verification)
                png_path = os.path.join(output_dir, "irradiance_preview.png")
                img_png = (img_inverted * 255).astype(np.uint8)
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
                # Get numpy array directly from pipeline
                generated_image_np = pipe(
                    prompt=prompt,
                    photo=photo,
                    num_inference_steps=inference_step,
                    height=new_height,
                    width=new_width,
                    generator=generator,
                    required_aovs=[aov_name],
                    output_type="np",  # Get numpy arrays directly
                ).images[0][0]

                # Resize numpy array using cv2
                generated_image_np = cv2.resize(
                    generated_image_np,
                    (old_width, old_height),
                    interpolation=cv2.INTER_LINEAR
                )

                # Store first sample for download (numpy format)
                if i == 0:
                    aov_images[aov_name] = generated_image_np

                # Convert numpy → PIL for Gradio gallery display (Gradio 3.48.0 requires PIL)
                # numpy is in (H, W, C) format, range [0, 1]
                display_image = Image.fromarray(
                    (np.clip(generated_image_np, 0, 1) * 255).astype(np.uint8)
                )
                return_list.append((display_image, f"Generated {aov_name} {i}"))

        # Create white wall × shadow visualization
        if "irradiance" in aov_images:
            # Get irradiance numpy array (already in (H, W, C) format)
            shadow_np = aov_images["irradiance"]

            # Invert: bright areas become lit, dark areas become shadowed
            shadow_inverted = 1.0 - shadow_np

            # Create white surface (same dimensions as shadow)
            white_surface = np.ones_like(shadow_inverted)

            # Multiply: white × shadow = shaded white surface (in gamma space for quick preview)
            shaded_white = white_surface * shadow_inverted

            # Convert numpy → PIL for Gradio display (Gradio 3.48.0 requires PIL)
            shaded_white_image = Image.fromarray(
                (np.clip(shaded_white, 0, 1) * 255).astype(np.uint8)
            )

            # Add to gallery with descriptive label
            return_list.append((shaded_white_image, "White Wall × Shadow (Debug Visualization)"))

        # Save images and get file paths
        saved_files = save_aov_images(aov_images)

        # Return: gallery, normal NPY, normal PNG, irradiance PNG preview, irradiance EXR for ST-RoomNet
        return (
            return_list,
            saved_files.get("normal_npy"),
            saved_files.get("normal_png"),
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

                gr.Markdown("### 📥 Normal Map Downloads")
                normal_npy_download = gr.File(label="Normal Map (NPY - NumPy array)")
                normal_png_download = gr.File(label="Normal Map Preview (PNG)")

                gr.Markdown("### 📥 Shadow Layer Downloads")
                shadow_preview = gr.File(label="Irradiance Preview (PNG)")
                shadow_download = gr.File(label="Shadow Layer (EXR - Use this in ST-RoomNet)")

        inputs = [
            photo,
            seed,
            inference_step,
            num_samples,
        ]
        outputs = [result_gallery, normal_npy_download, normal_png_download, shadow_preview, shadow_download]
        run_button.click(fn=callback, inputs=inputs, outputs=outputs, queue=True)

    return block


if __name__ == "__main__":
    demo = get_rgb2x_demo()
    demo.queue(max_size=1)
    demo.launch(share=True)
