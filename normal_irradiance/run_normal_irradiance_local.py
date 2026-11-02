#!/usr/bin/env python3
"""
Local (non-cloud) CLI runner for the normal_irradiance RGB->X model.

Why this exists
----------------
The original `gradio_demo_normal_irradiance.py` hardcodes:
    torch_dtype=torch.float16
    .to("cuda")
That combination only works on an NVIDIA GPU. On a Mac (CPU or Apple
Silicon MPS), float16 kernels for many ops (esp. GroupNorm/Conv used in
the UNet and VAE) are either missing or numerically unstable, which is
almost certainly the "weights mismatch"-flavored crash you hit before.

This script:
  - auto-detects the best available device: cuda > mps > cpu
  - uses float32 everywhere except actual CUDA (where fp16 is fine and fast)
  - has no Gradio dependency, so it's easier to debug from a terminal
  - downloads weights from HuggingFace (zheng95z/rgb-to-x) on first run,
    cached under normal_irradiance/model_cache_normal_irradiance/

Usage
-----
    cd normal_irradiance
    python run_normal_irradiance_local.py --photo path/to/room.jpg --steps 50

Notes
-----
  - CPU inference on a UNet-based diffusion pipeline is slow (minutes per
    AOV, not seconds). MPS (Apple Silicon GPU) will be much faster but
    less battle-tested for this diffusers version -- if MPS throws op
    errors, use --device cpu to force the safe fallback.
  - First run downloads ~4-5GB of weights from HuggingFace. Make sure
    you have disk space and are not on a network that blocks
    huggingface.co.
"""
import argparse
import os
import sys

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)


def pick_device(requested: str | None):
    import torch

    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser(description="Run RGB->X normal+irradiance locally")
    parser.add_argument("--photo", required=True, help="Input .png/.jpg/.jpeg/.exr image")
    parser.add_argument("--steps", type=int, default=50, help="Inference steps (fewer = faster, lower quality)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["cuda", "mps", "cpu"], default=None,
                         help="Force a device; default auto-detects best available")
    parser.add_argument("--max-side", type=int, default=768,
                         help="Resize longest side to this many px before inference (smaller = faster on CPU/MPS)")
    parser.add_argument("--out-dir", default="/tmp/rgb2x_output")
    args = parser.parse_args()

    import cv2
    import numpy as np
    import torch
    import torchvision
    from diffusers import DDIMScheduler
    from load_image import load_exr_image, load_ldr_image
    from pipeline_normal_irradiance import StableDiffusionAOVMatEstPipeline

    device = pick_device(args.device)
    # Only CUDA gets fp16; MPS/CPU use fp32 to avoid the exact class of
    # errors ("weights mismatch" / unsupported Half ops) hit previously.
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"[run_normal_irradiance_local] device={device} dtype={dtype}")

    cache_dir = os.path.join(CURRENT_DIR, "model_cache_normal_irradiance")
    print(f"[run_normal_irradiance_local] loading pipeline (cache_dir={cache_dir}) ...")
    pipe = StableDiffusionAOVMatEstPipeline.from_pretrained(
        "zheng95z/rgb-to-x",
        torch_dtype=dtype,
        cache_dir=cache_dir,
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config, rescale_betas_zero_snr=True, timestep_spacing="trailing"
    )
    pipe.set_progress_bar_config(disable=False)

    # --- load + preprocess image ---
    photo_path = args.photo
    if photo_path.lower().endswith(".exr"):
        photo = load_exr_image(photo_path, tonemaping=True, clamp=True)
    else:
        photo = load_ldr_image(photo_path, from_srgb=True)
    photo = photo.to(device=device, dtype=dtype)

    old_height, old_width = photo.shape[1], photo.shape[2]
    ratio = old_height / old_width
    max_side = args.max_side
    if old_height > old_width:
        new_height = max_side
        new_width = int(new_height / ratio)
    else:
        new_width = max_side
        new_height = int(new_width * ratio)
    new_width = max(8, new_width // 8 * 8)
    new_height = max(8, new_height // 8 * 8)
    photo = torchvision.transforms.Resize((new_height, new_width))(photo)

    required_aovs = ["normal", "irradiance"]
    prompts = {
        "normal": "Camera-space Normal",
        "irradiance": "Irradiance (diffuse lighting)",
    }

    os.makedirs(args.out_dir, exist_ok=True)
    generator = torch.Generator(device=device if device != "mps" else "cpu").manual_seed(args.seed)

    results = {}
    for aov_name in required_aovs:
        print(f"[run_normal_irradiance_local] generating {aov_name} ...")
        img_np = pipe(
            prompt=prompts[aov_name],
            photo=photo,
            num_inference_steps=args.steps,
            height=new_height,
            width=new_width,
            generator=generator,
            required_aovs=[aov_name],
            output_type="np",
        ).images[0][0]
        img_np = cv2.resize(img_np, (old_width, old_height), interpolation=cv2.INTER_LINEAR)
        results[aov_name] = img_np

        npy_path = os.path.join(args.out_dir, f"{aov_name}.npy")
        np.save(npy_path, img_np.astype(np.float32))
        png_path = os.path.join(args.out_dir, f"{aov_name}.png")
        cv2.imwrite(png_path, cv2.cvtColor((np.clip(img_np, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        print(f"  -> {npy_path}\n  -> {png_path}")

    print(f"[run_normal_irradiance_local] done. Outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
