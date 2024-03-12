import argparse
import time

import torch
from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image
from transformers import CLIPVisionModelWithProjection
from diffusers.utils import load_image


def run_pipeline(weight: float = 0.7, weight2: float = 0.3, strength: float = 0.5, prompt: str = '1cat, 4k, masterpiece', steps: int = 50,
                 image: str = None):
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "h94/IP-Adapter",
        subfolder="models/image_encoder",
        torch_dtype=torch.float16,
    )
    pipeline = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        image_encoder=image_encoder,
    ).to('cuda')
    pipeline.load_ip_adapter(
        "h94/IP-Adapter",
        subfolder="sdxl_models",
        weight_name=["ip-adapter-plus_sdxl_vit-h.safetensors", "ip-adapter-plus-face_sdxl_vit-h.safetensors"]
    )
    pipeline.set_ip_adapter_scale([weight, weight2])
    origin_image = load_image(image)
    style_images = [load_image(f"ziggy/img{i}.png") for i in range(10)]

    images = pipeline(
        prompt=prompt,
        # image=origin_image,
        # strength=strength,
        ip_adapter_image=[style_images, origin_image],
        negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
        num_inference_steps=steps,
        num_images_per_prompt=4,
    ).images
    for image in images:
        stamp = int(time.time() * 1000)
        image.save(f'output/{stamp}.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", default=0.7)
    parser.add_argument("--weight2", default=0.3)
    parser.add_argument("--strength", default=0.5)
    parser.add_argument("--steps", default=50)
    parser.add_argument("--image", default='https://static.interval.im/interval/8NWwFF8RRhs2HaBS.jpeg')
    parser.add_argument("--prompt", default='1man, 4k, masterpiece')
    args = parser.parse_args()

    run_pipeline(
        weight=float(args.weight),
        weight2=float(args.weight2),
        strength=float(args.strength),
        steps=int(args.steps),
        image=args.image,
        prompt=args.prompt,
    )
