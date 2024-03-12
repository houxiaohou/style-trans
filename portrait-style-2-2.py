"""
真人风格化

实现方式 2.2（图生图，ControlNet + IP-Adapter风格图）：
"""
import time

import torch
from diffusers import StableDiffusionXLControlNetImg2ImgPipeline, ControlNetModel
from diffusers.utils import load_image
from transformers import CLIPVisionModelWithProjection

from constants import BASE_MODEL, CONTROL_CANNY, CONTROL_DEPTH, IMAGE_CANNY, IMAGE_DEPTH, PROMPT, NEGATIVE


def portrait_trans(
        origin_image: str,
        control_image: str,
        prompt: str,
        control_type: str,
        style: str = 'vangogh',
        strength: float = 0.7,
        ip_adapter_scale: float = 0.7,
        prompt_2: str = None,
        negative: str = None,
):
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "h94/IP-Adapter",
        subfolder="models/image_encoder",
        torch_dtype=torch.float16,
    )
    controlnet = ControlNetModel.from_pretrained(
        CONTROL_CANNY if control_type == 'canny' else CONTROL_DEPTH,
        variant="fp16",
        use_safetensors=True,
        torch_dtype=torch.float16,
    ).to('cuda')
    pipeline = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
        BASE_MODEL,
        controlnet=controlnet,
        variant="fp16",
        use_safetensors=True,
        torch_dtype=torch.float16,
        image_encoder=image_encoder,
    ).to('cuda')
    pipeline.enable_model_cpu_offload()
    pipeline.load_ip_adapter(
        "h94/IP-Adapter",
        subfolder="sdxl_models",
        weight_name="ip-adapter-plus_sdxl_vit-h.safetensors"
    )
    style_images = [load_image(f"style/{style}/1-{i}.jpg") for i in range(8)]
    pipeline.set_ip_adapter_scale(ip_adapter_scale)
    images = pipeline(
        prompt=prompt,
        negative_prompt=negative,
        prompt_2=prompt_2,
        ip_adapter_image=[style_images],
        image=load_image(origin_image),
        control_image=load_image(control_image),
        strength=strength,
        num_inference_steps=30,
        num_images_per_prompt=4,
    ).images
    for image in images:
        stamp = int(time.time() * 1000)
        image.save(f'output/2_2/{control_type}/{strength}-{stamp}.png')


if __name__ == '__main__':
    for i in [0.4, 0.5, 0.6, 0.7]:
        # canny
        portrait_trans(
            IMAGE_CANNY,
            PROMPT,
            control_type='canny',
            negative=NEGATIVE,
            strength=i
        )
        # depth
        portrait_trans(
            IMAGE_DEPTH,
            PROMPT,
            control_type='depth',
            negative=NEGATIVE,
            strength=i
        )
