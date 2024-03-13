"""
真人风格化

实现方式 2.1（原图 + prompt）：
"""
import time

import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image

from constants import BASE_MODEL, PROMPT, NEGATIVE, IMAGE_ORIGIN


def portrait_trans(
        origin_image: str,
        prompt: str,
        strength: float = 0.7,
        negative: str = None,
):
    pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        BASE_MODEL,
        variant="fp16",
        use_safetensors=True,
        torch_dtype=torch.float16,
    ).to('cuda')
    images = pipeline(
        prompt=prompt,
        negative_prompt=negative,
        image=load_image(origin_image),
        strength=strength,
        num_inference_steps=50,
        num_images_per_prompt=4,
    ).images
    for image in images:
        stamp = int(time.time() * 1000)
        image.save(f'output/2_1/{strength}-{stamp}.png')


if __name__ == '__main__':
    for i in [0.3, 0.4, 0.5, 0.5, 0.6, 0.7]:
        portrait_trans(
            IMAGE_ORIGIN,
            PROMPT,
            negative=NEGATIVE,
            strength=i
        )
