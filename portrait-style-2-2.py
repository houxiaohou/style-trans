"""
真人风格化

实现方式 2.2（原图 + ControlNet + 风格 Prompt）：
"""
import time

import torch
from diffusers import StableDiffusionXLControlNetImg2ImgPipeline, ControlNetModel
from diffusers.utils import load_image

from constants import BASE_MODEL, CONTROL_CANNY, CONTROL_DEPTH, IMAGE_ORIGIN, IMAGE_CANNY, IMAGE_DEPTH, PROMPT, \
    NEGATIVE, PROMPT_2


def portrait_trans(
        origin_image: str,
        control_image: str,
        prompt: str,
        control_type: str,
        control_scale: float = 0.7,
        strength: float = 0.6,
        prompt_2: str = None,
        negative: str = None,
):
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
    ).to('cuda')
    pipeline.enable_model_cpu_offload()
    images = pipeline(
        prompt=prompt,
        negative_prompt=negative,
        prompt_2=prompt_2,
        # 原图
        image=load_image(origin_image),
        # 控制图
        control_image=load_image(control_image),
        controlnet_conditioning_scale=control_scale,
        # 原图强度
        strength=strength,
        num_inference_steps=50,
        num_images_per_prompt=4,
    ).images
    for image in images:
        stamp = int(time.time() * 1000)
        image.save(f'output/2_2/{control_type}/{strength}-{control_scale}-{stamp}.png')


if __name__ == '__main__':
    strength_numbers = [0.3, 0.35, 0.4, 0.5]
    scale_numbers = [0.4, 0.5, 0.6]
    for _strength in strength_numbers:
        for _scale in scale_numbers:
            # canny
            portrait_trans(
                IMAGE_ORIGIN,
                IMAGE_CANNY,
                PROMPT,
                control_type='canny',
                negative=NEGATIVE,
                strength=_strength,
                control_scale=_scale,
            )
            # depth
            portrait_trans(
                IMAGE_ORIGIN,
                IMAGE_DEPTH,
                PROMPT,
                control_type='depth',
                negative=NEGATIVE,
                strength=_strength,
                control_scale=_scale,
            )
