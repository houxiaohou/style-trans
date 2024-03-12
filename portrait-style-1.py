"""
真人风格化

实现方式一（文生图，ControlNet + 风格 Prompt）：
"""
import time

import torch
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
from diffusers.utils import load_image

from constants import BASE_MODEL, CONTROL_CANNY, CONTROL_DEPTH, IMAGE_CANNY, IMAGE_DEPTH, PROMPT, NEGATIVE

def portrait_trans(
        control_image: str,
        prompt: str,
        control_type: str,
        control_scale: float = 0.5,
        prompt_2: str = None,
        negative: str = None,
):
    controlnet = ControlNetModel.from_pretrained(
        CONTROL_CANNY if control_type == 'canny' else CONTROL_DEPTH,
        variant="fp16",
        use_safetensors=True,
        torch_dtype=torch.float16,
    ).to('cuda')
    pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
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
        image=load_image(control_image),
        controlnet_conditioning_scale=control_scale,
        num_inference_steps=30,
        num_images_per_prompt=4,
    ).images
    for image in images:
        stamp = int(time.time() * 1000)
        image.save(f'output/1_1/{control_type}/{control_scale}-{stamp}.png')


if __name__ == '__main__':
    # canny
    for i in [0.4, 0.5, 0.6, 0.7]:
        portrait_trans(
            IMAGE_CANNY,
            PROMPT,
            control_type='canny',
            negative=NEGATIVE,
            control_scale=i
        )
        # depth
        portrait_trans(
            IMAGE_DEPTH,
            PROMPT,
            control_type='depth',
            negative=NEGATIVE,
            control_scale=i
        )
