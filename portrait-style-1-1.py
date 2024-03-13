"""
真人风格化

实现方式 1.1（文生图，ControlNet + 风格 Prompt）：
"""
import time

import torch
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoPipelineForText2Image, T2IAdapter
from diffusers.utils import load_image

from constants import BASE_MODEL, CONTROL_CANNY, CONTROL_DEPTH, IMAGE_LINEART, PROMPT, NEGATIVE


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


def portrait_trans_lineart(
        control_image: str,
        prompt: str,
        negative: str,
        control_scale: float = 0.5
):
    """lineart"""
    # load adapter
    adapter = T2IAdapter.from_pretrained(
        "TencentARC/t2i-adapter-lineart-sdxl-1.0",
        torch_dtype=torch.float16,
        varient="fp16",
        use_safetensors=True,
    ).to("cuda")
    pipeline = AutoPipelineForText2Image.from_pretrained(
        BASE_MODEL,
        adapter=adapter,
        variant="fp16",
        use_safetensors=True,
        torch_dtype=torch.float16,
    ).to('cuda')
    images = pipeline(
        prompt=prompt,
        negative_prompt=negative,
        image=load_image(control_image),
        adapter_conditioning_scale=control_scale,
        num_inference_steps=30,
        num_images_per_prompt=4,
    ).images
    for image in images:
        stamp = int(time.time() * 1000)
        image.save(f'output/1_1/lineart/{control_scale}-{stamp}.png')


if __name__ == '__main__':
    for i in [0.4, 0.5, 0.6, 0.7]:
        # lineart
        portrait_trans_lineart(
            IMAGE_LINEART,
            PROMPT,
            negative=NEGATIVE,
            control_scale=i
        )
