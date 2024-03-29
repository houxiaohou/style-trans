"""
真人风格化

实现方式 1.3（文生图，ControlNet + IP-Adapter风格图 + IP-Adapter-FaceID）：
"""
import time

import torch
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
from diffusers.utils import load_image
from transformers import CLIPVisionModelWithProjection

from constants import BASE_MODEL, CONTROL_CANNY, CONTROL_DEPTH, IMAGE_CANNY, IMAGE_DEPTH, PROMPT, PROMPT_2, NEGATIVE, \
    IMAGE_ORIGIN, CONTROL_SCALE


def portrait_trans(
        origin_image: str,
        control_image: str,
        prompt: str,
        control_type: str,
        style: str = 'vangogh',
        ip_adapter_scale: float = 0.7,
        face_scale: float = 0.4,
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
    pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
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
        weight_name=["ip-adapter-plus_sdxl_vit-h.safetensors", "ip-adapter-plus-face_sdxl_vit-h.safetensors"]
    )
    style_images = [load_image(f"style/{style}/1-{i}.jpg") for i in [4]]
    pipeline.set_ip_adapter_scale([ip_adapter_scale, face_scale])
    images = pipeline(
        prompt=prompt,
        prompt_2=prompt_2,
        negative_prompt=negative,
        ip_adapter_image=[style_images, load_image(origin_image)],
        image=load_image(control_image),
        num_inference_steps=30,
        num_images_per_prompt=4,
    ).images
    for image in images:
        stamp = int(time.time() * 1000)
        image.save(f'output/1_3/{control_type}/face-{face_scale}-{stamp}.png')


if __name__ == '__main__':
    for i in [0.3, 0.4, 0.5, 0.6]:
        # depth
        portrait_trans(
            IMAGE_ORIGIN,
            IMAGE_DEPTH,
            PROMPT,
            control_type='depth',
            negative=NEGATIVE,
            ip_adapter_scale=0.7,
            face_scale=i,
        )
