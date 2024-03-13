"""
真人风格化

实现方式 1.5（文生图，prompt + IP-Adapter-FaceID）：
"""
import time
from urllib.request import urlopen

import cv2
import numpy as np

import torch
from diffusers import StableDiffusionXLPipeline
from insightface.app import FaceAnalysis
from ip_adapter.ip_adapter_faceid import IPAdapterFaceIDXL
from transformers import CLIPVisionModelWithProjection

from constants import BASE_MODEL, PROMPT, NEGATIVE, IMAGE_ORIGIN


def portrait_trans(
        origin_image: str,
        prompt: str,
        ip_adapter_scale: float = 0.7,
        negative: str = None,
):
    # face
    app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    req = urlopen(origin_image)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    image = cv2.imdecode(arr, -1)  # 'Load it as it is'
    faces = app.get(image)
    faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
    # image encoder
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "h94/IP-Adapter",
        subfolder="models/image_encoder",
        torch_dtype=torch.float16,
    )
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        BASE_MODEL,
        variant="fp16",
        use_safetensors=True,
        torch_dtype=torch.float16,
        image_encoder=image_encoder,
    )

    ip_model = IPAdapterFaceIDXL(pipeline, 'ip-adapter-faceid_sdxl.bin', 'cuda')
    images = ip_model.generate(
        faceid_embeds=faceid_embeds,
        prompt=prompt,
        negative_prompt=negative,
        scale=ip_adapter_scale,
        num_samples=4,
        num_inference_steps=30,
    ).images
    for image in images:
        stamp = int(time.time() * 1000)
        image.save(f'output/1_5/{ip_adapter_scale}-{stamp}.png')


if __name__ == '__main__':
    for i in [0.5, 0.6, 0.7]:
        portrait_trans(
            IMAGE_ORIGIN,
            PROMPT,
            negative=NEGATIVE,
            ip_adapter_scale=i
        )
