"""
宠物风格转换测试

实现方式：Img2Img + Style Prompt
"""
import time

import torch

from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image

from constants import BASE_MODEL


def trans(index: int, image: str, style: dict, prompt: str):
    folder = style.get('folder')

    pipeline: StableDiffusionXLImg2ImgPipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        'eienmojiki/Anything-XL',
        use_safetensors=True,
        torch_dtype=torch.float16,
    ).to('cuda')
    images = pipeline(
        prompt=style.get('prompt').replace('{prompt}', prompt),
        negative_prompt=style.get('negative'),
        image=load_image(f'{image}?imageView2/2/w/1024/h/1024'),
        num_inference_steps=30,
        num_images_per_prompt=4,
        guidance_scale=7.5,
        strength=style.get('strength'),
    ).images
    for image in images:
        stamp = int(time.time() * 1000)
        image.save(f'output/paw/{folder}/{index}-{stamp}.png')


if __name__ == '__main__':
    styles = [
        # {
        #     'name': 'Basquiat',
        #     'folder': 'basq',
        #     'prompt': 'Artwork by Jean-Michel Basquiat {prompt}. Neo-expressionism, street art influence, graffiti-inspired, raw, energetic, bold colors, dynamic composition, chaotic, layered, textural, expressive, spontaneous, distinctive, symbolic,energetic brushstrokes.',
        #     'negative': '(realistic:1.5), (photorealistic:1.5), calm, precise lines, conventional composition, subdued',
        #     'strength': 0.75
        # },
        {
            'name': 'Simple Vector Art',
            'folder': 'vector',
            'prompt': 'Simple Vector Art, {prompt}, 2D flat, simple shapes, minimalistic, professional graphic, flat color, high contrast, Simple Vector Art, 4k, masterpiece',
            'negative': 'ugly, deformed, noisy, blurry, low contrast, 3D, photo, realistic',
            'strength': 0.5,
        }
        # ,
        # {
        #     'name': 'popart',
        #     'folder': 'pop',
        #     'prompt': 'Simple Vector Art, {prompt}, 2D flat, simple shapes, minimalistic, professional graphic, flat color, high contrast, Simple Vector Art',
        #     'negative': 'ugly, deformed, noisy, blurry, low contrast, 3D, photo, realistic',
        #     'strength': 0.75,
        # },
    ]
    origin_images = [
        'https://static.interval.im/interval/XYcyfMZBGeXRnAPm.jpeg',
        'https://static.interval.im/interval/b2i4yZQmEzDtxSs8.jpeg',
        'https://static.interval.im/interval/D73yjc72hcBXQRza.jpeg',
        # 'https://static.interval.im/interval/p73SM4FTRKmh53cc.jpeg',
        # 'https://static.interval.im/interval/p73SM4FTRKmh53cc.jpeg',
        # 'https://static.interval.im/interval/esc2cberweJmkBzS.jpeg',
        # 'https://static.interval.im/interval/4Nsz4czmy8TWHA3b.jpeg',
        # 'https://static.interval.im/interval/2FsDfyeApJ5rsnYy.jpeg',
        # 'https://static.interval.im/interval/Pms4GWmRj5JkrnAd.jpeg',
        # 'https://static.interval.im/interval/2hRN7tC7aBkinWEC.jpeg',
        # 'https://static.interval.im/interval/pkhKm6zwkyh4hyyH.jpeg',
        # 'https://static.interval.im/interval/YG3hNFCKMDYrizJE.jpeg',
        # 'https://static.interval.im/interval/TjC4nRGJ7WtrEemp.jpeg',
        # 'https://static.interval.im/interval/QmGWeYiTr4xFFXzE.jpeg',
    ]
    for _style in styles:
        for _index, _image in enumerate(origin_images):
            trans(_index + 1, _image, _style, 'cat')
