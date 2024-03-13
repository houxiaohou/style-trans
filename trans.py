import torch
from diffusers import StableDiffusionXLPipeline


def trans():
    print('start')
    model = StableDiffusionXLPipeline.from_single_file(
        '/root/style-trans/blue_pencil-XL-v5.0.0.safetensors',
        torch_dtype=torch.float16,
        varient="fp16",
    )
    print('load finish')
    model.save_pretrained('/root/style-trans/blue-pencil')


if __name__ == '__main__':
    trans()
