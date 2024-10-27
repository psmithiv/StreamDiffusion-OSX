import sys
import os

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
    )
)

from utils.wrapper import StreamDiffusionWrapper

import torch

from config import Args
from pydantic import BaseModel, Field
from PIL import Image
import math

base_model = "stabilityai/sd-turbo"
img_size = 512
# base_model = "stabilityai/stable-diffusion-2-base"
# base_model = "stabilityai/sdxl-turbo" # exception
# base_model = "justinpinkney/miniSD" # Doesn't load 
# base_model = "lambdalabs/sd-image-variations-diffusers" #to delete
# base_model = "lambdalabs/miniSD-diffusers" # Ugly with 50 iterations
# base_model = "CompVis/stable-diffusion-v-1-4-original" # not loaded
# base_model = "runwayml/stable-diffusion-v1-5" # Only 512, slow but cool.Ugly with 50 iterations
# img_size = 256

default_prompt = "Portrait of The Joker halloween costume, face painting, with , glare pose, detailed, intricate, full of colour, cinematic lighting, trending on artstation, 8k, hyperrealistic, focused, extreme details, unreal engine 5 cinematic, masterpiece"
default_negative_prompt = "black and white, blurry, low resolution, pixelated,  pixel art, low quality, low fidelity"

page_content = """<h1 class="text-3xl font-bold">StreamDiffusion</h1>
<h3 class="text-xl font-bold">Image-to-Image SD-Turbo</h3>
<p class="text-sm">
    This demo showcases
    <a
    href="https://github.com/cumulo-autumn/StreamDiffusion"
    target="_blank"
    class="text-blue-500 underline hover:no-underline">StreamDiffusion
</a>
Image to Image pipeline using
    <a
    href="https://huggingface.co/stabilityai/sd-turbo"
    target="_blank"
    class="text-blue-500 underline hover:no-underline">SD-Turbo</a
    > with a MJPEG stream server.
</p>
"""


class Pipeline:
    class Info(BaseModel):
        name: str = "StreamDiffusion img2img"
        input_mode: str = "image"
        page_content: str = page_content

    class InputParams(BaseModel):
        prompt: str = Field(
            default_prompt,
            title="Prompt",
            field="textarea",
            id="prompt",
        )
        negative_prompt: str = Field(
            default_negative_prompt,
            title="Negative Prompt",
            field="textarea",
            id="negative_prompt",
        )
        width: int = Field(
            img_size, min=2, max=15, title="Width", disabled=True, hide=True, id="width"
        )
        height: int = Field(
            img_size, min=2, max=15, title="Height", disabled=True, hide=True, id="height"
        )

    def __init__(self, args: Args, device: torch.device, torch_dtype: torch.dtype, t_index_list: list = [1]):
        params = self.InputParams()
    
        self.stream = StreamDiffusionWrapper(
            model_id_or_path=base_model,
            use_tiny_vae=args.taesd,
            device=device,
            dtype=torch_dtype,
            t_index_list=t_index_list,
            frame_buffer_size=1,
            width=params.width,
            height=params.height,
            use_lcm_lora=False,
            output_type="pil",
            warmup=10,
            vae_id=None,
            acceleration=args.acceleration,
            mode="img2img",
            use_denoising_batch=True,
            cfg_type="none",
            use_safety_checker=args.safety_checker,
            # enable_similar_image_filter=True,
            # similar_image_filter_threshold=0.98,
            engine_dir=args.engine_dir,
        )

        self.last_prompt = default_prompt
        self.stream.prepare(
            prompt=default_prompt,
            # negative_prompt=default_negative_prompt,
            num_inference_steps=2,
            guidance_scale=0.0,
            t_index_list=t_index_list
        )

    def predict(self, params: "Pipeline.InputParams") -> Image.Image:
        image_tensor = self.stream.preprocess_image(params.image)
        # image_tensor -= image_tensor.min()
        # image_tensor /= image_tensor.max()
        output_image = self.stream(image=image_tensor, prompt=params.prompt)

        return output_image
