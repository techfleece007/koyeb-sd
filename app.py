import io
import os
from typing import Optional

import torch
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import Response
from PIL import Image

from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image

# ----------------------------
# Environment
# ----------------------------
MODEL_ID = os.getenv(
    "MODEL_ID",
    "stabilityai/stable-diffusion-xl-base-1.0"
)
HF_TOKEN = os.getenv("HF_TOKEN", None)
TORCH_DTYPE = torch.float16

app = FastAPI()

pipe: Optional[StableDiffusionXLImg2ImgPipeline] = None


def load_pipeline():
    global pipe
    if pipe is not None:
        return pipe

    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=TORCH_DTYPE,
        use_safetensors=True,
        token=HF_TOKEN,
        safety_checker=None,
    ).to("cuda")

    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()

    # ----------------------------
    # IP-ADAPTER (FACE)
    # ----------------------------
    pipe.load_ip_adapter(
        "h94/IP-Adapter",
        subfolder="sdxl_models",
        weight_name="ip-adapter-plus-face_sdxl.safetensors",
    )

    return pipe


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/edit")
async def edit(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    strength: float = Form(0.4),
    steps: int = Form(30),
    cfg: float = Form(6.0),
    seed: int = Form(-1),
    ip_scale: float = Form(0.9),  # 🔑 identity lock strength
):
    if strength <= 0 or strength > 1:
        raise HTTPException(400, "strength must be (0,1]")
    if ip_scale < 0 or ip_scale > 1.2:
        raise HTTPException(400, "ip_scale must be 0–1.2")

    image_bytes = await image.read()
    init_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    pipe = load_pipeline()
    pipe.set_ip_adapter_scale(ip_scale)

    generator = None
    if seed >= 0:
        generator = torch.Generator("cuda").manual_seed(seed)

    with torch.inference_mode():
        out = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            image=init_image,
            ip_adapter_image=init_image,  # SAME image = identity lock
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=cfg,
            generator=generator,
        ).images[0]

    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png")
