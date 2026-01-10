import io
import os
from typing import Optional, List, Tuple

import torch
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import Response
from PIL import Image

from diffusers import (
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLPipeline,
)
from transformers import CLIPVisionModelWithProjection

# -------------------------------------------------
# Environment
# -------------------------------------------------
MODEL_ID = os.getenv("MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0")
HF_TOKEN = os.getenv("HF_TOKEN", None)

TORCH_DTYPE = torch.float16

app = FastAPI(title="SDXL API (txt2img + img2img + IP-Adapter + LoRA)", version="2.0")

img2img_pipe: Optional[StableDiffusionXLImg2ImgPipeline] = None
txt2img_pipe: Optional[StableDiffusionXLPipeline] = None

# -------------------------------------------------
# Utilities
# -------------------------------------------------
def read_image(data: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Invalid image")


def resize_to_8(img: Image.Image) -> Image.Image:
    w, h = img.size
    w = max(512, (w // 8) * 8)
    h = max(512, (h // 8) * 8)
    return img.resize((w, h), Image.LANCZOS)


def parse_loras(loras: str) -> List[Tuple[str, str, float]]:
    """
    Format:
    repo/file.safetensors:weight,repo2/file2.safetensors:weight
    """
    if not loras:
        return []

    out = []
    for part in loras.split(","):
        try:
            path, weight = part.rsplit(":", 1)
            repo, file = path.rsplit("/", 1)
            out.append((repo, file, float(weight)))
        except Exception:
            continue
    return out


# -------------------------------------------------
# Pipeline loaders
# -------------------------------------------------
def load_img2img() -> StableDiffusionXLImg2ImgPipeline:
    global img2img_pipe
    if img2img_pipe is not None:
        return img2img_pipe

    torch.backends.cuda.matmul.allow_tf32 = True

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "h94/IP-Adapter",
        subfolder="models/image_encoder",
        torch_dtype=TORCH_DTYPE,
        token=HF_TOKEN,
    ).to("cuda")

    img2img_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=TORCH_DTYPE,
        use_safetensors=True,
        token=HF_TOKEN,
        image_encoder=image_encoder,
    ).to("cuda")

    # disable safety (defensive)
    if hasattr(img2img_pipe, "safety_checker"):
        img2img_pipe.safety_checker = None
    if hasattr(img2img_pipe, "requires_safety_checker"):
        img2img_pipe.requires_safety_checker = False

    img2img_pipe.enable_vae_slicing()
    img2img_pipe.enable_vae_tiling()

    img2img_pipe.load_ip_adapter(
        "h94/IP-Adapter",
        subfolder="sdxl_models",
        weight_name="ip-adapter-plus-face_sdxl_vit-h.safetensors",
    )

    return img2img_pipe


def load_txt2img() -> StableDiffusionXLPipeline:
    global txt2img_pipe
    if txt2img_pipe is not None:
        return txt2img_pipe

    torch.backends.cuda.matmul.allow_tf32 = True

    txt2img_pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=TORCH_DTYPE,
        use_safetensors=True,
        token=HF_TOKEN,
    ).to("cuda")

    if hasattr(txt2img_pipe, "safety_checker"):
        txt2img_pipe.safety_checker = None
    if hasattr(txt2img_pipe, "requires_safety_checker"):
        txt2img_pipe.requires_safety_checker = False

    txt2img_pipe.enable_vae_slicing()
    txt2img_pipe.enable_vae_tiling()

    return txt2img_pipe


# -------------------------------------------------
# API
# -------------------------------------------------
@app.get("/health")
def health():
    return {"ok": True}


# -------------------- TXT2IMG --------------------
@app.post("/generate")
async def generate(
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    width: int = Form(1024),
    height: int = Form(1024),
    steps: int = Form(30),
    cfg: float = Form(6.0),
    seed: int = Form(-1),
    loras: str = Form(""),
):
    if not prompt.strip():
        raise HTTPException(400, "prompt required")

    p = load_txt2img()

    if hasattr(p, "unload_lora_weights"):
        p.unload_lora_weights()

    lora_list = parse_loras(loras)
    if lora_list:
        names, weights = [], []
        for repo, file, w in lora_list:
            p.load_lora_weights(repo, weight_name=file, adapter_name=file)
            names.append(file)
            weights.append(w)
        if hasattr(p, "set_adapters"):
            p.set_adapters(names, adapter_weights=weights)

    generator = None
    if seed >= 0:
        generator = torch.Generator("cuda").manual_seed(seed)

    with torch.inference_mode():
        img = p(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=cfg,
            generator=generator,
        ).images[0]

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png")


# -------------------- IMG2IMG --------------------
@app.post("/edit")
async def edit(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    strength: float = Form(0.45),
    steps: int = Form(32),
    cfg: float = Form(6.0),
    seed: int = Form(-1),
    ip_scale: float = Form(0.7),
    loras: str = Form(""),
):
    img = resize_to_8(read_image(await image.read()))

    p = load_img2img()
    p.set_ip_adapter_scale(ip_scale)

    if hasattr(p, "unload_lora_weights"):
        p.unload_lora_weights()

    lora_list = parse_loras(loras)
    if lora_list:
        names, weights = [], []
        for repo, file, w in lora_list:
            p.load_lora_weights(repo, weight_name=file, adapter_name=file)
            names.append(file)
            weights.append(w)
        if hasattr(p, "set_adapters"):
            p.set_adapters(names, adapter_weights=weights)

    generator = None
    if seed >= 0:
        generator = torch.Generator("cuda").manual_seed(seed)

    with torch.inference_mode():
        out = p(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            image=img,
            ip_adapter_image=img,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=cfg,
            generator=generator,
        ).images[0]

    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png")
