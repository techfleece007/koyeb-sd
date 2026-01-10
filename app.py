import io
import os
import re
from typing import Optional, List, Tuple

import torch
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import Response
from PIL import Image

from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
)
from transformers import CLIPVisionModelWithProjection

# -------------------------------------------------
# Environment
# -------------------------------------------------
MODEL_ID = os.getenv("MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0")
HF_TOKEN = os.getenv("HF_TOKEN")

DTYPE = torch.float16
DEVICE = "cuda"

app = FastAPI(
    title="SDXL API (txt2img + img2img + IP-Adapter + LoRA)",
    version="2.3",
)

txt2img_pipe: Optional[StableDiffusionXLPipeline] = None
img2img_pipe: Optional[StableDiffusionXLImg2ImgPipeline] = None


# -------------------------------------------------
# Utils
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


def safe_adapter_name(name: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9_-]+", "_", name)
    return name.replace(".", "_") or "lora"


def parse_loras(loras: str) -> List[Tuple[str, str, float]]:
    """
    Format:
    repo/file.safetensors:0.7,repo2/file2.safetensors:0.4
    """
    if not loras:
        return []

    out = []
    for part in loras.split(","):
        path, w = part.rsplit(":", 1)
        repo, file = path.rsplit("/", 1)
        out.append((repo, file, float(w)))
    return out


def unload_loras_safe(pipe):
    if hasattr(pipe, "unload_lora_weights"):
        try:
            pipe.unload_lora_weights()
        except Exception:
            pass


def apply_loras(pipe, loras: str):
    unload_loras_safe(pipe)

    parsed = parse_loras(loras)
    if not parsed:
        return

    names, weights = [], []
    for repo, file, w in parsed:
        adapter = safe_adapter_name(file)
        pipe.load_lora_weights(repo, weight_name=file, adapter_name=adapter)
        names.append(adapter)
        weights.append(w)

    if hasattr(pipe, "set_adapters"):
        pipe.set_adapters(names, adapter_weights=weights)


# -------------------------------------------------
# Pipelines
# -------------------------------------------------
def load_txt2img() -> StableDiffusionXLPipeline:
    global txt2img_pipe
    if txt2img_pipe:
        return txt2img_pipe

    torch.backends.cuda.matmul.allow_tf32 = True

    txt2img_pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        token=HF_TOKEN,
        use_safetensors=True,
    ).to(DEVICE)

    txt2img_pipe.enable_vae_slicing()
    txt2img_pipe.enable_vae_tiling()

    if hasattr(txt2img_pipe, "safety_checker"):
        txt2img_pipe.safety_checker = None
    if hasattr(txt2img_pipe, "requires_safety_checker"):
        txt2img_pipe.requires_safety_checker = False

    return txt2img_pipe


def load_img2img() -> StableDiffusionXLImg2ImgPipeline:
    global img2img_pipe
    if img2img_pipe:
        return img2img_pipe

    torch.backends.cuda.matmul.allow_tf32 = True

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "h94/IP-Adapter",
        subfolder="models/image_encoder",
        torch_dtype=DTYPE,
        token=HF_TOKEN,
    ).to(DEVICE)

    img2img_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        token=HF_TOKEN,
        use_safetensors=True,
        image_encoder=image_encoder,
    ).to(DEVICE)

    img2img_pipe.enable_vae_slicing()
    img2img_pipe.enable_vae_tiling()

    if hasattr(img2img_pipe, "safety_checker"):
        img2img_pipe.safety_checker = None
    if hasattr(img2img_pipe, "requires_safety_checker"):
        img2img_pipe.requires_safety_checker = False

    img2img_pipe.load_ip_adapter(
        "h94/IP-Adapter",
        subfolder="sdxl_models",
        weight_name="ip-adapter-plus-face_sdxl_vit-h.safetensors",
    )

    return img2img_pipe


# -------------------------------------------------
# API
# -------------------------------------------------
@app.get("/health")
def health():
    return {"ok": True}


@app.post("/generate")
async def generate(
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    width: int = Form(1024),
    height: int = Form(1024),
    steps: int = Form(35),
    cfg: float = Form(7.0),
    seed: int = Form(-1),
    loras: str = Form(""),
):
    pipe = load_txt2img()
    apply_loras(pipe, loras)

    generator = None
    if seed >= 0:
        generator = torch.Generator(DEVICE).manual_seed(seed)

    with torch.inference_mode():
        img = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=cfg,
            generator=generator,
        ).images[0]

    buf = io.BytesIO()
    img.save(buf, "PNG")
    return Response(buf.getvalue(), media_type="image/png")


@app.post("/edit")
async def edit(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    strength: float = Form(0.55),
    steps: int = Form(35),
    cfg: float = Form(7.0),
    seed: int = Form(-1),
    ip_scale: float = Form(0.6),
    loras: str = Form(""),
):
    img = resize_to_8(read_image(await image.read()))

    pipe = load_img2img()
    pipe.set_ip_adapter_scale(float(ip_scale))
    apply_loras(pipe, loras)

    generator = None
    if seed >= 0:
        generator = torch.Generator(DEVICE).manual_seed(seed)

    with torch.inference_mode():
        out = pipe(
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
    out.save(buf, "PNG")
    return Response(buf.getvalue(), media_type="image/png")
