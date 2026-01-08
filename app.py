import io
import os
from typing import Optional, List, Tuple

import torch
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import Response
from PIL import Image

from diffusers import StableDiffusionXLImg2ImgPipeline
from transformers import CLIPVisionModelWithProjection

# ----------------------------
# Environment
# ----------------------------
MODEL_ID = os.getenv("MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0")
HF_TOKEN = os.getenv("HF_TOKEN", None)

TORCH_DTYPE = torch.float16

app = FastAPI(title="SDXL Img2Img API (IP-Adapter + HF LoRA)", version="1.4")

pipe: Optional[StableDiffusionXLImg2ImgPipeline] = None


# ----------------------------
# Utilities
# ----------------------------
def _read_image(file_bytes: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}")


def _resize_to_multiple_of_8(img: Image.Image) -> Image.Image:
    w, h = img.size
    w = max(512, (w // 8) * 8)
    h = max(512, (h // 8) * 8)
    return img.resize((w, h), Image.LANCZOS)


def parse_loras(loras_str: str) -> List[Tuple[str, str, float]]:
    """
    Format:
    "repo/file:0.5,repo2/file2:0.4"

    Example:
    "sheko007/koyeb-sd-lora/boltedonlipsXL-v1.safetensors:0.45"
    """
    if not loras_str:
        return []

    out = []
    for part in loras_str.split(","):
        name_weight = part.strip().rsplit(":", 1)
        if len(name_weight) != 2:
            continue
        path, weight = name_weight
        repo, file = path.rsplit("/", 1)
        out.append((repo, file, float(weight)))
    return out


# ----------------------------
# Pipeline loader
# ----------------------------
def _load_pipeline() -> StableDiffusionXLImg2ImgPipeline:
    global pipe
    if pipe is not None:
        return pipe

    torch.backends.cuda.matmul.allow_tf32 = True

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "h94/IP-Adapter",
        subfolder="models/image_encoder",
        torch_dtype=TORCH_DTYPE,
        token=HF_TOKEN,
    ).to("cuda")

    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=TORCH_DTYPE,
        use_safetensors=True,
        token=HF_TOKEN,
        image_encoder=image_encoder,
    ).to("cuda")

    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None
    if hasattr(pipe, "requires_safety_checker"):
        pipe.requires_safety_checker = False

    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()

    pipe.load_ip_adapter(
        "h94/IP-Adapter",
        subfolder="sdxl_models",
        weight_name="ip-adapter-plus-face_sdxl_vit-h.safetensors",
    )

    return pipe


# ----------------------------
# API
# ----------------------------
@app.get("/health")
def health():
    return {"ok": True}


@app.post("/edit")
async def edit(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    strength: float = Form(0.4),
    steps: int = Form(32),
    cfg: float = Form(6.0),
    seed: int = Form(-1),
    ip_scale: float = Form(0.9),

    # 🔥 HuggingFace LoRAs
    # Example:
    # sheko007/koyeb-sd-lora/boltedonlipsXL-v1.safetensors:0.45
    loras: str = Form(""),
):
    img_bytes = await image.read()
    init_image = _resize_to_multiple_of_8(_read_image(img_bytes))

    p = _load_pipeline()
    p.set_ip_adapter_scale(ip_scale)

    if hasattr(p, "unload_lora_weights"):
        p.unload_lora_weights()

    lora_list = parse_loras(loras)
    if lora_list:
        names, weights = [], []
        for repo, file, weight in lora_list:
            p.load_lora_weights(repo, weight_name=file, adapter_name=file)
            names.append(file)
            weights.append(weight)

        if hasattr(p, "set_adapters"):
            p.set_adapters(names, adapter_weights=weights)

    generator = None
    if seed >= 0:
        generator = torch.Generator("cuda").manual_seed(seed)

    with torch.inference_mode():
        out = p(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            image=init_image,
            ip_adapter_image=init_image,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=cfg,
            generator=generator,
        ).images[0]

    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png")
