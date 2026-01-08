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
LORA_DIR = os.getenv("LORA_DIR", "/app/loras")

TORCH_DTYPE = torch.float16

app = FastAPI(title="SDXL Img2Img API (IP-Adapter + LoRA)", version="1.3")

pipe: Optional[StableDiffusionXLImg2ImgPipeline] = None


# ----------------------------
# Utilities
# ----------------------------
def _read_image(file_bytes: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")


def _resize_to_multiple_of_8(
    img: Image.Image, min_side: int = 768, max_side: int = 1536
) -> Image.Image:
    w, h = img.size
    long_edge = max(w, h)

    scale = 1.0
    if long_edge < min_side:
        scale = min_side / float(long_edge)
    elif long_edge > max_side:
        scale = max_side / float(long_edge)

    if abs(scale - 1.0) > 1e-6:
        w = int(round(w * scale))
        h = int(round(h * scale))
        img = img.resize((w, h), Image.LANCZOS)

    w, h = img.size
    w = max(512, (w // 8) * 8)
    h = max(512, (h // 8) * 8)
    if (w, h) != img.size:
        img = img.resize((w, h), Image.LANCZOS)

    return img


def parse_loras(loras_str: str) -> List[Tuple[str, float]]:
    """
    Parse: "name:0.5,name2:0.4"
    """
    if not loras_str:
        return []

    out = []
    for part in loras_str.split(","):
        part = part.strip()
        if not part:
            continue

        if ":" in part:
            name, weight = part.split(":", 1)
            out.append((name.strip(), float(weight)))
        else:
            out.append((part.strip(), 0.5))
    return out


# ----------------------------
# Pipeline loader
# ----------------------------
def _load_pipeline() -> StableDiffusionXLImg2ImgPipeline:
    global pipe
    if pipe is not None:
        return pipe

    torch.backends.cuda.matmul.allow_tf32 = True

    # ✅ Correct image encoder for IP-Adapter Plus / Face (SDXL)
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

    # Defensive safety disable
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None
    if hasattr(pipe, "requires_safety_checker"):
        pipe.requires_safety_checker = False

    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()

    # IP-Adapter Plus Face (ViT-H)
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
    strength: float = Form(0.40),
    steps: int = Form(32),
    cfg: float = Form(6.0),
    seed: int = Form(-1),

    ip_scale: float = Form(0.90),

    # 🔥 LoRAs: "boltedonlipsXL-v1:0.45,LargeBreasts_By_Stable_Yogi_SD_V1:0.35"
    loras: str = Form(""),
):
    if not prompt or not prompt.strip():
        raise HTTPException(400, "prompt is required")
    if strength <= 0 or strength > 1:
        raise HTTPException(400, "strength must be in (0, 1]")
    if steps < 10 or steps > 80:
        raise HTTPException(400, "steps must be 10–80")
    if cfg < 1 or cfg > 15:
        raise HTTPException(400, "cfg must be 1–15")
    if ip_scale < 0 or ip_scale > 1.2:
        raise HTTPException(400, "ip_scale must be 0–1.2")

    img_bytes = await image.read()
    init_image = _read_image(img_bytes)
    init_image = _resize_to_multiple_of_8(init_image)

    p = _load_pipeline()
    p.set_ip_adapter_scale(ip_scale)

    # 🔄 Clear previously loaded LoRAs
    if hasattr(p, "unload_lora_weights"):
        p.unload_lora_weights()

    # 🔗 Load requested LoRAs
    lora_list = parse_loras(loras)
    if lora_list:
        adapter_names = []
        adapter_weights = []

        for name, weight in lora_list:
            path = os.path.join(LORA_DIR, f"{name}.safetensors")
            if not os.path.exists(path):
                raise HTTPException(400, f"LoRA not found: {name}")

            p.load_lora_weights(path, adapter_name=name)
            adapter_names.append(name)
            adapter_weights.append(weight)

        if hasattr(p, "set_adapters"):
            p.set_adapters(adapter_names, adapter_weights)

    generator = None
    if seed is not None and seed >= 0:
        generator = torch.Generator("cuda").manual_seed(seed)

    with torch.inference_mode():
        result = p(
            prompt=prompt,
            negative_prompt=negative_prompt.strip() or None,
            image=init_image,
            ip_adapter_image=init_image,
            strength=float(strength),
            num_inference_steps=int(steps),
            guidance_scale=float(cfg),
            generator=generator,
        ).images[0]

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png")
