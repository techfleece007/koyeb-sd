import io
import os
from typing import Optional

import torch
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import Response
from PIL import Image

from diffusers import StableDiffusionXLImg2ImgPipeline

# ----------------------------
# Environment
# ----------------------------
MODEL_ID = os.getenv("MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0")
HF_TOKEN = os.getenv("HF_TOKEN", None)

# SDXL runs great on fp16 on A100/A6000/L40S
TORCH_DTYPE = torch.float16

app = FastAPI(title="SDXL Img2Img API (IP-Adapter)", version="1.1")

pipe: Optional[StableDiffusionXLImg2ImgPipeline] = None


def _load_pipeline() -> StableDiffusionXLImg2ImgPipeline:
    """
    Loads SDXL Img2Img pipeline once, enables IP-Adapter (face),
    and applies defensive safety disable if present.
    """
    global pipe
    if pipe is not None:
        return pipe

    # Optional: speed/throughput improvement on Ampere+
    torch.backends.cuda.matmul.allow_tf32 = True

    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=TORCH_DTYPE,
        use_safetensors=True,
        token=HF_TOKEN,
    ).to("cuda")

    # ------------------------------------
    # Defensive safety disable (future-proof)
    # ------------------------------------
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None
    if hasattr(pipe, "requires_safety_checker"):
        pipe.requires_safety_checker = False

    # Performance helpers
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()

    # ----------------------------
    # IP-ADAPTER (FACE) for identity lock
    # ----------------------------
    pipe.load_ip_adapter(
        "h94/IP-Adapter",
        subfolder="sdxl_models",
        weight_name="ip-adapter-plus-face_sdxl.safetensors",
    )

    return pipe


def _read_image(file_bytes: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")


def _resize_to_multiple_of_8(img: Image.Image, min_side: int = 768, max_side: int = 1536) -> Image.Image:
    """
    Keep aspect ratio, clamp to [min_side, max_side] on the long edge,
    then force multiples of 8 (diffusion-friendly).
    """
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


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/edit")
async def edit(
    image: UploadFile = File(...),
    prompt: str = Form(...),

    # Optional knobs with safe defaults
    negative_prompt: str = Form(""),
    strength: float = Form(0.40),      # 0.25–0.50 typical for prompt-only edits
    steps: int = Form(32),             # 25–40 typical
    cfg: float = Form(6.0),            # 5–7 typical
    seed: int = Form(-1),

    # IP-Adapter identity lock strength
    ip_scale: float = Form(0.90),      # 0.6–1.1 typical
):
    # Validate inputs
    if not prompt or not prompt.strip():
        raise HTTPException(status_code=400, detail="prompt is required.")
    if strength <= 0 or strength > 1:
        raise HTTPException(status_code=400, detail="strength must be in (0, 1].")
    if steps < 10 or steps > 80:
        raise HTTPException(status_code=400, detail="steps must be between 10 and 80.")
    if cfg < 1 or cfg > 15:
        raise HTTPException(status_code=400, detail="cfg must be between 1 and 15.")
    if ip_scale < 0 or ip_scale > 1.2:
        raise HTTPException(status_code=400, detail="ip_scale must be between 0 and 1.2.")

    # Read input image
    img_bytes = await image.read()
    init_image = _read_image(img_bytes)
    init_image = _resize_to_multiple_of_8(init_image)

    # Load pipeline + set IP scale per request
    p = _load_pipeline()
    p.set_ip_adapter_scale(ip_scale)

    # Seed control (optional)
    generator = None
    if seed is not None and seed >= 0:
        generator = torch.Generator(device="cuda").manual_seed(seed)

    # Inference
    with torch.inference_mode():
        result = p(
            prompt=prompt,
            negative_prompt=negative_prompt.strip() or None,
            image=init_image,
            ip_adapter_image=init_image,          # identity reference
            strength=float(strength),
            num_inference_steps=int(steps),
            guidance_scale=float(cfg),
            generator=generator,
        ).images[0]

    # Return PNG
    buf = io.BytesIO()
    result.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")
