import io
import os
from typing import Optional

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image
from diffusers import StableDiffusionXLImg2ImgPipeline

# ----------------------------
# Configuration (via env vars)
# ----------------------------
MODEL_ID = os.getenv("MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0")
TORCH_DTYPE = torch.float16  # A100 supports fp16 well

# Optional: allow a HF token if the model is gated/private
HF_TOKEN = os.getenv("HF_TOKEN", None)

# Keep a single global pipeline instance (lazy-loaded)
pipe: Optional[StableDiffusionXLImg2ImgPipeline] = None

app = FastAPI(title="SDXL Img2Img API", version="1.0")


def load_pipe() -> StableDiffusionXLImg2ImgPipeline:
    """
    Loads the SDXL img2img pipeline once and reuses it across requests.
    Safety checker is disabled (self-hosted control).
    """
    global pipe
    if pipe is not None:
        return pipe

    # Improve performance / memory on GPU
    torch.backends.cuda.matmul.allow_tf32 = True

    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=TORCH_DTYPE,
        use_safetensors=True,
        token=HF_TOKEN,
        safety_checker=None,  # disable safety filtering in the pipeline
    )

    # Move to GPU
    pipe = pipe.to("cuda")

    # Performance helpers
    pipe.enable_vae_tiling()
    pipe.enable_vae_slicing()

    return pipe


def read_image(file_bytes: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/edit")
async def edit_image(
    image: UploadFile = File(...),
    prompt: str = Form(...),

    # Optional knobs (safe defaults)
    negative_prompt: str = Form(""),
    strength: float = Form(0.35),   # low strength = preserve identity more
    steps: int = Form(30),
    cfg: float = Form(6.0),
    seed: int = Form(-1),
):
    if strength <= 0 or strength > 1:
        raise HTTPException(status_code=400, detail="strength must be in (0, 1].")
    if steps < 10 or steps > 80:
        raise HTTPException(status_code=400, detail="steps must be between 10 and 80.")
    if cfg < 1 or cfg > 15:
        raise HTTPException(status_code=400, detail="cfg must be between 1 and 15.")

    img_bytes = await image.read()
    init_image = read_image(img_bytes)

    p = load_pipe()

    generator = None
    if seed is not None and seed >= 0:
        generator = torch.Generator(device="cuda").manual_seed(seed)

    # SDXL likes reasonable sizes; keep original size but clamp to multiples of 8
    w, h = init_image.size
    w = max(512, (w // 8) * 8)
    h = max(512, (h // 8) * 8)
    init_image = init_image.resize((w, h))

    # Run inference
    with torch.inference_mode():
        out = p(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt.strip() else None,
            image=init_image,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=cfg,
            generator=generator,
        ).images[0]

    # Return PNG
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")
