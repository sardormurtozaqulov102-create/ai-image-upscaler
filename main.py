"""
AI Photo Upscaler backend (Real-ESRGAN)

This is a hardened version of the original main.py. Key changes vs. the
original are marked with "# FIX:" comments so you can see exactly what
changed and why.
"""

import ssl

ssl._create_default_https_context = ssl._create_unverified_context
import io
import os
import sys
import threading
import time
from pathlib import Path

try:
    import torchvision.transforms.functional_tensor  # noqa: F401
except ModuleNotFoundError:
    import torchvision.transforms.functional as _functional
    sys.modules["torchvision.transforms.functional_tensor"] = _functional

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image

from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer

BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = BASE_DIR / "weights"
WEIGHTS_DIR.mkdir(exist_ok=True)

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", 20 * 1024 * 1024))  # 20 MB
MAX_INPUT_PIXELS = int(os.getenv("MAX_INPUT_PIXELS", 20_000_000))  # ~20MP, e.g. 5000x4000

app = FastAPI(title="AI Photo Upscaler", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_INFO = {
    "2": {
        "filename": "RealESRGAN_x2plus.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        "scale": 2,
    },
    "4": {
        "filename": "RealESRGAN_x4plus.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "scale": 4,
    },
}

_models: dict[str, RealESRGANer] = {}
_model_lock = threading.Lock()
_download_errors: dict[str, str] = {}


def build_network(scale: int) -> RRDBNet:
    return RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=scale,
    )


def _download_with_retries(info: dict, model_path: Path, attempts: int = 3) -> None:
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            load_file_from_url(
                url=info["url"],
                model_dir=str(WEIGHTS_DIR),
                file_name=info["filename"],
                progress=True,
            )
            return
        except Exception as exc:  # noqa: BLE001
            last_exc = exc

            if model_path.exists():
                try:
                    model_path.unlink()
                except OSError:
                    pass
            time.sleep(1.5 * attempt)
    raise RuntimeError(
        f"Model weight download failed after {attempts} attempts: {last_exc}"
    ) from last_exc


def get_upsampler(scale: int) -> RealESRGANer:
    key = str(scale)
    if key not in MODEL_INFO:
        raise ValueError("Scale must be 2 or 4")

    if key in _models:
        return _models[key]

    with _model_lock:
        if key in _models:
            return _models[key]

        info = MODEL_INFO[key]
        model_path = WEIGHTS_DIR / info["filename"]

        if not model_path.exists():
            try:
                _download_with_retries(info, model_path)
            except Exception as exc:  # noqa: BLE001
                _download_errors[key] = str(exc)
                raise

        use_cuda = torch.cuda.is_available()
        tile = int(os.getenv("REALESRGAN_TILE", "400"))

        try:
            upsampler = RealESRGANer(
                scale=info["scale"],
                model_path=str(model_path),
                model=build_network(info["scale"]),
                tile=tile,
                tile_pad=10,
                pre_pad=0,
                half=use_cuda,
                gpu_id=0 if use_cuda else None,
            )
        except Exception as exc:  # noqa: BLE001
            if model_path.exists():
                try:
                    model_path.unlink()
                except OSError:
                    pass
            raise RuntimeError(f"Failed to load model weights: {exc}") from exc

        _models[key] = upsampler
        return upsampler


@app.get("/")
def root():
    return {
        "message": "AI Photo Upscaler API ishlamoqda",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "supported_scales": [2, 4],
        "models_loaded": list(_models.keys()),
    }


@app.get("/health")
def health():
    return {
        "ok": True,
        "cuda": torch.cuda.is_available(),
        "models_loaded": list(_models.keys()),
    }


@app.post("/upscale")
async def upscale(
    file: UploadFile = File(...),
    scale: int = Form(2),
):
    if scale not in (2, 4):
        raise HTTPException(status_code=400, detail="scale faqat 2 yoki 4 bo'lishi kerak.")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Iltimos, rasm faylini yuklang.")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Fayl bo'sh.")

    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Fayl juda katta. Maksimum {MAX_UPLOAD_BYTES // (1024 * 1024)}MB.",
        )

    try:
        image = Image.open(io.BytesIO(raw))
        image.verify()  # FIX: catch truncated/corrupt images early with a clear error
        image = Image.open(io.BytesIO(raw)).convert("RGB")  # re-open after verify()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=400, detail=f"Rasm faylini o'qib bo'lmadi: {exc}"
        ) from exc

    original_width, original_height = image.size

    if original_width * original_height > MAX_INPUT_PIXELS:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Rasm o'lchami juda katta ({original_width}x{original_height}). "
                f"Iltimos kichikroq rasm yuklang."
            ),
        )

    try:
        input_np = np.array(image)
        upsampler = get_upsampler(scale)
        output_np, _ = upsampler.enhance(input_np, outscale=scale)
    except RuntimeError as exc:

        msg = str(exc)
        if "out of memory" in msg.lower():
            raise HTTPException(
                status_code=503,
                detail=(
                    "GPU/CPU xotirasi yetmadi. REALESRGAN_TILE qiymatini "
                    "kamaytiring (masalan 200) yoki kichikroq rasm yuklang."
                ),
            ) from exc
        raise HTTPException(
            status_code=500, detail=f"AI upscaling xatosi: {msg}"
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500,
            detail=f"AI upscaling xatosi: {type(exc).__name__}: {exc}",
        ) from exc

    output_image = Image.fromarray(output_np)
    output_buffer = io.BytesIO()
    output_image.save(output_buffer, format="PNG", optimize=True)
    output_buffer.seek(0)

    return StreamingResponse(
        output_buffer,
        media_type="image/png",
        headers={
            "Content-Disposition": 'attachment; filename="upscaled.png"',
            "X-Original-Size": f"{original_width}x{original_height}",
            "X-Output-Size": f"{output_image.width}x{output_image.height}",
            "X-Upscale-Scale": str(scale),
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):  # noqa: ANN001
    return JSONResponse(
        status_code=500,
        content={"detail": f"Kutilmagan server xatosi: {type(exc).__name__}: {exc}"},
    )
