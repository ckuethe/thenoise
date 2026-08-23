"""Focused HTTP API. A single generic /text2image endpoint serves whichever model
the runtime currently holds (the runtime loads exactly one model at a time).

Synchronous request/response: each generate() call blocks until the image is ready.
A per-model inference lock serializes concurrent requests.

The request carries only the shared, model-agnostic parameters. Per-model defaults
(including the "advanced" sampler params) are owned by the model class and are NOT
exposed here.
"""
from __future__ import annotations

import io
import logging
import os
import base64
from typing import List, Literal, Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

logger = logging.getLogger(__name__)

_UI_DIR = os.path.join(os.path.dirname(__file__), "ui")


class Text2ImageRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    width: Optional[int] = None
    height: Optional[int] = None

    steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    seed: Optional[int] = None
    upscale: bool = False
    upscale_factor: float = 1.0
    upscale_type: str = "refined"
    sampler: Optional[str] = None
    qwen_vae_enhance: bool = False
    film_grain: float = 0.0
    sharpening: float = 0.0
    lora_specs: Optional[List[str]] = None  # ["filename.safetensors:0.8", ...]
    pixel_upscaler: Optional[str] = None  # name (no .safetensors) in upscaler_dir
    out: Literal["png", "json"] = "png"

    def to_request(self):
        """Convert this wire request into a ``GenerateRequest`` for the controller."""
        from .models.config import GenerateRequest

        return GenerateRequest(
            prompt=self.prompt,
            negative_prompt=self.negative_prompt,
            width=self.width,
            height=self.height,
            steps=self.steps,
            guidance_scale=self.guidance_scale,
            seed=self.seed,
            upscale=self.upscale,
            upscale_factor=self.upscale_factor,
            upscale_type=self.upscale_type,
            sampler=self.sampler,
            qwen_vae_enhance=self.qwen_vae_enhance,
            film_grain=self.film_grain,
            sharpening=self.sharpening,
            lora_specs=self.lora_specs,
            pixel_upscaler=self.pixel_upscaler,
        )


def create_app(runtime) -> FastAPI:
    app = FastAPI(title="thenoise", version="0.1.0")

    @app.get("/", response_class=HTMLResponse)
    def ui():
        with open(os.path.join(_UI_DIR, "index.html"), encoding="utf-8") as f:
            return f.read()

    @app.get("/health")
    def health():
        return {"status": "ok", "models": runtime.available()}

    @app.get("/lora")
    def loras():
        """List available LoRA names (short, no .safetensors suffix)."""
        pipeline = runtime.pipeline
        if pipeline is None:
            return Response(status_code=503, content="no model is loaded")
        return {"loras": pipeline.list_loras()}

    @app.get("/upscalers")
    def upscalers():
        """List available pixel upscaler names (short, no .safetensors suffix).

        Pixel upscalers are a pixel-space / server concern and need no diffusion
        model, so this works even when no model is loaded.
        """
        return {"upscalers": runtime.pixel_upscalers.list()}

    @app.post("/text2image")
    def text2image(req: Text2ImageRequest):
        pipeline = runtime.pipeline
        if pipeline is None:
            return Response(status_code=503, content="no model is loaded")
        try:
            image = pipeline.generate(req.to_request())
        except Exception as e:  # surface generation errors cleanly
            logger.exception("generation failed")
            return Response(status_code=500, content=f"generation failed: {e}")

        buf = io.BytesIO()
        image.save(buf, format="PNG", pnginfo=getattr(image, "_pnginfo", None))
        content = buf.getvalue()

        if req.out == "json":
            return {"b64_json": base64.b64encode(content).decode("ascii")}
        return Response(
            content=content,
            media_type="image/png",
        )

    return app
