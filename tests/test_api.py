"""API tests using a fake runtime (no torch, no weights, no TestClient)."""
from __future__ import annotations

from thenoise.api import create_app, Text2ImageRequest
from thenoise.runtime import Settings, Runtime


def _fake_runtime(tmp_path=None):
    """Runtime with a fake pipeline + fake model (and optionally upscalers dir)."""
    class FakePipeline:
        def generate(self, request):
            self.last_request = request
            from PIL import Image
            return Image.new("RGB", (8, 8))

        def list_loras(self):
            return ["style", "pose"]

    runtime = Runtime(Settings())
    runtime._pipeline = FakePipeline()
    runtime._model = object()
    runtime._model_name = "fake"

    if tmp_path is not None:
        (tmp_path / "RealESRGAN_x4.safetensors").write_text("x")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "x2.safetensors").write_text("x")
        runtime._pixel_upscalers.upscaler_dir = str(tmp_path)
    return runtime


def _empty_runtime():
    return Runtime(Settings())


def _endpoint(app, path):
    for r in app.routes:
        if getattr(r, "path", None) == path:
            return r.endpoint
    raise AssertionError(f"no route {path}")


def test_upscalers_lists_names(tmp_path):
    app = create_app(_fake_runtime(tmp_path))
    res = _endpoint(app, "/upscalers")()
    assert res["upscalers"] == ["RealESRGAN_x4", "sub/x2"]


def test_upscalers_available_without_model():
    # Pixel upscalers are a pixel-space/server concern and need no diffusion model.
    app = create_app(_empty_runtime())
    res = _endpoint(app, "/upscalers")()
    assert res["upscalers"] == []


def test_text2image_passes_pixel_upscaler(tmp_path):
    runtime = _fake_runtime(tmp_path)
    app = create_app(runtime)
    req = Text2ImageRequest(prompt="a fox", width=512, height=512, pixel_upscaler="RealESRGAN_x4")
    res = _endpoint(app, "/text2image")(req)
    assert res.status_code == 200
    assert runtime._pipeline.last_request.pixel_upscaler == "RealESRGAN_x4"


def test_request_field_defaults_none():
    req = Text2ImageRequest(prompt="x", width=512, height=512)
    assert req.pixel_upscaler is None
