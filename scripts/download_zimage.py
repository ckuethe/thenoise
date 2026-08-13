"""Download the Z-Image-Turbo model artifacts into a local directory.

The DiT and VAE come from ``Comfy-Org/z_image_turbo`` (single-file bf16, no auth);
the Qwen3-4B text encoder and its tokenizer come from ``Tongyi-MAI/Z-Image-Turbo``
(the directory form, which ships ``config.json`` + the Qwen chat template needed by
the caption encoder).

  DiT           diffusion_models/z_image_turbo_bf16.safetensors
  VAE           vae/ae.safetensors                                (Flux VAE)
  Text encoder  text_encoder/  (config.json + 3 shards + index)   (Qwen3-4B)
  Tokenizer     tokenizer/     (vocab.json, tokenizer.json, ...)

Usage:
    python scripts/download_zimage.py --out ./models/zimage
"""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

COMFY_REPO = "Comfy-Org/z_image_turbo"
TONGYI_REPO = "Tongyi-MAI/Z-Image-Turbo"

# Single-file DiT / VAE from the ComfyUI export.
SINGLE_FILES = [
    ("dit", COMFY_REPO, "split_files/diffusion_models/z_image_turbo_bf16.safetensors"),
    ("vae", COMFY_REPO, "split_files/vae/ae.safetensors"),
]

# Directory-form text encoder (config + shards + index) from the official repo.
TEXT_ENCODER_FILES = [
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "model-00001-of-00003.safetensors",
    "model-00002-of-00003.safetensors",
    "model-00003-of-00003.safetensors",
]

# Qwen3 tokenizer (carries the chat template used by the caption encoder).
TOKENIZER_FILES = [
    "merges.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
]


def main() -> None:
    ap = argparse.ArgumentParser(description="Download Z-Image-Turbo model artifacts")
    ap.add_argument("--out", default="./models/zimage", help="output directory")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for name, repo, path in SINGLE_FILES:
        dest = hf_hub_download(repo, path, local_dir=str(out))
        print(f"{name:14s} -> {dest}")

    for file in TEXT_ENCODER_FILES:
        dest = hf_hub_download(TONGYI_REPO, f"text_encoder/{file}", local_dir=str(out))
        print(f"{'text_encoder':14s} -> {dest}")

    for file in TOKENIZER_FILES:
        dest = hf_hub_download(TONGYI_REPO, f"tokenizer/{file}", local_dir=str(out))
        print(f"{'tokenizer':14s} -> {dest}")


if __name__ == "__main__":
    main()
