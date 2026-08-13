"""Z-Image model loading utilities.

The DiT is the S3-DiT transformer; the text encoder is a Qwen3 model whose hidden
states feed the DiT's caption embedder. Both single-file and sharded (HF) DiT
checkpoints are accepted, and the text encoder is loaded from the ``text_encoder/``
directory with its sibling ``tokenizer/`` directory (which carries the Qwen chat
template used by the caption encoder).
"""
from __future__ import annotations

import logging
import os
from typing import Optional, Union

import torch

from thenoise.dit.zimage.models import ZImageTransformer2DModel
from thenoise.utils.safetensors import load_split_weights, strip_wrap_prefixes

logger = logging.getLogger(__name__)


ZIMAGE_DIT_CONFIG = dict(
    patch_size=2,
    f_patch_size=1,
    in_channels=16,
    dim=3840,
    n_layers=30,
    n_refiner_layers=2,
    n_heads=30,
    n_kv_heads=30,
    norm_eps=1e-5,
    qk_norm=True,
    cap_feat_dim=2560,
    rope_theta=256.0,
    t_scale=1000.0,
    axes_dims=(32, 48, 48),
    axes_lens=(1024, 512, 512),
)


def load_zimage_dit(
    dit_path: str,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    loading_device: Optional[Union[str, torch.device]] = None,
    config: Optional[dict] = None,
) -> ZImageTransformer2DModel:
    """Build the Z-Image S3-DiT on meta and load weights (assign=True).

    Accepts a single-file checkpoint (e.g. ComfyUI's ``z_image_turbo_bf16.safetensors``)
    or a sharded HF checkpoint (pass any ``model-0000X-of-0000Y`` shard path).
    """
    device = torch.device(device)
    loading_device = device if loading_device is None else torch.device(loading_device)
    cfg = dict(ZIMAGE_DIT_CONFIG)
    if config:
        cfg.update(config)

    logger.info(f"Loading Z-Image DiT weights from {dit_path}")
    with torch.device("meta"):
        dit = ZImageTransformer2DModel(**cfg)

    sd = load_split_weights(dit_path, device=str(loading_device), disable_mmap=True, dtype=dtype)
    sd = strip_wrap_prefixes(sd)

    dit.load_state_dict(sd, strict=True, assign=True)
    return dit


def load_zimage_text_encoder(
    path: str,
    dtype: torch.dtype = torch.bfloat16,
    device: Union[str, torch.device] = "cpu",
):
    """Load the Z-Image Qwen3 text encoder + tokenizer.

    ``path`` is the ``text_encoder/`` directory (``config.json`` + sharded
    ``model-*.safetensors`` + ``model.safetensors.index.json``). The tokenizer is
    loaded from the sibling ``tokenizer/`` directory (``os.path.dirname(path)``),
    which provides the Qwen chat template used by the caption encoder.

    Returns ``(text_encoder, tokenizer)`` where ``text_encoder`` is the bare Qwen3
    model (LM head dropped) whose ``hidden_states`` feed the DiT's caption embedder.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer_dir = os.path.join(os.path.dirname(path), "tokenizer")
    if not os.path.isdir(path):
        raise ValueError(
            f"Z-Image text encoder must be a directory, got {path!r}. "
            "Download it with `python scripts/download_zimage.py` (produces text_encoder/ + tokenizer/)."
        )
    if not os.path.isdir(tokenizer_dir):
        # Fall back to a tokenizer bundled inside the text_encoder dir, if present.
        tokenizer_dir = path

    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, local_files_only=True).model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)

    model.config.use_cache = False
    model = model.requires_grad_(False).to(device, dtype=dtype)
    logger.info(f"Loaded Z-Image text encoder. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model, tokenizer
