"""Illustrious (SDXL) adapter tests (no real weights / no GPU needed).

Covers the discrete euler schedule (timestep indices + sigma grid), the model
defaults, size rounding, the vendored CLIP tokenizer, and the CLIP-G / VAE /
UNet construction with random weights (CPU).
"""
from __future__ import annotations

import torch

from thenoise.dit.illustrious.sampling import (
    discrete_timesteps,
    get_alphas_cumprod,
    get_sigmas,
    sigma,
)
from thenoise.dit.illustrious.utils import ILLUSTRIOUS_TOKENIZER_CONFIG_DIR
from thenoise.models import IllustriousModel


def test_schedule_discrete_timesteps_descending():
    ts = discrete_timesteps(28)
    assert len(ts) == 28
    assert ts[0] == 999  # noise -> clean
    assert ts[-1] == 0
    assert ts == sorted(ts, reverse=True)


def test_schedule_sigmas_descending_with_trailing_zero():
    sigmas = get_sigmas(28)
    assert len(sigmas) == 29
    assert sigmas[-1] == 0.0
    assert all(s > n for s, n in zip(sigmas[:-1], sigmas[1:]))
    # largest sigma at t=999, smallest (t=0) just above 0.
    assert sigma(0) > 0.0
    assert sigma(999) > sigma(0)


def test_alphas_cumprod_bounds():
    abar = get_alphas_cumprod()
    assert abar.shape == (1000,)
    assert 0.0 < abar[0] < 1.0  # nearly clean
    assert abar[-1] < 0.05       # heavily noised


def test_illustrious_defaults():
    assert IllustriousModel.DEFAULT_STEPS == 28
    assert IllustriousModel.DEFAULT_GUIDANCE_SCALE == 5.5
    assert IllustriousModel.SAMPLER == "euler"
    assert IllustriousModel.LATENT_CHANNELS == 4
    assert IllustriousModel._VAE_SCALE == 8


def test_resolve_size_rounds_to_multiple_of_8():
    m = IllustriousModel.__new__(IllustriousModel)
    assert m.resolve_size(1000, 1000) == (1000, 1000)
    assert m.resolve_size(1000, 1002) == (1000, 1008)
    assert m.resolve_size(513, 511) == (520, 512)


def test_vendored_tokenizer_config_dir_exists():
    from pathlib import Path

    d = Path(ILLUSTRIOUS_TOKENIZER_CONFIG_DIR)
    assert d.is_dir()
    for required in ("vocab.json", "merges.txt", "tokenizer_config.json"):
        assert (d / required).is_file(), f"missing vendored tokenizer file {required}"


def test_find_illustrious_tokenizer_dir(tmp_path):
    from thenoise.dit.illustrious.utils import find_illustrious_tokenizer_dir

    # Downloader layout: <out>/tokenizer/ + <out>/split_files/text_encoders/file.safetensors
    out = tmp_path / "models"
    (out / "tokenizer").mkdir(parents=True)
    te = out / "split_files" / "text_encoders" / "clip_l_g.safetensors"
    assert find_illustrious_tokenizer_dir(str(te)) == str(out / "tokenizer")


def test_find_illustrious_tokenizer_dir_returns_none_without_tokenizer(tmp_path):
    from thenoise.dit.illustrious.utils import find_illustrious_tokenizer_dir

    te = tmp_path / "split_files" / "text_encoders" / "clip_l_g.safetensors"
    assert find_illustrious_tokenizer_dir(str(te)) is None


def test_tokenizer_loads_offline():
    from transformers import CLIPTokenizer

    tok = CLIPTokenizer.from_pretrained(ILLUSTRIOUS_TOKENIZER_CONFIG_DIR, local_files_only=True)
    ids = tok("a cat", padding="max_length", max_length=77, truncation=True, return_tensors="pt")
    assert ids.input_ids.shape == (1, 77)


def test_vae_decodes_latent_to_pixels():
    from thenoise.dit.illustrious.vae import AutoencoderKLIllustrious

    vae = AutoencoderKLIllustrious()
    latents = torch.randn(1, 4, 8, 8)
    pixels = vae.decode_to_pixels(latents)
    assert pixels.shape == (1, 3, 64, 64)
    assert (-1.0 <= pixels).all() and (pixels <= 1.0).all()


def test_size_embedding_dim():
    # 6 timestep_embeddings of 256 -> 1536, concatenated after pooled (1280) = 2816.
    from thenoise.dit.illustrious.models import timestep_embedding

    t = torch.tensor([1024.0])
    emb = timestep_embedding(t, 256)
    assert emb.shape == (1, 256)
    assert emb.dtype == torch.float32


def test_sigma_at_matches_sampling_grid():
    # ``_sigma_at`` (used for the UNet input scaling) must agree with the
    # standalone ``sigma(t)`` over the whole discrete grid.
    from thenoise.models.illustrious import IllustriousModel

    m = object.__new__(IllustriousModel)
    m._alphas_cumprod = get_alphas_cumprod()

    t = torch.arange(0, 1000, 37, dtype=torch.int64)
    s_at = m._sigma_at(t)
    assert s_at.shape == t.shape
    for i, ti in enumerate(t.tolist()):
        assert abs(float(s_at[i]) - sigma(ti)) < 1e-6
    # noisy end has the largest sigma; clean end approaches 0.
    assert float(s_at[0]) < float(s_at[-1])


def test_denoise_input_scaling_factor():
    # ComfyUI EPS ``calculate_input`` scales the model input by 1/sqrt(sigma^2+1).
    # At the noisiest step this is ~1/sigma_max (~0.04), so feeding the raw
    # latent (as the old code did) was ~26x too large and collapsed to gray.
    from thenoise.models.illustrious import IllustriousModel

    m = object.__new__(IllustriousModel)
    m._alphas_cumprod = get_alphas_cumprod()

    t = torch.tensor([999])
    sigma_hat = m._sigma_at(t)
    factor = 1.0 / torch.sqrt(sigma_hat**2 + 1)
    assert 0.0 < float(factor) < 1.0
    # the noisiest step's factor is well below 1 (input must be scaled down).
    assert float(factor) < 0.1
