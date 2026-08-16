"""Illustrious (SDXL-based anime) adapter.

Illustrious-XL is a Stable Diffusion XL derivative: an SDXL LDM UNet (with the
Illustrious transformer-depth reallocation), the SDXL dual-CLIP text encoders
(CLIP-L + CLIP-G -> 2048-dim cross-attention, CLIP-G pooled for the ADM vector),
and the SDXL VAE (4 channels, 8x compression). It is a discrete
noise-prediction (epsilon) model sampled with the shared ``euler`` sampler over
a discrete DDIM-style sigma grid.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import torch

from thenoise.dit.illustrious.models import timestep_embedding
from thenoise.dit.illustrious.sampling import (
    discrete_timesteps,
    get_alphas_cumprod,
    get_sigmas,
    sigmas_for_timesteps,
)
from thenoise.dit.illustrious.text import OpenClipTextTransformer
from thenoise.dit.illustrious.utils import (
    find_illustrious_tokenizer_dir,
    load_illustrious_dit,
    load_illustrious_text_encoders,
    load_illustrious_tokenizer,
)
from thenoise.dit.illustrious.lora import convert_hyper_sd_lora, lora_uses_diffusers_unet_keys
from thenoise.dit.illustrious.vae import AutoencoderKLIllustrious, load_illustrious_vae
from thenoise.models.base import Conditioning, DiffusionModel, Step, normalize_keys
from thenoise.models.config import ModelConfig, SamplingParams
from thenoise.utils.math import round_up

logger = logging.getLogger(__name__)

#: Pooled CLIP-G projection width (1280) and size-embedding width (6 x 256).
POOLED_DIM = 1280
SIZE_EMBED_DIM = 1536


class IllustriousModel(DiffusionModel):
    name = "illustrious"

    # SDXL defaults: ~28 euler steps, CFG ~5.5, 1024x1024. This matches the
    # widely-recommended Illustrious settings; higher steps / CFG tend to
    # over-saturate and hurt prompt adherence rather than help.
    DEFAULT_STEPS = 28
    DEFAULT_GUIDANCE_SCALE = 5.5
    DEFAULT_WIDTH = 1024
    DEFAULT_HEIGHT = 1024

    # The shared euler sampler reproduces the discrete DDIM/euler update when
    # ``denoise_step`` returns the predicted noise and the schedule's deltas are
    # sigma differences.
    SAMPLER = "euler"
    # SDXL is a discrete-epsilon model; only the euler solver is valid. A
    # requested ``er_sde`` falls back to euler with a warning (see create_sampler).
    SUPPORTED_SAMPLERS = ["euler"]

    # 4-channel SDXL latent, 8x spatial compression.
    LATENT_CHANNELS = 4
    _VAE_SCALE = 8

    MAX_SEQUENCE_LENGTH = 77

    @staticmethod
    def detect(f) -> bool:
        """True if this handle is an SDXL LDM UNet (Illustrious-compatible).

        The classic CompVis ``UNetModel`` layout is uniquely identified by the
        ``input_blocks`` / ``middle_block`` block lists together with the
        ``label_emb`` and ``time_embed`` conditioning MLPs. Keys are normalized
        first so repackaged checkpoints (``model.diffusion_model.``) resolve
        identically. Other registered models (flow DiTs) share none of these.
        """
        keys = list(normalize_keys(f.keys()))
        has_input = any(k.startswith("input_blocks.") for k in keys)
        has_middle = any(k.startswith("middle_block.") for k in keys)
        has_label = any(k.startswith("label_emb.") for k in keys)
        has_time = any(k.startswith("time_embed.") for k in keys)
        return has_input and has_middle and has_label and has_time

    def __init__(
        self,
        *,
        config: ModelConfig,
    ):
        super().__init__(config=config)

        # Enable TF32 for float32 matmuls (the CLIP text towers / weight loading
        # run some fp32 ops); suppresses the TensorFloat32 UserWarning and is
        # faster with negligible precision loss for inference.
        torch.set_float32_matmul_precision("high")

        logger.info("Loading Illustrious UNet from %s", config.dit_path)
        self.dit = load_illustrious_dit(config.dit_path, device=config.device, dtype=config.dtype)
        self.dit.eval().requires_grad_(False)

        logger.info(
            "Loading Illustrious text encoders (CLIP-L + CLIP-G) from %s",
            config.text_encoder_path,
        )
        self.clip_l, self.clip_g = load_illustrious_text_encoders(
            config.text_encoder_path, device=config.device, dtype=config.dtype
        )
        self.clip_l.eval().requires_grad_(False)
        self.clip_g.eval().requires_grad_(False)
        self.tokenizer = load_illustrious_tokenizer(
            find_illustrious_tokenizer_dir(config.text_encoder_path)
        )

        logger.info("Loading Illustrious SDXL VAE from %s", config.vae_path)
        self.vae = load_illustrious_vae(self.vae_path, device=self.device, disable_mmap=True)
        self.vae.to(self.dtype).eval().requires_grad_(False)

        # Discrete alphas_cumprod grid on-device, for per-step sigma lookup.
        # ComfyUI's EPS ``calculate_input`` scales the UNet input by
        # ``1/sqrt(sigma^2 + 1)`` (see ``BaseModel._apply_model``); without it
        # the noisiest input is ~sigma_max (~26) too large and denoise collapses.
        self._alphas_cumprod = get_alphas_cumprod(device=self.device)

        # Per-step cached ADM vector (pooled text + size embedding), built in
        # ``prepare_latent`` from the request's ``Conditioning``.
        self._y = None
        self._y_uncond = None

        logger.info("Illustrious model ready on %s (%s)", config.device, config.dtype)

    # ------------------------------------------------------------ kernels
    def encode_prompt(
        self,
        prompt: str,
        negative_prompt: str = "",
        *,
        guidance_scale: float,
    ) -> Conditioning:
        context, pooled = self._encode_prompt(prompt)
        null = None
        neg_pooled = None
        if guidance_scale > 1.0:
            neg_context, neg_pooled = self._encode_prompt(negative_prompt)
            null = neg_context
        return Conditioning(cond=context, null=null, pooled=pooled, neg_pooled=neg_pooled)

    def _get_lora_sd(self, filename: str) -> dict[str, torch.Tensor]:
        """Load a LoRA, converting Hyper-SD diffusers-keyed UNet LoRAs to LDM keys.

        Hyper-SD step-reduction LoRAs target the diffusers SDXL UNet layout
        (``lora_unet_down_blocks_*``); our UNet uses LDM keys. Auto-convert so
        ``--lora Hyper-SDXL-8steps-CFG-lora.safetensors --steps 8`` just works.
        """
        sd = super()._get_lora_sd(filename)
        if lora_uses_diffusers_unet_keys(sd):
            logger.info(
                "Converting diffusers-keyed UNet LoRA (%s) to LDM key naming", filename
            )
            sd = convert_hyper_sd_lora(sd)
        return sd

    def _encode_prompt(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(cross_attn_context [1,77,2048], pooled [1,1280])``."""
        dev = torch.device(self.device)
        # ComfyUI tokenizes CLIP-L and CLIP-G with *different* padding: CLIP-L
        # pads with EOS (49407) but CLIP-G pads with 0. Sharing one padded
        # sequence (EOS) for both made the CLIP-G cross-attention context attend
        # to ~66 spurious EOS tokens, corrupting prompt adherence (incoherent
        # subjects / generic anime faces).
        raw = self.tokenizer(prompt, truncation=True, max_length=self.MAX_SEQUENCE_LENGTH)
        base = raw["input_ids"][: self.MAX_SEQUENCE_LENGTH]
        pad = self.MAX_SEQUENCE_LENGTH - len(base)
        ids_l = torch.tensor([base + [49407] * pad], device=dev)  # CLIP-L pad = EOS
        ids_g = torch.tensor([base + [0] * pad], device=dev)      # CLIP-G pad = 0

        with torch.no_grad():
            out_l = self.clip_l(ids_l, output_hidden_states=True)
            hidden_l = out_l.hidden_states[-2]  # [1, 77, 768] penultimate
            hidden_g, pooled = self.clip_g(ids_g)  # [1, 77, 1280], [1, 1280]
            context = torch.cat([hidden_l, hidden_g], dim=-1)  # [1, 77, 2048]
        # The CLIP-G pooled text vector is passed to the UNet's ``label_emb``
        # unnormalized (``text_projection(eos @ ln_final)``), matching ComfyUI /
        # diffusers SDXL; the Linear label_emb was trained on that raw scale.
        return context.to(self.dtype), pooled.to(self.dtype)

    def init_latents(self, params: SamplingParams) -> torch.Tensor:
        dev = torch.device(self.device)
        shape = (
            1,
            self.dit.in_channels,
            params.height // self._VAE_SCALE,
            params.width // self._VAE_SCALE,
        )
        generator = torch.Generator(device=dev).manual_seed(params.seed)
        return torch.randn(shape, generator=generator, device=dev, dtype=self.dtype)

    def _size_embedding(self, height: int, width: int) -> torch.Tensor:
        """ComfyUI SDXL size embedding: 6 timestep_embeddings of 256, concatenated.

        Order: [height, width, crop_h, crop_w, target_height, target_width],
        with crop = (0, 0) and target = (height, width). Returns ``[1, 1536]``.
        """
        dev = torch.device(self.device)
        parts = []
        for value in (height, width, 0, 0, height, width):
            t = torch.tensor([float(value)], device=dev)
            parts.append(timestep_embedding(t, 256).to(torch.float32))
        return torch.cat(parts, dim=0).flatten().unsqueeze(0).to(dev)

    def prepare_latent(
        self,
        latents: torch.Tensor,
        cond: Conditioning,
        params: SamplingParams,
    ) -> torch.Tensor:
        # Scale the initial noise by the scheduler's max sigma (ComfyUI's flow
        # euler for the discrete SDXL EPS model: init = noise * sigma_max).
        sigmas = get_sigmas(params.steps)
        scaled = latents * sigmas[0]

        size_embeds = self._size_embedding(params.height, params.width)
        self._y = torch.cat([cond.pooled, size_embeds], dim=-1).to(self.dtype)
        if cond.null is not None and cond.neg_pooled is not None:
            self._y_uncond = torch.cat([cond.neg_pooled, size_embeds], dim=-1).to(self.dtype)
        else:
            self._y_uncond = None
        return scaled

    def schedule(self, params: SamplingParams) -> list[Step]:
        dev = torch.device(self.device)
        steps = params.steps
        ts = discrete_timesteps(steps)  # noise -> clean
        sigmas = sigmas_for_timesteps(ts)  # noise -> clean, trailing 0
        return [
            Step(
                t=torch.tensor(float(ts[i]), device=dev, dtype=torch.float32),
                delta=torch.tensor(sigmas[i] - sigmas[i + 1], device=dev, dtype=torch.float32),
            )
            for i in range(steps)
        ]

    def _sigma_at(self, t: torch.Tensor) -> torch.Tensor:
        """Sigma for discrete timestep index ``t`` (matches the sampling grid)."""
        abar = self._alphas_cumprod[t.to(torch.long)]
        return torch.sqrt((1.0 - abar) / abar)

    def denoise_step(
        self,
        latents: torch.Tensor,
        t: torch.Tensor,
        cond: Conditioning,
        guidance_scale: float,
        i: int,
    ) -> torch.Tensor:
        dev = torch.device(self.device)
        t_full = t.to(dev).reshape(1)
        context = cond.cond.to(dev, dtype=self.dtype)
        # ComfyUI EPS ``calculate_input``: the UNet expects the latent scaled by
        # ``1/sqrt(sigma^2 + 1)`` (the DDPM-space latent x_t), not the raw
        # EDM-space latent. Apply it before every (un)conditional forward.
        sigma = self._sigma_at(t_full).to(latents.dtype)
        scaled = latents / torch.sqrt(sigma**2 + 1)
        with torch.no_grad():
            eps = self.dit(scaled, t_full, self._y, context)
            if guidance_scale > 1.0 and self._y_uncond is not None and cond.null is not None:
                uncond = self.dit(
                    scaled, t_full, self._y_uncond, cond.null.to(dev, dtype=self.dtype)
                )
                eps = uncond + guidance_scale * (eps - uncond)
        # ComfyUI flow euler for the EPS model: the model output (eps) IS the
        # velocity; the sampler integrates ``x -= delta * eps``.
        return eps

    def finalize_latent(self, latents: torch.Tensor, params: SamplingParams) -> torch.Tensor:
        # The denoised latent is already in the UNet's scaled space; the VAE's
        # ``decode_to_pixels`` applies the 1/scaling_factor before decoding.
        return latents

    def resolve_size(self, width: int, height: int) -> tuple[int, int]:
        # SDXL latents are 8x compressed; round up to a multiple of 8.
        size = round_up(width, 8), round_up(height, 8)
        # Illustrious is trained near 1024x1024; strongly off-native sizes (e.g.
        # 512) produce incoherent output (observed: a red panda becomes a garden).
        if min(width, height) < 768 or max(width, height) > 1536:
            logger.warning(
                "Illustrious is trained near 1024x1024; requested %sx%s is far "
                "from native and may produce garbled results. Recommended ~1024x1024.",
                width,
                height,
            )
        return size

    def _upscale_format(self) -> str:
        raise NotImplementedError(
            "Illustrious (SDXL 4-channel latents) does not support latent upscale yet; "
            "no 4-channel upscaler weights are committed."
        )


__all__ = ["IllustriousModel"]
