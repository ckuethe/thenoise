"""Download checkpoints from Civitai via ``civitapy`` and split them into thenoise parts.

thenoise loads models from separate ``--dit`` / ``--vae`` / ``--text-encoder``
files, but Civitai SDXL/Illustrious checkpoints ship as one combined
``.safetensors`` (Stability-AI SDXL layout). This script downloads a model by its
Civitai model ID using the ``civitapy`` package, then splits the combined
checkpoint into the three parts, writing them alongside the downloaded file:

  <out>/Checkpoint/<modelid>_<name>_<creator>/<basemodel>/
    <combined>.safetensors
    diffusion_models/{name}_unet.safetensors        (--dit)
    vae/{name}_vae.safetensors                      (--vae)
    text_encoders/{name}_clip_l_g.safetensors       (--text-encoder)

Downloads are scoped by base-model filter so only versions whose base model
thenoise can actually run are fetched — unrelated checkpoints and LoRA/VAE-only
versions are skipped. By default every base model thenoise supports is allowed;
use ``--base`` (repeatable) to restrict to specific ones.

Usage:
    python scripts/download_civitai.py 1331249 --out ./models/bubbli
    python scripts/download_civitai.py 1331249 5678 --base anima --base krea --out ./models/foo
    python scripts/download_civitai.py 1331249 --no-base-model-filter --keep-combined
    python scripts/download_civitai.py "https://civitai.com/models/1302719/bla-bla-bla?modelVersionId=1591915"

Model targets may be a bare model ID or a full Civitai URL. A URL with
``?modelVersionId=<id>`` downloads only that specific version.
"""
from __future__ import annotations

import argparse
import logging
import urllib.parse
from pathlib import Path
from typing import Sequence

from safetensors.torch import load_file, save_file

try:
    from civitapy import CivitAIClient, CivitAIError, Model, ModelVersion
except ImportError:
    raise SystemExit(
        "civitapy is required to download from Civitai.\n"
        "Install it with:  uv pip install civitapy"
    )

logger = logging.getLogger(__name__)


def parse_target(s: str) -> tuple[int, int | None]:
    """Parse a model ID or Civitai URL into ``(model_id, version_id_or_None)``.

    Accepts a bare model ID (``1331249``), a model URL
    (``.../models/<id>/<slug>``), and a URL pinning one version
    (``...?modelVersionId=<id>``), on any Civitai domain (``civitai.com``,
    ``civitai.red``, ...).
    """
    try:
        return int(s), None
    except ValueError:
        pass
    parts = [p for p in urllib.parse.urlsplit(s).path.split("/") if p]
    if len(parts) < 2 or parts[0] != "models":
        raise ValueError(f"not a Civitai model URL or model ID: {s!r}")
    model_id = int(parts[1])
    version_id = None
    qs = urllib.parse.parse_qs(urllib.parse.urlsplit(s).query)
    if "modelVersionId" in qs:
        version_id = int(qs["modelVersionId"][0])
    return model_id, version_id

#: Stability-AI SDXL layout key prefixes inside a combined checkpoint.
UNET_PREFIX = "model.diffusion_model."
VAE_PREFIX = "first_stage_model."
CLIP_L_PREFIX = "conditioner.embedders.0.transformer."
CLIP_G_PREFIX = "conditioner.embedders.1.model."

#: Prediction-type marker tensors (``v_pred``, ``edm_mean``/``edm_std``, ...) that
#: some SDXL checkpoints carry at the top level. Preserved into the dit split so
#: ``SdxlModel`` can autodetect the prediction type.
PREDICTION_MARKERS = frozenset(
    {
        "v_pred",
        "ztsnr",
        "edm_mean",
        "edm_std",
        "edm_vpred.sigma_max",
        "edm_vpred.sigma_min",
    }
)

#: Human-friendly ``--base`` choices mapped to the Civitai base-model strings
#: each one covers. Kept in this script (not derived from the model catalog);
#: extend it when thenoise gains support for another base model.
BASE_MODEL_CHOICES: dict[str, list[str]] = {
    "illustrious": ["Illustrious", "SDXL 1.0"],
    "anima": ["Anima"],
    "zimage": ["ZImageTurbo", "Z-Image"],
    "flux-klein": ["Flux.2 Klein 4B", "Flux.2 Klein 9B", "Flux.2 Klein"],
    "krea": ["Krea 2", "Krea"],
}

#: Every base model thenoise supports (the default filter when ``--base`` is
#: unspecified).
DEFAULT_BASE_MODELS: list[str] = [
    base
    for bases in BASE_MODEL_CHOICES.values()
    for base in bases
]


def split_checkpoint(checkpoint: str, out: Path, name: str) -> tuple[Path, Path, Path]:
    """Split a combined SDXL/Illustrious checkpoint into thenoise parts.

    Loads ``checkpoint`` and partitions its state dict into a UNet (DiT), VAE,
    and a combined CLIP-L + CLIP-G text encoder file, saving them under ``out``
    with the given ``name`` stem:

      out/diffusion_models/{name}_unet.safetensors        (--dit)
      out/vae/{name}_vae.safetensors                      (--vae)
      out/text_encoders/{name}_clip_l_g.safetensors       (--text-encoder)

    Returns the (dit, vae, text_encoder) paths.

    Raises:
        ValueError: If any partition is empty (the file is not an SDXL/Illustrious
            combined checkpoint).
    """
    print(f"Loading {checkpoint} ...")
    sd = load_file(checkpoint)

    unet = {
        k[len(UNET_PREFIX):]: v
        for k, v in sd.items()
        if k.startswith(UNET_PREFIX)
    }
    # Keep prediction-type marker tensors (``v_pred``, ``edm_mean``/``edm_std``,
    # ...) so ``SdxlModel`` can autodetect the prediction type from the dit file.
    unet.update({k: v for k, v in sd.items() if k in PREDICTION_MARKERS})
    vae = {
        k[len(VAE_PREFIX):]: v
        for k, v in sd.items()
        if k.startswith(VAE_PREFIX)
        and (k.startswith(VAE_PREFIX + "decoder.") or "post_quant_conv" in k)
    }
    clip_l = {
        k[len(CLIP_L_PREFIX):]: v
        for k, v in sd.items()
        if k.startswith(CLIP_L_PREFIX) and not k.endswith("position_ids")
    }
    clip_g = {
        k[len(CLIP_G_PREFIX):]: v
        for k, v in sd.items()
        if k.startswith(CLIP_G_PREFIX)
    }
    for part_name, part in [("unet", unet), ("vae", vae), ("clip_l", clip_l), ("clip_g", clip_g)]:
        if not part:
            raise ValueError(
                f"partition {part_name!r} is empty: {checkpoint} may not be an "
                "SDXL/Illustrious combined checkpoint"
            )

    (out / "diffusion_models").mkdir(parents=True, exist_ok=True)
    (out / "vae").mkdir(parents=True, exist_ok=True)
    (out / "text_encoders").mkdir(parents=True, exist_ok=True)

    dit = out / "diffusion_models" / f"{name}_unet.safetensors"
    vae_path = out / "vae" / f"{name}_vae.safetensors"
    te = out / "text_encoders" / f"{name}_clip_l_g.safetensors"

    save_file(unet, str(dit))
    save_file(vae, str(vae_path))
    combined_te = {
        **{f"clip_l.{k}": v for k, v in clip_l.items()},
        **{f"clip_g.{k}": v for k, v in clip_g.items()},
    }
    save_file(combined_te, str(te))

    return dit, vae_path, te


def _split_paths(base: Path, name: str) -> tuple[Path, Path, Path]:
    """The three thenoise output paths for a split rooted at ``base``."""
    return (
        base / "diffusion_models" / f"{name}_unet.safetensors",
        base / "vae" / f"{name}_vae.safetensors",
        base / "text_encoders" / f"{name}_clip_l_g.safetensors",
    )


def _already_split(client: CivitAIClient, model_id: int, name: str) -> bool:
    """True if every version of ``model_id`` already has all three split files.

    Skips the download when the model was previously downloaded and split.
    """
    model = Model(**client.models_get(model_id))
    if model.type != "Checkpoint":
        return False
    for version in model.model_versions:
        base = Path(client._version_download_dir(model, version.base_model))
        if not all(p.exists() for p in _split_paths(base, name)):
            return False
    return True


def _already_split_version(client: CivitAIClient, version_id: int, name: str) -> bool:
    """True if a single version already has all three split files."""
    version = ModelVersion(**client.model_versions_get(version_id))
    model = Model(**client.models_get(version.model_id))
    base = Path(client._version_download_dir(model, version.base_model))
    return all(p.exists() for p in _split_paths(base, name))


def _split_downloaded(client: CivitAIClient, model_id: int, paths: Sequence[str], name: str) -> None:
    """Split every downloaded combined checkpoint under ``paths``.

    Prints a thenoise block per split checkpoint. Non-checkpoint or non-SDXL
    files are left in place (with a message) rather than failing the download.
    """
    import os

    for path in sorted(map(Path, paths)):
        print(f"Downloaded: {path}")

    # Only split checkpoints; a model may also carry LoRA/VAE/config files that
    # don't fit the combined SDXL layout.
    model_type = client.models_get(model_id).get("type")
    if model_type != "Checkpoint":
        print(f"Model type is {model_type!r} — not a checkpoint, skipping split.")
        return

    results: list[tuple[Path, Path, Path]] = []
    for path in sorted(map(Path, paths)):
        # Split files live next to the downloaded checkpoint (its parent dir), so
        # multiple models never overwrite one another.
        split_paths = _split_paths(path.parent, name)
        if all(p.exists() for p in split_paths):
            print(f"  (already split, skipping: {path.parent})")
            results.append(split_paths)
            continue
        try:
            results.append(split_checkpoint(str(path), path.parent, name))
        except ValueError as e:
            # Not a combined SDXL/Illustrious checkpoint (e.g. a LoRA or VAE
            # file); leave it in place rather than failing the whole download.
            print(f"  (skipping split: {e})")
        except CivitAIError as e:
            print(f"  (download error: {e})")

    if not results:
        print("\nNo combined checkpoints were split.")
        return

    # A model may ship one combined checkpoint (single part) or several versions
    # / parts (multi part); print a separate thenoise block per split checkpoint.
    print(f"\nDone. Split {len(results)} checkpoint(s). Point thenoise at:")
    for i, (dit, vae, te) in enumerate(results, 1):
        if len(results) > 1:
            print(f"  [{i}]")
        print(f"  --dit            {os.path.relpath(dit)}")
        print(f"  --vae            {os.path.relpath(vae)}")
        print(f"  --text-encoder   {os.path.relpath(te)}")


def download_model(model_id: int, out: Path, *, base_models: Sequence[str] | None, keep_combined: bool, name: str, progress: bool, version_id: int | None = None) -> None:
    import os

    if not os.environ.get("CIVITAI_TOKEN"):
        raise SystemExit(
            "CIVITAI_TOKEN is not set. Civitai requires a bearer token to download model files.\n"
            "Create an API key at https://civitai.com/account (Account -> API Keys), then set it:\n"
            "  export CIVITAI_TOKEN=<your-key>"
        )

    out.mkdir(parents=True, exist_ok=True)

    client = CivitAIClient(
        download_dir=str(out),
        base_models=list(base_models) if base_models else None,
    )
    if base_models:
        print(f"Base-model filter: {', '.join(base_models)}")
    else:
        print("Base-model filter: none (downloading every version)")

    if version_id is not None:
        if _already_split_version(client, version_id, name):
            print(f"Model {model_id} version {version_id} already downloaded and split — skipping.")
            return
        print(f"Downloading civitai model {model_id} version {version_id} ...")
        paths = client.download_model_version(version_id, progress=progress)
        if not paths:
            print("Nothing downloaded — version does not match the base-model filter.")
            return
    else:
        if _already_split(client, model_id, name):
            print(f"Model {model_id} already downloaded and split — skipping.")
            return
        print(f"Downloading civitai model {model_id} ...")
        paths = client.download_model(model_id, progress=progress)
        if not paths:
            print("Nothing downloaded — no version matched the filter.")
            return

    _split_downloaded(client, model_id, paths, name)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    # httpx (used by civitapy) logs every connection/retry at INFO; keep that
    # noise out of the downloader's output.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    ap = argparse.ArgumentParser(description="Download + split Civitai checkpoints via civitapy")
    ap.add_argument(
        "targets", nargs="+",
        help="Civitai model ID(s) or full model URL(s); a URL with "
             "?modelVersionId=<id> downloads only that version",
    )
    ap.add_argument("--out", default="./models/civitai", help="output directory")
    ap.add_argument(
        "--name", default="model",
        help="output filename stem (e.g. 'bubbli' -> bubbli_unet.safetensors)",
    )
    ap.add_argument(
        "--base",
        action="append",
        choices=sorted(BASE_MODEL_CHOICES),
        help="restrict downloads to one base model (repeatable; default: all supported)",
    )
    ap.add_argument(
        "--no-base-model-filter",
        action="store_true",
        help="disable the base-model filter and download every version",
    )
    ap.add_argument(
        "--keep-combined",
        action="store_true",
        help="keep the downloaded combined checkpoint after splitting",
    )
    ap.add_argument(
        "--no-progress",
        action="store_true",
        help="disable the per-file download progress bar",
    )
    args = ap.parse_args()

    if args.no_base_model_filter:
        base_models = None
    elif args.base:
        base_models = [
            base
            for name in args.base
            for base in BASE_MODEL_CHOICES[name]
        ]
    else:
        base_models = DEFAULT_BASE_MODELS

    for target in args.targets:
        model_id, version_id = parse_target(target)
        download_model(
            model_id,
            Path(args.out),
            base_models=base_models,
            keep_combined=args.keep_combined,
            name=args.name,
            progress=not args.no_progress,
            version_id=version_id,
        )


if __name__ == "__main__":
    main()
