"""Does the picture match the words?

The axis the suite was missing. Every other image check is about competence --
it decodes, it is the right size, it is more than one colour, its text is
legible -- and every one of them scored FLUX.2 klein and Z-Image Turbo
identically. None of them looks at whether the image is of what was asked for.

Two backends, because the literature disagrees about which is better and there
was no reason to guess which suits this machine:

  pickscore  yuvalkirstain/PickScore_v1, a CLIP-H fine-tune trained on 500k
             human preference pairs from Pick-a-Pic. Plain transformers.
  hpsv2      xswu/HPSv2, a CLIP-H fine-tune on the HPD v2 preference dataset.
             open_clip, and the checkpoint is a raw .pt rather than a repo.

Both are torch on MPS and both live behind the optional `metrics` dependency
group (687MB, 36 packages). When it is absent the check WARNS AND PASSES: a
missing optional tool that failed every candidate at once would look exactly
like a model regression.

Scores from the two are not comparable to each other -- different heads,
different scales. Compare candidates within one backend.
"""
from __future__ import annotations

from pathlib import Path

from harness.checks.base import CheckResult

BACKENDS = ("pickscore", "hpsv2")
PICKSCORE_REPO = "yuvalkirstain/PickScore_v1"
PICKSCORE_PROCESSOR = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
HPSV2_REPO = "xswu/HPSv2"
HPSV2_CHECKPOINT = "HPS_v2.1_compressed.pt"

# A 4GB model load per case would dominate a 19-second generation.
_CACHE: dict[str, object] = {}


class MetricsUnavailable(RuntimeError):
    """The optional scoring dependencies are not installed."""


def reset_cache() -> None:
    _CACHE.clear()


def cache_info() -> tuple:
    return tuple(sorted(_CACHE))


def _device():
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _pickscore():
    if "pickscore" not in _CACHE:
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:
            raise MetricsUnavailable(
                f"{exc}; install with: uv sync --group metrics") from exc
        processor = AutoProcessor.from_pretrained(PICKSCORE_PROCESSOR)
        model = AutoModel.from_pretrained(PICKSCORE_REPO).eval().to(_device())
        _CACHE["pickscore"] = (processor, model, torch)
    return _CACHE["pickscore"]


def _embedding(out):
    """The embedding, whichever shape this transformers version returns.

    transformers 4 returned a bare tensor from get_image_features /
    get_text_features; transformers 5 returns a BaseModelOutputWithPooling whose
    pooler_output IS the projected embedding. Handling both keeps a version
    bump from turning into an AttributeError deep inside a scoring run.
    """
    return out if hasattr(out, "norm") else out.pooler_output


def _score_pickscore(path: Path, prompt: str) -> float:
    from PIL import Image
    processor, model, torch = _pickscore()
    with Image.open(path) as im:
        image = im.convert("RGB")
    with torch.no_grad():
        pixels = processor(images=[image], return_tensors="pt")["pixel_values"]
        text = processor(text=[prompt], padding=True, truncation=True,
                         max_length=77, return_tensors="pt")
        image_emb = _embedding(
            model.get_image_features(pixel_values=pixels.to(_device())))
        text_emb = _embedding(model.get_text_features(
            **{k: v.to(_device()) for k, v in text.items()}))
        image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        # The logit scale is what the preference head was trained against, so
        # the raw cosine would not be on the scale the model learned.
        score = (model.logit_scale.exp() * (text_emb @ image_emb.T))[0][0]
    return float(score)


def _hpsv2():
    if "hpsv2" not in _CACHE:
        try:
            import open_clip
            import torch
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise MetricsUnavailable(
                f"{exc}; install with: uv sync --group metrics") from exc
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-H-14", pretrained=None, precision="fp32", device=_device())
        checkpoint = hf_hub_download(HPSV2_REPO, HPSV2_CHECKPOINT)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["state_dict"])
        model = model.eval().to(_device())
        tokenizer = open_clip.get_tokenizer("ViT-H-14")
        _CACHE["hpsv2"] = (model, preprocess, tokenizer, torch)
    return _CACHE["hpsv2"]


def _score_hpsv2(path: Path, prompt: str) -> float:
    from PIL import Image
    model, preprocess, tokenizer, torch = _hpsv2()
    with Image.open(path) as im:
        image = preprocess(im.convert("RGB")).unsqueeze(0).to(_device())
    text = tokenizer([prompt]).to(_device())
    with torch.no_grad():
        out = model(image, text)
        image_emb, text_emb = out[0], out[1]
        image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        score = model.logit_scale.exp() * (image_emb @ text_emb.T)[0][0]
    return float(score)


def _backend(name: str):
    if name not in BACKENDS:
        raise ValueError(
            f"unknown adherence backend {name!r}; known: {', '.join(BACKENDS)}")
    return _score_pickscore if name == "pickscore" else _score_hpsv2


def score(path: str | Path, prompt: str, backend: str = "pickscore") -> float:
    """How well the image at `path` matches `prompt`. Higher is better.

    Scores are comparable BETWEEN CANDIDATES ON ONE BACKEND, and not between
    backends: the two heads have different scales.
    """
    path = Path(path)
    scorer = _backend(backend)
    if not path.exists():
        raise FileNotFoundError(path)
    return float(scorer(path, prompt))


def check(path: str | Path, prompt: str, min_adherence: float | None = None,
          backend: str = "pickscore") -> CheckResult:
    """Measure prompt adherence; judge it only if a minimum was declared."""
    try:
        value = score(path, prompt, backend=backend)
    except MetricsUnavailable as exc:
        return CheckResult(
            True, "",
            [f"prompt adherence not measured: {exc} "
             f"(optional `metrics` dependency group)"])

    result = CheckResult(True, "")
    result.metrics = {"adherence": round(value, 4)}
    if min_adherence is not None and value < min_adherence:
        result.ok = False
        result.reason = (f"prompt adherence {value:.2f} is below the required "
                         f"{min_adherence}")
    return result
