"""Small helpers that sit between the attribution lens and ``shared/``.

Nothing here modifies ``shared/``. It only
* resolves a model name ('bert', 'DistilBERT', or a full Hugging Face id from
  ``shared.config.SENTIMENT_MODELS``) to the checkpoint id the loader expects,
* loads the classifier through ``shared.model_loader.load_classifier``,
* builds the lens's colours from ``shared.config``,
* lists every probe sentence, and
* calls ``attention_lens.extract.attention_rollout`` (with a guarded fallback).
"""
from __future__ import annotations

import warnings

import numpy as np

LABELS = ("NEGATIVE", "POSITIVE")


def canonical(model_name: str) -> str:
    """Map any spelling ('BERT', 'textattack/roberta-base-SST-2') to bert/distilbert/roberta."""
    n = model_name.lower()
    if "distil" in n:
        return "distilbert"
    if "roberta" in n:
        return "roberta"
    if "bert" in n:
        return "bert"
    raise ValueError(f"Unknown model name: {model_name!r}")


def resolve_checkpoint(model_name: str) -> str:
    """Hugging Face id for ``model_name`` using ``shared.config.SENTIMENT_MODELS``."""
    from shared.config import SENTIMENT_MODELS

    if model_name in SENTIMENT_MODELS:
        return model_name
    key = canonical(model_name)
    for checkpoint in SENTIMENT_MODELS:
        if canonical(checkpoint) == key:
            return checkpoint
    raise ValueError(f"{model_name!r} is not one of {list(SENTIMENT_MODELS)}")


def load_classifier(model_name: str):
    """Return ``(tokenizer, model)`` via ``shared.model_loader`` (cached there)."""
    from shared.model_loader import load_classifier as _load

    return _load(resolve_checkpoint(model_name))


def get_palette() -> dict[str, str]:
    """Colours taken from shared/config.py only.

    Blue/red (toward / against the prediction) are sampled from the config's
    COLOR_SCALE ("RdBu_r": low end blue, high end red); the rest are its constants.
    """
    from plotly.colors import sample_colorscale

    from shared import config

    blue, red, light_blue = sample_colorscale(config.COLOR_SCALE, [0.08, 0.92, 0.28])

    def hexify(c: str) -> str:
        if c.startswith("rgb"):
            r, g, b = (int(float(x)) for x in c[c.index("(") + 1:-1].split(",")[:3])
            return f"#{r:02x}{g:02x}{b:02x}"
        return c

    return {"positive": hexify(blue), "negative": hexify(red), "light_blue": hexify(light_blue),
            "accent": config.ACCENT, "accent_soft": config.ACCENT_SOFT,
            "neutral": config.NEUTRAL, "grid": config.GRID, "bg": config.BG_SOFT}


def probe_sentences() -> list[str]:
    """Every sentence in shared/sentences.py (ALL_PROBES + BIAS_PROBES), as strings."""
    from shared import sentences

    return [p.text for p in (*sentences.ALL_PROBES, *sentences.BIAS_PROBES)]


def _cls_row(out: np.ndarray, size: int) -> np.ndarray | None:
    if out.ndim == 1 and out.shape[0] == size:
        return out
    if out.ndim == 2 and out.shape == (size, size):
        return out[0]
    if out.ndim == 3 and out.shape[-1] == size:
        return out[-1][0]
    return None


def attention_rollout_cls(attn: np.ndarray) -> np.ndarray:
    """Rollout attention flowing into the [CLS] row, shape (seq,). ``attn`` is (L, H, S, S).

    Uses ``attention_lens.extract.attention_rollout``; only if that call fails
    does it fall back to a minimal Abnar & Zuidema implementation (with a warning).
    """
    size = attn.shape[-1]
    try:
        from attention_lens.extract import attention_rollout

        row = _cls_row(np.asarray(attention_rollout(attn), dtype=float), size)
        if row is not None:
            return row
    except Exception:
        pass
    warnings.warn("attention_lens.extract.attention_rollout unusable; using fallback rollout.")
    roll = np.eye(size)
    for layer in attn:
        a = layer.mean(0) + np.eye(size)
        a = a / a.sum(-1, keepdims=True)
        roll = a @ roll
    return roll[0]
