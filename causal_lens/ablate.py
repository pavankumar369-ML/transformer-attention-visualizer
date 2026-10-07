"""Run a classifier with chosen attention heads switched off.

We zero each removed head's slice of the attention context just before the
attention output projection. This is mathematically identical to
head_mask=0 but works in every transformers version (v5 may ignore head_mask).
"""
from contextlib import contextmanager
from typing import List, Tuple

import numpy as np
import torch

from shared.config import MAX_LENGTH, SENTIMENT_MODELS
from shared.model_loader import load_classifier


def short_key(model_name: str) -> str:
    """'distilbert' | 'roberta' | 'bert' from any model id (base or SST-2).

    Order matters: 'distilbert' and 'roberta' both contain 'bert'.
    """
    name = model_name.lower()
    for key in ("distilbert", "roberta", "bert"):
        if key in name:
            return key
    raise ValueError(f"Unknown model: {model_name}")


# short key -> sentiment checkpoint, taken from shared.config (never hard-coded here)
CLASSIFIERS = {short_key(m): m for m in SENTIMENT_MODELS}


def _attn_out_modules(model):
    return [m for n, m in model.named_modules()
            if n.endswith("attention.output.dense") or n.endswith("attention.out_lin")]


def grid_shape(model_key: str) -> Tuple[int, int]:
    _, model = load_classifier(CLASSIFIERS[short_key(model_key)])
    return model.config.num_hidden_layers, model.config.num_attention_heads


@contextmanager
def masked_heads(model, mask: np.ndarray):
    """mask: (layers, heads), 1 = keep, 0 = remove."""
    mods = _attn_out_modules(model)
    n_layers, n_heads = mask.shape
    assert len(mods) == n_layers, f"found {len(mods)} attention layers, mask has {n_layers}"
    head_dim = mods[0].in_features // n_heads
    handles = []
    for layer, mod in enumerate(mods):
        vec = torch.repeat_interleave(torch.as_tensor(mask[layer], dtype=torch.float32), head_dim)

        def pre(module, args, vec=vec):
            if not args:
                raise RuntimeError("attention output called with kwargs; hook needs positional input")
            return (args[0] * vec.to(args[0].dtype),) + tuple(args[1:])

        handles.append(mod.register_forward_pre_hook(pre))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def logits_with_mask(texts: List[str], mask: np.ndarray | None, model_key: str,
                     batch_size: int = 50) -> np.ndarray:
    """Raw logits. mask=None runs the untouched model (used for the identity test)."""
    tok, model = load_classifier(CLASSIFIERS[short_key(model_key)])
    out = []

    def run():
        for s in range(0, len(texts), batch_size):
            enc = tok(texts[s:s + batch_size], return_tensors="pt", padding=True,
                      truncation=True, max_length=MAX_LENGTH)
            out.append(model(**enc).logits.numpy())

    if mask is None:
        run()
    else:
        with masked_heads(model, mask):
            run()
    return np.concatenate(out)


def predict_with_mask(texts: List[str], mask: np.ndarray, model_name: str) -> np.ndarray:
    """Class probabilities with the masked heads switched off."""
    z = logits_with_mask(texts, mask, model_name)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def evaluate(texts, labels, mask, model_key) -> Tuple[float, float]:
    """(accuracy, mean probability of the correct class)."""
    probs = predict_with_mask(texts, mask, model_key)
    labels = np.asarray(labels)
    acc = float((probs.argmax(1) == labels).mean())
    return acc, float(probs[np.arange(len(labels)), labels].mean())
