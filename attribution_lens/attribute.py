"""Token attribution: SHAP, Integrated Gradients, last-layer attention, attention rollout.

All methods return one score per tokenizer token (special tokens included, so the
arrays line up with ``Encoded.tokens``). Scores for SHAP and IG are signed for the
*predicted class*: positive = pushes toward the prediction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import torch

from ._compat import attention_rollout_cls, canonical, load_classifier
from .predict import Encoded, encode_text, predict_proba_ids

METHODS = ("SHAP", "IG", "Attention", "Rollout")
METHOD_LABELS = {"SHAP": "SHAP", "IG": "Integrated Gradients",
                 "Attention": "Attention (last layer)", "Rollout": "Attention rollout"}
SHAP_MAX_EVALS = 300
IG_STEPS = 50
IG_DELTA_TOL = 0.05


@dataclass
class IGResult:
    """Integrated Gradients output."""

    scores: np.ndarray
    delta: float
    n_steps: int


@dataclass
class AttributionBundle:
    """Everything computed for one (text, model) pair."""

    text: str
    model_name: str
    family: str
    tokens: list[str]
    special: np.ndarray
    input_ids: torch.Tensor
    pred: int
    probs: np.ndarray
    scores: dict[str, np.ndarray] = field(default_factory=dict)
    ig_delta: float = float("nan")
    ig_steps: int = IG_STEPS


def _norm(s: str) -> str:
    return s.replace("##", "").replace("\u0120", "").strip().lower()


def _align_shap(values: np.ndarray, data: list[str], enc: Encoded) -> np.ndarray:
    """Map SHAP's own token list onto the tokenizer tokens."""
    n = len(enc.tokens)
    if len(values) == n:
        return values.astype(float)
    if len(values) == n - 2 and enc.special[0] and enc.special[-1]:
        return np.concatenate([[0.0], values, [0.0]])
    out = np.zeros(n)
    j = 0
    for i, tok in enumerate(enc.tokens):
        if enc.special[i]:
            continue
        while j < len(data) and _norm(str(data[j])) != _norm(tok):
            j += 1
        if j < len(data):
            out[i] = values[j]
            j += 1
    return out


@lru_cache(maxsize=512)
def _shap_cached(text: str, model_name: str, max_evals: int) -> tuple[float, ...]:
    import shap
    from transformers import pipeline

    tok, model = load_classifier(model_name)
    enc = encode_text(text, model_name)
    with torch.no_grad():
        probs = predict_proba_ids(enc.input_ids, model_name)[0]
    pred = int(probs.argmax())
    pipe = pipeline("text-classification", model=model, tokenizer=tok, top_k=None, device=model.device)
    explainer = shap.Explainer(pipe, shap.maskers.Text(tok))
    evals = max(max_evals, 2 * len(enc.tokens) + 1)  # partition explainer needs >= 2n+1
    sv = explainer([text], max_evals=evals, batch_size=16, silent=True)
    names = list(getattr(sv, "output_names", []) or [])
    label = model.config.id2label.get(pred, str(pred))
    col = names.index(label) if label in names else pred
    values = np.asarray(sv.values[0])
    values = values[:, col] if values.ndim == 2 else values
    return tuple(float(v) for v in _align_shap(values, list(sv.data[0]), enc))


def shap_scores(text: str, model_name: str, max_evals: int = SHAP_MAX_EVALS) -> np.ndarray:
    """SHAP value per token for the predicted class (cached; SHAP is slow)."""
    return np.array(_shap_cached(text, model_name, max_evals))


def integrated_gradients(text: str, model_name: str, n_steps: int = IG_STEPS,
                         adaptive: bool = True, max_steps: int = 400) -> IGResult:
    """Layer Integrated Gradients on the embedding layer with a [PAD] baseline.

    Baseline = same sentence, every non-special token replaced by [PAD]. If the
    convergence delta exceeds ``IG_DELTA_TOL`` and ``adaptive`` is set, n_steps is doubled.
    """
    from captum.attr import LayerIntegratedGradients

    tok, model = load_classifier(model_name)
    enc = encode_text(text, model_name)
    ids = enc.input_ids
    baseline = ids.clone()
    baseline[0, torch.from_numpy(~enc.special)] = tok.pad_token_id
    mask = torch.ones_like(ids)
    pred = int(predict_proba_ids(ids, model_name)[0].argmax())

    def forward_fn(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return torch.softmax(model(input_ids=input_ids, attention_mask=attention_mask).logits, dim=-1)

    lig = LayerIntegratedGradients(forward_fn, model.base_model.embeddings)
    steps = n_steps
    while True:
        attr, delta = lig.attribute(ids, baselines=baseline, additional_forward_args=(mask,),
                                    target=pred, n_steps=steps, return_convergence_delta=True,
                                    internal_batch_size=25)
        d = float(delta.abs().max())
        if not adaptive or d <= IG_DELTA_TOL or steps >= max_steps:
            break
        steps *= 2
    scores = attr.sum(dim=-1).squeeze(0).detach().cpu().numpy()
    return IGResult(scores=scores, delta=d, n_steps=steps)


@torch.no_grad()
def attention_scores(text: str, model_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (last-layer [CLS] attention averaged over heads, attention rollout from [CLS]).

    Rollout is delegated to ``attention_lens.extract.attention_rollout``.
    """
    _, model = load_classifier(model_name)
    enc = encode_text(text, model_name)
    out = model(input_ids=enc.input_ids, attention_mask=torch.ones_like(enc.input_ids),
                output_attentions=True)
    if not out.attentions:
        raise RuntimeError("No attentions returned: model must load with attn_implementation='eager'.")
    attn = torch.stack(out.attentions).squeeze(1).cpu().numpy()  # (L, H, S, S)
    return attn[-1].mean(axis=0)[0], attention_rollout_cls(attn)


def compute_attributions(text: str, model_name: str, max_evals: int = SHAP_MAX_EVALS) -> AttributionBundle:
    """Run all four methods on ``text`` and return an :class:`AttributionBundle`."""
    enc = encode_text(text, model_name)
    probs = predict_proba_ids(enc.input_ids, model_name)[0]
    ig = integrated_gradients(text, model_name)
    att, roll = attention_scores(text, model_name)
    return AttributionBundle(
        text=text, model_name=model_name, family=canonical(model_name), tokens=enc.tokens,
        special=enc.special, input_ids=enc.input_ids, pred=int(probs.argmax()), probs=probs,
        scores={"SHAP": shap_scores(text, model_name, max_evals), "IG": ig.scores,
                "Attention": att, "Rollout": roll},
        ig_delta=ig.delta, ig_steps=ig.n_steps)
