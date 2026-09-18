"""
Attention Lens - extraction.

Pulls raw attention weights out of BERT and reshapes them into something
a chart can consume. No plotting happens here; see render.py for that.

Shape reminder: model(**inputs).attentions is a tuple of length n_layers,
each element a tensor of shape (batch, n_heads, seq_len, seq_len).
Entry [b, h, i, j] = how much token i attends to token j, in head h.
Rows sum to 1.
"""

from typing import List, Optional

import numpy as np
import torch

from shared.model_loader import encode, load_base_model


def attention_matrices(text: str, model_name: Optional[str] = None):
    """Run one sentence through the encoder and return (tokens, attn).

    attn is a numpy array of shape (n_layers, n_heads, seq_len, seq_len).
    """
    if model_name:
        tokenizer, model = load_base_model(model_name)
    else:
        tokenizer, model = load_base_model()

    inputs = encode(tokenizer, text)
    with torch.no_grad():
        outputs = model(**inputs)

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    # tuple(n_layers) of (1, heads, seq, seq) -> (layers, heads, seq, seq)
    attn = torch.stack(outputs.attentions).squeeze(1).numpy()
    return tokens, attn


def select(attn: np.ndarray, layer: int = -1, head: Optional[int] = None) -> np.ndarray:
    """Pick one (seq, seq) matrix out of the full stack.

    layer: 0-indexed; -1 means the last layer.
    head:  0-indexed; None means average across all heads in that layer.

    Head-averaging is the honest default for a demo. Individual heads are
    noisy and cherry-picking one that happens to look clean is the easiest
    way to oversell what the model is doing.
    """
    layer_attn = attn[layer]
    if head is None:
        return layer_attn.mean(axis=0)
    return layer_attn[head]


def token_focus(
    matrix: np.ndarray,
    tokens: List[str],
    query_token: str,
    drop_special: bool = True,
):
    """What does one token attend to? Returns (labels, weights), sorted.

    This is the function behind the headline demo: ask what 'it' looks at
    and get 'animal' back.
    """
    if query_token not in tokens:
        raise ValueError(f"{query_token!r} is not in {tokens}")

    row = matrix[tokens.index(query_token)].copy()
    labels = list(tokens)

    if drop_special:
        keep = [i for i, t in enumerate(labels) if t not in ("[CLS]", "[SEP]")]
        row = row[keep]
        labels = [labels[i] for i in keep]
        total = row.sum()
        if total > 0:
            row = row / total  # renormalise after dropping specials

    order = np.argsort(row)[::-1]
    return [labels[i] for i in order], row[order]


def head_entropy(attn: np.ndarray, layer: int) -> np.ndarray:
    """Entropy per head in a layer. Low entropy = focused head.

    Useful for the "which heads are actually doing something" slide:
    plot entropy across heads and pick the sharpest one to show.
    """
    layer_attn = attn[layer]  # (heads, seq, seq)
    eps = 1e-12
    ent = -(layer_attn * np.log(layer_attn + eps)).sum(axis=-1)  # (heads, seq)
    return ent.mean(axis=-1)


def attention_rollout(attn: np.ndarray) -> np.ndarray:
    """Attention rollout (Abnar & Zuidema, 2020).

    Raw last-layer attention is a weak explanation because information has
    already been mixed across earlier layers. Rollout multiplies the
    residual-adjusted attention matrices together to approximate how much
    each input token contributes to each final position.

    Worth implementing: it is the difference between "we plotted a tensor"
    and "we read the interpretability literature".
    """
    n_layers, n_heads, seq, _ = attn.shape
    eye = np.eye(seq)

    rollout = eye
    for layer in range(n_layers):
        a = attn[layer].mean(axis=0)          # average heads
        a = a + eye                            # account for residual stream
        a = a / a.sum(axis=-1, keepdims=True)  # renormalise rows
        rollout = a @ rollout
    return rollout
