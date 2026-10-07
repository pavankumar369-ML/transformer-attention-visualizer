"""Prediction wrapper: sentences in, class probabilities out."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from shared.config import MAX_LENGTH
from shared.model_loader import encode

from ._compat import canonical, load_classifier


@dataclass
class Encoded:
    """One tokenised sentence (with [CLS]/[SEP] or <s>/</s>)."""

    input_ids: torch.Tensor   # (1, seq)
    tokens: list[str]
    special: np.ndarray       # (seq,) bool, True for special tokens
    family: str               # bert | distilbert | roberta


def encode_text(text: str, model_name: str) -> Encoded:
    """Tokenise ``text`` once; every method then works on identical tokens."""
    tok, _ = load_classifier(model_name)
    ids = encode(tok, text)["input_ids"]
    row = ids[0].tolist()
    special = np.array(tok.get_special_tokens_mask(row, already_has_special_tokens=True), dtype=bool)
    return Encoded(ids, tok.convert_ids_to_tokens(row), special, canonical(model_name))


@torch.no_grad()
def predict_proba(texts: list[str], model_name: str, batch_size: int = 32) -> np.ndarray:
    """Return array (n_texts, 2): probability of [negative, positive] sentiment."""
    tok, model = load_classifier(model_name)
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tok(list(texts[i:i + batch_size]), return_tensors="pt", padding=True,
                  truncation=True, max_length=MAX_LENGTH)
        out.append(torch.softmax(model(**enc).logits, dim=-1).cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0, 2))


@torch.no_grad()
def predict_proba_ids(input_ids: torch.Tensor, model_name: str, batch_size: int = 64) -> np.ndarray:
    """Probabilities for a batch of already-tokenised (and possibly masked) id rows (n, seq)."""
    _, model = load_classifier(model_name)
    out = []
    for i in range(0, input_ids.shape[0], batch_size):
        chunk = input_ids[i:i + batch_size]
        logits = model(input_ids=chunk, attention_mask=torch.ones_like(chunk)).logits
        out.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(out, axis=0)
