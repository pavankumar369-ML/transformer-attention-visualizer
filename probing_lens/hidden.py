"""
Probing Lens - hidden states, one vector per word.

Labels (POS, NER) are per word, but BERT works on subwords: "unbelievable"
may become ["un", "##bel", "##ievable"]. We keep the vector of the FIRST
subword of each word and drop the rest, so every word gets exactly one
vector per layer.

Words are located by character offset rather than by the tokenizer's own
word splitting, because BERT and RoBERTa split words differently and we want
the same word list from every model. For each word we ask the tokenizer which
token covers the word's first character (char_to_token); that token is the
first subword.

Shape reminder: model(...).hidden_states is a tuple of length n_layers + 1.
Entry 0 is the embedding layer, entry l is the output of transformer layer l.
"""

import re
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from shared.config import BASE_MODEL
from shared.model_loader import encode, load_base_model

# A word is a run of letters/digits (keeping contractions like "didn't"
# whole), or a run of punctuation.
_WORD_RE = re.compile(r"\w+(?:['’]\w+)*|[^\w\s]+")


def split_words(text: str) -> List[Tuple[str, int]]:
    """Split raw text into (word, start_char) pairs."""
    return [(m.group(), m.start()) for m in _WORD_RE.finditer(text)]


def load(model_name: str):
    """(tokenizer, model) from the shared loader.

    load_base_model() and load_base_model(BASE_MODEL) are separate lru_cache
    entries, so asking for BASE_MODEL by name would load a second copy of
    BERT next to the Attention lens's one.
    """
    return load_base_model() if model_name == BASE_MODEL else load_base_model(model_name)


def _first_subword(enc, batch_index: int, start: int, end: int) -> Optional[int]:
    """Index of the first token covering any character of text[start:end].

    Usually the first character is enough. Scanning the rest of the span
    covers characters the tokenizer normalises away; None means the word
    was truncated off the end.
    """
    for char in range(start, end):
        tok = enc.char_to_token(batch_index, char)
        if tok is not None:
            return tok
    return None


def _run(model, enc) -> np.ndarray:
    """Forward pass -> array (batch, n_layers + 1, seq, hidden)."""
    inputs = {k: v for k, v in enc.items() if k != "offset_mapping"}
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, output_attentions=False)
    return torch.stack(out.hidden_states, dim=1).numpy()


def word_vectors(text: str, model_name: str = BASE_MODEL) -> Tuple[List[str], np.ndarray]:
    """Words, and an array (n_layers + 1, n_words, hidden).

    Uses the shared encode(), so truncation matches the other lenses; words
    cut off by truncation are dropped.
    """
    tokenizer, model = load(model_name)
    spans = split_words(text)
    enc = encode(tokenizer, text)
    states = _run(model, enc)[0]  # (layers + 1, seq, hidden)

    words, idx = [], []
    for word, start in spans:
        tok = _first_subword(enc, 0, start, start + len(word))
        if tok is not None:
            words.append(word)
            idx.append(tok)
    return words, states[:, idx, :]


def batch_word_vectors(
    sentences: Sequence[Sequence[str]],
    model_name: str = BASE_MODEL,
    batch_size: int = 32,
    max_length: int = 512,
) -> List[np.ndarray]:
    """Pre-split sentences in, one (n_layers + 1, n_words, hidden) array out each.

    For the offline datasets, where the words (and their labels) are given.
    Words are joined with single spaces so every word starts a new token in
    every model. max_length is the model maximum rather than the shared 64,
    so no labelled word is ever truncated away.
    """
    tokenizer, model = load(model_name)
    results: List[Optional[np.ndarray]] = [None] * len(sentences)

    # Sort by length so each batch pads as little as possible.
    order = sorted(range(len(sentences)), key=lambda i: len(sentences[i]))
    for b in range(0, len(order), batch_size):
        chunk = order[b:b + batch_size]
        texts, starts = [], []
        for i in chunk:
            pos, offs = 0, []
            for w in sentences[i]:
                offs.append((pos, pos + len(w)))
                pos += len(w) + 1
            texts.append(" ".join(sentences[i]))
            starts.append(offs)

        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        states = _run(model, enc)

        for row, i in enumerate(chunk):
            idx = []
            for start, end in starts[row]:
                tok = _first_subword(enc, row, start, end)
                if tok is None:
                    raise ValueError(
                        f"Word at chars {start}-{end} of sentence {i} produced no "
                        f"token; raise max_length or check the text."
                    )
                idx.append(tok)
            results[i] = states[row][:, idx, :]

    return results  # type: ignore[return-value]
