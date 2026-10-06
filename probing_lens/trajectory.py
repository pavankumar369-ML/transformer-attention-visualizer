"""
Probing Lens - one word's path through the layers.

Two live measurements, both cosine similarity:

  self_similarity   a word against itself one layer earlier. Low values
                    mark the layers that rewrite the word the most.
  pair_similarity   the same word in two different sentences ("bank" by a
                    river vs "bank" with a paycheck). At layer 0 the two
                    vectors are nearly the same; where they drift apart is
                    where context starts to matter.

Caveat for the viva: contextual vectors are anisotropic (Ethayarajh, 2019) -
in upper layers, even unrelated words have fairly high cosine similarity.
So read pair_similarity by its shape across layers, not by absolute values.
"""

from typing import List, Optional, Tuple

import numpy as np

from probing_lens.hidden import word_vectors
from shared import sentences
from shared.config import BASE_MODEL

# (sentence A, sentence B, shared word, sense A, sense B)
WORD_SENSE_PAIRS: List[Tuple[str, str, str, str, str]] = [
    (sentences.AMBIGUITY[0].text, sentences.AMBIGUITY[1].text, "bank", "money", "river"),
    ("He swung the bat and hit the ball.", "A bat flew out of the dark cave.",
     "bat", "sport", "animal"),
    ("She drank cold water from the spring.", "The flowers bloom every spring.",
     "spring", "water", "season"),
]


def _cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity of two (n, d) arrays."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))


def find_word(words: List[str], target: str) -> int:
    """Index of the first word matching target, case-insensitively."""
    lowered = [w.lower() for w in words]
    if target.lower() not in lowered:
        raise ValueError(f"{target!r} is not in {words}")
    return lowered.index(target.lower())


def self_similarity(vectors: np.ndarray, word_index: int) -> np.ndarray:
    """Cosine of each layer l (1..L) with layer l-1, for one word.

    vectors is (n_layers + 1, n_words, hidden). Returns length n_layers;
    entry k is the similarity between layers k and k+1.
    """
    path = vectors[:, word_index, :]
    return _cos(path[:-1], path[1:])


def pair_similarity(
    text_a: str,
    text_b: str,
    word: str,
    model_name: str = BASE_MODEL,
) -> np.ndarray:
    """Cosine between the two uses of `word`, at every layer 0..L."""
    words_a, va = word_vectors(text_a, model_name)
    words_b, vb = word_vectors(text_b, model_name)
    return _cos(va[:, find_word(words_a, word), :], vb[:, find_word(words_b, word), :])


def separation_layer(similarity: np.ndarray) -> Optional[int]:
    """First layer where the pair has covered half its total drift.

    Threshold-free: "half-way from the starting similarity to the lowest
    similarity reached". None if the similarity never really drops.
    """
    start, low = similarity[0], similarity.min()
    if start - low < 0.05:
        return None
    halfway = start - (start - low) / 2
    return int(np.argmax(similarity <= halfway))
