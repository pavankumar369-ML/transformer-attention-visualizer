"""
Probing Lens - layer similarity with linear CKA (Kornblith et al., 2019).

CKA compares two sets of representations of the SAME inputs, row by row:
X is (n_words, d1), Y is (n_words, d2). It returns 0 (unrelated) to 1
(identical up to rotation and uniform scaling). Because it never compares
individual neurons, it works across layers and across models whose neurons
mean different things.
"""

from typing import Sequence

import numpy as np


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    num = np.linalg.norm(Y.T @ X, "fro") ** 2
    den = np.linalg.norm(X.T @ X, "fro") * np.linalg.norm(Y.T @ Y, "fro")
    return float(num / den)


def cka_matrix(layers_a: Sequence[np.ndarray], layers_b: Sequence[np.ndarray]) -> np.ndarray:
    """CKA between every layer of A and every layer of B.

    layers_a[i] and layers_b[j] are (n_words, hidden) for the same words.
    Pass the same list twice for one model's layer x layer map.
    """
    out = np.zeros((len(layers_a), len(layers_b)))
    for i, X in enumerate(layers_a):
        for j, Y in enumerate(layers_b):
            out[i, j] = linear_cka(X, Y)
    return out
