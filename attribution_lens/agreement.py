"""Agreement between attribution methods: Spearman rank correlation and top-3 overlap."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .attribute import METHODS

TOP_K = 3


@dataclass
class AgreementResult:
    """Two symmetric 4x4 tables indexed by method."""

    spearman: pd.DataFrame
    top3: pd.DataFrame


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return 0.0
    rho = spearmanr(a, b).correlation
    return 0.0 if rho is None or np.isnan(rho) else float(rho)


def _top_k(scores: np.ndarray, k: int) -> set[int]:
    return set(np.argsort(-scores, kind="stable")[:k].tolist())


def agreement_tables(scores: dict[str, np.ndarray], special: np.ndarray,
                     methods: tuple[str, ...] = METHODS, k: int = TOP_K) -> AgreementResult:
    """Pairwise Spearman rho and top-k token overlap (|A∩B|/k) over content tokens only.

    Constant score vectors (undefined rho) give 0. Tables are symmetric with 1.0 on the diagonal.
    """
    keep = ~np.asarray(special, dtype=bool)
    vecs = {m: np.asarray(scores[m], dtype=float)[keep] for m in methods}
    kk = max(1, min(k, int(keep.sum())))
    n = len(methods)
    rho = np.eye(n)
    ovl = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = vecs[methods[i]], vecs[methods[j]]
            rho[i, j] = rho[j, i] = _spearman(a, b)
            ovl[i, j] = ovl[j, i] = len(_top_k(a, kk) & _top_k(b, kk)) / kk
    return AgreementResult(pd.DataFrame(rho, index=methods, columns=methods),
                           pd.DataFrame(ovl, index=methods, columns=methods))
