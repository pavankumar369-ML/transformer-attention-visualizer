"""Faithfulness: deletion curves, comprehensiveness, sufficiency, AOPC, random control."""
from __future__ import annotations

import numpy as np
import torch

from ._compat import load_classifier
from .predict import predict_proba_ids

MAX_FRAC = 0.5   # delete up to 50% of content tokens
TOP_FRAC = 0.2   # comprehensiveness / sufficiency use the top 20%
N_RANDOM = 10


def rank_order(scores: np.ndarray, special: np.ndarray) -> np.ndarray:
    """Content-token positions sorted from most to least supportive of the prediction."""
    idx = np.flatnonzero(~np.asarray(special, dtype=bool))
    return idx[np.argsort(-np.asarray(scores)[idx], kind="stable")]


def _masked(ids: torch.Tensor, positions: np.ndarray, mask_id: int) -> torch.Tensor:
    v = ids.clone()
    if len(positions):
        v[torch.as_tensor(np.asarray(positions), dtype=torch.long)] = mask_id
    return v


def deletion_curve(ids: torch.Tensor, order: np.ndarray, model_name: str, pred: int,
                   mask_id: int, max_frac: float = MAX_FRAC, top_frac: float = TOP_FRAC) -> dict:
    """Replace the top-k ranked tokens by [MASK] for k = 0..max_frac*n and record the probability drop.

    ``ids`` is a 1-D id vector. Returns fractions, drops, aopc (mean drop over k>=1),
    comprehensiveness (drop with top 20% removed) and sufficiency (drop with only top 20% kept).
    """
    n = len(order)
    kmax = max(1, int(np.floor(max_frac * n)))
    k20 = min(n, max(1, int(round(top_frac * n))))
    rows = [_masked(ids, order[:k], mask_id) for k in range(kmax + 1)]
    rows.append(_masked(ids, order[:k20], mask_id))   # comprehensiveness variant
    rows.append(_masked(ids, order[k20:], mask_id))   # sufficiency variant (keep only top 20%)
    p = predict_proba_ids(torch.stack(rows), model_name)[:, pred]
    p0 = p[0]
    drops = p0 - p[:kmax + 1]
    return {"fractions": [k / n for k in range(kmax + 1)], "drops": drops.tolist(),
            "aopc": float(drops[1:].mean()), "comprehensiveness": float(p0 - p[kmax + 1]),
            "sufficiency": float(p0 - p[kmax + 2]), "k20": k20, "p0": float(p0)}


def drop_at_fraction(ids: torch.Tensor, scores: np.ndarray, special: np.ndarray,
                     fraction: float, model_name: str, pred: int) -> float:
    """Probability drop after masking the top ``fraction`` of content tokens (0.0 -> exactly 0)."""
    order = rank_order(scores, special)
    k = int(round(fraction * len(order)))
    tok, _ = load_classifier(model_name)
    row = ids.reshape(-1)
    p = predict_proba_ids(torch.stack([row, _masked(row, order[:k], tok.mask_token_id)]), model_name)[:, pred]
    return float(p[0] - p[1])


def faithfulness_report(ids: torch.Tensor, special: np.ndarray, scores: dict[str, np.ndarray],
                        model_name: str, pred: int, n_random: int = N_RANDOM, seed: int = 0) -> dict:
    """Curves for every method plus a 'Random' control (mean of ``n_random`` random orders)."""
    tok, _ = load_classifier(model_name)
    row = ids.reshape(-1)
    content = np.flatnonzero(~np.asarray(special, dtype=bool))
    report = {m: deletion_curve(row, rank_order(s, special), model_name, pred, tok.mask_token_id)
              for m, s in scores.items()}
    rng = np.random.default_rng(seed)
    runs = [deletion_curve(row, rng.permutation(content), model_name, pred, tok.mask_token_id)
            for _ in range(n_random)]
    report["Random"] = {
        "fractions": runs[0]["fractions"],
        "drops": np.mean([r["drops"] for r in runs], axis=0).tolist(),
        **{key: float(np.mean([r[key] for r in runs]))
           for key in ("aopc", "comprehensiveness", "sufficiency", "p0")},
        "k20": runs[0]["k20"],
    }
    return report
