"""Attribution & Faithfulness lens: compute() and render() for the app.

  compute(text, model_name) -> LensResult   numbers only, no Streamlit
  render(result)                            draws the Attribution tab

``LensResult.data`` keys: text, model_name, family, tokens, special, pred,
pred_label, pred_prob, probs, scores {SHAP, IG, Attention, Rollout} (one float
per token incl. special tokens), ig_delta, ig_steps, agreement {methods,
spearman, top3} (4x4), faithfulness {method: curve} incl. "Random".
Everything is plain lists/dicts/floats, so st.cache_data can pickle it.
"""
from __future__ import annotations

import numpy as np

from shared.contracts import LensResult

from ._compat import LABELS
from .agreement import agreement_tables
from .attribute import IG_DELTA_TOL, METHODS, SHAP_MAX_EVALS, compute_attributions
from .faithfulness import N_RANDOM, faithfulness_report
from .render import render_panels

LENS_KEY = "attribution"


def analyze(text: str, model_name: str, n_random: int = N_RANDOM,
            max_evals: int = SHAP_MAX_EVALS) -> dict:
    """Run all four methods, agreement and faithfulness; return the plain-dict payload."""
    b = compute_attributions(text, model_name, max_evals)
    agr = agreement_tables(b.scores, b.special)
    faith = faithfulness_report(b.input_ids, b.special, b.scores, model_name, b.pred, n_random=n_random)
    return {
        "text": text, "model_name": model_name, "family": b.family, "tokens": b.tokens,
        "special": b.special.tolist(), "pred": b.pred, "pred_label": LABELS[b.pred],
        "pred_prob": float(b.probs[b.pred]), "probs": b.probs.tolist(),
        "scores": {m: np.asarray(b.scores[m], dtype=float).tolist() for m in METHODS},
        "ig_delta": float(b.ig_delta), "ig_steps": int(b.ig_steps),
        "agreement": {"methods": list(METHODS), "spearman": agr.spearman.to_numpy().tolist(),
                      "top3": agr.top3.to_numpy().tolist()},
        "faithfulness": faith,
    }


def compute(text: str, model_name: str) -> LensResult:
    """Run the analysis (no plotting; the app wraps this in st.cache_data)."""
    p = analyze(text, model_name)
    notes = []
    if p["ig_delta"] > IG_DELTA_TOL:
        notes.append(f"Integrated Gradients has not fully converged (delta {p['ig_delta']:.3f}); "
                     "treat its scores as approximate.")
    n_content = int(np.sum(~np.array(p["special"])))
    if n_content < 4:
        notes.append("Very short sentence: faithfulness curves only have a few points.")
    return LensResult(lens=LENS_KEY, text=text, model_name=model_name, data=p, notes=notes)


def render(result: LensResult) -> None:
    """Draw the Attribution tab for a result returned by :func:`compute`."""
    import streamlit as st

    for note in result.notes:
        st.info(note)
    render_panels(result.data)
