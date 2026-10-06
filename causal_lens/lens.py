"""Causal lens: compute() + render() following the shared lens contract."""
from pathlib import Path

import numpy as np
import pandas as pd

from causal_lens.ablate import CLASSIFIERS, grid_shape, predict_with_mask
from causal_lens.importance import load_saved
from shared.contracts import LensResult
from shared.model_loader import load_classifier, tokens_of

DATA = Path(__file__).resolve().parents[1] / "data"
KEYS = {"bert-base-uncased": "bert", "distilbert-base-uncased": "distilbert", "roberta-base": "roberta"}
TOP_K = 10


def _key(model_name: str) -> str:
    return KEYS.get(model_name, model_name)


def _remove_top(prob_drop: np.ndarray, k: int) -> np.ndarray:
    m = np.ones(prob_drop.size)
    m[np.argsort(-prob_drop.ravel(), kind="stable")[:k]] = 0
    return m.reshape(prob_drop.shape)


def compute(text: str, model_name: str) -> LensResult:
    """Importance matrix + prediction before/after removing the TOP_K most important heads."""
    key = _key(model_name)
    tok, _ = load_classifier(CLASSIFIERS[key])
    shape = grid_shape(key)
    saved = load_saved(key)
    acc_drop, prob_drop = saved if saved else (np.zeros(shape), np.zeros(shape))
    before = float(predict_with_mask([text], np.ones(shape), key)[0, 1])
    after = float(predict_with_mask([text], _remove_top(prob_drop, TOP_K), key)[0, 1]) if saved else before
    summary = (f"P(positive) is {before:.2f}; removing the {TOP_K} most important heads makes it {after:.2f}."
               if saved else "Head importance not computed yet: run `python -m causal_lens.run_offline`.")
    return LensResult(tokens=tokens_of(tok, text), scores=acc_drop, model_name=model_name, summary=summary,
                      extra=dict(text=text, key=key, prob_drop=prob_drop, before=before, after=after,
                                 has_data=bool(saved)))


def render(result: LensResult) -> None:
    import streamlit as st
    from causal_lens.pruning import first_drop
    from causal_lens.render import importance_fig, pruning_fig

    key = result.extra["key"]
    st.subheader("Which heads matter?")
    if not result.extra["has_data"]:
        st.info(result.summary)
        return
    st.plotly_chart(importance_fig(result.scores), use_container_width=True)
    with st.expander("How to read this"):
        st.write("Each square is one attention head (rows = layers, columns = heads). We switched that single "
                 "head off and re-ran 200 movie-review sentences. Darker = accuracy fell more. Most squares are "
                 "pale: the model barely notices many heads missing.")

    st.subheader("How many heads does the model really need?")
    pr = DATA / f"pruning_{key}.csv"
    if pr.exists():
        df = pd.read_csv(pr)
        pct, heads = first_drop(df)
        st.plotly_chart(pruning_fig(df), use_container_width=True)
        st.caption(f"Removing the least important heads first, accuracy stays within 2 points of the original "
                   f"until {pct}% ({heads} heads) are removed.")
        with st.expander("How to read this"):
            st.write("Solid line: remove the least important heads first. Dashed line: remove heads in random "
                     "order. If the solid line stays high much longer, our importance scores identify heads that "
                     "really matter.")

    st.subheader("Switch heads off yourself")
    L, H = result.scores.shape
    opts = [f"L{l + 1}-H{h + 1}" for l in range(L) for h in range(H)]
    chosen = st.multiselect("Heads to remove (layer-head)", opts, key="causal_heads")
    mask = np.ones((L, H))
    for c in chosen:
        l, h = c[1:].split("-H")
        mask[int(l) - 1, int(h) - 1] = 0
    text = result.extra["text"]
    p0 = float(predict_with_mask([text], np.ones((L, H)), key)[0, 1])
    p1 = float(predict_with_mask([text], mask, key)[0, 1])
    c1, c2 = st.columns(2)
    c1.metric("P(positive) before", f"{p0:.3f}")
    c2.metric("P(positive) after", f"{p1:.3f}", delta=f"{p1 - p0:+.3f}")
    with st.expander("How to read this"):
        st.write("Pick heads to switch off and compare the model's prediction on the current sentence before "
                 "and after. A big change means those heads carry information the model uses.")
