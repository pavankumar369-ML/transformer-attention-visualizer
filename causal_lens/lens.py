"""Causal lens: compute(text, model_name) -> LensResult and render(result).

model_name is a sentiment checkpoint id from shared.config.SENTIMENT_MODELS.
Importance matrices and pruning curves come from data/ (written by
`python -m causal_lens.run_offline`); the live ablation runs on the user's sentence.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from causal_lens.ablate import CLASSIFIERS, grid_shape, predict_with_mask, short_key
from causal_lens.importance import load_saved
from shared.contracts import LensResult
from shared.model_loader import load_classifier, tokens_of

DATA = Path(__file__).resolve().parents[1] / "data"
TOP_K = 10
OFFLINE_HINT = "Head importance not computed yet. Run `python -m causal_lens.run_offline`."


def _remove_top(prob_drop: np.ndarray, k: int) -> np.ndarray:
    m = np.ones(prob_drop.size)
    m[np.argsort(-prob_drop.ravel(), kind="stable")[:k]] = 0
    return m.reshape(prob_drop.shape)


def compute(text: str, model_name: str) -> LensResult:
    """Importance matrix + prediction before/after removing the TOP_K most important heads."""
    key = short_key(model_name)
    tok, _ = load_classifier(CLASSIFIERS[key])
    shape = grid_shape(key)
    saved = load_saved(key)
    acc_drop, prob_drop = saved if saved else (np.zeros(shape), np.zeros(shape))
    before = float(predict_with_mask([text], np.ones(shape), key)[0, 1])
    after = float(predict_with_mask([text], _remove_top(prob_drop, TOP_K), key)[0, 1]) if saved else before
    summary = (f"The model gives this sentence a {before:.1%} chance of being positive. With the {TOP_K} "
               f"most important heads switched off, that becomes {after:.1%}."
               if saved else OFFLINE_HINT)
    return LensResult(lens="causal", text=text, model_name=model_name,
                      data=dict(key=key, tokens=tokens_of(tok, text), importance=acc_drop,
                                prob_drop=prob_drop, before=before, after=after, top_k=TOP_K,
                                has_data=bool(saved), summary=summary),
                      notes=[] if saved else [OFFLINE_HINT])


def render(result: LensResult) -> None:
    import streamlit as st
    from causal_lens.pruning import first_drop
    from causal_lens.render import importance_fig, pruning_fig

    d = result.data
    key, importance = d["key"], d["importance"]
    for n in result.notes:
        st.info(n)
    if not d["has_data"]:
        return
    st.caption(d["summary"])

    st.subheader("Which heads matter?")
    st.plotly_chart(importance_fig(importance), width="stretch")
    with st.expander("How to read this"):
        st.write("Each square is one attention head (rows = layers, columns = heads). We switched that single "
                 "head off and re-ran 200 movie-review sentences. Darker = accuracy fell more. Most squares are "
                 "pale: the model barely notices many heads missing.")

    st.subheader("How many heads does the model really need?")
    pr = DATA / f"pruning_{key}.csv"
    if pr.exists():
        df = pd.read_csv(pr)
        pct, heads = first_drop(df)
        st.plotly_chart(pruning_fig(df), width="stretch")
        st.caption(f"Removing the least important heads first, accuracy stays within 2 points of the original "
                   f"until {pct}% ({heads} heads) are removed.")
        with st.expander("How to read this"):
            st.write("Solid line: remove the least important heads first. Dashed line: remove heads in random "
                     "order. If the solid line stays high much longer, our importance scores identify heads that "
                     "really matter.")

    st.subheader("Switch heads off yourself")
    L, H = importance.shape
    opts = [f"L{l + 1}-H{h + 1}" for l in range(L) for h in range(H)]
    chosen = st.multiselect("Heads to switch off", opts, key=f"causal_heads_{key}",
                            placeholder="Pick heads by layer and number, for example L5-H3")
    mask = np.ones((L, H))
    for c in chosen:
        l, h = c[1:].split("-H")
        mask[int(l) - 1, int(h) - 1] = 0
    p0 = d["before"]
    p1 = float(predict_with_mask([result.text], mask, key)[0, 1])
    c1, c2 = st.columns(2)
    c1.metric("Chance of positive, all heads on", f"{p0:.1%}")
    c2.metric("Chance of positive, chosen heads off", f"{p1:.1%}",
              delta=f"{(p1 - p0) * 100:+.1f} points" if chosen else None, delta_color="off")
    with st.expander("How to read this"):
        st.write("Pick heads to switch off and compare the model's prediction on the current sentence before "
                 "and after. A big change means those heads carry information the model uses.")
