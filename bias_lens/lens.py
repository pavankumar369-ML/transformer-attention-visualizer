"""Bias lens: compute(text, model_name) -> LensResult and render(result).

model_name is a masked-LM id from shared.config.BIAS_MODELS.
The CrowS-Pairs benchmark is too slow to run per click, so it is read from
data/bias_results.csv (written by `python -m bias_lens.run_offline`).
The minimal-pair explorer runs live.
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from bias_lens.pair_diff import compare_pair
from bias_lens.swaps import swap_counterpart
from shared.contracts import LensResult

RESULTS = Path(__file__).resolve().parents[1] / "data" / "bias_results.csv"
OFFLINE_HINT = "Run `python -m bias_lens.run_offline` to produce the benchmark results."


def compute(text: str, model_name: str) -> LensResult:
    """Pair view for `text` vs its gender-swapped counterpart. No Streamlit."""
    other, swapped = swap_counterpart(text)
    cmp = compare_pair(text, other if swapped else text, model_name)
    notes = []
    if swapped and len(cmp["attr_diff"]):
        j = int(np.abs(cmp["attr_diff"]).argmax())
        wa = [cmp["tokens_a"][i] for i in cmp["swapped_a"]]
        wb = [cmp["tokens_b"][i] for i in cmp["swapped_b"]]
        summary = (f"Swapping {wa} → {wb} changes P(positive) from {cmp['p_pos_a']:.2f} to "
                   f"{cmp['p_pos_b']:.2f}; the largest attribution shift is on "
                   f"'{cmp['shared_labels'][j]}' ({cmp['attr_diff'][j]:+.3f}).")
    else:
        summary = "No gendered word to swap in this sentence, so the pair view shows no difference."
        notes.append("This sentence has no gendered word to swap. Type your own pair below.")
    if not RESULTS.exists():
        notes.append(OFFLINE_HINT)
    return LensResult(lens="bias", text=text, model_name=model_name,
                      data=dict(text_a=text, text_b=other if swapped else text,
                                swapped=swapped, cmp=cmp, summary=summary),
                      notes=notes)


_cached_compare = {}


def _compare_cached(a: str, b: str, model_name: str) -> dict:
    import streamlit as st
    if "fn" not in _cached_compare:
        _cached_compare["fn"] = st.cache_data(show_spinner="Comparing the pair…")(compare_pair)
    return _cached_compare["fn"](a, b, model_name)


def render(result: LensResult) -> None:
    import streamlit as st
    from bias_lens.render import pair_figs, stereotype_fig

    for n in result.notes:
        st.info(n)

    with st.expander("⚠ Content warning", expanded=False):
        st.write("This tab uses CrowS-Pairs, a research benchmark that contains offensive stereotypes "
                 "by design, so we can measure whether a model prefers them. Sentences are never shown "
                 "here; only aggregate scores. The benchmark measures a statistical preference, not harm, "
                 "and later work (Blodgett et al., 2021) found quality problems in some pairs.")

    st.subheader("Stereotype score by category")
    if RESULTS.exists():
        df = pd.read_csv(RESULTS)
        d = df[df["model"] == result.model_name]
        if d.empty:
            st.info(f"No benchmark results for {result.model_name} yet.")
        else:
            st.plotly_chart(stereotype_fig(d), use_container_width=True)
            sig = d[(d.category != "overall") & d.significantly_above_50]["category"].tolist()
            st.caption("Categories where this model assigns significantly higher likelihood to the "
                       "stereotyped sentence (p < 0.05, uncorrected): " + (", ".join(sig) if sig else "none")
                       + ". Change the model with the selector above.")
    else:
        st.info(OFFLINE_HINT)
    with st.expander("How to read this"):
        st.write("Each bar is the % of sentence pairs where the model finds the stereotyped sentence more "
                 "likely than its counterpart. 50% means no preference. The whisker is a 95% confidence "
                 "interval; dark bars are those whose interval does not cross 50%.")

    st.subheader("Minimal-pair explorer")
    tag = hashlib.md5(result.text.encode()).hexdigest()[:8]   # fresh inputs when the sentence changes
    a = st.text_input("Sentence A", result.data["text_a"], key=f"bias_a_{tag}")
    b = st.text_input("Sentence B", result.data["text_b"], key=f"bias_b_{tag}")
    cmp = _compare_cached(a, b, result.model_name)
    st.caption(result.data["summary"] if (a, b) == (result.data["text_a"], result.data["text_b"])
               else f"P(positive): A = {cmp['p_pos_a']:.2f}, B = {cmp['p_pos_b']:.2f}")
    fig_att, fig_attr = pair_figs(cmp)
    st.plotly_chart(fig_att, use_container_width=True)
    st.plotly_chart(fig_attr, use_container_width=True)
    with st.expander("How to read this"):
        st.write("Orange bars in rows A and B mark the words that differ between the two sentences. The bottom "
                 "row shows, for every shared word, how much its score changes when only that swap is made. "
                 "Attention shows where the model looks; attribution shows which words moved the sentiment "
                 "prediction when masked.")
