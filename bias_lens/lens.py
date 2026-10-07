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
from shared.sentences import BIAS_PAIRS

RESULTS = Path(__file__).resolve().parents[1] / "data" / "bias_results.csv"
OFFLINE_HINT = "Run `python -m bias_lens.run_offline` to produce the benchmark results."


def compute(text: str, model_name: str) -> LensResult:
    """Pair view for `text` vs its gender-swapped counterpart. No Streamlit."""
    other, swapped = swap_counterpart(text)
    # No gendered word to swap: show a built-in pair instead of an empty view.
    text_a, text_b = (text, other) if swapped else (BIAS_PAIRS[0][0].text, BIAS_PAIRS[0][1].text)
    cmp = compare_pair(text_a, text_b, model_name)
    notes = []
    if len(cmp["attr_diff"]):
        j = int(np.abs(cmp["attr_diff"]).argmax())
        wa = ", ".join(cmp["tokens_a"][i] for i in cmp["swapped_a"])
        wb = ", ".join(cmp["tokens_b"][i] for i in cmp["swapped_b"])
        lead = "" if swapped else "This sentence has no gendered word, so here is a built-in pair. "
        summary = (f"{lead}Swapping “{wa}” for “{wb}” moves the chance of a positive reading from "
                   f"{cmp['p_pos_a']:.0%} to {cmp['p_pos_b']:.0%}. The biggest shift is on "
                   f"“{cmp['shared_labels'][j]}” ({cmp['attr_diff'][j]:+.3f}).")
    else:
        summary = "The two sentences are identical, so there is no difference to show."
    if not RESULTS.exists():
        notes.append(OFFLINE_HINT)
    return LensResult(lens="bias", text=text, model_name=model_name,
                      data=dict(text_a=text_a, text_b=text_b,
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

    with st.expander("Content warning", expanded=False):
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
            st.plotly_chart(stereotype_fig(d), width="stretch")
            sig = d[(d.category != "overall") & d.significantly_above_50]["category"].tolist()
            st.caption("Categories where this model assigns significantly higher likelihood to the "
                       "stereotyped sentence (p < 0.05, uncorrected): " + (", ".join(sig) if sig else "none")
                       + ".")
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
    st.plotly_chart(fig_att, width="stretch")
    st.plotly_chart(fig_attr, width="stretch")
    with st.expander("How to read this"):
        st.write("Orange bars in rows A and B mark the words that differ between the two sentences. The bottom "
                 "row shows, for every shared word, how much its score changes when only that swap is made. "
                 "Attention shows where the model looks; attribution shows which words moved the sentiment "
                 "prediction when masked.")
