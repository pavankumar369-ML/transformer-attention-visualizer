"""Bias lens: compute() + render() following the shared lens contract."""
from pathlib import Path

import numpy as np
import pandas as pd

from bias_lens.pair_diff import compare_pair
from bias_lens.swaps import swap_counterpart
from shared.contracts import LensResult
from shared.sentences import BIAS_PAIRS

RESULTS = Path(__file__).resolve().parents[1] / "data" / "bias_results.csv"


def compute(text: str, model_name: str) -> LensResult:
    """Pair view for `text` vs its gender-swapped counterpart. No plotting."""
    other, swapped = swap_counterpart(text)
    cmp = compare_pair(text, other if swapped else text, model_name)
    scores = np.zeros(len(cmp["tokens_a"]))
    scores[cmp["shared_a"]] = cmp["attr_diff"]
    if swapped and len(cmp["attr_diff"]):
        j = int(np.abs(cmp["attr_diff"]).argmax())
        wa = [cmp["tokens_a"][i] for i in cmp["swapped_a"]]
        wb = [cmp["tokens_b"][i] for i in cmp["swapped_b"]]
        summary = (f"Swapping {wa} → {wb} changes P(positive) from {cmp['p_pos_a']:.2f} to "
                   f"{cmp['p_pos_b']:.2f}; the largest attribution shift is on "
                   f"'{cmp['shared_labels'][j]}' ({cmp['attr_diff'][j]:+.3f}).")
    else:
        summary = "No gendered word to swap in this sentence, so the pair view shows no difference."
    return LensResult(tokens=cmp["tokens_a"], scores=scores, model_name=model_name, summary=summary,
                      extra=dict(text_a=text, text_b=other if swapped else text, swapped=swapped, cmp=cmp))


def render(result: LensResult) -> None:
    import streamlit as st
    from bias_lens.render import pair_figs, stereotype_fig

    with st.expander("⚠ Content warning", expanded=False):
        st.write("This tab uses CrowS-Pairs, a research benchmark that contains offensive stereotypes "
                 "by design, so we can measure whether a model prefers them. Sentences are never shown "
                 "here; only aggregate scores. The benchmark measures a statistical preference, not harm, "
                 "and later work (Blodgett et al., 2021) found quality problems in some pairs.")

    st.subheader("Stereotype score by category")
    if not RESULTS.exists():
        st.info("No benchmark results yet. Run `python -m bias_lens.run_offline`.")
    else:
        df = pd.read_csv(RESULTS)
        models = list(df["model"].unique())
        default = models.index(result.model_name) if result.model_name in models else 0
        sel = st.selectbox("Model", models, index=default, key="bias_model")
        d = df[df["model"] == sel]
        st.plotly_chart(stereotype_fig(d), use_container_width=True)
        sig = d[(d.category != "overall") & d.significantly_above_50]["category"].tolist()
        st.caption("Categories where the model assigns significantly higher likelihood to the "
                   "stereotyped sentence (p < 0.05): " + (", ".join(sig) if sig else "none"))
    with st.expander("How to read this"):
        st.write("Each bar is the % of sentence pairs where the model finds the stereotyped sentence more "
                 "likely than its counterpart. 50% means no preference. The whisker is a 95% confidence "
                 "interval; dark bars are those whose interval does not cross 50%.")

    st.subheader("Minimal-pair explorer")

    @st.cache_data(show_spinner="Comparing the pair…")
    def _cmp(a, b, m):
        return compare_pair(a, b, m)

    a = st.text_input("Sentence A", result.extra["text_a"], key="bias_a")
    b = st.text_input("Sentence B", result.extra["text_b"], key="bias_b")
    cmp = _cmp(a, b, result.model_name)
    st.caption(f"P(positive): A = {cmp['p_pos_a']:.2f}, B = {cmp['p_pos_b']:.2f}")
    fig_att, fig_attr = pair_figs(cmp)
    st.plotly_chart(fig_att, use_container_width=True)
    st.plotly_chart(fig_attr, use_container_width=True)
    with st.expander("How to read this"):
        st.write("Orange bars in rows A and B mark the words that differ between the two sentences. The bottom "
                 "row shows, for every shared word, how much its score changes when only that swap is made. "
                 "Attention shows where the model looks; attribution shows which words moved the sentiment "
                 "prediction when masked.")
