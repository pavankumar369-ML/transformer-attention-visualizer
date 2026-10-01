"""
Multi-Lens Visualization Framework - main app.

One sentence in, four explanations out. The shell and shared state live
here; each lens lives in its own package.

Run locally:  streamlit run app/main.py
"""

import sys
from pathlib import Path

# Make `shared`, `attention_lens`, ... importable when run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import streamlit as st

from attention_lens import extract, render
from shared import sentences
from shared.config import ACCENT, BASE_MODEL, BERT_LAYERS, NEUTRAL

st.set_page_config(
    page_title="Multi-Lens Visualization Framework",
    page_icon="🔍",
    layout="wide",
)

# ------------------------------------------------------------------ style
st.markdown(
    f"""
    <style>
      .block-container {{ padding-top: 2.2rem; max-width: 1180px; }}
      h1, h2, h3 {{ color: {NEUTRAL}; letter-spacing: -0.01em; }}
      .tav-sub {{ color: #6C6F80; font-size: 0.95rem; margin-top: -0.6rem; }}
      .tav-why {{
        background: #F7F7F9; border-left: 3px solid {ACCENT};
        padding: 0.65rem 0.9rem; border-radius: 4px;
        font-size: 0.9rem; color: #4A4E69; margin: 0.6rem 0 1.1rem 0;
      }}
      .stTabs [data-baseweb="tab"] {{ font-size: 0.95rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------ input
@st.cache_data(show_spinner="Running BERT… (first launch downloads ~440 MB, one time only)")
def run_attention(text: str):
    return extract.attention_matrices(text)


st.title("Multi-Lens Visualization Framework")
st.markdown(
    '<p class="tav-sub">Four ways of looking inside a transformer, '
    "on one sentence at a time.</p>",
    unsafe_allow_html=True,
)

with st.container():
    left, right = st.columns([3, 2])

    with right:
        labels = [f"{p.text}" for p in sentences.ALL_PROBES]
        picked = st.selectbox(
            "Example sentences",
            options=["(type your own)"] + labels,
            help="Curated probes - each one is designed to expose a "
                 "specific behaviour.",
        )

    with left:
        default = "" if picked == "(type your own)" else picked
        text = st.text_input(
            "Sentence to analyse",
            value=default or sentences.ALL_PROBES[0].text,
        )

probe = next((p for p in sentences.ALL_PROBES if p.text == text), None)
if probe:
    st.markdown(
        f'<div class="tav-why"><b>What to look for:</b> {probe.why}</div>',
        unsafe_allow_html=True,
    )

if not text.strip():
    st.info("Enter a sentence above to begin.")
    st.stop()

try:
    tokens, attn = run_attention(text)
except Exception as exc:  # surface errors in the UI, not just the terminal
    st.error(f"Model failed to run: {exc}")
    st.stop()

# ------------------------------------------------------------------- tabs
tab_attn, tab_shap, tab_probe, tab_bias, tab_about = st.tabs(
    ["Attention", "Token Importance", "Layer Probing", "Bias Analysis", "About"]
)

# ---------------------------------------------------------- ATTENTION TAB
with tab_attn:
    st.subheader("What is the model looking at?")

    c1, c2, c3 = st.columns(3)
    layer = c1.slider("Layer", 0, BERT_LAYERS - 1, BERT_LAYERS - 1)
    head_mode = c2.selectbox("Head", ["average of all heads"] +
                             [f"head {h}" for h in range(attn.shape[1])])
    head = None if head_mode.startswith("average") else int(head_mode.split()[-1])

    real_tokens = [t for t in tokens if t not in ("[CLS]", "[SEP]")]
    # Default to the probe's focus token (e.g. "it") so the demo opens on
    # the interesting view instead of on "the".
    preferred = probe.focus[0] if probe and probe.focus else None
    default_idx = real_tokens.index(preferred) if preferred in real_tokens else 0
    query = c3.selectbox("Focus token", real_tokens, index=default_idx)

    matrix = extract.select(attn, layer=layer, head=head)

    st.markdown("**Arc view** - where this token sends its attention")
    fig_arc, ax_arc = plt.subplots(figsize=(max(7, 0.75 * len(tokens)), 3.2))
    render.arc(matrix, tokens, query, top_k=4, ax=ax_arc)
    st.pyplot(fig_arc, use_container_width=True)
    plt.close(fig_arc)

    labels, weights = extract.token_focus(matrix, tokens, query)
    top_label, top_weight = labels[0], weights[0]
    st.markdown(
        f"In layer **{layer}**, `{query}` attends most strongly to "
        f"**`{top_label}`** ({top_weight:.0%} of its non-special attention)."
    )

    with st.expander("Full attention matrix"):
        size = max(4.5, 0.42 * len(tokens))
        fig_hm, ax_hm = plt.subplots(figsize=(size, size))
        render.heatmap(matrix, tokens, ax=ax_hm)
        st.pyplot(fig_hm, use_container_width=False)
        plt.close(fig_hm)

    with st.expander("All heads in this layer"):
        fig_grid = render.head_grid(attn, tokens, layer)
        st.pyplot(fig_grid, use_container_width=True)
        plt.close(fig_grid)

    with st.expander("Attention rollout (all layers combined)"):
        st.caption(
            "Last-layer attention alone is a weak explanation - information "
            "has already been mixed by earlier layers. Rollout multiplies "
            "the residual-adjusted matrices to approximate input-to-output "
            "influence."
        )
        roll = extract.attention_rollout(attn)
        size = max(4.5, 0.42 * len(tokens))
        fig_r, ax_r = plt.subplots(figsize=(size, size))
        render.heatmap(roll, tokens, title="Attention rollout", ax=ax_r)
        st.pyplot(fig_r, use_container_width=False)
        plt.close(fig_r)

# --------------------------------------------------------------- STUB TABS
with tab_shap:
    st.subheader("Which words actually drove the prediction?")
    st.info(
        "🚧 **In development.**\n\n"
        "Target: run the sentiment classifier, compute SHAP values per "
        "token, render them as an inline highlighted sentence plus a bar "
        "chart. The interesting comparison is attention vs. attribution - "
        "high attention does not always mean high influence on the output."
    )

with tab_probe:
    st.subheader("What does each layer know?")
    st.info(
        "🚧 **In development.**\n\n"
        "Target: extract hidden states from all 12 layers, train a small "
        "logistic-regression probe per layer on a POS task and a sentiment "
        "task, plot accuracy against layer depth. Expected result: syntax "
        "peaks in the middle layers, semantics later."
    )

with tab_bias:
    st.subheader("Does the model treat these sentences differently?")
    st.info(
        "🚧 **In development.**\n\n"
        "Target: run minimal pairs from `shared.sentences.BIAS_PAIRS` "
        "(identical sentences, one swapped word), diff the attention and "
        "attribution, and surface where the two runs disagree."
    )

with tab_about:
    st.subheader("About this project")
    st.markdown(
        f"""
A transformer does not read a sentence word by word. For every token it
computes **attention weights** over every other token - a learned measure of
what context it needs in order to represent that word. Those weights are
numbers inside the model that nobody normally sees.

This tool makes them visible, and then adds three more views on top, because
attention alone is an incomplete explanation:

| Lens | Question it answers |
| --- | --- |
| Attention | Which tokens does the model look at? |
| Token Importance | Which tokens actually changed the output? |
| Layer Probing | What kind of information lives at each depth? |
| Bias Analysis | Do those patterns shift when we swap one loaded word? |

Base model: `{BASE_MODEL}`. No fine-tuning - every view is computed from a
pretrained checkpoint at inference time.
        """
    )
