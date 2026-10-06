"""
Multi-Lens Visualization Framework - main app.

One sentence in, five explanations out. The shell and shared state live
here; each lens lives in its own package.

Run locally:  streamlit run app/main.py
"""

import importlib
import importlib.util
import sys
from pathlib import Path

# Make `shared`, `attention_lens`, ... importable when run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import streamlit as st

from attention_lens import extract, render
from probing_lens import lens as probing_lens
from shared import sentences
from shared.config import (
    ACCENT,
    BASE_MODEL,
    BERT_LAYERS,
    BIAS_MODELS,
    NEUTRAL,
    PROBING_MODELS,
    SENTIMENT_MODELS,
)

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


@st.cache_data(show_spinner="Tracking words through the layers…")
def run_probing(text: str, model_name: str):
    return probing_lens.compute(text, model_name)


@st.cache_data(show_spinner="Running the lens…")
def run_lens(package: str, text: str, model_name: str):
    return importlib.import_module(f"{package}.lens").compute(text, model_name)


def lens_tab(package: str, models: dict, key: str, coming: str):
    """Draw a lens tab from its package's compute()/render().

    A lens appears automatically once `<package>/lens.py` exists, so lens
    authors never need to edit this file. Until then the tab shows `coming`.
    """
    # find_spec on "pkg.lens" raises if "pkg" itself is missing, so check both.
    if (importlib.util.find_spec(package) is None
            or importlib.util.find_spec(f"{package}.lens") is None):
        st.info(f"🚧 **In development.**\n\n{coming}")
        return
    model_name = st.selectbox(
        "Model", list(models), format_func=models.get, key=f"{key}_model"
    )
    try:
        result = run_lens(package, text, model_name)
    except Exception as exc:  # show the error in the tab, keep other tabs alive
        st.error(f"{package} failed: {exc}")
        return
    importlib.import_module(f"{package}.lens").render(result)


st.title("Multi-Lens Visualization Framework")
st.markdown(
    '<p class="tav-sub">Five ways of looking inside a transformer, '
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
tab_attn, tab_attr, tab_probe, tab_bias, tab_causal, tab_about = st.tabs(
    ["Attention", "Attribution", "Probing", "Bias", "Causal", "About"]
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
    st.pyplot(fig_arc, width="stretch")
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
        st.pyplot(fig_hm, width="content")
        plt.close(fig_hm)

    with st.expander("All heads in this layer"):
        fig_grid = render.head_grid(attn, tokens, layer)
        st.pyplot(fig_grid, width="stretch")
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
        st.pyplot(fig_r, width="content")
        plt.close(fig_r)

# --------------------------------------------------------------- STUB TABS
with tab_attr:
    st.subheader("Which words actually changed the answer?")
    lens_tab(
        "attribution_lens", SENTIMENT_MODELS, "attribution",
        "SHAP and Integrated Gradients per word, tested by deleting words "
        "and watching the prediction change.",
    )

with tab_probe:
    st.subheader("What does each layer know?")
    probe_model = st.selectbox(
        "Model", list(PROBING_MODELS), format_func=PROBING_MODELS.get,
        key="probing_model",
        help="Plain pretrained encoders, no fine-tuning. Switching model "
             "re-runs the live word tracker.",
    )
    try:
        probing_result = run_probing(text, probe_model)
    except Exception as exc:
        st.error(f"Probing lens failed: {exc}")
    else:
        probing_lens.render(probing_result, focus=probe.focus if probe else None)

with tab_bias:
    st.subheader("Does the model prefer stereotypes?")
    lens_tab(
        "bias_lens", BIAS_MODELS, "bias",
        "CrowS-Pairs benchmark scores per category, plus a minimal-pair "
        "explorer.",
    )

with tab_causal:
    st.subheader("Which attention heads actually matter?")
    lens_tab(
        "causal_lens", SENTIMENT_MODELS, "causal",
        "Switch attention heads off and measure what breaks.",
    )

with tab_about:
    st.subheader("About this project")
    st.markdown(
        f"""
A transformer does not read a sentence word by word. For every token it
computes **attention weights** over every other token - a learned measure of
what context it needs in order to represent that word. Those weights are
numbers inside the model that nobody normally sees.

This tool makes them visible, and then adds four more views on top, because
attention alone is an incomplete explanation:

| Lens | Question it answers |
| --- | --- |
| Attention | Which tokens does the model look at? |
| Attribution | Which tokens actually changed the output, and is that true? |
| Probing | What kind of information lives at each depth? |
| Bias | Does the model prefer stereotyped sentences? |
| Causal | Which attention heads actually matter? |

Base model: `{BASE_MODEL}`. No fine-tuning - every view is computed from a
pretrained checkpoint at inference time.
        """
    )
