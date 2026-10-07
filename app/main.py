"""
Transformer Attention Visualizer - a multi-lens visualization framework.

One sentence in, five views of how the model reads it. Only the lens that
is open runs, so switching views never recomputes the others.

Run locally:  streamlit run app/main.py
"""

import html
import importlib
import importlib.util
import inspect
import sys
import threading
from pathlib import Path

# Make `shared`, `attention_lens`, ... importable when run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from attention_lens import extract, views
from shared import sentences
from shared.cache import cached
from shared.config import APP_CLASSIFIER, APP_MODEL, APP_MODEL_DETAIL, APP_MODEL_NAME
from shared.theme import apply_theme

st.set_page_config(
    page_title="Transformer Attention Visualizer",
    page_icon="🔍",
    layout="wide",
)
apply_theme(st)

# ------------------------------------------------------------ examples
# Short names for the curated sentences, in the order of shared.sentences.
EXAMPLE_NAMES = [
    "Pronoun, tired", "Pronoun, wide", "Trophy and suitcase",
    "Not bad", "Not good",
    "Bank, money", "Bank, river",
    "Mixed review", "Sarcasm",
    "Agreement", "Nested clause",
]
EXAMPLES = dict(zip(EXAMPLE_NAMES, sentences.ALL_PROBES))

LENSES = ["Attention", "Attribution", "Probing", "Bias", "Causal", "About"]

LENS_COPY = {
    "Attention": ("Which words does the model look at?",
                  "Every word spreads its attention across the sentence. Pick a word to see where it looks.",
                  f"{APP_MODEL_NAME}"),
    "Attribution": ("Which words changed the prediction?",
                    "Four methods score each word, then we delete words to test whether those scores hold up.",
                    f"{APP_MODEL_NAME} fine-tuned for sentiment (SST-2)"),
    "Probing": ("What does each layer know?",
                "Small classifiers try to read grammar and meaning out of each layer, with a control task to keep them honest.",
                f"{APP_MODEL_NAME}, with saved results for BERT and RoBERTa"),
    "Bias": ("Does the model prefer stereotypes?",
             "Benchmark scores from CrowS-Pairs, and a side-by-side view of this sentence with its gendered words swapped.",
             f"{APP_MODEL_NAME}"),
    "Causal": ("Which attention heads actually matter?",
               "Switch heads off and measure how much the prediction moves. This is the only lens that tests cause, not correlation.",
               f"{APP_MODEL_NAME} fine-tuned for sentiment (SST-2)"),
}


# ------------------------------------------------------------ state
def _use_example():
    name = st.session_state.get("example")
    if name:
        st.session_state.sentence = EXAMPLES[name].text


def _typed():
    st.session_state.example = None


def _keep_lens():
    # A segmented control can be clicked off; keep the last lens instead.
    if st.session_state.lens is None:
        st.session_state.lens = st.session_state.get("_last_lens", "Attention")
    st.session_state._last_lens = st.session_state.lens


if "sentence" not in st.session_state:
    st.session_state.sentence = sentences.ALL_PROBES[0].text
    st.session_state.example = EXAMPLE_NAMES[0]

# ------------------------------------------------------------ masthead
st.markdown(
    f"""
    <div class="tav-top">
      <div>
        <div class="tav-name">Transformer Attention Visualizer</div>
        <div class="tav-tag">See how a transformer reads one sentence, through five independent views.</div>
      </div>
      <div class="tav-model">{APP_MODEL_NAME}<span>{APP_MODEL_DETAIL}</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

text = st.text_input("Sentence", key="sentence", on_change=_typed,
                     placeholder="Type any English sentence", label_visibility="collapsed")
st.pills("Examples", EXAMPLE_NAMES, key="example", on_change=_use_example)

probe = next((p for p in sentences.ALL_PROBES if p.text == text), None)
if probe:
    st.markdown(f'<p class="tav-note"><b>What to look for.</b> {html.escape(probe.why)}</p>',
                unsafe_allow_html=True)

if not text.strip():
    st.info("Type a sentence above, or pick one of the examples.")
    st.stop()

st.write("")
lens = st.segmented_control("View", LENSES, default="Attention", key="lens",
                            on_change=_keep_lens, label_visibility="collapsed") or "Attention"


def lens_header(name: str):
    question, explainer, model = LENS_COPY[name]
    st.markdown(
        f'<div class="tav-lens-head"><h2>{question}</h2><p>{explainer}</p>'
        f'<div class="tav-meta">Model: {model}</div></div>',
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------ cached runs
# Two layers of caching: Streamlit's in-memory cache for this session, and
# shared.cache on disk so results survive restarts.
@st.cache_data(show_spinner="Reading the sentence…")
def run_attention(sentence: str):
    return cached("attention_lens", sentence, APP_MODEL,
                  lambda: extract.attention_matrices(sentence, APP_MODEL))


@st.cache_data(show_spinner="Running the lens. The first run on a new sentence takes a few seconds…")
def run_lens(package: str, sentence: str, model_name: str):
    return cached(package, sentence, model_name,
                  lambda: importlib.import_module(f"{package}.lens").compute(sentence, model_name))


@st.cache_resource
def _preload():
    """While the reader looks at the first view, load the heavier lenses in the background.

    Importing SHAP and Captum and loading the sentiment model takes several
    seconds. Doing it here means the Attribution and Causal views open
    quickly when the user gets to them.
    """
    def work():
        try:
            from shared.model_loader import load_classifier, load_mlm
            load_classifier(APP_CLASSIFIER)
            load_mlm(APP_MODEL)
            for package in ("attribution_lens", "causal_lens", "bias_lens", "probing_lens"):
                importlib.import_module(f"{package}.lens")
        except Exception:
            pass          # preloading is best-effort; the view loads normally if this fails
    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    return thread


_preload()


def show_lens(package: str, model_name: str, **render_kwargs):
    """Run one lens and draw it. Errors stay inside this view."""
    if (importlib.util.find_spec(package) is None
            or importlib.util.find_spec(f"{package}.lens") is None):
        st.info("This view is not available in this copy of the project.")
        return
    try:
        result = run_lens(package, text, model_name)
    except Exception as exc:
        st.error(f"This view could not run on this sentence: {exc}")
        return
    module = importlib.import_module(f"{package}.lens")
    accepted = inspect.signature(module.render).parameters
    kwargs = {k: v for k, v in render_kwargs.items() if k in accepted}
    try:
        module.render(result, **kwargs)
    except Exception as exc:
        st.error(f"This view could not be drawn: {exc}")


# ------------------------------------------------------------ attention
def attention_view():
    try:
        tokens, attn = run_attention(text)
    except Exception as exc:
        st.error(f"The model could not run: {exc}")
        return
    n_layers, n_heads = attn.shape[0], attn.shape[1]
    content = views.content_index(tokens)
    if not content:
        st.info("This sentence has no words to analyse.")
        return

    # Default focus: the probe's focus word, else the first word.
    default_word = probe.focus[0] if probe and probe.focus else None
    default_pos = next((i for i in content if tokens[i] == default_word), content[0])

    c1, c2, c3 = st.columns([2, 3, 2], gap="large")
    query = c1.selectbox(
        "Focus word", content, index=content.index(default_pos),
        format_func=lambda i: views.display(tokens[i]) + (f" ({i})" if [tokens[j] for j in content].count(tokens[i]) > 1 else ""),
        key=f"focus_{hash(text)}",
    )
    layer = c2.slider("Layer", 1, n_layers, n_layers, help="1 is closest to the input, the last layer closest to the output.")
    head_choice = c3.selectbox("Heads", ["All heads, averaged"] + [f"Head {h + 1}" for h in range(n_heads)],
                               help="Averaging is the honest default. Single heads are noisy.")
    head = None if head_choice.startswith("All") else int(head_choice.split()[-1]) - 1
    matrix = extract.select(attn, layer=layer - 1, head=head)

    word = views.display(tokens[query])
    st.markdown(f"### Where “{html.escape(word)}” looks")
    st.markdown(views.token_strip_html(tokens, matrix, query), unsafe_allow_html=True)

    idx, w = views.focus_weights(matrix, tokens, query)
    others = [o for o in range(len(idx)) if idx[o] != query]
    if others:
        top = max(others, key=lambda o: w[o])
        st.caption(f"In layer {layer}, “{word}” gives {w[top]:.0%} of its attention to "
                   f"“{views.display(tokens[idx[top]])}”. Darker words get more. "
                   f"[CLS] and [SEP] are left out and the rest rescaled to 100%.")

    left, right = st.columns([3, 1], gap="large")
    with left:
        st.plotly_chart(views.arc_figure(tokens, matrix, query), width="stretch")
    with right:
        st.markdown("**Strongest links**")
        st.markdown(views.ranked_html(tokens, matrix, query), unsafe_allow_html=True)

    st.markdown("### Go deeper")
    show_matrix = st.toggle("Full attention matrix for this layer")
    if show_matrix:
        st.plotly_chart(views.heatmap_figure(tokens, matrix), width="stretch")
        st.caption("Each row is a word; each cell is how much it attends to the word in that column. "
                   "Rows add up to 100%, including [CLS] and [SEP].")

    show_heads = st.toggle(f"Every head in layer {layer}")
    if show_heads:
        st.plotly_chart(views.head_grid_figure(tokens, attn[layer - 1]), width="stretch")
        st.caption("Heads specialise. Some look at the next or previous word, some park on [SEP], "
                   "a few track grammar.")

    show_rollout = st.toggle("Attention rollout across all layers")
    if show_rollout:
        st.plotly_chart(views.heatmap_figure(tokens, extract.attention_rollout(attn)), width="stretch")
        st.caption("One layer's attention ignores the mixing done by the layers below it. Rollout "
                   "multiplies the layers together to estimate how much each input word reaches each "
                   "output position (Abnar and Zuidema, 2020).")


# ------------------------------------------------------------ about
def about_view():
    st.markdown('<div class="tav-lens-head"><h2>About this project</h2>'
                '<p>A transformer never reads left to right. For every word it computes attention '
                'weights over every other word: a learned measure of which context it needs. '
                'Those numbers sit inside the model where nobody normally sees them.</p></div>',
                unsafe_allow_html=True)
    st.markdown(
        """
Plotting attention is the obvious first step, and on its own it is not enough: a word can receive a lot of
attention without changing the model's answer. So this tool looks at the same sentence five ways and lets
the views check each other.

| View | Question it answers | How |
| --- | --- | --- |
| Attention | Which words does the model look at? | Attention weights, head grid, attention rollout |
| Attribution | Which words changed the prediction, and is that true? | SHAP, Integrated Gradients, deletion tests |
| Probing | What does each layer know? | Linear probes with control tasks, layer similarity (CKA) |
| Bias | Does the model prefer stereotypes? | CrowS-Pairs benchmark, gender-swapped sentence pairs |
| Causal | Which attention heads actually matter? | Switching heads off and measuring the change |

**Model.** Everything runs live on DistilBERT, a smaller version of BERT with 6 layers instead of 12.
The saved experiments also cover BERT and RoBERTa, and the Probing view compares all three.

**A note on what attention proves.** High attention means information flowed along that link. It does not
mean the link caused the output. That gap is the reason the other four views exist.
        """
    )


# ------------------------------------------------------------ router
if lens == "About":
    about_view()
else:
    lens_header(lens)
    if lens == "Attention":
        attention_view()
    elif lens == "Attribution":
        show_lens("attribution_lens", APP_CLASSIFIER)
    elif lens == "Probing":
        show_lens("probing_lens", APP_MODEL, focus=probe.focus if probe else None)
    elif lens == "Bias":
        show_lens("bias_lens", APP_MODEL)
    elif lens == "Causal":
        show_lens("causal_lens", APP_CLASSIFIER)
