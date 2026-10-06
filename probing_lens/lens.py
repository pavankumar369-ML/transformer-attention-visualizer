"""
Probing Lens - what the app calls.

  compute(text, model_name) -> LensResult   numbers only, no Streamlit
  render(result)                            draws the Probing tab

Probe curves and CKA maps are too slow to train on every click, so they are
read from files written by run_offline.py. The word tracker is cheap and
runs live on the user's sentence.
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from probing_lens import render as plots
from probing_lens.hidden import word_vectors
from probing_lens.trajectory import (
    WORD_SENSE_PAIRS,
    pair_similarity,
    self_similarity,
    separation_layer,
)
from shared.config import BASE_MODEL, PROBING_MODELS
from shared.contracts import LensResult

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_CSV = DATA_DIR / "probing_results.csv"
CKA_JSON = DATA_DIR / "cka_results.json"

OFFLINE_HINT = "Run `python -m probing_lens.run_offline` to produce it."


def display_name(model_name: str) -> str:
    return PROBING_MODELS.get(model_name, model_name)


def load_probe_results() -> Optional[pd.DataFrame]:
    if not RESULTS_CSV.exists():
        return None
    return pd.read_csv(RESULTS_CSV)


def load_cka() -> Dict[str, np.ndarray]:
    """{"modelA|modelB": matrix}; rows are layers of A, columns of B."""
    if not CKA_JSON.exists():
        return {}
    raw = json.loads(CKA_JSON.read_text())
    return {k: np.array(v) for k, v in raw.items()}


@lru_cache(maxsize=8)
def _pairs(model_name: str) -> List[dict]:
    """The built-in word-sense pairs. Independent of the user's sentence."""
    out = []
    for text_a, text_b, word, sense_a, sense_b in WORD_SENSE_PAIRS:
        sim = pair_similarity(text_a, text_b, word, model_name)
        out.append(dict(
            word=word, text_a=text_a, text_b=text_b, senses=[sense_a, sense_b],
            similarity=sim, separation=separation_layer(sim),
        ))
    return out


def compute(text: str, model_name: str = BASE_MODEL) -> LensResult:
    words, vectors = word_vectors(text, model_name)
    n_layers = vectors.shape[0] - 1
    self_sim = (
        np.stack([self_similarity(vectors, i) for i in range(len(words))])
        if words else np.zeros((0, n_layers))
    )

    notes = []
    results = load_probe_results()
    if results is not None:
        results = results[results.model == model_name]
    if results is None or results.empty:
        notes.append(f"No saved probe results for {display_name(model_name)}. {OFFLINE_HINT}")
        results = None

    cka = load_cka()
    if not cka:
        notes.append(f"No saved CKA maps. {OFFLINE_HINT}")

    return LensResult(
        lens="probing",
        text=text,
        model_name=model_name,
        data=dict(
            words=words,
            n_layers=n_layers,
            self_similarity=self_sim,   # (n_words, n_layers)
            probe_results=results,      # DataFrame or None
            cka=cka,
            pairs=_pairs(model_name),
        ),
        notes=notes,
    )


# ------------------------------------------------------------------ render
def _how_to_read(text: str):
    import streamlit as st
    with st.expander("How to read this"):
        st.markdown(text)


def _cka_label(key: str) -> str:
    a, b = key.split("|")
    if a == b:
        return f"{display_name(a)} layers"
    return f"{display_name(a)} vs {display_name(b)}"


def render(result: LensResult, focus: Optional[List[str]] = None):
    """Draw the four panels. focus = the probe's focus words, if any."""
    import streamlit as st

    d = result.data
    name = display_name(result.model_name)
    for note in result.notes:
        st.info(note)

    # ---------------------------------------------------- 1. probe curves
    st.markdown("#### 1 · What each layer knows")
    df = d["probe_results"]
    if df is not None:
        best = {t: int(g.loc[g.accuracy.idxmax(), "layer"]) for t, g in df.groupby("task")}
        st.markdown(
            f"In **{name}**, part of speech is easiest to read off **layer "
            f"{best.get('pos', '–')}**, named entities **layer {best.get('ner', '–')}**, "
            f"sentiment **layer {best.get('sentiment', '–')}**."
        )
        st.plotly_chart(plots.probe_curves(df))
        baselines = (
            df.groupby("task").baseline.first() if "baseline" in df else pd.Series(dtype=float)
        )
        base_txt = ", ".join(
            f"{plots.TASK_LABELS[t]} {baselines[t]:.0%}" for t in ("pos", "ner", "sentiment")
            if t in baselines
        )
        _how_to_read(
            "For every layer we froze the model and trained a tiny linear classifier "
            "(a *probe*) to predict a property from that layer's word vectors. "
            "Higher = the information is easier to read at that depth.\n\n"
            "- **Solid lines** are real tasks; the dot marks each task's best layer.\n"
            "- **Dashed line** is the *control task*: every word is given a random "
            "but fixed tag. A probe can only score well on it by memorising words, "
            "so it shows how much of the POS score memorisation alone could buy.\n"
            + (f"- Always guessing the most common label would score: {base_txt}. "
               "NER is mostly the 'not an entity' tag, so its line starts high.\n"
               if base_txt else "")
            + "- DistilBERT has 6 layers, not 12. Compare models by *relative* depth "
            "(e.g. DistilBERT layer 3 ≈ BERT layer 6), not by layer number."
        )

        # ------------------------------------------------ 2. selectivity
        st.markdown("#### 2 · How much of the grammar score is real?")
        st.plotly_chart(plots.selectivity_bars(df))
        st.caption("Taller bar = the layer really encodes grammar, not just word identity.")
        _how_to_read(
            "Each bar is *POS accuracy − control accuracy* (Hewitt & Liang, 2019). "
            "A layer can have high accuracy but low selectivity: then the probe is "
            "mostly recognising words it has seen, not reading grammatical structure."
        )

    # ---------------------------------------------------------- 3. CKA
    st.markdown("#### 3 · Which layers do similar work?")
    cka = d["cka"]
    if cka:
        keys = sorted(cka, key=lambda k: (k.split("|")[0] != k.split("|")[1], k))
        own = f"{result.model_name}|{result.model_name}"
        choice = st.radio(
            "Compare", keys, index=keys.index(own) if own in keys else 0,
            format_func=_cka_label, horizontal=True, key="probing_cka",
        )
        a, b = choice.split("|")
        st.plotly_chart(
            plots.cka_heatmap(cka[choice], display_name(a), display_name(b)),
        )
        _how_to_read(
            "Each cell compares two layers using CKA: run the same few thousand words "
            "through both, and ask how similar the two sets of vectors are "
            "(1 = same information, arranged the same way; 0 = unrelated). "
            "Warm square blocks along the diagonal are groups of neighbouring layers "
            "doing similar work.\n\n"
            "In a *model vs model* map, a bright diagonal means the two models build "
            "their representations in the same order; a bright cell off the diagonal "
            "means one model reaches a stage earlier than the other."
        )

    # -------------------------------------------------- 4. word tracker
    st.markdown("#### 4 · How one word's meaning changes as it goes up (live)")
    words = d["words"]
    if words:
        lowered = [w.lower() for w in words]
        default = next((lowered.index(f.lower()) for f in (focus or [])
                        if f.lower() in lowered), 0)
        idx = st.selectbox(
            "Word from your sentence", range(len(words)), index=default,
            format_func=lambda i: f"{words[i]}  (word {i + 1})",
            key=f"probing_word|{result.text}",  # fresh default per sentence
        )
        st.plotly_chart(
            plots.self_similarity_line(d["self_similarity"][idx], words[idx]),
        )
        sims = d["self_similarity"][idx]
        st.caption(
            f"'{words[idx]}' changes most going into layer {int(np.argmin(sims)) + 1} "
            f"(similarity {sims.min():.2f} with the layer before)."
        )

    pairs = d["pairs"]
    pair_default = next((i for i, p in enumerate(pairs) if p["word"] in
                         [w.lower() for w in words]), 0)
    p_idx = st.selectbox(
        "Word-sense pair", range(len(pairs)), index=pair_default,
        format_func=lambda i: f"'{pairs[i]['word']}': {pairs[i]['senses'][0]} vs "
                              f"{pairs[i]['senses'][1]}",
        key="probing_pair",
    )
    pair = pairs[p_idx]
    st.markdown(
        f"> A: *{pair['text_a']}*  \n> B: *{pair['text_b']}*"
    )
    st.plotly_chart(
        plots.pair_similarity_line(pair["similarity"], pair["word"], pair["senses"],
                                   pair["separation"]),
    )
    sim = pair["similarity"]
    if pair["separation"] is not None:
        caption = (
            f"The two '{pair['word']}'s start almost identical and have drifted half-way "
            f"apart by layer {pair['separation']}: that is where {name} starts using "
            "context to tell the senses apart."
        )
        # Upper layers can pull every vector back together (anisotropy).
        if sim[-1] - sim.min() > (sim[0] - sim.min()) / 2:
            caption += (
                f" They drift back together after layer {int(np.argmin(sim))}, though: "
                f"in {name}'s upper layers most vectors point the same way, so raw "
                "cosine stops separating the senses."
            )
        st.caption(caption)
    else:
        st.caption(f"The two '{pair['word']}'s never clearly separate in {name}.")
    _how_to_read(
        "**Top chart:** cosine similarity between the chosen word's vector and its own "
        "vector one layer earlier. 1 = the layer left it unchanged; dips show the layers "
        "that rewrite it the most.\n\n"
        "**Bottom chart:** the same word in two sentences with different meanings. At "
        "layer 0 the model has only seen the word itself, so the two vectors are nearly "
        "identical. As layers mix in context they drift apart. The dotted line marks the "
        "layer where they have covered half of that drift.\n\n"
        "Read the *shape*, not the absolute numbers: in upper layers even unrelated "
        "words have fairly high cosine similarity (Ethayarajh, 2019)."
    )
