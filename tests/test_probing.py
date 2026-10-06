"""Probing lens checks. Run with: pytest -q tests/test_probing.py

The word-alignment tests matter most: if the first-subword rule is off by
one, every probe result is still a plausible-looking number, just for the
wrong word.
"""

import numpy as np
import pytest
import torch

from probing_lens import lens
from probing_lens.hidden import batch_word_vectors, load, word_vectors
from probing_lens.probe import control_labels, majority_baseline, probe_accuracy
from probing_lens.similarity import linear_cka
from probing_lens.trajectory import pair_similarity, separation_layer, self_similarity
from shared import sentences
from shared.config import BASE_MODEL, LIGHT_MODEL
from shared.model_loader import encode


@pytest.mark.parametrize("text", ["unbelievable results", "unbelievably tokenizable"])
def test_one_vector_per_word_not_per_subword(text):
    # bert-base-uncased keeps "unbelievable" whole but splits the second
    # sentence into 8 subwords, so both cases are covered.
    words, vecs = word_vectors(text, BASE_MODEL)
    assert words == text.split()
    assert vecs.shape[1] == 2


def test_vector_is_first_subword():
    """The word's vector must be the hidden state of its FIRST subword."""
    text = "embeddings are unbelievably tokenizable"
    tokenizer, model = load(BASE_MODEL)
    words, vecs = word_vectors(text, BASE_MODEL)
    enc = encode(tokenizer, text)
    with torch.no_grad():
        states = model(**enc, output_hidden_states=True).hidden_states
    word_ids = enc.word_ids()
    first = [word_ids.index(w) for w in sorted({w for w in word_ids if w is not None})]
    assert len(enc["input_ids"][0]) - 2 > len(words)  # some words were split
    for layer in (0, 6, 12):
        expected = states[layer][0, first].numpy()
        assert np.allclose(vecs[layer], expected, atol=1e-5)


@pytest.mark.parametrize("model_name, n_layers", [(BASE_MODEL, 12), (LIGHT_MODEL, 6)])
def test_hidden_state_shape(model_name, n_layers):
    words, vecs = word_vectors("The keys to the cabinet were on the table.", model_name)
    assert vecs.shape == (n_layers + 1, len(words), 768)
    assert words[-1] == "."


def test_roberta_aligns_to_the_same_words():
    text = "The keys to the cabinet were on the table."
    bert_words, _ = word_vectors(text, BASE_MODEL)
    roberta_words, vecs = word_vectors(text, "roberta-base")
    assert roberta_words == bert_words
    assert vecs.shape == (13, len(bert_words), 768)


def test_batch_matches_live_extraction():
    """Offline (batched, pre-split) and live paths give the same vectors."""
    sents = [["the", "river", "bank", "flooded"], ["he", "sat", "down"]]
    batched = batch_word_vectors(sents, BASE_MODEL)
    for words, vecs in zip(sents, batched):
        live_words, live = word_vectors(" ".join(words), BASE_MODEL)
        assert live_words == words
        assert np.allclose(vecs, live, atol=1e-4)


def test_cka_identity_and_scale_invariance():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 32))
    assert linear_cka(X, X) == pytest.approx(1.0)
    assert linear_cka(X, 5 * X) == pytest.approx(1.0)
    assert linear_cka(X, rng.normal(size=(200, 32))) < 0.3


def test_control_labels_consistent_per_word():
    words = ["The", "cat", "sat", "on", "the", "mat", "cat", "the"]
    labels = [0, 1, 2, 3, 0, 1, 1, 0]
    ctrl = control_labels(words, labels)
    by_word = {}
    for w, c in zip(words, ctrl):
        by_word.setdefault(w.lower(), set()).add(int(c))
    assert all(len(v) == 1 for v in by_word.values())
    assert set(ctrl) <= set(labels)
    assert np.array_equal(ctrl, control_labels(words, labels))  # reproducible


def test_probe_on_random_vectors_scores_near_majority_baseline():
    rng = np.random.default_rng(0)
    y = rng.choice(3, size=1000, p=[0.7, 0.2, 0.1])
    X = rng.normal(size=(1000, 20))
    acc = probe_accuracy(X[:800], y[:800], X[800:], y[800:])
    assert abs(acc - majority_baseline(y[:800], y[800:])) < 0.08


def test_self_similarity_and_separation():
    words, vecs = word_vectors("He sat on the bank and watched the river flow.", BASE_MODEL)
    sim = self_similarity(vecs, words.index("bank"))
    assert sim.shape == (12,)
    assert np.all((sim > -1.0001) & (sim < 1.0001))

    pair = pair_similarity(sentences.AMBIGUITY[0].text, sentences.AMBIGUITY[1].text, "bank")
    assert pair.shape == (13,)
    assert pair[0] > 0.95                 # same word, same position, no context yet
    assert pair[-1] < pair[0]             # context pulls the senses apart
    assert separation_layer(np.array([1.0, 1.0, 1.0])) is None


@pytest.mark.parametrize(
    "text", [p.text for p in sentences.ALL_PROBES + sentences.BIAS_PROBES]
)
def test_compute_runs_on_every_probe_sentence(text):
    result = lens.compute(text, BASE_MODEL)
    assert result.lens == "probing"
    assert len(result.data["words"]) > 0
    assert result.data["self_similarity"].shape == (len(result.data["words"]), 12)
