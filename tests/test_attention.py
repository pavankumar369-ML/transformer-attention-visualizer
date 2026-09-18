"""Sanity checks. Run with: pytest -q

These are cheap but worth having: they catch the class of bug where a
reshape silently transposes the matrix and every downstream plot is wrong
in a way that still looks plausible.
"""

import numpy as np
import pytest

from attention_lens import extract
from shared import sentences

SENTENCE = sentences.COREFERENCE[0].text


@pytest.fixture(scope="module")
def run():
    return extract.attention_matrices(SENTENCE)


def test_shape_matches_tokens(run):
    tokens, attn = run
    n_layers, n_heads, s1, s2 = attn.shape
    assert s1 == s2 == len(tokens)
    assert n_layers == 12 and n_heads == 12


def test_rows_sum_to_one(run):
    _, attn = run
    sums = attn.sum(axis=-1)
    assert np.allclose(sums, 1.0, atol=1e-4)


def test_special_tokens_present(run):
    tokens, _ = run
    assert tokens[0] == "[CLS]" and tokens[-1] == "[SEP]"


def test_head_average_differs_from_single_head(run):
    _, attn = run
    avg = extract.select(attn, layer=-1, head=None)
    h0 = extract.select(attn, layer=-1, head=0)
    assert not np.allclose(avg, h0)


def test_token_focus_sorted_and_normalised(run):
    tokens, attn = run
    matrix = extract.select(attn, layer=-1)
    labels, weights = extract.token_focus(matrix, tokens, "it")
    assert "[CLS]" not in labels
    assert np.all(np.diff(weights) <= 1e-9)   # descending
    assert weights.sum() == pytest.approx(1.0, abs=1e-4)


def test_rollout_rows_sum_to_one(run):
    _, attn = run
    roll = extract.attention_rollout(attn)
    assert np.allclose(roll.sum(axis=-1), 1.0, atol=1e-3)
