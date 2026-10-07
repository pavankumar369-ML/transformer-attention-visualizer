"""Attention views: fast checks, no model needed."""

import numpy as np

from attention_lens import views

TOKENS = ["[CLS]", "the", "animal", "was", "tir", "##ed", "because", "it", ".", "[SEP]"]


def _matrix(n, seed=0):
    m = np.random.default_rng(seed).random((n, n))
    return m / m.sum(axis=1, keepdims=True)


def test_focus_weights_drop_specials_and_sum_to_one():
    idx, w = views.focus_weights(_matrix(len(TOKENS)), TOKENS, query=7)
    assert 0 not in idx and len(TOKENS) - 1 not in idx
    assert np.isclose(w.sum(), 1.0)


def test_display_joins_wordpieces():
    assert views.display("##ed") == "ed" and views.display("it") == "it"


def test_strip_and_ranked_html_mark_the_focus_word():
    m = _matrix(len(TOKENS))
    strip = views.token_strip_html(TOKENS, m, 7)
    assert 'class="tav-tok q"' in strip and "[CLS]" not in strip
    ranked = views.ranked_html(TOKENS, m, 7, k=3)
    assert ranked.count('class="pct"') == 3


def test_figures_build():
    n = len(TOKENS)
    m = _matrix(n)
    assert len(views.arc_figure(TOKENS, m, 7).data) >= 2
    assert views.heatmap_figure(TOKENS, m).data[0].z.shape == (n, n)
    layer = np.stack([_matrix(n, s) for s in range(12)])
    assert len(views.head_grid_figure(TOKENS, layer).data) == 12
