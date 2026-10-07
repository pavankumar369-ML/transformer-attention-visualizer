"""Tests for the Attribution & Faithfulness lens.

Set ATTR_TEST_MODEL (a key of shared.config.SENTIMENT_MODELS, or 'bert'/'roberta') to test another model.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from attribution_lens._compat import probe_sentences
from shared.config import CLASSIFIER_MODEL
from shared.contracts import LensResult
from attribution_lens.agreement import agreement_tables
from attribution_lens.attribute import (METHODS, compute_attributions, integrated_gradients,
                                        shap_scores)
from attribution_lens.faithfulness import drop_at_fraction
from attribution_lens.lens import compute
from attribution_lens.predict import encode_text, predict_proba

MODEL = os.environ.get("ATTR_TEST_MODEL", CLASSIFIER_MODEL)
DEMO = "The animal didn't cross the street because it was too tired."


@pytest.fixture(scope="module")
def bundle():
    """All four attributions for the demo sentence (computed once)."""
    return compute_attributions(DEMO, MODEL)


def test_predict_proba_sums_to_one():
    texts = [DEMO, "I love this movie.", "This was a terrible, boring film."]
    p = predict_proba(texts, MODEL)
    assert p.shape == (len(texts), 2)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-5)


def test_shap_and_ig_one_score_per_token():
    n = len(encode_text(DEMO, MODEL).tokens)
    assert len(shap_scores(DEMO, MODEL)) == n
    assert len(integrated_gradients(DEMO, MODEL).scores) == n


def test_ig_convergence_delta_below_threshold():
    assert integrated_gradients(DEMO, MODEL).delta < 0.05


def test_removing_zero_percent_gives_zero_drop(bundle):
    drop = drop_at_fraction(bundle.input_ids, bundle.scores["SHAP"], bundle.special,
                            0.0, MODEL, bundle.pred)
    assert drop == 0.0


def test_agreement_table_symmetric_unit_diagonal(bundle):
    res = agreement_tables(bundle.scores, bundle.special)
    for table in (res.spearman, res.top3):
        arr = table.to_numpy()
        assert arr.shape == (4, 4)
        assert list(table.index) == list(METHODS)
        assert np.allclose(arr, arr.T)
        assert np.allclose(np.diag(arr), 1.0)


@pytest.mark.slow
def test_compute_runs_on_all_probe_sentences():
    failures = []
    for text in probe_sentences():
        try:
            res = compute(text, MODEL)
            assert isinstance(res, LensResult) and res.lens == "attribution"
            n = len(res.data["tokens"])
            assert all(len(res.data["scores"][m]) == n for m in METHODS)
        except Exception as exc:  # collect so one failure shows all
            failures.append(f"{text!r}: {exc!r}")
    assert not failures, "\n".join(failures)
