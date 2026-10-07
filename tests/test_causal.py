import numpy as np
import pytest

from causal_lens.ablate import CLASSIFIERS, grid_shape, logits_with_mask, short_key
from shared.config import SENTIMENT_MODELS

KEY = "distilbert"
TEXTS = ["a gorgeous, moving film.", "dull and lifeless from start to finish."]


def test_short_key_handles_every_app_model_name():
    assert {short_key(m) for m in SENTIMENT_MODELS} == {"distilbert", "bert", "roberta"}
    assert short_key("bert-base-uncased") == "bert"
    assert short_key("distilbert-base-uncased") == "distilbert"
    assert short_key("roberta-base") == "roberta"
    assert set(CLASSIFIERS) == {"distilbert", "bert", "roberta"}


def test_all_ones_mask_matches_original_logits():
    L, H = grid_shape(KEY)
    base = logits_with_mask(TEXTS, None, KEY)
    masked = logits_with_mask(TEXTS, np.ones((L, H)), KEY)
    assert np.allclose(base, masked, atol=1e-6)


def test_removing_a_whole_layer_changes_logits():
    L, H = grid_shape(KEY)
    m = np.ones((L, H))
    m[0, :] = 0
    base = logits_with_mask(TEXTS, None, KEY)
    assert not np.allclose(base, logits_with_mask(TEXTS, m, KEY), atol=1e-4)


@pytest.mark.parametrize("key", ["distilbert", "bert", "roberta"])
def test_importance_matrix_shape(key):
    from causal_lens.importance import load_saved
    saved = load_saved(key)
    if saved is None:
        pytest.skip("run causal_lens.run_offline first")
    assert saved[0].shape == grid_shape(key)


def test_compute_runs_on_every_sentence_and_app_model():
    from causal_lens.lens import compute
    from shared.contracts import LensResult
    from shared.sentences import ALL_PROBES, BIAS_PROBES
    first_model = next(iter(SENTIMENT_MODELS))
    for p in ALL_PROBES + BIAS_PROBES:
        r = compute(p.text, first_model)
        assert isinstance(r, LensResult) and r.lens == "causal" and r.data["tokens"]
    for model_name in SENTIMENT_MODELS:
        r = compute(ALL_PROBES[0].text, model_name)
        assert r.data["importance"].shape == grid_shape(model_name)
