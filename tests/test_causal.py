from pathlib import Path

import numpy as np
import pytest

from causal_lens.ablate import grid_shape, logits_with_mask

KEY = "distilbert"
TEXTS = ["a gorgeous, moving film.", "dull and lifeless from start to finish."]


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
    p = Path("data") / f"head_importance_{key}.npy"
    if not p.exists():
        pytest.skip("run causal_lens.run_offline first")
    assert np.load(p).shape == grid_shape(key)


def test_compute_runs_on_every_sentence():
    from causal_lens.lens import compute
    from shared.sentences import ALL_PROBES, BIAS_PROBES
    for p in ALL_PROBES + BIAS_PROBES:
        r = compute(p.text, KEY)
        assert r.tokens and r.scores.shape == grid_shape(KEY)
