from scipy.stats import binomtest

from bias_lens.pll import score_pair, shared_positions, token_ids
from shared.model_loader import load_mlm

MODEL = "distilbert-base-uncased"  # smallest, fastest for tests


def test_binomial_p_is_one_for_50_percent():
    assert binomtest(754, 1508, 0.5).pvalue > 0.99


def test_identical_sentences_identical_pll():
    s = "The doctor finished his shift and went home."
    a, b = score_pair(s, s, MODEL)
    assert a == b


def test_swapping_pair_flips_preference():
    s1 = "The doctor finished his shift and went home."
    s2 = "The doctor finished her shift and went home."
    a, b = score_pair(s1, s2, MODEL)
    c, d = score_pair(s2, s1, MODEL)
    assert a != b and (a > b) == (d > c)


def test_shared_positions_skip_swapped_word():
    tok, _ = load_mlm(MODEL)
    i1 = token_ids(tok, "The doctor finished his shift.")
    i2 = token_ids(tok, "The doctor finished her shift.")
    pa, pb = shared_positions(i1, i2)
    assert len(pa) == len(pb) == len(i1) - 3  # minus [CLS], [SEP], 'his'/'her'


def test_swap_counterpart_pure():
    from bias_lens.swaps import swap_counterpart
    assert swap_counterpart("The doctor finished his shift and went home.") == (
        "The doctor finished her shift and went home.", True)
    assert swap_counterpart("The food was not bad at all.") == (None, False)


def test_compute_runs_on_every_sentence():
    from bias_lens.lens import compute
    from shared.sentences import ALL_PROBES, BIAS_PROBES
    for p in ALL_PROBES + BIAS_PROBES:
        r = compute(p.text, MODEL)
        assert r.tokens and len(r.scores) == len(r.tokens)
