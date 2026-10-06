"""Minimal-pair explorer: compare one sentence pair token by token.

Two signals per sentence:
  * attention received  - mean over all layers/heads/query tokens (base encoder)
  * occlusion attribution - drop in P(positive) when one token is masked (classifier)

TODO(swap-in): when attention_lens.extract and Member A's attribution functions
are on main, replace `attention_received` / `occlusion_attribution` with calls to
them. Never copy their code.
"""
import numpy as np
import torch

from bias_lens.pll import shared_positions
from causal_lens.ablate import CLASSIFIERS
from shared.config import MAX_LENGTH
from shared.model_loader import load_base_model, load_classifier, tokens_of

MODEL_TO_KEY = {"bert-base-uncased": "bert", "distilbert-base-uncased": "distilbert",
                "roberta-base": "roberta"}


@torch.no_grad()
def attention_received(text: str, model_name: str) -> np.ndarray:
    tok, model = load_base_model(model_name)
    enc = tok(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    att = torch.stack(model(**enc).attentions)          # (L, 1, H, T, T)
    return att[:, 0].mean(dim=(0, 1)).mean(dim=0).numpy()  # mean over L,H, then over queries


@torch.no_grad()
def occlusion_attribution(text: str, model_name: str):
    """Return (attribution per token incl. specials = 0, P(positive))."""
    key = MODEL_TO_KEY.get(model_name, model_name)
    tok, model = load_classifier(CLASSIFIERS[key])
    ids = tok(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)["input_ids"][0]
    T = len(ids)
    batch = ids.repeat(max(T - 1, 1), 1)                 # row 0 = original, row p = token p masked
    for p in range(1, T - 1):
        batch[p, p] = tok.mask_token_id
    p_pos = torch.softmax(model(input_ids=batch).logits, dim=-1)[:, 1].numpy()
    attr = np.zeros(T)
    attr[1:T - 1] = p_pos[0] - p_pos[1:T - 1]
    return attr, float(p_pos[0])


def compare_pair(sent_a: str, sent_b: str, model_name: str) -> dict:
    tok, _ = load_base_model(model_name)
    ta, tb = tokens_of(tok, sent_a), tokens_of(tok, sent_b)
    ids_a = tok(sent_a, truncation=True, max_length=MAX_LENGTH)["input_ids"]
    ids_b = tok(sent_b, truncation=True, max_length=MAX_LENGTH)["input_ids"]
    pa, pb = shared_positions(ids_a, ids_b)
    att_a, att_b = attention_received(sent_a, model_name), attention_received(sent_b, model_name)
    attr_a, p_a = occlusion_attribution(sent_a, model_name)
    attr_b, p_b = occlusion_attribution(sent_b, model_name)
    sa, sb = set(pa), set(pb)
    return dict(
        tokens_a=ta, tokens_b=tb, att_a=att_a, att_b=att_b, attr_a=attr_a, attr_b=attr_b,
        p_pos_a=p_a, p_pos_b=p_b, shared_a=pa, shared_b=pb,
        swapped_a=[i for i in range(1, len(ta) - 1) if i not in sa],
        swapped_b=[i for i in range(1, len(tb) - 1) if i not in sb],
        att_diff=att_b[pb] - att_a[pa] if pa else np.array([]),
        attr_diff=attr_b[pb] - attr_a[pa] if pa else np.array([]),
        shared_labels=[ta[i] for i in pa],
    )
