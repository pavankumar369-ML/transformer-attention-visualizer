"""Pseudo-log-likelihood (PLL) scoring with a masked language model.

PLL(sentence) = sum over scored positions i of
                log P(true token_i | sentence with token_i masked)

We score only tokens the two sentences SHARE, so swapped words do not
change how many tokens each sentence contributes.
"""
from difflib import SequenceMatcher
from typing import List, Tuple

import torch

from shared.config import MAX_LENGTH
from shared.model_loader import load_mlm


def token_ids(tokenizer, sentence: str) -> List[int]:
    return tokenizer(sentence, truncation=True, max_length=MAX_LENGTH)["input_ids"]


def shared_positions(ids_a: List[int], ids_b: List[int]) -> Tuple[List[int], List[int]]:
    """Positions of tokens present in both sequences (aligned), excluding
    the special first/last tokens. Returns (positions_in_a, positions_in_b)."""
    sm = SequenceMatcher(a=ids_a, b=ids_b, autojunk=False)
    pa, pb = [], []
    for blk in sm.get_matching_blocks():
        for k in range(blk.size):
            i, j = blk.a + k, blk.b + k
            if 0 < i < len(ids_a) - 1 and 0 < j < len(ids_b) - 1:
                pa.append(i)
                pb.append(j)
    return pa, pb


@torch.no_grad()
def pll(sentence: str, shared_positions: List[int], model_name: str,
        batch_size: int = 32) -> float:
    """Sum of log P(true token | sentence with that token masked)."""
    if not shared_positions:
        return 0.0
    tok, model = load_mlm(model_name)
    ids = tok(sentence, truncation=True, max_length=MAX_LENGTH,
              return_tensors="pt")["input_ids"][0]
    total = 0.0
    for s in range(0, len(shared_positions), batch_size):
        chunk = shared_positions[s:s + batch_size]
        batch = ids.repeat(len(chunk), 1)          # one masked copy per position
        rows = torch.arange(len(chunk))
        cols = torch.tensor(chunk)
        batch[rows, cols] = tok.mask_token_id
        logits = model(input_ids=batch).logits     # (n, T, vocab)
        logp = torch.log_softmax(logits[rows, cols], dim=-1)
        total += logp[rows, ids[cols]].sum().item()
    return total


def score_pair(sent_a: str, sent_b: str, model_name: str) -> Tuple[float, float]:
    """PLL of both sentences over their shared tokens."""
    tok, _ = load_mlm(model_name)
    ids_a, ids_b = token_ids(tok, sent_a), token_ids(tok, sent_b)
    pa, pb = shared_positions(ids_a, ids_b)
    return pll(sent_a, pa, model_name), pll(sent_b, pb, model_name)
