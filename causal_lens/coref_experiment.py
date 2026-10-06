"""Part 2, Step 5: does the impressive "it -> animal" attention actually matter?

For a masked sentence, find the heads where "it" attends most to its referent,
switch off the top-k, and measure how the model's preference for the correct
word over the wrong word changes (log P(correct) - log P(wrong) at [MASK]).
Control: switch off k RANDOM heads (20 draws). If the coreference heads change
the margin no more than random heads do, the attention looked meaningful but
was not used.

Usage:  python -m causal_lens.coref_experiment [bert-base-uncased]
"""
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from causal_lens.ablate import masked_heads
from shared.config import MAX_LENGTH
from shared.model_loader import load_mlm

ITEMS = [
    dict(name="animal", referent="animal", correct="tired", wrong="wide",
         template="The animal didn't cross the street because it was too [MASK]."),
    dict(name="trophy", referent="suitcase", correct="small", wrong="big",
         template="The trophy doesn't fit in the suitcase because it is too [MASK]."),
]
DATA = Path(__file__).resolve().parents[1] / "data"


def _clean(t):
    return t.lstrip("Ġ▁")


def _find(tokens, word, last=False):
    idx = [i for i, t in enumerate(tokens) if _clean(t) == word]
    if not idx:
        raise ValueError(f"'{word}' not found in {tokens}")
    return idx[-1] if last else idx[0]


def _word_id(tok, model_name, word):
    ids = tok.encode((" " if "roberta" in model_name else "") + word, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"'{word}' is not a single token for {model_name}")
    return ids[0]


@torch.no_grad()
def _margin(model, enc, pos, cid, wid, mask=None):
    with (masked_heads(model, mask) if mask is not None else nullcontext()):
        lp = torch.log_softmax(model(**enc).logits[0, pos], dim=-1)
    return float(lp[cid] - lp[wid])


@torch.no_grad()
def run(model_name="bert-base-uncased", ks=(1, 5, 10), n_random=20):
    tok, model = load_mlm(model_name)
    rows = []
    for item in ITEMS:
        text = item["template"].replace("[MASK]", tok.mask_token)
        enc = tok(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
        toks = tok.convert_ids_to_tokens(enc["input_ids"][0])
        it, ref = _find(toks, "it", last=True), _find(toks, item["referent"])
        pos = toks.index(tok.mask_token)
        cid, wid = _word_id(tok, model_name, item["correct"]), _word_id(tok, model_name, item["wrong"])
        att = torch.stack(model(**enc, output_attentions=True).attentions)[:, 0, :, it, ref].numpy()  # (L,H)
        L, H = att.shape
        base = _margin(model, enc, pos, cid, wid)
        for k in ks:
            top = np.argsort(-att.ravel())[:k]
            m = np.ones(L * H)
            m[top] = 0
            after = _margin(model, enc, pos, cid, wid, m.reshape(L, H))
            rnd = []
            for s in range(n_random):
                r = np.ones(L * H)
                r[np.random.default_rng(s).choice(L * H, k, replace=False)] = 0
                rnd.append(_margin(model, enc, pos, cid, wid, r.reshape(L, H)))
            rows.append(dict(model=model_name, item=item["name"], k=k, heads=[(int(i // H) + 1, int(i % H) + 1) for i in top],
                             margin_before=base, margin_after_coref_heads=after,
                             margin_after_random_mean=np.mean(rnd), margin_after_random_std=np.std(rnd)))
            print(f"{item['name']} k={k}: before {base:.2f} | coref heads off {after:.2f} | "
                  f"random off {np.mean(rnd):.2f} ± {np.std(rnd):.2f}", flush=True)
    df = pd.DataFrame(rows)
    DATA.mkdir(exist_ok=True)
    df.to_csv(DATA / f"coref_ablation_{model_name}.csv", index=False)
    return df


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "bert-base-uncased")
