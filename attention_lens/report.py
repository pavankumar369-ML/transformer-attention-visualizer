"""
Attention lens - the numbers behind the findings.

Prints three tables for docs/findings.md and the presentation:

  1. Coreference: per layer, how much "it" attends to "animal" and to
     "street" in the tired/wide sentence pair (heads averaged, [CLS]/[SEP]
     left out and the rest rescaled, exactly as the app shows it), plus the
     single head that separates the pair best.
  2. Attention sinks: per layer, the share of all attention that lands on
     [CLS] and [SEP].
  3. Head behaviour: per layer, how many heads mostly look at the previous
     or next word, and the average focus (entropy) of the heads.

Run from the project folder:
    python -m attention_lens.report                      # DistilBERT, the app's model
    python -m attention_lens.report --model bert-base-uncased
Takes a few seconds once the model is cached.
"""

import argparse

import numpy as np

from attention_lens import extract, views
from shared.config import APP_MODEL

TIRED = "The animal didn't cross the street because it was too tired."
WIDE = "The animal didn't cross the street because it was too wide."


def _pos(tokens, word):
    return tokens.index(word)


def coreference_table(model_name: str):
    """Per layer: it->animal and it->street, for both sentences."""
    rows = []
    t_tok, t_att = extract.attention_matrices(TIRED, model_name)
    w_tok, w_att = extract.attention_matrices(WIDE, model_name)
    for layer in range(t_att.shape[0]):
        out = {"layer": layer + 1}
        for name, tok, att in (("tired", t_tok, t_att), ("wide", w_tok, w_att)):
            m = extract.select(att, layer=layer)
            idx, w = views.focus_weights(m, tok, _pos(tok, "it"))
            share = dict(zip(idx, w))
            out[f"{name}_animal"] = share[_pos(tok, "animal")]
            out[f"{name}_street"] = share[_pos(tok, "street")]
        rows.append(out)

    # Best single head: largest (animal - street) in "tired" plus (street - animal) in "wide".
    best = None
    for layer in range(t_att.shape[0]):
        for head in range(t_att.shape[1]):
            score = 0.0
            for tok, att, sign in ((t_tok, t_att, 1), (w_tok, w_att, -1)):
                idx, w = views.focus_weights(att[layer, head], tok, _pos(tok, "it"))
                share = dict(zip(idx, w))
                score += sign * (share[_pos(tok, "animal")] - share[_pos(tok, "street")])
            if best is None or score > best[0]:
                best = (score, layer + 1, head + 1)
    return rows, best


def sink_table(model_name: str):
    """Per layer: mean share of attention that lands on special tokens."""
    tok, att = extract.attention_matrices(TIRED, model_name)
    special = [i for i, t in enumerate(tok) if t in views.SPECIAL]
    return [(layer + 1, float(att[layer][:, :, special].sum(-1).mean())) for layer in range(att.shape[0])]


def head_behaviour(model_name: str):
    """Per layer: heads that mostly attend to the previous / next word, and mean entropy."""
    tok, att = extract.attention_matrices(TIRED, model_name)
    n = len(tok)
    out = []
    for layer in range(att.shape[0]):
        prev = nxt = 0
        for head in range(att.shape[1]):
            m = att[layer, head]
            prev += np.mean([m[i, i - 1] for i in range(1, n)]) > 0.5
            nxt += np.mean([m[i, i + 1] for i in range(n - 1)]) > 0.5
        out.append((layer + 1, int(prev), int(nxt), float(extract.head_entropy(att, layer).mean())))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=APP_MODEL)
    model = ap.parse_args().model

    rows, best = coreference_table(model)
    print(f"\nModel: {model}")
    print("\n1. Where does 'it' look? (share of its attention, heads averaged)")
    print("layer | tired: animal  street | wide: animal  street")
    for r in rows:
        print(f"{r['layer']:>5} | {r['tired_animal']:>13.0%} {r['tired_street']:>7.0%} |"
              f" {r['wide_animal']:>11.0%} {r['wide_street']:>7.0%}")
    print(f"Best single head for the pair: layer {best[1]}, head {best[2]} (separation score {best[0]:.2f})")

    print("\n2. Share of all attention landing on [CLS] and [SEP]")
    for layer, share in sink_table(model):
        print(f"layer {layer:>2}: {share:.0%}")

    print("\n3. Head behaviour (heads giving most of their attention to the previous / next word)")
    print("layer | previous-word heads | next-word heads | mean entropy (lower = more focused)")
    for layer, prev, nxt, ent in head_behaviour(model):
        print(f"{layer:>5} | {prev:>19} | {nxt:>15} | {ent:.2f}")


if __name__ == "__main__":
    main()
