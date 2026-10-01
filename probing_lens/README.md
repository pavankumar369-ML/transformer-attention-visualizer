# Layer Probing Lens

## What you are building
BERT has 12 layers. They do not all do the same job. This lens shows what
kind of linguistic information is recoverable at each depth.

## Method
1. Run a labelled dataset through the encoder with `output_hidden_states=True`.
2. For each layer, take the token (or `[CLS]`) embeddings.
3. Train a small logistic-regression classifier — the *probe* — on those
   frozen embeddings for two tasks:
   - **syntactic**: part-of-speech tag prediction
   - **semantic**: sentiment or entailment
4. Plot probe accuracy against layer index, one line per task.

## The result to aim for
The published pattern is that syntax is most recoverable in the middle
layers and semantics later. If your curves reproduce that, say so. If they
do not, say that too and discuss why — a negative result you can explain is
worth more in a viva than a plot you cannot defend.

## Deliverables
- `probe.py` — extract hidden states, train probes, return accuracy per layer
- `render.py` — the accuracy-vs-depth line chart
- Wire into the **Layer Probing** tab of `app/main.py`

## Dataset
Small is fine. A few thousand tokens from the `conll2003` POS tags and a
1–2k sample of SST-2 will run in minutes on CPU. Cache the extracted
hidden states to `data/` so you are not re-running BERT on every tweak.

## Rules
- Import the model from `shared.model_loader`, never load your own.
- Work on branch `feature/probing-lens`, open a PR into `main`.
