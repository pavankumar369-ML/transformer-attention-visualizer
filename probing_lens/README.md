# Layer Probing Lens

**Question:** what does each layer of the model know?

BERT stacks 12 layers. The published claim (Tenney et al., 2019) is that early
layers hold word form, middle layers grammar, and later layers meaning. This
lens tests that claim with controls, and shows live how one word's vector
changes as it moves up the layers.

Models: `bert-base-uncased` (12 layers), `distilbert-base-uncased` (6),
`roberta-base` (12). All plain encoders, loaded through
`shared/model_loader.py`. Layer 0 is the embedding layer, so BERT gives 13
sets of vectors.

## Files

```
hidden.py        hidden states per layer, one vector per word (first subword)
data.py          conll2003 + SST-2 subsamples (python -m probing_lens.data prints examples)
probe.py         logistic-regression probes, majority baseline, control task
similarity.py    linear CKA
trajectory.py    a word vs its previous layer; one word in two senses
render.py        the Plotly figures
lens.py          compute(text, model_name) -> LensResult, render(result)
run_offline.py   the full experiment -> data/probing_results.csv, data/cka_results.json
```

## Reproducing the results

```bash
python -m probing_lens.run_offline                       # all three models
python -m probing_lens.run_offline --model roberta-base  # one model
```

First run downloads the models and datasets, extracts vectors to
`data/vectors/` (float16, a few hundred MB per model, git-ignored) and then
trains the probes. Later runs reuse the cached vectors; `--force` re-extracts.
The two result files are small and committed, so the app works on a fresh
clone without running this.

## Method

| Task | Data | Unit | Train / test |
| --- | --- | --- | --- |
| Part of speech | conll2003 `pos_tags` | word | 1,000 / 250 sentences |
| Named entities | conll2003 `ner_tags` | word | same sentences |
| Sentiment | GLUE SST-2 | sentence (mean of word vectors) | 1,000 train / 250 validation |

SST-2's test split is unlabelled, so its test sentences come from validation.

- **Subwords.** A word's vector is the hidden state of its *first* subword.
  Words are found by character offset, so BERT and RoBERTa produce the same
  word list even though their tokenizers split text differently. That is
  what makes cross-model CKA possible.
- **Probe.** `StandardScaler` + `LogisticRegression(max_iter=1000, random_state=0)`,
  one per layer per task. Scaling keeps the probe linear but lets lbfgs
  converge: a few BERT dimensions are far larger than the rest.
- **Control task** (Hewitt & Liang, 2019). Every lower-cased word type gets a
  random POS tag drawn from the real tag distribution, fixed across train and
  test. *Selectivity = real accuracy − control accuracy.*
- **Baseline.** `probing_results.csv` also records the majority-class accuracy
  for each task. NER is ~80% "O", so raw NER accuracy needs that context.
- **CKA.** Linear CKA on the first 3,000 conll test words, layer × layer for
  each model and model × model for every pair.
- **Word tracker** (live). Cosine of a word with itself one layer earlier, and
  of one word in two senses ("bank": money vs river). The separation layer is
  the first layer where the pair has covered half of its total drop in
  similarity, which avoids picking an arbitrary threshold.

## Tests

`pytest -q tests/test_probing.py`: word alignment (one vector per word, and
that vector is the first subword's), shapes for BERT and DistilBERT, offline
and live extraction agree, CKA identity and scale invariance, control-label
consistency, random-vector probe ≈ majority baseline, and `compute()` on every
sentence in `shared/sentences.py`.
