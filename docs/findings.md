# Findings log

One entry per lens, minimum. This file is what turns a tool into a project —
in the report and in the viva, "here is what we built" is worth much less
than "here is what we found."

Format: sentence, what you expected, what happened, what you concluded.

---

## Attention lens

Numbers for the findings below come from `python -m attention_lens.report`
(DistilBERT by default; add `--model bert-base-uncased` for BERT).

### Finding 1 — coreference shifts on a single word
- **Probe:** the Winograd pair in `shared.sentences.COREFERENCE[0:2]`
- **Expected:** `it` attends to `animal` in the `tired` version, `street` in
  the `wide` version
- **Observed:** _fill in — layer, head, actual weights_
- **Conclusion:** _fill in_

### Finding 2 — head specialisation
- **Probe:** any sentence, head grid view, first layer vs last layer
- **Expected:** early heads look positional (diagonal / previous-token),
  later heads look more selective
- **Observed:** _fill in_
- **Conclusion:** _fill in_

### Finding 3 — `[CLS]` and `[SEP]` as attention sinks
- **Expected:** a large share of attention lands on special tokens,
  especially in later layers — heads with nothing useful to do "park" there
- **Observed:** _fill in_
- **Conclusion:** _fill in — this is a good honesty point: a lot of raw
  attention mass is not linguistically meaningful at all_

---

## Attribution lens

Numbers come from `python -m attribution_lens.run_experiment`, which writes
`data/attribution_results.csv` and `data/attribution_summary.csv`.

### Finding 1: does attention agree with attribution?
- **Expected:** low Spearman correlation between attention and SHAP / IG
  (Jain and Wallace, 2019).
- **Observed:** _fill in from the "Spearman with SHAP" column_
- **Conclusion:** _fill in_

### Finding 2: which explanation is most faithful?
- **Expected:** SHAP or IG first, attention lower, random last (AOPC).
- **Observed:** _fill in_
- **Conclusion:** _fill in_

### Finding 3: the sarcasm sentence
- **Probe:** "Wow, it broke in one day. Truly impressive engineering."
- **Observed:** _prediction, and the words SHAP and IG mark most strongly_
- **Conclusion:** _fill in_

---

## Layer probing lens

**Setup.** Linear probes (standardised logistic regression) on frozen hidden
states, one per layer. POS and NER: 1,000 train / 250 test conll2003
sentences (14,969 / 3,453 words, mirror `eriktks/conll2003`). Sentiment:
1,000 SST-2 train / 250 validation sentences, mean of word vectors. Control
task as in Hewitt & Liang (2019). Majority-class baselines: POS 18.5%,
NER 81.6%, sentiment 50.0%. Every number below is reproduced by
`python -m probing_lens.run_offline` and stored in `data/probing_results.csv`
and `data/cka_results.json`.

| Model | Best POS layer | Best NER layer | Best sentiment layer | Peak POS selectivity |
| --- | --- | --- | --- | --- |
| BERT | 6 (87.4%) | 7 (96.1%) | 9 (87.6%) | 37.9% (layer 9) |
| DistilBERT | 3 (88.2%) | 4 (95.9%) | 6 (79.2%) | 36.6% (layer 6) |
| RoBERTa | 3 (90.9%) | 8 (97.0%) | 8 (84.4%) | 41.2% (layer 8) |

### Finding 1: BERT reproduces the order of the pipeline, but the POS peak is flat
- **Expected:** grammar (POS) peaks earlier than meaning (NER, sentiment).
- **Observed:** in BERT the peaks are in that order: POS at layer 6, NER at 7,
  sentiment at 9. RoBERTa (POS 3, NER 8, sentiment 8) and DistilBERT
  (POS 3, NER 4, sentiment 6 of 6) show the same order. The POS curve is
  nearly flat, though. In BERT every layer from 2 to 9 scores between 86.2%
  and 87.4%, so "layer 6" is the top of a plateau, not a sharp peak.
  Sentiment rises the most with depth (69% at layer 0 to 88% at layer 9).
- **Caveat:** with 250 test sentences, one standard error on sentiment
  accuracy is about 2.3 points. BERT's jump at layer 9 (87.6% against 82–84%
  around it) should be read as "upper-middle layers", not "exactly layer 9".
- **Conclusion:** the ordering claim of Tenney et al. holds here. The
  stronger claim that grammar lives in one particular layer does not: POS is
  readable almost equally well from most of the network.

### Finding 2: accuracy and selectivity disagree
- **Observed:** BERT's embedding layer gets 81.5% POS accuracy, but the
  control task gets 70.6%, so selectivity is only 10.9 points. Most of the
  score at layer 0 comes from recognising words. With depth, POS accuracy
  barely moves (about 86–87%), while control accuracy falls steadily from
  70.6% to 47.8% as each word's vector mixes in its context. Selectivity
  therefore rises to 37.9% at layer 9, three layers after the accuracy peak.
  RoBERTa behaves the same way and is more selective at every layer
  (19.1% at layer 0, peaking at 41.2% at layer 8).
- **Conclusion:** the layer that scores best is not the layer that encodes
  grammar most selectively. Rising selectivity comes mainly from
  memorisation getting harder, not from POS getting easier. Without the
  control task we would have reported layer 6 and missed this.
  NER needs the same care: its baseline is 81.6% (most words are "O"), so
  the 96% peak is a gain of about 15 points, not of 96.

### Finding 3: where context starts to matter (word-sense pairs)
- **Probe:** `shared.sentences.AMBIGUITY` ("bank" money vs river). "bank" is
  word 5 in both sentences, so at layer 0 the two vectors are identical
  (cosine 1.00).
- **Observed (BERT):** cosine drops to 0.78 at layer 1 and has fallen half
  of its total drop by **layer 2**. It reaches its lowest point, 0.40, at
  layer 9, then rises slightly to 0.48 at layer 12. The two extra pairs in
  the tracker, "bat" and "spring", reach their half-way point at layer 4.
  DistilBERT separates by layers 1–3.
- **Observed (RoBERTa):** similarity only dips to about 0.80–0.90 in layers 1–4
  and then climbs back to 0.93–0.95 at layer 12, for all three pairs.
- **Conclusion:** in BERT, context starts changing the word from the first
  layer or two and keeps changing it into the upper-middle layers. RoBERTa's
  curve does **not** mean it ignores context. Its NER and sentiment probes
  are as good as BERT's. Its upper layers are highly anisotropic (almost all
  vectors point the same way, Ethayarajh 2019), so raw cosine stops telling
  senses apart. Comparing cosine values across models is unsafe, and the app
  flags this case in its caption.

### Finding 4: models organise their layers differently (CKA)
- **BERT on itself:** a smooth band. Neighbouring layers have CKA
  0.90–0.96, and similarity fades with distance. Layer 12 stands apart:
  0.77 with layer 11 and below 0.6 with layers 0–7. This fits the last layer
  being specialised for the masked-word objective.
- **RoBERTa on itself:** more uniform. Layers 4–11 all have CKA ≥ 0.80
  with each other, one broad block rather than a gradual drift.
- **BERT vs RoBERTa:** no diagonal. The highest value anywhere is 0.74, and
  BERT layers 4–9 match RoBERTa layers 3–9 about equally well (0.65–0.74).
  Even the embedding layers only reach 0.71, because the tokenisers differ
  (uncased WordPiece vs cased byte-level BPE). The two models end up with
  similar overall representations but do not build them in a matching
  layer-by-layer order.
- **BERT vs DistilBERT (bonus):** a clear diagonal at about 2:1.
  DistilBERT layer 1 matches BERT 1–2 (0.94), layer 3 matches BERT 5
  (0.91), layer 4 matches BERT 8 (0.89), and layer 6 matches BERT 12 (0.82).
  DistilBERT is initialised from every second BERT layer and trained to
  imitate it. This is the fair way to compare the two: by relative depth,
  not by layer number.

---

## Bias and Causal lenses

See `docs/findings_member_c.md`.
