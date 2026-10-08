# Project guide

This guide explains the Transformer Attention Visualizer for two kinds of reader. Each
section starts with **In plain words**, for anyone, and then **Under the hood**, for
readers who want the technical detail. Every screen, chart and control in the app is
covered.

---

## 1. The idea in one minute

**In plain words.** Language models such as the ones behind search engines and chatbots
are built from a design called a *transformer*. They are very good at language, but
nobody can easily see *why* they give the answer they give. This project is a set of
five "lenses" that look inside one of these models while it reads a sentence you type.
Each lens answers one question, and together they give an honest picture: what the
model looks at, what actually drives its decision, what it has learned at each stage,
whether it carries social stereotypes, and which of its parts it really needs.

**Under the hood.** The app runs DistilBERT, a 6-layer, 12-head transformer encoder
distilled from BERT. Each lens is an interpretability method from the research
literature: attention visualisation and rollout, feature attribution (SHAP and
Integrated Gradients) with faithfulness tests, linear probing with control tasks and
CKA, the CrowS-Pairs bias benchmark, and attention-head ablation. Saved experiments
extend three of the lenses to BERT (12 layers) and RoBERTa (12 layers).

### A worked example

Take the sentence *"The animal didn't cross the street because it was too tired."*
A person knows that "it" means the animal, because streets don't get tired. Change one
word, "tired" to "wide", and "it" now means the street. The model has to work this out
from context alone. The Attention lens shows which words "it" is looking at in each
case. The Causal lens then tests whether those looks matter, by switching off the parts
of the model responsible and checking whether it still gets the answer right.

---

## 2. Key terms

| Term | Plain meaning |
| --- | --- |
| Transformer | The neural network design behind modern language models. It reads all the words of a sentence at once. |
| Token | A piece of text the model works with. Usually a word; long words are split, so "unbelievably" may become "un", "##believ", "##ably". |
| [CLS] and [SEP] | Special tokens the model adds at the start and end of every sentence. |
| Attention | For each word, a set of percentages over every other word, saying how much it "looks at" each one. Each word's percentages add up to 100%. |
| Layer | One stage of processing. DistilBERT has 6, stacked in order; BERT and RoBERTa have 12. |
| Head | One of 12 parallel attention units inside each layer. Each head can learn to look for something different. |
| Hidden state | The list of 768 numbers that represents a word at a given layer. |
| Sentiment | Whether a sentence sounds positive or negative. |
| Probe | A very small classifier trained to read one kind of information out of a layer. |
| Benchmark | A standard test set that lets results be compared with published research. |

---

## 3. The main screen

**Sentence box.** The large box at the top. Type any English sentence, then press Enter.
Every view analyses whatever is in this box.

**Examples.** The row of rounded buttons under the box, such as *Pronoun, tired* or
*Sarcasm*. Clicking one puts a carefully chosen sentence into the box. Each example is
designed to expose one behaviour:

| Example | What it tests |
| --- | --- |
| Pronoun, tired / Pronoun, wide | Whether "it" is linked to the animal or the street; one word changes the answer |
| Trophy and suitcase | A classic hard pronoun puzzle (a Winograd schema) |
| Not bad / Not good | Whether "not" flips the meaning of the word after it |
| Bank, money / Bank, river | Whether the same word gets a different meaning from context |
| Mixed review | A sentence with one negative and one positive half |
| Sarcasm | Positive-sounding words with a negative meaning |
| Agreement | Whether "were" is linked to "keys", not the nearer word "cabinet" |
| Nested clause | A sentence with a clause inside a clause |

**What to look for.** A short note under the examples that tells you what each example
is designed to show. It disappears when you type your own sentence.

**View selector.** The row of six buttons: Attention, Attribution, Probing, Bias,
Causal and About. Only the selected view runs, which keeps the app fast.

**Model label.** The top right corner shows the model running live: DistilBERT,
66 million parameters, 6 layers, 12 heads.

**Speed.** The first time a new sentence is opened in a view, the model has to run, which
takes from one to several seconds. Results are then saved on disk, so the same sentence
opens instantly afterwards, even after the app is restarted.

---

## 4. Attention view

*Question: which words does the model look at?*

**In plain words.** You pick one word, and the sentence is coloured by how much
attention that word pays to every other word. Darker blue means more attention.

### Controls

| Control | What it does |
| --- | --- |
| **Focus word** | The word whose attention you want to see. For the pronoun examples it starts on "it". |
| **Layer** | Which stage of the model to look at, from 1 (closest to the raw words) to 6 (closest to the final answer). Attention patterns change a lot between layers. |
| **Heads** | "All heads, averaged" combines the 12 heads in the layer. Picking a single head shows just that one. The average is the honest default, because single heads are noisy and it is easy to cherry-pick one that looks impressive. |

### What you see

1. **The tinted sentence.** Every word is shaded by the focus word's attention to it.
   Hovering over a word shows the exact percentage. [CLS] and [SEP] are hidden here and
   the remaining percentages are rescaled to add up to 100%, because the special tokens
   absorb a lot of attention without carrying meaning.
2. **A one-line summary** naming the word the focus word looks at most in this layer,
   and its share of the attention.
3. **Arc chart.** Curved lines from the focus word to the four words it looks at most.
   Thicker, darker arcs mean more attention, and the percentage sits on top of each arc.
4. **Strongest links.** The top five words as a small bar list.

### Go deeper (switches)

| Switch | What appears |
| --- | --- |
| **Full attention matrix for this layer** | A grid where each row is a word and each column is a word it can look at. The colour of each square is the attention. This is the raw view, special tokens included. |
| **Every head in layer N** | Twelve small grids, one per head. Some heads always look at the next word, some at the previous word, some at [SEP]. This shows that heads specialise. |
| **Attention rollout across all layers** | One layer's attention ignores what the layers below already mixed together. Rollout multiplies all the layers to estimate how much each input word reaches each final position (Abnar and Zuidema, 2020). |

**Under the hood.** Attention weights come from the model's forward pass with
`output_attentions=True` and eager attention, giving an array of shape
(layers, heads, tokens, tokens). Rollout adds the identity matrix to account for the
residual connection, renormalises each row and multiplies the layers in order. All
charts are Plotly, built only when a switch is turned on.

---

## 5. Attribution view

*Question: which words changed the prediction, and can we trust that explanation?*

**In plain words.** This view uses a version of the model trained to decide whether a
sentence sounds positive or negative. It shows what the model decided, which words
pushed it towards that decision, and then tests those explanations by deleting the
words they say are important. If deleting them changes the answer a lot, the
explanation was right.

### What you see

1. **Prediction.** Positive or Negative, and how confident the model is. Sentences that
   are not reviews, like the pronoun examples, still get a prediction; the model simply
   picks whichever side the wording leans towards.
2. **SHAP row and Integrated Gradients row.** The sentence twice, each word shaded by
   one method. Blue means the word pushes towards the prediction; red means it pushes
   against it; darker means stronger. Hover for the exact score. When both rows
   highlight the same words, the explanation is more trustworthy.
3. **Faithfulness chart.** The horizontal axis is the share of words removed; the
   vertical axis is how much the model's confidence drops. Each method removes its own
   most important words first. A line that rises quickly means the method found the
   words that really matter. The dashed grey line removes words at random; a method
   close to it is no better than guessing. The number in brackets in the legend, AOPC,
   is the average drop, so higher is better.
4. **Agreement heatmap.** How similarly the four methods rank the words, from 1 (same
   ranking) to 0 (unrelated) to -1 (opposite). Low agreement between attention and SHAP
   is direct evidence that attention is not the same as explanation.

**Under the hood.** Model: `distilbert-base-uncased-finetuned-sst-2-english`. SHAP uses a
text masker with up to 300 evaluations. Integrated Gradients uses Captum's
`LayerIntegratedGradients` on the embedding layer, a `[PAD]` baseline and 50 steps; the
convergence delta is reported under "How to read this". The two attention baselines are
the last layer's [CLS] attention and attention rollout. Faithfulness follows ERASER
(DeYoung et al., 2020): comprehensiveness, sufficiency and AOPC, with ten random
orderings as the control. Agreement is Spearman rank correlation.

---

## 6. Probing view

*Question: what does each layer know?*

**In plain words.** The model processes a sentence in stages. This view asks what kind
of knowledge is available at each stage: grammar (is this word a noun or a verb?),
names (is this a person or a place?) and feeling (is this sentence positive?). It does
this by training a tiny "reader" on each layer and checking how well it does.

### What you see

1. **What each layer knows.** One line per task: part of speech, named entities and
   sentiment, with a dot on each task's best layer. The dashed line is a control task
   (see below).
2. **How much of the grammar score is real.** Bars showing grammar accuracy minus control
   accuracy for each layer. A tall bar means the layer genuinely encodes grammar, not
   just the identity of familiar words.
3. **Which layers do similar work.** A grid comparing every pair of layers; 1 means they
   hold the same information. The **Compare** options show each model on its own
   (DistilBERT, BERT, RoBERTa) or one model against another, for example BERT against
   DistilBERT.
4. **How one word's meaning changes going up the layers.** Pick a word from your
   sentence and see how much its representation changes at each layer. The
   **Word-sense pair** menu compares "bank" (money) with "bank" (river), and two other
   pairs, showing the layer where the model starts to tell the two meanings apart.

**Under the hood.** Hidden states are taken from every layer, one vector per word using
its first sub-word token. Probes are standardised logistic regression on frozen vectors:
part of speech and named entities from CoNLL-2003 (1,000 training and 250 test
sentences), sentiment from SST-2. The control task follows Hewitt and Liang (2019): each
word type gets a random but fixed label, and selectivity is real accuracy minus control
accuracy. Layer similarity is linear CKA (Kornblith et al., 2019). The probe curves and
CKA maps are computed offline for all three models; the word tracker runs live.

---

## 7. Bias view

*Question: does the model prefer stereotyped sentences?*

**In plain words.** CrowS-Pairs is a published test of 1,508 sentence pairs. In each
pair, one sentence expresses a stereotype and the other is the same sentence with the
group changed. If the model finds the stereotyped version more natural more than half the
time, it has absorbed that stereotype from its training text.

### What you see

1. **Content warning.** A closed note explaining that the benchmark contains offensive
   stereotypes by design. The app never shows the sentences themselves, only scores.
2. **Stereotype score by category.** One bar per category (age, gender, religion and so
   on) plus an overall bar. The centre line is 50%, meaning no preference. Bars to the
   right mean the model prefers the stereotype. The thin whisker is a 95% confidence
   range; dark bars are the ones whose range stays clear of 50%, so the preference is
   unlikely to be chance.
3. **Minimal-pair explorer.** Your sentence and the same sentence with gendered words
   swapped (he and she, his and her, man and woman, and so on), compared side by side.
   If your sentence has no gendered word, a built-in pair about a doctor is shown. You can
   edit both sentences. The charts show attention and attribution for each sentence, and
   a third row shows the difference on the words they share.

**Under the hood.** Each sentence is scored by pseudo-log-likelihood with the masked
language model: each shared word is hidden in turn and the log-probability of the true
word is added up. The model prefers the stereotype when that sentence scores higher. The
confidence interval comes from 1,000 bootstrap resamples, and significance from an exact
binomial test against 50%, with a Bonferroni correction across the nine categories in
the write-up. Scores are computed offline. The explorer uses attention and occlusion
attribution from the sentiment model.

---

## 8. Causal view

*Question: which attention heads actually matter?*

**In plain words.** The other views can only show what goes on together. This view
tests cause and effect: it switches parts of the model off and checks what breaks. If
switching a part off makes the model worse, that part was doing real work.

### What you see

1. **Summary line.** The model's chance that your sentence is positive, with all heads on
   and with the 10 most important heads switched off.
2. **Which heads matter.** A grid of every head: rows are layers, columns are heads. Blue
   means removing that head lowers accuracy; amber means removing it slightly raises it;
   white means no effect. Hover for the exact change in accuracy points.
3. **How many heads does the model really need.** The horizontal axis is the share of
   heads removed. The solid line removes the least important heads first; the dashed line
   removes them in random order. The caption says how many heads can go before accuracy
   falls more than 2 points below the original.
4. **Switch heads off yourself.** Choose heads by layer and number, for example L5-H3,
   and compare the chance of a positive reading before and after.

**Under the hood.** A head is removed by zeroing its slice of the attention output
before the output projection, using forward hooks. Importance is the accuracy change on
200 SST-2 validation sentences when a single head is removed. Pruning removes heads in
10% steps, compared with the mean of five random orders. A separate experiment
(`causal_lens/coref_experiment.py`) removes the heads where "it" attends most to its
referent and measures the model's preference for the correct word.

---

## 9. About view

A short explanation of the project, the table of the five views, and the note that
high attention does not prove cause.

---

## 10. How the project maps to the syllabus (CISWE310L)

| Module | Where it appears |
| --- | --- |
| 1. Introduction to NLP and text preprocessing | Sub-word tokenisation, special tokens, aligning sub-words back to words |
| 2. Language modelling and text classification | Masked language model scoring in the Bias view; sentiment classification in Attribution and Causal |
| 3. POS tagging and named entity recognition | Part-of-speech and named-entity probes in the Probing view |
| 4. Syntax analysis | Agreement and nested-clause examples; grammar selectivity per layer; syntax-tracking heads in the head grid |
| 5. Semantic analysis | Word sense ("bank"), pronoun resolution, the word-meaning tracker |
| 6. Machine translation | Not covered directly. The transformer and its attention mechanism were introduced for translation (Vaswani et al., 2017) |
| 7. Advanced NLP techniques | Transformers, attention, BERT-family models, interpretability methods |
| 8. Contemporary issues | Social bias measurement; explainability and whether explanations can be trusted |

---

## 11. Engineering notes

- **One contract for every view.** Each view has `compute(text, model_name)` and
  `render(result)` and returns a `LensResult`. The app finds views automatically, so team
  members never had to edit the main app file.
- **Shared foundation.** One model loader, one configuration file, one theme and one set
  of example sentences, so all five views analyse the same thing and look consistent.
- **Speed.** Only the open view runs; results are cached in memory and on disk; heavy
  libraries load in the background; a warm-up script pre-computes the examples.
- **Quality.** 50+ automated tests, run on every pull request by GitHub Actions. All work
  went through branches and reviewed pull requests.
