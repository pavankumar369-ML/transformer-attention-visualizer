# Transformer Attention Visualizer

### A multi-lens visualization framework for interpreting transformer language models

Type one English sentence and see how a transformer model reads it, through five
independent views: where it looks, which words drive its decision, what each layer
knows, whether it prefers stereotypes, and which of its parts it actually needs.

Course project for **ISWE310L Natural Language Processing**, VIT Vellore.

---

## Why five views instead of one

Inside a transformer, every word computes **attention weights** over every other word:
a learned measure of which context it needs. Plotting those weights is the usual first
step, and on its own it is misleading. A word can receive a lot of attention without
changing the model's answer (Jain and Wallace, 2019). So each view here asks a different
question about the same sentence, and the views check each other.

| View | Question it answers | Method |
| --- | --- | --- |
| **Attention** | Which words does the model look at? | Attention weights per layer and head, head grid, attention rollout |
| **Attribution** | Which words changed the prediction, and is that explanation true? | SHAP, Integrated Gradients, deletion (faithfulness) tests |
| **Probing** | What does each layer know? | Linear probes for POS, NER and sentiment, control tasks, CKA layer similarity |
| **Bias** | Does the model prefer stereotyped sentences? | CrowS-Pairs benchmark (1,508 pairs), gender-swapped sentence pairs |
| **Causal** | Which attention heads actually matter? | Switching heads off one at a time, progressive pruning |

A plain-language walkthrough of every view, chart and button is in
[`docs/PROJECT_GUIDE.md`](docs/PROJECT_GUIDE.md).

---

## Key results

Measured on saved experiments in `data/`. Full write-ups are in `docs/findings.md`
and `docs/findings_member_c.md`.

- **Layers build up meaning in order.** In BERT, part of speech is easiest to read at
  layer 6, named entities at layer 7 and sentiment at layer 9. DistilBERT and RoBERTa
  keep the same order.
- **The best-scoring layer is not the most honest one.** A control task shows that
  BERT's grammar signal is most selective at layer 9 (about 38 points above
  memorisation), three layers after its accuracy peak.
- **All three models prefer stereotyped sentences more often than chance.** On
  CrowS-Pairs, DistilBERT prefers the stereotype in 55.6% of pairs, BERT in 58.5% and
  RoBERTa in 59.9%; 50% would mean no preference. Religion is the most consistent
  category across models.
- **Most attention heads are redundant.** Accuracy stays within 2 points of the original
  until about 40% of DistilBERT's heads and 60% of BERT's and RoBERTa's are removed.
- **Some attention patterns are really used.** Switching off the five RoBERTa heads where
  "it" attends most to "animal" collapses the model's preference for the correct word
  (from +4.98 to +0.54), while five random heads barely move it (+4.82).

---

## Quick start

Requires Python 3.11 or newer. A CPU is enough.

```bash
git clone https://github.com/pavankumar369-ML/transformer-attention-visualizer.git
cd transformer-attention-visualizer

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python app/warmup.py               # optional, one time: pre-computes the example sentences
streamlit run app/main.py          # then open http://localhost:8501
```

The app runs live on **DistilBERT** (66M parameters, 6 layers, 12 heads), plus its
sentiment-tuned version for the Attribution and Causal views. The first launch downloads
both models, about 260 MB each. The warm-up step takes a few minutes and makes every
example sentence open instantly afterwards, even after a restart.

### Reproducing the experiments

The app reads saved results from `data/`. To regenerate them:

```bash
python -m probing_lens.run_offline                  # probes and CKA, three models
python -m bias_lens.run_offline                     # CrowS-Pairs scores
python -m causal_lens.run_offline                   # head importance and pruning
python -m attribution_lens.run_experiment --n-sst2 200
python -m attention_lens.report                     # attention numbers for the findings
```

### Tests

```bash
pytest -q
```

The full suite downloads and runs all three models, so it takes 20 to 30 minutes on a
laptop. GitHub Actions runs it on every pull request.

---

## Project structure

```
app/               Streamlit app, warm-up script
shared/            model loading, configuration, theme, disk cache, the example sentences
attention_lens/    attention extraction and views, rollout, findings report
attribution_lens/  SHAP, Integrated Gradients, faithfulness tests, agreement
probing_lens/      hidden states, probes, control tasks, CKA, word-meaning tracker
bias_lens/         pseudo-log-likelihood scoring, CrowS-Pairs benchmark, pair explorer
causal_lens/       head ablation, importance matrix, pruning, coreference experiment
data/              saved experiment results read by the app
docs/              project guide, findings, report drafts, demo script
tests/             one test file per view
```

Every view exposes the same two functions, `compute(text, model_name)` and
`render(result)`, and returns a `LensResult` (`shared/contracts.py`). The app finds each
view automatically, runs only the one that is open, and caches results in memory and on
disk.

---

## Team

| Member | Responsibility | Pull request |
| --- | --- | --- |
| **Pavan Kumar** ([@pavankumar369-ML](https://github.com/pavankumar369-ML)) | Project lead. Shared foundation, Attention view, app design, integration, speed, testing setup | [#2](https://github.com/pavankumar369-ML/transformer-attention-visualizer/pull/2), [#5](https://github.com/pavankumar369-ML/transformer-attention-visualizer/pull/5), [#6](https://github.com/pavankumar369-ML/transformer-attention-visualizer/pull/6), [#7](https://github.com/pavankumar369-ML/transformer-attention-visualizer/pull/7) |
| **Arun Eshwar Senthil Kumar** ([@Arun-Eshwar-Senthil-Kumar](https://github.com/Arun-Eshwar-Senthil-Kumar)) | Attribution and faithfulness view | [#4](https://github.com/pavankumar369-ML/transformer-attention-visualizer/pull/4) |
| **K S Sai Chandan** ([@kssaichandan](https://github.com/kssaichandan)) | Probing view: probes, control tasks, CKA, word tracker | [#1](https://github.com/pavankumar369-ML/transformer-attention-visualizer/pull/1) |
| **Yuvasri S M** ([@Yuvasri-S-M](https://github.com/Yuvasri-S-M)) | Bias and Causal views: CrowS-Pairs benchmark, head ablation | [#3](https://github.com/pavankumar369-ML/transformer-attention-visualizer/pull/3) |

---

## Limitations

- Attention, attribution and probing show **correlation**; only the Causal view tests
  cause, and only one head at a time.
- CrowS-Pairs measures a **likelihood preference**, not harm, and some of its pairs have
  known quality problems (Blodgett et al., 2021).
- Head importance is measured on 200 sentences, so differences of 1 to 2 points are noise.
- The live app uses one model. Comparisons with BERT and RoBERTa come from the saved
  experiments.

---

## Built with

PyTorch, Hugging Face Transformers, Streamlit, Plotly, SHAP, Captum, scikit-learn,
Hugging Face Datasets.

## References

- Vaswani et al. (2017). *Attention Is All You Need.*
- Clark et al. (2019). *What Does BERT Look At? An Analysis of BERT's Attention.*
- Jain and Wallace (2019). *Attention is not Explanation.*
- Abnar and Zuidema (2020). *Quantifying Attention Flow in Transformers.*
- Sundararajan et al. (2017). *Axiomatic Attribution for Deep Networks.*
- Lundberg and Lee (2017). *A Unified Approach to Interpreting Model Predictions.*
- DeYoung et al. (2020). *ERASER: A Benchmark to Evaluate Rationalized NLP Models.*
- Tenney et al. (2019). *BERT Rediscovers the Classical NLP Pipeline.*
- Hewitt and Liang (2019). *Designing and Interpreting Probes with Control Tasks.*
- Kornblith et al. (2019). *Similarity of Neural Network Representations Revisited.*
- Nangia et al. (2020). *CrowS-Pairs: A Challenge Dataset for Measuring Social Biases in Masked Language Models.*
- Blodgett et al. (2021). *Stereotyping Norwegian Salmon: An Inventory of Pitfalls in Fairness Benchmark Datasets.*
- Michel et al. (2019). *Are Sixteen Heads Really Better than One?*

## License

MIT
