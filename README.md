# Transformer Attention Visualizer

### A Multi-Lens Visualization Framework for Interpreting Transformer Language Models

Type a sentence. See four different explanations of how BERT understood it.

> *"The animal didn't cross the street because **it** was too tired."*
> → `it` attends to **animal**
>
> *"The animal didn't cross the street because **it** was too wide."*
> → `it` attends to **street**
>
> One word changed. The model's internal wiring changed with it. This tool
> shows you that happening.

---

## Why this exists

A transformer does not read left to right. For every token it computes
**attention weights** over every other token — a learned score for how much
context each word needs. Those weights are real numbers inside the model that
nobody normally sees.

Plotting them is the obvious first move, and it is also not enough: a token
can receive high attention without influencing the output at all. So this
project takes four passes at the same question instead of one.

| Lens | Question | Method |
| --- | --- | --- |
| **Attention** | What is the model looking at? | Raw attention weights + attention rollout |
| **Token Importance** | What actually changed the answer? | SHAP attribution over a classifier |
| **Layer Probing** | What does each layer know? | Logistic probes on frozen hidden states |
| **Bias Analysis** | Does any of this shift on a loaded word? | Minimal-pair diffing |

---

## Screenshots

<!-- Replace once the app is running. A GIF of the arc view updating as you
     change the focus token is the single highest-value asset in this README. -->

| Arc view | Head grid |
| --- | --- |
| _coming soon_ | _coming soon_ |

**Live demo:** _deploy link goes here_

---

## Quick start

```bash
git clone https://github.com/<org>/transformer-attention-visualizer.git
cd transformer-attention-visualizer

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
streamlit run app/main.py
```

First run downloads `bert-base-uncased` (~420 MB). Everything after that is
cached. CPU is fine — there is no training in the Attention lens.

---

## What's in the box

```
shared/            model loading, config, the shared probe sentences
attention_lens/    attention extraction, heatmaps, arc diagrams, rollout
shap_lens/         token attribution
probing_lens/      layer-wise probing classifiers
bias_dashboard/    minimal-pair comparison + final integration
app/               the Streamlit app that stitches all four together
tests/             sanity checks (attention rows sum to 1, etc.)
docs/              report material, figures, demo script
```

Everything imports from `shared/`. One model, one tokenizer, one sentence
set — so four independent lenses stay comparable.

---

## The probe set

`shared/sentences.py` is not a random list of sentences. Each one is chosen
to expose a specific behaviour, and carries a `why` field explaining what to
look for:

- **Coreference** — Winograd-style pairs where one swapped word should move
  the pronoun's attention to a different noun
- **Negation** — does `not` actually flip the weight of the word it scopes over?
- **Word sense** — `bank` as money vs. `bank` as riverside
- **Syntax** — agreement attraction, centre-embedded clauses
- **Sentiment** — mixed sentiment, and sarcasm where surface words lie
- **Bias** — minimal pairs differing only in a gendered word

---

## A note on what attention does and doesn't prove

Attention weights are **not** a complete explanation of model behaviour, and
the interpretability literature is fairly direct about this. A high weight
means information flowed along that edge, not that the edge caused the
output. That limitation is the reason this project has four lenses instead
of one, and it is discussed rather than glossed over in `docs/`.

---

## Author

**Pavan Kumar** — design, implementation, and research write-up
[GitHub @pavankumar369-ML](https://github.com/pavankumar369-ML) ·
[LinkedIn](https://www.linkedin.com/in/pindiprolu-phani-pavan-kumar-236280385)

## Roadmap

- [x] Shared infrastructure — model loader, probe set, theme
- [x] Attention lens — heatmap, arc view, head grid, attention rollout
- [ ] Token Importance lens — SHAP attribution vs. attention
- [ ] Layer Probing lens — POS and sentiment probes across 12 layers
- [ ] Bias lens — minimal-pair diffing
- [ ] Live deployment + demo GIF

---

## Built with

`transformers` · `torch` · `streamlit` · `matplotlib` · `shap` · `scikit-learn`

## References

- Vaswani et al., *Attention Is All You Need* (2017)
- Clark et al., *What Does BERT Look At? An Analysis of BERT's Attention* (2019)
- Abnar & Zuidema, *Quantifying Attention Flow in Transformers* (2020)
- Jain & Wallace, *Attention is not Explanation* (2019)
- Tenney et al., *BERT Rediscovers the Classical NLP Pipeline* (2019)

## License

MIT
