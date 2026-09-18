# Token Importance Lens — owner: Person B

## What you are building
Attention shows what the model *looked at*. It does not show what actually
*changed the answer*. This lens answers the second question using SHAP.

## Deliverables
- `attribute.py` — given a sentence, return one importance score per token
  for the sentiment classifier (`shared.config.CLASSIFIER_MODEL`).
- `render.py` — two views:
  1. the sentence rendered inline with each word background-shaded by score
     (red = pushed toward negative, blue = pushed toward positive)
  2. a horizontal bar chart of the top tokens
- Wire it into the **Token Importance** tab of `app/main.py`.

## The interesting result to aim for
Pick a sentence where attention and attribution disagree. `"The battery life
is terrible but the camera is stunning."` is a good candidate — attention
spreads widely, but attribution should concentrate on the two adjectives.
That disagreement is the finding worth putting in the report.

## Getting started
```python
import shap
from shared.model_loader import load_classifier

tokenizer, model = load_classifier()
# shap.Explainer works with a HF pipeline; see shap docs for "text" explainers
```

## Rules
- Import the model from `shared.model_loader`, never load your own.
- Test on the probes in `shared.sentences`, not on ad-hoc sentences.
- Work on branch `feature/shap-lens`, open a PR into `main`.
