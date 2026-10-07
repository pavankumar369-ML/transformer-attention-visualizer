# Attribution & Faithfulness Lens (Member A)

Answers: **which words actually changed the model's answer, and can we trust that explanation?**

## Methods
SHAP, Integrated Gradients (`[PAD]` baseline, Captum), last-layer `[CLS]` attention,
attention rollout (`attention_lens.extract.attention_rollout`).
Each is scored by deleting its top-ranked words (`[MASK]`) and measuring the probability drop
(comprehensiveness, sufficiency, AOPC) against a random-order control, and compared by
Spearman correlation and top-3 overlap.

## Files
`predict.py` probabilities | `attribute.py` the four methods | `agreement.py` 4x4 tables |
`faithfulness.py` deletion tests | `lens.py` `compute()` / `render()` | `render.py` Streamlit panels |
`run_experiment.py` batch run | `_compat.py` the only file touching `shared/` APIs.

## Use
```python
from attribution_lens import compute, render
result = compute("The food was not bad at all.", "distilbert-base-uncased-finetuned-sst-2-english")
render(result)   # inside a Streamlit tab; the app wires it automatically
# model_name: any key of shared.config.SENTIMENT_MODELS (or the short names distilbert | bert | roberta)
```
Batch experiment: `python -m attribution_lens.run_experiment --n-sst2 200`
(writes `data/attribution_results.csv`). Tests: `pytest tests/test_attribution.py -q`.
