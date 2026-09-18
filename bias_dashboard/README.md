# Bias Analysis Lens — owner: Person D

## What you are building
Two jobs: the bias lens itself, and final integration of all four tabs.

### Part 1 — the lens
`shared.sentences.BIAS_PAIRS` holds minimal pairs: sentences identical
except for one swapped word (`his` / `her`, `doctor` / `nurse`).

For each pair:
- run both sentences through the Attention lens and the SHAP lens
- diff the results
- surface where the two runs disagree, and by how much

A clean framing for the UI: show both sentences side by side with the same
visualization, and a third panel showing the delta.

### Part 2 — integration
You own the final app feeling like one product rather than four:
- consistent colour scale across all tabs (`shared.config.COLOR_SCALE`)
- consistent token rendering
- the landing/About tab
- deployment to Streamlit Community Cloud

## Framing this honestly
Be careful with claims. A difference in attention weights between `his` and
`her` is *evidence of differential processing*, not proof of harm. Write it
that way in the report — measured claims survive questioning, inflated ones
do not.

## Deliverables
- `compare.py` — run a minimal pair, return per-token deltas
- `render.py` — side-by-side + delta view
- Wire into the **Bias Analysis** tab of `app/main.py`
- Deployed live URL in the root README

## Rules
- Import from `shared.model_loader` and reuse `attention_lens` / `shap_lens`
  functions rather than reimplementing them.
- Work on branch `feature/bias-lens`, open a PR into `main`.
