# Demo script (5 minutes)

Rehearse this. A live demo that wanders is worse than a slide.

## 0:00 — The hook (45s)

Open on the arc view, sentence already loaded:

> "The animal didn't cross the street because **it** was too tired."

Point at the thick arc from `it` to `animal`.

> "Nobody told the model what 'it' refers to. It worked that out, and this
> is us reading that decision off the weights."

Switch the dropdown to the `too wide` variant. The arc moves to `street`.

> "One word changed. The model rewired."

That 45 seconds is the whole project. Everything after is depth.

## 0:45 — Why one view isn't enough (60s)

Open the SHAP tab on the mixed-sentiment sentence.

> "Attention says the model *looked* here. It doesn't say the answer
> *depended* on it. There's a well-known paper called 'Attention is not
> Explanation' making exactly this point — so we added a second lens that
> measures influence on the output instead of information flow."

Show a case where the two disagree. That contrast is your strongest slide.

## 1:45 — Depth (60s)

Layer probing tab. Accuracy-vs-depth curves.

> "Syntax is most recoverable around the middle layers, semantics later.
> The model builds up a pipeline it was never explicitly taught."

## 2:45 — Why it matters (60s)

Bias tab. Minimal pair, side by side, delta panel.

> "Same sentence, one swapped word. Here's where the internal processing
> diverges. We're not claiming harm from this alone — we're showing the
> measurement, which is the part that's usually invisible."

## 3:45 — Close (45s)

Back to the About tab. One line each on the four lenses, then:

> "Four views, one model, one sentence. Live at <URL>, code on GitHub."

## Q&A prep — the questions you will get

**"Doesn't attention already explain the model?"**
No, and we can say why. High attention means information flowed along that
edge, not that it caused the output. Jain & Wallace (2019) showed you can
often find different attention distributions that produce the same
prediction. That limitation is the reason for lenses 2–4.

**"Did you train anything?"**
The attention and bias lenses use a pretrained checkpoint at inference time.
The probing lens trains small logistic-regression probes on frozen
embeddings — the transformer itself stays frozen, which is the point of
probing.

**"What's attention rollout?"**
Last-layer attention alone ignores twelve layers of prior mixing. Rollout
multiplies the residual-adjusted attention matrices across layers to
approximate input-to-output influence.

**"Why should I trust the bias result?"**
We don't claim harm from attention differences alone — we show a
measurable divergence in internal processing on minimal pairs, and say
exactly that.
