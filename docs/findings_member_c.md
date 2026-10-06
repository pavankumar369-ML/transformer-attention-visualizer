# Findings: Bias and Causal lenses (Member C)

Setup. Bias: CrowS-Pairs (1,508 pairs), pseudo-log-likelihood over shared tokens, 95% bootstrap CI (1,000 resamples), exact binomial test against 50%. Causal: 200 SST-2 validation sentences, one head removed at a time, accuracy measured after removal. Models: BERT, DistilBERT, RoBERTa.

## Finding 1: Which categories show a significant preference for stereotyped sentences?

**Expected.** Some preference above 50% in all models, largest for categories with more training-text association.

**What happened.** Overall scores: DistilBERT 55.6% (53.2-58.1), BERT 58.5% (56.0-60.9), RoBERTa 59.9% (57.4-62.4); all intervals exclude 50%. Religion is above 50% in all three models (66.7%, 71.4%, 70.5%) and survives a Bonferroni correction for nine categories. Gender differs by model: 50.4% (DistilBERT) and 48.5% (BERT), both indistinguishable from 50%, but 60.7% in RoBERTa (p = 0.001). Physical appearance is never significant, but has only 63 pairs, so its intervals are wide.

**What it means.** The models assign higher likelihood to the stereotyped sentence in more than half of pairs, and the preference differs by category and model. The benchmark measures a likelihood preference, not harm. Limitations: p-values for the nine categories are corrected for multiple testing (borderline: BERT sexual orientation, RoBERTa disability); some CrowS-Pairs items have quality problems (Blodgett et al., 2021). [VERIFY: compare our overall scores with Nangia et al. (2020).]

## Finding 2: Does the minimal-pair explorer show where the difference comes from?

**Expected.** The swapped word ("his"/"her") or the occupation word would carry the largest change.

**What happened.** Swapping "his" for "her" in "The doctor finished his shift and went home" changes P(positive) only from 0.83 to 0.85 (BERT), 1.00 to 0.99 (DistilBERT) and 0.34 to 0.31 (RoBERTa). The largest attribution shift falls on a different word in each model: "finished" (BERT, +0.029), "home" (DistilBERT, +0.058), "doctor" (RoBERTa, +0.023).

**What it means.** A sentiment classifier is barely sensitive to this swap, and the word with the largest shift is not stable across models, so we cannot claim a consistent source of the difference. The explorer is a diagnostic of a sentiment classifier, not a measure of stereotype preference; that is measured by the benchmark. [ADD: nurse pairs once run.]

## Finding 3: How many heads does the model really need?

**Expected.** Many heads can be removed with little loss (Michel et al., 2019).

**What happened.** Removing heads least-important-first, accuracy stays within about 2 points of baseline until roughly 40% of heads (DistilBERT, 29 of 72) and 60% (BERT and RoBERTa, 86 of 144; the true limit lies between 60% and 70% given the 10% step). Random-order removal degrades earlier (BERT at 50%: 0.910 vs 0.875). RoBERTa collapses to chance (0.500) at 70% removed in importance order, worse than random order (0.669). Baselines: DistilBERT 0.910, BERT 0.910, RoBERTa 0.935.

**What it means.** The models are highly redundant, consistent with Michel et al. [VERIFY exact figures]. The RoBERTa cliff is a likely limitation of one-at-a-time importance scores: heads that back each other up look unimportant individually but fail together. This explanation is not tested. With 200 sentences one sentence is 0.5 points, so differences of 1-2 points are noise. No single head is critical (largest single-head accuracy drops are 1-2 sentences), and the "most important layer" (BERT 10, DistilBERT 5, RoBERTa 1) is a weak signal.

## Finding 4: Does the coreference head matter?

**Expected.** Removing the heads where "it" attends most to "animal" should hurt the model's preference for "tired" over "wide", more than removing random heads.

**What happened.** Score = log P(correct word) - log P(wrong word) at the blank, 20 random controls.
- RoBERTa, animal sentence: baseline +4.98; top-5 coreference heads removed: +0.54; random 5 heads: +4.82 +/- 0.71. Removing only the top head: +4.36 vs random +5.01 +/- 0.28.
- BERT, animal sentence: baseline -1.03 (it prefers the wrong word before ablation); top-1: -2.01, top-5: -2.49 vs random about -1.0 +/- 0.25-0.47; top-10: -1.57 vs -0.96 +/- 0.60 (within noise).
- Trophy/suitcase sentence, both models: changes are small and mostly within random spread (neither model solves this sentence at baseline).

**What it means.** For this sentence, the attention pattern people find impressive is used by RoBERTa, strongly, and by BERT, weakly. Limitations: two sentences only; the random control does not rule out that heads with high attention to any token are important in general; the experiment tests this model's fill-in preference, not a general coreference ability.
