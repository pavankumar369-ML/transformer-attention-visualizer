# Report draft: Member C

## Bias Analysis

**Question.** Does the model assign higher likelihood to stereotyped sentences than to their minimally different counterparts, and is the effect larger than chance?

**Method.** We use CrowS-Pairs (Nangia et al., 2020), 1,508 sentence pairs across nine categories. Each pair differs in a few words. For each masked language model (BERT, DistilBERT, RoBERTa) we compute a pseudo-log-likelihood (PLL): each shared word is masked in turn and the log-probability of the true word is summed. Only words the two sentences share are scored, so the swapped words do not change how many words each sentence contributes. The model "prefers the stereotype" if the stereotyped sentence has the higher PLL. The stereotype score is the percentage of pairs where this happens; 50% means no preference. We report a 95% bootstrap interval (1,000 resamples of pairs) and an exact binomial test against 50%, and correct for nine categories with Bonferroni.

**Results.** Overall scores are 55.6% (DistilBERT, CI 53.2-58.1), 58.5% (BERT, 56.0-60.9) and 59.9% (RoBERTa, 57.4-62.4). Religion is the most consistent category (66.7%, 71.4%, 70.5%). Gender differs between models: near 50% for DistilBERT and BERT, 60.7% for RoBERTa. [VERIFY and compare with Nangia et al.] A minimal-pair explorer compares attention and occlusion attribution for a pair such as "his"/"her". For the doctor pair, the sentiment prediction barely changes (0.83 to 0.85 for BERT), and the word with the largest attribution shift differs between models.

**Limitations.** The benchmark measures a likelihood preference, not harm. Some pairs have quality problems (Blodgett et al., 2021). Small categories (physical appearance, 63 pairs; disability, 60) have wide intervals. The score depends on the scoring method.

## Causal Head Ablation

**Question.** Which attention heads actually matter for a model's prediction? Attention maps show correlation; ablation tests cause.

**Method.** We fine-tuned sentiment checkpoints of each model, evaluated on 200 SST-2 validation sentences. For each head, we switch it off (zeroing its contribution before the attention output projection) and record the change in accuracy and correct-class probability, giving a layers x heads importance matrix. We then remove heads progressively, least important first, in 10% steps, and compare with random removal (mean of five orders). We also test the "it" to "animal" attention pattern: we switch off the heads where "it" attends most to its referent in fill-in-the-blank sentences and compare the effect with removing random heads.

**Results.** Baseline accuracy is 0.910 (DistilBERT), 0.910 (BERT), 0.935 (RoBERTa). Accuracy stays within about 2 points until roughly 40% (DistilBERT) and 60% (BERT, RoBERTa) of heads are removed, and importance order outperforms random order, supporting heavy redundancy (cf. Michel et al., 2019 [VERIFY]). No single head matters much. RoBERTa falls to chance at 70% removed in importance order, probably because heads that back each other up appear unimportant one at a time. For the coreference sentence, switching off the five top "it" to "animal" heads reduces RoBERTa's preference for "tired" from +4.98 to +0.54 versus +4.82 for random heads, so this attention is used; the effect is weaker in BERT.

**Limitations.** 200 sentences give a resolution of 0.5 points per sentence. The coreference experiment uses two sentences. One-at-a-time importance cannot capture joint effects.

## Summary table

| Model | Overall stereotype score (95% CI) | Categories significantly above 50% | Heads removable before a 2-point drop | Most important layer |
|---|---|---|---|---|
| BERT | 58.5% (56.0-60.9) | age, nationality, race-color, religion, sexual-orientation, socioeconomic | about 60% (86 of 144) | layer 10 (weak) |
| DistilBERT | 55.6% (53.2-58.1) | age, religion, sexual-orientation | about 40% (29 of 72) | layer 5 (weak) |
| RoBERTa | 59.9% (57.4-62.4) | age, disability, gender, nationality, religion, sexual-orientation, socioeconomic | about 60% (86 of 144) | layer 1 (weak) |

(Significance here is the uncorrected p < 0.05 column; state the Bonferroni-corrected subset in the text.)

## Slide (one slide)

**Title:** Change something, measure what moves.
- Bias: models assign higher likelihood to stereotyped sentences in 56-60% of CrowS-Pairs (50% = no preference); religion is the most consistent category.
- Causal: 40-60% of attention heads can be removed with little accuracy loss; but the "it" to "animal" heads are used by RoBERTa (+4.98 to +0.54 on removal).
- Honest limits: benchmark quality, 200-sentence resolution, two-sentence coreference test.
