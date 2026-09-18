# Findings log

One entry per lens, minimum. This file is what turns a tool into a project —
in the report and in the viva, "here is what we built" is worth much less
than "here is what we found."

Format: sentence, what you expected, what happened, what you concluded.

---

## Attention lens

### Finding 1 — coreference shifts on a single word
- **Probe:** the Winograd pair in `shared.sentences.COREFERENCE[0:2]`
- **Expected:** `it` attends to `animal` in the `tired` version, `street` in
  the `wide` version
- **Observed:** _fill in — layer, head, actual weights_
- **Conclusion:** _fill in_

### Finding 2 — head specialisation
- **Probe:** any sentence, head grid view, layer 0 vs layer 11
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

## Token importance lens
_Person B_

---

## Layer probing lens
_Person C_

---

## Bias lens
_Person D_
