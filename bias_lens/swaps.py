"""Build a gender-swapped counterpart of a sentence (pure Python, no model)."""
import re

from shared.sentences import BIAS_PAIRS

SWAPS = {
    "his": "her", "her": "his", "he": "she", "she": "he", "him": "her",
    "man": "woman", "woman": "man", "men": "women", "women": "men",
    "boy": "girl", "girl": "boy", "father": "mother", "mother": "father",
    "brother": "sister", "sister": "brother", "husband": "wife", "wife": "husband",
}


def swap_counterpart(text: str):
    """Return (counterpart_text or None, did_swap).

    1. If `text` is one half of a curated pair in shared/sentences.BIAS_PAIRS,
       return the other half.
    2. Otherwise swap every gendered word (his<->her, man<->woman, ...).
    3. If there is none, return (None, False).
    """
    for a, b in BIAS_PAIRS:
        if text == a.text:
            return b.text, True
        if text == b.text:
            return a.text, True

    changed = False

    def repl(m):
        nonlocal changed
        w = m.group(0)
        new = SWAPS.get(w.lower())
        if new is None:
            return w
        changed = True
        return new.capitalize() if w[0].isupper() else new

    out = re.sub(r"\b\w+\b", repl, text)
    return (out, True) if changed else (None, False)
