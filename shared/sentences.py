"""
The shared probe set.

All four lenses analyse these same sentences. That is what turns four
separate modules into one coherent demo: the examiner types nothing, clicks
"The animal didn't cross the street...", and watches four different
explanations of the same sentence appear.

Each entry has a `why` field - the one-line reason the sentence is in the
set. Use it as the caption in the UI so the audience knows what to look for.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class Probe:
    text: str
    category: str
    why: str
    focus: List[str] = field(default_factory=list)  # tokens worth pointing at


COREFERENCE = [
    Probe(
        text="The animal didn't cross the street because it was too tired.",
        category="coreference",
        why="'it' should attend to 'animal' - the tired thing is the animal.",
        focus=["it", "animal"],
    ),
    Probe(
        text="The animal didn't cross the street because it was too wide.",
        category="coreference",
        why="Same sentence, one word changed - now 'it' should attend to 'street'.",
        focus=["it", "street"],
    ),
    Probe(
        text="The trophy doesn't fit in the suitcase because it is too small.",
        category="coreference",
        why="Classic Winograd pair; 'it' resolves to the suitcase.",
        focus=["it", "suitcase"],
    ),
]

NEGATION = [
    Probe(
        text="The food was not bad at all.",
        category="negation",
        why="Does the model let 'not' flip the weight of 'bad'?",
        focus=["not", "bad"],
    ),
    Probe(
        text="I would not say the service was good.",
        category="negation",
        why="Negation scoped over a positive word - tests indirect sentiment.",
        focus=["not", "good"],
    ),
]

AMBIGUITY = [
    Probe(
        text="He went to the bank to deposit his paycheck.",
        category="word-sense",
        why="'bank' as a financial institution - watch 'deposit' and 'paycheck'.",
        focus=["bank", "deposit"],
    ),
    Probe(
        text="He sat on the bank and watched the river flow.",
        category="word-sense",
        why="Same word, different sense - context should shift attention to 'river'.",
        focus=["bank", "river"],
    ),
]

SENTIMENT = [
    Probe(
        text="The battery life is terrible but the camera is stunning.",
        category="sentiment",
        why="Mixed sentiment - which half dominates the prediction?",
        focus=["terrible", "stunning"],
    ),
    Probe(
        text="Wow, it broke in one day. Truly impressive engineering.",
        category="sentiment",
        why="Sarcasm - surface words are positive, meaning is negative.",
        focus=["impressive", "broke"],
    ),
]

SYNTAX = [
    Probe(
        text="The keys to the cabinet were on the table.",
        category="syntax",
        why="Agreement attraction - 'were' agrees with 'keys', not 'cabinet'.",
        focus=["keys", "were"],
    ),
    Probe(
        text="The man who the woman greeted left the building.",
        category="syntax",
        why="Centre-embedded clause; tests long-range syntactic linking.",
        focus=["man", "left"],
    ),
]

# Bias templates are minimal pairs: identical sentences, one swapped word.
# The Bias lens diffs the two runs; the other lenses can display them too.
BIAS_PAIRS = [
    (
        Probe(
            text="The doctor finished his shift and went home.",
            category="bias",
            why="Occupation-gender template (male form).",
            focus=["doctor", "his"],
        ),
        Probe(
            text="The doctor finished her shift and went home.",
            category="bias",
            why="Occupation-gender template (female form).",
            focus=["doctor", "her"],
        ),
    ),
    (
        Probe(
            text="The nurse finished his shift and went home.",
            category="bias",
            why="Counter-stereotype pairing for the same occupation slot.",
            focus=["nurse", "his"],
        ),
        Probe(
            text="The nurse finished her shift and went home.",
            category="bias",
            why="Stereotype-aligned pairing for the same occupation slot.",
            focus=["nurse", "her"],
        ),
    ),
]

ALL_PROBES: List[Probe] = (
    COREFERENCE + NEGATION + AMBIGUITY + SENTIMENT + SYNTAX
)

# Flat list of bias probes, for lenses that just want sentences.
BIAS_PROBES: List[Probe] = [p for pair in BIAS_PAIRS for p in pair]


def by_category(category: str) -> List[Probe]:
    """All probes in one category, bias included."""
    pool = ALL_PROBES + BIAS_PROBES
    return [p for p in pool if p.category == category]


def categories() -> List[str]:
    seen = []
    for p in ALL_PROBES + BIAS_PROBES:
        if p.category not in seen:
            seen.append(p.category)
    return seen
