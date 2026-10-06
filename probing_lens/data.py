"""
Probing Lens - datasets.

Small on purpose, so the whole experiment runs on a laptop CPU:

  conll2003   1,000 train / 250 test sentences, per-word POS and NER tags
  glue/sst2   1,000 train / 250 test sentences, one sentiment label each

SST-2's GLUE test split has no labels (they are all -1), so its 250 test
sentences come from the validation split instead.

Subsampling is seeded, so every run and every model sees the same sentences.
That matters: CKA compares two models row by row, which only works if row i
is the same word in both.
"""

import random
from typing import Dict, List

from datasets import load_dataset

# The canonical "conll2003" is a script dataset, which newer versions of
# `datasets` refuse to run. Try the original first, then a mirror, then the
# auto-converted parquet copy that the Hub keeps for script datasets.
CONLL_SOURCES = [
    ("eriktks/conll2003", None),
    ("eriktks/conll2003", "refs/convert/parquet"),
    ("conll2003", None),
]
SST2_SOURCE = ("nyu-mll/glue", "sst2")

N_TRAIN = 1000
N_TEST = 250
SEED = 0


def _sample(split, n: int, seed: int) -> List[int]:
    idx = list(range(len(split)))
    random.Random(seed).shuffle(idx)
    return sorted(idx[:n])


def load_conll(n_train: int = N_TRAIN, n_test: int = N_TEST, seed: int = SEED) -> Dict:
    """{"train": [...], "test": [...], "source": str}

    Each item is {"words": [...], "pos": [...], "ner": [...]}.
    """
    ds, source, errors = None, None, []
    for name, revision in CONLL_SOURCES:
        try:
            ds = load_dataset(name, revision=revision)
            source = name if revision is None else f"{name}@{revision}"
            break
        except Exception as exc:  # try the next mirror
            errors.append(f"{name}@{revision}: {exc}")
    if ds is None:
        raise RuntimeError("Could not load conll2003:\n" + "\n".join(errors))

    def keep(ex):
        # Drop document separators and empty lines.
        return len(ex["tokens"]) > 0 and ex["tokens"][0] != "-DOCSTART-"

    out = {"source": source}
    for split, n in (("train", n_train), ("test", n_test)):
        data = ds[split].filter(keep)
        out[split] = [
            {"words": ex["tokens"], "pos": ex["pos_tags"], "ner": ex["ner_tags"]}
            for ex in data.select(_sample(data, n, seed))
        ]
    return out


def load_sst2(n_train: int = N_TRAIN, n_test: int = N_TEST, seed: int = SEED) -> Dict:
    """{"train": [...], "test": [...], "source": str}

    Each item is {"words": [...], "label": 0 or 1}. SST-2 sentences are
    already lower-cased and space-tokenised, so splitting on spaces gives
    the words.
    """
    ds = load_dataset(*SST2_SOURCE)
    out = {"source": "/".join(SST2_SOURCE)}
    for split, src, n in (("train", "train", n_train), ("test", "validation", n_test)):
        data = ds[src]
        out[split] = [
            {"words": ex["sentence"].split(), "label": ex["label"]}
            for ex in data.select(_sample(data, n, seed))
        ]
    return out


if __name__ == "__main__":
    # Week-1 check: print five examples from each dataset.
    conll = load_conll()
    print(f"conll2003 from {conll['source']}: "
          f"{len(conll['train'])} train / {len(conll['test'])} test")
    for ex in conll["train"][:5]:
        print(list(zip(ex["words"], ex["pos"], ex["ner"])))

    sst = load_sst2()
    print(f"\nsst2 from {sst['source']}: {len(sst['train'])} train / {len(sst['test'])} test")
    for ex in sst["train"][:5]:
        print(ex["label"], " ".join(ex["words"]))
