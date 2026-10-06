"""
Probing Lens - probes and control tasks.

A probe is a deliberately weak classifier (logistic regression) trained on
FROZEN hidden states from one layer. If even a linear model can read a
property off a layer, that property is explicitly encoded there. A deep
probe could learn the task itself and tell us little about the layer.

The catch (Hewitt & Liang, 2019): a probe can score well by memorising word
identities - "the" is always DT - without the layer encoding any grammar.
The control task measures that. Every word TYPE gets a random but fixed
label, drawn from the real label distribution. The only way to do well on
it is to memorise words, so

    selectivity = real accuracy - control accuracy

is the part of the score that memorisation cannot explain.
"""

from typing import Dict, List, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 0


def make_probe():
    """Standardise, then logistic regression.

    Scaling is still linear, so the probe is no more powerful. It only fixes
    the optimiser: a few BERT dimensions are an order of magnitude larger
    than the rest, and unscaled lbfgs crawls on them.
    """
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=SEED),
    )


def probe_accuracy(X_train, y_train, X_test, y_test) -> float:
    """Train one probe, return test accuracy."""
    clf = make_probe()
    clf.fit(np.asarray(X_train, dtype=np.float32), y_train)
    return float(clf.score(np.asarray(X_test, dtype=np.float32), y_test))


def majority_baseline(y_train, y_test) -> float:
    """Accuracy of always guessing the most common training label."""
    values, counts = np.unique(y_train, return_counts=True)
    return float(np.mean(np.asarray(y_test) == values[np.argmax(counts)]))


def control_labels(
    words: Sequence[str],
    labels: Sequence[int],
    seed: int = SEED,
) -> np.ndarray:
    """Hewitt & Liang control task: one random, fixed label per word type.

    Pass train and test words together so a word keeps its label across the
    split. Word types are lower-cased, because BERT-uncased cannot tell
    "The" from "the" and the control should not ask it to.
    """
    rng = np.random.default_rng(seed)
    values, counts = np.unique(labels, return_counts=True)
    probs = counts / counts.sum()

    types = sorted({w.lower() for w in words})  # sorted = reproducible
    drawn = rng.choice(values, size=len(types), p=probs)
    lookup: Dict[str, int] = dict(zip(types, drawn.tolist()))
    return np.array([lookup[w.lower()] for w in words])


def layer_curve(
    train_layers: List[np.ndarray],
    y_train,
    test_layers: List[np.ndarray],
    y_test,
) -> List[float]:
    """Probe accuracy for every layer. *_layers[l] is (n_items, hidden)."""
    return [
        probe_accuracy(Xtr, y_train, Xte, y_test)
        for Xtr, Xte in zip(train_layers, test_layers)
    ]


def selectivity(accuracy: Sequence[float], control_accuracy: Sequence[float]) -> np.ndarray:
    return np.asarray(accuracy) - np.asarray(control_accuracy)
