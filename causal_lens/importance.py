"""Score every head by the damage its removal causes (SST-2 validation).

Matrices are saved as .csv (small, and committable: the repo's .gitignore
excludes data/*.npy).
"""
from pathlib import Path

import numpy as np

from causal_lens.ablate import evaluate, grid_shape

DATA = Path(__file__).resolve().parents[1] / "data"


def load_sst2(n: int = 200, seed: int = 0):
    from datasets import load_dataset
    ds = load_dataset("nyu-mll/glue", "sst2", split="validation").shuffle(seed=seed).select(range(n))
    return [s.strip() for s in ds["sentence"]], list(ds["label"])


def compute_importance(model_key: str, n: int = 200):
    """Returns (acc_drop, prob_drop, baseline_acc, baseline_prob); matrices are (layers, heads)."""
    texts, labels = load_sst2(n)
    L, H = grid_shape(model_key)
    base_acc, base_prob = evaluate(texts, labels, np.ones((L, H)), model_key)
    print(f"[{model_key}] baseline accuracy {base_acc:.3f} (expect ~0.88-0.95)", flush=True)
    acc_drop = np.zeros((L, H))
    prob_drop = np.zeros((L, H))
    for l in range(L):
        for h in range(H):
            m = np.ones((L, H))
            m[l, h] = 0
            acc, prob = evaluate(texts, labels, m, model_key)
            acc_drop[l, h] = base_acc - acc
            prob_drop[l, h] = base_prob - prob
        print(f"  layer {l + 1}/{L} done", flush=True)
    DATA.mkdir(exist_ok=True)
    np.savetxt(DATA / f"head_importance_{model_key}.csv", acc_drop, delimiter=",", fmt="%.6f")
    np.savetxt(DATA / f"head_importance_prob_{model_key}.csv", prob_drop, delimiter=",", fmt="%.6f")
    return acc_drop, prob_drop, base_acc, base_prob


def _load(stem: str, key: str):
    for ext, loader in ((".csv", lambda p: np.loadtxt(p, delimiter=",", ndmin=2)), (".npy", np.load)):
        p = DATA / f"{stem}_{key}{ext}"
        if p.exists():
            return loader(p)
    return None


def load_saved(model_key: str):
    """(acc_drop, prob_drop) from data/, or None if the offline run has not happened."""
    acc = _load("head_importance", model_key)
    if acc is None:
        return None
    prob = _load("head_importance_prob", model_key)
    return acc, (prob if prob is not None else acc)
