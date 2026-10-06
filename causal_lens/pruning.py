"""Progressive pruning: least-important-first vs random order."""
from pathlib import Path

import numpy as np
import pandas as pd

from causal_lens.ablate import evaluate, grid_shape
from causal_lens.importance import load_sst2

DATA = Path(__file__).resolve().parents[1] / "data"


def _mask_removing(order: np.ndarray, n_remove: int, shape) -> np.ndarray:
    flat = np.ones(shape[0] * shape[1])
    flat[order[:n_remove]] = 0
    return flat.reshape(shape)


def prune_curve(model_key: str, prob_drop: np.ndarray, n: int = 200,
                n_random: int = 5, step_pct: int = 10) -> pd.DataFrame:
    """Order by prob_drop (finer than accuracy drop, which has many ties)."""
    texts, labels = load_sst2(n)
    shape = grid_shape(model_key)
    total = shape[0] * shape[1]
    imp_order = np.argsort(prob_drop.ravel(), kind="stable")  # least important first
    rows = []
    for pct in range(0, 101, step_pct):
        k = round(total * pct / 100)
        acc_imp, _ = evaluate(texts, labels, _mask_removing(imp_order, k, shape), model_key)
        rand = []
        for seed in range(n_random):
            order = np.random.default_rng(seed).permutation(total)
            rand.append(evaluate(texts, labels, _mask_removing(order, k, shape), model_key)[0])
        rows.append(dict(model=model_key, pct_removed=pct, heads_removed=k,
                         acc_importance=acc_imp, acc_random_mean=np.mean(rand),
                         acc_random_std=np.std(rand)))
        print(f"  {pct}% removed: importance {acc_imp:.3f} random {np.mean(rand):.3f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(DATA / f"pruning_{model_key}.csv", index=False)
    return df


def first_drop(df: pd.DataFrame, points: float = 0.02):
    """Largest % removed before accuracy falls more than `points` below the unpruned baseline."""
    base = df.loc[df.pct_removed == 0, "acc_importance"].iloc[0]
    bad = df[df.acc_importance < base - points]
    if bad.empty:
        return int(df.pct_removed.max()), int(df.heads_removed.max())
    first = bad.iloc[0]
    prev = df[df.pct_removed < first.pct_removed].iloc[-1]
    return int(prev.pct_removed), int(prev.heads_removed)
