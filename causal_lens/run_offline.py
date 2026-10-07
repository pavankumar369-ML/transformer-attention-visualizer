"""Usage (repo root):  python -m causal_lens.run_offline [distilbert bert roberta]
Saves data/head_importance_<m>.npy, head_importance_prob_<m>.npy, pruning_<m>.csv"""
import sys

from causal_lens.importance import compute_importance
from causal_lens.pruning import first_drop, prune_curve

if __name__ == "__main__":
    for m in (sys.argv[1:] or ["distilbert", "bert", "roberta"]):
        print(f"== {m} ==", flush=True)
        acc_drop, prob_drop, base_acc, _ = compute_importance(m)
        curve = prune_curve(m, prob_drop)
        pct, heads = first_drop(curve)
        top_layer = int(acc_drop.sum(axis=1).argmax()) + 1
        print(f"[{m}] removable before >2pt drop: {heads} heads ({pct}%); "
              f"most important layer (summed acc drop): {top_layer}")
