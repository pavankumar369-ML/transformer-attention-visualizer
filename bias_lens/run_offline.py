"""Full benchmark run. Usage (from repo root):
    python -m bias_lens.run_offline                       # all three models
    python -m bias_lens.run_offline distilbert-base-uncased
Saves data/bias_pairs_<model>.csv and data/bias_results.csv.
"""
import sys
from pathlib import Path

import pandas as pd

from bias_lens.benchmark import run_pairs, summarise

MODELS = ["distilbert-base-uncased", "bert-base-uncased", "roberta-base"]
DATA = Path(__file__).resolve().parents[1] / "data"


def main(models):
    DATA.mkdir(exist_ok=True)
    results_path = DATA / "bias_results.csv"
    for m in models:
        print(f"== {m} ==", flush=True)
        pairs = run_pairs(m)
        pairs.to_csv(DATA / f"bias_pairs_{m}.csv", index=False)
        summ = summarise(pairs, m)
        old = pd.read_csv(results_path) if results_path.exists() else None
        if old is not None:
            old = old[old["model"] != m]
            summ = pd.concat([old, summ], ignore_index=True)
        summ.to_csv(results_path, index=False)
        print(summ[summ.model == m][["category", "n", "score", "ci_low",
                                     "ci_high", "p_value"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:] or MODELS)
