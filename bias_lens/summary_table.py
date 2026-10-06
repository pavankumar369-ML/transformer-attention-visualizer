"""Print the report summary table. Run after both offline scripts:
    python -m bias_lens.summary_table
"""
from pathlib import Path

import numpy as np
import pandas as pd

from causal_lens.pruning import first_drop

DATA = Path(__file__).resolve().parents[1] / "data"
PAIRS = [("BERT", "bert-base-uncased", "bert"), ("DistilBERT", "distilbert-base-uncased", "distilbert"),
         ("RoBERTa", "roberta-base", "roberta")]

if __name__ == "__main__":
    res = pd.read_csv(DATA / "bias_results.csv")
    print("| Model | Overall stereotype score (95% CI) | Categories significantly above 50% | "
          "Heads removable before accuracy drops 2 points | Most important layer |")
    print("|---|---|---|---|---|")
    for name, mlm, key in PAIRS:
        r = res[res.model == mlm]
        o = r[r.category == "overall"].iloc[0] if len(r) else None
        cats = r[(r.category != "overall") & r.significantly_above_50]["category"].tolist() if len(r) else []
        bias = f"{o.score:.1f}% ({o.ci_low:.1f}–{o.ci_high:.1f})" if o is not None else "(not run)"
        pr, imp = DATA / f"pruning_{key}.csv", DATA / f"head_importance_{key}.npy"
        if pr.exists() and imp.exists():
            _, heads = first_drop(pd.read_csv(pr))
            total = np.load(imp).size
            layer = int(np.load(imp).sum(axis=1).argmax()) + 1
            c = f"{heads} of {total}"
            lay = f"layer {layer}"
        else:
            c = lay = "(not run)"
        print(f"| {name} | {bias} | {', '.join(cats) or 'none'} | {c} | {lay} |")
