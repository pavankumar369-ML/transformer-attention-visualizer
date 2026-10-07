"""Batch experiment: all four methods on probe sentences + SST-2 validation, all three models.

Run from the repo root:  python -m attribution_lens.run_experiment --n-sst2 200
Writes data/attribution_results.csv (one row per model x sentence x method) and
data/attribution_summary.csv (the report table).
"""
from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from shared.config import SENTIMENT_MODELS

from ._compat import probe_sentences
from .attribute import METHOD_LABELS, METHODS
from .lens import analyze

MODELS = tuple(SENTIMENT_MODELS)  # the three checkpoints in shared.config


def sst2_sample(n: int, seed: int = 0) -> list[tuple[str, int]]:
    """``n`` random (sentence, label) pairs from the SST-2 validation split."""
    from datasets import load_dataset

    ds = load_dataset("glue", "sst2", split="validation")
    idx = np.random.default_rng(seed).choice(len(ds), size=min(n, len(ds)), replace=False)
    return [(ds[int(i)]["sentence"].strip(), int(ds[int(i)]["label"])) for i in idx]


def rows_for(p: dict, source: str, sid: int, true_label: int | None) -> list[dict]:
    """Flatten one analysis payload into one CSV row per method (+ Random)."""
    shap_i = METHODS.index("SHAP")
    base = {"model": p["family"], "source": source, "sentence_id": sid, "text": p["text"],
            "n_tokens": int(np.sum(~np.array(p["special"]))), "pred": p["pred_label"],
            "pred_prob": p["pred_prob"], "true_label": true_label, "ig_delta": p["ig_delta"]}
    out = []
    for name in (*METHODS, "Random"):
        f = p["faithfulness"][name]
        row = {**base, "method": name if name == "Random" else METHOD_LABELS[name],
               "comprehensiveness": f["comprehensiveness"], "sufficiency": f["sufficiency"],
               "aopc": f["aopc"], "spearman_with_shap": np.nan, "top3_overlap_with_shap": np.nan}
        if name in METHODS:
            j = METHODS.index(name)
            row["spearman_with_shap"] = p["agreement"]["spearman"][shap_i][j]
            row["top3_overlap_with_shap"] = p["agreement"]["top3"][shap_i][j]
        out.append(row)
    return out


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Report table: mean comprehensiveness, sufficiency, AOPC, Spearman with SHAP per method."""
    order = [*(METHOD_LABELS[m] for m in METHODS), "Random"]
    g = df.groupby("method")[["comprehensiveness", "sufficiency", "aopc", "spearman_with_shap"]].mean()
    return g.reindex(order)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-sst2", type=int, default=200)
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--out", default="data/attribution_results.csv")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="only first N sentences (smoke test)")
    args = ap.parse_args()

    items: list[tuple[str, str, int | None]] = [(t, "probe", None) for t in probe_sentences()]
    items += [(t, "sst2", y) for t, y in sst2_sample(args.n_sst2, args.seed)]
    if args.limit:
        items = items[:args.limit]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for model in args.models:
        for sid, (text, source, label) in enumerate(items):
            try:
                rows += rows_for(analyze(text, model), source, sid, label)
            except Exception:
                print(f"[skip] {model} #{sid}: {text[:50]!r}\n{traceback.format_exc(limit=2)}")
            if (sid + 1) % 25 == 0:
                print(f"{model}: {sid + 1}/{len(items)}")
                pd.DataFrame(rows).to_csv(out, index=False)  # checkpoint
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    summary = summarise(df)
    summary.to_csv(out.with_name("attribution_summary.csv"))
    print(f"\nSaved {len(df)} rows to {out}\n")
    print(summary.round(3).to_string())
    for model, part in df.groupby("model"):
        print(f"\n{model}\n{summarise(part).round(3).to_string()}")


if __name__ == "__main__":
    main()
