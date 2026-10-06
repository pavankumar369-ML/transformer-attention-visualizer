"""
Probing Lens - the full offline experiment.

    python -m probing_lens.run_offline                      # all three models
    python -m probing_lens.run_offline --model roberta-base # just one

For each model:
  1. extract word vectors for conll2003 and sentence vectors for SST-2,
     cached per layer as float16 in data/vectors/<model>/ (git-ignored)
  2. train a probe per layer for POS, the POS control task, NER, sentiment
  3. write the rows into data/probing_results.csv
  4. compute layer x layer CKA, plus cross-model CKA for every pair of
     models already cached, into data/cka_results.json

The two result files are small and committed, so the app works on a fresh
clone without re-running any of this.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from probing_lens import data as datasets
from probing_lens.hidden import batch_word_vectors
from probing_lens.probe import control_labels, majority_baseline, probe_accuracy
from probing_lens.similarity import cka_matrix
from shared.config import PROBING_MODELS

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
VECTOR_DIR = DATA_DIR / "vectors"
RESULTS_CSV = DATA_DIR / "probing_results.csv"
CKA_JSON = DATA_DIR / "cka_results.json"

# CKA uses this many conll test words. Same words, same order, every model.
CKA_WORDS = 3000


def _slug(model_name: str) -> str:
    return model_name.replace("/", "__")


def _flat(items: List[dict], key: str) -> np.ndarray:
    return np.array([x for item in items for x in item[key]])


# ------------------------------------------------------------ 1. vectors
def cache_vectors(model_name: str, conll: Dict, sst: Dict, force: bool = False) -> Path:
    """Write <set>_L<layer>.npy for conll_train/test and sst2_train/test."""
    out = VECTOR_DIR / _slug(model_name)
    done = out / "done.json"
    if done.exists() and not force:
        print(f"  vectors cached in {out}")
        return out
    out.mkdir(parents=True, exist_ok=True)

    for name, items, pool in (
        ("conll_train", conll["train"], False),
        ("conll_test", conll["test"], False),
        ("sst2_train", sst["train"], True),
        ("sst2_test", sst["test"], True),
    ):
        t0 = time.time()
        per_sentence = batch_word_vectors([x["words"] for x in items], model_name)
        if pool:
            # Sentence vector = mean of its word vectors.
            stacked = np.stack([v.mean(axis=1) for v in per_sentence], axis=1)
        else:
            stacked = np.concatenate(per_sentence, axis=1)  # (layers+1, words, hidden)
        for layer, X in enumerate(stacked):
            np.save(out / f"{name}_L{layer}.npy", X.astype(np.float16))
        print(f"  {name}: {stacked.shape} in {time.time() - t0:.0f}s")

    done.write_text(json.dumps({"n_layers": int(stacked.shape[0] - 1)}))
    return out


def load_layers(model_name: str, name: str) -> List[np.ndarray]:
    folder = VECTOR_DIR / _slug(model_name)
    n_layers = json.loads((folder / "done.json").read_text())["n_layers"]
    return [np.load(folder / f"{name}_L{l}.npy") for l in range(n_layers + 1)]


# ------------------------------------------------------------- 2. probes
def run_probes(model_name: str, conll: Dict, sst: Dict, jobs: int) -> pd.DataFrame:
    tr, te = load_layers(model_name, "conll_train"), load_layers(model_name, "conll_test")
    s_tr, s_te = load_layers(model_name, "sst2_train"), load_layers(model_name, "sst2_test")

    words_tr, words_te = _flat(conll["train"], "words"), _flat(conll["test"], "words")
    pos_tr, pos_te = _flat(conll["train"], "pos"), _flat(conll["test"], "pos")
    ner_tr, ner_te = _flat(conll["train"], "ner"), _flat(conll["test"], "ner")
    sent_tr = np.array([x["label"] for x in sst["train"]])
    sent_te = np.array([x["label"] for x in sst["test"]])

    # Control labels over train + test together, so a word keeps its label.
    ctrl = control_labels(
        np.concatenate([words_tr, words_te]), np.concatenate([pos_tr, pos_te])
    )
    ctrl_tr, ctrl_te = ctrl[: len(words_tr)], ctrl[len(words_tr):]

    jobs_list = []
    for layer in range(len(tr)):
        jobs_list += [
            ("pos", layer, tr[layer], pos_tr, te[layer], pos_te),
            ("pos_control", layer, tr[layer], ctrl_tr, te[layer], ctrl_te),
            ("ner", layer, tr[layer], ner_tr, te[layer], ner_te),
            ("sentiment", layer, s_tr[layer], sent_tr, s_te[layer], sent_te),
        ]

    def one(task, layer, Xtr, ytr, Xte, yte):
        t0 = time.time()
        acc = probe_accuracy(Xtr, ytr, Xte, yte)
        print(f"  layer {layer:2d} {task:12s} {acc:.3f}  ({time.time() - t0:.0f}s)", flush=True)
        return task, layer, acc

    scores = Parallel(n_jobs=jobs)(delayed(one)(*j) for j in jobs_list)
    acc = {(t, l): a for t, l, a in scores}

    baseline = {
        "pos": majority_baseline(pos_tr, pos_te),
        "ner": majority_baseline(ner_tr, ner_te),
        "sentiment": majority_baseline(sent_tr, sent_te),
    }
    rows = []
    for layer in range(len(tr)):
        for task in ("pos", "ner", "sentiment"):
            rows.append(dict(
                model=model_name, task=task, layer=layer,
                accuracy=round(acc[(task, layer)], 4),
                control_accuracy=round(acc[("pos_control", layer)], 4) if task == "pos" else None,
                baseline=round(baseline[task], 4),
            ))
    return pd.DataFrame(rows)


def save_results(new: pd.DataFrame):
    if RESULTS_CSV.exists():
        old = pd.read_csv(RESULTS_CSV)
        new = pd.concat([old[~old.model.isin(new.model.unique())], new])
    order = {m: i for i, m in enumerate(PROBING_MODELS)}
    new = new.sort_values(["model", "task", "layer"], key=lambda s: s.map(order) if s.name == "model" else s)
    new.to_csv(RESULTS_CSV, index=False)


# --------------------------------------------------------------- 3. CKA
def update_cka():
    """Recompute every CKA map whose models have cached vectors."""
    cached = [m for m in PROBING_MODELS if (VECTOR_DIR / _slug(m) / "done.json").exists()]
    layers = {m: [X[:CKA_WORDS] for X in load_layers(m, "conll_test")] for m in cached}

    out = {}
    for i, a in enumerate(cached):
        for b in cached[i:]:
            t0 = time.time()
            out[f"{a}|{b}"] = np.round(cka_matrix(layers[a], layers[b]), 4).tolist()
            print(f"  CKA {a} x {b} in {time.time() - t0:.0f}s")
    CKA_JSON.write_text(json.dumps(out))


# ------------------------------------------------------------- summary
def summary(df: pd.DataFrame) -> str:
    lines = [
        "| Model | Best POS layer | Best NER layer | Best sentiment layer | Peak POS selectivity |",
        "| --- | --- | --- | --- | --- |",
    ]
    for model in PROBING_MODELS:
        m = df[df.model == model]
        if m.empty:
            continue
        cells = []
        for task in ("pos", "ner", "sentiment"):
            r = m[m.task == task].loc[lambda g: g.accuracy.idxmax()]
            cells.append(f"{int(r.layer)} ({r.accuracy:.1%})")
        pos = m[m.task == "pos"]
        sel = pos.accuracy - pos.control_accuracy
        peak = pos.loc[sel.idxmax()]
        cells.append(f"{sel.max():.1%} (layer {int(peak.layer)})")
        lines.append(f"| {PROBING_MODELS[model]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", choices=list(PROBING_MODELS), action="append",
                        help="model to run (repeatable); default: all three")
    parser.add_argument("--force", action="store_true", help="re-extract cached vectors")
    parser.add_argument("--jobs", type=int, default=4, help="probes trained in parallel")
    args = parser.parse_args()

    print("Loading datasets…")
    conll, sst = datasets.load_conll(), datasets.load_sst2()
    print(f"  conll2003 from {conll['source']}, sst2 from {sst['source']}")

    for model in args.model or list(PROBING_MODELS):
        print(f"\n=== {model}")
        cache_vectors(model, conll, sst, force=args.force)
        save_results(run_probes(model, conll, sst, args.jobs))
        print(f"  results -> {RESULTS_CSV}")

    print("\nCKA…")
    update_cka()
    print(f"  -> {CKA_JSON}")

    print("\n" + summary(pd.read_csv(RESULTS_CSV)))


if __name__ == "__main__":
    main()
