"""
Pre-compute every view for every example sentence, once.

After this, the examples open instantly, even after you restart the app.
Typed sentences are still computed live (and then saved too).

Run from the project folder:  python app/warmup.py
Takes a few minutes on a laptop CPU. Safe to stop and re-run: finished
results are kept.
"""

import importlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attention_lens import extract                                    # noqa: E402
from shared import sentences                                          # noqa: E402
from shared.cache import cached                                       # noqa: E402
from shared.config import APP_CLASSIFIER, APP_MODEL                   # noqa: E402

JOBS = [
    ("attribution_lens", APP_CLASSIFIER),
    ("probing_lens", APP_MODEL),
    ("bias_lens", APP_MODEL),
    ("causal_lens", APP_CLASSIFIER),
]


def main():
    probes = sentences.ALL_PROBES
    start = time.time()
    for n, probe in enumerate(probes, 1):
        text = probe.text
        print(f"[{n}/{len(probes)}] {text}")
        t = time.time()
        cached("attention_lens", text, APP_MODEL, lambda: extract.attention_matrices(text, APP_MODEL))
        print(f"    attention    {time.time() - t:5.1f}s")
        for package, model in JOBS:
            t = time.time()
            try:
                lens = importlib.import_module(f"{package}.lens")
                cached(package, text, model, lambda: lens.compute(text, model))
                print(f"    {package.split('_')[0]:<12} {time.time() - t:5.1f}s")
            except Exception as exc:
                print(f"    {package.split('_')[0]:<12} FAILED: {exc}")
    print(f"Done in {time.time() - start:.0f}s. Cached results are in data/cache/.")


if __name__ == "__main__":
    main()
