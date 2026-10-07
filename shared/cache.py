"""
Disk cache for lens results.

Streamlit's own cache lives in memory, so every restart recomputed every
sentence from scratch. This keeps each result on disk (data/cache/), so a
sentence that has been analysed once opens instantly from then on, even
after a restart.

The cache key includes a fingerprint of the lens's source code: when
anyone edits a lens, its old results are ignored automatically.

data/cache/ is in .gitignore. Fill it with:  python app/warmup.py
"""

import hashlib
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Callable, TypeVar

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / "cache"

T = TypeVar("T")


@lru_cache(maxsize=16)
def _fingerprint(package: str) -> str:
    """Hash of the package's .py files plus shared/, so code edits invalidate results."""
    h = hashlib.sha1()
    for folder in (ROOT / package, ROOT / "shared"):
        for path in sorted(folder.glob("*.py")):
            h.update(path.name.encode())
            h.update(path.read_bytes())
    return h.hexdigest()[:12]


def _path(package: str, text: str, model_name: str) -> Path:
    key = hashlib.sha1(f"{_fingerprint(package)}|{model_name}|{text}".encode()).hexdigest()[:20]
    return CACHE_DIR / package / f"{key}.pkl"


def cached(package: str, text: str, model_name: str, compute: Callable[[], T]) -> T:
    """Return the saved result if there is one, otherwise compute and save it."""
    path = _path(package, text, model_name)
    if path.exists():
        try:
            with path.open("rb") as f:
                return pickle.load(f)
        except Exception:          # stale or unreadable file: just recompute
            path.unlink(missing_ok=True)
    value = compute()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:              # caching is a bonus; never fail the view over it
        pass
    return value
